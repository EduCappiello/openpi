"""Ground truth for 'is this wave done' -- reads the checkpoint dir, not the ledger.

The ledger records intent (which demos a wave covers); the filesystem records fact
(whether training actually finished). The orchestrator trusts the filesystem, and
syncs the ledger's status field to match on every check so `cat episode_ledger.json`
stays an accurate human-readable log.

Deliberately does NOT invoke train.py to answer this question: launching the script
just to discover a no-op costs ~2 minutes of dataset-scan + XLA compile per call,
measured on this box (see WAVE_TRAINING.md). Checking {checkpoint_dir}/{final_step}/
params/ on disk is instant and exactly what orbax itself uses to decide resume vs
fresh-init (openpi/training/checkpoints.py:initialize_checkpoint_dir).
"""
"""Ground truth for 'is this wave done' -- reads the checkpoint dir, not the ledger.

The ledger records intent (which demos a wave covers); the filesystem records fact
(whether training actually finished). The orchestrator trusts the filesystem, and
syncs the ledger's status field to match on every check so `cat episode_ledger.json`
stays an accurate human-readable log.

Deliberately does NOT invoke train.py to answer this question: launching the script
just to discover a no-op costs ~2 minutes of dataset-scan + XLA compile per call,
measured on this box (see WAVE_TRAINING.md). Checking {checkpoint_dir}/{final_step}/
params/ on disk is instant and exactly what orbax itself uses to decide resume vs
fresh-init (openpi/training/checkpoints.py:initialize_checkpoint_dir).
"""
import os
import pathlib
import sys

# Waves whose final checkpoints live on the prior box / in HF storage and are NOT
# expected on this pod's filesystem. The orchestrator treats them as complete without
# a disk check, or it would retrain already-consumed demos.
# NOTE (2026-08-14): wave1 & wave2 were fully fine-tuned (step 14999) and uploaded to
# HF (0Corvid0/pi05-b1k-waves, folders wave1_d30_38_38ep / wave2_d38_46_46ep). Their
# local checkpoints were pruned after upload, so without listing them here the disk-only
# completeness check below would wrongly report them QUEUED and the orchestrator would
# re-train already-consumed demos (re-staging ~206 GB and re-hitting EDQUOT). Verified
# present in HF 2026-08-14. arm0_monolithic is still the pretrained warm-start base and
# is excluded from the wave loop in next_incomplete_wave() below.
#
# NOTE (2026-08-15): wave3 & wave4 were also completed at step 14999 and uploaded to HF
# (folders wave3_d46_54_54ep / wave4_d54_62_62ep, verified present 2026-08-15). wave3's
# local checkpoint was pruned by the orchestrator after wave4's upload. Both are listed
# here so the disk-only check does not re-select an already-consumed wave (the 2026-08-15
# bug that tried to re-train wave3 after wave4). KEEP THIS SET IN SYNC WITH THE PRUNE
# GATE IN run_waves.sh -- a wave is only safe to prune once it is recorded here.
#
# NOTE (2026-08-19): wave7 & wave8 are also complete (step 14999) and uploaded to HF
# (folders wave7_d80_90_90ep / wave8_d90_100_100ep, verified present). Their local
# checkpoints were deleted during disk cleanup (wave8 is re-downloaded from HF by the
# b1k_families backbone_foundation warm-start). Adding them here prevents the wave
# orchestrator from re-training wave7/wave8 and keeps the b1k_families gate (families
# run only after ALL waves COMPLETE) open.
REMOTE_COMPLETE_WAVES = {
    "wave1_d30_38",
    "wave2_d38_46",
    "wave3_d46_54",
    "wave4_d54_62",
    "wave5_d62_70",
    "wave6_d70_80",
    "wave7_d80_90",
    "wave8_d90_100",
}


# TrainConfig.checkpoint_base_dir / assets_base_dir are "./outputs/..." -- relative
# paths resolved (via .resolve()) against the CURRENT WORKING DIRECTORY at call time,
# not against this file's location. Every entry point in this package must run with
# cwd == the openpi repo root, or checkpoint_dir silently resolves to a nonexistent
# path and every wave reads as incomplete. Enforce it here rather than trusting callers.
_OPENPI_ROOT = pathlib.Path(__file__).resolve().parents[1]
os.chdir(_OPENPI_ROOT)
sys.path.insert(0, str(_OPENPI_ROOT / "src"))
sys.path.insert(0, str(_OPENPI_ROOT / "b1k_waves"))
import openpi.training.config as C
import ledger


def wave_config_name(wave_name):
    return f"pi05_b1k_{wave_name}"


def is_wave_complete(wave):
    """True iff this wave's final-step checkpoint params/ exist on disk."""
    if wave["name"] in REMOTE_COMPLETE_WAVES:
        return True
    name = wave_config_name(wave["name"])
    if name not in C._CONFIGS_DICT:
        return False
    cfg = C.get_config(name)
    final_step = cfg.num_train_steps - 1
    return (cfg.checkpoint_dir / str(final_step) / "params").exists()


def partial_progress(wave):
    """Highest locally-saved step for a wave, or None if no checkpoint exists yet."""
    name = wave_config_name(wave["name"])
    if name not in C._CONFIGS_DICT:
        return None
    cfg = C.get_config(name)
    if not cfg.checkpoint_dir.exists():
        return None
    steps = sorted(int(p.name) for p in cfg.checkpoint_dir.iterdir() if p.name.isdigit())
    return steps[-1] if steps else None


def sync_ledger_status():
    """Refresh every wave's status field against on-disk fact. Returns the updated ledger."""
    d = ledger.load()
    changed = False
    for w in d["waves"]:
        if w["name"] == "arm0_monolithic":
            continue  # trained outside this config-generation path; status set manually
        if is_wave_complete(w):
            step = C.get_config(wave_config_name(w["name"])).num_train_steps - 1
            new_status = f"COMPLETE (step {step})"
        else:
            p = partial_progress(w)
            new_status = f"IN_PROGRESS (step {p})" if p is not None else "QUEUED"
        if w.get("status") != new_status:
            w["status"] = new_status
            changed = True
    if changed:
        ledger.save(d)
    return d


def next_incomplete_wave(d=None):
    d = d or sync_ledger_status()
    for w in d["waves"]:
        if w["name"] == "arm0_monolithic":
            continue
        if w["name"] in REMOTE_COMPLETE_WAVES:
            continue
        if "COMPLETE" not in w.get("status", ""):
            return w
    return None


def _find(d, name):
    for w in d["waves"]:
        if w["name"] == name:
            return w
    return None


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        # Exit 0 iff the named wave's final checkpoint exists on disk. Used by
        # run_waves.sh right after train.py exits, to verify completion from fact
        # rather than trusting train.py's own exit code (which is 0 on a clean crash
        # recovery path too, e.g. a caught exception that still returns cleanly).
        d = sync_ledger_status()
        w = _find(d, sys.argv[2])
        if w is None:
            print(f"unknown wave {sys.argv[2]!r}")
            sys.exit(2)
        ok = "COMPLETE" in w.get("status", "")
        print(w["status"])
        sys.exit(0 if ok else 1)

    d = sync_ledger_status()
    for w in d["waves"]:
        print(f"{w['name']:20s} demos[{w['demo_lo']:3d},{w['demo_hi']:3d})  {w.get('status','?')}")
    nxt = next_incomplete_wave(d)
    print(f"\nnext to run: {nxt['name'] if nxt else '(none -- all waves complete)'}")
