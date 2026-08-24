"""Ground truth for 'is this family done' -- reads the checkpoint dir, not the ledger.

Mirrors b1k_waves/wave_status.py. The ledger records intent (tasks/demos); the
filesystem records fact (whether training finished). The orchestrator trusts the
filesystem and syncs the ledger's status field on every check.

Completion is DISK-ONLY. Historically a family whose local checkpoint was cleaned
up but whose params live on HF was reported COMPLETE via an HF-upload probe; that
fallback is disabled for the full-FT clean-slate re-run so stale HF uploads can
never suppress a needed re-train (mirrors the empty REMOTE_COMPLETE_WAVES in the
wave pipeline). upload_family.py still handles the HF push itself.

Never invoke train.py to answer this question: checking
{checkpoint_dir}/{final_step}/params on disk is instant and exactly what orbax
uses to decide resume vs fresh-init.
"""
import pathlib
import sys

# checkpoint_base_dir/assets_base_dir are "./outputs/..." -- relative paths
# resolved against the CWD, so every entry point must run with cwd == the openpi
# repo root. Enforce it here rather than trusting callers.
_OPENPI_ROOT = pathlib.Path(__file__).resolve().parents[1]
import os
os.chdir(_OPENPI_ROOT)
sys.path.insert(0, str(_OPENPI_ROOT / "src"))
sys.path.insert(0, str(_OPENPI_ROOT / "b1k_families"))
import openpi.training.config as C
import family_ledger as ledger

# HF repo where completed families are uploaded (gated, owner-only fetch).
# Used only by upload_family.py; family_status.py completion is disk-only now.
UPLOAD_REPO_ID = "0Corvid0/pi05-b1k-families"


def family_config_name(name):
    return f"pi05_b1k_{name}"


def is_family_complete(f):
    """True iff this family's final-step params/ exists on disk (completion is disk-only)."""
    name = family_config_name(f["name"])
    if name in C._CONFIGS_DICT:
        cfg = C.get_config(name)
        final_step = cfg.num_train_steps - 1
        if (cfg.checkpoint_dir / str(final_step) / "params").exists():
            return True
    return False


def partial_progress(f):
    """Highest locally-saved step for a family, or None."""
    name = family_config_name(f["name"])
    if name not in C._CONFIGS_DICT:
        return None
    cfg = C.get_config(name)
    if not cfg.checkpoint_dir.exists():
        return None
    steps = sorted(int(p.name) for p in cfg.checkpoint_dir.iterdir() if p.name.isdigit())
    return steps[-1] if steps else None


def sync_ledger_status():
    """Refresh every family's status field against on-disk fact (no HF fallback)."""
    d = ledger.load()
    changed = False
    for f in d["families"]:
        name = family_config_name(f["name"])
        final_step = None
        if name in C._CONFIGS_DICT:
            cfg = C.get_config(name)
            if (cfg.checkpoint_dir / str(cfg.num_train_steps - 1) / "params").exists():
                final_step = cfg.num_train_steps - 1
        if final_step is not None:
            new_status = f"COMPLETE (step {final_step})"
        else:
            p = partial_progress(f)
            new_status = f"IN_PROGRESS (step {p})" if p is not None else "QUEUED"
        if f.get("status") != new_status:
            f["status"] = new_status
            changed = True
    if changed:
        ledger.save(d)
    return d


def is_finished(status):
    """A family is finished iff its final checkpoint is on disk (COMPLETE)."""
    return "COMPLETE" in status


def next_incomplete_family(d=None):
    d = d or sync_ledger_status()
    for f in d["families"]:
        if not is_finished(f.get("status", "")):
            return f
    return None


def _find(d, name):
    for f in d["families"]:
        if f["name"] == name:
            return f
    return None


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        # Exit 0 iff the named family's final checkpoint exists on disk.
        d = sync_ledger_status()
        f = _find(d, sys.argv[2])
        if f is None:
            print(f"unknown family {sys.argv[2]!r}")
            sys.exit(2)
        ok = is_finished(f.get("status", ""))
        print(f["status"])
        sys.exit(0 if ok else 1)

    d = sync_ledger_status()
    for f in d["families"]:
        print(f"{f['name']:24s} tasks({len(f['task_ids']):3d})  {f.get('status', '?')}")
    nxt = next_incomplete_family(d)
    print(f"\nnext to run: {nxt['name'] if nxt else '(none -- all families complete)'}")
