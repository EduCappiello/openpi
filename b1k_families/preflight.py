"""Pre-flight health + resource checks before launching a family training run.

Runs from the openpi repo root (checkpoint paths are cwd-relative). Checks, in
order:
  1. cwd guard      -- must be the openpi repo root.
  2. lerobot guard  -- venv lerobot == version pinned in uv.lock (the wensi-ai
                       lerobot release/b1k fork, 0.5.2; v3.0 dataset layout support).
  3. warm-start     -- the --warm-start <path> checkpoint has params/ on disk.
  4. VRAM           -- XLA_PYTHON_CLIENT_MEM_FRACTION auto-computed from free VRAM:
                          FRAC = (MIN_FREE - SAFETY_MIB) / TOTAL
                      capped at B1K_MEM_FRACTION_CAP, floored at
                      B1K_MEM_FRACTION_FLOOR (abort below floor). Manual override
                      via B1K_MEM_FRACTION if set.
  5. disk           -- require >= B1K_MIN_DISK_GB free.
  6. HF token       -- warn (non-fatal) if no token; uploads will fail, not train.

Prints `FRACTION=<float>` as the final line; exit 0 iff all hard checks pass.
"""
import os
import pathlib
import shutil
import subprocess
import sys

# ---- config (env-overridable) ----------------------------------------------------------
_TOTAL_MIB = None  # set dynamically from nvidia-smi; fallback (MiB) per H100 80GB
SAFETY_MIB = int(os.environ.get("B1K_MEM_SAFETY_MIB", "4096"))
CAP = float(os.environ.get("B1K_MEM_FRACTION_CAP", "0.85"))
# NOTE (build agent, 2026-08-06): floor tuned to the TRUE training need, not XLA
# preallocation. Real peak for batch-64/horizon-50 pi05 is ~35-48 GB/GPU; 0.60
# (~49 GB free) covers the worst case with margin. This lets a modest model server
# (e.g. Qwen3.6 27B via ollama, ~13 GB/GPU) coexist with training. It does NOT
# permit a ~44 GB/GPU DeepSeek V4 Flash alongside batch-64 training (44+48 > 80);
# that combination requires a smaller batch (e.g. 32) or dedicated GPUs.
# 2-GPU pod with the resident model server (~28 GB/GPU): best achievable fraction is
# (81559-28666-4096)/81559 ~= 0.59, so the floor must sit below that or every run FATALs.
FLOOR = float(os.environ.get("B1K_MEM_FRACTION_FLOOR", "0.59"))
MIN_DISK_GB = float(os.environ.get("B1K_MIN_DISK_GB", "40"))
REQUIRED_LEROBOT = "0.4.4"

_OPENPI_ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---- checks ---------------------------------------------------------------------------
def check_cwd():
    # Must already have chdir'd? No: we chdir to openpi root here (like wave_status).
    # But if the user ran us from the wrong place, resolve() on --warm-start is literal,
    # so we chdir for safety and treat "not the root" as the guard only if impossible.
    if not (pathlib.Path.cwd().resolve() / "src").exists():
        print("FATAL: cwd is not the openpi repo root (relative ckpt/assets paths would misresolve)")
        return False
    return True


def _required_lerobot():
    """Version pinned in uv.lock (single source of truth; tracks the wensi-ai fork)."""
    lock = _OPENPI_ROOT / "uv.lock"
    try:
        text = lock.read_text()
    except OSError:
        return REQUIRED_LEROBOT
    for block in text.split("[[package]]"):
        if 'name = "lerobot"' in block:
            for line in block.splitlines():
                line = line.strip()
                if line.startswith("version = "):
                    return line.split("=", 1)[1].strip().strip('"')
    return REQUIRED_LEROBOT


def check_lerobot():
    required = _required_lerobot()
    try:
        import lerobot
        ok = lerobot.__version__ == required
    except ImportError:
        ok = False
    if ok:
        print(f"lerobot OK ({required})")
    else:
        print(f"FATAL: lerobot wrong/missing (need {required}). Run b1k_waves/ensure_lerobot.py then retry.")
    return ok


def check_warm_start(path):
    p = pathlib.Path(path)
    resolved = p if p.is_absolute() else (pathlib.Path.cwd() / p)
    # The warm-start arg is the params DIRECTORY itself (CheckpointWeightLoader
    # reads params_path directly), so check the dir's existence, not dir/params.
    ok = resolved.exists()
    print(f"warm-start: {path} -> {'OK' if ok else 'MISSING (no such dir)'}")
    return ok


def _query_gpu():
    """Return (list[int] free_MiB, total_MiB) from nvidia-smi, or (None, None)."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None, None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return None, None
    rows = []
    for line in out.splitlines():
        parts = [p.strip().strip('MiB') for p in line.split(',')]
        if len(parts) >= 2:
            try:
                rows.append((int(parts[0]), int(parts[1])))
            except ValueError:
                pass
    if not rows:
        return None, None
    return [free for _, free in rows], rows[0][0]


def check_vram():
    override = os.environ.get("B1K_MEM_FRACTION")
    if override:
        frac = float(override)
        print(f"VRAM: using manual B1K_MEM_FRACTION override = {frac:.3f}")
        return frac, True

    frees, total = _query_gpu()
    total = total or int(os.environ.get("B1K_GPU_TOTAL_MIB", "81559"))
    if frees is None or not frees:
        print(f"WARN: nvidia-smi unavailable -- using floor {FLOOR} (conservative)")
        return FLOOR, True

    min_free = min(frees)
    total = max(total, min_free)  # safety
    frac = (min_free - SAFETY_MIB) / total
    print(f"VRAM: min free {min_free} MiB / {total} MiB total (safety {SAFETY_MIB} MiB) -> raw FRAC {frac:.3f}")
    frac = min(frac, CAP)
    if frac < FLOOR:
        print(f"FATAL: computed FRAC {frac:.3f} < floor {FLOOR:.3f}. Need >= {FLOOR:.3f} "
              f"(approx {int(FLOOR*total)} MiB free per GPU).")
        print("  Free VRAM: stop the llama.cpp/ollama serving processes, then re-run.")
        try:
            subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory,name",
                            "--format=csv,noheader"], text=True)
        except Exception:  # noqa: BLE001
            pass
        return frac, False
    print(f"VRAM: approved FRAC = {frac:.3f} (cap {CAP}, floor {FLOOR})")
    return frac, True


def check_disk():
    # Data/checkpoints live on the big volume (B1K_DATA_VOLUME), not on "/" --
    # check the volume that actually holds the lerobot cache.
    vol = os.environ.get("B1K_DATA_VOLUME", "/tmp/hf-cache")
    free = shutil.disk_usage(vol).free / 1e9
    ok = free >= MIN_DISK_GB
    print(f"disk: {free:.0f} GB free at {vol} (require {MIN_DISK_GB:.0f} GB) -> {'OK' if ok else 'FATAL'}")
    return ok


def check_hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        print("HF_TOKEN set (uploads OK)")
        return True
    print("WARN: HF_TOKEN not set -- training will still run, but upload_family.py will fail (non-fatal).")
    return True


def main():
    # Ensure a stable cwd for relative-path checks.
    if not (pathlib.Path.cwd().resolve() / "src").exists():
        os.chdir(_OPENPI_ROOT)

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-start", required=True, help="path to warm-start params dir")
    args = ap.parse_args()

    hard_ok = True
    hard_ok &= check_cwd()
    hard_ok &= check_lerobot()
    hard_ok &= check_warm_start(args.warm_start)
    frac, vram_ok = check_vram()
    hard_ok &= vram_ok
    hard_ok &= check_disk()
    check_hf_token()  # non-fatal

    print(f"FRACTION={frac:.3f}")
    sys.exit(0 if hard_ok else 1)


if __name__ == "__main__":
    main()
