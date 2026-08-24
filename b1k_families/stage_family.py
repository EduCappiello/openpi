"""Stage a task-filtered subset of BEHAVIOR-1K RGB media, in parallel.

Adapted from b1k_waves/stage_wave.py for super-family experts: instead of a
contiguous demo block over all 100 tasks, stage the episodes of a specific set
of TASK IDs within a demo window (demos 90-99 for families). Files already on
disk are skipped, so overlapping dual-family tasks (10, 23, 58, 79) do not
re-download.

Unlike the wave pipeline the lerobot cache has already been pruned, so the meta/
parquet files must be fetched before staging can filter by task. This is handled
by ensure_meta() -- a cheap snapshot_download restricted to meta/*.
"""
import argparse
import glob
import os
import pathlib
import sys
import time

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download, snapshot_download

REPO = "behavior-1k/2026-challenge-demos"
ROOT = pathlib.Path("/root/.cache/huggingface/lerobot/behavior-1k/2026-challenge-demos")
STREAMS = ["zed_link_camera_0", "left_realsense_link_camera_0", "right_realsense_link_camera_0"]

# Per-file wall-clock budget. Wave files are ~100-210 MB; 6 minutes is a stall
# detector, not a bandwidth limit (measured 40-130 MB/s with parallel workers).
PER_FILE_TIMEOUT_S = 360
MAX_ATTEMPTS = 4

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")  # per-HTTP-request timeout


def ensure_meta(workers=16):
    """Fetch the meta/ parquet files (task/demo index) if not already present.

    meta/ is a small fraction of the repo but required to know which files a
    task's episodes live in. No-op once present.
    """
    if glob.glob(str(ROOT / "meta/episodes/*/*.parquet")):
        return True
    print(f"meta/ not staged -- downloading meta/* (~small) into {ROOT}", flush=True)
    os.makedirs(ROOT, exist_ok=True)
    try:
        snapshot_download(REPO, repo_type="dataset", local_dir=str(ROOT),
                          allow_patterns=["meta/*"], max_workers=workers)
    except Exception as e:  # noqa: BLE001
        print(f"meta download failed: {e}", flush=True)
        return False
    return bool(glob.glob(str(ROOT / "meta/episodes/*/*.parquet")))


def meta():
    cols = ["episode_index", "length", "task_index", "demo_index_within_task",
            "data/chunk_index", "data/file_index"] + \
           [f"videos/observation.rgb.{s}/{k}" for s in STREAMS for k in ["chunk_index", "file_index"]]
    return pd.concat([pd.read_parquet(f, columns=cols)
                      for f in sorted(glob.glob(str(ROOT / "meta/episodes/*/*.parquet")))], ignore_index=True)


def family_files(df, task_ids, lo, hi):
    sub = df[(df["task_index"].isin(task_ids)) &
             (df["demo_index_within_task"] >= lo) & (df["demo_index_within_task"] < hi)]
    paths = set()
    for _, r in sub[["data/chunk_index", "data/file_index"]].drop_duplicates().iterrows():
        paths.add(f"data/chunk-{int(r['data/chunk_index']):03d}/file-{int(r['data/file_index']):03d}.parquet")
    for s in STREAMS:
        u = sub[[f"videos/observation.rgb.{s}/chunk_index", f"videos/observation.rgb.{s}/file_index"]].drop_duplicates()
        for _, r in u.iterrows():
            paths.add(f"videos/observation.rgb.{s}/chunk-{int(r.iloc[0]):03d}/file-{int(r.iloc[1]):03d}.mp4")
    return sorted(paths), sub


def _fetch_batch(paths, workers, per_file_timeout):
    """One pass over `paths`. Returns the subset that failed or timed out."""
    failed = []

    def get(p):
        hf_hub_download(REPO, p, repo_type="dataset", local_dir=str(ROOT))
        return p

    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(get, p): p for p in paths}
        done = 0
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                fut.result(timeout=per_file_timeout)
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(paths)}", flush=True)
            except Exception as e:  # noqa: BLE001 -- includes TimeoutError from .result()
                print(f"  FAILED {p}: {type(e).__name__}: {e}", flush=True)
                failed.append(p)
    return failed


def stage(task_ids, lo, hi, workers=16, dry_run=False):
    """Returns (episodes_covered, n_fetched, n_still_missing)."""
    if not ensure_meta(workers):
        print("FATAL: could not stage meta/", flush=True)
        return 0, 0, -1  # sentinel: -1 missing means metadata failure
    df = meta()
    paths, sub = family_files(df, task_ids, lo, hi)
    todo = [p for p in paths if not (ROOT / p).exists()]
    free = os.statvfs("/").f_bavail * os.statvfs("/").f_frsize / 1e9
    print(f"family {len(task_ids)} tasks, demos {lo}..{hi}: {len(sub)} episodes, "
          f"{sub.length.sum()/1e6:.1f}M frames")
    print(f"files: {len(paths)} total | {len(paths)-len(todo)} already local | {len(todo)} to fetch")
    print(f"disk free: {free:.0f} GB")
    if dry_run or not todo:
        return len(sub), 0, len(todo)

    remaining = todo
    fetched_total = 0
    w = workers
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"attempt {attempt}/{MAX_ATTEMPTS}: fetching {len(remaining)} files with {w} workers", flush=True)
        failed = _fetch_batch(remaining, w, PER_FILE_TIMEOUT_S)
        fetched_total += len(remaining) - len(failed)
        if not failed:
            remaining = []
            break
        remaining = failed
        w = max(2, w // 2)  # back off concurrency -- observed to clear CDN stalls
        if attempt < MAX_ATTEMPTS:
            time.sleep(10 * attempt)  # brief backoff before retrying

    still_missing = [p for p in remaining if not (ROOT / p).exists()]
    print(f"staged {fetched_total} files; {len(still_missing)} still missing after {MAX_ATTEMPTS} attempts")
    return len(sub), fetched_total, len(still_missing)


def parse_tasks(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="comma-separated task IDs, e.g. '8,14,15'")
    ap.add_argument("--lo", type=int, default=90)
    ap.add_argument("--hi", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    task_ids = parse_tasks(a.tasks)
    if not task_ids:
        print("FATAL: --tasks must be a non-empty comma-separated list")
        sys.exit(1)
    _, _, missing = stage(task_ids, a.lo, a.hi, a.workers, a.dry_run)
    # -1 sentinel (meta failure) and any still-missing file are fatal.
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
