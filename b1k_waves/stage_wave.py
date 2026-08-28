"""Stage one contiguous wave of BEHAVIOR-1K RGB media, in parallel.

Never fetch single-threaded: measured on this box, sequential hf_hub_download runs
at 0.7-2.4 MB/s vs 40-130 MB/s with 16 workers. Depth streams are skipped -- pi0.5
never reads them and they are ~half the bytes.

Self-healing against a hang observed in production: one file's download once sat idle
for 10+ minutes with no error and no progress (the underlying CDN client retries
internally and can wedge). A per-file result() timeout turns that into a bounded
failure instead of an indefinite stall, and failed/timed-out files are retried with
fewer concurrent workers -- mirrors the manual recovery that worked the first time
this happened (16 workers -> 4 workers cleared the same class of stall).
"""
import argparse
import glob
import os
import pathlib
import sys
import time

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download

REPO = "behavior-1k/2026-challenge-demos"
ROOT = pathlib.Path("/root/.cache/huggingface/lerobot/behavior-1k/2026-challenge-demos")
STREAMS = ["zed_link_camera_0", "left_realsense_link_camera_0", "right_realsense_link_camera_0"]

# Per-file wall-clock budget. Wave files are ~100-210 MB; even at a pessimistic
# 1 MB/s (far below the 40-130 MB/s measured with parallel workers) that is under
# 4 minutes, so 6 minutes is a stall detector, not a bandwidth limit.
PER_FILE_TIMEOUT_S = 240
MAX_ATTEMPTS = 5

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"


def meta():
    cols = ["episode_index", "length", "task_index", "demo_index_within_task",
            "data/chunk_index", "data/file_index"] + \
           [f"videos/observation.rgb.{s}/{k}" for s in STREAMS for k in ["chunk_index", "file_index"]]
    return pd.concat([pd.read_parquet(f, columns=cols)
                      for f in sorted(glob.glob(str(ROOT / "meta/episodes/*/*.parquet")))], ignore_index=True)


def wave_files(df, lo, hi):
    sub = df[(df.demo_index_within_task >= lo) & (df.demo_index_within_task < hi)]
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


def stage(lo, hi, workers=16, dry_run=False):
    """Returns (episodes_covered, n_fetched, n_still_missing)."""
    df = meta()
    paths, sub = wave_files(df, lo, hi)
    todo = [p for p in paths if not (ROOT / p).exists()]
    vol = str(ROOT)  # data volume (lerobot cache root), not "/"
    free = os.statvfs(vol).f_bavail * os.statvfs(vol).f_frsize / 1e9
    print(f"wave demos {lo}..{hi}: {len(sub)} episodes, {sub.length.sum()/1e6:.1f}M frames")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    _, _, missing = stage(a.lo, a.hi, a.workers, a.dry_run)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
