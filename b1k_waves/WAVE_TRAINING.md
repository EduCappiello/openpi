# Wave Training — Status & Design

Documents the unattended `run_waves.sh` pipeline that fine-tunes `pi0.5` on the
BEHAVIOR-1K 2026-challenge demos across GPUs. Written 2026-07-24 from a live
inspection of the running system; referenced by `run_waves.sh` and
`wave_status.py` as the design doc (it didn't exist yet — this is it).

## TL;DR status as of 2026-07-24 06:43 UTC

**Training is healthy and on schedule.** No action needed.

| Wave | Demos | Status |
|---|---|---|
| `arm0_monolithic` | 0–30 | ✅ COMPLETE (step 49999) |
| `wave1_d30_38` | 30–38 | ✅ COMPLETE (step 14999) |
| `wave2_d38_46` | 38–46 | 🔄 **IN PROGRESS** — step ~13,100 / 15,000, ~1.4 s/it, ETA ~45 min |
| `wave3_d46_54` | 46–54 | ⏳ QUEUED |
| `wave4_d54_62` | 54–62 | ⏳ QUEUED |
| `wave5_d62_70` | 62–70 | ⏳ QUEUED |

- Orchestrator `run_waves.sh` (pid 30900) has been running since 2026-07-24 03:22Z,
  found wave2's trainer already active (pid 16595, started 01:32Z) and is correctly
  waiting on it rather than double-launching.
- Both H100s at 100% util, ~62 GB/80 GB memory each — normal, matches the reference
  batch-64 config.
- Latest checkpoint for wave2 saved cleanly at step 12500 (async orbax save,
  8.4s, old step 10000 pruned per `keep_period`/save_interval housekeeping).
- Disk: 145 GB free / 373 GB total (62% used) — comfortable headroom.
- `--no-wandb-enabled`: W&B logging is intentionally off for these runs; progress
  is tracked via the `tqdm`-style lines in `train_wave2.log` and via
  `wave_status.py`/the episode ledger, not a dashboard.

## What the pipeline does

`run_waves.sh` is meant to be started once (login, cron `@reboot`, or a systemd
unit) and left alone. Per the script's own header, it re-derives all state from
disk on every iteration — checkpoints, the episode ledger, and the lerobot cache —
never from its own memory, so it is safe to kill and restart at any point.

Loop, per wave:
1. **lerobot version guard** (`ensure_lerobot.py`) — the one real outage this
   campaign hit was a `uv sync` elsewhere silently downgrading lerobot
   0.4.4 → 0.3.4, which can't read the v3.0 dataset layout and crashes training
   on next launch. Checked and fixed before anything else runs.
2. **Reclaim disk** (`reclaim.py`) — deletes cached media not needed by any
   incomplete wave.
3. **Stage media** (`stage_wave.py`) for the wave's demo range, with retries.
4. **Attach-or-launch**: if a `scripts/train.py <config>` process for this wave
   is already running (e.g. launched manually, or by a still-running earlier
   invocation), it waits on that pid instead of starting a second one — two
   processes writing the same orbax checkpoint dir would corrupt it and fight
   over GPU memory. This is exactly the situation observed right now: wave2's
   trainer (pid 16595) was already running when `run_waves.sh` started at 03:22Z.
5. **Launch/resume in the foreground** with up to 3 crash-retries, always with
   `--resume` (safe unconditionally — empty checkpoint dir means fresh warm
   start from the prior wave's params, partial dir means resume that wave).
6. **Verify completion from disk**, not from `train.py`'s exit code — a caught
   exception can still exit 0. `wave_status.py --check <wave>` checks whether
   `{checkpoint_dir}/{final_step}/params` exists.
7. Repeat until `next_wave_info.py` reports `DONE`.

A `flock` on `run_waves.lock` makes a second concurrent invocation of the script
exit immediately instead of racing the first.

## Ground truth: ledger vs. filesystem

`episode_ledger.json` records **intent** (seed 20260721, 100 tasks × 200
episodes/task, which demo indices belong to which wave). `wave_status.py`
treats the **filesystem** as the fact of record and syncs the ledger's
`status` field to match on every check — so the ledger stays a human-readable
log but is never trusted for the actual go/no-go decision. (Its `status`
field can lag the live `tqdm` step count between syncs — e.g. it read
"step 5000" for wave2 while the trainer was actually past 13,000 — that's
expected staleness, not a bug.)

## Prior run: `arm0_monolithic` and the old supervisor

Before this wave pipeline existed, a simpler per-process supervisor
(`supervise_arm0.sh`, logs under `/root/b1k-logs/`) ran the original monolithic
baseline (`pi05_b1k`, demos 0–30, 50,000 steps). Its log shows a restart loop
from 2026-07-21 21:32Z to 2026-07-22 03:13Z: the trainer had already reached
`50.0kit/50.0kit` (i.e. finished) each time, but exited fast enough after
finishing that the supervisor's liveness check misread it as a crash and
relaunched with `--resume` roughly every 13 minutes. This was harmless —
`--resume` against a complete checkpoint is a no-op — but cosmetically noisy
in `supervisor.log`. That supervisor process is no longer running (superseded
by `run_waves.sh`); `arm0_monolithic` is confirmed complete at step 49999 and
is excluded from the wave loop (`wave_status.py` special-cases it: "trained
outside this config-generation path; status set manually").

## Key files

- `run_waves.sh` — orchestrator entrypoint.
- `episode_ledger.json` — wave definitions and synced status.
- `ledger.py` / `wave_status.py` / `next_wave_info.py` — ledger I/O and
  completion logic.
- `stage_wave.py` / `reclaim.py` / `ensure_lerobot.py` — data-staging and env
  guards run before each wave.
- `train_wave1.log`, `train_wave2.log`, ... — raw stdout/stderr of each wave's
  `scripts/train.py` invocation.
- `run_waves.log` / `run_waves_stdout.log` — orchestrator-level log (lerobot
  guard results, staging results, launch/attach decisions).
- Checkpoints: `../outputs/checkpoints/pi05_b1k_wave{N}_d{lo}_{hi}/wave{N}_d{lo}_{hi}/{step}/`
  (~13 GB per wave's kept checkpoint; `keep_period` prunes intermediate steps).
- Config: `../src/openpi/training/config.py`, `_make_wave_configs()` — wave
  `TrainConfig`s are generated programmatically (batch size 64, `pi05` model,
  `action_horizon=50`) rather than hand-declared per wave.

## Verifying status yourself

```bash
cd /root/dev/b1k-baselines/baselines/openpi
.venv/bin/python b1k_waves/wave_status.py          # dumps synced ledger status
tail -f b1k_waves/train_wave2.log                   # live progress of the active wave
tail -f b1k_waves/run_waves.log                     # orchestrator decisions
nvidia-smi                                          # GPU utilization/memory
```
