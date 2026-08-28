# Wave Training — Status & Design

Documents the unattended `run_waves.sh` pipeline that fine-tunes `pi0.5` on the
BEHAVIOR-1K 2026-challenge demos across GPUs. Referenced by `run_waves.sh` and
`wave_status.py` as the design doc.

> ## 🔄 POD MOVED (2026-08-27): 4×H100 → 2×H100, campaign COMPLETE
>
> **All 8 waves are now COMPLETE (step 14999) and uploaded to `0Corvid0/pi05-b1k-waves`**,
> and all 5 families (backbone + F1..F4) are COMPLETE and uploaded too. The machine was
> reinstalled from the personal backup on **2026-08-27** and the training work now lives on a
> **2× H100 80 GB, 20-core** pod (was 4× H100, 40-core). `fsdp_devices=2`, `batch_size=32`,
> `num_workers=8`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.60` (was 0.75); data + venv + checkpoints
> live on `/tmp` (14 TB), not `/` (66 GB PVC); the working copy is at
> `/tmp/b1k/BEHAVIOR-1K/b1k-baselines/baselines/openpi` (was `/root/BEHAVIOR-1K/...`).
> **the model server** holds ~28 GB/GPU and is kept running (it was a standalone inference server at ~45 GB/GPU).
> See `AGENT_PRIMER.md` for the full old-vs-new table. Everything below is still accurate for
> *how* the pipeline works; treat its machine-specific numbers (GPU count, disk, paths,
> VRAM) as historical.
>
> ---
>
> ## ⚠️ CURRENT STATE (2026-08-15, OLD POD): FULL-FT WAVE CHAIN MID-PROGRESS
>
> **This is the FULL fine-tuning campaign** (`gemma_2b`, full FT, `action_horizon=32`,
> `fsdp_devices=4`) replacing the prior LoRA run. It is **mid-way through the wave chain**:
>
> - **wave1–wave6 are COMPLETE** (step 14999) and uploaded to `0Corvid0/pi05-b1k-waves`
>   (folders `wave1_d30_38_38ep` … `wave6_d70_80_80ep`). Their local checkpoints were pruned (or
>   deleted 2026-08-15), so they are registered in **`REMOTE_COMPLETE_WAVES`** in `wave_status.py`
>   (verified present in HF 2026-08-15). Without this, the disk-only completeness check would
>   wrongly report them QUEUED and the orchestrator would re-train already-consumed demos.
> - **wave7 is the active wave (IN_PROGRESS at step 2500).** It resumes from its own valid local
>   checkpoint at `outputs/checkpoints/pi05_b1k_wave7_d80_90/wave7_d80_90/2500` (~30 GB) and
>   warms from wave6's final checkpoint (uploaded; wave6's local copy was deleted in the
>   2026-08-15 cleanup). **wave8 is QUEUED**; the family STAR pipeline (F1..F4) is gated until
>   all waves COMPLETE.
> - **wave7 crash on 2026-08-15 was disk-full (EDQUOT), not a corrupt checkpoint.** It died at
>   the step-5000 checkpoint write with `OS error 122: EDQUOT Disk quota exceeded` on all 3
>   launch attempts → `FATAL`. The disk had filled to 100% because completed waves 5 & 6 (29 GB
>   each) and stale media were still on disk. **Recovery (2026-08-15):** deleted local checkpoints
>   for wave5 & wave6 (both verified uploaded), the stale wave7 `5000.orbax-checkpoint-tmp-*`
>   leftovers, and 237 stale `data/*.parquet` files not needed by waves 7/8 (~18.9 GB) → **~82 GB
>   freed** (~81 GB free). Also added `wave5_d62_70` & `wave6_d70_80` to `REMOTE_COMPLETE_WAVES`
>   so the prune safety-gate releases them on future runs (this was the gap that let the disk
>   fill: they were uploaded but not in the set, so the prune kept them as fallback forever).
> - **wave3 crash on 2026-08-14 was environmental, not a corrupt checkpoint.** It died three
>   ways across retries: (a) GPU OOM (`Cuda failure 2 'out of memory'`) while `the inference server`
>   held ~44 GB/GPU; (b) `std::system_error: Resource temporarily unavailable` at the step-7500
>   checkpoint save — OpenBLAS `pthread_create failed for thread N of 64`, i.e. thread/CPU
>   exhaustion because OpenBLAS/OMP/JAX spawn up to 64 threads against the pod's **40-CPU cgroup
>   quota** (`/sys/fs/cgroup/cpu.max`); (c) the same thread failure on retry 3. Fixed in
>   `run_waves.sh` by capping `OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/`MKL_NUM_THREADS=40` on
>   the `train.py` launch. A stale `7500.orbax-checkpoint-tmp-*` leftover from the interrupted
>   save was removed; the step-5000 checkpoint is intact and is the resume point.
> - **Model config is FULL FT**: `_b1k_wave_model()` = `Pi0Config(pi05=True, action_horizon=32)`
>   → `paligemma_variant="gemma_2b"` (no LoRA), `action_dim` default 32,
>   `fsdp_devices=4`. `get_freeze_filter()` returns `Nothing` → every weight is trainable.
> - **Frozen validation split** (behavior_training_split.md): a single versioned 18,500-train /
>   1,500-val split (`frozen_val_split.json`) shared by every variant. `config.py` reads it for
>   both `_make_wave_configs()` and `_make_family_configs()`, then **intersects it with each
>   run's staged demo/task subset** (see "Dataset staging — Option B implemented" below) so the
>   trainer only references media that `stage_wave.py` / `stage_family.py` actually download.
> - **VRAM**: a standalone inference server may hold ~45 GB/GPU. Full FT needs
>   `XLA_PYTHON_CLIENT_MEM_FRACTION=0.75` (~60 GB/GPU) and dies **silently** (no traceback,
>   whole tmux pane dies) if it cannot preallocate. Check `nvidia-smi` and free the GPU before
>   launching. See "Launch checklist" below.

The history below describes the old (LoRA) campaign. It is kept for reference; it no
longer reflects current training state. Use the sections marked "Model config (full FT)"
and "Launch checklist" for the current run.

## Prior campaign status (LoRA — superseded 2026-08-12)

**All waves (arm0–wave8) had completed.** The chain ran on a prior 2× H100 box (arm0 → wave1 → … → wave5). This pod is a **4× H100, 40-CPU cgroup quota** machine where `wave5_d62_70` was re-trained and completed all 15,000 steps on 2026-07-31. Waves 6–8 ran sequentially on this pod, finishing at ~13:12 UTC on 2026-08-02.

| Wave | Demos (per task) | Steps | Old status |
|---|---|---|---|
| `arm0_monolithic` | 0–30 | 49,999 | ✅ COMPLETE (warm-start base — still used) |
| `wave1_d30_38` … `wave5_d62_70` | 30–70 | 15,000 | ✅ COMPLETE (LoRA-era, superseded) |
| `wave6_d70_80` … `wave8_d90_100` | 70–100 | 15,000 | ✅ COMPLETE (LoRA-era, superseded) |

- The old LoRA-era checkpoints were mirrored in `IntelligentDecisionLab/pi05-b1k-monolithic-model`
  and **deleted locally**; the FULL-FT chain re-trains them. As of 2026-08-15 wave1–wave4 are
  re-trained & re-uploaded (see the current-state block above); wave5–wave8 are still pending.

## Training Timeline (This Pod)

| Wave | Started | Completed | Duration |
|---|---|---|---|
| wave5_d62_70 (re-train) | 2026-07-31 | 2026-07-31 | ~8h |
| wave6_d70_80 | 2026-08-01T15:29Z | 2026-08-01T19:04Z | ~3.7h |
| wave7_d80_90 | 2026-08-02T05:47Z | 2026-08-02T09:30Z | ~3.7h |
| wave8_d90_100 | 2026-08-02T09:31Z | 2026-08-02T13:12Z | ~3.7h |

## Model config (full FT) — current run

`_b1k_wave_model()` in `src/openpi/training/config.py` builds the model for every wave and
family config:

```python
def _b1k_wave_model():
    return pi0_config.Pi0Config(pi05=True, action_horizon=32)
```

- `paligemma_variant` defaults to **`gemma_2b`** → **full fine-tuning**, no LoRA. The SigLIP
  vision encoder, 2B PaliGemma VLM, 300M action expert, and heads are all trainable
  (`get_freeze_filter()` → `nnx.Nothing`).
- `action_horizon = 32` (matches the monolith spec in behavior_training_split.md).
- `action_dim` stays at the Pi0Config default **32** (matches the canonical b1k checkpoints).
- `fsdp_devices = 2` (this 2-GPU pod; was 4 on the old pod) shards the ~2.4B model + fp32 Adam
  optimizer states across both H100s at `batch_size=32` (was 64), keeping per-GPU VRAM
  ~50 GB — fits beside the model server's ~28 GB/GPU. `num_workers=8` (20-core cgroup quota).

**VRAM:** full FT at batch 32 is lighter than the old 4-GPU batch-64 run but still far heavier
than the LoRA era. Tune the XLA preallocation pool with `XLA_PYTHON_CLIENT_MEM_FRACTION`
(default **0.60** ≈ 49 GB/GPU in `run_waves.sh`). The the model server model server (~28 GB/GPU) is
**kept running** on this pod; if you free it, the fraction can go back up toward 0.75.

## Dataset staging — Option B implemented (2026-08-12)

Earlier drafts of the frozen split made every wave/family train on the full **18,500-episode
corpus**, which the incremental per-wave/per-family staging could never satisfy on this pod's
~226 GB disk — lerobot would have attempted to download the whole ~3 TB on demand. **This is
resolved by intersecting the frozen split with what staging actually downloads.**

`_make_wave_configs()` and `_make_family_configs()` in `src/openpi/training/config.py` now
subset the shared frozen split **before** building each `TrainConfig`:

- **Waves** — keep frozen train/val episodes whose `demo = id % 200` lies in that wave's
  `demo_lo..demo_hi` window (`stage_wave.py --lo/--hi` downloads exactly this window for all
  100 tasks). So each wave trains on its own frozen∩window subset (~735–935 train eps).
- **Families** — keep frozen episodes whose demo ∈ `[demo_lo,demo_hi)` **and** `task = id//200`
  ∈ the family's `task_ids` (`stage_family.py --tasks` downloads exactly this set). So each
  family trains on frozen∩task∩window (~74–290 train eps).

Because every indexed episode is staged, the loader never pulls media on demand. Verification
(disjoint train/val, window/task membership, per-wave/per-family counts) passed for all 8 waves
and all 5 families. The family ledger's own `train_episodes`/`val_episodes` lists are a legacy
contiguous-tail variant and are **ignored** whenever the frozen split exists (it always does).

**Dataset & disk facts (verified on this pod):**
- The lerobot cache `/root/.cache/huggingface/lerobot/behavior-1k/2026-challenge-demos` starts
  **empty**; all media is fetched on demand from HF `behavior-1k/2026-challenge-demos`
  (RGB-only — depth is skipped by `stage_wave.py` / `stage_family.py`).
- Waves: `stage_wave.py` downloads one demo window at a time (**~66 GB** for 800 eps, **~82 GB**
  for 1000 eps). Peak disk ≈ one window + the current wave's kept checkpoint (~13 GB) + the arm0
  warm-start base (~7 GB) ≈ **~110 GB**.
- Families: `stage_family.py --tasks` downloads a task-filtered slice of demos[90,100) — backbone
  (32 t) ≈ **~26 GB**, F2/F1 (25–26 t) ≈ ~20–21 GB, F3 (13 t) ≈ 11 GB, F4 (8 t) ≈ 6.5 GB.
- Disk is **not** unlimited on this pod — it filled to 100% on 2026-08-13 (`OS error 122:
  EDQUOT Disk quota exceeded` crashed wave3's checkpoint write at step 5000) and again on
  2026-08-15 (EDQUOT crashed wave7 at step 5000; completed-wave checkpoints 5 & 6 + stale media
  were the culprits). Freeing space is a real, recurring need. Keep the `model-server/` and `the model server`
  model-server files, but do not let the lerobot media cache and completed-wave checkpoints
  accumulate: `reclaim.py` (run before each wave) frees stale media, and the orchestrator prunes
  a completed wave's local checkpoint only *after* its upload succeeds **and** its name is in
  `REMOTE_COMPLETE_WAVES`. After the 2026-08-15 cleanup (deleted wave5 & wave6 local checkpoints,
  stale wave7 `5000.orbax-checkpoint-tmp-*` leftovers, and 237 stale `data/*.parquet` files) the
  pod has **~81 GB free**. Disk can be checked at any time with `df -h /`.

  **Disk-management gotcha:** `reclaim.py` only frees stale **`.mp4` video** files (it filters
  on `.mp4`), so old-wave `data/*.parquet` (depth/observation) accumulate and are never auto-
  reclaimed. To find & free them manually, run (from the `b1k_waves/` dir) the union of
  `stage_wave.wave_files` for the *incomplete* waves (currently wave7 = demos[80,90) and wave8 =
  demos[90,100)) and delete any `data/chunk-*/file-*.parquet` + `videos/.../file-*.mp4` on disk
  outside that set. Keep `REMOTE_COMPLETE_WAVES` in `wave_status.py` in sync with every new
  upload, or the prune safety-gate will keep uploaded waves' checkpoints forever and the disk
  will fill again.

## Launch checklist (current run)

> **2026-08-27 pod:** paths below are the old pod's. On this machine the openpi root is
> `/tmp/b1k/BEHAVIOR-1K/b1k-baselines/baselines/openpi`, check disk with `df -h /tmp`, and the
> VRAM co-tenant is **the model server (~28 GB/GPU, kept running)** — prealloc is 0.60, not 0.75.
> Verified-good quick check: `bash smoke_test.sh` from the openpi root (100-step smoke run).

**Check the GPU first — `the inference server` may be holding ~45 GB/GPU and silently kills full FT.**

1. Check free VRAM. If < ~60 GB/GPU, free it (the standalone inference server only — never the agent's model server):
   ```bash
   nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader   # expect ~80 GB free/GPU
    pkill -f "inference-server"     # frees ~45 GB/GPU (do this yourself — an agent must not kill it on your behalf)
   ```
   A full-FT launch with insufficient VRAM dies **silently** (no traceback) right after the
   dataset-init logs, while XLA preallocates ~60 GB/GPU. The whole `behavior` tmux pane/process
   group dies. Symptom: `train.py` disappears moments after `hf_dataset cache HIT`, no error in
   `train_wave*.log`. (Alternatively, if you instead see `Cuda failure 2 'out of memory'` in the
   NCCL logs, that is the same the inference server VRAM contention mid-training.)
2. **CPU thread exhaustion is fixed in `run_waves.sh`** (caps `OPENBLAS_NUM_THREADS` /
   `OMP_NUM_THREADS` / `MKL_NUM_THREADS` — **20** on this pod, was 40). If a wave still dies at
   a checkpoint save with `std::system_error: Resource temporarily unavailable` /
   `pthread_create failed`, that is the 20-core cgroup quota (`/sys/fs/cgroup/cpu.max`) being
   over-subscribed; verify the thread caps are applied on the `train.py` line in
   `run_waves.sh` before re-launching.
3. Launch the wave chain in the `behavior` tmux session (already created; the shell is at the
   openpi root):
   ```bash
   tmux send-keys -t behavior 'bash b1k_waves/run_waves.sh 2>&1 | tee ~/train_pipeline.log' C-m
   ```
    (To create it fresh if missing, on this pod:
    `tmux new-session -d -s behavior -c /tmp/b1k/BEHAVIOR-1K/b1k-baselines/baselines/openpi`)
 4. `run_waves.sh` auto-fetches the arm0 warm-start base from HF (idempotent), then runs
    **wave5 → wave8** in order (wave1–wave4 are already COMPLETE and skipped). Confirm
    progress:
   ```bash
   tail -f b1k_waves/run_waves.log          # orchestrator decisions
   tail -f b1k_waves/train_wave*.log        # "[I] Progress on: Xit/15.0kit ..."
   .venv/bin/python b1k_waves/wave_status.py
   ```
5. After all waves are `COMPLETE`, run the family (STAR) pipeline — see
   `b1k_families/OPERATOR_RUNBOOK.md` §3.

## What the pipeline does

`run_waves.sh` is meant to be started once (login, cron `@reboot`, or a systemd
unit) and left alone. Per the script's own header, it re-derives all state from
disk on every iteration — checkpoints, the episode ledger, and the lerobot cache —
never from its own memory, so it is safe to kill and restart at any point.

Loop, per wave:
1. **lerobot version guard** (`ensure_lerobot.py`) — the real outages this campaign
   hit were a stray `uv sync` silently downgrading lerobot (breaking v3.0 dataset
   layout), and -- after `uv.lock` moved to the wensi-ai/lerobot `release/b1k` fork
   (0.5.2) -- a stale PyPI 0.4.4 install lacking the fork's `accelerate` runtime dep,
   which breaks the data-loader import path. The guard compares the installed version
   to the one pinned in `uv.lock` and, on mismatch, runs `uv sync --frozen` to rebuild
   the exact locked env. Checked before anything else runs.
2. **Reclaim disk** (`reclaim.py`) — deletes cached media not needed by any
   incomplete wave.
3. **Stage media** (`stage_wave.py`) for the wave's demo range, with retries.
3b. **Compute norm stats** (`compute_norm_stats.py --config-name pi05_b1k`) — a long-running IO-heavy pass over all staged episodes (~22 k on arm0). Run inside a tmux session so it survives disconnects; progress can be monitored with `tmux capture-pane -t <session> -p`. All waves share the same norm stats from the arm0 assets dir (`outputs/assets/pi05_b1k/`), so this only needs to run once before any wave. If running for a fresh base config, expect 3–4 h on a single CPU.
4. **Warm-start**: the **config default chain** (`_make_wave_configs()`) wires each wave to inherit the **local** previous-wave checkpoint (wave1 → arm0 base, which `run_waves.sh` fetches from HF via `snapshot_download()` if absent). No `B1K_WARM_START_PARAMS` override is needed. GPU memory reserved with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.60` (~49 GB) on this pod; with `fsdp_devices=2` / batch 32 full-FT the per-GPU footprint fits beside the model server's ~28 GB.
5. **Attach-or-launch**: if a `scripts/train.py <config>` process for this wave is already running (e.g. launched manually, or by a still-running earlier invocation), it waits on that pid instead of starting a second one — two processes writing the same orbax checkpoint dir would corrupt it and fight over GPU memory.
6. **Launch/resume in the foreground** with up to 3 crash-retries, always with `--resume` (safe unconditionally — empty checkpoint dir means fresh warm start from the prior wave's params, partial dir means resume that wave).
7. **Verify completion from disk**, not from `train.py`'s exit code — a caught exception can still exit 0. `wave_status.py --check <wave>` checks whether `{checkpoint_dir}/{final_step}/params` exists.
8. **Upload to the gated repo** (`upload_wave.py`) — pushes the completed wave's `params/` + `assets/` into `0Corvid0/pi05-b1k-waves/{name}_{demo_hi}ep`. Upload failure is non-fatal (logs a warning, checkpoint stays on disk for re-upload). The old org repo `IntelligentDecisionLab/pi05-b1k-monolithic-model` can no longer receive uploads — it is private and has hit its private-storage limit (HTTP 403).
9. **Prune older local checkpoints**: after successful upload, deletes all `outputs/checkpoints/pi05_b1k_wave*/` dirs except the current wave's own dir (kept as the next wave's local fallback). Frees ~13 GB per pruned wave.
10. Repeat until `next_wave_info.py` reports `DONE`.

A `flock` on `run_waves.lock` makes a second concurrent invocation of the script exit immediately instead of racing the first.

### Warm-start pitfalls (read before debugging wave failures)

**Never symlink params into `~/.cache/openpi/...`**. `download.py:64` calls `.resolve()` which follows symlinks to their real target, then `_should_invalidate_cache()` (`download.py:192`) does `relative_to(cache_dir)` — if the target is outside cache (e.g. `/root/BEHAVIOR-1K/...`), this raises a fatal `ValueError`. A 6.7 GB symlink attempted as a "pre-seeded cache" hit exactly this in Aug-2026 and crashed all 3 launch retries. **Use real copies if you must pre-populate the cache**, but prefer local-first warm-start which avoids any download entirely.

**`hf://` directory downloads fail silently on this pod**. `download.py:114-121` runs `fs.get()` in a thread and never calls `future.result()`, so exceptions are swallowed; `params.partial` is never created and `shutil.move` at line 92 raises `FileNotFoundError`. This is why warm-start is **local-first** and `run_waves.sh` fetches the arm0 base with the `snapshot_download()` function (`python -c`; the `hf download` CLI also works) rather than fsspec `hf://`. The fsspec bug lives upstream in openpi's `src/openpi/shared/download.py`; if/when it is fixed, hf:// can again be used for the base.

## Ground truth: ledger vs. filesystem

`episode_ledger.json` records **intent** (seed 20260721, 100 tasks × 200
episodes/task, which demo indices belong to which wave). `wave_status.py`
treats the **filesystem** as the fact of record and syncs the ledger's
`status` field to match on every check — so the ledger stays a human-readable
log but is never trusted for the actual go/no-go decision. (Its `status`
field can lag the live `tqdm` step count between syncs. That is expected
staleness, not a bug.)

## Prior run: `arm0_monolithic` and the old supervisor

Before this wave pipeline existed, a simpler per-process supervisor script ran the original monolithic
baseline (`pi05_b1k`, demos 0–30, 50,000 steps) on the prior 2× H100 box. (The archived copy was
`supervise_arm0.sh`; monitoring is now done interactively by polling `wave_status.py`,
`nvidia-smi`, and log tails — there are no helper monitor scripts in this folder.) Its log showed a restart loop
from 2026-07-21 21:32Z to 2026-07-22 03:13Z: the trainer had already reached
`50.0kit/50.0kit` (i.e. finished) each time, but exited fast enough after
finishing that the supervisor's liveness check misread it as a crash and
relaunched with `--resume` roughly every 13 minutes. This was harmless —
`--resume` against a complete checkpoint is a no-op — but cosmetically noisy
in `supervisor.log`. That supervisor process has been superseded by `run_waves.sh`;
`arm0_monolithic` is confirmed complete at step 49999 and excluded from the wave loop.
The arm0 checkpoint is preserved in HF as `arm0_monolithic_30ep/params` under
`IntelligentDecisionLab/pi05-b1k-monolithic-model`. (The legacy
`pi05-b1k-arm0-monolithic-bucket-30ep` bucket repo now 404s and must not be used.)

## Key files

- `run_waves.sh` — orchestrator entrypoint.
- `episode_ledger.json` — wave definitions and synced status.
- `ledger.py` / `wave_status.py` / `next_wave_info.py` — ledger I/O and
  completion logic.
- `stage_wave.py` / `reclaim.py` / `ensure_lerobot.py` — data-staging and env
  guards run before each wave.
- Logs: `train_wave{N}_d{lo}_{hi}.log` and `run_waves.log` live in this directory (`b1k_waves/`). Training progress appears as `[I] Progress on: Xit/Ykit rate:Z it/s` lines (not `Step N:` format).
- **On this pod**, training runs inside the tmux session named `behavior`. Detach with Ctrl-b d.
- Checkpoints: `../outputs/checkpoints/pi05_b1k_wave{N}_d{lo}_{hi}/wave{N}_d{lo}_{hi}/{step}/` (~13 GB per wave's kept checkpoint; `keep_period` prunes intermediate steps). Pruned after upload to the gated waves repo.
- Config: `../src/openpi/training/config.py`, `_make_wave_configs()` — wave `TrainConfig`s are generated programmatically (batch size 32, `pi05` model, `action_horizon=32`, `fsdp_devices=2`, `num_workers=8`, **full FT `gemma_2b`**) rather than hand-declared per wave.

## Verifying status / continuing waves

**Check if training is running:**
```bash
cd /tmp/b1k/BEHAVIOR-1K/b1k-baselines/baselines/openpi

# 1. Is train.py alive?
ps aux | grep 'train\.py' | grep -v grep

# 2. GPU utilization (running = ~100% on all 4 GPUs, ~62 GB each)
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# 3. Current progress (tail the log for latest step count and ETA):
tail -5 b1k_waves/train_wave6_d70_80.log

# 4. Warm-start source used (from run_waves.log):
grep 'warm-start:' b1k_waves/run_waves.log | tail -3

# 5. Wave status from ledger:
.venv/bin/python b1k_waves/wave_status.py
```

**Healthy signs:**
- `ps aux` shows PID with high CPU (300%+) and running `scripts/train.py`.
- All 4 GPUs at ~100% util, ~62 GB each.
- Log tail shows `[I] Progress on: Xit/15.0kit rate:... remaining:...`.

**Startup timing:** Wave7 launched at 2026-08-02T06:00 UTC, first training step logged at 06:12 — **~12 min cold start** (stage_wave + hf_dataset cache materialization + warm-start weight restore + XLA pmap compile). The `pipe_write` state during XLA compilation is expected and can last this long; do not kill the process before ~15 min from launch.

**Unhealthy signs:**
- GPU memory at model load (~78 GB) but utilization at 0% for >15 min → likely stuck in warm-start download or XLA compile deadlock (check log for Traceback).
- `train_wave*.log` shows `FileNotFoundError` on `params.partial` → a stale `hf://` warm-start was attempted. Fix: confirm `run_waves.sh` is using the local config-default chain (no `B1K_WARM_START_PARAMS` override) and that the arm0 base / prior-wave checkpoint is on disk; remove stale cache at `~/.cache/openpi/IntelligentDecisionLab/...`.
- No PID running but wave not complete → pipeline stopped; re-run `bash b1k_waves/run_waves.sh` to resume.

**Continue training after disconnect/reboot:**
```bash
tmux send-keys -t behavior 'cd /tmp/b1k/BEHAVIOR-1K/b1k-baselines/baselines/openpi && bash b1k_waves/run_waves.sh 2>&1 | tee ~/train_pipeline.log' C-m
```
The script is idempotent: it picks up the next incomplete wave and resumes from the last local checkpoint. Waves advance automatically — when the current wave completes, `run_waves.sh` uploads it to HF, prunes old checkpoints, then starts the next wave (no manual intervention needed).

## Verifying status yourself

```bash
cd /tmp/b1k/BEHAVIOR-1K/b1k-baselines/baselines/openpi
.venv/bin/python b1k_waves/wave_status.py          # synced ledger status + next wave
nvidia-smi                                          # GPU utilization/memory
tmux attach -t behavior                             # live trainer output (detach: Ctrl-b d)
tmux capture-pane -t behavior -p -S -40             # tail the pane without attaching
```

Healthy signs (what a *running* run looks like):
- The `train_wave*.log` tail shows the progress line
  `[I] Progress on: Xit/15.0kit rate:... remaining:...` (the current format; the older
  `Step N: grad_norm=..., loss=...` + tqdm bar was from the LoRA-era run and is not produced now).
- `nvidia-smi` shows one `scripts/train.py` process on **every** GPU, each at
  ~100% util and ~62 GB memory (fits beside the model server's ~16 GB on an 80 GB H100).
- Loss in the low tens on step 0 for a warm-started model, then decreasing.

Two gotchas learned the hard way:
1. **Don't conclude "it's running" from init logs alone.** Lines like
   `CheckpointManager created`, `Loaded norm stats`, and
   `data_config: DataConfig(...)` are all printed *before* the model is
   restored and before any GPU work. The first real confirmation is a
   `Step 0:` line in the pane, plus nonzero util in `nvidia-smi`.
2. **A traceback may linger in the tmux scrollback** from a previous crashed
   run and show up in `capture-pane -S <large-N>`. Always check the tail
   (`-S -40`) and the current PID before judging the run's health.

## HF checkpoints

**The old org repo `IntelligentDecisionLab/pi05-b1k-monolithic-model` is private and has hit
its private-storage limit, so it can no longer receive uploads (HTTP 403 on the LFS batch
endpoint).** Completed-wave uploads now go to a **public + gated** repo owned by `0Corvid0`
— public so it avoids the private-storage cap, gated so only approved accounts can access it:

- **Waves:** `0Corvid0/pi05-b1k-waves` → folders `{wave_name}_{demo_hi}ep/`
- **Families:** `0Corvid0/pi05-b1k-families` → folders `{family_name}_100ep/`
  (separate repo; `family_status.py`, `upload_family.py`, and `run_family_experts.sh` all
  target it.)

Only the `arm0_monolithic_30ep/` base remains on the org repo
(`IntelligentDecisionLab/pi05-b1k-monolithic-model`) and is deliberately **kept** there —
it is the pretrained warm-start base the wave chain inherits from; do not remove it.

As of 2026-08-15 the waves repo holds the full-FT checkpoints uploaded so far (wave5 & wave6
are now uploaded too; wave7 is training locally at step 2500 and not yet uploaded, wave8 is
still pending):

| Folder | Wave | Demos (per task) | Steps | Content | Status |
|---|---|---|---|---|---|
| `arm0_monolithic_30ep/` (org repo) | arm0 | 0–30 | 49,999 | params + assets | base (not re-trained) |
| `wave1_d30_38_38ep/` | wave1 | 30–38 | 14,999 | params + assets | COMPLETE — uploaded |
| `wave2_d38_46_46ep/` | wave2 | 38–46 | 14,999 | params + assets | COMPLETE — uploaded |
| `wave3_d46_54_54ep/` | wave3 | 46–54 | 14,999 | params + assets | COMPLETE — uploaded |
| `wave4_d54_62_62ep/` | wave4 | 54–62 | 14,999 | params + assets | COMPLETE — uploaded |
| `wave5_d62_70_70ep/` | wave5 | 62–70 | 14,999 | params + assets | COMPLETE — uploaded |
| `wave6_d70_80_80ep/` | wave6 | 70–80 | 14,999 | params + assets | COMPLETE — uploaded |
| — (not yet uploaded) | wave7 | 80–90 | 2,500 | params + assets (local) | IN_PROGRESS — resume from disk |

The old LoRA-era `wave5/6/7/8` folders were **deleted** from the org repo on 2026-08-13
(superseded by the full-FT re-run); the full-FT re-run re-creates them in the waves repo as
each wave completes.

Each folder contains `params/` (deployable OCDBT weights) and `assets/` (norm stats). `train_state/` is excluded to keep uploads small.

### Usage

**Warm-start training** (set env var before launching train.py or run_waves.sh for manual/debug overrides):
```bash
# Local-first is used automatically by run_waves.sh (config default chain). Manual override:
B1K_WARM_START_PARAMS="./outputs/checkpoints/pi05_b1k/arm0_monolithic/49999/params"
```

The local path is passed to `CheckpointWeightLoader` directly. Avoid the `hf://` fsspec form (silently fails on this pod); use the `hf download` CLI to materialize a base checkpoint to disk first.

**Serving a policy** (from the gated waves repo; access must be approved first):
```bash
--policy.dir=hf://0Corvid0/pi05-b1k-waves/wave2_d38_46_46ep/params
```

## Manual launch & continuing training

The gated waves repo (`0Corvid0/pi05-b1k-waves`) replaces the old per-wave repos. Wave `TrainConfig`s are auto-generated from `episode_ledger.json` by `_make_wave_configs()` in `src/openpi/training/config.py`. Warm-start uses the **config default chain** — each wave inherits the previous wave's local params (wave1 → the arm0 base). You can still override with `B1K_WARM_START_PARAMS` (a local path) for manual/debug runs; the `hf://` fsspec form is unreliable on this pod.

**Launching via run_waves.sh (recommended — sequential, self-managing):**
```bash
cd /tmp/b1k/BEHAVIOR-1K/b1k-baselines/baselines/openpi
tmux new-session -d -s behavior "bash b1k_waves/run_waves.sh"   # or attach an existing 'behavior' session
tmux send-keys -t behavior "bash b1k_waves/run_waves.sh" C-m
```

The script fetches the arm0 base from HF if absent, then handles warm-start (config default chain, local), training, upload to the gated waves repo, and pruning of older checkpoints. Waves 1–8 run in order.

**Manual single-wave launch (advanced):**
```bash
cd /tmp/b1k/BEHAVIOR-1K/b1k-baselines/baselines/openpi
B1K_WARM_START_PARAMS="./outputs/checkpoints/pi05_b1k/arm0_monolithic/49999/params" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.60 \
.venv/bin/python scripts/train.py pi05_b1k_wave1_d30_38 --no-wandb-enabled
```

Notes:
- **Full FT + FSDP (2-GPU pod).** `_b1k_wave_model()` + generated configs use `fsdp_devices=2`,
  sharding the 2.4B model + fp32 Adam states across both GPUs. `jax.device_count()` = 2;
  `batch_size=32` → 16 samples/GPU (same per-GPU sample count as the old 4-GPU batch-64 run).
- **`XLA_PYTHON_CLIENT_MEM_FRACTION=0.60` (~49 GB)** is the preallocation pool; with
  `fsdp_devices=2` full-FT fits beside the model server's ~28 GB/GPU (kept running on this pod). If you
  free the model server, the fraction can go back up toward 0.75.
- Each wave is 15,000 steps on **its own frozen∩demo-window train subset** (~735–935 episodes;
  see "Dataset staging — Option B implemented"). To download data without training, run
  `stage_wave.py` directly — do **not** run `run_waves.sh`, which stages **and trains**.

## The checkpoint download trap (and how to verify integrity)

Symptom: training dies moments after the init logs with

```
ValueError: FAILED_PRECONDITION: Error reading "params.PaliGemma.img.
Transformer.encoderblock.MultiHeadDotProductAttention_0.out.kernel.value/0.1.0.0"
in OCDBT database at local file ".../intelligent_decision_lab/params/":
Truncated Zstd-compressed stream; at byte 0
```

Root cause: the HF checkpoint download was **incomplete** — one LFS chunk
(`params/ocdbt.process_0/d/aebc8d45b9dcca4ac4de0bcbd96965ad`, ~1.1 GB) was
missing. The dir "looked" complete (5.7 GB) and even the HF hub cache was
present as a stub (`models--.../refs/main` with no blobs). Orbax's OCDBT
database then fails on the truncated zstd stream at restore time.

Verify a downloaded checkpoint before trusting it:

```bash
cd outputs/checkpoints/intelligent_decision_lab
# 1. Every LFS-tracked chunk declared in .gitattributes must exist on disk
for f in $(grep -o "params/ocdbt.process_0/d/[a-f0-9]*" .gitattributes); do
  [ -f "$f" ] || echo "MISSING: $f"
done
# 2. Sanity-check total size (complete wave5 checkpoint ≈ 6.7 GB; a partial
#    download can look large while still missing a 1+ GB chunk)
du -sh .                                   # expect ~6.7G
ls params/ocdbt.process_0/d/ | wc -l       # must equal the count in .gitattributes
# 3. The HF hub cache should contain the blobs, not just a refs stub
ls ~/.cache/huggingface/hub/models--0Corvid0--pi05-b1k-waves/blobs/
```

Fix: delete the incomplete dir and re-download so the hub cache materializes
the missing chunks:

```bash
rm -rf outputs/checkpoints/intelligent_decision_lab
.venv/bin/hf download 0Corvid0/pi05-b1k-waves \
  --local-dir outputs/checkpoints/intelligent_decision_lab --repo-type model
```

Then re-run the integrity checks above before relaunching training.
