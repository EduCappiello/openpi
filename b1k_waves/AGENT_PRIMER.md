# b1k-waves — Agent Primer

**Read this BEFORE `WAVE_TRAINING.md`. It tells you the current state of affairs and where to look.**

## What's happening right now (2026-08-28)

> **2026-08-28 — merged `train` (allenwang's correlated-noise work) + this pod's 2-GPU state.**
> `git pull`/merge brought in the `EduCappiello/openpi` `train` tip `00f75a7` ("feat: add noise
> test") and re-applied the local 2-GPU config values. What is new in this tree:
>
> - **Correlated flow-matching noise** — `src/openpi/training/noise.py` (new module:
>   `load_cholesky` + `sample_correlated_noise`). `train.py` threads a precomputed Cholesky
>   factor of a shrinkage-regularized empirical action covariance through `train_step`, and
>   `Pi0.compute_loss` now accepts an optional `noise=` override (default behavior unchanged:
>   iid N(0,I)). Inference matches via `Policy(noise_cholesky=...)` / `policy_config`.
> - **4 new task-0 comparison configs** — `pi05_b1k_task0_{lora,full}_{gauss,corr}` in
>   `config.py` (`_make_task0_noise_configs`): {LoRA, Full-FT} × {Gaussian, correlated} on a
>   100-demo task-0 subset (90 train / 10 val). `common` has `noise_real_action_dim=23`,
>   `val_log_interval=200`, `val_num_batches=5`.
> - **`scripts/compute_action_covariance.py`** (new) — builds the `action_cholesky.npy` the
>   `*_corr` configs need (run it AFTER `compute_norm_stats.py --config-name
>   pi05_b1k_task0_lora_gauss`). **`scripts/serve_b1k_patched.py`** (new) — serves policies
>   against the 2026-era `omnigibson.eval` layout.
> - **Validation loop in `train.py`** — `val_step` + `_make_val_data_loader` read
>   `config.val_repo_id`/`val_episodes_index` (previously unused) and log `val_loss` every
>   `val_log_interval` steps. Val always uses iid noise, so it's comparable across variants.
> - **Dependency changes (allenwang's `pyproject.toml`/`uv.lock`)** — lerobot moved from the
>   HF `huggingface/lerobot` rev to **`wensi-ai/lerobot` branch `release/b1k` (0.5.2)**;
>   `transformers==5.5.4` (was 4.53.2); `pandas>=2.2.3`; torch built cu128.
> - **Fix applied in this merge (2026-08-28)** — the fork's `lerobot.utils.utils` imports
>   `accelerate` at module load, but `accelerate` only lived in the fork's `training`/`smolvla`
>   extras (openpi requests `lerobot[dataset]`), so a clean `uv sync --frozen` produced a broken
>   env (`ModuleNotFoundError: No module named 'accelerate'`). `accelerate>=1.10.0,<2.0.0` is
>   now declared in openpi's `pyproject.toml` + `uv.lock`. Also: `ensure_lerobot.py` no longer
>   force-downgrades to PyPI `lerobot==0.4.4` (`--no-deps`) — it now enforces the version
>   pinned in `uv.lock` and runs `uv sync --frozen` on mismatch; `preflight.py` reads the same
>   pinned version instead of hard-coding `0.4.4`.
> - **Hardware in configs = this pod (2-GPU).** Wave/family blocks use `fsdp_devices=2`,
>   `batch_size=32`, `num_workers=8` (the old 4-GPU values are gone with the old pod). The 4
>   task0 configs use `fsdp_devices=1` (lora) / `2` (full).
> - **task0 asset paths (fixed 2026-08-28):** allenwang's configs pointed at
>   `/mnt/train-data-1-hdd/...` (his storage, absent here). Now local:
>   `assets_base_dir=./outputs/assets`, `checkpoint_base_dir=./outputs/checkpoints`,
>   norm stats at `./outputs/assets/pi05_b1k_task0_lora_gauss` (all 4 configs share it —
>   compute ONCE with the `lora_gauss` name), `noise_cholesky_path=
>   ./outputs/assets/pi05_b1k_task0/action_cholesky.npy` (same default in
>   `scripts/compute_action_covariance.py`). Run order per config:
>   `compute_norm_stats.py --config-name pi05_b1k_task0_lora_gauss` →
>   `compute_action_covariance.py` → the 4 `train.py` runs. Data/val episodes auto-fetch
>   from HF (verified). Also fixed a merge bug: the `_CONFIGS = [...]` line was
>   duplicated (17 configs double-registered in the list).
> - **`family_ledger.json` reset** (allenwang's commit): F1–F4 `COMPLETE (step 14999)` →
>   `QUEUED` (backbone stays COMPLETE). If a families re-run is intended, this is the starting
>   state; local checkpoints for F1–F4 were pruned on this pod.

> **⚠️ Pod-compat state (2026-08-28): this pod runs `exp/pod-compat-2gpu`, NOT `train`.**
> The 2-GPU values, `/mnt→./outputs` task0 path fix, `accelerate` dep fix, guard updates,
> and the duplicate-`_CONFIGS` bug fix were moved OFF `train` and live on
> **`exp/pod-compat-2gpu`** (open a PR → train per the policy below). Remote `train` is
> back at allenwang's `00f75a7` (his code, his 4-GPU path defaults — it does NOT run on
> this pod as-is: no `accelerate` in the lock, `/mnt/train-data-1-hdd` paths, `fsdp_devices=4`).
> The local working tree is checked out on `exp/pod-compat-2gpu` so training keeps working.
> When the PR merges: `git fetch origin && git merge origin/train` (or rebase the exp
> branch) and switch back to `train`.

### How to open the pod-compat PR (when allenwang is around)

This repo is a **fork** (`Physical-Intelligence/openpi` → `wensi-ai/openpi` →
`EduCappiello/openpi`), so GitHub's PR form may pre-fill "base repository" with
`wensi-ai/openpi` or `Physical-Intelligence/openpi`. **That is wrong for our purposes** —
the PR must stay inside `EduCappiello/openpi` (the repo allenwang works in). Steps:

1. Open **https://github.com/EduCappiello/openpi/compare/train...exp/pod-compat-2gpu?expand=1**
   (this URL already has both sides set correctly inside the right repo).
2. Verify the form reads: **base repository `EduCappiello/openpi` : branch `train`** ←
   **head repository `EduCappiello/openpi` : branch `exp/pod-compat-2gpu`**.
   If the base repository shows `wensi-ai/...` or `Physical-Intelligence/...`, change the
   "base repository" dropdown to `EduCappiello/openpi`.
3. Title: `pod-compat: 2-GPU values + task0 local paths + accelerate/_CONFIGS fixes`.
   Body: "Pod-compatibility for the 2xH100 pod (the old 4-GPU pod is gone): 2-GPU hardware
   values, task0 configs `/mnt/train-data-1-hdd` paths -> local `./outputs/...`, added
   `accelerate` to deps (the lerobot fork imports it at module load), fixed a duplicated
   `_CONFIGS` line. No new training method or checkpoint. Verified with a 100-step smoke
   run on 2xH100 (exit 0, val_loss logged)."
4. Create the PR and tag allenwang. He reviews + merges; then:
   `git fetch origin && git checkout train && git merge origin/train`.

### 🚦 Branching & PR policy (2026-08-28, from allenwang)

Training and evaluation are being kept isolated. `train` is the **training** branch and is
currently compatible with the **evaluation** branch. The rules:

1. **`train` is the integration point.** Do not commit new training-method experiments
   directly to `train`.
2. **To try a different training method: branch off `train`** (e.g.
   `git checkout -b exp/<name> train`), develop + train there.
3. **New checkpoint / working result → open a PR against `train`.** allenwang reviews it
   and pulls whatever **inference-side** code the evaluation needs from the PR into the
   evaluation branch himself. So: the PR must be self-contained (training config, model
   changes, any inference/serve changes it implies, and the checkpoint location), and you
   do NOT edit the evaluation branch.
4. Keep the evaluation branch isolated — never push training experiment code to it.

This keeps the codebase maintainable: `train` stays a clean, known-good training state;
each experiment is a reviewable branch/PR; evaluation stays decoupled and only receives
vetted inference-side pieces.

**Practical consequence for the correlated-noise work:** the 4 `pi05_b1k_task0_*`
configs on `train` are the shared baseline. If you run variants (different beta, noise
schedule, LoRA rank, etc.), do it on an `exp/*` branch and PR the winner — do not mutate
the task0 configs on `train` directly.

**Campaign state: ALL WAVES AND ALL FAMILIES are COMPLETE (step 14999) and uploaded to HF.**
There is nothing left in the queue — `run_waves.sh` is a no-op and
`run_family_experts.sh` would retrain from scratch (local checkpoints are pruned; completion is
disk-only via `REMOTE_COMPLETE_WAVES` / the ledgers).

**This work has moved to a NEW machine** (reinstalled 2026-08-27 from the personal backup).
The old 4×H100 pod is gone. Key differences from every doc written before 2026-08-27:

| | Old pod (≤ 2026-08-15) | This pod (2026-08-27) |
|---|---|---|
| GPUs | 4× H100 80 GB | **2× H100 80 GB** (driver 570.86.10, CUDA 12.8) |
| CPU | 40-core cgroup quota | **20-core** cgroup quota (`/sys/fs/cgroup/cpu.max` = 2000000/100000); RAM ~2 TB |
| Root disk | ~226 GB, data on `/` | `/` is a **66 GB** gpfs PVC (code/small caches only); **data, venv, checkpoints live on `/tmp` (14 TB LV)** |
| Working copy | `/root/BEHAVIOR-1K/...` | **`/tmp/b1k/BEHAVIOR-1K/b1k-baselines/baselines/openpi`** (re-clone + overlay per the backup README) |
| Full-FT config | `fsdp_devices=4`, batch 64, `num_workers=32` | **`fsdp_devices=2`, batch 32, `num_workers=8`** (edited in `config.py` for this pod) |
| XLA mem fraction | 0.75 | **0.60** in `run_waves.sh`; families: `preflight.py` computes ~0.593 live, floor lowered to **0.59** |
| VRAM co-tenant | a standalone inference server ~45 GB/GPU | **the model server** (`the model server's inference process`) **~28 GB/GPU — kept running on purpose**; ~52 GB free/GPU for training |
| Thread caps | 40 | **20** (capped in both orchestrators) |
| HF dataset cache | `~/.cache/b1k_hf_subset` on `/` | **`B1K_HF_DATASET_CACHE=/tmp/hf-dataset-cache`** (set in both orchestrators) |

Path indirections (symlinks, keep them): `~/.cache/huggingface/lerobot → /tmp/hf-cache/lerobot`
(lets the hardcoded `ROOT` in `stage_wave.py`/`stage_family.py` land on the big volume) and
`~/.cache/jax → /tmp/jax-cache`. The HF token lives at `~/.cache/huggingface/token`
(auto-discovered; the demo dataset is public, the arm0 base repo is private).

**Verified on this pod (2026-08-27):** 100-step smoke run of `pi05_b1k_wave8_d90_100`
(warm-started from the arm0 base) — both GPUs at ~100% util, loss 0.0425 → 0.0199, checkpoint
written, then pruned. Re-verify anytime with `bash smoke_test.sh` (repo root). `check_env.py`
passes. Staged data: wave8 demo window (1,000 eps) = **111 GB** on `/tmp`.

**Re-verified post-merge (2026-08-28):** after pulling allenwang's `train` (lerobot 0.5.2
fork, transformers 5.5.4, `accelerate` dep fix), `uv sync --frozen` → `check_env.py` →
`preflight.py --warm-start ...` → `ensure_lerobot.py` all pass, and a 100-step
`smoke_test.sh` run of `pi05_b1k_wave8_d90_100` trains through the new data-loader (lerobot
0.5.2 reads the v3.0 demo dataset fine). All 45 configs import, including the 4 new
`pi05_b1k_task0_*` noise configs.

### ⚡ Current operational status (what to do right now)

Nothing is running. If you start a NEW training run (e.g. retrain a wave/family, or a new
experiment):

1. Check VRAM: `nvidia-smi --query-gpu=index,memory.free --format=csv,noheader`. Expect ~52
   GB free/GPU (the model server holds ~28 GB/GPU by design). If free VRAM < ~47 GB/GPU, ask the user to
   stop the model server — do NOT kill it yourself.
2. Confirm disk headroom on the DATA volume: `df -h /tmp` (not `/` — `/` is a 66 GB PVC).
3. Launch in tmux from the openpi root:
   `tmux send-keys -t behavior 'cd /tmp/b1k/BEHAVIOR-1K/b1k-baselines/baselines/openpi && bash b1k_waves/run_waves.sh 2>&1 | tee ~/train_pipeline.log' C-m`
4. Confirm it actually started (not just init logs): `ps aux | grep 'train\.py'` + GPU util ~100%
   on both GPUs. First training step appears after a ~12 min cold start; don't kill before ~15 min.
5. Monitor with `.venv/bin/python b1k_waves/wave_status.py` and `tail -f b1k_waves/run_waves.log`.

**Batch-size caveat (2 GPUs):** the configs keep 15,000 steps but batch 32 (was 64), so a
re-run sees **half the samples** of the original 4-GPU run. If matching the original data
coverage matters, double `num_train_steps` (and the cosine `decay_steps`) for that run.

### ⚠️ 2026-08-15 disk-full (EDQUOT) incident & cleanup

wave7 crashed at its step-5000 checkpoint write with
`OS error 122: EDQUOT Disk quota exceeded` on all 3 launch attempts → `FATAL`. The disk had
filled to 100% because completed waves 5 & 6 (29 GB each) and a large chunk of stale media were
still on disk. **Recovery (2026-08-15):**

- Deleted local checkpoints for **wave5 & wave6** (both verified uploaded to
  `0Corvid0/pi05-b1k-waves`), the stale wave7 `5000.orbax-checkpoint-tmp-*` leftovers, and 237
  stale `data/*.parquet` files not needed by waves 7/8 (~18.9 GB). Freed **~82 GB** total.
- Added **`wave5_d62_70` & `wave6_d70_80` to `REMOTE_COMPLETE_WAVES`** (`wave_status.py`) so the
  prune safety-gate in `run_waves.sh` releases them on future runs — this was the gap that let
  the disk fill: wave5/6 were uploaded but not in that set, so the prune kept them as local
  fallback forever.

**Two recurring disk-pressure root causes to keep in mind:**
1. A completed wave's checkpoint lingers unless its name is in `REMOTE_COMPLETE_WAVES`. Keep that
   set in sync with every new upload (`wave_status.py`), or the disk quietly fills again.
2. `reclaim.py` only deletes stale **`.mp4` video** files, never `data/*.parquet` — so old-wave
   depth/observation parquet accumulate. Reclaim manually as needed (see the stale-file computation
   in `WAVE_TRAINING.md` "Disk management").

### Key facts to know

- **Pod (2026-08-27+)**: 2× H100 80 GB, 20-core cgroup quota. **the model server** holds ~28 GB/GPU and
  is kept running (the agent's model server) — full FT runs alongside it at
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.60` (~49 GB/GPU prealloc; ~52 GB free). If it can't
  preallocate, training dies silently (no traceback). Do not kill the model server on behalf
  of the user; tell them it needs freeing.
- **Full FT, not LoRA.** `_b1k_wave_model()` returns `Pi0Config(pi05=True, action_horizon=32)`
  → `paligemma_variant="gemma_2b"` (no LoRA), so `get_freeze_filter()` is `nnx.Nothing`
  (everything trainable). Configs set `fsdp_devices=2` and `batch_size=32` to shard the ~2.4B
  model + fp32 Adam states across the 2× H100 (both GPUs verified at ~100% util, 2026-08-27).
- **Frozen validation split** (`b1k_waves/frozen_val_split.json`): 18,500 train / 1,500 val
  episodes (~15/task), disjoint. `config.py` then intersects it with each run's staged
  subset — per wave (`frozen ∩ demo_lo..demo_hi`) and per family (`frozen ∩ task_ids ∩
  demo_lo..demo_hi`) — so the trainer only references media `stage_wave.py`/`stage_family.py`
  actually download (Option B; ~735–935 train eps/wave, ~74–290 train eps/family). No on-demand
  corpus pull. **Disk is a real constraint on this pod** — it filled to 100% on 2026-08-13
  (EDQUOT crashed wave3 at step 5000) and again on 2026-08-15 (EDQUOT crashed wave7 at step
  5000; waves 5/6 checkpoints + stale media were the culprits). Keep the lerobot media cache and
  completed-wave checkpoints pruned: `reclaim.py` runs before each wave (but only frees `.mp4`,
  see note), and the orchestrator deletes a completed wave's local checkpoint only after its
  upload succeeds AND its name is in `REMOTE_COMPLETE_WAVES`. After the 2026-08-15 cleanup the
  pod has **~81 GB free** (check anytime with `df -h /`).
- **Warm-start is LOCAL-FIRST.** The fsspec `hf://` directory download silently fails on this
  pod. `run_waves.sh` fetches the arm0 params once via `snapshot_download` and chains each
  wave from the previous wave's local checkpoint. Never use symlinks for cache pre-seeding
  (a symlink resolving outside `~/.cache/openpi` raises a fatal `ValueError`).
- **arm0 warm-start base** lives at
  `outputs/checkpoints/pi05_b1k/arm0_monolithic/49999/params` (auto-downloaded from
  `IntelligentDecisionLab/pi05-b1k-monolithic-model/arm0_monolithic_30ep/params` by
  `run_waves.sh` on first run).

### Quick status check

```bash
cd /tmp/b1k/BEHAVIOR-1K/b1k-baselines/baselines/openpi
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader   # ~52 GB free/GPU = ready (the model server resident); <47 GB = ask user to free VRAM
df -h /tmp                                                                   # data volume (14 TB); NOT df -h /
tmux ls                                                                # 'behavior' session = where run_waves.sh runs
tmux capture-pane -t behavior -p -S -40                                 # live trainer output without attaching
ps aux | grep 'train\.py' | grep -v grep                                # is training actually running?
ls outputs/checkpoints/pi05_b1k/arm0_monolithic/49999/params            # arm0 warm-start base present?
.venv/bin/python b1k_waves/wave_status.py                               # waves ledger status
.venv/bin/python b1k_families/family_status.py                          # families ledger status
```

## Full documentation

For complete pipeline design, warm-start gotchas, checkpoint integrity checks, and HF repo
details: read **`WAVE_TRAINING.md`** in this directory.
