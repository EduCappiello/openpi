# b1k-waves — Agent Primer

**Read this BEFORE `WAVE_TRAINING.md`. It tells you the current state of affairs and where to look.**

## What's happening right now (2026-08-15)

**This is the FULL fine-tuning campaign** (`gemma_2b`, full FT, `action_horizon=32`,
`fsdp_devices=4`). It is **mid-way through the wave chain**, not deferred.

| Wave | Status | Notes |
|---|---|---|
| arm0_monolithic (demos 0–30) | base | pretrained warm-start base, auto-fetched from HF |
| wave1_d30_38 (30–38) | **COMPLETE (step 14999)** | full-FT, uploaded to HF; excluded from re-train |
| wave2_d38_46 (38–46) | **COMPLETE (step 14999)** | full-FT, uploaded to HF; excluded from re-train |
| wave3_d46_54 (46–54) | **COMPLETE (step 14999)** | full-FT, uploaded to HF; excluded from re-train |
| wave4_d54_62 (54–62) | **COMPLETE (step 14999)** | full-FT, uploaded to HF; excluded from re-train |
| wave5_d62_70 (62–70) | **COMPLETE (step 14999)** | full-FT, uploaded to HF; excluded from re-train |
| wave6_d70_80 (70–80) | **COMPLETE (step 14999)** | full-FT, uploaded to HF; excluded from re-train |
| wave7_d80_90 (80–90) | **IN_PROGRESS (step 2500)** | resumes from local checkpoint at step 2500 |
| wave8_d90_100 (90–100) | **QUEUED** | run after wave7 completes |
| STAR families (F1..F4) | QUEUED | gated until all waves COMPLETE |

**wave1–wave6 are done and uploaded to `0Corvid0/pi05-b1k-waves`** (folders
`wave1_d30_38_38ep` … `wave6_d70_80_80ep`, step 14999). Each uploaded wave's local checkpoint was
pruned (or manually deleted 2026-08-15 — see below), and all six are registered in
`REMOTE_COMPLETE_WAVES` in `wave_status.py` — otherwise the disk-only completeness check would
wrongly mark them QUEUED and the orchestrator would re-train already-consumed demos (re-staging
~206 GB and re-hitting EDQUOT).

**wave7 is the active wave.** Its valid local checkpoint is at
`outputs/checkpoints/pi05_b1k_wave7_d80_90/wave7_d80_90/2500` (~30 GB). It is NOT yet uploaded.
It warm-starts from wave6's final checkpoint (uploaded; wave6's local copy was deleted in the
2026-08-15 cleanup).

### ⚡ Current operational status (what to do right now)

**Training is currently PAUSED — no `train.py` is running.** `run_waves.sh` stopped with
`FATAL: wave7_d80_90 did not complete after 3 launch attempts` because the **disk filled to
100%** (see below). To resume:

1. Check VRAM: `nvidia-smi --query-gpu=index,memory.free --format=csv,noheader`. If free VRAM
   is < 60 GB/GPU, another process is hogging it — **tell the user it must be killed**; do NOT
   kill it yourself.
2. Confirm disk headroom: `df -h /` — expect ~81 GB free (freed by the 2026-08-15 cleanup). The
   step-5000 checkpoint write for wave7 needs ~30 GB free, and wave8's staging needs ~26 GB, so
   re-check `df -h /` before and during wave8.
3. Once VRAM is ~80 GB free/GPU, launch the pipeline:
   `tmux send-keys -t behavior 'bash b1k_waves/run_waves.sh 2>&1 | tee ~/train_pipeline.log' C-m`
4. Confirm it actually started (not just init logs): `ps aux | grep 'train\.py'` + GPU util ~100%
   on all 4 GPUs. First training step appears after a ~12 min cold start; don't kill before ~15 min.
5. Monitor with `.venv/bin/python b1k_waves/wave_status.py` and `tail -f b1k_waves/run_waves.log`.

`run_waves.sh` will skip wave1–wave6 (COMPLETE), then **resume wave7 from step 2500** and run
wave7 → wave8 automatically (stage → train → upload to `0Corvid0/pi05-b1k-waves` → prune).

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

- **Pod**: 4× H100. Another workload may hold ~45 GB/GPU.
  Check `nvidia-smi` and kill it (`kill <pid>`) if free VRAM is < ~60 GB/GPU — full FT needs
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.75` (~60 GB/GPU) and dies silently (no traceback) if it
  can't preallocate. Do not kill it on behalf of the user; tell them it needs freeing.
- **Full FT, not LoRA.** `_b1k_wave_model()` returns `Pi0Config(pi05=True, action_horizon=32)`
  → `paligemma_variant="gemma_2b"` (no LoRA), so `get_freeze_filter()` is `nnx.Nothing`
  (everything trainable). Configs set `fsdp_devices=4` to shard the ~2.4B model + fp32 Adam
  states across the 4× H100.
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
cd /root/BEHAVIOR-1K/b1k-baselines/baselines/openpi
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader   # free VRAM ~80 GB/GPU = ready; <60 GB = another process hogging
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
