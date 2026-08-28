# b1k_families — Operator Runbook

Foolproof, step-by-step guide to launch, monitor, troubleshoot, and resume the
BEHAVIOR-1K **backbone_foundation** fine-tune. This is the operational playbook;
`b1k_families/README.md` is the design reference.

> ## 🔄 POD MOVED (2026-08-27): read before any command below
>
> This runbook was written for the old 4×H100 pod. On this 2×H100, 20-core pod:
> - openpi root is **`/tmp/b1k/BEHAVIOR-1K/b1k-baselines/baselines/openpi`** (replace every
>   `/root/BEHAVIOR-1K/...` below);
> - `fsdp_devices=2`, `batch_size=32`, `num_workers=8` (were 4/64/32);
> - VRAM co-tenant is **the model server, ~28 GB/GPU, kept running** — expect **~52 GB free/GPU**, not
>   "~55–60 GB after killing inference server". `preflight.py` floor is now **0.59** (live FRAC
>   ≈ 0.593); disk is checked on **`/tmp/hf-cache`** (14 TB), not `/`;
> - disk is no longer tight: the demos[90,100) window (111 GB) is **already staged** and the
>   wave8 warm-start params (11 GB) are **already fetched** to the conventional path.
> See `b1k_waves/AGENT_PRIMER.md` for the full old-vs-new table.

> ## ✅ CURRENT STATE (2026-08-21): CAMPAIGN COMPLETE — backbone + F1..F4 all trained & uploaded
>
> **Phase 1 + Phase 2 are DONE.** All five checkpoints (`backbone_foundation` + `F1..F4`, each
> 15,000 steps / step 14999) are complete and uploaded to `0Corvid0/pi05-b1k-families/<name>_100ep/`.
> **Local checkpoints have been CLEANED** (`cleanup_family.py --yes` on all five + the wave8 warm-start
> dir) — completion is verified **disk-only**, so re-running `run_family_experts.sh` now would find no
> local checkpoints and **retrain from scratch**. Do NOT relaunch unless you intend a full re-run;
> to resume experts, first restore their `params/` from HF per §3a-style download.
>
> ---
> ## 🗄️ PRIOR STATE (2026-08-19): RESET → single backbone_foundation
>
> The b1k_families campaign was **reset**. The old **5-expert STAR pipeline**
> (`backbone_foundation` → `F1..F4`) is **archived**, not active:
>
> - HF archive: `0Corvid0/pi05-b1k-families/archive_2026-08-16/` holds the 5 old
>   checkpoints (`backbone_foundation_100ep/`, `F1..F4_100ep/`) moved there on 2026-08-19.
> - Local snapshot of the old ledger: `b1k_families/family_ledger.old.json`.
>
> **Phase 1 (`backbone_foundation`) — COMPLETE, verified 2026-08-20.** It is a **full-FT
> monolithic policy over all 100 tasks** (demos 90–100, 15,000 steps), warm-started from the
> **wave8** checkpoint on HF (`0Corvid0/pi05-b1k-waves/wave8_d90_100_100ep/params`). Trained to
> step 14999 and uploaded to `0Corvid0/pi05-b1k-families/backbone_foundation_100ep/`; the
> active ledger now reports it `COMPLETE (step 14999)`.
>
> **Phase 2 — NEXT: F1..F4 force-expert families**, each warm-started from that backbone.
> Copy-paste ledger snippet and launch commands are in the **Phase 2 (§6B)** section below.
> The local checkpoint at `outputs/checkpoints/pi05_b1k_backbone_foundation/backbone_foundation/14999`
> is their ONLY warm-start source — do **NOT** delete it (`cleanup_family.py`) until all four
> experts are uploaded to HF. (The archived old backbone was a different, 32-task model.)
>
> - **Wave chain is COMPLETE** (`next_wave_info` → `DONE`; all 8 waves recorded complete).
>   The local wave7/wave8 checkpoints were deleted for disk; the wave8 warm-start is
>   re-downloaded from HF (step 1). Do **not** re-train waves.
> - **Model config is FULL FT**: `_b1k_wave_model()` = `Pi0Config(pi05=True, action_horizon=32)`
>   → `gemma_2b` (no LoRA), `fsdp_devices=2` on this 2-GPU pod (was 4). See
>   `b1k_waves/WAVE_TRAINING.md` "Model config (full FT)".
> - Training data = **frozen split ∩ all tasks ∩ demos[90,100)** (Option B). The frozen split
>   (`b1k_waves/frozen_val_split.json`) is intersected per family in `config.py`.
> - Free the GPU (stop the standalone inference server) before launching.

**Every command is copy-paste safe.** Run each from the openpi repo root:

```bash
cd /tmp/b1k/BEHAVIOR-1K/b1k-baselines/baselines/openpi
```

---

## 1. What this pipeline does (read once)

**Phase 1 (done)** — trains **one** policy, `backbone_foundation`, on **all 100 tasks**
(demos 90–100):

```
0Corvid0/pi05-b1k-waves
   wave8_d90_100_100ep (full-FT) ──► pi05_b1k_backbone_foundation  (15,000 steps · all 100 tasks)
                                        │
                                        └─► upload to 0Corvid0/pi05-b1k-families/backbone_foundation_100ep/
```

**Phase 2 (next)** — the four force-expert families warm-start from that backbone:

    pi05_b1k_backbone_foundation ──► F4_heavy_grasp → F3_surface_contact
                                     → F2_actuation_transfer → F1_constrained_insertion
                                          │
                                          └─► upload each to 0Corvid0/pi05-b1k-families/<name>_100ep/

See **§6B** for the Phase 2 steps. Invariant across both phases:

- All models are **full fine-tuning (no LoRA)** — `_b1k_wave_model()` = `Pi0Config(pi05=True,
  action_horizon=32)` → `gemma_2b`, `fsdp_devices=4` (verified in `src/openpi/training/config.py`).
- **Warm-start** uses only the source's `params/` dir (`CheckpointWeightLoader`) — optimizer state
  and LR-schedule position start clean at each stage.
- **Training data** = frozen split ∩ family tasks ∩ demos[90,100) (Option B).
- **Completion** is verified **from disk** (`{checkpoint_dir}/14999/params`), never from
  `train.py`'s exit code.

The orchestrator (`run_family_experts.sh`) is generic over the ledger: it iterates whatever
families are listed (a `while true` loop until DONE). Today only `backbone_foundation` is in the
ledger; §6B-2 appends F1..F4 before the Phase 2 launch.

---

## 2. Pre-flight: prepare GPUs + environment

The orchestrator's `preflight.py` blocks launch if GPUs are over-subscribed or the disk is
too full. Check everything **before** starting training.

### 2a. Find VRAM-hogging processes — one thing (on this pod)

This pod runs **the model server** plus the trainer. (The old pod also had a standalone inference server hog at 40+ GB/GPU — it does not exist here.)

| Thing | Binary / process | Parent | Typical VRAM | Must you stop it? |
|---|---|---|---|---|
| **Model-server daemon** | `the model server daemon` | init/supervisor | ~0 (daemon) | **NO — keep it running** |
| **Model-server inference child** | `the model server's inference process` | `the model server daemon` | **~28 GB/GPU** when a model is loaded | **NO — training is tuned to run alongside it** |

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
nvidia-smi
ps aux | grep -iE 'inference|model[- ]server' | grep -v grep
```

### 2b. (Old pod) Stop the VRAM hogs

**Not applicable on this pod** — the model server is the only co-tenant and it stays up. If the user
ever frees it, do so by PID (`kill <pid>`), never `pkill` by pattern.

### 2c. Confirm free VRAM meets the requirement

`preflight.py` computes `FRAC = (free_MiB - safety) / total` and aborts if below its floor.
**This pod's floor is `0.59`** (in `preflight.py`, or override with `B1K_MEM_FRACTION_FLOOR`);
with the model server resident the live FRAC is ≈ 0.593 and passes. Verify **at least ~47 GB free per
GPU** (the model server ~28 GB + prealloc ~49 GB + safety):

```bash
nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader
```

### 2d. Confirm free disk

Preflight aborts if free disk `< 40 GB` (env `B1K_MIN_DISK_GB`). **On this pod the check
runs on the data volume `/tmp/hf-cache` (14 TB), not `/`** (a 66 GB PVC holding only the
backup repo + small caches):

```bash
df -h /tmp | tail -1
```

Current pod state (2026-08-27): the demos[90,100) window is **already staged (111 GB)** and
the wave8 warm-start params are **already fetched (11 GB)**. A fresh 5-family re-run needs
~29 GB per final checkpoint.

### 2e. HuggingFace token (for uploads + the wave8 download)

Training does not need a token (the demo dataset is public). Downloading wave8 and
uploading the finished checkpoint need auth. A valid token cached at
`~/.cache/huggingface/token` works without `HF_TOKEN` set (the current account is
`0Corvid0`); preflight may still print `WARN: HF_TOKEN not set` — that is a **false alarm**
when the cached token exists.

### 2f. Dry-run preflight (optional but recommended)

Point it at the wave8 warm-start params dir you download in step 1:

```bash
.venv/bin/python b1k_families/preflight.py \
  --warm-start ./outputs/checkpoints/pi05_b1k_wave8_d90_100/wave8_d90_100/14999/params
```

- Exit **0** → all hard checks pass; last line prints `FRACTION=<float>`.
- Exit **1** → a hard check failed; output names the reason (VRAM/disk/lerobot/warm-start). See §5.

---

## 3. Start training — Phase 1: backbone_foundation (DONE)

Steps below are the record of how Phase 1 was launched (completed & uploaded 2026-08-20).
**For Phase 2 (F1..F4) skip to §6B and run only that section's snippet + launch.**

### 3a. Download the wave8 warm-start from HF

The local wave8 checkpoint was deleted for disk. Download its `params/` from HF to the
conventional warm-start path:

```bash
mkdir -p outputs/checkpoints/pi05_b1k_wave8_d90_100/wave8_d90_100/14999
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="0Corvid0/pi05-b1k-waves",
    allow_patterns="wave8_d90_100_100ep/params/*",
    local_dir=".wave8_dl",
)
PY
mv .wave8_dl/wave8_d90_100_100ep/params \
   outputs/checkpoints/pi05_b1k_wave8_d90_100/wave8_d90_100/14999/params
rm -rf .wave8_dl
```

> **Gotcha:** the `mkdir -p` must include the `14999` level — the `mv` target is a
> directory, so its parent `.../14999/` must already exist or the `mv` fails with
> "No such file or directory".

Verify the params dir exists (the warm-start argument is the **params directory itself**):

```bash
ls -la outputs/checkpoints/pi05_b1k_wave8_d90_100/wave8_d90_100/14999/params
```

### 3b. Populate the reset ledger with the backbone_foundation entry

`config.py` generates the training config **from `b1k_families/family_ledger.json`**. The
ledger is reset (`families: []`), so add one entry: **all 100 tasks**, demos 90–100,
15,000 steps, warm-started from wave8. Global episode index = `200 × task + demo`.

```bash
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "b1k_families")
import family_ledger as L
d = L.load()
d["wave8_final_params"] = "./outputs/checkpoints/pi05_b1k_wave8_d90_100/wave8_d90_100/14999/params"
# Phase 1 set this to "" (only backbone_foundation was trained). It MUST be re-set to the
# local backbone params before launching F1..F4 — see §6B.
d["backbone_final_params"] = ""   # Phase 1: unused. Phase 2: REQUIRED for F1..F4 warm-start.
d["families"] = [{
    "name": "backbone_foundation",
    "task_ids": list(range(100)),   # all 100 tasks
    "demo_lo": 90,
    "demo_hi": 100,
    "num_train_steps": 15000,
    "warm_start": "wave8",
    "status": "QUEUED",
}]
L.save(d)
PY
```

Confirm the config resolves and monitoring shows exactly one family, QUEUED:

```bash
.venv/bin/python b1k_families/family_status.py
.venv/bin/python b1k_families/next_family_info.py   # NAME=backbone_foundation, WARM=<wave8 params path>
```

### 3c. Launch the pipeline (in tmux)

> Old pod: free the GPU first (standalone inference server). **This pod: no action needed — the model server
> stays running and training is tuned for it.** The wave gate (`next_wave == DONE`) is
> already satisfied.

Start the orchestrator in a detached tmux session so it survives your terminal closing.
Prefer the long-lived `behavior` session via `tmux send-keys`; otherwise (this pod's path):

```bash
tmux new-session -d -s behavior_family \
  'cd /tmp/b1k/BEHAVIOR-1K/b1k-baselines/baselines/openpi && bash b1k_families/run_family_experts.sh 2>&1 | tee ~/family_train_pipeline.log'
```

Attach / detach anytime:

```bash
tmux attach -t behavior_family        # watch live
# Ctrl-b then "d"                       ...detach and leave it running
tmux ls                               # confirm the session exists
```

> `run_family_experts.sh` is **single-instance** (flock on `b1k_families/run_families.lock`).

### What it does automatically, in order

1. `next_family_info.py` → decides the next family (here: `backbone_foundation`) or `DONE`.
2. `preflight.py` → VRAM/disk/lerobot/cwd/warm-start checks; computes the XLA memory fraction.
3. `ensure_lerobot.py` → enforces the lerobot version pinned in `uv.lock` (currently the
   wensi-ai/lerobot `release/b1k` fork, 0.5.2 — v3.0 dataset layout support); on mismatch it
   runs `uv sync --frozen`.
4. `stage_family.py --tasks 0..99 --lo 90 --hi 100` → downloads demos[90,100) media for all
   100 tasks (idempotent; files already on disk are skipped).
5. `scripts/train.py pi05_b1k_backbone_foundation --resume --no-wandb-enabled` → trains to
   15,000 steps; completion verified from disk.
6. `upload_family.py` → pushes the checkpoint to HF (non-fatal; keeps local on failure).
7. **No auto-delete.** The checkpoint persists until you run `cleanup_family.py --yes`.

---

## 4. Live monitoring & health checks

### 4a. Ledger status

```bash
.venv/bin/python b1k_families/family_status.py
```

Healthy state:

```
backbone_foundation      tasks(100)  QUEUED

next to run: backbone_foundation
```

Status values: `QUEUED` · `IN_PROGRESS (step N)` · `COMPLETE (step 14999)`.

### 4b. What runs next?

```bash
.venv/bin/python b1k_families/next_family_info.py
```

Prints `KEY=VALUE` lines (or `DONE`):

```
NAME=backbone_foundation
CONFIG=pi05_b1k_backbone_foundation
TASKS=0,1,2,3,...,99
LO=90
HI=100
STEPS=15000
WARM=./outputs/checkpoints/pi05_b1k_wave8_d90_100/wave8_d90_100/14999/params
```

### 4c. Tail live logs

```bash
tail -f b1k_families/run_families.log          # orchestrator (preflight/stage/upload decisions)
tail -f b1k_families/train_*.log               # the training loop
tail -f ~/family_train_pipeline.log            # everything the tmux process printed
```

### 4d. Healthy-execution signs

| Signal | Healthy state |
|---|---|
| `nvidia-smi` | ~100% GPU utilization on **both** GPUs, ~77 GB VRAM used per GPU (the model server ~28 + trainer ~49) |
| Orchestrator log | `lerobot OK (0.5.2)`, `FRACTION=<float>` (≈0.593 on this pod), `warm-start: ... -> OK` |
| Training log | `[I] Progress on: 123it/15.0kit rate:... remaining:...` |

15000 steps → the total reads `15.0kit`. At ~1.3 it/s on the old 4×H100 box, one run ≈
~3.2 h; **budget ~2× that on this 2-GPU pod** (batch 32, lower per-step rate — estimate).

---

## 5. Failure recovery & resume

### 5a. The pipeline is idempotent — just re-run it

Kill it, the pod reboots, a crash occurs — none of it corrupts state. The script
re-derives progress from disk each iteration (checkpoint dir, ledger, lerobot cache).
Re-run the tmux command from §3c and it resumes from where it left off.

**Resume mechanics:** every launch uses `--resume`. A missing checkpoint dir → fresh
warm-start from wave8 params; a partial dir → resumes from the latest completed checkpoint
step. If complete, the orchestrator skips and prints `all families complete. nothing to do.`
— **run it again anytime, it is always safe.**

**Startup-hang rule of thumb (learned 2026-08-20):** a healthy start prints its first
`Progress on:` line within ~10–15 min of launch (JAX startup + data-loader init). If the train log
stops at `local_batch_size: ...` / scipy-sklearn import tracebacks with **0% GPU util and zero log
growth for >20 min**, it is deadlocked — just `kill <train.py pid>` and re-run the orchestrator;
it retries (attempt N/3) or you relaunch tmux, both safe.

### 5b. Troubleshooting `preflight.py` aborts (exit 1)

| FATAL message | Meaning | Fix |
|---|---|---|
| `computed FRAC X < floor 0.590` | Not enough free VRAM | Ask the user to free VRAM (the model server is the co-tenant), re-check (§2c), re-run |
| `disk: X GB free at /tmp/hf-cache (require 40 GB)` | Disk too full | Free ≥ 40 GB on the data volume, re-run (rare on 14 TB) |
| `lerobot wrong/missing (need 0.5.2)` | Wrong lerobot | Run `b1k_waves/ensure_lerobot.py`, re-run |
| `warm-start: <path> -> MISSING` | Wave8 params not downloaded | Do §3a, re-run |

After fixing, **just re-run the orchestrator** — it resumes, it does not restart.

### 5c. If the warm-start checkpoint seems missing

The warm-start argument is the **params directory itself**:

```bash
ls -la outputs/checkpoints/pi05_b1k_wave8_d90_100/wave8_d90_100/14999/params
```

If missing, re-download per §3a (the local wave8 was deleted for disk; it lives on HF).

### 5d. `family_status.py --check` exit codes

```bash
.venv/bin/python b1k_families/family_status.py --check backbone_foundation
```

- Exit **0** → `COMPLETE (step 14999)`.
- Exit **1** → not complete yet (`QUEUED` / `IN_PROGRESS`).
- Exit **2** → unknown family name.

---

## 6. Architecture reference & task mapping

`backbone_foundation` trains on **all 100 tasks**, demos 90–100, from the wave8 checkpoint.
`config.py` intersects the shared frozen split (`b1k_waves/frozen_val_split.json`) with
`task_ids ∩ demos[90,100)` (Option B), so every indexed episode is staged by
`stage_family.py` (no on-demand corpus pull). Global episode index = `200 × task + demo`
(task = `id // 200`, demo = `id % 200`).

| Family | n_tasks | Task IDs |
|---|---|---|
| `backbone_foundation` | 100 | 0,1,2,…,99 |

Source of truth: `b1k_families/family_ledger.json` (episodes, warm-start). Config name:
`pi05_b1k_backbone_foundation`.

The old 5-expert STAR checkpoints (old 32-task `backbone_foundation` + old `F1..F4`) are
**archived**: HF `0Corvid0/pi05-b1k-families/archive_2026-08-16/` and local ledger snapshot
`b1k_families/family_ledger.old.json`. Do **not** use the archived old backbone as a warm-start —
it is a different (32-task) model. Phase 2 re-trains `F1..F4` on top of the NEW full-FT
100-task backbone; see §6B.

---

## 6B. Phase 2 — train F1..F4 force-expert families from the new backbone

> **Status: NEXT.** Phase 1 (`backbone_foundation`) is COMPLETE and uploaded (2026-08-20).
> Phase 2 trains four expert families, each warm-started from the NEW local backbone
> checkpoint. The orchestrator already supports this end-to-end — only the ledger entry is missing.

**How it works:** `config.py` resolves an expert's warm-start from the ledger field
`backbone_final_params` (STAR topology: backbone ← wave8, experts ← backbone). Each expert trains
demos 90–100 on its own task subset; frozen-split intersection and staging are automatic via
`stage_family.py --tasks ...`. The experts are **independent of each other** — list order in the
ledger is execution order (recommended: F4→F3→F2→F1, matching the archive's upload order). One
tmux launch runs all four back-to-back (`run_family_experts.sh` loops until DONE).

### 6B-1. Free the GPU

**This pod: nothing to free** — the model server stays running and the pipeline is tuned for it (floor
0.59). Just confirm ≥ ~47 GB free per GPU (§2c) and ≥ 40 GB on `/tmp/hf-cache` (§2d). Disk is
**not** tight on this pod (14 TB data volume); the §6B-5 cleanup guidance is optional, not
required.

### 6B-2. Wire the ledger (copy-paste; run from openpi repo root)

```bash
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "b1k_families")
import family_ledger as L
d = L.load()
# REQUIRED: experts warm-start from the NEW full-FT 100-task backbone (local, step 14999).
# Do NOT point at HF or at the archived old (32-task) backbone.
d["backbone_final_params"] = "./outputs/checkpoints/pi05_b1k_backbone_foundation/backbone_foundation/14999/params"
existing = {f["name"]: f for f in d["families"]}
specs = [
    ("F4_heavy_grasp",          [8, 14, 15, 16, 23, 58, 74, 88]),
    ("F3_surface_contact",      [31, 32, 33, 36, 37, 57, 60, 68, 69, 71, 79, 80, 93]),
    ("F2_actuation_transfer",   [0, 3, 10, 17, 25, 30, 38, 39, 40, 41, 42, 43, 44, 45, 46, 49,
                                 51, 62, 66, 72, 79, 90, 95, 96, 98]),
    ("F1_constrained_insertion",[4, 5, 9, 10, 11, 12, 19, 22, 23, 26, 34, 35, 48, 55, 56, 58,
                                 63, 67, 75, 77, 78, 81, 86, 89, 92, 99]),
]
for name, tasks in specs:
    if name not in existing:   # idempotent — safe to re-run after partial progress
        d["families"].append({
            "name": name,
            "task_ids": tasks,
            "demo_lo": 90,
            "demo_hi": 100,
            "num_train_steps": 15000,
            "warm_start": "backbone",
            "status": "QUEUED",
        })
L.save(d)
print("ledger updated:", [f["name"] for f in d["families"]])
PY
```

Verify: `next_family_info.py` must print `NAME=F4_heavy_grasp ... WARM=./outputs/checkpoints/
pi05_b1k_backbone_foundation/backbone_foundation/14999/params`. If `WARM=` is empty, the ledger
edit did not take — do NOT launch (an expert would silently fresh-init instead of warm-starting).

### 6B-3. Launch (same tmux pattern as §3c)

```bash
tmux new-session -d -s behavior_family \
  'cd /tmp/b1k/BEHAVIOR-1K/b1k-baselines/baselines/openpi && bash b1k_families/run_family_experts.sh 2>&1 | tee ~/family_train_pipeline.log'
```

The orchestrator loops `F4 → F3 → F2 → F1` (each ~15,000 steps ≈ 3.3 h on the old 4×H100 pod;
**budget ~2× on this 2-GPU pod**), then uploads each on completion and exits DONE.
Monitoring is identical to §4: `family_status.py`, `next_family_info.py`,
`tail -f b1k_families/train_F*.log` (fresh log per expert).

### 6B-4. Per-expert upload & disk management

After EACH expert's `upload_family.py` succeeds, verify the folder on HF
(`0Corvid0/pi05-b1k-families/<name>_100ep/`) and then free its ~29 GB:

```bash
.venv/bin/python b1k_families/cleanup_family.py F4_heavy_grasp --yes   # only after upload verified
```

On the old pod disk was the binding constraint (~100 GB free; 4 experts ≈ +116 GB if never
cleaned). **On this 14 TB pod it is not** — cleaning after upload is optional (still a good
habit); the pipeline stays idempotent (§5a) either way.

**Never delete the backbone checkpoint itself until ALL four experts are COMPLETE on disk AND
uploaded:** `cleanup_family.py backbone_foundation --yes` only at the very end.

### 6B-5. Completion criteria (Phase 2)

All of: `family_status.py` shows F1..F4 `COMPLETE (step 14999)`; next_family_info prints DONE;
HF top-level folders include all four `<name>_100ep`; README table rows appended by the uploader.

---

## 7. Post-training: upload & cleanup

Upload the completed checkpoint to the gated repo `0Corvid0/pi05-b1k-families` (needs HF
auth — §2e). This is a full-FT re-run; it uploads to the same `backbone_foundation_100ep/`
folder name the archive used, but now under the repo root:

```bash
.venv/bin/python b1k_families/upload_family.py backbone_foundation
```

Uploads `params/` + `assets/` → `0Corvid0/pi05-b1k-families/backbone_foundation_100ep/`
and appends a row to the repo README. Fails (exit 1) if the family isn't complete on disk.
`--repo-id` overrides the target repo.

Delete the local checkpoint **only after** confirming the upload:

```bash
.venv/bin/python b1k_families/cleanup_family.py backbone_foundation --yes
```

- Refuses (exit 1) without `--yes`.
- Refuses to delete a family that is not COMPLETE on disk.
- `--require-upload` only checks COMPLETE-on-disk and prints a WARN — upload status is not
  tracked in the ledger; confirm uploads manually first.

Verify it landed:

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import HfApi
info = HfApi().model_info("0Corvid0/pi05-b1k-families", repo_type="model")
folders = sorted({s.rfilename.split('/')[0] for s in info.siblings
                  if s.rfilename not in ("README.md", ".gitattributes")})
print("gated:", info.gated)
print("top-level folders:", folders)
PY
```

Expected: `gated: auto` and `folders == ['archive_2026-08-16', 'backbone_foundation_100ep']`
(the archive holds the old 5 checkpoints; the new `backbone_foundation_100ep` is the fresh
full-FT run). If the folder is missing, re-run the upload.

> **HF storage note (free account `0Corvid0`):** the new run replaces
> `backbone_foundation_100ep` in-place (net storage ≈ unchanged at ~134 GB). The old F1–F4
> checkpoints remain under `archive_2026-08-16/` as the durable archive.

---

## 8. Quick reference

| Action | Command |
|---|---|
| Download wave8 warm-start | see §3a |
| Populate ledger (backbone_foundation) | see §3b |
| Launch pipeline (detached) | `tmux new-session -d -s behavior_family 'cd /tmp/b1k/BEHAVIOR-1K/b1k-baselines/baselines/openpi && bash b1k_families/run_family_experts.sh 2>&1 \| tee ~/family_train_pipeline.log'` |
| Attach / detach tmux | `tmux attach -t behavior_family` / `Ctrl-b d` |
| LEDGER status | `.venv/bin/python b1k_families/family_status.py` |
| Next to run | `.venv/bin/python b1k_families/next_family_info.py` |
| Orchestrator log | `tail -f b1k_families/run_families.log` |
| Training log | `tail -f b1k_families/train_*.log` |
| Verify family | `.venv/bin/python b1k_families/family_status.py --check backbone_foundation` |
| Preflight dry-run | `.venv/bin/python b1k_families/preflight.py --warm-start <wave8 params_dir>` |
| Show GPU/VRAM | `nvidia-smi --query-gpu=index,memory.total,memory.free,utilization.gpu --format=csv` |
| Show GPU procs | `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader` |
| Stop VRAM hog | (this pod) none — the model server stays running; old pod: `kill <pid>` (never `the model server`) |
| Set HF token | `export HF_TOKEN=hf_your_token` |
| Upload backbone | `.venv/bin/python b1k_families/upload_family.py backbone_foundation` |
| Verify gated-upload | see §7 snippet (prints `gated` + folders) |
| Delete family ckpt | `.venv/bin/python b1k_families/cleanup_family.py backbone_foundation --yes` |
