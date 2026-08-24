# BEHAVIOR-1K Training Split Guide

**For:** Charles
**Context:** Monolith π0.5 baseline retrain + follow-up experiments (torque sidecar, BDDL stage separation)

---

## TL;DR

Before staging any data for the monolith retrain, carve out and freeze a small internal validation slice (~15 episodes/task, ~7.5% of the corpus). Every subsequent training run — monolith, +torque, +torque+BDDL, and eventually the force-routed MoE — trains on the remaining episodes and gets checked against this same frozen slice. Do this **now, before the monolith trains on anything**, not after.

---

## 1. What this slice is for

We have two axes we want to add on top of the monolith baseline: torque sidecar information and BDDL-based stage separation. Before spending the expensive official-style live-rollout eval (~362s wall-clock/episode, ~7x slower than real-time) on every variant, we want a fast, cheap internal signal for whether each addition is actually helping.

This slice is **not** a replacement for the official BEHAVIOR evaluation. The official challenge scores models on live simulator rollouts under held-out initial-condition instances generated at eval time (10 per task, via BDDL sampling) — those are never sourced from the demonstration corpus at all, so there's no leakage risk there regardless of what we do here. This internal slice exists purely so we can iterate quickly and cheaply before committing to a full sim-rollout eval run.

## 2. Why it has to be carved out *before* the monolith trains — not after

If the monolith trains on all 20,000 episodes first, and we designate a slice as "validation" afterward, the monolith has already seen and partially memorized those specific trajectories. Its score on that slice is inflated — it's measuring recall, not generalization. Two things go wrong from there:

- If we then exclude that slice from the torque/BDDL fine-tuning runs, we're comparing a monolith that's seen the eval data against variants that haven't. Any apparent difference is contaminated by that asymmetry, not by the thing we're actually trying to measure.
- If we don't bother excluding it from later runs either (the easier mistake, since it's already staged), nothing was ever actually held out, and the whole exercise was pointless.

So the slice needs to be defined and frozen **before staging starts**, so that the monolith, both fine-tuned variants, and later the MoE all train on the exact same (20k − slice) corpus and are all checked against the exact same untouched episodes. That's what makes the internal numbers comparable across every model in the pipeline, not just self-consistent within each one.

## 3. Sizing and selection

- **Target: ~15 episodes/task** (within the 10–20/task range), landing around 7.5% of the full 20,000-episode corpus.
- **Stratify by task.** Pull the slice evenly across all 100 tasks — don't let it skew toward tasks that happen to be easier to sample or already over-staged. Every task needs held-out episodes, or we lose signal on exactly the tasks where torque/BDDL conditioning might matter most (contact-rich, multi-stage tasks).
- Random selection within each task's 200 episodes is fine — no need for anything fancier than a seeded shuffle.

## 4. Process

1. **Generate the manifest first.** Before touching the rotating-shard staging pipeline, produce a fixed list of held-out episode IDs per task (e.g. `val_manifest.json`, keyed by `task/episode_index`). Seed the RNG and commit the manifest itself somewhere versioned — it needs to be reproducible, not regenerated per run.
2. **Exclude the manifest from every staging pass.** This is the step most likely to break silently: since shards rotate in and out over time as we work toward the full 20k, it's easy for a "held-out" episode to get staged into a training run by accident if the exclusion isn't enforced at the staging layer itself, not just remembered informally. The staging script should check every episode it stages against the manifest and skip anything on it.
3. **Train the monolith on the remaining ~18,500 episodes.**
4. **Torque sidecar and BDDL stage-separation fine-tunes also train on the same (20k − slice) corpus** — same exclusion, same manifest. These are additive features/annotations on top of the existing demos (torque via OmniGibson/MuJoCo replay, BDDL stages via predicate evaluation), not new episodes, so this doesn't change the staging footprint beyond the derived data itself.
5. **Score every variant against the same frozen slice** for the internal check: monolith baseline, +torque, +torque+BDDL. Same slice, same eval code path, every time.
6. **Reserve official live-rollout eval for after the internal slice shows a variant is worth it.** Don't burn full sim-eval budget on a variant that doesn't clear the cheap internal check first.

## 5. What this is not

- **Not the official BEHAVIOR eval set.** That's generated live at evaluation time from held-out initial-condition instances, independent of anything in the demo corpus. This slice sits upstream of that, as a cheap internal gate.
- **Not a reason to shrink the monolith's training budget meaningfully.** ~7.5% is a small, deliberate trade for having one clean comparison point shared across every model we train. It should not turn into "let's hold out more just in case" — more held-out data just means fewer episodes for a baseline we already want at full 200/task scale.
- **Not something to redo per experiment.** One manifest, frozen once, reused by everything downstream — including the eventual force-routed MoE, so it's directly comparable against the monolith and both fine-tuned variants on the same internal yardstick.

## 6. Action items

**Status (2026-08-12):** the split is now generated, versioned, and wired into both
config builders. The current run is a **full-FT (`gemma_2b`, `action_horizon=32`) clean-slate
re-run from wave1** using this frozen split.

- [x] Generate stratified 15/task validation manifest, seeded and versioned → `frozen_val_split.json` (`b1k_waves/create_frozen_val_split.py`)
- [x] Wire the manifest into every config → `_b1k_frozen_split()` in `config.py` reads `frozen_val_split.json`, then each `TrainConfig` keeps only the frozen episodes that staging actually downloads — per wave (frozen ∩ `demo_lo..demo_hi`) and per family (frozen ∩ `task_ids` ∩ `demo_lo..demo_hi`). **Option B (2026-08-12).**
- [x] Confirm zero leakage per subset — for all 8 waves + 5 families, `train ∩ val = ∅`, all indexed demos within the window, all family tasks within `task_ids` (verified after the Option B edit)
- [x] Stage each run's training data at full fine-tune, `action_horizon=32` — `run_waves.sh` stages + trains each wave's window; `run_family_experts.sh` stages each family's task slice. No manifest-exclusion needed at the staging layer, because `config.py` already restricts `episodes_index` to staged media.
- [ ] Build lightweight internal scoring path against the manifest slice (reused for monolith, +torque, +torque+BDDL, MoE)
