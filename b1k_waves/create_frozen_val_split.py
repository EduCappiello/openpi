"""Generate the deterministic, frozen train/val episode split for BEHAVIOR-1K.

Implements the internal validation-slice directive from b1k_waves/behavior_training_split.md:
~15 episodes/task (~7.5% of the 100-task / 20000-episode corpus = 1500 held-out episodes),
frozen BEFORE any retraining run so every model variant (Monolith, +Torque, +Torque+BDDL,
and the eventual force-routed MoE) trains on the same remaining episodes and is evaluated
against the exact same held-out slice.

Output: b1k_waves/frozen_val_split.json (deterministic, seeded, versioned). Consumers
(config.py _make_wave_configs / _make_family_configs, and future MoE builders) load it and
set episodes_index=train_episodes, val_episodes_index=val_episodes.

Selection modes:
  * seeded-shuffle (DEFAULT): per-task seeded RNG shuffle over that task's 200 episodes,
    first `val_per_task` become held out. Matches the coworker directive "random selection
    within each task's 200 episodes is fine ... seeded shuffle".
    CAVEAT: shuffling shreds mp4 file locality. Ledger.ledger.py documents a measured ~3.6x
    disk/staging cost vs a contiguous block (426 GB vs 118 GB for a 12 eps/task pick). Use
    --contiguous if disk/staging cost matters more than within-task randomness.
  * --contiguous: deterministic block 200t+185 .. 200t+199 (i.e. the tail of each task's
    block). Cheapest to stage, no RNG.
"""
import argparse
import json
import pathlib
import random

LEDGER_SEED = 20260721  # matches b1k_waves/ledger.py SEED so the whole pipeline shares one seed.
EPISODES_PER_TASK = 200
NUM_TASKS = 100
DEFAULT_VAL_PER_TASK = 15

OUT = pathlib.Path(__file__).with_name("frozen_val_split.json")


def gidx(task: int, demo: int) -> int:
    """Global lerobot episode_index for (task, demo-within-task)."""
    return EPISODES_PER_TASK * task + demo


def build(seed: int, val_per_task: int, *, contiguous: bool) -> dict:
    train: list[int] = []
    val: list[int] = []
    rng = random.Random(seed)
    for t in range(NUM_TASKS):
        if contiguous:
            # Tail block of the task, matching ledger.py's contiguous-block convention.
            val_lo = EPISODES_PER_TASK - val_per_task
            val += [gidx(t, i) for i in range(val_lo, EPISODES_PER_TASK)]
            train += [gidx(t, i) for i in range(val_lo)]
        else:
            episodes = list(range(EPISODES_PER_TASK))
            rng.shuffle(episodes)
            val += [gidx(t, i) for i in episodes[:val_per_task]]
            train += [gidx(t, i) for i in episodes[val_per_task:]]
    return {
        "seed": seed,
        "episodes_per_task": EPISODES_PER_TASK,
        "num_tasks": NUM_TASKS,
        "val_per_task": val_per_task,
        "mode": "contiguous_tail" if contiguous else "seeded_shuffle",
        "num_train": len(train),
        "num_val": len(val),
        "val_fraction": round(len(val) / (len(train) + len(val)), 4),
        "train_episodes": train,
        "val_episodes": val,
        "tasks": list(range(NUM_TASKS)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=LEDGER_SEED, help="RNG seed (only used in seeded-shuffle mode).")
    ap.add_argument("--val-per-task", type=int, default=DEFAULT_VAL_PER_TASK,
                    help=f"Held-out episodes per task (default {DEFAULT_VAL_PER_TASK} = 7.5%).")
    ap.add_argument("--contiguous", action="store_true",
                    help="Use the deterministic contiguous tail block 200t+185..200t+199 instead of a seeded shuffle.")
    ap.add_argument("--out", type=pathlib.Path, default=OUT, help="Output JSON path.")
    args = ap.parse_args()

    result = build(args.seed, args.val_per_task, contiguous=args.contiguous)
    # Atomic write so an interrupted run cannot leave a partial manifest.
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.rename(args.out)

    print(f"Wrote {args.out}  ({result['mode']}, val_per_task={result['val_per_task']})")
    print(f"  train={result['num_train']}  val={result['num_val']}  "
          f"val_fraction={result['val_fraction']}")
    print(f"  first val per task0: {result['val_episodes'][0]}..{result['val_episodes'][14]}")


if __name__ == "__main__":
    main()
