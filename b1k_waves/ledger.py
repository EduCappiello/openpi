"""Deterministic episode ledger for BEHAVIOR-1K wave training.

Records which episodes each training run consumed so waves never overlap and any
run can be reproduced exactly.

Design note -- why waves are CONTIGUOUS blocks of demo_index_within_task and not
a seeded shuffle: lerobot v3.0 packs ~6.6 consecutive episodes per mp4. Measured
on this box, a seeded-random pick of 12 eps/task touches 426 GB of video files
while the contiguous block 30..41 covers the same 1200 episodes in 118 GB -- a
3.6x disk penalty for zero data benefit. Reproducibility therefore comes from
this ledger (an explicit, recorded schedule), not from an RNG that shreds file
locality. SEED is still recorded and used for the train/val split within a wave.
"""
import json, pathlib, datetime

SEED = 20260721
EPISODES_PER_TASK = 200          # task t occupies global episodes [200t, 200t+200)
NUM_TASKS = 100
LEDGER = pathlib.Path(__file__).with_name("episode_ledger.json")


def gidx(task, demo):
    """Global lerobot episode_index for (task, demo-within-task)."""
    return EPISODES_PER_TASK * task + demo


def _blank():
    return {"seed": SEED, "episodes_per_task": EPISODES_PER_TASK,
            "num_tasks": NUM_TASKS, "waves": []}


def load():
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else _blank()


def save(d):
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2))
    tmp.rename(LEDGER)          # atomic: an interrupted write cannot corrupt the ledger


def consumed(d):
    """Set of demo indices already used by any recorded wave."""
    s = set()
    for w in d["waves"]:
        s |= set(range(w["demo_lo"], w["demo_hi"]))
    return s


def next_lo(d):
    c = consumed(d)
    return max(c) + 1 if c else 0


def plan_wave(n_per_task, val_per_task, name, d=None):
    """Carve the next contiguous block of n_per_task demos off the unused pool."""
    d = d or load()
    lo = next_lo(d)
    hi = lo + n_per_task
    if hi > EPISODES_PER_TASK:
        raise ValueError(f"wave {lo}..{hi} exceeds {EPISODES_PER_TASK} episodes/task")
    if any(max(lo, w["demo_lo"]) < min(hi, w["demo_hi"]) for w in d["waves"]):
        raise ValueError(f"wave {lo}..{hi} overlaps an existing wave")
    # val is the TAIL of the block, so train indices stay contiguous from lo
    val_lo = hi - val_per_task
    train = [gidx(t, i) for t in range(NUM_TASKS) for i in range(lo, val_lo)]
    val   = [gidx(t, i) for t in range(NUM_TASKS) for i in range(val_lo, hi)]
    return {"name": name, "demo_lo": lo, "demo_hi": hi, "val_lo": val_lo,
            "seed": SEED, "n_per_task": n_per_task, "val_per_task": val_per_task,
            "train_episodes": train, "val_episodes": val,
            "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")}


def commit(wave, d=None):
    d = d or load()
    d["waves"].append(wave)
    save(d)
    return d
