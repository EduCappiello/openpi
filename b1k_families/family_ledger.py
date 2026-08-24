"""Deterministic episode ledger for BEHAVIOR-1K super-family expert training.

Records which tasks and demo episodes each expert uses so the four force
super-families (plus the backbone foundation) never overlap on intent and any
run is reproducible. Reuses the wave ledger's global episode layout: task t
occupies [200t, 200t+200); this pipeline trains on demos 90-99 (9 train + 1 val).

Warm-start topology (star, NOT a chain): backbone_foundation inherits weights
from the wave8 checkpoint; every expert inherits directly from the completed
backbone_foundation checkpoint. See each family's ``warm_start`` field.
"""
import json
import os
import pathlib
import datetime

EPISODES_PER_TASK = 200          # task t occupies global episodes [200t, 200t+200)
NUM_TASKS = 100
LEDGER = pathlib.Path(__file__).with_name("family_ledger.json")


def load():
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else _blank()


def save(d):
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2))
    tmp.rename(LEDGER)          # atomic: an interrupted write cannot corrupt the ledger


def _blank():
    return {"seed": 20260721, "episodes_per_task": EPISODES_PER_TASK,
            "num_tasks": NUM_TASKS, "families": []}


def families_in_order(d=None):
    """Families in the order the orchestrator runs them (ledger order)."""
    d = d or load()
    return d.get("families", [])


def warm_start_params(f, d=None):
    """Resolve a family's warm-start param path (star topology)."""
    d = d or load()
    if f.get("warm_start") == "backbone":
        return d["backbone_final_params"]
    return d["wave8_final_params"]


def gidx(task, demo):
    """Global lerobot episode_index for (task, demo-within-task)."""
    return EPISODES_PER_TASK * task + demo


if __name__ == "__main__":
    d = load()
    for f in families_in_order(d):
        print(f"{f['name']:24s} tasks={len(f['task_ids']):3d} "
              f"ntrain={len(f['train_episodes']):4d} nval={len(f['val_episodes']):4d} "
              f"warm={f['warm_start']} -> {warm_start_params(f, d)}")
