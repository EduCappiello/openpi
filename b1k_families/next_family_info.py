"""Print the next incomplete family as shell-sourceable KEY=VALUE lines, or "DONE".

Consumed by run_family_experts.sh via `eval "$(...)"`. Exit code 0 whenever the
query itself succeeded (including the all-done case); nonzero means the ledger or
config could not be read, which the caller treats as fatal.
"""
import sys

import family_status as FS
import family_ledger as ledger
import openpi.training.config as C


def main():
    d = FS.sync_ledger_status()
    f = FS.next_incomplete_family(d)
    if f is None:
        print("DONE")
        return 0
    config_name = FS.family_config_name(f["name"])
    cfg = C.get_config(config_name)
    print(f"NAME={f['name']}")
    print(f"CONFIG={config_name}")
    print(f"TASKS={','.join(str(t) for t in f['task_ids'])}")
    print(f"LO={f['demo_lo']}")
    print(f"HI={f['demo_hi']}")
    print(f"STEPS={cfg.num_train_steps}")
    print(f"WARM={ledger.warm_start_params(f, d)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
