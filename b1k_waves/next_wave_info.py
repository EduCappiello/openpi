"""Print the next incomplete wave as shell-sourceable KEY=VALUE lines, or "DONE".

Consumed by run_waves.sh via `eval "$(next_wave_info.py)"`. Exit code is 0 whenever
the query itself succeeded (including the "nothing left to run" case) -- nonzero only
means the ledger/config could not be read at all, which the caller treats as fatal.
"""
import sys

import wave_status as WS
import openpi.training.config as C


def main():
    d = WS.sync_ledger_status()
    w = WS.next_incomplete_wave(d)
    if w is None:
        print("DONE")
        return 0
    config_name = WS.wave_config_name(w["name"])
    cfg = C.get_config(config_name)
    print(f"NAME={w['name']}")
    print(f"CONFIG={config_name}")
    print(f"LO={w['demo_lo']}")
    print(f"HI={w['demo_hi']}")
    print(f"STEPS={cfg.num_train_steps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
