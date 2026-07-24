"""Guard against the actual root cause of the one real training outage this campaign hit.

A `uv sync` in this venv (run by something other than this pipeline -- observed once,
cause not fully identified) silently downgraded lerobot from 0.4.4 (v3.0 dataset
support, what arm0 and every wave trained under) to 0.3.4 (v2.1-only, pinned by
uv.lock's git rev). The downgrade produces no import error -- training launches,
gets through norm-stats loading, and only dies when LeRobotDatasetMetadata tries to
read a v2.1-layout file (`meta/tasks.jsonl`) that does not exist in a v3.0 repo.

Run this before every wave launch. It is a cheap import + version string compare,
not a network call, so it's fine to call unconditionally.
"""
import pathlib
import subprocess
import sys

REQUIRED = "0.4.4"
VENV = pathlib.Path(__file__).resolve().parents[1] / ".venv"
# Absolute path: cron/unattended shells often have a minimal PATH that lacks
# ~/.local/bin, so `uv` by bare name is not reliable outside an interactive login shell.
UV_BIN = pathlib.Path.home() / ".local" / "bin" / "uv"


def current_version():
    try:
        import lerobot
        return lerobot.__version__
    except ImportError:
        return None


def fix():
    print(f"lerobot version wrong or missing (need {REQUIRED}) -- reinstalling --no-deps", flush=True)
    import os
    env = dict(os.environ, VIRTUAL_ENV=str(VENV))
    uv = str(UV_BIN) if UV_BIN.exists() else "uv"
    subprocess.run(
        [uv, "pip", "install", "--no-deps", f"lerobot=={REQUIRED}"],
        env=env, check=True,
    )


def main():
    v = current_version()
    if v == REQUIRED:
        print(f"lerobot {v} OK")
        return 0
    print(f"lerobot version is {v!r}, expected {REQUIRED!r}")
    fix()
    # re-check in a fresh subprocess since the current interpreter already imported
    # the wrong version and won't see the reinstall without a restart
    out = subprocess.run(
        [str(VENV / "bin" / "python"), "-c", "import lerobot; print(lerobot.__version__)"],
        capture_output=True, text=True,
    )
    new_v = out.stdout.strip()
    if new_v != REQUIRED:
        print(f"FIX FAILED: still {new_v!r} after reinstall. stderr:\n{out.stderr}", file=sys.stderr)
        return 1
    print(f"lerobot fixed -> {new_v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
