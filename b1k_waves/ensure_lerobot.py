"""Guard against the actual root cause of the real training outages this campaign hit.

Historically a stray `uv sync` silently downgraded lerobot to a rev that cannot read
the v3.0 demo layout (it expects v2.1 `meta/tasks.jsonl`), and -- since 2026-08-28,
when `uv.lock` moved to the wensi-ai/lerobot `release/b1k` fork (0.5.2) -- a stale
install of PyPI lerobot 0.4.4 (which lacks the fork's `accelerate` runtime dep) broke
the data-loader import path. Both failures are silent until dataset load.

The canonical env is whatever `uv.lock` pins (currently lerobot 0.5.2 from the
wensi-ai fork). This guard compares the installed version to the one in uv.lock and,
on mismatch, runs `uv sync --frozen` to rebuild the exact locked env -- no ad-hoc
version strings, no `--no-deps` git installs that skip transitive deps.

Run this before every wave/family launch. It is a cheap import + string compare when
the env is correct, so it's fine to call unconditionally.
"""
import pathlib
import subprocess
import sys

REQUIRED = None  # resolved from uv.lock at runtime (single source of truth)
LOCK = pathlib.Path(__file__).resolve().parents[1] / "uv.lock"
VENV = pathlib.Path(__file__).resolve().parents[1] / ".venv"
# Absolute path: cron/unattended shells often have a minimal PATH that lacks
# ~/.local/bin, so `uv` by bare name is not reliable outside an interactive login shell.
UV_BIN = pathlib.Path.home() / ".local" / "bin" / "uv"


def required_version():
    """Reads the lerobot version pinned in uv.lock (the single source of truth)."""
    if not LOCK.exists():
        return None
    text = LOCK.read_text()
    # [[package]] name = "lerobot" ... version = "X.Y.Z"
    for block in text.split("[[package]]"):
        if 'name = "lerobot"' in block:
            for line in block.splitlines():
                line = line.strip()
                if line.startswith("version = "):
                    return line.split("=", 1)[1].strip().strip('"')
    return None


def current_version():
    try:
        import lerobot
        return lerobot.__version__
    except ImportError:
        return None


def fix(required):
    print(f"lerobot version wrong or missing (need {required}) -- running `uv sync --frozen`", flush=True)
    import os
    root = VENV.parent
    env = dict(os.environ, VIRTUAL_ENV=str(VENV))
    uv = str(UV_BIN) if UV_BIN.exists() else "uv"
    subprocess.run(
        [uv, "sync", "--frozen"],
        cwd=str(root),
        env=env,
        check=True,
    )


def main():
    required = required_version()
    if required is None:
        print(f"FIX FAILED: could not read the pinned lerobot version from {LOCK}", file=sys.stderr)
        return 1
    v = current_version()
    if v == required:
        print(f"lerobot {v} OK")
        return 0
    print(f"lerobot version is {v!r}, expected {required!r} (per uv.lock)")
    fix(required)
    # re-check in a fresh subprocess since the current interpreter already imported
    # the wrong version and won't see the reinstall without a restart
    out = subprocess.run(
        [str(VENV / "bin" / "python"), "-c", "import lerobot; print(lerobot.__version__)"],
        capture_output=True, text=True,
    )
    new_v = out.stdout.strip()
    if new_v != required:
        print(f"FIX FAILED: still {new_v!r} after reinstall. stderr:\n{out.stderr}", file=sys.stderr)
        return 1
    print(f"lerobot fixed -> {new_v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
