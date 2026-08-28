"""Fail fast if the venv drifts from the versions B1K training requires.

Why this exists: `uv sync` resolves from uv.lock, which pins lerobot to a git rev
that installs 0.4.4's PREDECESSOR (0.3.4, dataset layout v2.1) and drags datasets
and av back below lerobot 0.4.4's own minimums. The 2026 demos are v3.0, so after
a sync training either dies immediately (v2.1 looks for meta/tasks.jsonl, which
v3.0 does not have) or -- worse -- limps at ~100x slowdown with mismatched
datasets/pyarrow. Run this before every launch.
"""
import sys

REQUIRED = {
    # package: (predicate, human-readable requirement, why it matters)
    "lerobot":  (lambda v: v >= (0, 4, 4), ">=0.4.4",      "v3.0 dataset layout; 0.3.4 expects v2.1 meta/tasks.jsonl"),
    "datasets": (lambda v: (4,) <= v < (5,), ">=4.0.0,<5", "lerobot 0.4.4 requires it; 3.x cannot read v3.0 'List' features"),
    "av":       (lambda v: (15,) <= v < (16,), ">=15,<16", "lerobot 0.4.4 requires it"),
    "numpy":    (lambda v: v == (1, 26, 4), "==1.26.4",    "jax 0.5.3 is built against it"),
}


def _ver(mod):
    from importlib.metadata import version
    return tuple(int(x) for x in version(mod).split(".")[:3] if x.isdigit())


def main() -> int:
    bad = []
    for pkg, (ok, want, why) in REQUIRED.items():
        try:
            v = _ver(pkg)
        except Exception as e:
            bad.append(f"  {pkg}: NOT INSTALLED ({e})")
            continue
        got = ".".join(map(str, v))
        mark = "OK  " if ok(v) else "BAD "
        print(f"  {mark}{pkg:10s} {got:12s} (need {want})")
        if not ok(v):
            bad.append(f"  {pkg}: got {got}, need {want} -- {why}")

    # v3.0 layout must be physically present, not just the right version number.
    from pathlib import Path
    meta = Path("/root/.cache/huggingface/lerobot/behavior-1k/2026-challenge-demos/meta")
    if not (meta / "tasks.parquet").exists():
        bad.append(f"  dataset: {meta}/tasks.parquet missing -- not a v3.0 checkout")

    if bad:
        print("\nENVIRONMENT BROKEN -- do not launch training:")
        print("\n".join(bad))
        print('\nFix:\n  cd <openpi repo root>\n'
                '  .venv/bin/python b1k_waves/ensure_lerobot.py\n'
                '  # (rebuilds the exact env uv.lock pins -- currently the wensi-ai/lerobot\n'
                '  #  release/b1k fork 0.5.2, which needs `accelerate`; never --no-deps to PyPI 0.4.4)')
        return 1
    print("\nenvironment OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
