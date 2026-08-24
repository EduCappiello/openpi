"""Plot training loss & grad_norm for the 5 b1k family runs.

Reads the `Step N: grad_norm=..., loss=..., param_norm=...` lines from the
`train_*.log` files in this directory and produces a two-panel figure
(loss vs step, grad_norm vs step), one line per family.

Usage:
    .venv/bin/python b1k_families/plot_family_losses.py
Output:
    b1k_families/family_losses.png
"""

import argparse
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP_RE = re.compile(
    r"Step (\d+): grad_norm=([0-9.eE+-]+), loss=([0-9.eE+-]+), param_norm=([0-9.eE+-]+)"
)

FAMILIES = [
    ("backbone_foundation", "Backbone foundation", "#1f77b4"),
    ("F4_heavy_grasp", "F4 heavy_grasp", "#2ca02c"),
    ("F3_surface_contact", "F3 surface_contact", "#ff7f0e"),
    ("F2_actuation_transfer", "F2 actuation_transfer", "#d62728"),
    ("F1_constrained_insertion", "F1 constrained_insertion", "#9467bd"),
]


def parse_log(path: pathlib.Path) -> tuple[list[float], list[float]]:
    steps: list[float] = []
    losses: list[float] = []
    grad_norms: list[float] = []
    seen: set[int] = set()
    for line in path.read_text().splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        if step in seen:  # avoid double-counting if a step line appears twice
            continue
        seen.add(step)
        steps.append(step)
        losses.append(float(m.group(3)))
        grad_norms.append(float(m.group(2)))
    return steps, losses, grad_norms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent,
    )
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    out = args.out or (args.logs_dir / "family_losses.png")

    fig, (ax_loss, ax_grad) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("b1k family training (full-FT, pi05, action_horizon=32)", fontsize=13)

    for fname, label, color in FAMILIES:
        path = args.logs_dir / f"train_{fname}.log"
        if not path.exists():
            print(f"WARN: {path} not found, skipping")
            continue
        steps, losses, grad_norms = parse_log(path)
        if not steps:
            print(f"WARN: no loss lines parsed from {path}")
            continue
        ax_loss.plot(steps, losses, label=label, color=color, lw=1.5)
        ax_grad.plot(steps, grad_norms, label=label, color=color, lw=1.5)

    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Loss vs step (log scale)")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=8)

    ax_grad.set_xlabel("Step")
    ax_grad.set_ylabel("grad_norm")
    ax_grad.set_title("Grad norm vs step")
    ax_grad.grid(True, alpha=0.3)
    ax_grad.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
