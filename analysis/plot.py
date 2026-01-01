# Note: Parts of this code were developed with the assistance of ChatGPT.

# Libraries
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Academic plotting style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

CHECKPOINT_DIR = "checkpoints"
FIGURE_DIR = "analysis/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# Load checkpoints
def load_checkpoints():
    paths = sorted(glob(os.path.join(CHECKPOINT_DIR, "*.pth")))
    checkpoints = []
    for p in paths:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        ckpt["path"] = p
        checkpoints.append(ckpt)
    return checkpoints

def select_experiments(checkpoints, *, init, scheduler, budget):
    return [
        c for c in checkpoints
        if c.get("init") == init
        and c.get("scheduler") == scheduler
        and c.get("budget", 30) == budget
    ]

# MAIN FIGURE 1
# SGD vs Adam (same init & scheduler)
def plot_optimizer_comparison(checkpoints, init="he", scheduler="cosine", budget=5):
    subset = select_experiments(
        checkpoints, init=init, scheduler=scheduler, budget=budget
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    max_loss = max(
        max(c["train_loss_curve"] + c.get("val_loss_curve", []))
        for c in subset
    )

    for i, c in enumerate(subset):
        epochs = np.arange(1, len(c["train_loss_curve"]) + 1)

        label = c["optimizer"].upper()

        ax.plot(
            epochs,
            c["train_loss_curve"],
            color=COLORS[i],
            linestyle="-",
            label=f"{label} (Train)"
        )

        ax.plot(
            epochs,
            c["val_loss_curve"],
            color=COLORS[i],
            linestyle="--",
            label=f"{label} (Val)"
        )

    ax.set_ylim(0, max_loss * 1.05)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(
        f"Optimizer Comparison ({init.capitalize()}, {scheduler.capitalize()}, Budget={budget})"
    )
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/optimizer_comparison_budget{budget}.png")
    plt.close()

# MAIN FIGURE 2
# Generalization Gap
def plot_generalization_gap(checkpoints, init="he", scheduler="cosine", budget=5):
    subset = select_experiments(
        checkpoints, init=init, scheduler=scheduler, budget=budget
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    max_gap = max(
        max(abs(np.array(c["gap_loss_curve"])))
        for c in subset
    )

    for i, c in enumerate(subset):
        epochs = np.arange(1, len(c["gap_loss_curve"]) + 1)
        ax.plot(
            epochs,
            c["gap_loss_curve"],
            marker="o",
            color=COLORS[i],
            label=c["optimizer"].upper()
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(-max_gap * 1.1, max_gap * 1.1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Δ Loss (Val − Train)")
    ax.set_title(
        f"Generalization Gap ({init.capitalize()}, {scheduler.capitalize()}, Budget={budget})"
    )
    ax.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/generalization_gap_budget{budget}.png")
    plt.close()

# MAIN FIGURE 3
# Test F1 comparison
def plot_f1_comparison(checkpoints, init="he", scheduler="cosine", budget=5):
    subset = select_experiments(
        checkpoints, init=init, scheduler=scheduler, budget=budget
    )

    labels = [c["optimizer"].upper() for c in subset]
    f1s = [c["test_f1"] for c in subset]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(labels, f1s, color=COLORS[:len(labels)], edgecolor="black")

    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{b.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Macro F1 Score")
    ax.set_title(
        f"Test Performance ({init.capitalize()}, {scheduler.capitalize()}, Budget={budget})"
    )

    plt.tight_layout()
    plt.savefig(f"{FIGURE_DIR}/f1_comparison_budget{budget}.png")
    plt.close()

# APPENDIX
# Individual loss curves
def plot_individual_losses(checkpoints):
    for i, c in enumerate(checkpoints):
        epochs = np.arange(1, len(c["train_loss_curve"]) + 1)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, c["train_loss_curve"], label="Train", linewidth=2)
        ax.plot(epochs, c["val_loss_curve"], label="Validation", linewidth=2)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(
            f"Experiment {i+1} ({c['optimizer'].upper()}, {c['init']}, {c['scheduler']})"
        )
        ax.legend()

        plt.tight_layout()
        plt.savefig(
            f"{FIGURE_DIR}/appendix_experiment_{i+1:02d}.png"
        )
        plt.close()

# Main
def main():
    checkpoints = load_checkpoints()
    if not checkpoints:
        raise RuntimeError("No checkpoints found.")

    # MAIN FIGURES
    plot_optimizer_comparison(checkpoints, init="he", scheduler="cosine", budget=5)
    plot_generalization_gap(checkpoints, init="he", scheduler="cosine", budget=5)
    plot_f1_comparison(checkpoints, init="he", scheduler="cosine", budget=5)

    # APPENDIX
    plot_individual_losses(checkpoints)

    print("Academic figures generated (clean & interpretable).")

if __name__ == "__main__":
    main()