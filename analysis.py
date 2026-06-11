"""Analyze benchmark JSON results and create summary visualizations.

Reads all *_results.json files from the results/ directory and generates:
    - Accuracy-fairness tradeoff scatter plot
    - Convergence curves per solution
    - F1 / Precision / Recall bar chart
    - Resource (time) comparison
    - Summary table (printed to console)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CLASS_NAMES, PROJECT_ROOT
from utils import load_json_files


RESULTS_DIR = Path(PROJECT_ROOT) / "results"

SOLUTION_LABELS = {
    "baseline": "Baseline",
    "batch_norm": "BatchNorm",
    "group_norm": "GroupNorm",
    "layer_norm": "LayerNorm",
    "instance_norm": "InstanceNorm",
    "focal_loss": "FocalLoss",
    "regularization": "L2 Reg.",
}

# Colour palette for consistent styling across charts
SOLUTION_COLORS = {
    "baseline": "#6c757d",
    "batch_norm": "#0d6efd",
    "group_norm": "#198754",
    "layer_norm": "#ffc107",
    "instance_norm": "#dc3545",
    "focal_loss": "#6f42c1",
    "regularization": "#fd7e14",
}


def _label(solution):
    return SOLUTION_LABELS.get(solution, solution)


def _color(solution):
    return SOLUTION_COLORS.get(solution, "#333333")


# ---------------------------------------------------------------------------
# Load & normalise results
# ---------------------------------------------------------------------------


def load_results():
    """Read all result JSON files and return round-level records."""
    records = []
    for result in load_json_files(RESULTS_DIR):
        solution = result.get("solution_name", "unknown")
        for rd in result.get("rounds", []):
            records.append({
                "solution": solution,
                "round": rd.get("round", 0),
                "accuracy": rd.get("accuracy", 0.0),
                "fairness": rd.get("fairness_score", 0.0),
                "avg_train_loss": rd.get("avg_train_loss", 0.0),
                "macro_precision": rd.get("macro_precision", 0.0),
                "macro_recall": rd.get("macro_recall", 0.0),
                "macro_f1": rd.get("macro_f1", 0.0),
                "round_time_sec": rd.get("round_time_sec", 0.0),
                "class_accuracy": rd.get("class_accuracy", {}),
            })
    return pd.DataFrame(records)


def summarize_results(df):
    """Create one summary row per solution from the final round."""
    if df.empty:
        return pd.DataFrame()
    final = df.sort_values("round").groupby("solution").tail(1)
    summary = final[[
        "solution", "accuracy", "fairness", "macro_precision",
        "macro_recall", "macro_f1", "round_time_sec",
    ]].copy()
    summary["accuracy_pct"] = summary["accuracy"]  # already in %
    return summary.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_accuracy_fairness_tradeoff(summary):
    """Scatter: accuracy vs fairness, one point per solution."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in summary.iterrows():
        ax.scatter(
            row["fairness"],
            row["accuracy_pct"],
            s=120,
            color=_color(row["solution"]),
            edgecolors="white",
            linewidths=1.5,
            zorder=5,
        )
        ax.annotate(
            _label(row["solution"]),
            (row["fairness"], row["accuracy_pct"]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=9,
        )

    ax.set_xlabel("Fairness Score", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy–Fairness Tradeoff (BreakHis)", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "accuracy_fairness_tradeoff.png", dpi=200)
    plt.close(fig)


def plot_convergence_curves(df):
    """Line plot: accuracy over rounds for each solution."""
    solutions = sorted(df["solution"].unique())
    n = len(solutions)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, sol in zip(axes_flat, solutions):
        sol_df = df[df["solution"] == sol].sort_values("round")
        ax.plot(
            sol_df["round"], sol_df["accuracy"],
            marker="o", markersize=3, color=_color(sol), linewidth=1.5,
        )
        ax.set_title(_label(sol), fontsize=11, fontweight="bold")
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle("Convergence Curves (BreakHis)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "convergence_curves.png", dpi=200)
    plt.close(fig)


def plot_f1_precision_recall(summary):
    """Grouped bar chart: macro F1, precision, recall per solution."""
    solutions = summary["solution"].tolist()
    labels = [_label(s) for s in solutions]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, summary["macro_precision"], width, label="Precision", color="#0d6efd")
    ax.bar(x, summary["macro_recall"], width, label="Recall", color="#198754")
    ax.bar(x + width, summary["macro_f1"], width, label="F1 Score", color="#6f42c1")

    ax.set_xlabel("Normalization Technique", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision / Recall / F1 Comparison (BreakHis)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "f1_precision_recall.png", dpi=200)
    plt.close(fig)


def plot_training_time(summary):
    """Bar chart: total round time per solution."""
    solutions = summary["solution"].tolist()
    labels = [_label(s) for s in solutions]
    colors = [_color(s) for s in solutions]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, summary["round_time_sec"], color=colors)
    ax.set_xlabel("Solution")
    ax.set_ylabel("Final Round Time (sec)")
    ax.set_title("Training Time Comparison", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_time_comparison.png", dpi=200)
    plt.close(fig)


def plot_loss_curves(df):
    """Overlay plot: training loss over rounds for all solutions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for sol in sorted(df["solution"].unique()):
        sol_df = df[df["solution"] == sol].sort_values("round")
        ax.plot(
            sol_df["round"], sol_df["avg_train_loss"],
            label=_label(sol), color=_color(sol), linewidth=1.5,
        )
    ax.set_xlabel("Round")
    ax.set_ylabel("Average Training Loss")
    ax.set_title("Training Loss Convergence (BreakHis)", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "loss_curves.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def print_summary_table(summary):
    """Print a formatted table of final-round metrics."""
    display = summary.copy()
    display["Solution"] = display["solution"].map(_label)
    display = display[[
        "Solution", "accuracy_pct", "fairness",
        "macro_precision", "macro_recall", "macro_f1",
        "round_time_sec",
    ]]
    display.columns = [
        "Solution", "Acc (%)", "Fairness",
        "Precision", "Recall", "F1",
        "Time (s)",
    ]
    print("\n" + "=" * 80)
    print("  BENCHMARK SUMMARY — BreakHis Breast Cancer (5 Clients, Non-IID)")
    print("=" * 80)
    print(display.to_string(
        index=False,
        float_format=lambda v: f"{v:.4f}",
    ))
    print("=" * 80)


def print_per_class_accuracy(df):
    """Print per-class accuracy for each solution (final round)."""
    final = df.sort_values("round").groupby("solution").tail(1)
    for _, row in final.iterrows():
        sol = row["solution"]
        ca = row.get("class_accuracy", {})
        if not isinstance(ca, dict) or not ca:
            continue
        print(f"\n{_label(sol)} — Per-Class Accuracy:")
        for class_id in sorted(ca.keys(), key=int):
            class_name = CLASS_NAMES[int(class_id)] if int(class_id) < len(CLASS_NAMES) else f"Class {class_id}"
            print(f"  {class_name}: {ca[class_id]:.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()

    if df.empty:
        print(f"No JSON results found in {RESULTS_DIR}")
        return

    summary = summarize_results(df)

    # Generate all plots
    plot_accuracy_fairness_tradeoff(summary)
    plot_convergence_curves(df)
    plot_f1_precision_recall(summary)
    plot_training_time(summary)
    plot_loss_curves(df)

    # Console output
    print_summary_table(summary)
    print_per_class_accuracy(df)
    print(f"\nSaved analysis plots to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
