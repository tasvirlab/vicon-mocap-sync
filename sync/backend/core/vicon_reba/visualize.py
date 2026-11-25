"""Publication-quality plots for the Vicon REBA pipeline.

Generates:
  1. angles_timeseries.png  — all REBA-critical angles over time (two rows of subplots)
  2. reba_timeseries.png    — REBA total score + score_a/score_b over time with risk bands
  3. reba_segment_heatmap.png — per-body-segment score bar chart for peak frame

All figures saved at 300 DPI in the specified output directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for scripts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       9,
    "axes.titlesize":  10,
    "axes.labelsize":  9,
    "legend.fontsize": 8,
    "figure.dpi":      150,
})

RISK_BANDS = [
    (1,  1,  "#d5f5e3", "Negligible"),
    (2,  3,  "#fcf3cf", "Low"),
    (4,  7,  "#fdebd0", "Medium"),
    (8,  10, "#fadbd8", "High"),
    (11, 15, "#e8daef", "Very High"),
]


def _add_risk_bands(ax: plt.Axes, y_max: float = 15) -> None:
    """Draw horizontal risk-band shading on a REBA score axis."""
    for lo, hi, color, label in RISK_BANDS:
        ax.axhspan(lo - 0.5, hi + 0.5, alpha=0.25, color=color, label=label, zorder=0)


def plot_angles(
    angles_df: pd.DataFrame,
    out_dir: Path,
    time_col: str = "time_s",
) -> Path:
    """Plot all REBA-critical joint angles over time.

    Args:
        angles_df: DataFrame from angles.angles_to_dataframe().
        out_dir:   Directory to save the figure.
        time_col:  Column to use as the x-axis (default: 'time_s').

    Returns:
        Path to the saved figure.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t = angles_df[time_col].to_numpy() if time_col in angles_df.columns else np.arange(len(angles_df))

    panels = [
        ("Trunk", [
            ("trunk_total_deg",    "Total inclination"),
            ("trunk_sagittal_deg", "Sagittal (fwd+/bwd−)"),
            ("trunk_lateral_deg",  "Lateral bend"),
            ("trunk_twist_deg",    "Twist (sho vs hip bar)"),
        ]),
        ("Neck", [
            ("neck_flex_deg",    "Flexion from trunk"),
            ("neck_lateral_deg", "Lateral bend"),
        ]),
        ("Right arm", [
            ("upper_arm_r_deg",     "Upper arm elevation"),
            ("upper_arm_r_abd_deg", "Shoulder abduction"),
            ("lower_arm_r_deg",     "Elbow flexion"),
            ("wrist_r_deg",         "Wrist deviation"),
        ]),
        ("Left arm", [
            ("upper_arm_l_deg",     "Upper arm elevation"),
            ("upper_arm_l_abd_deg", "Shoulder abduction"),
            ("lower_arm_l_deg",     "Elbow flexion"),
            ("wrist_l_deg",         "Wrist deviation"),
        ]),
        ("Legs", [
            ("knee_r_deg", "Right knee flexion"),
            ("knee_l_deg", "Left knee flexion"),
            ("hip_r_deg",  "Right hip angle"),
            ("hip_l_deg",  "Left hip angle"),
        ]),
    ]

    n_rows = len(panels)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.8 * n_rows), sharex=True)
    fig.suptitle("REBA-Critical Joint Angles (Vicon Ground Truth)", fontsize=11, y=1.01)

    for ax, (title, series) in zip(axes, panels):
        for col, label in series:
            if col in angles_df.columns:
                ax.plot(t, angles_df[col].to_numpy(), label=label, linewidth=0.9)
        ax.set_ylabel("Degrees")
        ax.set_title(title)
        ax.legend(loc="upper right", ncol=2, framealpha=0.7)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")

    axes[-1].set_xlabel("Time (s)" if time_col == "time_s" else "Frame")
    plt.tight_layout()

    out = out_dir / "angles_timeseries.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_reba_timeseries(
    reba_df: pd.DataFrame,
    out_dir: Path,
    time_col: np.ndarray | None = None,
) -> Path:
    """Plot REBA total score and Score A / Score B over time with risk bands.

    Args:
        reba_df:  DataFrame from reba.score_timeseries().
        out_dir:  Output directory.
        time_col: 1-D array of timestamps; falls back to frame index if None.

    Returns:
        Path to the saved figure.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t = time_col if time_col is not None else np.arange(len(reba_df))

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # ── Top: REBA total ──────────────────────────────────────────────────────
    ax = axes[0]
    _add_risk_bands(ax)
    ax.plot(t, reba_df["reba_total"].to_numpy(), color="#2c3e50", linewidth=1.2, zorder=5)
    ax.set_ylim(0, 15.5)
    ax.set_ylabel("REBA Total Score")
    ax.set_title("REBA Total Score Over Time")

    # Risk band legend
    patches = [mpatches.Patch(color=c, alpha=0.5, label=lbl) for _, _, c, lbl in RISK_BANDS]
    ax.legend(handles=patches, loc="upper left", ncol=5, fontsize=7, framealpha=0.8)

    # Mark peak
    peak_idx = int(reba_df["reba_total"].idxmax())
    peak_t   = t[peak_idx]
    peak_val = reba_df["reba_total"].iloc[peak_idx]
    ax.axvline(peak_t, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.annotate(f"Peak {peak_val}",
                xy=(peak_t, peak_val),
                xytext=(peak_t + (t[-1] - t[0]) * 0.02, peak_val + 0.5),
                fontsize=8, color="red")
    ax.grid(True, alpha=0.25)

    # ── Bottom: Score A and Score B ──────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(t, reba_df["score_a"].to_numpy(), label="Score A (trunk+neck+legs+activity)",
             color="#e74c3c", linewidth=1.0, zorder=5)
    ax2.plot(t, reba_df["score_b"].to_numpy(), label="Score B (arm+coupling)",
             color="#3498db", linewidth=1.0, zorder=5)
    ax2.set_ylabel("Sub-Score")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Score A and Score B Components")
    ax2.legend(loc="upper right", framealpha=0.8)
    ax2.grid(True, alpha=0.25)
    ax2.set_ylim(0, 13)

    plt.tight_layout()
    out = out_dir / "reba_timeseries.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_segment_heatmap(
    reba_df: pd.DataFrame,
    out_dir: Path,
    summary: dict | None = None,
) -> Path:
    """Bar chart of per-segment scores at the peak REBA frame.

    Args:
        reba_df:  DataFrame from reba.score_timeseries().
        out_dir:  Output directory.
        summary:  Optional output of reba.summarise() to annotate the title.

    Returns:
        Path to the saved figure.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    peak_idx = int(reba_df["reba_total"].idxmax())
    peak     = reba_df.iloc[peak_idx]

    segments = {
        "Trunk":       (int(peak["trunk_score"]),      5),
        "Neck":        (int(peak["neck_score"]),        3),
        "Legs":        (int(peak["legs_score"]),        4),
        "Upper Arm":   (int(peak["upper_arm_adj"]),     6),
        "Lower Arm":   (int(peak["lower_arm_score"]),   2),
        "Wrist":       (int(peak["wrist_adj"]),         2),
    }

    labels = list(segments.keys())
    scores = [v[0] for v in segments.values()]
    maxes  = [v[1] for v in segments.values()]
    ratios = [s / m for s, m in zip(scores, maxes)]

    colors = []
    for r in ratios:
        if r < 0.34:
            colors.append("#2ecc71")
        elif r < 0.67:
            colors.append("#e67e22")
        else:
            colors.append("#e74c3c")

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, scores, color=colors, edgecolor="white", height=0.55)
    ax.barh(labels, maxes, color="lightgrey", edgecolor="white", height=0.55, alpha=0.3)

    for bar, score, mx in zip(bars, scores, maxes):
        ax.text(
            bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{score}/{mx}",
            va="center", ha="left", fontsize=9,
        )

    peak_reba = int(peak["reba_total"])
    risk = str(peak["risk_level"]).replace("_", " ").title()
    title = f"Segment Scores at Peak REBA Frame (frame {peak_idx})"
    if summary:
        title += f"\nPeak REBA: {peak_reba}  |  Risk: {risk}  |  Mean: {summary['mean']:.1f}"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Score")
    ax.set_xlim(0, max(maxes) + 1.5)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out = out_dir / "reba_segment_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_all(
    angles_df: pd.DataFrame,
    reba_df: pd.DataFrame,
    out_dir: Path,
    summary: dict | None = None,
) -> list[Path]:
    """Convenience: run all three plots and return list of saved paths."""
    time_arr = angles_df["time_s"].to_numpy() if "time_s" in angles_df.columns else None
    paths = [
        plot_angles(angles_df, out_dir),
        plot_reba_timeseries(reba_df, out_dir, time_col=time_arr),
        plot_segment_heatmap(reba_df, out_dir, summary=summary),
    ]
    return paths
