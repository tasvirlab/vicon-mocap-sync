"""Publication-style figures for Channel 2 (posture duration) and Channel 3
(trunk-motion dynamics) — the two ergonomic channels beyond frame-wise REBA.

These render what channels.py already computed; they do NOT recompute or
re-filter anything, so a figure can never disagree with the JSON/CSV a trial
already wrote. Same separation visualize.py uses for Channel 1.

Figure design follows the three-channel framing:

  Channel 2 — "how long and how often is the risky posture sustained?"
    (a) signed angle trace per segment, with the bout-entry threshold and every
        detected bout shaded, so onset/offset are readable against the signal
        that produced them;
    (b) a Gantt-style bout timeline, one lane per segment, which is the figure
        the duration channel is actually about — count, placement, and length
        of bouts across the trial at a glance;
    (c) per-bout duration bars against the provisional ISO 11226 holding-time
        limit for that bout's mean angle.

  Channel 3 — "how rapidly or abruptly does the trunk move?"
    Three stacked rows sharing a time axis — signed angle, angular velocity,
    angular acceleration — with all three anatomical planes overlaid in each,
    plus a Marras-predictor bar panel. Stacking (rather than a 3x3 grid) keeps
    the vertical read: a velocity peak lines up with the angle slope that
    produced it and the acceleration spike that follows.

Deliberate honesty constraints, mirroring the modules these figures render:
  - ISO 11226 anchors are PROVISIONAL. Every ISO mark is dashed, labelled
    "provisional", and the caption says so — a reader must not mistake a
    dashed line here for a validated endurance limit.
  - Angles are SIGNED (flexion/extension, left/right) and plotted signed, with
    a zero line. The bout threshold is drawn at +/-threshold because bout
    detection uses |angle|. Do not "simplify" these traces to magnitudes.
  - Marras probability is a HIGH-RISK-GROUP probability, never an individual
    injury probability, and is only shown when the model is applicable.
"""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display in the container
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "axes.titleweight": "bold",
    "legend.fontsize":  8,
    "figure.dpi":       150,
})
# NOTE: savefig bbox is deliberately NOT "tight". A caption or annotation wider
# than its axes makes a tight bbox expand the canvas around it, which shunted
# every panel into the right half of the image and left a large void on top.
# Margins are reserved explicitly via gridspec top/bottom instead.

# Plane identity colors, shared by both figures so a reader can carry them over.
PLANE_COLORS = {
    "trunk_sagittal": "#0EA5E9",   # sky   — flexion / extension
    "trunk_lateral":  "#F59E0B",   # amber — lateral bending
    "trunk_twist":    "#8B5CF6",   # violet— axial rotation
}
PLANE_LABELS = {
    "trunk_sagittal": "Sagittal (flexion +/ extension −)",
    "trunk_lateral":  "Frontal (lateral bending)",
    "trunk_twist":    "Transverse (axial rotation)",
}

SEGMENT_LABELS = {
    "trunk":       "Trunk",
    "neck":        "Neck",
    "upper_arm_r": "Upper arm (R)",
    "upper_arm_l": "Upper arm (L)",
}
# Compact forms for the timeline lanes and the duration-bar ticks, where the
# long names ran off the left margin / collided with each other.
SHORT_LABELS = {"trunk": "Trunk", "neck": "Neck",
                "upper_arm_r": "Arm R", "upper_arm_l": "Arm L"}

# Default colouring is by SEGMENT IDENTITY — neutral, carries no ISO verdict.
SEGMENT_COLORS = {
    "trunk":       "#0EA5E9",
    "neck":        "#8B5CF6",
    "upper_arm_r": "#10B981",
    "upper_arm_l": "#14B8A6",
}

# ISO verdict colours, used ONLY when the ISO layer is explicitly on. The angle
# verdict and the duration verdict are deliberately different colours: one red
# meaning both is what let an out-of-range ANGLE be read as a holding-time
# exceedance (a 2.9 s bout "exceeding" a limit drawn in the hundreds of seconds).
_DUR_OK        = "#10B981"   # duration within the provisional holding-time limit
_DUR_EXCEEDED  = "#EF4444"   # duration beyond it — a genuine duration exceedance
_ANGLE_BAD     = "#B45309"   # ANGLE beyond ISO's upper anchor: not recommended at
                             # ANY duration. Not a duration exceedance.
_NO_CRITERION  = "#64748B"   # below ISO's lower anchor: no duration criterion

# Exposure partition of the observed interval. Unmeasured is deliberately a
# neutral grey, never blended into "acceptable" — an occluded frame is not
# evidence of a safe posture.
_EXP_HAZARD     = "#EF4444"
_EXP_ACCEPTABLE = "#10B981"
_EXP_UNMEASURED = "#94A3B8"


def _fig_to_png(fig) -> bytes:
    """Render a figure to PNG bytes and close it (containers have no display)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def _time_axis(n: int, fps: float, time_s: np.ndarray | None) -> np.ndarray:
    """Trial-relative seconds. A supplied time column is XCP-anchored (it does
    not start at 0 — see c3d_pipeline's time_s note), so it is shifted to start
    at zero FOR DISPLAY ONLY; the underlying data is never modified."""
    if time_s is not None and len(time_s) == n:
        t = np.asarray(time_s, dtype=float)
        if np.isfinite(t).any():
            return t - np.nanmin(t)
    return np.arange(n, dtype=float) / max(fps, 1e-9)


# np.gradient is one-sided (lower-order) at the array ends. This many samples at
# each end are treated as derivative-boundary and excluded from peak marking.
_EDGE_FRAMES = 2


def _peak_idx_interior(y: np.ndarray, *, exclude_edges: bool) -> int | None:
    """Index of max |y|, optionally ignoring the derivative-boundary samples."""
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).any():
        return None
    if not exclude_edges or y.size <= 2 * _EDGE_FRAMES + 1:
        return int(np.nanargmax(np.abs(y)))
    inner = np.abs(y[_EDGE_FRAMES:-_EDGE_FRAMES])
    if not np.isfinite(inner).any():
        return None
    return int(np.nanargmax(inner)) + _EDGE_FRAMES


def _peak_abs(df: pd.DataFrame, col: str) -> float | None:
    """Peak |value| of a column, excluding derivative-boundary samples."""
    if col not in df.columns:
        return None
    y = df[col].to_numpy(dtype=float)
    k = _peak_idx_interior(y, exclude_edges=True)
    return None if k is None else abs(float(y[k]))


def _draw_rep_bands(ax, rep_marks) -> None:
    """Light vertical markers at each detected repetition, so a bout in Channel 2
    and a velocity peak in Channel 3 can be lined up against the same cycle."""
    if rep_marks is None:
        return
    for i, tm in enumerate(np.asarray(rep_marks, dtype=float)):
        if np.isfinite(tm):
            ax.axvline(tm, color="#94A3B8", ls="-", lw=0.8, alpha=0.55, zorder=1,
                       label="detected repetition" if i == 0 else None)


def _draw_cycle_spans(ax, cycles) -> None:
    """Shade each segmented onset->completion span and tick its boundaries.

    Only fully-resolved cycles are shaded: a cycle whose onset or completion
    never crossed its level has no defensible extent, and drawing one anyway
    would put a boundary on the figure that the numbers deny."""
    if not cycles:
        return
    drawn = False
    for c in cycles:
        on, off = c.get("onset_s"), c.get("completion_s")
        if on is None or off is None:
            continue
        ax.axvspan(on, off, facecolor="#0EA5E9", alpha=0.07, zorder=0,
                   label="segmented cycle (onset→completion)" if not drawn else None)
        for x, style in ((on, "|"), (off, "|")):
            ax.plot([x], [ax.get_ylim()[0]], style, color="#0284C7", ms=8,
                    mew=1.4, zorder=4, clip_on=False)
        drawn = True


# ── Channel 1 — REBA posture severity ────────────────────────────────────────

# REBA action levels (Hignett & McAtamney 2000). Fixed bands — do not retune.
REBA_BANDS = [
    (1,  1,  "#D1FAE5", "1 — Negligible"),
    (2,  3,  "#FEF3C7", "2 — Low"),
    (4,  7,  "#FED7AA", "3 — Medium"),
    (8,  10, "#FECACA", "4 — High"),
    (11, 15, "#E9D5FF", "5 — Very high"),
]

# The six body-segment sub-scores REBA composes, in the order the sheet uses.
_SEGMENT_SCORES = [
    ("trunk_score",     "Trunk"),
    ("neck_score",      "Neck"),
    ("legs_score",      "Legs"),
    ("upper_arm_adj",   "Upper arm"),
    ("lower_arm_score", "Lower arm"),
    ("wrist_adj",       "Wrist"),
]


def plot_reba(reba_df: pd.DataFrame, summary: dict, *, fps: float = 60.0,
              time_s=None, title: str = "", rep_marks=None,
              angle_reference: str = "") -> bytes:
    """Channel 1 figure: frame-wise REBA score, action-level bands, and the
    segment sub-scores at the peak frame.

    Three panels:
      (A) REBA total over time on the action-level bands, with invalid
          (occluded) frames marked — they are excluded from every statistic and
          must not be read as a real, let alone worst-case, posture;
      (B) time spent in each action level (the distribution behind the mean);
      (C) segment sub-scores at the peak frame — which body parts drove it.

    `summary` is reba.summarise() output; nothing is recomputed here.
    """
    if "reba_total" not in reba_df.columns:
        raise ValueError("reba frame has no 'reba_total' column")

    t = _time_axis(len(reba_df), fps, time_s)
    total = reba_df["reba_total"].to_numpy(dtype=float)
    invalid = (reba_df["risk_level"].to_numpy() == "invalid"
               if "risk_level" in reba_df.columns else np.zeros(len(reba_df), bool))
    shown = np.where(invalid, np.nan, total)   # never draw an invalid frame

    fig = plt.figure(figsize=(11, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0],
                          hspace=0.42, wspace=0.22,
                          top=0.915, bottom=0.135, left=0.075, right=0.985)

    # ── (A) REBA over time on the action-level bands ─────────────────────────
    axa = fig.add_subplot(gs[0, :])
    y_max = max(float(np.nanmax(shown)) if np.isfinite(shown).any() else 1.0, 7.0)
    for lo, hi, color, label in REBA_BANDS:
        if lo - 0.5 <= y_max + 0.5:
            axa.axhspan(lo - 0.5, hi + 0.5, color=color, alpha=0.55, zorder=0)
            if lo <= y_max + 0.5:
                axa.text(0.004, (min(hi, y_max + 0.5) + lo - 0.5) / 2, label,
                         transform=axa.get_yaxis_transform(), va="center",
                         fontsize=7, color="#475569", zorder=2)
    _draw_rep_bands(axa, rep_marks)
    axa.plot(t, shown, color="#0F172A", lw=1.3, zorder=3)

    pk = summary.get("peak_frame")
    if pk is not None and 0 <= int(pk) < len(t):
        axa.plot(t[int(pk)], total[int(pk)], "o", ms=6, color="#DC2626",
                 mec="white", mew=1.0, zorder=5,
                 label=f"peak REBA {summary.get('max', '?')}")
        axa.legend(loc="upper right", framealpha=0.92)
    if invalid.any():
        for i in np.flatnonzero(invalid):
            axa.axvline(t[i], color="#94A3B8", lw=0.6, alpha=0.35, zorder=1)

    axa.set_ylabel("REBA total")
    axa.set_ylim(0.5, y_max + 0.5)
    axa.set_xlabel("time (s, trial-relative)")
    axa.set_title(
        f"Frame-wise REBA score  —  mean {summary.get('mean', float('nan')):.1f}, "
        f"median {summary.get('median', float('nan')):.1f}, peak {summary.get('max', '?')}",
        loc="left")
    axa.grid(axis="y", alpha=0.15, lw=0.5)

    # ── (B) time in each action level ────────────────────────────────────────
    axb = fig.add_subplot(gs[1, 0])
    dist = summary.get("risk_distribution", {}) or {}
    n_valid = max(int(summary.get("n_valid_frames", 0)), 1)
    order = [lab.split(" — ")[1].lower() for _, _, _, lab in REBA_BANDS]
    keys = [k for k in dist] or []
    # Keep the sheet's own ordering where the labels match, else fall back to
    # whatever risk_level strings the scorer produced.
    ordered = [k for k in order if k in dist] + [k for k in keys if k not in order]
    if ordered:
        pct = [100.0 * dist[k] / n_valid for k in ordered]
        colors = []
        for k in ordered:
            match = [c for _, _, c, lab in REBA_BANDS if lab.split(" — ")[1].lower() == k]
            colors.append(match[0] if match else "#CBD5E1")
        bars = axb.bar(np.arange(len(ordered)), pct, color=colors,
                       edgecolor="#94A3B8", lw=0.6)
        for bar, p, k in zip(bars, pct, ordered):
            axb.text(bar.get_x() + bar.get_width() / 2, p, f"{p:.0f}%",
                     ha="center", va="bottom", fontsize=8)
        axb.set_xticks(np.arange(len(ordered)))
        axb.set_xticklabels([k.capitalize() for k in ordered], fontsize=8)
        axb.set_ylim(0, max(pct) * 1.22 if pct else 1)
    axb.set_ylabel("% of valid frames")
    axb.set_title("Time in each REBA action level", loc="left")
    axb.grid(axis="y", alpha=0.18, lw=0.5)

    # ── (C) segment sub-scores at the peak frame ─────────────────────────────
    axc = fig.add_subplot(gs[1, 1])
    labels, vals = [], []
    for col, lab in _SEGMENT_SCORES:
        key = "peak_" + col.replace("_score", "").replace("_adj", "")
        v = summary.get(key)
        if v is None and col in reba_df.columns and pk is not None:
            v = reba_df[col].iloc[int(pk)]
        if v is not None and np.isfinite(float(v)):
            labels.append(lab)
            vals.append(float(v))
    if vals:
        x = np.arange(len(vals))
        axc.bar(x, vals, color="#0EA5E9", width=0.6)
        for xi, v in zip(x, vals):
            axc.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        axc.set_xticks(x)
        axc.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
        axc.set_ylim(0, max(vals) * 1.25)
    axc.set_ylabel("sub-score")
    axc.set_title(f"Segment sub-scores at the peak frame (#{pk})", loc="left")
    axc.grid(axis="y", alpha=0.18, lw=0.5)

    fig.suptitle(
        f"Preliminary Vicon-derived REBA reference{('  ·  ' + title) if title else ''}",
        fontsize=12, fontweight="bold", y=0.978)
    n_drop = int(summary.get("n_dropped_frames", 0))
    fig.text(0.5, 0.052,
             f"REBA severity computed from the same signed anatomical angles as Channels 2 and 3, "
             f"with the SAME deterministic REBA implementation and task metadata used for the "
             f"Vicon-derived reference. Load/coupling/activity are entered task metadata, never "
             f"inferred from pose. {n_drop} of {summary.get('n_frames', '?')} frames were invalid "
             f"(occluded marker) and are excluded from every statistic and from peak detection. "
             f"This is a Vicon-derived reference, not expert-validated REBA."
             + (f" {angle_reference}" if angle_reference else ""),
             ha="center", va="top", fontsize=7.5, color="#475569", wrap=True)
    return _fig_to_png(fig)


# ── Channel 2 — posture duration ─────────────────────────────────────────────

def plot_posture_duration(ch2: dict, angles: dict[str, np.ndarray],
                          fps: float, *, time_s=None, title: str = "",
                          rep_marks=None, show_iso: bool | None = None,
                          angle_reference: str = "") -> bytes:
    """Channel 2 figure: angle traces + bout shading, bout timeline, durations.

    Three panels, in the order a reader needs them:
      (A) per-segment signed angle with the bout threshold and every detected
          bout shaded — onset/offset readable against the signal that made them;
      (B) bout timeline, one lane per segment, ONE NEUTRAL COLOUR PER SEGMENT;
      (C) bout durations on a LINEAR axis, with longest/total/%-of-task called out.

    ISO 11226 colouring and limit lines appear ONLY when `show_iso` resolves
    true, which by default follows `ch2["iso_assessment"]["applicable"]`. For a
    repetitive dynamic lifting trial that is False, so the figure carries no ISO
    colouring at all: a red/green chart is read long before its caption, and ISO
    11226 covers static postures under minimal external force — which a loaded
    lifting cycle is not.

    Args:
        ch2:    channels.compute_posture_duration() output.
        angles: the same signed angle dict that produced it (for the traces).
        fps:    sample rate of those angle series.
        time_s: optional time column; shifted to trial-relative for display.
        rep_marks: optional repetition times (s, trial-relative) drawn as light
                   vertical bands so Channels 2 and 3 line up temporally.
        show_iso: force the ISO layer on/off; None = follow iso_assessment.
    """
    segs: dict = ch2.get("segments", {})
    if not segs:
        raise ValueError("posture_duration payload has no segments to plot")

    iso_ok = ch2.get("iso_assessment", {}).get("applicable", False)
    show_iso = iso_ok if show_iso is None else bool(show_iso)

    seg_keys = [k for k in ("trunk", "neck", "upper_arm_r", "upper_arm_l") if k in segs]
    n_trace = len(seg_keys)
    all_bouts = [(k, b) for k in seg_keys for b in segs[k].get("bouts", [])]
    n_static = sum(1 for _, b in all_bouts if b.get("is_static_hold") is True)
    static_evaluated = any(segs[k].get("static_evaluated") for k in seg_keys)

    fig = plt.figure(figsize=(11, 1.75 * n_trace + 6.6))
    gs = fig.add_gridspec(
        n_trace + 3, 1,
        height_ratios=[1.0] * n_trace + [0.16 * max(n_trace, 2) + 0.75, 1.30, 0.95],
        hspace=0.62, top=0.935, bottom=0.115, left=0.085, right=0.985)

    def _bout_color(b: dict, seg_key: str) -> str:
        """Colour carries SEGMENT identity; the static/dynamic distinction is
        carried by hatching instead (see _bout_hatch), and the ISO holding-time
        verdict only ever applies to a bout that already passed BOTH the static
        test and the external-force test."""
        if b.get("iso_interpretation") == "limit_exceeded":
            return _DUR_EXCEEDED
        return SEGMENT_COLORS.get(seg_key, "#10B981")

    def _bout_hatch(b: dict) -> str | None:
        """Hatch a STATIC high-risk hold. An unhatched bar is an above-threshold
        bout that includes movement — it is NOT a sustained static load, which is
        the single most important thing this figure must not let a reader
        assume."""
        return "///" if b.get("is_static_hold") is True else None

    # ── (A) per-segment angle trace with bouts shaded ────────────────────────
    t_ref = None
    ax0 = None
    for row, key in enumerate(seg_keys):
        d = segs[key]
        ax = fig.add_subplot(gs[row, 0], sharex=ax0) if ax0 is not None \
            else fig.add_subplot(gs[row, 0])
        ax0 = ax0 or ax
        series = np.asarray(angles.get(d.get("angle_channel", ""), []), dtype=float)
        t = _time_axis(len(series), fps, time_s)
        t_ref = t if t_ref is None else t_ref

        _draw_rep_bands(ax, rep_marks)
        ax.axhline(0, color="#94A3B8", lw=0.7, zorder=1)
        ax.plot(t, series, color="#1E293B", lw=1.1, zorder=3)

        # Entry/exit thresholds. NOTE the key: the layer renamed this to
        # entry_threshold_deg, and reading the old angle_threshold_deg silently
        # returned None, which removed the threshold lines from every panel
        # without any error. Read the current key and label it on the plot so a
        # missing line is visible rather than invisible.
        thr = d.get("entry_threshold_deg")
        thr_off = d.get("exit_threshold_deg")
        if thr is not None:
            for sign in (+1, -1):
                ax.axhline(sign * thr, color="#DC2626", ls=":", lw=1.0, zorder=2,
                           label="entry threshold" if sign > 0 else None)
            if thr_off is not None and abs(thr_off - thr) > 1e-9:
                for sign in (+1, -1):
                    ax.axhline(sign * thr_off, color="#F59E0B", ls=":", lw=0.8,
                               alpha=0.75, zorder=2,
                               label="exit threshold (hysteresis)" if sign > 0 else None)
            # No in-axes text label: the dotted lines plus the legend carry it,
            # and a red caption box on every panel crowded the traces. The
            # numeric values live in the caption band, the API response, and the
            # Analyse table.

        for b in d.get("bouts", []):
            t0 = t[b["onset_frame"]] if b["onset_frame"] < len(t) else t[-1]
            t1 = t[b["offset_frame"]] if b["offset_frame"] < len(t) else t[-1]
            ax.axvspan(t0, t1, facecolor=_bout_color(b, key), alpha=0.16, zorder=0,
                       hatch=_bout_hatch(b), edgecolor="#475569", lw=0.0)

        ax.set_ylabel("angle (deg)")
        ax.set_title(f"{SEGMENT_LABELS.get(key, key)}  —  {d.get('n_bouts', 0)} bout(s), "
                     f"{d.get('total_high_risk_s', 0.0):.1f} s above threshold "
                     f"({d.get('percent_task_time', 0.0):.0f}% of task)", loc="left")
        ax.grid(alpha=0.18, lw=0.5)
        if row == 0 and ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", framealpha=0.9)
        if row < n_trace - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("time (s, trial-relative)")

    # ── (B) bout timeline ────────────────────────────────────────────────────
    axt = fig.add_subplot(gs[n_trace, 0], sharex=ax0)
    _draw_rep_bands(axt, rep_marks)
    for lane, key in enumerate(seg_keys):
        for b in segs[key].get("bouts", []):
            t0 = t_ref[b["onset_frame"]] if b["onset_frame"] < len(t_ref) else t_ref[-1]
            t1 = t_ref[b["offset_frame"]] if b["offset_frame"] < len(t_ref) else t_ref[-1]
            axt.barh(lane, max(t1 - t0, 1e-3), left=t0, height=0.55,
                     color=_bout_color(b, key), edgecolor="white", lw=0.6, zorder=3,
                     hatch=_bout_hatch(b))
    axt.set_yticks(range(len(seg_keys)))
    axt.set_yticklabels([SHORT_LABELS.get(k, k) for k in seg_keys])
    # An empty lane above the first segment holds the legend, so it can never
    # sit on top of a bout bar (it hid a whole trunk bout before).
    axt.set_ylim(len(seg_keys) - 0.5, -1.5)
    axt.set_xlabel("time (s, trial-relative)")
    axt.set_title("Above-threshold posture bouts (hysteresis-gated; onset/offset back-dated)",
                  loc="left")
    axt.grid(axis="x", alpha=0.18, lw=0.5)
    if show_iso:
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=_DUR_OK,       label="duration within ISO limit"),
            plt.Rectangle((0, 0), 1, 1, color=_DUR_EXCEEDED, label="duration exceeds ISO limit"),
            plt.Rectangle((0, 0), 1, 1, color=_ANGLE_BAD,    label="ANGLE not recommended (any duration)"),
            plt.Rectangle((0, 0), 1, 1, color=_NO_CRITERION, label="no ISO duration criterion"),
        ]
        axt.legend(handles=handles, loc="upper right", framealpha=0.95, ncol=2, fontsize=7)
    else:
        handles = [plt.Rectangle((0, 0), 1, 1, color=SEGMENT_COLORS.get(k, "#10B981"),
                                 label=SEGMENT_LABELS.get(k, k)) for k in seg_keys]
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="#CBD5E1", hatch="///",
                                     edgecolor="#475569",
                                     label="static hold (S=1)" if n_static else
                                           "static hold (none detected)"))
        axt.legend(handles=handles, loc="upper right", framealpha=0.95,
                   ncol=len(seg_keys) + 1, fontsize=7)

    # ── (C) bout durations, LINEAR axis ──────────────────────────────────────
    # Linear, not log: the primary result is how each bout's length compares with
    # the others and with the trial, all inside one order of magnitude. The log
    # axis existed only to fit ISO's minutes-scale limits alongside seconds-scale
    # bouts; with the ISO layer off there is nothing left to compress.
    axd = fig.add_subplot(gs[n_trace + 1, 0])
    labels = [f"{SHORT_LABELS.get(k, k)}\n#{i}"
              for k in seg_keys
              for i, _ in enumerate(segs[k].get("bouts", []), start=1)]
    durs   = [b["duration_s"] for _, b in all_bouts]
    colors = [_bout_color(b, k) for k, b in all_bouts]
    if durs:
        x = np.arange(len(durs))
        axd.bar(x, durs, color=colors, width=0.62, zorder=3,
                hatch=[_bout_hatch(b) for _, b in all_bouts], edgecolor="#475569", lw=0.5)
        for xi, dv in zip(x, durs):
            axd.text(xi, dv, f"{dv:.1f}", ha="center", va="bottom", fontsize=7, zorder=4)
        axd.set_xticks(x)
        axd.set_xticklabels(labels, fontsize=7)
        axd.set_xlim(-0.7, len(durs) - 0.3)
        y_top = max(durs) * 1.24
        if show_iso:
            drawn = False
            for xi, (_, b) in zip(x, all_bouts):
                lim = b.get("iso_limit_s")
                if lim is not None:
                    axd.plot([xi - 0.38, xi + 0.38], [lim, lim], color="#1E293B",
                             ls="--", lw=1.3, zorder=5,
                             label=None if drawn else "ISO 11226 holding-time limit (provisional)")
                    drawn = True
                    y_top = max(y_top, lim * 1.24)
            if drawn:
                axd.legend(loc="upper left", framealpha=0.9)
        axd.set_ylim(0, y_top)
    else:
        axd.text(0.5, 0.5, "No above-threshold bouts detected in this trial",
                 ha="center", va="center", color="#64748B", transform=axd.transAxes)
        axd.set_xticks([])
    axd.set_ylabel("bout duration (s)")
    axd.set_title("Bout duration", loc="left")
    axd.grid(axis="y", alpha=0.18, lw=0.5)

    # ── (D) exposure partition: hazardous vs acceptable vs unmeasured ────────
    axe = fig.add_subplot(gs[n_trace + 2, 0])
    y = np.arange(len(seg_keys))
    haz = [segs[k].get("exposure", {}).get("hazardous_exposure_s", 0.0) for k in seg_keys]
    acc = [segs[k].get("exposure", {}).get("acceptable_posture_s", 0.0) for k in seg_keys]
    unm = [segs[k].get("exposure", {}).get("unmeasured_s", 0.0) for k in seg_keys]
    axe.barh(y, haz, height=0.55, color=_EXP_HAZARD, label="hazardous (above threshold)")
    axe.barh(y, acc, height=0.55, left=haz, color=_EXP_ACCEPTABLE,
             label="acceptable (below threshold)")
    axe.barh(y, unm, height=0.55, left=np.add(haz, acc), color=_EXP_UNMEASURED,
             hatch="xx", edgecolor="white", lw=0.5, label="unmeasured (occluded)")
    span = max([h + a + u for h, a, u in zip(haz, acc, unm)] or [1.0])
    for i, k in enumerate(seg_keys):
        e = segs[k].get("exposure", {})
        pct = e.get("percent_hazardous_of_measured")
        if pct is None or haz[i] <= 0:
            continue
        txt = f"{haz[i]:.1f}s ({pct:.0f}%)"
        # A short bar cannot hold its own label — put it outside in dark ink
        # instead of letting white text spill across the axis.
        if haz[i] < 0.22 * span:
            axe.text(haz[i] + 0.01 * span, i, txt, ha="left", va="center",
                     fontsize=7.5, color="#1E293B", fontweight="bold", zorder=5)
        else:
            axe.text(haz[i] / 2, i, txt, ha="center", va="center",
                     fontsize=7.5, color="white", fontweight="bold", zorder=5)
    axe.set_yticks(y)
    axe.set_yticklabels([SHORT_LABELS.get(k, k) for k in seg_keys])
    # Empty lane below the last segment holds the legend clear of every bar.
    axe.set_ylim(len(seg_keys) + 0.4, -0.6)
    axe.set_xlabel("time (s)")
    axe.set_title("Exposure partition of the observed interval "
                  "(% is of MEASURED time, excluding occlusion)", loc="left")
    axe.grid(axis="x", alpha=0.18, lw=0.5)
    axe.legend(loc="lower center", framealpha=0.95, ncol=3, fontsize=7)

    hyst = ch2.get("hysteresis", {})
    fig.suptitle(
        f"Preliminary Vicon-derived posture-bout reference{('  ·  ' + title) if title else ''}",
        fontsize=12, fontweight="bold", y=0.982)
    totals = " | ".join(
        f"{SHORT_LABELS.get(k, k)}: {segs[k]['n_bouts']} bouts, "
        f"{segs[k]['total_high_risk_s']:.1f} s total, "
        f"longest {segs[k]['longest_bout_s']:.1f} s, {segs[k]['percent_task_time']:.0f}% of task"
        for k in seg_keys)
    sc = ch2.get("static_criteria", {}) or {}
    if not static_evaluated:
        static_note = (" Static classification NOT EVALUATED — the low-motion thresholds "
                       "(eps_omega, eps_theta) have not been derived and frozen, so no bout "
                       "is claimed to be a static hold.")
    else:
        static_note = (f" {n_static} of {len(all_bouts)} bouts met the static criteria "
                       f"(duration > {sc.get('min_duration_s', 4)} s, median |omega| < eps_omega, "
                       f"SD(theta) < eps_theta) and are hatched.")
    iso_line = ("ISO 11226 holding-time comparison is evaluated PER BOUT (static AND minimal "
                "external force), not per task; anchors remain unverified against "
                "ISO 11226:2000.")
    fig.text(0.5, 0.070, totals, ha="center", va="top", fontsize=7.5, color="#334155", wrap=True)
    fig.text(0.5, 0.040,
             f"Above-threshold bouts include the rising and falling phases of a movement, so they "
             f"are NOT static-posture durations.{static_note} Hysteresis: entry/exit angle "
             f"thresholds from REBA segment categories, plus d_on/d_off "
             f"{hyst.get('enter_s', '?')} s / exit {hyst.get('exit_s', '?')} s at "
             f"{ch2.get('fps', fps):.4g} Hz, with onset and offset back-dated to the first and last "
             f"qualifying frame. {iso_line}"
             + (f" {angle_reference}" if angle_reference else ""),
             ha="center", va="top", fontsize=7.5, color="#475569", wrap=True)
    return _fig_to_png(fig)


# ── Channel 3 — trunk dynamics ───────────────────────────────────────────────

def plot_trunk_dynamics(dyn_df: pd.DataFrame, ch3: dict,
                        *, time_s=None, title: str = "", rep_marks=None) -> bytes:
    """Channel 3 figure: signed angle / velocity / acceleration + summary table.

    Derivative boundaries are drawn as hatched end zones and excluded from peak
    marking: `np.gradient` is 2nd-order in the interior but one-sided at the
    first and last sample, so those points are lower-order and must not be read
    as ordinary measurements (they are where spurious extrema tend to land).
    """
    fps = float(ch3.get("fps") or 60.0)
    planes = [p for p in PLANE_COLORS if f"{p}_deg" in dyn_df.columns]
    if not planes:
        raise ValueError("trunk_dynamics frame has none of the expected trunk columns")

    t_col = time_s
    if t_col is None and "time_s" in dyn_df.columns:
        t_col = dyn_df["time_s"].to_numpy()
    t = _time_axis(len(dyn_df), fps, t_col)

    # Taller table panel: the summary grew to 11 rows once cycle segmentation
    # (duration / rise-return / recovery / duty cycle) was added, and the old
    # 0.95 ratio let it overflow into the caption band.
    fig = plt.figure(figsize=(11, 12.4))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 1.0, 1.55], hspace=0.44,
                          top=0.945, bottom=0.115, left=0.085, right=0.985)

    rows = [
        ("_deg",       "angle (°)",             "Signed trunk angle"),
        ("_vel_deg_s", "velocity (°/s)",        "Trunk angular velocity"),
        ("_acc_deg_s2", "acceleration (°/s²)",  "Trunk angular acceleration"),
    ]
    ax0 = None
    for row, (suffix, ylabel, heading) in enumerate(rows):
        ax = fig.add_subplot(gs[row, 0], sharex=ax0) if ax0 is not None \
            else fig.add_subplot(gs[row, 0])
        ax0 = ax0 or ax
        _draw_rep_bands(ax, rep_marks)
        if row == 0:
            _draw_cycle_spans(ax, (ch3.get("repetition") or {}).get("cycles"))
        ax.axhline(0, color="#94A3B8", lw=0.7, zorder=1)
        # Derivative boundary: one-sided (lower-order) for velocity/acceleration.
        # Marked on every row so the same time span is excluded consistently.
        if row > 0 and len(t) > 2 * _EDGE_FRAMES:
            for x0, x1 in ((t[0], t[_EDGE_FRAMES]), (t[-1 - _EDGE_FRAMES], t[-1])):
                ax.axvspan(x0, x1, facecolor="none", edgecolor="#94A3B8",
                           hatch="///", lw=0.0, alpha=0.5, zorder=0)
        for p in planes:
            col = f"{p}{suffix}"
            if col not in dyn_df.columns:
                continue
            y = dyn_df[col].to_numpy(dtype=float)
            ax.plot(t, y, color=PLANE_COLORS[p], lw=1.15,
                    label=PLANE_LABELS[p] if row == 0 else None, zorder=3)
            # Mark the peak |value| — the quantity the Marras predictors report.
            # Boundary samples are excluded: a one-sided derivative there can
            # manufacture the largest value in the whole trace.
            k = _peak_idx_interior(y, exclude_edges=(row > 0))
            if k is not None:
                ax.plot(t[k], y[k], "o", ms=4.5, color=PLANE_COLORS[p],
                        mec="white", mew=0.8, zorder=4)
        ax.set_ylabel(ylabel)
        ax.set_title(heading, loc="left")
        ax.grid(alpha=0.18, lw=0.5)
        if row == 0:
            # Headroom so the legend never sits on top of the traces — the
            # sagittal peak is the tallest feature and reads first.
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + (hi - lo) * 0.30)
            ax.legend(loc="upper center", framealpha=0.92, ncol=3, fontsize=7.5)
        if row < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("time (s, trial-relative)")

    # ── Task-level summary table (replaces the bar panel) ────────────────────
    # A table, not bars: these five quantities are in three different units, so a
    # shared "magnitude" axis compared things that are not comparable — and when
    # an input is missing the bar panel simply rendered empty, which is not a
    # reportable result. A table states the value AND its availability.
    axp = fig.add_subplot(gs[3, 0])
    axp.axis("off")
    pred = ch3.get("marras_predictors", {}) or {}
    rep = ch3.get("repetition", {}) or {}
    marras = ch3.get("marras_probability", {}) or {}

    def _fmt(v, unit=""):
        return "—" if v is None else f"{float(v):.1f}{unit}"

    n_reps = rep.get("n_reps")
    # Task-aware labelling: a kneeling trial's trunk-flexion cadence is a
    # MOVEMENT CYCLE rate, not a lift rate. Only a lift task gets the lift label,
    # and only a lift task carries a non-None lift_rate_per_hr into Marras.
    is_lift = rep.get("cycle_kind") == "lift"
    cycle_word = "Lift" if is_lift else "Movement-cycle"
    lift_rate = rep.get("lift_rate_per_hr")
    cycle_rate = rep.get("cycle_rate_per_hr", lift_rate)
    plane = rep.get("counting_channel", "trunk_sagittal_deg")
    _tim = rep.get("cycle_timing") or {}
    plane_word = {"trunk_sagittal_deg": "sagittal", "trunk_lateral_deg": "frontal",
                  "trunk_twist_deg": "transverse"}.get(plane, plane)
    rows_tbl = [
        ("Maximum sagittal flexion",  _fmt(pred.get("peak_trunk_sagittal_deg"), "°"),
         "yes" if pred.get("peak_trunk_sagittal_deg") is not None else "no"),
        ("Mean twisting velocity",    _fmt(pred.get("mean_twist_vel_deg_s"), "°/s"),
         "yes" if pred.get("mean_twist_vel_deg_s") is not None else "no"),
        ("Maximum lateral velocity",  _fmt(pred.get("peak_lateral_vel_deg_s"), "°/s"),
         "yes" if pred.get("peak_lateral_vel_deg_s") is not None else "no"),
        (f"{cycle_word} rate (within-trial)",
         "—" if cycle_rate is None else f"{float(cycle_rate):.0f} /h",
         "within-trial only" if cycle_rate else "no"),
        ("Maximum external-load moment", "—", "no (task metadata)"),
        (f"Cycles counted ({plane_word} plane)",
         "—" if n_reps is None else str(int(n_reps)),
         "yes" if n_reps is not None else "no"),
        ("Mean cycle duration (onset→completion)",
         _fmt(_tim.get("mean_cycle_duration_s"), " s"),
         (f"{_tim.get('n_cycles_complete', 0)}/{_tim.get('n_cycles_segmented', 0)} "
          f"fully segmented") if _tim else "no"),
        ("Mean rise / return time",
         (f"{_fmt(_tim.get('mean_rise_time_s'))} / "
          f"{_fmt(_tim.get('mean_return_time_s'), ' s')}") if _tim else "—",
         "yes" if _tim.get("mean_rise_time_s") is not None else "no"),
        ("Mean recovery between cycles",
         _fmt(_tim.get("mean_recovery_s"), " s"),
         "within-trial only" if _tim.get("mean_recovery_s") is not None else "no"),
        ("Duty cycle (time inside a cycle)",
         _fmt(_tim.get("duty_cycle_percent"), " %"),
         "within-trial only" if _tim.get("duty_cycle_percent") is not None else "no"),
        ("Peak sagittal acceleration", _fmt(_peak_abs(dyn_df, "trunk_sagittal_acc_deg_s2"), "°/s²"),
         "yes (separate output, not a Marras predictor)"),
    ]
    tbl = axp.table(
        cellText=[[a, b, c] for a, b, c in rows_tbl],
        colLabels=["Variable", "Vicon-derived value", "Available"],
        cellLoc="left", colLoc="left", loc="upper center",
        colWidths=[0.36, 0.24, 0.40])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.28)
    for (r, _c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#F1F5F9")
            cell.set_text_props(fontweight="bold")
    axp.set_title("Task-level trunk-motion summary "
                  "(Marras et al., 1993 predictors + repetition)", loc="left")

    if marras.get("applicable") and marras.get("probability") is not None:
        verdict = (f"Marras high-risk-GROUP probability: {marras['probability']:.3f} "
                   f"— a group-level probability, NOT an individual injury probability.")
    else:
        missing = []
        if not is_lift:
            missing.append("a lift rate (this task's cycles are movement cycles, "
                           "not lifts, so the counted cadence is not a lift rate)")
        elif not lift_rate:
            missing.append("lift rate")
        missing.append("maximum external-load moment")
        verdict = ("Marras probability was NOT calculated because "
                   + " and ".join(missing)
                   + (" were" if len(missing) > 1 else " was") + " unavailable.")

    cutoff = ch3.get("cutoff_hz")
    fig.suptitle(
        f"Preliminary Vicon-derived trunk-kinematics reference"
        f"{('  ·  ' + title) if title else ''}",
        fontsize=12, fontweight="bold", y=0.982)
    fig.text(0.5, 0.072, verdict, ha="center", va="top", fontsize=7.5,
             color="#334155", wrap=True)
    fig.text(0.5, 0.043,
             f"Signed plane-specific angles, unwrapped before differentiation at {fps:.4g} Hz with "
             f"cutoff_hz = {cutoff} "
             f"({'Vicon path — markers already 6 Hz zero-phase filtered upstream' if cutoff is None else 'RGB/model path'}). "
             f"Filled markers show each channel's peak |value|; hatched end zones are the "
             f"derivative boundary ({_EDGE_FRAMES} frames), where np.gradient is one-sided and "
             f"lower-order — excluded from peak marking and not to be used as ordinary samples. "
             f"Angular acceleration is a separate dynamics output, NOT one of the five final "
             f"Marras predictors. Cycles counted on the {plane_word} plane."
             + (f" NOTE: {rep['plane_warning']}" if rep.get("plane_warning") else ""),
             ha="center", va="top", fontsize=7.5, color="#475569", wrap=True)
    return _fig_to_png(fig)


# ── Self-check ───────────────────────────────────────────────────────────────

def _demo() -> None:
    """Assert-based self-check: both figures render non-trivial PNG bytes."""
    from . import channels as _ch
    from . import posture_duration as _dur

    fps = 60.0
    t = np.arange(0, 6, 1 / fps)
    sag = 60 + 60 * np.sin(2 * np.pi * 0.4 * t)
    small = 5 * np.sin(2 * np.pi * 0.4 * t)
    angles = {
        "trunk_sagittal_deg": sag,
        "trunk_lateral_deg":  small,
        "trunk_twist_deg":    small,
        "neck_flex_deg":      30 + small,
        "upper_arm_r_deg":    70 + small,
        "upper_arm_l_deg":    15 + small,
    }

    # Primary descriptors are produced with no ISO involvement at all.
    ch2 = _ch.compute_posture_duration(angles, fps, task_id="SUB01_lifting_T01")
    png2 = plot_posture_duration(ch2, angles, fps, title="SELFCHECK")
    assert png2[:8] == b"\x89PNG\r\n\x1a\n", "channel-2 output is not a PNG"
    assert len(png2) > 20_000, f"channel-2 PNG suspiciously small: {len(png2)}B"
    tr = ch2["segments"]["trunk"]
    for k in ("n_bouts", "total_high_risk_s", "longest_bout_s", "percent_task_time"):
        assert k in tr, f"missing primary descriptor {k}"
    assert 0.0 <= tr["percent_task_time"] <= 100.0

    # STATIC STATUS IS NEVER GUESSED. With no frozen eps, every bout reports
    # "not evaluated" — a dynamic transition must never be shown as a static hold.
    assert tr["static_evaluated"] is False
    assert all(b["is_static_hold"] is None for b in tr["bouts"])
    assert ch2["static_criteria"]["thresholds_frozen"] is False

    # THE REGRESSION THAT MATTERS: an ANGLE beyond ISO's upper anchor is an angle
    # verdict, never a holding-time exceedance. A 3 s bout at 95 deg must not be
    # reported as exceeding a limit measured in hundreds of seconds.
    t_deep = np.arange(0, 24, 1 / fps)
    deep = np.where((t_deep % 12) < 6, 95.0, 0.0)     # 6 s holds, smooth
    rdeep = _dur.segment_duration_bouts(
        "trunk", deep, fps, angular_vel_deg_s=np.gradient(deep, 1 / fps),
        eps_omega=2.0, eps_theta=1.0, load_score=0)
    for b in rdeep["bouts"]:
        if b["is_static_hold"]:
            assert b["iso_holding_time_ratio"] is None, b
            assert b["iso_interpretation"] == "angle_not_recommended", b

    # ISO ratio is gated by external force (criterion 5), which is entered
    # metadata — a loaded static hold gets no holding-time ratio.
    rload = _dur.segment_duration_bouts(
        "trunk", deep, fps, angular_vel_deg_s=np.gradient(deep, 1 / fps),
        eps_omega=2.0, eps_theta=1.0, load_score=3)
    assert all(b["iso_holding_time_ratio"] is None for b in rload["bouts"])

    # THRESHOLD LINES MUST BE DRAWN. Renaming angle_threshold_deg ->
    # entry_threshold_deg once removed them from every panel with no error at
    # all, so assert the figure actually reads the key the layer emits.
    tr_seg = ch2["segments"]["trunk"]
    assert "entry_threshold_deg" in tr_seg and "exit_threshold_deg" in tr_seg, tr_seg.keys()
    assert tr_seg["entry_threshold_deg"] == 20.0, tr_seg["entry_threshold_deg"]
    assert ch2["segments"]["upper_arm_r"]["entry_threshold_deg"] == 45.0

    # A figure rendered WITH static bouts must still render (hatching path).
    t12 = np.arange(0, 12, 1 / fps)
    hold = {"trunk_sagittal_deg": np.where(t12 < 8, 30.0 + 0.3 * np.sin(2 * np.pi * 0.05 * t12), 0.0)}
    ch2_static = _ch.compute_posture_duration(
        hold, fps, task_id="SUB01_forward_trunk_bending_T01", load_score=0,
        frozen_thresholds={"trunk": (2.0, 1.0)})
    assert ch2_static["segments"]["trunk"]["n_static_holds"] == 1
    assert plot_posture_duration(ch2_static, hold, fps)[:8] == b"\x89PNG\r\n\x1a\n"

    dyn_df, ch3 = _ch.compute_trunk_dynamics(angles, fps, "SUB01_lifting_T01")
    reps = ch3.get("repetition", {}).get("peak_times_s")
    png3 = plot_trunk_dynamics(dyn_df, ch3, title="SELFCHECK", rep_marks=reps)
    assert png3[:8] == b"\x89PNG\r\n\x1a\n", "channel-3 output is not a PNG"
    assert len(png3) > 20_000, f"channel-3 PNG suspiciously small: {len(png3)}B"

    # Derivative-boundary exclusion: a huge spike planted in the first sample
    # must not be chosen as the peak of a differentiated channel.
    y = np.zeros(200); y[0] = 1e6; y[100] = 10.0
    assert _peak_idx_interior(y, exclude_edges=True) == 100
    assert _peak_idx_interior(y, exclude_edges=False) == 0

    # Channel 1 renders from the same angles, with entered task metadata.
    from . import reba as _reba
    ang_full = dict(angles)
    ang_full.update({
        "neck_lateral_deg": np.zeros_like(t), "upper_arm_r_abd_deg": np.zeros_like(t),
        "upper_arm_l_abd_deg": np.zeros_like(t), "lower_arm_r_deg": 60 + np.zeros_like(t),
        "lower_arm_l_deg": 60 + np.zeros_like(t), "wrist_r_deg": np.zeros_like(t),
        "wrist_l_deg": np.zeros_like(t), "knee_max_deg": 20 + np.zeros_like(t),
    })
    reba_df = _reba.score_timeseries(ang_full, load=1, coupling=0, activity=1)
    reba_sum = _reba.summarise(reba_df)
    png1 = plot_reba(reba_df, reba_sum, fps=fps, title="SELFCHECK", rep_marks=reps)
    assert png1[:8] == b"\x89PNG\r\n\x1a\n", "channel-1 output is not a PNG"
    assert len(png1) > 20_000, f"channel-1 PNG suspiciously small: {len(png1)}B"

    # An unlabelled upload must NOT default to ISO-applicable: if the task cannot
    # be named, it cannot be claimed static and minimally loaded.
    ch2_unlabelled = _ch.compute_posture_duration(angles, fps, task_id="")
    assert ch2_unlabelled["iso_assessment"]["applicable"] is False,         "an unknown task must fail SAFE on ISO applicability, not permissive"

    # A trial with no bouts must still render (empty-state path), not raise.
    calm = {k: np.zeros_like(v) for k, v in angles.items()}
    ch2_calm = _ch.compute_posture_duration(calm, fps, task_id="SUB01_lifting_T01")
    assert plot_posture_duration(ch2_calm, calm, fps)[:8] == b"\x89PNG\r\n\x1a\n"

    # A payload with no segments at all is an error, not a blank figure.
    try:
        plot_posture_duration({"segments": {}}, angles, fps)
        raise AssertionError("expected ValueError for an empty segments payload")
    except ValueError:
        pass

    print("channel_figures self-check OK "
          f"(ch2 {len(png2)//1024} KB, ch3 {len(png3)//1024} KB)")


if __name__ == "__main__":
    _demo()
