"""Self-contained regression checks for c3d_pipeline.py's fallback-CSV merge,
4-state source classification, and identity/rate validation.

Run: python -m core.test_c3d_pipeline
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.c3d_pipeline import (
    OUTPUT_COLUMNS, classify_joint_source, find_fully_valid_window,
    _merge_fallback, _validate_fallback_csv,
)


def test_find_fully_valid_window_trims_both_ends():
    # HD missing at the start (frames 0-1) and end (frame 5); PELCEN always
    # present. The fully-valid window is the INTERSECTION: frames 2-4.
    df = pd.DataFrame({
        "HD_X":     [np.nan, np.nan, 1.0, 2.0, 3.0, np.nan],
        "PELCEN_X": [1.0,    1.0,    1.0, 1.0, 1.0, 1.0],
    })
    lo, hi = find_fully_valid_window(df, ["HD_X", "PELCEN_X"])
    assert (lo, hi) == (2, 4), (lo, hi)


def test_find_fully_valid_window_excludes_permanently_missing_joint():
    # RTOE is NaN for the entire trial (never placed) — it must not block
    # trimming on the other, consistently-tracked joints.
    df = pd.DataFrame({
        "HD_X":     [np.nan, 1.0, 2.0, 3.0],
        "PELCEN_X": [1.0,    1.0, 1.0, 1.0],
        "RTOE_X":   [np.nan, np.nan, np.nan, np.nan],
    })
    lo, hi = find_fully_valid_window(df, ["HD_X", "PELCEN_X", "RTOE_X"])
    assert (lo, hi) == (1, 3), (lo, hi)


def test_find_fully_valid_window_raises_when_never_simultaneously_valid():
    # HD and PELCEN never overlap — no frame has both.
    df = pd.DataFrame({
        "HD_X":     [1.0, 1.0, np.nan, np.nan],
        "PELCEN_X": [np.nan, np.nan, 1.0, 1.0],
    })
    try:
        find_fully_valid_window(df, ["HD_X", "PELCEN_X"])
        raise AssertionError("expected ValueError when no frame has both joints valid")
    except ValueError as e:
        assert "consistently-tracked joints valid" in str(e)


def test_classify_joint_source():
    orig      = np.array([True,  False, False, False, True])
    interp    = np.array([False, True,  False, False, False])
    recovered = np.array([False, False, True,  False, False])
    final     = np.array([True,  True,  True,  False, True])

    got = classify_joint_source(orig, interp, recovered, final)
    expected = np.array(["c3d", "interpolated", "recovered_csv", "invalid", "c3d"], dtype=object)
    assert list(got) == list(expected), got

    # final_valid always wins, even if an upstream mask disagrees (defends
    # against a future reordering of the pipeline breaking mutual exclusivity).
    orig2  = np.array([True])
    interp2 = np.array([False])
    rec2   = np.array([False])
    final2 = np.array([False])   # e.g. despiked/dropped after the snapshot
    assert classify_joint_source(orig2, interp2, rec2, final2)[0] == "invalid"


def _fake_output_df(frames: np.ndarray, x_values: dict) -> pd.DataFrame:
    """Build a minimal OUTPUT_COLUMNS-shaped DataFrame; joints not in
    x_values get an all-NaN X/Y/Z."""
    data = {"Frame": frames}
    for col in OUTPUT_COLUMNS[1:]:
        name = col[:-2]
        data[col] = x_values.get(name, np.full(len(frames), np.nan))
    return pd.DataFrame(data)[OUTPUT_COLUMNS]


def test_merge_fallback_recovered_mask():
    frames = np.array([1, 2, 3, 4])
    df_c3d = _fake_output_df(frames, {"HD": np.array([1.0, np.nan, np.nan, 4.0])})
    df_fb  = _fake_output_df(frames, {"HD": np.array([np.nan, 2.0, np.nan, 40.0])})

    merged, filled, unfillable, recovered = _merge_fallback(df_c3d, df_fb)

    # frame 2: C3D NaN, CSV has 2.0 -> recovered
    assert merged.loc[1, "HD_X"] == 2.0
    # frame 3: both NaN -> stays NaN, "unfillable"
    assert np.isnan(merged.loc[2, "HD_X"])
    # frame 1/4: C3D already had a value -> C3D wins, CSV's 40.0 at frame 4 ignored
    assert merged.loc[0, "HD_X"] == 1.0
    assert merged.loc[3, "HD_X"] == 4.0

    assert filled.get("HD") == 1
    assert unfillable.get("HD") == 1
    assert list(recovered["HD"]) == [False, True, False, False]


def test_validate_fallback_csv_rejects_rate_mismatch():
    frames = np.array([1, 2, 3])
    df_fb = _fake_output_df(frames, {"PELCEN": np.full(3, 900.0)})
    try:
        _validate_fallback_csv(60, 100.0, frames, frames, df_fb)
        raise AssertionError("expected ValueError for sample-rate mismatch")
    except ValueError as e:
        assert "sample rate" in str(e)


def test_validate_fallback_csv_rejects_zero_overlap():
    fb_frames  = np.array([1000, 1001, 1002])
    c3d_frames = np.array([1, 2, 3])
    df_fb = _fake_output_df(fb_frames, {"PELCEN": np.full(3, 900.0)})
    try:
        _validate_fallback_csv(100, 100.0, fb_frames, c3d_frames, df_fb)
        raise AssertionError("expected ValueError for zero frame overlap")
    except ValueError as e:
        assert "overlap" in str(e)


def test_validate_fallback_csv_warns_on_bad_units_and_filename():
    frames = np.array([1, 2, 3])
    # Z values ~0.9 instead of ~900mm -> looks like metres, not millimetres.
    df_fb = _fake_output_df(frames, {"PELCEN": np.full(3, 0.9)})
    warnings = _validate_fallback_csv(
        100, 100.0, frames, frames, df_fb,
        task_name="SUB04_lifting_T01", fallback_filename="SUB09_kneeling_T02.csv",
    )
    joined = " ".join(warnings)
    assert "millimetres" in joined
    assert "identifying tokens" in joined

    # Matching filename tokens -> no identity warning.
    df_fb_ok = _fake_output_df(frames, {"PELCEN": np.full(3, 900.0)})
    warnings_ok = _validate_fallback_csv(
        100, 100.0, frames, frames, df_fb_ok,
        task_name="SUB04_lifting_T01", fallback_filename="SUB04_lifting_T01_raw.csv",
    )
    assert warnings_ok == []


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
