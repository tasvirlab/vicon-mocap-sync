"""Cross-consistency check across the three independent 21-joint name lists
in this codebase.

The lists intentionally use DIFFERENT internal orders (rtmpose_pipeline.py /
vicon_reba/load.py put PELCEN at index 12, matching the raw Vicon export
column order; dataset_export.py puts PELCEN at index 3, matching the
training loss_construction.py contract). That's safe today because every
cross-module read goes through name-keyed CSV column lookup, never raw
positional trust — but nothing previously caught it if the three lists ever
drifted to disagree on WHICH 21 names exist. This test is that catch.

No consolidation is done here — each list keeps its own order for its own
module. This only asserts the underlying joint SET agrees everywhere, and
that dataset_export's order specifically still matches the frozen training-loss
contract.

Run: python -m utils.test_joint_names
"""

from __future__ import annotations

from core import rtmpose_pipeline
from core import dataset_export
from core.vicon_reba import load as vicon_load

# The frozen training-loss loss_construction.py order (Pelvis = index 3) — see
# dataset_export.py's own module docstring/assert for the authoritative source.
LOSS_CONTRACT_ORDER = [
    "HD", "C7", "T10", "PELCEN", "LSHO", "RSHO", "LELB", "RELB", "LWR", "RWR",
    "LHAND", "RHAND", "LHIP", "RHIP", "LKNE", "RKNE", "LANK", "RANK",
    "LTOE", "RTOE", "L45",
]


def test_all_lists_have_21_unique_names():
    for label, names in [
        ("rtmpose_pipeline", rtmpose_pipeline.JOINT_NAMES),
        ("vicon_reba.load", vicon_load.JOINT_NAMES),
        ("dataset_export", dataset_export.JOINT_NAMES),
    ]:
        assert len(names) == 21, f"{label}: expected 21 joints, got {len(names)}"
        assert len(set(names)) == 21, f"{label}: contains duplicate joint names"


def test_all_lists_agree_on_the_same_joint_set():
    rtm_set  = set(rtmpose_pipeline.JOINT_NAMES)
    load_set = set(vicon_load.JOINT_NAMES)
    ds_set   = set(dataset_export.JOINT_NAMES)

    assert rtm_set == load_set, (
        f"rtmpose_pipeline vs vicon_reba.load joint sets diverge: "
        f"only in rtmpose={rtm_set - load_set}, only in load={load_set - rtm_set}"
    )
    assert rtm_set == ds_set, (
        f"rtmpose_pipeline vs dataset_export joint sets diverge: "
        f"only in rtmpose={rtm_set - ds_set}, only in dataset_export={ds_set - rtm_set}"
    )


def test_dataset_export_order_matches_loss_contract():
    assert dataset_export.JOINT_NAMES == LOSS_CONTRACT_ORDER, (
        "dataset_export.JOINT_NAMES no longer matches the frozen training-loss "
        "loss_construction.py order — this is the array order actually "
        "written to the training .npz; do not re-index without also "
        "updating the model's loss/eval code."
    )
    assert dataset_export.JOINT_NAMES.index("PELCEN") == 3, \
        "Pelvis must stay at index 3 (the model's root-centering index)"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
