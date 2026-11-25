# Engineering notes

Conventions that the numerical results depend on. Change them only together with the tests that lock them.

## Architecture

- `frontend/` is one Electron window. Its main process starts two FastAPI services from each conda environment's own `python.exe` (not `conda run`, which races when two runs start together and splits arguments at spaces).
- Calibration service: port 8765. Sync pipeline service: port 8766. Both bind to `127.0.0.1` only.
- Each frontend module (`calib/`, `sync/`) has its own API client and `components/ui`; the design tokens, theme and shell are shared.
- Output folders are configurable (Settings → Folders) and persisted in `calibration/config.json` and `sync/config.json`.
- Both services answer only requests from the app's own pages (`localhost` or `127.0.0.1` on ports 5173 and 5174) and only for loopback host names. To allow another origin, for example a packaged build, set `VICONSYNC_ALLOWED_ORIGINS` (comma separated) before starting the backends.
- `VICONSYNC_RAW_VICON_DIR` (optional) points at a folder of raw Vicon exports, one subfolder per subject, so Sync can auto-match the XCP file and C3D Convert the cleaned CSV next to the original export.

## Coordinates and units

| Source | Convention |
|---|---|
| Vicon | right-handed, Z up, millimetres |
| 2D keypoints | image pixels, origin top-left; normalized per frame by root-centering on the pelvis and dividing by half the image width |
| 3D training target | metres, root-centred on the pelvis, no fitted scaler |

The canonical 21-joint order has the pelvis at index 3: `HD, C7, T10, PELCEN, LSHO, RSHO, LELB, RELB, LWR, RWR, LHAND, RHAND, LHIP, RHIP, LKNE, RKNE, LANK, RANK, LTOE, RTOE, L45`.

## Synchronization

1. The XCP file gives the Vicon capture start time (local clock); the video container gives its creation time (UTC).
2. `sync_offset = video_UTC - vicon_UTC`. The camera timestamp is quantised to whole seconds, so this alone is only good to about ±1 s.
3. Optional refinement correlates video optical flow (restricted to the subject's projected bounding box) with marker velocity and, when calibration is supplied, projected Vicon motion. The signal with the highest Pearson `r` wins; the correction is applied only when `r >= 0.30`.
4. `needs_review` is set when the signals disagree by more than 2 frames, when `r < 0.55`, or when the peak is near the search boundary. Dataset export blocks any trial whose review flag is missing or true unless `manual_review_approved: true` is set in that trial's `metadata.json`.
5. The ground-reaction-force derivative is plotted for diagnosis but never votes: force follows acceleration, optical flow follows speed, so they are about 90° out of phase.

A static pose looks aligned under any offset. Verify the overlay during the fast phase of the movement.

## C3D conversion

- Output columns per joint: coordinates plus `{joint}_Source` with values `c3d`, `interpolated`, `recovered_csv` or `invalid`.
- The CSV is trimmed to the first and last frame where every consistently tracked joint is valid. A joint present in fewer than 5 % of frames is treated as never placed.
- `time_s` is not re-zeroed at the trim boundary; Sync and Overlay rely on it being anchored to the capture start.
- Zero-phase Butterworth filtering (6 Hz) happens once, at the native 100 Hz, before the 100 Hz to 60 Hz resample.
- A fallback Nexus CSV is accepted only if its rate matches the C3D and the frame ranges overlap.

## Angles and kinematics

- Trunk sagittal, lateral and twist angles and the neck flexion angle are **signed**. Take `abs()` only at the point of use (REBA thresholds). Differentiating a magnitude manufactures acceleration spikes at neutral crossings.
- Trunk lateral angle is `atan2(t_right, sqrt(t_up² + t_forward²))`; the naive `atan2(|t_right|, t_up)` reads deep flexion as a lateral bend.
- Angles are measured in a per-frame pelvis frame (`right = RHIP - LHIP`, `up = world Z`, `forward = up × right`), so they do not depend on facing direction.
- `kinematics` owns all derivatives. Vicon-derived signals use `cutoff_hz=None` (already filtered); model-derived signals use `cutoff_hz=6.0` so both are band-limited identically.
- Derivative boundary samples are one-sided and are excluded from peak marking.
- Upper-arm elevation is trunk-relative, so it tracks trunk flexion during lifting; do not read an arm bout as independent evidence there.

## REBA channels

The REBA stage writes three separate outputs and never fuses them: posture severity (`reba_scores.csv`), posture duration (`posture_duration.json`) and trunk dynamics (`trunk_dynamics.csv/json`).

- Load, coupling, activity and unilateral flags are entered task metadata, never inferred from pose.
- A posture duration "bout" includes the rising and falling phases of a movement; it is not a static-hold duration unless flagged `is_static_hold`.
- ISO 11226 anchors are provisional (`iso_11226_values_verified: false`). Angle verdict (`angle_status`) and duration verdict (`duration_status`) are independent fields.
- The Marras probability is a high-risk-group probability for eligible repetitive material-handling tasks, never an individual injury probability.

## 2D pose

- Keypoints map to the 21 joints; `HD` is the mean of eyes and ears (not the nose), hands use the middle-finger MCP, feet the small toe.
- A joint below the confidence threshold (default 0.3) is NaN and invalidates that frame for window QC.
- Subject selection is guided by the projected Vicon pelvis; frames with no detection within 250 px emit NaN rather than a bystander.
- `T10` and `L45` are interpolated from `C7` and the pelvis centre, so they carry no independent evidence.

## Dataset export

- Windows are 243 frames with an 81 frame step; a window needs at least 90 % valid 2D frames and no interior gap over 30 frames. The 3D target allows gaps of at most 5 frames.
- Splits are by subject (train, validation, test), never by frame. Any fitted parameter (the spine ratio) is fitted on training subjects only and then frozen.
- Provenance arrays (`confidence`, `visibility_mask`, `interpolation_mask`, `target_3d_source`) travel with the windows.

## Output layout

One folder per trial: `outputs/<subject>/<NN_task>/<subject>_<task>_T<NN>/` with `metadata.json`, `sync.json`, `synced_60hz.csv`, `synced_vicon2d_60hz.csv`, `frame_mapping.csv`, `rgb.mp4`, `pose2d_rtmpose.csv`, `reba/` and `qa/`. A folder without `metadata.json` is invisible to discovery. Build paths with `utils/task_paths.py`; never hardcode file names.
