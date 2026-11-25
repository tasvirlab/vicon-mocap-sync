# Within-Task High-Risk Posture-Duration Assessment Layer

**Status:** IMPLEMENTED; NOT VALIDATED. Every quantity below is computed from a
Vicon-derived reference. No performance claim, agreement statistic, or
validated-threshold claim is made in this document.

Implementation: `backend/core/vicon_reba/posture_duration.py` (layer),
`channels.py` (orchestration), `channel_figures.py` (figures).
Regression tests: `python -m core.vicon_reba.posture_duration` (12 checks).

---

## 1. Purpose and scope

The layer converts a continuous, filtered monocular- or Vicon-derived 3D pose
sequence into stable temporal events, and quantifies **when** and **for how
long** a body segment occupies a high-risk posture. It applies to **both dynamic
and static** construction tasks.

It is deliberately **not** named "static-holding duration analysis": lifting,
handling, carrying and twisting contain dynamic movement, and a period above an
angular threshold is not by itself a sustained static load. Static-posture
evaluation is a **conditional secondary branch** (§4).

The outputs are **temporal exposure descriptors**. They are not a new ergonomic
risk score, and they are not combined with REBA severity or trunk dynamics into
a composite index.

---

## 2. Notation and primary outputs

For body segment `s`, let `θ̃_s(t)` be the filtered signed anatomical angle
series (pelvis-centred, `atan2`-based; see `angles.py`), sampled at `f_s` Hz over
an observed task interval of duration `T_trial`.

### 2.1 High-risk state (dual threshold)

A high-risk state is defined by a **pre-specified ergonomic threshold derived
from an established instrument** — the REBA body-segment category boundary — not
by a value chosen after inspecting results:

| Segment | Entry `θ_on` | REBA basis | Exit `θ_off` |
|---|---|---|---|
| Trunk | 20° | `trunk_base` > 20° → score 3 (moderate flexion) | 15° |
| Neck | 20° | `neck_base` > 20° → score 2 (top category) | 15° |
| Upper arm | 45° | `upper_arm_base` > 45° → score 3 (45–90° elevation) | 40° |

Per-segment tuning rationale: the entry threshold is the angle at which that
segment's REBA base score first reaches its *substantially awkward* category —
score ≥ 3 where the scale has four levels (trunk, upper arm), or the top category
(2) where the scale has only two (neck). Boundaries are read from the same
`reba.py` tables the severity channel uses, so the duration layer and the
severity layer cannot drift apart.

The **exit threshold** sits a fixed band (`HYSTERESIS_BAND_DEG = 5°`) below entry,
forming a Schmitt trigger:

```
state(t) = 1  if  |θ̃_s(t)| ≥ θ_on          (from the low state)
state(t) = 1  if  |θ̃_s(t)| ≥ θ_off         (from the high state)
state(t) = 0  otherwise;  NaN (occluded) is always 0 and never bridges a bout
```

The band is an **algorithm parameter**, pre-specified before analysis and subject
to sensitivity analysis. It is not an ergonomic threshold and is not claimed to
be validated.

*Measured effect (synthetic, 60 Hz, signal oscillating ±1.5° about the 20°
boundary at 3 Hz):* a single threshold produces **23 state transitions**; the
dual threshold produces **0**.

### 2.2 Temporal gating and bouts

Consecutive-frame gating is applied on top of the angle hysteresis: a bout opens
only after `d_on` seconds of continuous high state and closes only after `d_off`
seconds of continuous low state (defaults `d_on = d_off = 0.5 s`). Onset and
offset are **back-dated** to the first and final qualifying frames, so the gating
debounces without shortening the reported event.

Each detected event:

```
B_{s,i} = [ t^on_{s,i} , t^off_{s,i} ]
H_{s,i} = t^off_{s,i} − t^on_{s,i}
```

### 2.3 Reported quantities

```
n_s          = number of bouts
H_{s,i}      = individual bout duration
max_i H_{s,i}= longest bout
T^high_s     = Σ_i H_{s,i}                     (total high-risk time)
P^high_s     = 100 · T^high_s / T_trial        (% of observed task time)
```

A dynamic transition that crosses the posture threshold **is** a legitimate bout.
It is **not** thereby a sustained static load — that is decided in §4.

### 2.4 Exposure partition — hazardous vs acceptable vs unmeasured

The observed interval is partitioned **three ways**, per segment:

```
T_observed = T_hazardous + T_acceptable + T_unmeasured
T_hazardous  = Σ_i H_{s,i}                      (time inside an above-threshold bout)
T_unmeasured = frames with a non-finite angle   (occlusion / dropped marker)
T_acceptable = T_measured − T_hazardous
```

> ⚠️ **Two-way partition would be wrong.** Computing
> `acceptable = observed − hazardous` folds occluded frames into acceptable
> posture time, asserting a safe posture that was never observed.
> Frames lost to occlusion would be banked that way.
> Occlusion is its own category and is hatched separately in the figure.

Percentages are reported against **both** denominators, because they differ
whenever occlusion exists and the choice changes the number:

- `percent_hazardous_of_observed` — includes occluded time in the denominator
- `percent_hazardous_of_measured` — excludes it (report this one for exposure)

**Bout time ≥ raw frame-wise time.** Hysteresis and `d_off` deliberately hold a
bout open across a brief dip below threshold, so `T_hazardous` exceeds a simple
frame count of above-threshold samples. Both are reported
(`above_threshold_frame_s`, `bout_minus_framewise_s`) so the difference is
visible rather than arguable.

---

## 3. What this layer does not do

- It does not extrapolate a short trial to an occupational exposure. Report
  *"four repetitions were detected within the observed 19.2-second interval"*.
  A normalised within-trial rate may be reported **descriptively only**, and
  never presented as the worker's actual hourly work rate.
- It does not create "low / medium / high" duration categories. No such
  categories exist in this layer, because none are available from a validated
  standard for these quantities.
- It does not treat every above-threshold period as static loading.

---

## 4. Conditional static-posture branch

A bout is classified as a **static high-risk hold** only when all criteria hold:

```
S_{s,i} = 1 [ H_{s,i} > 4 s
              ∧ median |ω_s(t)| < ε_ω,s
              ∧ SD(θ_s(t))      < ε_θ,s ]        for t ∈ B_{s,i}
```

with a fifth criterion evaluated by the caller, which holds the task metadata:

5. the activity falls within the applicability of the selected static-posture
   instrument — operationalised as **no or minimal external force**, tested
   against the *entered* REBA load/force score (`load ≤ 0`). Never inferred from
   pose.

`ω_s` is taken from the **same kinematics path** the trunk-dynamics channel uses
(`cutoff_hz = None` on the Vicon path, since markers are already 6 Hz zero-phase
filtered upstream), so a "static" verdict cannot disagree with the dynamics
channel.

### 4.1 Derivation of ε_ω and ε_θ — REQUIRED BEFORE USE

`S_{s,i}` returns **`None` ("not evaluated")** whenever the thresholds have not
been derived and frozen. The layer never substitutes a default. Until they are derived, no bout is reported as a static hold.

The derivation procedure (`estimate_static_thresholds()`):

1. estimated from the **static standing/calibration trial** recorded for each
   participant (separate from the eight task trials), and controlled holding
   trials where available;
2. computed **only** from participants in the training/validation pool — a
   test-pool subject passed to this function is a leakage error the function
   cannot detect and the caller must prevent;
3. taken as the *p*-th percentile (default *p* = 95, pre-specified) of the
   per-trial `median |ω|` and `SD(θ)` distributions;
4. **frozen to disk** (`outputs/_calibration/static_thresholds.json`) before any
   test-set evaluation, with `n_trials` and `n_frames` provenance recorded;
5. varied in a sensitivity analysis (percentile and band) and reported
   transparently.

Run it with:

```
python -m tools.derive_static_thresholds --train SUB01,SUB02,SUB03
```

`--train` is mandatory and has no default. Add `--dry-run` to inspect before
writing; the script refuses to overwrite an existing frozen file without
`--force`, because re-freezing after test evaluation invalidates the split.

### 4.1.1 Screening the calibration trials

A calibration trial is usable only if the person actually held still. Inspect each trial's per-channel
median `|ω|` and `SD(θ)` before freezing: a trial in which the person moved inflates the threshold and
makes the static test too permissive. Use only trials that pass this check.

### 4.1.2 Neutral-reference correction

Some channels (notably `neck_flex_deg`) do not read 0° in a person's neutral standing posture, because the
marker-based segment axes do not coincide with the anatomical ones. Angles are therefore expressed relative to
each subject's own measured neutral pose (`core/vicon_reba/neutral_offset.py`), derived from a still static
capture and frozen to `outputs/_calibration/neutral_offsets.json` by the same command that derives ε.

- The offsets are subtracted so that neutral reads 0°.
- Arm channels are deliberately **not** offset (listed in `EXCLUDED_CHANNELS`): the static pose holds the arms
  away from the body, so that reading is the pose, not a bias. A regression test asserts that an arm offset is refused.
- When no offset exists for a subject, angles stay **raw**, the response warns, and the figure caption says so.
  Another subject's neutral pose is never substituted.

### 4.2 Holding-time ratio (ISO 11226, conditional)

For a bout that is static **and** within the minimal-external-force scope:

```
R^hold_{s,i} = H_{s,i} / L_s(θ̄_{s,i})
```

`L_s` is the instrument-based acceptable holding-time limit for that segment and
mean posture angle. Interpretation is binary, with **no invented gradations**:

- `R^hold ≤ 1` — selected holding-time limit not exceeded
- `R^hold > 1` — selected holding-time limit exceeded

`R^hold` is withheld (returned `None`) with an explicit reason in every other
case: `not_static`, `static_status_not_evaluated`, `external_force_unknown`,
`external_force_exceeds_iso_scope`, `no_duration_criterion` (below the lower
anchor), or `angle_not_recommended` (beyond the upper anchor).

> **The angle verdict and the duration verdict are separate.** ISO rejects a
> posture beyond its upper anchor *regardless of duration*; that is an **angle**
> verdict and is reported as `angle_not_recommended`, never as a holding-time
> exceedance. Conflating the two previously caused a 2.9 s bout to be reported as
> exceeding a limit of several hundred seconds.

> **ISO status:** the holding-time anchors in `_ISO_CURVES` remain
> **unverified** against ISO 11226:2000 (`iso_11226_values_verified = False`).
> ISO output must not be reported as validated until the equations, anchors,
> units, and scope are checked against the standard.

---

## 4a. Cycle segmentation (Channel 3 companion)

`repetition.segment_cycles()` resolves each detected peak into
**onset → peak → completion**, so duty cycle and recovery become derivable.

For peak *i*, the crossing level is taken from the **local valley on each side**:

```
L_left  = valley_left  + f · (θ_peak − valley_left)
L_right = valley_right + f · (θ_peak − valley_right)          f = 0.10
```

onset = last sample before the peak still below `L_left`; completion = first
sample after the peak below `L_right`; both searches bounded by the neighbouring
peaks so one cycle's onset cannot land inside the previous cycle.

> **Why per-side local valleys, not the prominence base.** For the tallest peak
> in a trial the prominence base runs to the global minimum. This
> can put the level for the biggest cycle far below the valleys that bracket it, leaving that cycle with
> **both** boundaries unresolved. Per-side local valleys segment every cycle.

> **Unresolved boundaries are reported as `None`, never substituted.** Falling
> back to the search bound looked tidier but manufactured numbers: it placed
> cycle 4's onset exactly on cycle 3's peak, giving a 3.57 s "rise time" no
> movement made and a −0.90 s "recovery" that read as overlapping cycles.

Per-cycle outputs: `onset_s`, `peak_s`, `completion_s`, `rise_time_s`,
`return_time_s`, `cycle_duration_s`, `recovery_to_next_s`, `onset_resolved`,
`completion_resolved`. Trial-level: mean cycle duration, mean rise/return, mean
and minimum recovery, total cycle time, and `duty_cycle_percent` — **a
within-trial descriptor, not a shift-level duty cycle.**

`f = 0.10` is a pre-specified algorithm parameter (the conventional
movement-onset fraction), not a validated ergonomic threshold. Vary it in the
sensitivity analysis and report it.

### Task-aware cycle naming and plane

`cycle_kind(task_id)` returns `lift` only for material-handling tasks and
`movement_cycle` otherwise; a non-lift task returns `lift_rate_per_hr = None` so
its cadence **cannot reach the Marras model**, which was built from lifting data.
`cycle_channel(task_id)` counts Task 8 (trunk flexion + axial twisting) on the
transverse plane, everything else on the sagittal. When another plane's excursion
exceeds the counted one by >1.5×, a `plane_warning` is reported — never a silent
auto-switch, which would change counts between runs.

## 5. Task-applicability

Static classification is decided **per bout for every task** — there is no a
priori dynamic/static task class. See `task_applicability.md` for the table of
which outputs are valid for which task.

---

## 6. Reporting template

> For each segment we report the number of above-threshold posture bouts, each
> bout's onset, offset and duration, the longest bout, the total high-risk time
> `T^high_s`, and the percentage of observed task time `P^high_s`. Bouts were
> detected with entry/exit angle thresholds derived from REBA segment categories
> (trunk 20°/15°, neck 20°/15°, upper arm 45°/40°) and `d_on = d_off = 0.5 s`
> consecutive-frame gating, with onset and offset back-dated to the first and
> final qualifying frames. Bouts include the rising and falling phases of a
> movement and are therefore not static-posture durations; a bout was additionally
> classified as a static hold only when it exceeded 4 s with median angular
> velocity and angular variability below thresholds derived from static
> calibration trials of training participants only. [State ε values and the
> sensitivity analysis here once derived.] ISO 11226 holding-time comparison was
> applied only to static bouts under minimal external force.
