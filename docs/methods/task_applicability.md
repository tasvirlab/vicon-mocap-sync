# Task-Applicability Table — which outputs are valid for which task

**Status:** IMPLEMENTED; NOT VALIDATED. Rows describe which quantities are
*methodologically applicable*, not which have been shown accurate.

Companion to `posture_duration_layer.md`. Enforced in code by
`posture_duration.segment_duration_bouts()` (per-bout gates) and
`channels.compute_trunk_dynamics()` (Marras eligibility).

---

## Design decision: no a priori task class

Static classification is decided **per bout, for every task**. There is no
dynamic/static task dichotomy applied in advance, because a task label is a weak
proxy for what the static-posture instrument actually requires: it both
over-blocks (a genuinely sustained hold occurring inside a "handling" trial) and
under-blocks (a loaded static hold in a task not on an exclusion list).

The gates that decide applicability are therefore properties of the **bout**, not
the task name:

| Gate | Test | Source |
|---|---|---|
| Above-threshold | `\|θ̃_s\| ≥ θ_on` with hysteresis + `d_on`/`d_off` | pose |
| Static hold `S` | `H > 4 s` ∧ `median\|ω\| < ε_ω` ∧ `SD(θ) < ε_θ` | pose + frozen ε |
| ISO scope | entered REBA load/force score ≤ 0 | **entered task metadata** |
| Marras eligible | repetitive material-handling task | task protocol |

---

## Output applicability by task

The eight locked tasks. "Per bout" means the output is produced whenever the
bout-level gate is satisfied, regardless of task.

| # | Task | Bout timing, duration, count, `T^high`, `P^high` | Repetition count + within-trial rate | Trunk angle / velocity / acceleration | Static-hold classification `S` | ISO `R^hold` | Marras probability |
|---:|---|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | Manual lifting | ✅ | ✅ | ✅ | per bout | per bout † | eligible ‡ |
| 2 | Overhead work | ✅ | ✅ desc. | ✅ | per bout | per bout † | ✗ |
| 3 | Forward trunk bending | ✅ | ✅ desc. | ✅ | per bout | per bout † | ✗ |
| 4 | Squatting | ✅ | ✅ desc. | ✅ | per bout | per bout † | ✗ |
| 5 | Kneeling | ✅ | ✅ desc. | ✅ | per bout | per bout † | ✗ |
| 6 | Handling | ✅ | ✅ | ✅ | per bout | per bout † | ✗ |
| 7 | Load carrying | ✅ | ✅ desc. | ✅ | per bout | per bout † | ✗ |
| 8 | Trunk flexion + axial twisting | ✅ | ✅ | ✅ | per bout | per bout † | ✗ |

**†** `R^hold` additionally requires the entered load/force score to be 0 (ISO's
no-or-minimal-external-force scope). A loaded task therefore yields
`external_force_exceeds_iso_scope` for every bout even when `S = 1`. This is why
the column is not a task-level ✗: the gate is real data, not a task label.

**‡** Marras eligibility is set by `MARRAS_ELIGIBLE_TASKS` (currently `lifting`).
The probability is only computed when all five predictors are available; the
maximum external-load moment is task metadata not derivable from pose, so the
model currently reports `applicable: false`.

**"desc."** — repetition count is reported **descriptively** for the observed
interval only. A within-trial rate may accompany it, but must never be presented
as the worker's actual hourly work rate.

---

## Quantities that are never task-conditional

| Output | Always reported | Note |
|---|---|---|
| Bout onset / offset / duration | ✅ | Primary temporal exposure descriptor |
| Number of bouts, longest bout | ✅ | |
| `T^high_s`, `P^high_s` | ✅ | `P^high` is relative to the observed interval |
| Entry/exit thresholds used | ✅ | REBA-derived; recorded per segment in the output |
| Signed trunk angle / velocity / acceleration | ✅ | Angular acceleration is a separate output, **not** a Marras predictor |

---

## Quantities deliberately absent

| Not produced | Why |
|---|---|
| "Low / medium / high" duration categories | No validated standard defines them for these quantities |
| Composite risk index across channels | The three channels are reported separately by design |
| Shift-level / cumulative exposure, TWA | Trials are ~10–20 s; extrapolation is not defensible |
| Individual injury probability | Marras output is a high-risk-**group** probability |
| Expert-validated REBA agreement | No independent expert scoring in this software |

---

## Interpretation caveat: segment non-independence

During trunk-flexion tasks the upper-arm angle is measured relative to the trunk
(the REBA definition), so passively hanging arms track trunk flexion by geometry.
As a result an arm bout can be long, and strongly correlated with the trunk, without the arms being
loaded. Raising the arm entry threshold to the REBA score-3 boundary (45°) reduces but does not eliminate
this. When reporting per-segment
bouts for trunk-flexion tasks, state that an arm bout is **partly a geometric
consequence of trunk flexion**, not independent evidence of arm exposure.
