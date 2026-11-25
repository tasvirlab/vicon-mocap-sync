// ── Subject-first task ID composition — outputs/SUB01/<task>/T01/ ──────────────
// Mirrors backend/utils/task_paths.py build_task_id()/split_task_id() exactly.

export const SUBJECTS: string[] = Array.from(
  { length: 10 },
  (_, i) => `SUB${String(i + 1).padStart(2, '0')}`,
)

export function buildTaskId(subjectId: string, task: string, trial: string): string {
  const taskClean = task.trim().replace(/\s+/g, '_')
  return `${subjectId}_${taskClean}_${trial}`
}

/** Splits a composite task_id back into parts, or null if it doesn't match
 * the SUB<NN>_<task>_T<NN> pattern (e.g. a legacy/unassigned id). */
export function splitTaskId(taskId: string): { subjectId: string; task: string; trial: string } | null {
  const m = taskId.match(/^(SUB\d+)_(.+)_(T\d+)$/)
  if (!m) return null
  return { subjectId: m[1], task: m[2], trial: m[3] }
}

/** Normalizes a free-typed trial value to "T01" form (accepts "1", "01", "T1", "T01"). */
export function normalizeTrial(raw: string): string {
  const digits = raw.replace(/^[Tt]/, '').trim()
  const n = parseInt(digits, 10)
  if (Number.isNaN(n)) return raw.trim() || 'T01'
  return `T${String(n).padStart(2, '0')}`
}

/** "T01" -> "T02" — for advancing to the next trial of the same subject/task
 * without retyping it (the 8 tasks are each recorded 3x per subject). */
export function incrementTrial(trial: string): string {
  const n = parseInt(trial.replace(/^[Tt]/, ''), 10)
  return normalizeTrial(String((Number.isNaN(n) ? 0 : n) + 1))
}
