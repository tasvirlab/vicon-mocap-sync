import { reactive, watch } from 'vue'

const STORAGE_KEY = 'viconcalib_settings_v2'

export const SETTINGS_DEFAULTS = {
  // Not a calibration default — the subject currently being recorded. Kept
  // here (rather than a separate store) purely so it persists across pages
  // and reloads via the same localStorage write; resetToDefaults() below
  // deliberately does not touch it.
  subject_id: '',
  board: {
    dictionary:       'DICT_6X6_250',
    cols:             7,
    rows:             10,
    square_length_mm: 40.0,
    marker_length_mm: 26.0,
    min_corners:      8,
  },
  extraction: {
    sample_interval_frames: 5,
    blur_threshold:         80.0,
    target_accepted_frames: 50,
    min_accepted_frames:    15,
    rotate:                 'ccw90',   // default for iPhone portrait MOV
  },
  calibration: {
    rms_accept_threshold: 0.5,
    rms_warn_threshold:   1.0,
  },
  solver: {
    use_ransac:                       true,
    ransac_reprojection_threshold_px: 8.0,
    ransac_confidence:                0.999,
    refine_lm:                        true,
  },
}

export type AppSettings = typeof SETTINGS_DEFAULTS

function loadSettings(): AppSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) {
      const p = JSON.parse(raw)
      return {
        subject_id:  p.subject_id ?? SETTINGS_DEFAULTS.subject_id,
        board:       { ...SETTINGS_DEFAULTS.board,       ...(p.board       ?? {}) },
        extraction:  { ...SETTINGS_DEFAULTS.extraction,  ...(p.extraction  ?? {}) },
        calibration: { ...SETTINGS_DEFAULTS.calibration, ...(p.calibration ?? {}) },
        solver:      { ...SETTINGS_DEFAULTS.solver,      ...(p.solver      ?? {}) },
      }
    }
  } catch { /* ignore corrupt data */ }
  return JSON.parse(JSON.stringify(SETTINGS_DEFAULTS))
}

// Module-level reactive — shared across all consumers without duplicate localStorage reads
const settings = reactive<AppSettings>(loadSettings())

// Any mutation from any page (Settings form, board params on Intrinsic, rotate
// buttons on Extrinsic, etc.) persists automatically — no page has to remember
// to call save()/update() itself.
watch(settings, (s) => localStorage.setItem(STORAGE_KEY, JSON.stringify(s)), { deep: true })

export function useSettings() {
  function save(cfg: AppSettings) {
    Object.assign(settings.board,       cfg.board)
    Object.assign(settings.extraction,  cfg.extraction)
    Object.assign(settings.calibration, cfg.calibration)
    Object.assign(settings.solver,      cfg.solver)
  }

  function resetToDefaults() {
    const d = JSON.parse(JSON.stringify(SETTINGS_DEFAULTS)) as AppSettings
    Object.assign(settings.board,       d.board)
    Object.assign(settings.extraction,  d.extraction)
    Object.assign(settings.calibration, d.calibration)
    Object.assign(settings.solver,      d.solver)
  }

  function update(path: string[], val: unknown) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let obj: any = settings
    for (let i = 0; i < path.length - 1; i++) obj = obj[path[i]]
    obj[path[path.length - 1]] = val
  }

  return { settings, save, resetToDefaults, update }
}
