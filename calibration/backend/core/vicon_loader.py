"""
vicon_loader.py
===============
Load Vicon trajectory data from CSV or C3D files.
Returns {marker_name: np.ndarray (N_frames, 3)} in mm.
Auto-detects format by file extension.
"""

from __future__ import annotations
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CSV reader (Nexus export format)
# ─────────────────────────────────────────────────────────────────────────────

class ViconCSVReader:

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Vicon CSV not found: {filepath}")
        self._markers: Dict[str, np.ndarray] = {}
        self._frame_rate: float = 100.0
        self._n_frames: int = 0

    def parse(self) -> Dict[str, np.ndarray]:
        with open(self.filepath, newline="", encoding="utf-8-sig") as f:
            raw_lines = f.readlines()

        name_row_idx, _, data_start_idx = self._find_structure(raw_lines)
        self._frame_rate = self._extract_frame_rate(raw_lines, name_row_idx)
        marker_names = self._parse_marker_names(raw_lines[name_row_idx])
        self._markers = self._parse_data(marker_names, raw_lines[data_start_idx:])
        self._n_frames = next(iter(self._markers.values())).shape[0] if self._markers else 0
        return self._markers

    def get_frame(self, frame_idx: int) -> Dict[str, np.ndarray]:
        return {
            name: arr[frame_idx]
            for name, arr in self._markers.items()
            if frame_idx < len(arr)
        }

    @property
    def marker_names(self) -> List[str]:
        return list(self._markers.keys())

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def frame_rate(self) -> float:
        return self._frame_rate

    # ------------------------------------------------------------------
    def _find_structure(self, lines):
        """Locate the Trajectories section in a Vicon Nexus CSV export.

        Nexus exports often contain a Devices (force plate) section first,
        then a Trajectories section.  We must skip to Trajectories so we
        read marker XYZ data, not force-plate data.

        Expected layout after the "Trajectories" keyword:
          +0  "Trajectories"
          +1  frame-rate  (e.g. "100")
          +2  marker-names row  (e.g. ",,Test1:LSHO,,,...")
          +3  "Frame,Sub Frame,X,Y,Z,..."
          +4  units row  (e.g. ",,mm,mm,mm,...")
          +5  first data row
        """
        for i, line in enumerate(lines):
            if line.strip().upper().startswith("TRAJECTORIES"):
                name_row  = i + 2
                data_start = i + 5
                return name_row, name_row + 1, data_start

        # Fallback: look for a row that looks like a marker-names header
        for i, line in enumerate(lines):
            s = line.strip().upper()
            if "FRAME" in s and "SUB" in s:
                return i - 1, i, i + 2
            if 0 < i < 15:
                parts = [p.strip() for p in line.split(",") if p.strip()]
                if len(parts) > 3 and not all(self._is_numeric(p) for p in parts):
                    return i, i + 1, i + 2

        return 0, 1, 2

    def _extract_frame_rate(self, lines, name_row_idx) -> float:
        """Read the frame-rate number that immediately precedes the marker-names row."""
        for line in reversed(lines[max(0, name_row_idx - 3): name_row_idx]):
            try:
                val = float(line.strip().split(",")[0])
                if 10.0 <= val <= 2000.0:
                    return val
            except ValueError:
                pass
        return 100.0

    def _parse_marker_names(self, name_line: str) -> List[Optional[str]]:
        """Extract marker names from the Nexus header row.

        Nexus encodes names as  "Subject:MarkerName"  (e.g. "Test1:LEFT 11").
        We want the part AFTER the colon as the marker name.
        Empty cells and frame/subframe labels are returned as None and then
        forward-filled so every column carries its parent marker name.
        """
        parts = [p.strip() for p in name_line.split(",")]
        result = []
        for p in parts:
            if not p or p.upper() in ("FRAME", "SUB FRAME", "SUBFRAME"):
                result.append(None)
            else:
                # "Test1:LEFT 11" -> "LEFT 11";  bare name -> keep as-is
                name = p.split(":", 1)[-1].strip() if ":" in p else p.strip()
                result.append(name or None)
        # Forward-fill: empty X/Y/Z slots inherit the preceding marker name
        filled, last = [], None
        for name in result:
            if name is not None:
                last = name
            filled.append(last)
        return filled

    def _parse_data(self, col_names: List[Optional[str]],
                    data_lines: List[str]) -> Dict[str, np.ndarray]:
        groups: Dict[str, List[int]] = {}
        for idx, name in enumerate(col_names):
            if name:
                groups.setdefault(name, []).append(idx)

        rows = []
        for line in data_lines:
            if not line.strip():
                continue
            parts = line.split(",")
            rows.append([self._to_float(p) for p in parts])

        if not rows:
            return {}

        # Pad to uniform width — Vicon CSVs sometimes have trailing commas
        # or missing columns on gap-filled rows, causing inhomogeneous shape.
        max_cols = max(len(r) for r in rows)
        for r in rows:
            if len(r) < max_cols:
                r.extend([np.nan] * (max_cols - len(r)))

        arr = np.array(rows, dtype=np.float64)
        markers = {}
        for name, cols in groups.items():
            xyz_cols = cols[:3]
            if all(c < arr.shape[1] for c in xyz_cols):
                markers[name] = arr[:, xyz_cols]
        return markers

    @staticmethod
    def _is_numeric(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    @staticmethod
    def _to_float(s: str) -> float:
        try:
            return float(s.strip())
        except ValueError:
            return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# C3D reader (via ezc3d)
# ─────────────────────────────────────────────────────────────────────────────

class ViconC3DReader:

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"C3D file not found: {filepath}")
        self._markers: Dict[str, np.ndarray] = {}
        self._frame_rate: float = 100.0
        self._n_frames: int = 0

    def parse(self) -> Dict[str, np.ndarray]:
        try:
            import ezc3d
        except ImportError:
            raise ImportError(
                "ezc3d is required for C3D support. "
                "Install: pip install ezc3d"
            )
        c = ezc3d.c3d(str(self.filepath))
        self._frame_rate = float(c["header"]["points"]["frame_rate"])
        labels: List[str] = c["parameters"]["POINT"]["LABELS"]["value"]
        points = c["data"]["points"]   # (4, M, F)
        self._n_frames = points.shape[2]

        self._markers = {}
        for m_idx, label in enumerate(labels):
            name = label.strip()
            if not name:
                continue
            xyz = points[:3, m_idx, :].T.copy()
            occluded = points[3, m_idx, :] == 0
            xyz[occluded] = np.nan
            self._markers[name] = xyz

        return self._markers

    def get_frame(self, frame_idx: int) -> Dict[str, np.ndarray]:
        return {
            name: arr[frame_idx]
            for name, arr in self._markers.items()
            if frame_idx < len(arr)
        }

    @property
    def marker_names(self) -> List[str]:
        return list(self._markers.keys())

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def frame_rate(self) -> float:
        return self._frame_rate


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def load_vicon_file(filepath: str):
    """Auto-detect format and return the appropriate reader (unparsed)."""
    ext = Path(filepath).suffix.lower()
    if ext == ".c3d":
        return ViconC3DReader(filepath)
    elif ext in (".csv", ".txt"):
        return ViconCSVReader(filepath)
    else:
        raise ValueError(f"Unsupported Vicon file format: {ext!r}")


def load_vicon_markers(filepath: str) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Parse a Vicon file and return (markers_dict, frame_rate).
    markers_dict: {marker_name: (N_frames, 3) array in mm}
    """
    reader = load_vicon_file(filepath)
    markers = reader.parse()
    logger.info(
        f"Loaded {len(markers)} markers × {reader.n_frames} frames "
        f"@ {reader.frame_rate} Hz  from {Path(filepath).name}"
    )
    return markers, reader.frame_rate
