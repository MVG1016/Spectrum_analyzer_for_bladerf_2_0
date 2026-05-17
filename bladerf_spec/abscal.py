"""Absolute-power dBm offset table.

Frequency → dBm offset, linearly interpolated. Loaded once from
absolute_calibration.csv at the project root. Falls back to a flat -77 dB
offset (the historical hardcoded value) if the file is missing or malformed,
so behaviour is preserved on a fresh checkout.
"""
import os
import sys

import numpy as np

ABS_CAL_FILENAME = "absolute_calibration.csv"

_ABS_CAL_LOADED = False
_ABS_CAL_FREQS = np.array([400e6, 800e6, 1500e6, 2400e6, 3000e6, 5000e6])
_ABS_CAL_OFFSETS = np.array([-77.0, -77.0, -77.0, -77.0, -77.0, -77.0])


def _abs_cal_default_path() -> str:
    """Return the path to absolute_calibration.csv at the project root.

    The file lives next to main.py (one level up from this package).
    """
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, ABS_CAL_FILENAME)


def _load_absolute_calibration():
    global _ABS_CAL_LOADED, _ABS_CAL_FREQS, _ABS_CAL_OFFSETS
    if _ABS_CAL_LOADED:
        return
    _ABS_CAL_LOADED = True
    path = _abs_cal_default_path()
    if not os.path.exists(path):
        print(f"Note: {path} not found — using built-in flat -77 dB offset")
        return
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        if data.shape[1] < 2 or len(data) == 0:
            raise ValueError("expected columns: freq_hz, offset_db")
        order = np.argsort(data[:, 0])
        _ABS_CAL_FREQS = data[order, 0]
        _ABS_CAL_OFFSETS = data[order, 1]
        print(f"Absolute calibration loaded: {path} "
              f"({len(_ABS_CAL_FREQS)} points, "
              f"{_ABS_CAL_FREQS.min()/1e6:.0f}–{_ABS_CAL_FREQS.max()/1e6:.0f} MHz)")
    except Exception as e:
        print(f"Warning: failed to parse {path}: {e}. Using flat -77 dB.")


def get_calibration(freq_hz: float) -> float:
    """Absolute-power dBm offset for the given RX frequency (interpolated)."""
    _load_absolute_calibration()
    return float(np.interp(freq_hz, _ABS_CAL_FREQS, _ABS_CAL_OFFSETS))


def update_absolute_calibration_csv(freq_hz: float, new_offset_db: float):
    """Insert (or replace) a row in absolute_calibration.csv and invalidate cache."""
    global _ABS_CAL_LOADED
    path = _abs_cal_default_path()

    rows: list = []
    if os.path.exists(path):
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
            rows = [[float(r[0]), float(r[1])] for r in data]
        except Exception as e:
            print(f"Warning: existing cal file unreadable ({e}); starting fresh")

    matched = False
    for r in rows:
        if abs(r[0] - freq_hz) < 1.0:
            r[1] = new_offset_db
            matched = True
            break
    if not matched:
        rows.append([float(freq_hz), float(new_offset_db)])

    rows.sort(key=lambda r: r[0])

    with open(path, "w") as f:
        f.write("freq_hz,offset_db\n")
        for fr, off in rows:
            f.write(f"{int(fr)},{off}\n")

    _ABS_CAL_LOADED = False  # force reload on next get_calibration()
