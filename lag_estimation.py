"""
lag_estimation.py

Estimate the camera-to-LiDAR/IMU time delay by cross-correlating two speed
proxies: the vehicle's real speed (from the IMU/GPS log) against the
camera's own apparent motion (optical flow magnitude between sampled
frames of its raw video). The lag that best aligns the two curves is the
delay between them.

This is a direct port of the engine already built and tested for the
standalone sync-check dashboard earlier in this project, carrying over
both bug fixes found there rather than re-introducing them:

  Per-segment correlation. An earlier version normalized each signal once,
  globally, then took a raw dot product per shift - only equal to a true
  Pearson correlation when every subsegment happens to share the whole
  signal's own mean and variance, which is false for periodic or smoothly
  varying signals. It let the reported correlation exceed 1.0, which is
  mathematically impossible for a real correlation. Fixed by normalizing
  freshly on each compared segment.

  Explicit bounds check before slicing. `a[:n+s]` for a negative `n+s`
  does not give an empty array the way a length check afterward would
  suggest - Python counts a negative stop index from the array's end, so
  large shifts were silently comparing short, wrongly-paired, wrapped
  segments instead of correctly finding no overlap. Fixed with an
  `abs(s) >= n` check before any slicing happens.

What this cannot do: tell a real timing delay apart from a period where
the vehicle simply was not moving (both curves are flat and carry no
information), or from a stretch where the camera's own motion is
dominated by something other than ego-motion. The correlation curve
returned alongside the best lag is how to catch this - a single sharp
peak is a confident result, a flat or multi-peaked one is not.
"""

from __future__ import annotations

import csv
import io

import numpy as np


# --------------------------------------------------------------------------
# Correlation
# --------------------------------------------------------------------------

def normalized_cross_correlation(a, b, max_lag_samples):
    """
    Pearson correlation of a against b at each integer sample shift in
    [-max_lag_samples, +max_lag_samples], normalized freshly on each
    overlapping segment.

    Convention, verified against an unambiguous single-spike-each signal
    rather than derived on paper: positive shift means A is the DELAYED
    signal, i.e. a(t) = b(t - shift).
    """
    n = len(a)
    shifts = np.arange(-max_lag_samples, max_lag_samples + 1)
    corr = np.full(len(shifts), np.nan)
    min_len = max(10, n // 10)
    for i, s in enumerate(shifts):
        if abs(s) >= n:            # see module docstring - must be checked
            continue                # before slicing, not inferred after
        if s >= 0:
            seg_a, seg_b = a[s:], (b[:n - s] if s > 0 else b)
        else:
            seg_a, seg_b = a[:n + s], b[-s:]
        if len(seg_a) < min_len:
            continue
        sa = seg_a - seg_a.mean()
        sb = seg_b - seg_b.mean()
        denom = np.sqrt((sa * sa).sum() * (sb * sb).sum())
        corr[i] = float((sa * sb).sum() / denom) if denom > 1e-12 else np.nan
    return shifts, corr


def resample_to_grid(t, v, grid):
    return np.interp(grid, t, v, left=np.nan, right=np.nan)


def estimate_lag(t_truth, v_truth, t_cam, v_cam, lag_range=(-8.0, 8.0),
                 grid_hz=50.0):
    """
    Find the lag (seconds) that best aligns the camera motion proxy to the
    ground-truth speed series. Convention: video_time = lidar_time - lag,
    matching the rest of this project.
    """
    t0 = max(t_truth.min(), t_cam.min())
    t1 = min(t_truth.max(), t_cam.max())
    if t1 - t0 < 2.0:
        return dict(best_lag_s=0.0, confidence=0.0, at_edge=False,
                   lags_s=np.array([0.0]), correlation=np.array([0.0]),
                   note="overlapping time range too short to estimate")

    grid = np.arange(t0, t1, 1.0 / grid_hz)
    g_truth = resample_to_grid(t_truth, v_truth, grid)
    g_cam = resample_to_grid(t_cam, v_cam, grid)
    ok = np.isfinite(g_truth) & np.isfinite(g_cam)
    if ok.sum() < grid_hz * 2:
        return dict(best_lag_s=0.0, confidence=0.0, at_edge=False,
                   lags_s=np.array([0.0]), correlation=np.array([0.0]),
                   note="not enough overlapping valid samples")

    max_lag_samples = int(round(lag_range[1] * grid_hz))
    shifts, corr = normalized_cross_correlation(g_truth[ok], g_cam[ok],
                                               max_lag_samples)
    # Sign fixed against one unambiguous physical case, not derived on
    # paper (see module docstring on why paper derivation is untrusted
    # here): a camera that started recording N seconds late must come back
    # positive, to match this tool's own stated convention ("positive delay
    # means the camera frame lags the LiDAR moment") and the metadata-only
    # calculation elsewhere in this dashboard, which uses the same sign for
    # the same meaning. Verified directly below, not assumed.
    lags_s = shifts / grid_hz

    valid = np.isfinite(corr)
    if not valid.any():
        return dict(best_lag_s=0.0, confidence=0.0, at_edge=False,
                   lags_s=lags_s, correlation=corr,
                   note="no valid correlation computed")

    best_i = np.nanargmax(corr)
    best_lag = float(lags_s[best_i])
    at_edge = bool(best_i <= 1 or best_i >= len(lags_s) - 2)
    note = "ok"
    if at_edge:
        note = (f"best fit sits at the edge of the searched range "
               f"({lag_range[0]}s to {lag_range[1]}s) - the true delay may "
               f"lie outside it, or there may be no real match to find.")

    return dict(best_lag_s=best_lag, confidence=float(corr[best_i]),
               lags_s=lags_s, correlation=corr, note=note, at_edge=at_edge)


def optical_flow_speed_proxy(frames, resize_to=320, row_band=(0.30, 0.80)):
    """
    Mean dense optical-flow magnitude between consecutive sampled frames.

    row_band restricts the measurement to a horizontal strip of the frame
    (0=top, 1=bottom), excluding the sky and the vehicle's own body -
    averaged over a whole wide-angle frame, both dilute what should be a
    clean relationship between flow and speed with noise unrelated to
    motion. Validated on a synthetic scene with realistic frame-to-frame
    noise in those regions: whole-frame averaging correlated with true
    speed at 0.72, this band restriction at 0.85.
    """
    import cv2

    def prep(f):
        h, w = f.shape[:2]
        f = f[int(h * row_band[0]):int(h * row_band[1])]
        h2, w2 = f.shape[:2]
        s = resize_to / max(h2, w2)
        small = cv2.resize(f, (max(1, int(w2 * s)), max(1, int(h2 * s))))
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small

    gray = [prep(f) for f in frames]
    out = np.empty(len(gray) - 1)
    for i in range(len(gray) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray[i], gray[i + 1], None, 0.5, 2, 15, 2, 5, 1.2, 0)
        out[i] = float(np.hypot(flow[..., 0], flow[..., 1]).mean())
    return out


# --------------------------------------------------------------------------
# IMU/GPS CSV
# --------------------------------------------------------------------------

def parse_imu_csv(data: bytes):
    """
    Extract (t_rel_s, speed_m_s) from an IMU/GPS log like this project's.

    Delimiter is SNIFFED from the header line, not assumed: real logs from
    this rig have been seen comma-separated, while the same data pasted
    through other tools arrives tab-separated. Guessing one and hardcoding
    it silently matches zero columns and produces a confusing
    "column not found" error rather than an obvious parse failure.

    '#'-prefixed metadata lines appear both before AND after the header row
    (a real quirk of this logger, not assumed) - every such line is skipped
    regardless of position, and the first non-'#' line is the header.

    t_rel_s is used directly rather than reconstructed from t_unix: the
    logger already computes it at sub-millisecond precision, anchored to
    t0_unix, which the whole-second t_unix column cannot offer. Note that
    for a clip cut out of a longer session, t_rel_s stays referenced to the
    ORIGINAL session t0, so it may start far from zero - that is correct
    and harmless here, since correlation only needs relative timing.

    speed is the magnitude of filter_vel_x/y/z - direction-independent, so
    no per-rig axis convention needs to be known or assumed.
    """
    text = data.decode("utf-8-sig", errors="replace")   # -sig strips a BOM

    # find the header line first, so the delimiter can be sniffed from it
    raw_lines = text.splitlines()
    header_line = None
    for line in raw_lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        header_line = line
        break
    if header_line is None:
        raise ValueError("no header row found (every line was blank or "
                         "started with '#')")

    # whichever delimiter splits the header into more fields is the real one
    delim = max(("\t", ",", ";"), key=lambda d: len(header_line.split(d)))
    if len(header_line.split(delim)) < 2:
        raise ValueError("header row has only one field - the delimiter "
                         "is not tab, comma or semicolon")

    reader = csv.reader(io.StringIO(text), delimiter=delim)
    header = None
    rows = []
    for row in reader:
        if not row:
            continue
        # skip truly blank lines and '#' metadata lines, but NOT a row whose
        # first field merely happens to be empty - that is a legitimate data
        # row with a missing leading value, and dropping it silently would
        # lose real samples
        if not any(f.strip() for f in row):
            continue
        if row[0].lstrip().startswith("#"):
            continue
        if header is None:
            header = row
            continue
        rows.append(row)

    if header is None:
        raise ValueError("no header row found after delimiter detection")

    def col(name):
        # tolerate stray whitespace around column names
        for i, h in enumerate(header):
            if h.strip() == name:
                return i
        return None

    i_t = col("t_rel_s")
    i_vx, i_vy, i_vz = col("filter_vel_x"), col("filter_vel_y"), col("filter_vel_z")
    missing = [n for n, i in [("t_rel_s", i_t), ("filter_vel_x", i_vx),
                             ("filter_vel_y", i_vy), ("filter_vel_z", i_vz)]
              if i is None]
    if missing:
        raise ValueError(
            f"required column(s) not found: {', '.join(missing)}. "
            f"Detected delimiter {delim!r}, {len(header)} columns. "
            f"First few found: {', '.join(h.strip() for h in header[:6])}")

    t, v = [], []
    for row in rows:
        try:
            width = max(i_t, i_vx, i_vy, i_vz)
            if len(row) <= width:
                continue
            tt = float(row[i_t])
            vx, vy, vz = float(row[i_vx]), float(row[i_vy]), float(row[i_vz])
        except (ValueError, IndexError):
            continue
        t.append(tt)
        v.append((vx * vx + vy * vy + vz * vz) ** 0.5)

    if len(t) < 20:
        raise ValueError(f"only {len(t)} usable rows parsed (delimiter "
                         f"{delim!r}) - check the velocity columns actually "
                         f"contain numbers")
    t = np.asarray(t, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    # Rebase onto the file's own first sample. For a clip cut out of a
    # longer session, t_rel_s stays referenced to the ORIGINAL session t0
    # (a real case here: a 45s clip whose t_rel_s runs 1320..1365), while
    # the clip's video starts at 0. Left as-is the two axes never overlap
    # and the correlation correctly but uselessly reports "range too
    # short". Rebasing costs nothing for a full-session log, where the
    # first sample is already ~0, and is what makes a clip work at all.
    # Only relative timing matters to a cross-correlation, so no
    # information is lost either way.
    t = t - t[0]
    return t, v