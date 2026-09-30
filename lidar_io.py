"""Read a short time window out of a VLP-16 pcap and draw it, fast enough to scrub.

This exists so video and LiDAR can be eyeballed together BEFORE a clip is
exported. The exporters cut the two streams on different clocks, and the
disagreement is invisible until something downstream comes out wrong - by which
point a gigabyte has been written. Rendering a couple of rotations lets the
misalignment be seen and dialled out first.

Seeking matters: the pcaps run to 2+ GB, and a scrub that rescans from byte 0
is unusable. Every VLP-16 data packet is the same size, so when a file is
uniform the packet at any index sits at a computable offset and the window is
found by binary search - a handful of seeks instead of two million. Files that
are not uniform fall back to a forward scan, which is correct but slow, so the
caller is told which happened.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GLOBAL_HDR = 24
REC_HDR = 16
VLP_FRAME = 1248          # 42 B eth/ip/udp + 1206 B payload
VLP_PAYLOAD = 1206
STRIDE = REC_HDR + VLP_FRAME

BLOCKS = 12
CH_PER_BLOCK = 32
# VLP-16 ring elevations, in firing order within a block
VERT_DEG = np.array([-15, 1, -13, 3, -11, 5, -9, 7,
                     -7, 9, -5, 11, -3, 13, -1, 15], dtype=np.float32)
VERT_RAD = np.radians(VERT_DEG)
SIN_V, COS_V = np.sin(VERT_RAD), np.cos(VERT_RAD)


@dataclass
class PcapIndex:
    path: Path
    uniform: bool
    n_packets: int
    first_ts: float
    last_ts: float

    @property
    def span_s(self) -> float:
        return self.last_ts - self.first_ts


def _rec_ts(f, off: int) -> float:
    f.seek(off)
    ts, frac, _incl, _orig = struct.unpack("<IIII", f.read(REC_HDR))
    return ts + frac / 1e6


def index_pcap(path: str | Path) -> PcapIndex:
    """Cheap structural probe - no full scan when the file is uniform."""
    path = Path(path)
    size = path.stat().st_size
    body = size - GLOBAL_HDR
    uniform = body > 0 and body % STRIDE == 0
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic not in (0xa1b2c3d4, 0xa1b23c4d):
            raise ValueError(f"not a little-endian pcap: 0x{magic:08x}")
        if uniform:
            n = body // STRIDE
            # Confirm the stride guess on a few records rather than trusting
            # arithmetic - a file can divide evenly by luck.
            for i in (0, n // 2, n - 1):
                f.seek(GLOBAL_HDR + i * STRIDE + 8)
                incl = struct.unpack("<I", f.read(4))[0]
                if incl != VLP_FRAME:
                    uniform = False
                    break
        if uniform:
            n = body // STRIDE
            return PcapIndex(path, True, n,
                             _rec_ts(f, GLOBAL_HDR),
                             _rec_ts(f, GLOBAL_HDR + (n - 1) * STRIDE))
        # Non-uniform: walk it once, tolerating junk.
        f.seek(GLOBAL_HDR)
        first = last = None
        n = 0
        while True:
            h = f.read(REC_HDR)
            if len(h) < REC_HDR:
                break
            ts, frac, incl, _ = struct.unpack("<IIII", h)
            if incl > 65535:
                break
            t = ts + frac / 1e6
            first = t if first is None else first
            last = t
            n += 1
            f.seek(incl, 1)
        return PcapIndex(path, False, n, first or 0.0, last or 0.0)


def _bisect_offset(f, idx: PcapIndex, t: float) -> int:
    """First record index whose timestamp is >= t (uniform files only)."""
    lo, hi = 0, idx.n_packets - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if _rec_ts(f, GLOBAL_HDR + mid * STRIDE) < t:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _decode(payloads: np.ndarray) -> tuple:
    """Vectorised VLP-16 decode of an (N, 1206) uint8 array -> xyz, intensity.

    Each block holds two firings of 16 rings. The second firing happens halfway
    to the next block's azimuth, so it is interpolated rather than reusing the
    block azimuth - skipping that smears every other ring by up to a degree,
    which is exactly the kind of error that looks like bad calibration later.
    """
    n = payloads.shape[0]
    blocks = payloads[:, :BLOCKS * 100].reshape(n, BLOCKS, 100)

    az_raw = (blocks[:, :, 2].astype(np.uint16)
              | (blocks[:, :, 3].astype(np.uint16) << 8))
    az = az_raw.astype(np.float32) / 100.0                    # (n, 12) degrees

    data = blocks[:, :, 4:].reshape(n, BLOCKS, CH_PER_BLOCK, 3)
    dist_raw = (data[:, :, :, 0].astype(np.uint16)
                | (data[:, :, :, 1].astype(np.uint16) << 8))
    dist = dist_raw.astype(np.float32) * 0.002                # 2 mm units
    inten = data[:, :, :, 2].astype(np.uint8)

    # azimuth of the next block, to interpolate the second firing
    az_next = np.empty_like(az)
    az_next[:, :-1] = az[:, 1:]
    az_next[:, -1] = az[:, -1] + (az[:, -1] - az[:, -2])
    d_az = (az_next - az) % 360.0
    half = d_az / 2.0

    az_full = np.empty((n, BLOCKS, CH_PER_BLOCK), dtype=np.float32)
    az_full[:, :, :16] = az[:, :, None]
    az_full[:, :, 16:] = (az + half)[:, :, None]
    az_full %= 360.0

    az_rad = np.radians(az_full)
    sin_v = np.tile(SIN_V, 2)[None, None, :]
    cos_v = np.tile(COS_V, 2)[None, None, :]

    xy = dist * cos_v
    x = xy * np.sin(az_rad)
    y = xy * np.cos(az_rad)
    z = dist * sin_v

    keep = dist > 0.05
    xyz = np.stack([x[keep], y[keep], z[keep]], axis=1)
    return xyz, inten[keep]


def read_window(path: str | Path, t_start: float, t_end: float,
                idx: PcapIndex | None = None,
                max_packets: int = 4000) -> tuple:
    """Points captured in [t_start, t_end]. Returns (xyz, intensity, n_pkts)."""
    idx = idx or index_pcap(path)
    payloads = []
    with open(path, "rb") as f:
        if idx.uniform:
            i = _bisect_offset(f, idx, t_start)
            f.seek(GLOBAL_HDR + i * STRIDE)
            while i < idx.n_packets and len(payloads) < max_packets:
                h = f.read(REC_HDR)
                if len(h) < REC_HDR:
                    break
                ts, frac, incl, _ = struct.unpack("<IIII", h)
                t = ts + frac / 1e6
                if t > t_end:
                    break
                buf = f.read(incl)
                if incl == VLP_FRAME and len(buf) == VLP_FRAME:
                    payloads.append(np.frombuffer(buf[42:], dtype=np.uint8))
                i += 1
        else:
            f.seek(GLOBAL_HDR)
            while len(payloads) < max_packets:
                h = f.read(REC_HDR)
                if len(h) < REC_HDR:
                    break
                ts, frac, incl, _ = struct.unpack("<IIII", h)
                if incl > 65535:
                    break
                t = ts + frac / 1e6
                buf = f.read(incl)
                if t < t_start:
                    continue
                if t > t_end:
                    break
                if incl == VLP_FRAME and len(buf) == VLP_FRAME:
                    payloads.append(np.frombuffer(buf[42:], dtype=np.uint8))

    if not payloads:
        return np.empty((0, 3), np.float32), np.empty(0, np.uint8), 0
    arr = np.stack(payloads)
    xyz, inten = _decode(arr)
    return xyz, inten, arr.shape[0]


def render(xyz: np.ndarray, inten: np.ndarray, view: str = "bev",
           rng: float = 30.0, px: int = 520) -> np.ndarray:
    """Rasterise to an RGB image without matplotlib - fast enough to scrub.

    'bev' is the top-down road view, best for spotting where a pile or a parked
    vehicle sits relative to the track. 'front' is the forward elevation, which
    reads more like the camera and is the easier one to match against a frame.
    """
    img = np.zeros((px, px, 3), dtype=np.uint8)
    img[:] = (14, 14, 18)
    if xyz.shape[0] == 0:
        return img

    if view == "bev":
        a, b, col = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        lo_c, hi_c = -2.0, 3.0
    else:                                   # forward elevation, +Y ahead
        fwd = xyz[:, 1] > 0
        if not fwd.any():
            return img
        a, b, col = xyz[fwd, 0], xyz[fwd, 2], xyz[fwd, 1]
        lo_c, hi_c = 0.0, rng

    u = ((a + rng) / (2 * rng) * (px - 1)).astype(np.int32)
    v = ((rng - b) / (2 * rng) * (px - 1)).astype(np.int32)
    if view == "front":
        v = ((6.0 - b) / 12.0 * (px - 1)).astype(np.int32)

    ok = (u >= 0) & (u < px) & (v >= 0) & (v < px)
    u, v, col = u[ok], v[ok], col[ok]
    if u.size == 0:
        return img

    c = np.clip((col - lo_c) / max(hi_c - lo_c, 1e-6), 0, 1)
    # turbo-ish ramp: blue -> green -> yellow -> red
    r = np.clip(1.6 * c - 0.4, 0, 1)
    g = np.clip(1.4 * np.sin(np.pi * c), 0, 1)
    bl = np.clip(1.2 - 2.0 * c, 0, 1)
    img[v, u, 0] = (r * 255).astype(np.uint8)
    img[v, u, 1] = (g * 255).astype(np.uint8)
    img[v, u, 2] = (bl * 255).astype(np.uint8)

    # sensor origin + range rings so distances are readable at a glance
    cx = cy = px // 2
    for rr in (5, 10, 20):
        if rr >= rng:
            continue
        rad = int(rr / rng * (px / 2))
        th = np.linspace(0, 2 * np.pi, 720)
        yy = (cy + rad * np.sin(th)).astype(np.int32)
        xx = (cx + rad * np.cos(th)).astype(np.int32)
        m = (yy >= 0) & (yy < px) & (xx >= 0) & (xx < px)
        if view == "bev":
            img[yy[m], xx[m]] = np.maximum(img[yy[m], xx[m]], (55, 55, 62))
    if view == "bev":
        img[cy - 2:cy + 3, cx - 2:cx + 3] = (255, 255, 255)
    return img
