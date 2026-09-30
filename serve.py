#!/usr/bin/env python3
"""
MUMMAS data curation dashboard - local server.

    python serve.py                 -> http://localhost:8005
    python serve.py --port 9000
    python serve.py --no-browser

Stop it with Ctrl+C. Standard library only, plus numpy for the LiDAR decode.

WHAT THIS IS FOR
    Reaching a moment in a session by PLAYBACK TIME, watching the camera and
    both VLP-16s run together at that moment, dialling out the offset between
    them, and only then cutting a clip.

WHY THE LAG CONTROL EXISTS
    The exporters map video playback position P to wall clock as t0 + P. That
    is wrong: t0 is when the run was stamped, but the camera is an RTSP pull
    and its first frame only lands once six streams have been negotiated,
    several seconds later. The LiDAR's own timestamps are honest - on
    run_20260721_072016_8278 the pcap agrees with the session clock to 0.7 s
    at the end of a 39-minute run - so the whole error sits in the camera's
    playback-to-wall-clock mapping. Playback P is really t0 + lag + P.

    Nothing here ever rewrites a timestamp. The lag only moves which packets
    are SELECTED, so LiDAR, IMU and GPS stay locked to each other - which is
    what per-point motion deskewing downstream depends on.

PREPARING A WINDOW
    Scrubbing a 2.3 GB mp4 and a 2.2 GB pcap live is not pleasant, so a window
    is prepared once: small upright lens videos, and LiDAR rotations packed
    at 10 Hz across the window PLUS a pad either side. The pad is what makes
    the lag slider free to move - it just picks a different frame index.

Routes
    /                              the dashboard
    /api/sessions                  every session found under session_roots
    /api/prepare       POST        build a window · GET ?job= for progress
    /api/export        POST        cut the clip via extract_session_clip.py
    /w/<id>/meta.json              what was prepared
    /w/<id>/video/lensN.mp4        upright preview video, range-served
    /w/<id>/lidar/lidarN.bin       packed rotations, Range-served
    /assets/<file>                 fonts and artwork

Nothing under session_roots is ever written to.
"""

import argparse
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

import lidar_io
import lag_estimation

ROOT = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(ROOT, "app")
ASSETS = os.path.join(ROOT, "assets")

with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as fh:
    CFG = json.load(fh)

# Prepared windows run to ~100 MB each and are thrown away freely, so they do
# NOT belong beside the app: this folder lives in OneDrive, and every prepare
# would be uploaded. Default to the machine's temp area instead, and let
# config point somewhere else if a bigger scratch disk is preferred.
CACHE = CFG.get("cache_dir") or os.path.join(
    tempfile.gettempdir(), "mummas_curation_cache")

FFMPEG = CFG.get("ffmpeg") or "ffmpeg"
PREVIEW_H = int(CFG.get("preview_height", 480))
LIDAR_HZ = int(CFG.get("lidar_hz", 10))
LIDAR_PX = int(CFG.get("lidar_px", 460))
LAG_PAD = float(CFG.get("lag_pad_s", 8.0))

JOBS = {}
JOBS_LOCK = threading.Lock()

# ───────────────────────────────────────────── manual uploads
#
# An uploaded session is given the exact same shape _sessions() produces -
# run, run_dir, l1, l2, t0, duration, lenses, lens_durations - so _prepare()
# and everything after it runs completely unmodified. The only new code is
# how this dict gets built and where it is looked up from.
UPLOADS = {}
UPLOADS_LOCK = threading.Lock()

# parsed (t_rel_s, speed_m_s) arrays per run, once an IMU/GPS csv has been
# attached via /api/attach_imu - kept separate from the session dict itself
# so attaching one doesn't need to touch _sessions()/_register_upload at all
IMU_DATA = {}
IMU_DATA_LOCK = threading.Lock()


def _parse_multipart(body: bytes, content_type: str):
    """Return {field_name: (filename_or_None, bytes)} for one multipart
    form-data body. Minimal but correct: stdlib http.server has nothing
    like Flask's request.files, so this exists to stand in for it."""
    m = re.search(r'boundary="?([^";]+)"?', content_type)
    if not m:
        raise ValueError("no multipart boundary in Content-Type")
    boundary = ("--" + m.group(1)).encode("utf-8")
    parts = body.split(boundary)
    out = {}
    for part in parts:
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        if b"\r\n\r\n" not in part:
            continue
        head, data = part.split(b"\r\n\r\n", 1)
        data = data.rstrip(b"\r\n")
        disp = None
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-disposition"):
                disp = line.decode("utf-8", "replace")
        if not disp:
            continue
        nm = re.search(r'name="([^"]*)"', disp)
        fn = re.search(r'filename="([^"]*)"', disp)
        if not nm:
            continue
        out[nm.group(1)] = (fn.group(1) if fn else None, data)
    return out


def _probe_duration(path):
    """Video duration in seconds via ffprobe - serve.py has no cv2/video
    library of its own, only ffmpeg by subprocess, so this matches that."""
    ffprobe = FFMPEG.replace("ffmpeg.exe", "ffprobe.exe").replace(
        "ffmpeg", "ffprobe")
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30)
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def _register_upload(files):
    """files: {field_name: (filename_or_None, bytes)} from _parse_multipart.
    Saves everything under CACHE/uploads/<run>/, in the same on-disk shape
    _sessions() would have found on the filesystem, and returns the run id."""
    run = f"upload_{int(time.time() * 1000)}"
    run_dir = os.path.join(CACHE, "uploads", run)
    os.makedirs(run_dir, exist_ok=True)

    l1 = l2 = None
    for unit, field in ((1, "pcap1"), (2, "pcap2")):
        if field in files and files[field][1]:
            p = os.path.join(run_dir, f"l{unit}_raw_uploaded.pcap")
            with open(p, "wb") as fh:
                fh.write(files[field][1])
            if unit == 1:
                l1 = p
            else:
                l2 = p

    lenses, lens_durations = [], {}
    for key, (fname, data) in files.items():
        m = re.match(r"^lens(\d)$", key)
        if not m or not data:
            continue
        n = int(m.group(1))
        lens_dir = os.path.join(run_dir, f"LENS{n}")
        os.makedirs(lens_dir, exist_ok=True)
        vp = os.path.join(lens_dir, f"video_lens{n}.mp4")
        with open(vp, "wb") as fh:
            fh.write(data)
        lenses.append(n)
        lens_durations[str(n)] = round(_probe_duration(vp), 3)
    lenses.sort()

    # Metadata (capture_stats.json / session_meta.json / run_t0.json), when
    # the user attaches it. capture_stats.json is the one that actually
    # matters: only it carries mic_end_unix AND the per-lens durations, and
    # the metadata-only delay needs both. The other two are accepted and
    # read for t0 because they are tiny and sit in the same folder, so
    # taking all three is more robust than making the user work out which
    # one holds what.
    meta_t0 = meta_mic_start = meta_mic_end = None
    meta_lens_durations = {}
    for field in ("capture_stats", "session_meta", "run_t0"):
        if field not in files or not files[field][1]:
            continue
        try:
            j = json.loads(files[field][1].decode("utf-8", "replace"))
        except Exception:
            continue
        # run_t0's fields sit at the top level; session_meta nests the same
        # shape under "run_t0" - accept either without guessing which file
        # this is from its name
        nested = j.get("run_t0") if isinstance(j.get("run_t0"), dict) else {}
        if meta_t0 is None:
            meta_t0 = j.get("t0_unix") or nested.get("t0_unix")
        if meta_mic_start is None:
            meta_mic_start = j.get("mic_start_unix")
        if meta_mic_end is None:
            meta_mic_end = j.get("mic_end_unix")
        per = j.get("per_lens")
        if isinstance(per, dict):
            for k, vv in per.items():
                if isinstance(vv, dict) and vv.get("duration_sec") is not None:
                    meta_lens_durations[str(k)] = round(
                        float(vv["duration_sec"]), 3)

    # Two different anchors are needed here, and conflating them broke
    # LiDAR display once already:
    #
    #   meta_t0    the logger's stamped session start. The metadata-only
    #              delay is meaningless against any other origin, since
    #              mic_end and the lens durations are both measured from it.
    #   pcap_t0    the uploaded pcap's own first-packet time. _prepare()
    #              computes its LiDAR window as t0 + start_s, so this is
    #              what makes playback time 0 mean "the start of the data
    #              you actually uploaded".
    #
    # For a full session they are the same. For an uploaded EXCERPT they
    # are not: a 65 s pcap cut from 701 s into a 20 min session sits 701 s
    # away from meta_t0, so anchoring playback on meta_t0 puts every
    # sensible playback time outside the pcap and shows no LiDAR at all.
    # So t0 (what _prepare uses) stays on the pcap, and meta_t0 is carried
    # separately for the metadata calculation alone.
    pcap_t0 = None
    for p in (l1, l2):
        if p:
            try:
                pcap_t0 = lidar_io.index_pcap(p).first_ts
                break
            except Exception:
                pass
    t0 = pcap_t0 if pcap_t0 is not None else (meta_t0 or time.time())

    # The logger's own recorded per-lens durations beat ffprobe on the
    # uploaded file: an uploaded clip may have been cut out of the full
    # session, so its own length says nothing about when that lens started
    # in the original run. Only fall back to the probed value for a lens
    # the metadata does not cover.
    for k, vv in meta_lens_durations.items():
        lens_durations[k] = vv

    duration = max(lens_durations.values(), default=0.0)
    s = {
        "run": run, "day": "", "stamp": "", "run_dir": run_dir, "day_dir": "",
        "t0": t0, "duration": round(duration, 3), "lenses": lenses or [1],
        "l1": l1, "l2": l2, "lens_durations": lens_durations,
        "mic_start": meta_mic_start, "mic_end": meta_mic_end,
        "meta_t0": meta_t0,
        "has_meta": bool(meta_mic_end is not None and meta_lens_durations
                        and meta_t0 is not None),
        "manual": True,
    }
    with UPLOADS_LOCK:
        UPLOADS[run] = s
    return s



# ───────────────────────────────────────────── sessions

def _sessions():
    """Every run that has a camera folder, a t0 and at least one pcap.

    The tree is by sensor, not by run: CAMERA/run_<stamp>_<id>/LENSn, while
    LIDAR/IMU/AQI sit at day level named by the same stamp. So the camera run
    is the anchor and the rest are found from its stamp.
    """
    out = []
    for root in CFG.get("session_roots", []):
        if not os.path.isdir(root):
            continue
        for day in sorted(os.listdir(root)):
            day_dir = os.path.join(root, day)
            cam_dir = os.path.join(day_dir, "CAMERA")
            if not os.path.isdir(cam_dir):
                continue
            for run in sorted(os.listdir(cam_dir)):
                run_dir = os.path.join(cam_dir, run)
                stats_p = os.path.join(run_dir, "capture_stats.json")
                if not os.path.isfile(stats_p):
                    continue
                try:
                    with open(stats_p, encoding="utf-8") as fh:
                        stats = json.load(fh)
                except Exception:
                    continue
                t0 = stats.get("t0_unix")
                if not t0:
                    continue
                m = re.match(r"run_(\d{8})_(\d{6})_", run)
                if not m:
                    continue
                stamp = (f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]}"
                         f"_{m.group(2)[:2]}-{m.group(2)[2:4]}-{m.group(2)[4:6]}")
                per = stats.get("per_lens", {})
                lenses = sorted(int(k) for k in per)
                dur = per.get("1", {}).get("duration_sec") or (
                    max((v.get("duration_sec", 0) for v in per.values()),
                        default=0))

                def pcap(unit):
                    p = os.path.join(day_dir, "LIDAR", f"LIDAR {unit}",
                                     f"l{unit}_raw_{stamp}.pcap")
                    return p if os.path.isfile(p) and \
                        os.path.getsize(p) > 50_000 else None

                l1, l2 = pcap(1), pcap(2)
                if not (l1 or l2):
                    continue
                out.append({
                    "run": run, "day": day, "stamp": stamp,
                    "run_dir": run_dir, "day_dir": day_dir,
                    "t0": t0, "duration": round(dur, 3),
                    "lenses": lenses, "l1": l1, "l2": l2,
                    # Per-lens durations fan out in 0.1 s steps because the
                    # logger opens the six RTSP streams sequentially; surface
                    # it rather than pretend the rig starts as one.
                    "lens_durations": {k: round(v.get("duration_sec", 0), 3)
                                       for k, v in per.items()},
                    "mic_start": stats.get("mic_start_unix"),
                    "mic_end": stats.get("mic_end_unix"),
                })
    return out


def _session_by_run(run):
    with UPLOADS_LOCK:
        if run in UPLOADS:
            return UPLOADS[run]
    for s in _sessions():
        if s["run"] == run:
            return s
    return None


# ───────────────────────────────────────────── prepare

def _lidar_coverage(path):
    try:
        idx = lidar_io.index_pcap(path)
        return idx, idx.first_ts, idx.last_ts
    except Exception:
        return None, None, None


def _prepare(job_id, run, start_s, dur_s, lenses):
    def note(**kw):
        with JOBS_LOCK:
            JOBS[job_id].update(kw)

    try:
        s = _session_by_run(run)
        if not s:
            note(state="error", msg=f"session {run} not found")
            return
        wdir = os.path.join(CACHE, job_id)
        os.makedirs(os.path.join(wdir, "video"), exist_ok=True)

        t0 = s["t0"]
        total = len(lenses) + 1
        step = 0

        # ---- lens videos: upright, small, exact cut (a re-encode is needed
        # for the rotation anyway, so there is no keyframe snapping here) ----
        made = []
        for ln in lenses:
            src = os.path.join(s["run_dir"], f"LENS{ln}", f"video_lens{ln}.mp4")
            if not os.path.isfile(src):
                continue
            dst = os.path.join(wdir, "video", f"lens{ln}.mp4")
            cmd = [FFMPEG, "-y", "-v", "error",
                   "-ss", f"{start_s:.3f}", "-i", src, "-t", f"{dur_s:.3f}",
                   "-vf", f"transpose=2,scale=-2:{PREVIEW_H}",
                   "-an", "-c:v", "libx264", "-preset", "veryfast",
                   "-crf", "26", "-movflags", "+faststart", dst]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and os.path.isfile(dst):
                made.append(ln)
            step += 1
            note(state="running", pct=int(100 * step / total),
                 msg=f"lens {ln}")

        # ---- LiDAR frames across the window plus a pad on both sides, so
        # moving the lag slider is only an index shift -------------------
        lo = t0 + start_s - LAG_PAD
        hi = t0 + start_s + dur_s + LAG_PAD
        n_frames = int((hi - lo) * LIDAR_HZ)
        units = {}
        for unit in (1, 2):
            p = s.get(f"l{unit}")
            if not p:
                units[str(unit)] = {"ok": False, "reason": "no pcap"}
                continue
            idx, first, last = _lidar_coverage(p)
            if idx is None:
                units[str(unit)] = {"ok": False, "reason": "unreadable pcap"}
                continue
            # A unit that stopped early is a real failure mode - LiDAR 2 quit
            # 265 s before the end of run_20260721_072016_8278 - so report the
            # gap instead of emitting blank frames that read as bad sync.
            if last < lo or first > hi:
                units[str(unit)] = {
                    "ok": False,
                    "reason": (f"no data here; covers "
                               f"{first - t0:.0f}s to {last - t0:.0f}s"),
                    "covers": [round(first - t0, 1), round(last - t0, 1)]}
                continue
            # One binary per unit, drawn client-side on a canvas, exactly as
            # the visualisation dashboard does it: xyz as int16 centimetres
            # followed by uint8 intensity, 7 bytes a point. The browser pulls
            # a single rotation with a Range request, so disk is the only
            # cost of the pad - nothing extra is held in memory.
            os.makedirs(os.path.join(wdir, "lidar"), exist_ok=True)
            binp = os.path.join(wdir, "lidar", f"lidar{unit}.bin")
            frames = []
            off = 0
            with open(binp, "wb") as fb:
                for i in range(n_frames):
                    t = lo + i / LIDAR_HZ
                    if t < first or t > last:
                        frames.append([0, 0])
                        continue
                    try:
                        xyz, inten, _ = lidar_io.read_window(
                            p, t, t + 1.0 / LIDAR_HZ, idx)
                    except Exception:
                        frames.append([0, 0])
                        continue
                    n = int(xyz.shape[0])
                    if n == 0:
                        frames.append([0, 0])
                        continue
                    P = np.clip(xyz * 100.0, -32760, 32760).astype("<i2")
                    fb.write(P.tobytes())
                    fb.write(np.asarray(inten, dtype=np.uint8).tobytes())
                    frames.append([off, n])
                    off += n * 7
                    if i % 25 == 0:
                        note(state="running",
                             pct=int(100 * (step + i / max(n_frames, 1))
                                     / total),
                             msg=f"LiDAR {unit}  {i}/{n_frames}")
            got = sum(1 for f in frames if f[1])
            units[str(unit)] = {"ok": got > 0, "frames": frames,
                                "n_filled": got,
                                "bytes": os.path.getsize(binp),
                                "covers": [round(first - t0, 1),
                                           round(last - t0, 1)]}
        step += 1

        meta = {
            "id": job_id, "run": run, "t0": t0,
            "start_s": start_s, "duration_s": dur_s,
            "lenses": made, "lidar": units,
            "lidar_hz": LIDAR_HZ, "lag_pad_s": LAG_PAD,
            "lidar_t_lo": lo, "n_frames": n_frames,
            "preview_height": PREVIEW_H,
        }
        with open(os.path.join(wdir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        note(state="done", pct=100, msg="ready", meta=meta)
    except Exception as exc:
        note(state="error", msg=f"{exc}",
             trace=traceback.format_exc()[-1500:])


# ───────────────────────────────────────────── http

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    # -- helpers --
    def _send(self, code, body=b"", ctype="application/json", extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj), "application/json")

    def _file(self, path, ctype=None):
        """Serve a file, honouring Range so the browser can scrub video."""
        if not os.path.isfile(path):
            self._send(404, "not found", "text/plain")
            return
        size = os.path.getsize(path)
        ctype = ctype or mimetypes.guess_type(path)[0] or \
            "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        code = 200
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
                end = min(end, size - 1)
                if start > end:
                    self._send(416, "bad range", "text/plain")
                    return
                code = 206
        length = end - start + 1
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as fh:
            fh.seek(start)
            left = length
            while left > 0:
                chunk = fh.read(min(262144, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    # -- routing --
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        u = urlparse(self.path)
        p = unquote(u.path)
        q = parse_qs(u.query)

        if p == "/" or p == "/index.html":
            # No-store specifically for the page itself: _file() sets no
            # cache header at all, which left the browser free to keep
            # showing an old copy of this file after every edit, even past
            # an ordinary reload. Video and LiDAR bin serving deliberately
            # keep using plain _file() below, unchanged - they are large,
            # static once written, and Range-served, so normal caching
            # there is correct and worth keeping.
            ip = os.path.join(APP, "index.html")
            if not os.path.isfile(ip):
                return self._send(404, "not found", "text/plain")
            with open(ip, "rb") as fh:
                body = fh.read()
            return self._send(200, body, "text/html")   # _send always
            # sets Cache-Control: no-store on its own - no extra needed

        if p.startswith("/assets/"):
            return self._file(os.path.join(ASSETS, os.path.basename(p)))

        if p == "/api/sessions":
            try:
                with UPLOADS_LOCK:
                    uploaded = list(UPLOADS.values())
                strip = lambda s: (
                    {k: v for k, v in s.items()
                     if k not in ("l1", "l2", "run_dir", "day_dir")}
                    | {"l1": bool(s["l1"]), "l2": bool(s["l2"])})
                return self._json({
                    "export_root": CFG.get("export_root", ""),
                    # uploads first - the one you just added is the one
                    # you are about to work with
                    "sessions": [strip(s) for s in uploaded]
                              + [strip(s) for s in _sessions()]})
            except Exception as exc:
                return self._json({"error": str(exc)}, 500)

        if p == "/api/prepare":
            jid = (q.get("job") or [""])[0]
            with JOBS_LOCK:
                j = dict(JOBS.get(jid) or {})
            if not j:
                return self._json({"error": "no such job"}, 404)
            return self._json(j)

        if p.startswith("/w/"):
            rest = p[3:].split("/", 1)
            if len(rest) == 2:
                return self._file(os.path.join(CACHE, rest[0],
                                               *rest[1].split("/")))

        return self._send(404, "not found", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        p = unquote(u.path)

        if p == "/api/upload":
            ctype = self.headers.get("Content-Type", "")
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else b""
            try:
                files = _parse_multipart(body, ctype)
                if not any(k in files for k in ("pcap1", "pcap2")):
                    return self._json(
                        {"error": "at least one pcap file is required"}, 400)
                if not any(re.match(r"^lens\d$", k) for k in files):
                    return self._json(
                        {"error": "at least one lens video is required"}, 400)
                s = _register_upload(files)
                return self._json({"run": s["run"], "lenses": s["lenses"],
                                   "l1": bool(s["l1"]), "l2": bool(s["l2"]),
                                   "duration": s["duration"], "t0": s["t0"],
                                   "has_meta": s.get("has_meta", False)})
            except Exception as exc:
                return self._json({"error": f"{type(exc).__name__}: {exc}"},
                                  500)

        if p == "/api/attach_imu":
            ctype = self.headers.get("Content-Type", "")
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else b""
            try:
                files = _parse_multipart(body, ctype)
                run = files.get("run", (None, b""))[1].decode("utf-8", "replace")
                if not run or _session_by_run(run) is None:
                    return self._json({"error": "unknown session - prepare "
                                       "or upload it first"}, 400)
                if "imu" not in files or not files["imu"][1]:
                    return self._json({"error": "no imu file in the request"},
                                      400)
                t, v = lag_estimation.parse_imu_csv(files["imu"][1])
                with IMU_DATA_LOCK:
                    IMU_DATA[run] = (t, v)
                return self._json({"rows": int(len(t))})
            except Exception as exc:
                return self._json({"error": f"{type(exc).__name__}: {exc}"},
                                  400)

        b = self._body()

        if p == "/api/lag_estimate":
            run = b.get("run")
            lens = int(b.get("lens", 1))
            with IMU_DATA_LOCK:
                imu = IMU_DATA.get(run)
            if imu is None:
                return self._json({"error": "attach an IMU/GPS csv for this "
                                   "session first"}, 400)
            s = _session_by_run(run)
            if s is None:
                return self._json({"error": "unknown session"}, 404)
            video = os.path.join(s["run_dir"], f"LENS{lens}",
                                 f"video_lens{lens}.mp4")
            if not os.path.isfile(video):
                return self._json({"error": f"no video for lens {lens} in "
                                   f"this session"}, 404)
            try:
                import cv2
            except ImportError:
                return self._json({"error": "cv2 (opencv-python) is not "
                                   "installed - needed only for this "
                                   "optical-flow check, nothing else in the "
                                   "dashboard needs it"}, 500)
            try:
                cap = cv2.VideoCapture(video)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # sampled across the WHOLE video, capped so a long session
                # does not take unreasonably long to read through once
                step = max(1, n_frames // 300)
                frames = []
                idx = 0
                while idx < n_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, fr = cap.read()
                    if not ok:
                        break
                    frames.append(fr)
                    idx += step
                cap.release()
                if len(frames) < 5:
                    return self._json({"error": "could not read enough "
                                       "frames from this video"}, 500)
                flow = lag_estimation.optical_flow_speed_proxy(frames)
                t_flow = (np.arange(len(frames) - 1) + 0.5) * step / fps
                t_imu, v_imu = imu
                res = lag_estimation.estimate_lag(t_imu, v_imu, t_flow, flow,
                                                  lag_range=(-8.0, 8.0))
                return self._json({
                    "best_lag_s": res["best_lag_s"],
                    "confidence": res["confidence"],
                    "at_edge": res["at_edge"], "note": res["note"],
                    "lags_s": res["lags_s"].tolist(),
                    "correlation": [None if not np.isfinite(c) else float(c)
                                   for c in res["correlation"]]})
            except Exception as exc:
                traceback.print_exc()
                return self._json({"error": f"{type(exc).__name__}: {exc}"},
                                  500)

        if p == "/api/prepare":
            run = b.get("run")
            try:
                start_s = float(b.get("start_s", 0))
                dur_s = float(b.get("duration_s", 20))
            except (TypeError, ValueError):
                return self._json({"error": "bad times"}, 400)
            lenses = [int(x) for x in (b.get("lenses") or [1])]
            jid = f"w{int(time.time() * 1000)}"
            with JOBS_LOCK:
                JOBS[jid] = {"state": "running", "pct": 0, "msg": "starting"}
            threading.Thread(target=_prepare,
                             args=(jid, run, start_s, dur_s, lenses),
                             daemon=True).start()
            return self._json({"job": jid})

        if p == "/api/export":
            run = b.get("run")
            try:
                start_s = float(b.get("start_s", 0))
                dur_s = float(b.get("duration_s", 20))
                lag = float(b.get("lag", 0))
            except (TypeError, ValueError):
                return self._json({"error": "bad numbers"}, 400)
            # The folder comes from the page so it can be changed without
            # editing config.json; config only supplies the starting value.
            out_root = str(b.get("out") or CFG.get("export_root") or "clips")
            if not os.path.isdir(out_root):
                return self._json(
                    {"error": f"export folder does not exist: {out_root}"}, 400)
            pipeline = CFG.get("pipeline") or "."
            script = os.path.join(pipeline, "extract_session_clip.py")
            if not os.path.isfile(script):
                return self._json(
                    {"error": f"extract_session_clip.py not found in "
                              f"{pipeline}"}, 400)
            # centre is the playback time that was typed; the flat layout
            # names the folder after that moment's unix time, the way the
            # older MUMMAS Data clips do.
            try:
                centre_s = float(b.get("centre_s", start_s + dur_s / 2))
            except (TypeError, ValueError):
                centre_s = start_s + dur_s / 2
            mm, ss = int(start_s // 60), start_s % 60
            cmd = [sys.executable, "-u", "extract_session_clip.py",
                   "--run", run, "--start", f"{mm}:{ss:04.1f}",
                   "--duration", f"{dur_s}",
                   "--centre", f"{centre_s}",
                   "--sensor-offset", f"{lag}",
                   "--layout", "flat",
                   "--out", out_root]
            r = subprocess.run(cmd, cwd=pipeline, capture_output=True,
                               text=True)
            return self._json({"rc": r.returncode, "cmd": " ".join(cmd),
                               "out": (r.stdout or "")[-4000:],
                               "err": (r.stderr or "")[-2000:]})

        return self._send(404, "not found", "text/plain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8005)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--clear-cache", action="store_true")
    a = ap.parse_args()

    if a.clear_cache and os.path.isdir(CACHE):
        # Never let a locked file take the server down with it - OneDrive and
        # virus scanners both hold handles open, and a cache that refuses to
        # clear is not a reason to refuse to start.
        failed = []
        shutil.rmtree(CACHE, onerror=lambda f, path, e: failed.append(path))
        if failed:
            print(f"note: {len(failed)} cache entries could not be removed "
                  f"(in use); carrying on")
    os.makedirs(CACHE, exist_ok=True)

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://localhost:{a.port}"
    print(f"MUMMAS data curation dashboard  ->  {url}")
    print("Ctrl+C to stop.")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()