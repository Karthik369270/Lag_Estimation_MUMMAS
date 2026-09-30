# MUMMAS Data Curation Dashboard

Finds the right clip in a MUMMAS session, dials out the camera/LiDAR timing
offset, and exports the clip with that offset applied.

The problem it exists for: the logger starts LiDAR, IMU and GPS on one
clock, but the camera is an RTSP pull whose first frame only lands once six
streams have finished negotiating — several seconds later, and by a
different amount on every run. That offset is invisible until something
downstream comes out wrong, by which point a large clip has already been
written. This lets it be measured and corrected first.

## Running it

```bash
python serve.py
```

Opens <http://localhost:8005>. Options: `--port N`, `--no-browser`.

Requirements: Python 3.9+, `numpy`, and ffmpeg on disk (path set in
`config.json`). `opencv-python` is needed **only** for the IMU/optical-flow
estimate — everything else works without it, and the dashboard says so
plainly rather than failing silently if it is missing.

```bash
pip install numpy opencv-python
```

## Layout

```
serve.py            HTTP server, session scan, window preparation, export
lag_estimation.py   cross-correlation lag estimate + IMU/GPS CSV parsing
lidar_io.py         VLP-16 pcap indexing, windowed reads, rasterising
config.json         paths and preview settings — edit per machine
app/index.html      the whole frontend
assets/             logos and fonts (optional; absence is cosmetic only)
```

`index.html` must sit inside `app/`. `serve.py` reads it from there and will
404 if it is moved alongside `serve.py`.

## Setting up on a new machine

**`config.json` is per-machine and is not in the repo** (it is gitignored,
because every path in it is specific to one computer). A fresh clone has no
`config.json` and will not start until you make one:

```powershell
copy config.example.json config.json
```

Then open `config.json` and edit it. Two rules that between them cause most
first-run failures:

**Leave `ffmpeg` empty.** The server finds ffmpeg by itself - on PATH, then
in the usual Windows install locations - and prints where it found it on
startup. Only set a path if it reports finding nothing.

**Never leave an `<angle bracket>` placeholder in a path.** A half-replaced
one (say `ffmpeg-<version>-full_build` with only the username filled in) is
a path that does not exist, and it surfaces later as a bare
`[WinError 2] The system cannot find the file specified` when you press
Prepare Window - which reads like a missing pcap or video and sends you
looking in entirely the wrong place. The server now detects this at startup
and says so, and falls back to autodetection rather than failing.

If you only ever upload files through the browser, `session_roots` can stay
empty and `pipeline` can stay `"."`. Those matter only for scanned sessions
and for export.

## config.json

| key | meaning |
| --- | --- |
| `session_roots` | folders that **contain** day-stamped folders (`30062026`, …). Point at the parent, not at a day folder itself. |
| `export_root` | default output directory for exported clips |
| `pipeline` | folder holding `extract_session_clip.py` (export only) |
| `ffmpeg` | leave empty to autodetect; a full path only if that fails |
| `cache_dir` | scratch space for prepared windows; defaults to the system temp dir |
| `preview_height`, `lidar_px`, `lidar_hz`, `lag_pad_s` | preview size and how far the lag slider can travel |

## Two ways to load a session

**Scanned** — set `session_roots` and sessions appear in the dropdown
automatically, read from the real folder tree.

**Uploaded** — attach a pcap, a lens video, and optionally the session's
JSON metadata directly in the browser. Useful when the folder tree is not
mounted, or for a clip already cut out of a session. Export is unavailable
for uploaded sessions, because `extract_session_clip.py` works from the
scanned tree and has no way to find files that bypassed it; the panel hides
itself rather than offering a button that cannot work.

## Measuring the camera lag

Two independent methods. They are deliberately not wired into each other —
if they agree, that is real corroboration, not one tool confirming itself.

### Metadata-only (preferred where available)

Pure arithmetic from `capture_stats.json`:

```
delay = (mic_end_unix − t0_unix) − per_lens[N].duration_sec
```

If the session ran 211.84 s but this lens only recorded 209.90 s, then
1.94 s is missing, and — since RTSP negotiation happens at startup — that
missing time is at the front.

No video decoding, no correlation, no confidence caveat. Appears
automatically once a session and lens are selected, and updates per lens.

**The assumption it rests on, stated rather than buried:** that every lens
stopped at the same real moment. `mic_end_unix` is a single session-level
field, so a lens that stopped *early* would also show missing time, and this
calculation would wrongly attribute it to a late start. Nothing in the file
distinguishes the two cases. It also assumes container duration reflects
real recorded time — dropped frames would land directly in the result.

Observed on one real session: lens 1 at 4.639 s down to lens 6 at 4.039 s,
decreasing smoothly — exactly the signature of six streams negotiating in
sequence. That internal consistency is good supporting evidence, not proof.

### IMU/optical-flow cross-correlation

Attach the session's IMU/GPS CSV, then estimate. It cross-correlates the
camera's own optical-flow motion against speed magnitude from
`filter_vel_x/y/z`, using the `t_rel_s` column directly.

Reports a confidence (0–1) alongside the number, and flags when the best fit
sits at the edge of the searched range — which means the true value is
outside it, or there is no real match to find.

**This method is unreliable on short clips.** Measured across four ~40 s
clips: +3.76 s (conf 0.56), −8.00 s (conf 0.39, pinned to the range edge),
+0.64 s (conf 0.84), −6.94 s (conf 0.71). A 12-second spread with sign
flips, on a rig that should have no such variation. With only ~40 s of
signal and a ±8 s search, many shifts fit nearly as well as the best one and
the correlation latches onto noise. Treat anything below ~0.85 confidence as
unusable, and prefer full sessions over clips.

### Verifying either of them

Set the lag slider to the estimate, find a sharp event — a hard stop, a fast
turn — and check whether the LiDAR sweep and the camera image agree. Then
set it to 0 and look again. If the estimate visibly lines up and 0 does not,
that is confirmation from a genuinely independent third method.

## Notes on behaviour that has caused confusion

**Two different time anchors exist, and conflating them breaks things.**
`t0` is what playback time 0 means; for an uploaded excerpt it is the pcap's
own first-packet time, so playback lands on real data. `meta_t0` is the
logger's stamped session start, used only for the metadata calculation,
which is meaningless against any other origin. For a full session they
coincide. For a 65 s pcap cut from 701 s into a session they do not, and
using the wrong one either hides all LiDAR or produces a nonsense delay.

**"NO DATA HERE; COVERS 701S TO 766S"** means the requested playback window
does not overlap the pcap — normal when a short pcap excerpt is paired with
a full-length video. Ask for a playback time inside the stated range.

**`ConnectionAbortedError` / `WinError 10053` in the console** is harmless.
The browser closed a connection mid-request — routine when navigating away
or cancelling a video fetch. Python's `http.server` logs it noisily.

**IMU CSVs vary between sessions.** Some are comma-separated, some
tab-separated; the delimiter is sniffed from the header rather than assumed.
A clip's `t_rel_s` may stay referenced to the original session's `t0` (one
real case runs 1320–1365 s), so it is rebased onto its own first sample —
harmless for a full log, and what makes a clip work at all.

## API

| route | method | purpose |
| --- | --- | --- |
| `/api/sessions` | GET | list scanned and uploaded sessions |
| `/api/upload` | POST | attach pcap / video / metadata JSON |
| `/api/prepare` | POST, GET | build a preview window; poll with `?job=` |
| `/api/attach_imu` | POST | attach an IMU/GPS CSV for a session |
| `/api/lag_estimate` | POST | optical-flow vs IMU lag estimate |
| `/api/export` | POST | cut the clip (scanned sessions only) |
