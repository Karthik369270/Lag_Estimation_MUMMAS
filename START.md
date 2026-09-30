# MUMMAS · Data Curation Dashboard

Reach a moment by **playback time**, watch the camera and both VLP-16s run
together at that moment, dial out the offset between them, then cut the clip.

```
start.bat                 or      python serve.py
```

Opens <http://localhost:8005>. Ctrl+C to stop.

---

## How it is used

1. **Pick a session.** The list shows which LiDARs that run has and how long
   the video is. Sessions missing LiDAR 2 are marked `--`.
2. **Type a playback time** (`22:00`, or plain seconds) and how much you want
   before and after it.
3. **Prepare window.** Small upright lens video plus LiDAR frames either side
   of the window. Roughly 10–20 s for a 20 s window.
4. **Play, and move the lag slider** until the LiDAR scene agrees with what
   the camera shows. Pick something unambiguous — a pole, a parked vehicle, a
   kerb line — and slide until it sits in the same place.
   **Drag** either LiDAR pane to swing the view, **scroll** to zoom it
   (4 m to 120 m across; the current range is in the corner readout).
5. **Export.** Set the folder at the top of the export panel — it starts from
   `export_root` in `config.json` and can be changed per export. Clips are
   named by `extract_session_clip.py`: `clip_<date>_<run>_<MMSS>_<dur>s`.

The session, time, window, lens, lag and export folder are remembered in the
browser, so a reload puts you back where you were rather than on the first
session in the list.

---

## Why there is a lag control at all

The exporters map video playback position `P` to wall clock as `t0 + P`. That
is wrong. `t0` is when the run was *stamped*; the camera is an RTSP pull and
its first frame only arrives once six streams have been negotiated, several
seconds later. So playback `P` is really at `t0 + lag + P`.

The LiDAR is not the problem. Its timestamps are honest: on
`run_20260721_072016_8278` the pcap agrees with the session clock to **0.7 s**
at the end of a 39-minute run. A clock that was genuinely wrong would be off
at *both* ends; this one is off only at the start, because the LiDAR capture
simply opens ~3 s before `t0` is stamped. The whole error sits in the camera's
playback-to-wall-clock mapping.

**Nothing here ever rewrites a timestamp.** The lag only changes which packets
are *selected*. LiDAR, IMU and GPS all shift together, so their mutual
alignment — which per-point motion deskewing depends on — stays intact.
Rewriting LiDAR times to "fix" the video offset would trade a visible 3 s
video error for an invisible 3 s pose error.

---

## What it will tell you

- **A LiDAR that stopped early.** LiDAR 2 quit 265 s before the end of
  `run_20260721_072016_8278`, and is absent entirely from 5 of 10 sessions.
  Where a unit has no data for the chosen time, the pane says which part of
  the session that unit actually covers, rather than drawing an empty frame
  that reads as bad sync.
- **The lens stagger.** The six RTSP streams open about 0.1 s apart, so the
  rig spans roughly 0.5 s end to end. The session note reports the spread. One
  global lag cannot align all six perfectly.

---

## Layout

```
serve.py         local server, standard library + numpy
config.json      where sessions live, where clips go, which ffmpeg
lidar_io.py      fast VLP-16 pcap reader and renderer
app/index.html   the dashboard
assets/          LEMON MILK faces and artwork
cache/           prepared windows; safe to delete at any time
```

`python serve.py --clear-cache` empties `cache/` on the way up.

Reading is one-way: nothing under `session_roots` is ever written to. Export
is delegated to `extract_session_clip.py` in the pipeline folder, so a clip cut
here is cut by the same code path as one cut from the command line.

## The lag is still an estimate

The slider defaults to 3.5 s, which is an eyeballed figure, not a measured
one. Now that both streams are side by side the slider *is* the measuring
instrument — converge on a number and it can become a recorded per-run value
instead of a default. The rigorous version is a shared event: one sharp jolt
read both as a video frame index and as the `acc_z` spike in the IMU.
