# UPDATE — Researcher Power Pack

This document covers two waves of work. The current state of the app is
the union of both. Older features from wave 1 are preserved in wave 2.

---

# Wave 4 — Hybrid GPU packaging: one Windows installer for every machine (2026-05-29)

The Windows distributable is now **one universal installer** that works on any
PC and gains GPU acceleration on demand — no per-GPU builds, no env vars.

**How it works.** The build bakes a **universal CPU PyTorch** into the
installer (staged as a data tree, not frozen into the PYZ — a frozen torch
can't be overridden, since PyInstaller's frozen importer beats `sys.path`).
On first launch, if an NVIDIA GPU is detected, the app offers to **download
the matching CUDA build** of torch+torchvision (cu128 for RTX 50-series /
Blackwell, cu118 for older drivers) into a per-user folder
(`%LocalAppData%\CCTV-YOLO\torch_runtime`); `runtime_hook.py` puts that folder
first on `sys.path` on the next launch, so the app then runs on the GPU. A
restart is required (torch can't be hot-swapped). The app **always works on
the baked CPU torch** if the download is skipped, fails, or is interrupted —
the `.torch_ready` marker is written last, so a half-install is never used.

- New `cctv_yolo/gpu_runtime.py` — stdlib-only GPU detection, pinned-wheel
  resolution against the live PyTorch index (sha256-verified), download +
  unzip into the per-user runtime dir. Pins: cu128 → torch 2.8.0 / tv 0.23.0;
  cu118 → torch 2.7.1 / tv 0.22.1.
- New `cctv_yolo/gpu_setup_dialog.py` — first-run + Settings dialog with a
  progress bar, cancel, and a restart prompt. Wired into `main.py` (first run)
  and Settings → **"Set up / repair GPU acceleration"**.
- `cctv_yolo.spec` (Windows): excludes torch/torchvision from the bundle,
  stages the CPU build under `torch_cpu_baseline/`, freezes torch's
  pure-Python deps, and **explicitly bundles the MSVC runtime**
  (`vcruntime140_1.dll` etc.) that torch needs — PyInstaller no longer
  analyzes torch's DLLs, so this can't be left to chance.
- `build_windows.bat` now bakes the CPU baseline (pinned torch 2.8.0+cpu).
  The old build-time CUDA auto-detect (`detect_torch_variant.py`) is removed —
  GPU selection moved to runtime. macOS is unchanged (torch baked, MPS auto).
- Verified by two adversarial multi-agent reviews (the live PyTorch index was
  fetched to confirm wheel URLs/sha256/self-containment). Windows-machine
  testing still required before wide distribution.

---

# Wave 3 — Cross-platform build & runtime hardening (2026-05-28)

This wave is not about new features — it's about making the v2 desktop
app **actually build and run** on the machines testers have, across
Windows and macOS. A multi-agent cross-platform audit surfaced a stack
of blockers (the Windows build could hang forever; Training was dead in
every shipped binary; newest-generation GPUs ran on CPU or crashed) plus
a tail of reliability bugs. This is the authoritative list of what
shipped this round.

## Windows build now actually builds (the blocker)

- **Root cause of a tester's "build hangs forever / KeyboardInterrupt
  during pip install torch":** their `python` was Python 3.14, and
  PyTorch ships **no wheels for 3.13/3.14**, so pip's resolver thrashed
  indefinitely trying to find a satisfiable version. The script also
  blindly reused a venv that had been created with 3.14.
- `build_windows.bat` now:
  - **(a)** selects a supported interpreter, preferring the
    `py -3.12` / `py -3.11` / `py -3.10` launcher, and refuses to
    proceed on 3.13/3.14 with a clear message + a python.org download
    link.
  - **(b)** revalidates a **reused** venv's Python version and
    auto-recreates it if the version is unsupported.
  - **(c)** fixes the step numbering (`[4/4]` → `[4/5]`) and adds a
    torch-install verification that warns loudly if a CPU-only wheel
    was silently resolved when CUDA was requested.

## GPU / CUDA: Blackwell (RTX 50-series) support

- New **`detect_torch_variant.py`** auto-detects the GPU/driver via
  `nvidia-smi` and picks the right torch wheel index: **cu128**
  (Blackwell / driver CUDA ≥ 12.8), cu126, cu124, cu121, cu118, or
  cpu. The old fixed `cu118` default could not drive Blackwell GPUs
  (compute `sm_120`) — they ran on CPU or crashed with "no kernel
  image is available for execution on the device". A tester's
  **RTX 5070 on CUDA 13.0** now resolves to cu128.
- `gpu_info.detect_device()` now validates the GPU's compute
  capability against the bundled wheel's compiled kernels: an
  RTX 50-series GPU with a cu118 wheel is reported as
  **CPU-with-a-fixable-reason** instead of a green "GPU active" that
  crashes on first inference. CPU-fallback guidance is now
  GPU-generation-aware (it recommends cu128 for the newest GPUs).
- `processor.py` now turns the cryptic CUDA "no kernel image is
  available" error into an actionable message ("rebuild with cu128").
- `model_compare.py` now passes `device=` on track/predict and routes
  through the centralized detector, so **Model Compare uses the GPU**
  (and the same Blackwell-safety check) instead of silently running on
  CPU.

## Training works in the shipped build (was completely broken)

- The Training tab spawned `sys.executable -m ultralytics`. In a
  PyInstaller-frozen app `sys.executable` is the **GUI exe itself**, so
  this relaunched the app instead of training — the entire
  active-learning loop was **dead in every shipped .exe/.dmg** (it only
  ever worked in dev). Training now runs **in-process** via the
  Ultralytics Python API on the worker thread, with:
  - epoch-callback progress reporting,
  - a clean cooperative stop,
  - stdout/stderr redirected (the windowed build has no console),
  - DataLoader workers disabled in frozen builds (they would otherwise
    re-spawn the GUI).

## macOS distribution

- `build_mac.sh` now **ad-hoc code-signs** the `.app`
  (`codesign --deep --sign -`) so it isn't blocked as "damaged and
  can't be opened" on other Macs — it downgrades the experience to the
  normal right-click → **Open** prompt. (Full notarization still
  requires a paid Apple Developer ID; noted as the proper path.)
- Hardened the `du`/`df` disk-space preflight against `set -e`.
- Bumped the bundle's `LSMinimumSystemVersion` to **12.0** to match the
  real PySide6/torch floor (and the README).

## Reliability fixes

- **Live tab:** the stream worker (QThread + VideoCapture/VideoWriter)
  is now stopped and joined on app close and before restart, instead of
  being orphaned.
- **Performance tab:** the confusion-matrix computation now runs on a
  worker thread instead of freezing the GUI on long sessions.
- **NAS:** config read now forces UTF-8; Windows unmount uses the bare
  drive (`"Z:"`) instead of `"Z:\"` so `net use /delete` actually
  works.
- **Model downloader** no longer sets a process-wide socket timeout
  (which leaked into every other socket and was never restored); reads
  now honor the intended 30 s timeout.
- **CSV export** writers now force UTF-8 (non-ASCII ROI/class names no
  longer abort export on Windows), and the end-to-end smoke test's
  `open()`-encoding sweep was strengthened to also catch the
  `Path.open("w")` form that previously slipped through.

Found via a 34-finding multi-agent cross-platform audit; the remaining low-severity and sample-rate-specific findings are tracked for a follow-up.

---

# Wave 2 — Active-learning loop, occlusion handling, anomalies, clips, CLI (2026-05-01)

This wave focuses on the **occlusion / re-identification edge case** the
researcher flagged (a vehicle that's tracked, briefly covered, then
reappears and gets a new track ID), and on closing the active-learning
loop with confusion-matrix evaluation. The base app now ships with **9
tabs** (was 8).

## Occlusion / track-gap handling — the headliner

A new toolbar button **"Find Occlusions"** in the review window scans
every track pair and surfaces likely-same-vehicle-through-occlusion
suggestions, scored 0..1.

Scoring blends:
- **Velocity continuation**: predicts where track A would have been at
  track B's first frame using A's terminal velocity, then measures
  pixel offset to B's actual entry point.
- **Bbox-size similarity**: vehicles don't change apparent size much
  in a short occlusion.
- **Class match**: car-to-car beats car-to-truck.
- **Time gap**: shorter is better (configurable max default 90 frames).
- **Velocity-direction cosine**: does B keep moving in A's heading?

The dialog (`OcclusionSuggestionsDialog`) lists every candidate with
its sub-scores. Each row has a **Merge** button that performs the
existing merge-with-gap-interpolation, but flags every newly created
interpolated frame with `"occluded": True`.

**Visual marking** of those frames is everywhere they're rendered:
- **Canvas**: pink dotted bbox (instead of green dashed) and a
  `[occluded]` label.
- **Timeline minimap**: thicker pink vertical accent lines, separate
  from the green interpolation marks.
- **Annotated video / clips**: same pink color, thicker pen.
- **Saved JSON**: `frames[*].occluded = true` so external tools can
  filter on it.

Also: `occlusion.mark_track_uncertain_segments(track)` flags
intra-track gaps (≤ 5 frames missing inside a single track), so even
without merging the reviewer sees where the model lost the vehicle.

## Visual / CLIP search across all sessions

`visual_search.py` builds a file-backed embedding index over centroid
crops of every track in every session. Two query modes wired into the
**Cross-session Search** dialog:
- **Image query** (always available with torchvision): pick a track,
  find the most-similar tracks across every session.
- **Text query** ("red truck", "police car") if `open-clip-torch` is
  installed.

Backend auto-detected:
1. `open_clip` (ViT-B-32) — best, supports text and image.
2. `torchvision` ResNet50 — image-only fallback.
3. None — dialog explains how to install.

Index lives at `~/Documents/CCTV-YOLO/config/visual_index.{json,npy}`.
Entry count + backend name shown after build.

## Auto-clip extraction + supercut

`clips.py::find_events` scans a session for noteworthy moments —
needs-review tracks, low-confidence tracks, occluded segments, ROI
entries — and renders a 6-second annotated MP4 around each one (2s
pre / 4s post by default; configurable). Optional supercut concatenates
every clip into a single highlight reel with title cards between
events. Wired as **"Auto-clip events + supercut"** in the Analytics
tab.

## Anomaly detection from learned baselines

`anomaly.py` builds a per-ROI / per-hour-of-day baseline from every
session, then z-scores the target session's count and class-share
against that. Surfaces any metric beyond the threshold (default
±2σ) with metric, ROI, hour, value, baseline mean+std, z-score.

Wired into the new **Insights** tab. Hour is inferred from
`processed_at` plus event offset within the video — so a 2 pm clip
with traffic peaking 30 minutes in puts the spike at hour 14.

## Confusion matrix + per-class metrics

`confusion.py` runs a fresh prediction pass with any chosen model
against a held-out corrected session, IoU-matches predictions to the
correction ground truth (≥ 0.5), and produces:
- Confusion matrix with a `background` pseudo-class (covers FP/FN).
- Per-class precision / recall / F1.
- A self-rendered PNG of the matrix (no matplotlib dep).

Wired into **Insights** with a model picker, stride control, log
streaming, metrics table, and the rendered PNG. This is what closes
the active-learning loop — train on corrections, then *prove* the
trained model improved.

## Auto pixels-per-meter calibration

`calibration.py` runs two heuristics in parallel:
1. **Track-velocity** — assume the median car speed is ~30 mph
   (configurable), divide median pixel-velocity by that.
2. **Scene geometry** — Hough-line detection + RANSAC vanishing
   point + median car bbox height treated as 1.5 m.

A new **Auto** button next to the speed PPM spinbox in **Analytics**
fills in the better of the two and reports both for inspection.

## HTML session report

`report.py::render_html_report` produces a single self-contained HTML
file per session with summary stats, by-class table, embedded heatmap
(base64), OD matrix, top-10 speeds, anomalies, and a link or embedded
annotated MP4. Drop the file in any browser. Wired as
**"Generate Report"** in Analytics.

## Dataset health dashboard

`dataset_health.py` walks every corrected session and reports:
- Class & subclass balance.
- Bbox-size distribution (small / medium / large per COCO).
- Median area, median aspect ratio.
- Frame coverage (annotated frames / total frames).
- Train/val split count.

Plus heuristic warnings: "Class X is 1.5%", "85% imbalanced", "<200
total bboxes", "small bboxes dominate", "no val set yet".

Wired in **Insights**. Run before training to catch bias.

## Before/after corrections playback

`before_after.py` renders a side-by-side MP4 of raw tracks vs
corrected tracks. Frames where the bbox sets differ get a red banner
+ "DIFF" label, so scrubbing through quickly shows what changed.
Wired as **"Render Before/After"** in Analytics.

## Headless CLI

`python -m cctv_yolo.cli` exposes all major operations from the
shell — perfect for cron / overnight runs:

```
process [video | --folder PATH --recursive] [--model X] [--conf 0.25]
annotate <session_id> [--blur-lp]
heatmap <session_id> [--sigma 12]
timeseries <session_id> [--bucket 60]
speeds <session_id> --ppm 22.5
report <session_id> [--embed-video]
train [--base yolov8n.pt --epochs 30 --imgsz 640]
list-sessions
```

Operates on the same `~/Documents/CCTV-YOLO/` data root as the GUI.

## Subclass / occlusion-aware UI everywhere

- Class-change dialog now stores `track["subclass"]`; subclasses also
  surface in annotated export labels, search results, and visual
  index entries.
- Canvas + minimap + annotated MP4 + clips + before/after all
  recognize and visually distinguish `occluded` frames.

## New / changed files (wave 2)

```
occlusion.py           Gap-candidate scoring + intra-track flagging
occlusion_dialog.py    Suggestions table, accept/reject, "Apply all"
visual_search.py       Embedding index (CLIP / ResNet) + query
clips.py               Event detection + clip rendering + supercut
anomaly.py             Per-ROI/per-hour baseline + z-score detection
confusion.py           IoU-matched eval + PNG renderer
calibration.py         Track-velocity + vanishing-point ppm estimation
report.py              Self-contained HTML report generator
dataset_health.py      Class balance + bbox stats + warnings
before_after.py        Side-by-side raw-vs-corrected video
cli.py                 argparse entrypoint for headless ops
insights_tab.py        Health + anomalies + confusion-matrix UI
```

Edited: `main_window.py` (Insights tab + shortcuts),
`review_window.py` (occlusion button + merge-pair helper),
`video_canvas.py` (occluded box style),
`timeline_minimap.py` (occluded marks),
`analytics_tab.py` (auto-calibrate, clips, before/after, report),
`search_dialog.py` (visual search panel).

## New keyboard shortcut

| Shortcut | Action |
|---|---|
| Ctrl+9 | Switch to Insights tab |

(Ctrl+1..8 still target Preprocessing/Batch/Correction/Performance/Analytics/Training/Models/Live with adjusted indices.)

## Wave-2 verification

```text
$ python3 ast-parse cctv_yolo/*.py        → 46/46 OK
$ python3 -c "import every wave-2 module" → all imports succeed
$ QT_QPA_PLATFORM=offscreen MainWindow    → 9 tabs, clean shutdown
$ occlusion.find_gap_candidates synth     → score 0.93 for the
                                            "same car through gap" pair
$ calibration.calibrate_from_tracks       → returns ppm with sample count
$ confusion.evaluate + render PNG         → matrix axes correct, PNG saved
$ clips.find_events                       → ROI entries + needs-review found
```

Real-data verification (running visual search at scale, training a
model with real corrections, RTSP streams) needs the corresponding
inputs / hardware to fully exercise.

---

# Wave 1 — Researcher Power Pack (2026-05-01)

This update lands a large set of researcher-grade features in
`software2/` (the PySide6 native desktop app). Existing functionality
— preprocessing, correction, performance, ROI editing, NAS — was kept
intact and extended. The base app now ships with eight tabs instead of
three.

## TL;DR — what's new

| Capability | Where | Notes |
|---|---|---|
| Folder-wide / batch processing | **Batch** tab | Persistent queue, watch folder, priority, resume-on-crash |
| Watch folder auto-ingest | **Batch** tab | Drop a video into a watched folder → auto-queued |
| Annotated MP4 export | **Analytics** tab | Burns boxes + ROIs + counter; optional license-plate blur |
| Path-density heatmap (PNG) | **Analytics** tab | Gaussian-smoothed bbox-center accumulation |
| Origin-destination matrix | **Analytics** tab | Per-pair counts between ROIs (CSV + on-screen table) |
| Per-minute time-series CSV | **Analytics** tab | Configurable bucket, per-class & per-ROI columns |
| Speed estimation | **Analytics** tab | Pixel-to-meter calibration → mph/kph per track |
| Direction-of-travel per ROI | **Analytics** tab | N/S/E/W classification per track-entry vector |
| License-plate blur | **Analytics** tab + `lp_blur.py` | Optional plate detector, fallback to bottom-25% blur |
| Active-learning queue | **Training** tab | Sessions ranked by mean confidence + flag density |
| Build YOLO dataset from corrections | **Training** tab | Walks every corrected session, samples frames, writes `data.yaml` |
| One-click retrain | **Training** tab | Shells out to `yolo detect train`, streams logs into the UI |
| Versioned model output | `~/Documents/CCTV-YOLO/models/` | Trained `.pt` saved as `trained_<timestamp>.pt`, immediately picks up in pickers |
| Models browser + import | **Models** tab | List, set active (`★`), import a `.pt` |
| A/B model comparison | **Models** tab | Run two models on a video, side-by-side stats + Δ |
| Live RTSP / webcam | **Live** tab | RTSP URL or webcam index, FPS-throttled detection |
| Real-time alert engine | **Live** tab + `alerts.py` | Loiter, wrong-way, speed cap rules |
| Cross-session search | **File → Cross-session Search…** (Ctrl+F) | Filter tracks across ALL sessions by class / ROI / conf / length |
| Timeline minimap | **Review window** (above frame bar) | Per-track lanes, low-conf density, edited-frame markers, click-to-seek |
| Finer vehicle subclasses | Class-change dialog | sedan/SUV/pickup/box-truck/etc. saved to `track["subclass"]` |

## New / changed files

### New modules in `software2/cctv_yolo/`

```
batch_queue.py        Persistent queue + watch folder + scheduler
batch_tab.py          UI for the Batch tab
annotated_export.py   Render an annotated MP4 from track data
lp_blur.py            License-plate detection + blur (optional model)
analytics.py          Heatmap, OD matrix, time-series, speeds, direction
analytics_tab.py      UI for the Analytics tab
training.py           COCO/YOLO dataset build + training subprocess
training_tab.py       Active-learning queue + train UI
model_compare.py      Run two YOLO models, diff stats
models_tab.py         Models list + A/B compare UI
alerts.py             Stateful rule engine (loiter / wrong-way / speed)
live_stream.py        RTSP/webcam capture + detect + alert worker
live_tab.py           Live stream UI
search_dialog.py      Cross-session track search dialog
timeline_minimap.py   Click-to-seek timeline strip with low-conf hot zones
```

### Edited

```
main_window.py        Wires every new tab + Cross-session Search action,
                      shutdown for batch queue
review_window.py      Adds the timeline minimap above the frame bar,
                      keeps it in sync with frame nav and edits
dialogs.py            Class-change dialog now also picks a subclass
```

## Architecture notes

- **Existing patterns preserved.** Every new background job is a
  `QThread` subclass with `progress` / `finished` / `error` signals
  exactly like `processing.ProcessingWorker`. UI tabs follow the
  `__init__(data_manager)` → `refresh()` shape so `MainWindow` can
  refresh them uniformly on mode switch / review close.
- **No data-format breakage.** Tracks/corrections JSON is unchanged;
  the optional `subclass` field is additive and ignored by older code.
  Annotated export and analytics consume `corrections` first, falling
  back to raw `tracks` (same priority as the review window).
- **NAS-aware.** Everything goes through `DataManager`, so all new
  features automatically work in NAS mode (corrections/exports land
  under `_cctv_processing/` on the share).

## Feature deep-dive

### 1. Batch tab — persistent queue + watch folder
- File: `batch_queue.py` (`BatchQueueManager`, `BatchQueueStore`,
  `WatchFolderWorker`).
- Persisted to `~/Documents/CCTV-YOLO/config/batch_queue.json`.
- Resume-on-crash: anything marked `processing` at startup is reset to
  `queued` and resumes automatically.
- Priority sort + `scheduled_at` (ISO datetime) for "process at 2am
  when GPU is free" workflows.
- Watch folder polls a directory recursively; stable-size check
  prevents queueing files mid-copy.
- Drag in a folder via "Add Folder (recursive)…" to queue every video
  underneath it.

### 2. Annotated video export
- `annotated_export.py::annotate_video` writes an MP4 with bounding
  boxes, ROI overlays, class labels, per-frame counter, and
  per-class/per-ROI HUD.
- Optional `blur_lp=True` → calls `LicensePlateBlurrer`.
  - Drop a plate detection `.pt` at
    `~/Documents/CCTV-YOLO/models/license_plate.pt`
    (or set `CCTV_YOLO_LP_MODEL`) and it'll be picked up.
  - With no plate model present, falls back to a heavy gaussian blur
    on the bottom 25% of each vehicle bbox so privacy still works.

### 3. Analytics module
All pure functions — feed them a track dict, get back a file path or
data structure. Each is also exposed in the Analytics tab UI:

- `render_heatmap` — accumulates bbox centers into a gaussian density
  map and blends onto a representative frame. JET colormap.
- `origin_destination_matrix` — for each track, picks first ROI it
  enters as origin and last ROI it leaves as destination; returns the
  full square matrix plus totals.
- `time_series_csv` — buckets by configurable seconds-per-bucket,
  emits per-class and per-ROI columns. Drop into Excel and you have a
  time-of-day histogram.
- `estimate_speeds` — needs `pixels_per_meter`. Returns avg + peak
  mph/kph per track with outlier trimming.
- `direction_of_travel` — N/S/E/W classification per ROI based on
  smoothed entry-vector direction.

### 4. Training + active learning
- `session_review_priority` — single track-data dict in, score out.
  Lower score = more interesting to review (low confidence, lots of
  short fragments, lots of `needs_review`-flagged tracks).
- `rank_sessions_by_uncertainty` — applies that to every uncorrected
  session. The Training tab's top table shows them sorted lowest-score
  first.
- `build_yolo_dataset` — walks every session that has corrections,
  samples frames (configurable stride), writes a YOLO-format dataset
  + `data.yaml` to `~/Documents/CCTV-YOLO/training/ds_<timestamp>/`.
- `TrainingWorker` shells out to `python -m ultralytics detect train`
  with the dataset, streams stdout into the UI, and copies the
  resulting `best.pt` into `~/Documents/CCTV-YOLO/models/` as
  `trained_<timestamp>.pt`. The Models picker in **Preprocessing**,
  **Live**, and **Models** tabs picks it up automatically.

### 5. Models tab — A/B comparison
- `model_compare.run_model_track` runs `yolo.track` end-to-end and
  reports `total_tracks`, `total_detections`, `mean_conf`,
  `median_track_length`, and `by_class` counts.
- The tab runs A then B sequentially with two progress bars and shows
  a Δ column highlighted green/red. Use stride > 1 to speed up
  comparison runs on long videos.

### 6. Live tab + alerts
- `live_stream.LiveStreamWorker` opens any cv2-compatible source
  (RTSP URL, webcam index, file path), throttles to a max FPS, runs
  `model.track` per frame with `persist=True`, and emits a `QImage`
  for display + `dict` of stats.
- `alerts.AlertEngine` — stateful, frame-fed rule engine:
  - **loiter** — track present ≥ N seconds.
  - **wrong-way** — smoothed dx vector beyond a threshold (negative
    threshold catches "shouldn't be moving left").
  - **speed** — pixel-velocity cap.
  - One-shot: each (track_id, rule) only fires once.
- The frame view burns red `! LOITER` etc. labels for visibility; a
  side panel logs every alert with timestamp.

### 7. Cross-session search
- `File → Cross-session Search…` (Ctrl+F) opens a modal dialog.
- Filters: class, ROI name (matches across sessions), min/max avg
  confidence, min track length, "needs-review only".
- Double-click a result row → opens the review window for that
  session.

### 8. Timeline minimap
- `timeline_minimap.TimelineMinimap` is a self-contained QWidget. The
  review window mounts it just above the frame controls.
- Per-track horizontal lanes (color = vehicle class, alpha = avg
  confidence).
- Bottom strip shows low-confidence detection density.
- Vertical accent lines mark interpolated/edited frames.
- Click anywhere to seek; live playhead tracks the current frame.

### 9. Finer vehicle subclasses
- `dialogs.VEHICLE_SUBCLASSES` maps each primary class to a list of
  subclass options (sedan, SUV, pickup, box-truck, semi, etc.).
- `ClassChangeDialog` now has a second dropdown that re-populates
  when the primary class changes.
- Subclass is stored on the track as `track["subclass"]`. Annotated
  export and search both surface it. Older sessions without a
  subclass still work — empty/missing means "unspecified".

## Verification

Smoke-tested in this build:

```text
$ python3 ast-parse cctv_yolo/*.py        → 34/34 OK
$ python3 -c "import every new module"    → all imports succeed
$ QT_QPA_PLATFORM=offscreen python3 main  → MainWindow builds, 8 tabs
$ analytics OD/time-series/speed/direction synthetic test  → correct
$ AlertEngine.loiter rule fires after threshold            → correct
$ training.session_review_priority on synthetic data       → correct
```

Real-data workflows (training run, RTSP stream, GPU-accelerated batch
jobs) need the corresponding hardware/inputs to fully exercise — the
code path is in place but ground-truth verification on actual hardware
is your call.

## Not in this drop (future work)

- **Per-reviewer attribution / multi-reviewer.** The plumbing is
  there (subclass/needs_review fields), but a "current reviewer"
  signed-in identity isn't yet — corrections still write as a single
  flat JSON file.
- **Live alerts to webhooks (Slack/email).** Alert engine emits
  `Alert` events into the UI; sending them to external systems is a
  one-function-call extension on `LiveTab._on_alert` once you decide
  where they go.
- **Re-process all sessions with a new model.** The Models tab can
  compare two models on a single video; "rerun every session against
  trained_X.pt" is a natural follow-up button on top of the Batch
  queue.

## Keyboard shortcuts added

| Shortcut | Action |
|---|---|
| Ctrl+4 | Switch to Batch tab |
| Ctrl+5 | Switch to Analytics tab |
| Ctrl+6 | Switch to Training tab |
| Ctrl+7 | Switch to Models tab |
| Ctrl+8 | Switch to Live tab |
| Ctrl+F | Cross-session search dialog |

(Existing Ctrl+1/2/3 still target Preprocessing/Correction/Performance,
adjusted for the new tab order.)

## Data-directory layout (after this update)

```
~/Documents/CCTV-YOLO/
├── data/
│   ├── videos/
│   ├── tracks/
│   ├── corrections/
│   └── exports/
│       └── <session_id>/
│           ├── coco_annotations.json
│           ├── annotations.json
│           ├── labeled/                 (existing)
│           ├── <session>_annotated.mp4  ← NEW
│           ├── <session>_heatmap.png    ← NEW
│           ├── <session>_od_matrix.csv  ← NEW
│           ├── <session>_timeseries.csv ← NEW
│           └── <session>_speeds.csv     ← NEW
├── models/
│   ├── yolov8m.pt                       (existing)
│   ├── trained_<timestamp>.pt           ← NEW (training output)
│   └── license_plate.pt                 ← NEW (optional, you provide)
├── training/                            ← NEW
│   └── ds_<timestamp>/
│       ├── data.yaml
│       ├── images/{train,val}/
│       ├── labels/{train,val}/
│       └── runs/<run-name>/weights/best.pt
└── config/
    ├── nas.json
    ├── model_config.json
    ├── processing_rois.json
    ├── global_roi.json
    └── batch_queue.json                 ← NEW (persistent queue)
```
