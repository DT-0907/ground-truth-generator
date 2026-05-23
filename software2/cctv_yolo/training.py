"""
Training + active-learning loop.

Pipeline:
  1. Aggregate every session that has corrections.
  2. Sample annotated frames (every N frames per track) into a YOLO
     dataset on disk: ``{data_root}/training/<run-id>/{images,labels}/...``.
  3. Write ``data.yaml`` with class names.
  4. Spawn `yolo detect train` (Ultralytics) as a subprocess and stream
     stdout into the UI.
  5. Save the best.pt as ``models/<run-id>.pt`` and bump
     ``last_model``.

Active-learning queue ranks sessions by mean confidence of tracks that
have NOT been corrected yet — surface those first for review.
"""
from __future__ import annotations
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
from PySide6.QtCore import QObject, QThread, Signal


# Fallback class order used when nothing has been corrected yet.
_COMMON_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]


def discover_classes(data_manager) -> list[str]:
    """PRD J9-1 — return the union of class names seen across every
    correction (falls back to ``_COMMON_CLASSES`` if none exist)."""
    found: list[str] = []
    seen: set[str] = set()
    for c in _COMMON_CLASSES:
        if c not in seen:
            found.append(c)
            seen.add(c)
    try:
        sessions = data_manager.get_sessions()
    except Exception:
        return found
    for s in sessions:
        if not s.get("has_corrections"):
            continue
        data = data_manager.load_corrections(s["id"])
        if not data:
            continue
        for tr in data.get("tracks", []):
            cls = (tr.get("class") or "vehicle").strip()
            if cls and cls not in seen:
                found.append(cls)
                seen.add(cls)
    return found


# ---------------------------------------------------------------------------
# Active-learning queue
# ---------------------------------------------------------------------------

def session_review_priority(track_data: dict) -> dict:
    """Lower score = higher review priority.

    Combines:
    * mean confidence of uncorrected tracks (lower → more uncertain)
    * fraction of short tracks (which are usually fragments)
    * fraction flagged needs_review
    """
    tracks = track_data.get("tracks", [])
    if not tracks:
        # All keys present so callers can always look them up — empty session
        # is high-priority for review (score 1.0).
        return {
            "score": 1.0,
            "tracks": 0,
            "low_conf": 0,
            "short": 0,
            "needs_review": 0,
            "mean_conf": 0.0,
        }

    confs = []
    short = 0
    needs = 0
    for t in tracks:
        confs.append(t.get("avg_confidence", 0.0))
        if t.get("needs_review"):
            needs += 1
        if (t.get("end_frame", 0) - t.get("start_frame", 0)) < 10:
            short += 1

    mean_conf = sum(confs) / max(1, len(confs))
    low = sum(1 for c in confs if c < 0.4)
    # Score: lower is more interesting to review
    score = mean_conf - 0.05 * (short / max(1, len(tracks))) \
            - 0.1 * (needs / max(1, len(tracks)))

    return {
        "score": round(score, 4),
        "tracks": len(tracks),
        "low_conf": low,
        "short": short,
        "needs_review": needs,
        "mean_conf": round(mean_conf, 3),
    }


def rank_sessions_by_uncertainty(data_manager) -> list[dict]:
    """Return sessions ordered by review priority (most uncertain first)."""
    sessions = data_manager.get_sessions()
    out = []
    for s in sessions:
        if s.get("has_corrections"):
            continue  # already corrected
        data = data_manager.load_tracks(s["id"])
        if not data:
            continue
        prio = session_review_priority(data)
        out.append({
            "session_id": s["id"],
            "video_name": s["video_name"],
            **prio,
        })
    out.sort(key=lambda x: x["score"])
    return out


# ---------------------------------------------------------------------------
# Dataset builder — corrections -> YOLO format
# ---------------------------------------------------------------------------

def build_yolo_dataset(
    data_manager,
    output_root: Path,
    sample_every_n: int = 5,
    val_split: float = 0.1,
    progress_callback=None,
    restrict_session_ids: list[str] | None = None,
    roi: dict | None = None,
) -> dict:
    """Walk every session that has corrections, sample frames, write
    YOLO-formatted training data.

    PRD J2: ``restrict_session_ids`` filters to a specific set (used by
    'Build from unused corrections').

    Returns
    -------
    dict
        ``{"images": int, "labels": int, "classes": [...], "yaml_path": str}``
    """
    output_root = Path(output_root)
    images_train = output_root / "images" / "train"
    images_val = output_root / "images" / "val"
    labels_train = output_root / "labels" / "train"
    labels_val = output_root / "labels" / "val"
    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    # PRD J9-1 — discover classes dynamically across every correction, so
    # the index stays stable as new classes appear.
    class_index: dict[str, int] = {}
    for c in discover_classes(data_manager):
        class_index[c] = len(class_index)

    sessions = data_manager.get_sessions()
    corrected = [s for s in sessions if s.get("has_corrections")]
    if restrict_session_ids is not None:
        keep = set(restrict_session_ids)
        corrected = [s for s in corrected if s["id"] in keep]
    total = max(1, len(corrected))

    images_written = 0
    labels_written = 0

    # Deterministic-ish split: every 10th session goes to val
    for idx, s in enumerate(corrected):
        sid = s["id"]
        data = data_manager.load_corrections(sid)
        if not data:
            continue
        video_path = data_manager.get_video_path(sid)
        if not video_path or not video_path.exists():
            print(f"[training] skip {sid}: video missing")
            continue

        # Index detections by frame
        per_frame: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
        # PRD J8 — optional ROI filter: only keep detections whose bbox
        # center falls inside the ROI.
        if roi is not None:
            try:
                from cctv_yolo.processor import _bbox_center_in_roi
            except Exception:
                _bbox_center_in_roi = None
        else:
            _bbox_center_in_roi = None
        for tr in data.get("tracks", []):
            cls = tr.get("class") or "vehicle"
            if cls not in class_index:
                class_index[cls] = len(class_index)
            cls_idx = class_index[cls]
            for fd in tr.get("frames", []):
                if fd.get("interpolated"):
                    continue
                bbox = fd["bbox"]
                if roi is not None and _bbox_center_in_roi is not None:
                    if not _bbox_center_in_roi(bbox, roi):
                        continue
                per_frame[fd["frame"]].append((cls_idx, bbox))

        if not per_frame:
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[training] skip {sid}: cannot open video")
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if w <= 0 or h <= 0:
            cap.release()
            continue

        sorted_frames = sorted(per_frame.keys())[::max(1, sample_every_n)]
        is_val = (idx % 10) == 0

        for fnum in sorted_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
            ret, frame = cap.read()
            if not ret:
                continue
            stem = f"{sid}_f{fnum:06d}"
            img_dir = images_val if is_val else images_train
            lbl_dir = labels_val if is_val else labels_train
            img_path = img_dir / f"{stem}.jpg"
            lbl_path = lbl_dir / f"{stem}.txt"

            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            images_written += 1

            with open(lbl_path, "w") as f:
                for cls_idx, bbox in per_frame[fnum]:
                    x1, y1, x2, y2 = bbox
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    if bw <= 0 or bh <= 0:
                        continue
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    bw = max(0.0, min(1.0, bw))
                    bh = max(0.0, min(1.0, bh))
                    f.write(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    labels_written += 1
        cap.release()

        if progress_callback:
            progress_callback(int((idx + 1) / total * 100))

    # Write data.yaml
    classes_sorted = sorted(class_index.items(), key=lambda x: x[1])
    class_names = [c for c, _ in classes_sorted]
    yaml_path = output_root / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_root.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names: [" + ", ".join(f"'{c}'" for c in class_names) + "]\n")

    return {
        "images": images_written,
        "labels": labels_written,
        "classes": class_names,
        "yaml_path": str(yaml_path),
        "root": str(output_root),
        "corrected_sessions": len(corrected),
    }


# ---------------------------------------------------------------------------
# Training subprocess
# ---------------------------------------------------------------------------

class TrainingWorker(QThread):
    """Run `yolo detect train` as a subprocess, stream lines to UI."""

    log_line = Signal(str)
    progress = Signal(int)         # 0..100 (best-effort, parsed from stdout)
    finished_ok = Signal(str)      # final model path
    failed = Signal(str)

    def __init__(
        self,
        data_yaml: str,
        base_model: str = "yolov8n.pt",
        epochs: int = 30,
        imgsz: int = 640,
        batch: int = 16,
        run_name: str = "",
        models_dir: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.data_yaml = data_yaml
        self.base_model = base_model
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.run_name = run_name or dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.models_dir = Path(models_dir) if models_dir else None
        self._proc: Optional[subprocess.Popen] = None
        self._stop = False

    def stop(self):
        self._stop = True
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def _collect_meta(self, src: Path, dest: Path, stamp: str) -> dict:
        """Assemble provenance metadata for the trained model.

        Pulls ``best_val_map`` out of the Ultralytics results CSV when
        available (sibling of best.pt under ``runs/<name>/``).
        """
        dataset_root = Path(self.data_yaml).parent
        # The dataset folder name is also the build_id used by training_tab.
        dataset_id = dataset_root.name

        best_val_map: float | None = None
        try:
            results_csv = src.parent.parent / "results.csv"  # ../weights/best.pt → ../results.csv
            if results_csv.exists():
                lines = results_csv.read_text().strip().splitlines()
                if len(lines) >= 2:
                    headers = [h.strip() for h in lines[0].split(",")]
                    # Prefer mAP50-95 over mAP50 if both exist
                    for key in ("metrics/mAP50-95(B)", "metrics/mAP50(B)",
                                "metrics/mAP50-95", "metrics/mAP50"):
                        if key in headers:
                            col = headers.index(key)
                            # Take the last (final-epoch) row
                            last = [c.strip() for c in lines[-1].split(",")]
                            if col < len(last):
                                try:
                                    best_val_map = round(float(last[col]), 4)
                                except ValueError:
                                    pass
                            break
        except Exception:
            pass

        return {
            "base_model": self.base_model,
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "dataset_id": dataset_id,
            "build_id": dataset_id,
            "run_name": self.run_name,
            "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
            "best_val_map": best_val_map,
        }

    def run(self):
        try:
            cmd = [
                sys.executable, "-m", "ultralytics",
                "detect", "train",
                f"data={self.data_yaml}",
                f"model={self.base_model}",
                f"epochs={self.epochs}",
                f"imgsz={self.imgsz}",
                f"batch={self.batch}",
                f"name={self.run_name}",
                f"project={Path(self.data_yaml).parent / 'runs'}",
                "verbose=True",
                "exist_ok=True",
            ]
            self.log_line.emit("$ " + " ".join(cmd))
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            epoch_re = re.compile(r"^\s*(\d+)/(\d+)\b")
            best_path = None
            for line in self._proc.stdout:
                if self._stop:
                    break
                line = line.rstrip()
                if not line:
                    continue
                self.log_line.emit(line)

                m = epoch_re.match(line)
                if m:
                    cur, tot = int(m.group(1)), int(m.group(2))
                    if tot > 0:
                        self.progress.emit(int(cur / tot * 100))

                # Catch the "best.pt" path from the summary
                if "best.pt" in line:
                    for tok in line.split():
                        if tok.endswith("best.pt"):
                            best_path = tok
                            break

            self._proc.wait()
            if self._proc.returncode != 0 and not self._stop:
                self.failed.emit(f"yolo train exited with code {self._proc.returncode}")
                return
            if self._stop:
                self.failed.emit("Training stopped by user")
                return

            # Find best.pt in the runs/<name>/weights dir
            runs_dir = Path(self.data_yaml).parent / "runs" / self.run_name / "weights"
            if best_path and Path(best_path).exists():
                src = Path(best_path)
            elif runs_dir.exists():
                src = runs_dir / "best.pt"
            else:
                self.failed.emit("Could not find best.pt after training")
                return

            if not src.exists():
                self.failed.emit(f"best.pt missing at {src}")
                return

            # Copy into models dir with a versioned name
            if self.models_dir is None:
                from cctv_yolo.paths import get_models_dir
                self.models_dir = get_models_dir()
            self.models_dir.mkdir(parents=True, exist_ok=True)
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = self.models_dir / f"trained_{stamp}.pt"
            shutil.copy2(src, dest)
            self.log_line.emit(f"Saved trained model: {dest}")

            # PRD K2-4 — write sidecar provenance JSON so the Models tab can
            # surface base_model / epochs / dataset_id / best_val_map etc.
            try:
                meta = self._collect_meta(src, dest, stamp)
                meta_path = dest.with_suffix(".meta.json")
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
                self.log_line.emit(f"Wrote sidecar metadata: {meta_path.name}")
            except Exception as e:
                # Sidecar failure shouldn't block the model from being usable
                self.log_line.emit(f"Warning: couldn't write sidecar meta: {e}")

            self.finished_ok.emit(str(dest))
        except Exception as e:
            import traceback
            self.log_line.emit(traceback.format_exc())
            self.failed.emit(str(e))


# ---------------------------------------------------------------------------
# Dataset building worker (separate so UI doesn't freeze)
# ---------------------------------------------------------------------------

class DatasetBuildWorker(QThread):
    progress = Signal(int)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, data_manager, output_root: Path,
                 sample_every_n: int = 5, val_split: float = 0.1,
                 roi: dict | None = None, parent=None):
        super().__init__(parent)
        self.dm = data_manager
        self.output_root = output_root
        self.sample_every_n = sample_every_n
        self.val_split = val_split
        self.roi = roi
        # PRD J2: set by training_tab._build_dataset(restrict_to=...) before .start()
        self._restrict_session_ids: list[str] | None = None

    def run(self):
        try:
            stats = build_yolo_dataset(
                self.dm,
                self.output_root,
                sample_every_n=self.sample_every_n,
                val_split=self.val_split,
                progress_callback=lambda p: self.progress.emit(p),
                restrict_session_ids=self._restrict_session_ids,
                roi=self.roi,
            )
            # PRD J4b — write a manifest so combined datasets and ROI builds
            # are self-describing on disk.
            try:
                manifest = {
                    "build_type": "single",
                    "built_at": dt.datetime.now().isoformat(),
                    "sample_every_n": self.sample_every_n,
                    "val_split": self.val_split,
                    "roi": self.roi,
                    "class_names": stats.get("classes", []),
                    "image_counts": {"total": stats.get("images", 0)},
                    "source_sessions": self._restrict_session_ids or "all_corrected",
                }
                (Path(self.output_root) / "manifest.json").write_text(
                    json.dumps(manifest, indent=2)
                )
            except OSError:
                pass
            self.finished_ok.emit(stats)
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# PRD J4b — combine existing datasets
# ---------------------------------------------------------------------------

def combine_datasets(
    parent_dataset_paths: list[Path],
    output_path: Path,
    val_split: float = 0.1,
) -> Path:
    """Merge multiple YOLO datasets into one.

    * Hardlinks files on POSIX, falls back to copy on Windows or cross-device.
    * Re-indexes class IDs into the union class list, rewriting label files.
    * On filename collisions across parents, files are prefixed by parent id.
    * Writes ``manifest.json`` and a unified ``data.yaml``.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    images_train = output_path / "images" / "train"
    images_val = output_path / "images" / "val"
    labels_train = output_path / "labels" / "train"
    labels_val = output_path / "labels" / "val"
    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    parents = [Path(p) for p in parent_dataset_paths]

    # 1) Build the union class list & per-parent remap.
    union: list[str] = []
    seen: set[str] = set()
    parent_classes: dict[str, list[str]] = {}
    for p in parents:
        from cctv_yolo.training_history import dataset_summary
        info = dataset_summary(p)
        parent_classes[p.name] = info["classes"]
        for c in info["classes"]:
            if c not in seen:
                union.append(c)
                seen.add(c)
    if not union:
        union = list(_COMMON_CLASSES)

    class_remap: dict[str, dict[int, int]] = {}
    for parent_name, classes in parent_classes.items():
        class_remap[parent_name] = {old: union.index(c) for old, c in enumerate(classes) if c in union}

    # 2) Walk each parent and stage images + remapped labels.
    used_names: set[str] = set()
    image_counts = {"train": 0, "val": 0}

    def _link_or_copy(src: Path, dest: Path) -> None:
        if dest.exists():
            return
        try:
            os.link(src, dest)
        except (OSError, NotImplementedError):
            shutil.copy2(src, dest)

    source_sessions: set[str] = set()

    for parent in parents:
        remap = class_remap.get(parent.name, {})
        for split in ("train", "val"):
            src_img = parent / "images" / split
            src_lbl = parent / "labels" / split
            if not src_img.exists():
                continue
            dest_img = images_train if split == "train" else images_val
            dest_lbl = labels_train if split == "train" else labels_val
            for img in src_img.iterdir():
                if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                name = img.name
                if name in used_names:
                    name = f"{parent.name}__{name}"
                used_names.add(name)
                # Track session ID (everything before "_f<frame>")
                stem = img.stem
                m = re.match(r"(.+)_f\d+$", stem)
                if m:
                    source_sessions.add(m.group(1))
                _link_or_copy(img, dest_img / name)
                # Rewrite label with remapped class indices.
                lbl_src = src_lbl / (img.stem + ".txt")
                if lbl_src.exists():
                    new_stem = Path(name).stem
                    out_lbl = dest_lbl / (new_stem + ".txt")
                    try:
                        lines = []
                        for raw in lbl_src.read_text().splitlines():
                            parts = raw.strip().split()
                            if len(parts) < 5:
                                continue
                            try:
                                old_idx = int(parts[0])
                            except ValueError:
                                continue
                            new_idx = remap.get(old_idx, old_idx)
                            lines.append(" ".join([str(new_idx)] + parts[1:]))
                        out_lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
                    except OSError:
                        pass
                image_counts[split] += 1

    # 3) Write data.yaml (union classes).
    yaml_path = output_path / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_path.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(union)}\n")
        f.write("names: [" + ", ".join(f"'{c}'" for c in union) + "]\n")

    # 4) Manifest.
    manifest = {
        "build_type": "combined",
        "built_at": dt.datetime.now().isoformat(),
        "parents": [p.name for p in parents],
        "class_names": union,
        "class_remap": {k: {str(o): n for o, n in v.items()} for k, v in class_remap.items()},
        "image_counts": image_counts,
        "source_sessions": sorted(source_sessions),
        "val_split": val_split,
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return output_path
