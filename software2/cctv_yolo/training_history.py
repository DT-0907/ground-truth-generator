"""
Training-history helpers — dataset diffing, mAP extraction, and model-vs-model
comparison summaries used by the J7 "Promote?" prompt.

These are intentionally tiny pure functions so they're easy to unit-test and
re-use from the Performance tab if needed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------

def dataset_summary(dataset_dir: Path) -> dict:
    """Cheap-to-compute summary of a YOLO dataset folder.

    Returns ``{train_images, val_images, classes, manifest, name}``.
    Missing folders simply count as zero.
    """
    dataset_dir = Path(dataset_dir)
    train_dir = dataset_dir / "images" / "train"
    val_dir = dataset_dir / "images" / "val"
    train_n = len(list(train_dir.glob("*.jpg"))) if train_dir.exists() else 0
    val_n = len(list(val_dir.glob("*.jpg"))) if val_dir.exists() else 0

    classes: list[str] = []
    yaml_path = dataset_dir / "data.yaml"
    if yaml_path.exists():
        try:
            txt = yaml_path.read_text()
            m = re.search(r"names\s*:\s*\[(.*?)\]", txt, re.DOTALL)
            if m:
                inner = m.group(1)
                classes = [
                    s.strip().strip("'").strip('"')
                    for s in inner.split(",")
                    if s.strip()
                ]
        except OSError:
            pass

    manifest: dict = {}
    mf = dataset_dir / "manifest.json"
    if mf.exists():
        try:
            manifest = json.loads(mf.read_text())
        except (json.JSONDecodeError, OSError):
            manifest = {}

    return {
        "name": dataset_dir.name,
        "path": str(dataset_dir),
        "train_images": train_n,
        "val_images": val_n,
        "classes": classes,
        "manifest": manifest,
    }


def list_datasets(training_root: Path) -> list[dict]:
    """List every ``ds_*`` folder under the training root, newest first."""
    training_root = Path(training_root)
    if not training_root.exists():
        return []
    folders = [p for p in training_root.iterdir() if p.is_dir() and p.name.startswith("ds_")]
    folders.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [dataset_summary(p) for p in folders]


# ---------------------------------------------------------------------------
# Training-run metrics
# ---------------------------------------------------------------------------

def latest_run_metrics(dataset_dir: Path) -> dict:
    """Read ``runs/<run>/results.csv`` and pull the last-row val mAP / loss.

    Ultralytics writes ``epoch,train/box_loss,...,metrics/mAP50(B),mAP50-95(B),...``
    Returns ``{epochs, map50, map5095, best_fitness}`` — missing values are ``None``.
    """
    dataset_dir = Path(dataset_dir)
    runs_dir = dataset_dir / "runs"
    if not runs_dir.exists():
        return {}
    # Pick the most recently modified run
    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not candidates:
        return {}
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    csv = candidates[0] / "results.csv"
    if not csv.exists():
        return {}
    try:
        lines = [ln.strip() for ln in csv.read_text().splitlines() if ln.strip()]
        if len(lines) < 2:
            return {}
        header = [h.strip() for h in lines[0].split(",")]
        last = [c.strip() for c in lines[-1].split(",")]
        row = dict(zip(header, last))
    except OSError:
        return {}

    def _f(key_substrs: list[str]) -> float | None:
        for k, v in row.items():
            kl = k.lower()
            if all(s in kl for s in key_substrs):
                try:
                    return float(v)
                except ValueError:
                    return None
        return None

    return {
        "epochs": int(float(row.get("epoch", 0))) if row.get("epoch") else None,
        "map50": _f(["map50"]),
        "map5095": _f(["map50-95"]),
        "box_loss": _f(["val", "box_loss"]) or _f(["box_loss"]),
    }


# ---------------------------------------------------------------------------
# Model comparison summary (used by the J7 promote prompt)
# ---------------------------------------------------------------------------

def compare_models_summary(
    current_model_path: Path | None,
    new_model_path: Path,
    new_dataset_dir: Path | None = None,
) -> dict:
    """Build a side-by-side dict for the promote prompt.

    Each side has ``{name, size_mb, modified, training_images, val_map50,
    val_map5095, epochs}``. Missing data falls back to ``None`` cleanly.
    """
    def _side(model_path: Path | None, ds_dir: Path | None) -> dict:
        if model_path is None or not Path(model_path).exists():
            return {"name": "(none)", "size_mb": None, "modified": None,
                    "training_images": None, "val_map50": None,
                    "val_map5095": None, "epochs": None}
        p = Path(model_path)
        side: dict = {
            "name": p.name,
            "size_mb": round(p.stat().st_size / (1024 * 1024), 1),
            "modified": p.stat().st_mtime,
        }
        meta = p.with_suffix(p.suffix + ".meta.json")
        if not meta.exists():
            meta = p.parent / (p.stem + ".meta.json")
        meta_d: dict = {}
        if meta.exists():
            try:
                meta_d = json.loads(meta.read_text())
            except (json.JSONDecodeError, OSError):
                meta_d = {}
        side["training_images"] = meta_d.get("training_images")
        side["val_map50"] = meta_d.get("val_map50")
        side["val_map5095"] = meta_d.get("val_map5095")
        side["epochs"] = meta_d.get("epochs")
        # If dataset dir was passed, derive missing fields from it.
        if ds_dir is not None and side["val_map50"] is None:
            metrics = latest_run_metrics(Path(ds_dir))
            side["val_map50"] = side["val_map50"] or metrics.get("map50")
            side["val_map5095"] = side["val_map5095"] or metrics.get("map5095")
            side["epochs"] = side["epochs"] or metrics.get("epochs")
            if side["training_images"] is None:
                ds = dataset_summary(Path(ds_dir))
                side["training_images"] = ds["train_images"] + ds["val_images"]
        return side

    return {
        "current": _side(current_model_path, None),
        "new": _side(new_model_path, new_dataset_dir),
    }
