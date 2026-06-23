"""Single source of truth for the app's object *class sets*.

Historically the detectable/labelable classes were hardcoded as the five COCO
vehicle types in ~10 different modules. This module replaces all of that with a
small, data-driven registry backed by ``config/class_sets.json``.

A *class set* is a named vocabulary of classes the user can detect, relabel
tracks into, train on, and see in stats. Built-in presets ship with the app and
cannot be deleted; the user can also create/edit/delete custom sets (e.g. the
FHWA 13-class vehicle scheme).

Two distinct roles, intentionally kept separate:

* **Label vocabulary** — the full list of class names (+ colors) the active set
  defines. Drives the relabel dropdown, per-class stat cards, colors, training.
* **Detection mapping** — for a *stock* COCO model, which COCO class IDs to keep
  during inference and what name to give each. Built from the active set's
  ``coco_id`` seeds. A *custom-trained* model is read directly from
  ``model.names`` instead (so a real FHWA model "just works").

Stock-model detection is necessarily coarse: COCO can't tell a 5-axle truck from
a 6-axle one, so several FHWA classes share no COCO seed. Detection emits the
nearest COCO match (truck -> "Class 5 ...") and the user refines the precise
class in Correction or by training a custom model. This is the "coarse seed +
relabel" behavior chosen at design time.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

from cctv_yolo import paths
from cctv_yolo.theme import (
    ERROR,
    OFFWHITE,
    PINK,
    PURPLE,
    ROI_COLOR_ROTATION,
    YELLOW,
)

logger = logging.getLogger(__name__)

_CONFIG_NAME = "class_sets.json"
_lock = threading.RLock()
_cache: Optional[dict] = None


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------
# Colors for the original five vehicle types match the historical
# theme.CLASS_COLORS values exactly so existing visuals don't shift.
_VEHICLE_CLASSES = [
    {"name": "car",        "color": PINK,      "coco_id": 2},
    {"name": "truck",      "color": "#C76EB1", "coco_id": 7},
    {"name": "bus",        "color": "#7A2A7A", "coco_id": 5},
    {"name": "motorcycle", "color": YELLOW,    "coco_id": 3},
    {"name": "bicycle",    "color": ERROR,     "coco_id": 1},
]

# Pedestrian == COCO "person" (class id 0). Given a distinct, palette-harmonised
# accent so it stands apart from the vehicle classes.
_PEDESTRIAN = {"name": "pedestrian", "color": "#7AC4D4", "coco_id": 0}

# FHWA 13-category vehicle classification. Only classes a stock COCO model can
# coarsely seed carry a ``coco_id`` (and each COCO id is used at most once so the
# detection map stays unambiguous). Everything else is reached by relabeling or
# a custom model.
_FHWA_13 = [
    {"name": "Class 1: Motorcycles",                         "coco_id": 3},
    {"name": "Class 2: Passenger Cars",                      "coco_id": 2},
    {"name": "Class 3: Pickups, Vans (2-axle, 4-tire)",      "coco_id": None},
    {"name": "Class 4: Buses",                               "coco_id": 5},
    {"name": "Class 5: 2-Axle, 6-Tire Single-Unit Trucks",   "coco_id": 7},
    {"name": "Class 6: 3-Axle Single-Unit Trucks",           "coco_id": None},
    {"name": "Class 7: 4+ Axle Single-Unit Trucks",          "coco_id": None},
    {"name": "Class 8: 4-or-Fewer Axle Single-Trailer",      "coco_id": None},
    {"name": "Class 9: 5-Axle Single-Trailer Trucks",        "coco_id": None},
    {"name": "Class 10: 6+ Axle Single-Trailer Trucks",      "coco_id": None},
    {"name": "Class 11: 5-or-Fewer Axle Multi-Trailer",      "coco_id": None},
    {"name": "Class 12: 6-Axle Multi-Trailer Trucks",        "coco_id": None},
    {"name": "Class 13: 7+ Axle Multi-Trailer Trucks",       "coco_id": None},
]


def _with_rotated_colors(classes: list[dict]) -> list[dict]:
    """Assign palette-harmonised colors to any class missing one."""
    out = []
    for i, c in enumerate(classes):
        c = dict(c)
        if not c.get("color"):
            c["color"] = ROI_COLOR_ROTATION[i % len(ROI_COLOR_ROTATION)]
        out.append(c)
    return out


def _builtin_sets() -> dict:
    return {
        "coco_vehicles": {
            "name": "COCO Vehicles",
            "builtin": True,
            "classes": [dict(c) for c in _VEHICLE_CLASSES],
        },
        "coco_vehicles_ped": {
            "name": "COCO Vehicles + Pedestrian",
            "builtin": True,
            "classes": [dict(c) for c in _VEHICLE_CLASSES] + [dict(_PEDESTRIAN)],
        },
        "fhwa_13": {
            "name": "FHWA 13 Classes",
            "builtin": True,
            "classes": _with_rotated_colors(_FHWA_13),
        },
    }


def _default_config() -> dict:
    return {"active": "coco_vehicles", "sets": _builtin_sets()}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _config_path() -> Path:
    return paths.get_config_dir() / _CONFIG_NAME


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _load() -> dict:
    """Load (and lazily migrate) the config, caching the result."""
    global _cache
    with _lock:
        if _cache is not None:
            return _cache
        path = _config_path()
        cfg: dict
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.exception("class_sets.json unreadable; recreating defaults")
                cfg = _default_config()
        else:
            cfg = _default_config()

        # Migration: ensure all built-in presets are present (so upgrades that
        # add a new preset show it) and the active id is valid.
        cfg.setdefault("sets", {})
        for sid, sdef in _builtin_sets().items():
            if sid not in cfg["sets"]:
                cfg["sets"][sid] = sdef
            else:
                # keep user edits but guarantee the builtin flag
                cfg["sets"][sid]["builtin"] = True
        if cfg.get("active") not in cfg["sets"]:
            cfg["active"] = "coco_vehicles"

        _cache = cfg
        # Persist any migration so the file on disk stays current.
        try:
            _atomic_write(path, cfg)
        except OSError:
            logger.warning("Couldn't persist class_sets.json migration")
        return _cache


def reload() -> None:
    """Drop the in-memory cache; next access re-reads from disk."""
    global _cache
    with _lock:
        _cache = None


def _save(cfg: dict) -> None:
    with _lock:
        _atomic_write(_config_path(), cfg)
        global _cache
        _cache = cfg


# ---------------------------------------------------------------------------
# Public read API
# ---------------------------------------------------------------------------

def list_sets() -> list[dict]:
    """Return [{id, name, builtin, classes:[...]}] for every set."""
    cfg = _load()
    out = []
    for sid, sdef in cfg["sets"].items():
        out.append({
            "id": sid,
            "name": sdef.get("name", sid),
            "builtin": bool(sdef.get("builtin")),
            "classes": [dict(c) for c in sdef.get("classes", [])],
        })
    return out


def active_id() -> str:
    return _load()["active"]


def active_set() -> dict:
    cfg = _load()
    sid = cfg["active"]
    sdef = cfg["sets"][sid]
    return {
        "id": sid,
        "name": sdef.get("name", sid),
        "builtin": bool(sdef.get("builtin")),
        "classes": [dict(c) for c in sdef.get("classes", [])],
    }


def class_names(set_id: Optional[str] = None) -> list[str]:
    """Ordered list of class names for a set (active set by default)."""
    cfg = _load()
    sid = set_id or cfg["active"]
    sdef = cfg["sets"].get(sid, {})
    return [c["name"] for c in sdef.get("classes", [])]


def color_for(class_name: Optional[str]) -> Optional[str]:
    """Return the hex color the active set assigns to ``class_name``.

    Case-insensitive. Returns None if the active set doesn't define it (callers
    fall back to their own default — see theme.class_color).
    """
    if not class_name:
        return None
    target = class_name.lower()
    for c in active_set()["classes"]:
        if c["name"].lower() == target:
            return c.get("color")
    return None


# ---------------------------------------------------------------------------
# Detection mapping
# ---------------------------------------------------------------------------

def coco_detect_map(set_id: Optional[str] = None) -> dict[int, str]:
    """``{coco_class_id: class_name}`` for stock-COCO inference.

    Built from the active set's ``coco_id`` seeds. Falls back to the base
    vehicle map if the set defines no COCO seeds at all, so a fully-custom set
    still yields *some* detections on a stock model rather than zero boxes.
    """
    cfg = _load()
    sid = set_id or cfg["active"]
    sdef = cfg["sets"].get(sid, {})
    out: dict[int, str] = {}
    for c in sdef.get("classes", []):
        cid = c.get("coco_id")
        if cid is None:
            continue
        out.setdefault(int(cid), c["name"])
    if not out:
        # Safety net: detect vehicles under their COCO names.
        return {c["coco_id"]: c["name"] for c in _VEHICLE_CLASSES}
    return out


def _looks_like_coco(names: dict) -> bool:
    """Heuristic: does this model's ``names`` dict look like stock COCO?"""
    if not names or len(names) < 80:
        return False
    try:
        return str(names.get(0)).lower() == "person" and str(names.get(2)).lower() == "car"
    except Exception:
        return False


def detect_mapping_for_model(model) -> tuple[Optional[list[int]], dict[int, str]]:
    """Return ``(classes_filter, id_to_name)`` for a loaded YOLO model.

    * Stock COCO model -> filter to the active set's COCO seeds; names from the
      active set (the coarse-seed behavior).
    * Custom-trained model -> no class filter (``None``); names read straight
      from ``model.names`` so its native classes flow through unchanged.
    """
    names = getattr(model, "names", None) or {}
    # names can be a dict or list depending on Ultralytics version.
    if isinstance(names, (list, tuple)):
        names = {i: n for i, n in enumerate(names)}
    if _looks_like_coco(names):
        m = coco_detect_map()
        return list(m.keys()), m
    return None, {int(k): str(v) for k, v in names.items()}


# ---------------------------------------------------------------------------
# Public write API (used by the Manage Class Sets UI)
# ---------------------------------------------------------------------------

def set_active(set_id: str) -> None:
    cfg = _load()
    if set_id not in cfg["sets"]:
        raise KeyError(f"Unknown class set: {set_id}")
    cfg = dict(cfg)
    cfg["active"] = set_id
    _save(cfg)


def _new_set_id(name: str, existing: dict) -> str:
    base = "".join(ch if ch.isalnum() else "_" for ch in name.strip().lower()) or "set"
    sid = base
    n = 1
    while sid in existing:
        n += 1
        sid = f"{base}_{n}"
    return sid


def create_set(name: str, classes: list[dict], activate: bool = True) -> str:
    """Create a custom set. ``classes`` is a list of {name, color?, coco_id?}."""
    cfg = _load()
    cfg = json.loads(json.dumps(cfg))  # deep copy
    sid = _new_set_id(name, cfg["sets"])
    cfg["sets"][sid] = {
        "name": name.strip() or sid,
        "builtin": False,
        "classes": _with_rotated_colors([_normalize_class(c) for c in classes]),
    }
    if activate:
        cfg["active"] = sid
    _save(cfg)
    return sid


def update_set(set_id: str, *, name: Optional[str] = None,
               classes: Optional[list[dict]] = None) -> None:
    """Update a set's name and/or class list. Built-in sets can be edited too
    (e.g. recoloring), but a guard in the UI warns before doing so."""
    cfg = _load()
    if set_id not in cfg["sets"]:
        raise KeyError(f"Unknown class set: {set_id}")
    cfg = json.loads(json.dumps(cfg))
    sdef = cfg["sets"][set_id]
    if name is not None:
        sdef["name"] = name.strip() or set_id
    if classes is not None:
        sdef["classes"] = _with_rotated_colors([_normalize_class(c) for c in classes])
    _save(cfg)


def delete_set(set_id: str) -> None:
    cfg = _load()
    if set_id not in cfg["sets"]:
        return
    if cfg["sets"][set_id].get("builtin"):
        raise ValueError("Built-in class sets cannot be deleted.")
    cfg = json.loads(json.dumps(cfg))
    del cfg["sets"][set_id]
    if cfg.get("active") == set_id:
        cfg["active"] = "coco_vehicles"
    _save(cfg)


def _normalize_class(c: dict) -> dict:
    """Coerce a user-supplied class dict into the stored shape."""
    name = str(c.get("name", "")).strip()
    out: dict = {"name": name}
    if c.get("color"):
        out["color"] = str(c["color"])
    cid = c.get("coco_id")
    out["coco_id"] = int(cid) if cid not in (None, "") else None
    return out
