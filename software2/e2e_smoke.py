#!/usr/bin/env python3
"""End-to-end integration smoke test.

Runs the cross-cutting checks the PRD calls out (Part N — N2 consistency
sweep + the most important automated bits of the canonical 20-step
workflow). Doesn't require a GUI display — uses Qt's offscreen platform.

Usage:
    QT_QPA_PLATFORM=offscreen ./build_venv/bin/python e2e_smoke.py
    # or from inside the venv:
    python e2e_smoke.py

Exits 0 on success, non-zero on failure. Safe to run repeatedly — creates
a uniquely-named test group and deletes it at the end.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
import uuid
from pathlib import Path

# Force offscreen so this can run headless / in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication


_PASS = "[ PASS ]"
_FAIL = "[ FAIL ]"
_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    tag = _PASS if ok else _FAIL
    line = f"{tag} {label}"
    if detail:
        line += f"  -- {detail}"
    print(line)
    if not ok:
        _failures.append(label)


def main() -> int:
    app = QApplication(sys.argv)

    from cctv_yolo.data_manager import DataManager
    from cctv_yolo.main_window import MainWindow
    from cctv_yolo.paths import get_data_root
    from cctv_yolo.widgets.open_location_bar import OpenLocationBar

    dm = DataManager()
    print(f"\nData root resolved to: {get_data_root()}\n")
    print("=" * 60)
    print("CCTV-YOLO end-to-end smoke test")
    print("=" * 60)
    print()

    w = MainWindow(dm)
    w.show()
    check("MainWindow constructs", w.tabs.count() == 9, f"{w.tabs.count()} tabs")

    # Set up signal sniffers
    events = {"corr": [], "grp": [], "cmp": []}
    dm.corrections_changed.connect(lambda sid: events["corr"].append(sid))
    dm.groups_changed.connect(lambda: events["grp"].append("fired"))
    w.training_tab.compare_models_requested.connect(
        lambda a, b: events["cmp"].append((a, b))
    )

    # 1. corrections_changed signal fires on save
    sessions = dm.get_sessions()
    if not sessions:
        print("\n(no sessions in data folder; skipping data-touching checks)")
        return 0

    sid_with_corrections = next(
        (s["id"] for s in sessions if s.get("has_corrections")), None
    )
    if sid_with_corrections:
        data = dm.load_session_data(sid_with_corrections)
        dm.save_corrections(sid_with_corrections, data)
        check("corrections_changed fires on save",
              events["corr"] == [sid_with_corrections],
              f"received: {events['corr']}")

        # Atomic backup + _version field
        bak = dm.corrections_dir / ".bak"
        n_bak = len(list(bak.glob(f"{sid_with_corrections}-*.json"))) if bak.exists() else 0
        check("Atomic save backup rotation",
              bak.exists() and n_bak >= 1,
              f"bak/ exists={bak.exists()}, backups={n_bak}")
        check("_version field added on save",
              dm.load_corrections(sid_with_corrections).get("_version") == 2,
              f"version={dm.load_corrections(sid_with_corrections).get('_version')}")

    # 2. Groups end-to-end (with unique name to avoid stale collisions)
    test_name = f"_E2E_{uuid.uuid4().hex[:6]}"
    gid = dm.create_group(test_name, color="#982598")
    member_ids = [s["id"] for s in sessions[:2]]
    if member_ids:
        dm.add_to_group(gid, member_ids)
    check("Groups create + add + groups_changed",
          len(events["grp"]) >= 2 and len(dm.get_sessions_in_group(gid)) == len(member_ids),
          f"groups_changed fired {len(events['grp'])}x, members={len(dm.get_sessions_in_group(gid))}")

    # Cross-tab visibility
    w.performance_tab._populate_groups()
    perf_names = [w.performance_tab.group_combo.itemText(i)
                  for i in range(w.performance_tab.group_combo.count())]
    check("Performance tab sees new group",
          any(test_name in n for n in perf_names),
          f"combo: {perf_names}")

    w.analytics_tab._refresh_groups()
    ana_names = [w.analytics_tab.group_combo.itemText(i)
                 for i in range(w.analytics_tab.group_combo.count())]
    check("Analytics tab sees new group",
          any(test_name in n for n in ana_names),
          f"combo: {ana_names}")

    # 3. Iterative training loop
    unused = dm.list_unused_corrections()
    check("list_unused_corrections returns list",
          isinstance(unused, list),
          f"{len(unused)} sessions")

    # 4. PRD J7 — Training -> Performance handoff
    starting_tab = w.tabs.currentIndex()
    w.training_tab.compare_models_requested.emit("yolov8n.pt", "yolov8m.pt")
    check("J7 signal received",
          events["cmp"] == [("yolov8n.pt", "yolov8m.pt")],
          f"signal: {events['cmp']}")
    check("J7 tab switched to Performance",
          w.tabs.currentWidget() is w.performance_tab)
    check("J7 model A pre-filled",
          w.performance_tab.compare_model_a.currentText() == "yolov8n.pt")
    check("J7 model B pre-filled",
          w.performance_tab.compare_model_b.currentText() == "yolov8m.pt")

    # 5. PRD I2 — Insights has 4 sub-tabs
    sub_titles = [w.insights_tab.tabs.tabText(i)
                  for i in range(w.insights_tab.tabs.count())]
    expected_sub = {"Session", "Group", "Dataset", "Multi"}
    have_sub = {t for t in sub_titles if any(e in t for e in expected_sub)}
    check("Insights has 4 sub-tabs (Session/Group/Dataset/Multi)",
          len(have_sub) == 4,
          f"found: {sub_titles}")

    # 6. PRD C12 — OpenLocationBar in every tab
    ol_counts = {}
    for tab_name in ('preprocessing_tab', 'batch_tab', 'correction_tab',
                      'performance_tab', 'analytics_tab', 'insights_tab',
                      'training_tab', 'models_tab', 'live_tab'):
        ol_counts[tab_name] = len(getattr(w, tab_name).findChildren(OpenLocationBar))
    check("OpenLocationBar in every tab",
          all(v >= 1 for v in ol_counts.values()),
          f"per-tab: {ol_counts}")

    # 7. Model dropdown reflects only installed
    installed = set(dm.list_models())
    prep_items = {w.preprocessing_tab.model_combo.itemText(i)
                  for i in range(w.preprocessing_tab.model_combo.count())}
    check("Preprocessing dropdown matches installed models",
          installed == prep_items or (not installed and "(no models installed)" in prep_items),
          f"installed={installed}, dropdown={prep_items}")

    # 8. ROI propagation (corrections JSON -> analytics)
    if sid_with_corrections:
        data = dm.load_session_data(sid_with_corrections)
        rois_in = data.get("rois", [])
        from cctv_yolo import analytics as A
        od = A.origin_destination_matrix(data)
        rois_out = od.get("rois", [])
        check("ROI propagation corrections -> analytics OD",
              (len(rois_in) == 0 and len(rois_out) == 0) or len(rois_out) >= 1,
              f"in={len(rois_in)}, out={len(rois_out)}")

    # 9. Atomic write helper used in data_manager
    src_count = Path("cctv_yolo/data_manager.py").read_text().count("_atomic_write_json")
    check("_atomic_write_json used widely in data_manager",
          src_count >= 10,
          f"{src_count} references")

    # 10. Duplicate-group-name auto-suffix
    test_name2 = f"_E2E_{uuid.uuid4().hex[:6]}"
    g1 = dm.create_group(test_name2)
    g2 = dm.create_group(test_name2)
    g2_name = dm.get_group(g2)["name"]
    check("Duplicate-group-name auto-suffix",
          g2_name == f"{test_name2} (2)",
          f"second group named: {g2_name}")
    dm.delete_group(g1)
    dm.delete_group(g2)

    # 11. Safe class-color lookup (regression: KeyError 'unknown' bug)
    from cctv_yolo.theme import class_color, CLASS_COLORS
    check("class_color('unknown') doesn't raise",
          class_color("unknown") == CLASS_COLORS["unknown"])
    check("class_color(None) doesn't raise",
          class_color(None) == CLASS_COLORS["unknown"])
    check("class_color('totally_unknown_class') doesn't raise",
          class_color("zebra") == CLASS_COLORS["unknown"])
    check("class_color('CAR') (wrong case) resolves",
          class_color("CAR") == CLASS_COLORS["car"])

    # 12. Required deps importable (regression: matplotlib missing on Windows)
    required = ("ultralytics", "matplotlib", "matplotlib.pyplot",
                "yaml", "PIL", "cv2", "torch", "PySide6")
    missing_required = []
    for dep in required:
        try:
            __import__(dep)
        except ImportError as e:
            missing_required.append(f"{dep} ({e})")
    check("Required deps importable",
          not missing_required,
          ", ".join(missing_required) or "all present")

    # 13. Recommended deps (Ultralytics transitive — non-fatal on dev
    # machines without a full venv re-install, but ALL must be present
    # in PyInstaller-bundled builds)
    recommended = ("scipy", "pandas", "psutil", "seaborn")
    missing_rec = []
    for dep in recommended:
        try:
            __import__(dep)
        except ImportError:
            missing_rec.append(dep)
    if missing_rec:
        # Print as a warning, not a failure — local venv may not have
        # them yet but the build script ensures the .exe / .app does.
        print(f"[ WARN ] Recommended deps missing locally: {missing_rec}")
        print(f"         (Pip install them or re-run build_*.sh / .bat;")
        print(f"         the spec ensures the frozen build has all of them.)")

    # 13. processor.py imports cleanly (catches Ultralytics submodule
    # import-cascade regressions like the matplotlib one)
    try:
        from cctv_yolo import processor  # noqa: F401
        proc_ok, proc_err = True, ""
    except Exception as e:
        proc_ok, proc_err = False, str(e)
    check("cctv_yolo.processor imports cleanly", proc_ok, proc_err)

    # 14. Every text-mode open() in cctv_yolo/ specifies encoding=
    # (regression: on Windows, default cp1252 bombs on UTF-8 JSON)
    import re
    bad_open: list[str] = []
    open_pat = re.compile(
        r'open\([^)]*?,\s*["\']([rwa]\+?)["\'](?!\s*,\s*encoding)',
    )
    for py in Path("cctv_yolo").rglob("*.py"):
        if "__pycache__" in str(py):
            continue
        text = py.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if "encoding=" in line:
                continue
            if open_pat.search(line):
                bad_open.append(f"{py}:{i}")
    check("Every text-mode open() specifies encoding=",
          not bad_open,
          (f"{len(bad_open)} sites missing encoding=: "
           + ", ".join(bad_open[:3]) + ("..." if len(bad_open) > 3 else ""))
          if bad_open else "all good")

    # Cleanup
    dm.delete_group(gid)

    # ----------------------------------------------------------------
    print()
    print("=" * 60)
    if _failures:
        print(f"FAIL — {len(_failures)} check(s) failed:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("PASS — every check succeeded.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
