"""
Headless CLI -- runs the same pipelines as the GUI from the command line.

Usage examples:

    python -m cctv_yolo.cli process --folder /path/to/videos --recursive
    python -m cctv_yolo.cli process /path/to/video.mp4 --model yolov8m.pt
    python -m cctv_yolo.cli annotate <session_id>
    python -m cctv_yolo.cli heatmap <session_id>
    python -m cctv_yolo.cli timeseries <session_id> --bucket 60
    python -m cctv_yolo.cli speeds <session_id> --ppm 22.5
    python -m cctv_yolo.cli report <session_id>
    python -m cctv_yolo.cli train --epochs 30 --base yolov8n.pt
    python -m cctv_yolo.cli list-sessions

All commands operate on the same data root as the desktop app.
"""
from __future__ import annotations
import os
import argparse
import sys
from pathlib import Path

# Pin native math libraries to a single thread BEFORE torch is imported (see
# processor.py): CLI `process` runs detection in-process, and multi-threaded
# torch/OpenCV with the duplicate-OpenMP shim heap-corrupts on Windows.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from cctv_yolo.data_manager import DataManager


def cmd_process(args):
    dm = DataManager()
    from cctv_yolo.processor import process_video

    targets: list[Path] = []
    if args.folder:
        folder = Path(args.folder)
        if not folder.exists():
            print(f"Folder not found: {folder}")
            return 1
        if args.recursive:
            it = folder.rglob("*")
        else:
            it = folder.iterdir()
        for p in it:
            if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
                targets.append(p)
    elif args.video:
        targets.append(Path(args.video))
    else:
        print("Provide either --folder or a video path.")
        return 1

    for v in targets:
        sid = v.stem
        print(f"[process] {v.name} as {sid}")
        process_video(
            video_path=str(v),
            output_dir=str(dm.tracks_dir),
            model_name=args.model,
            conf_threshold=args.conf,
            session_id=sid,
            models_dir=str(dm.models_dir),
        )
    print(f"Processed {len(targets)} video(s).")
    return 0


def cmd_annotate(args):
    dm = DataManager()
    from cctv_yolo.annotated_export import annotate_video
    track_data = dm.load_session_data(args.session_id)
    if not track_data:
        print(f"No data for session {args.session_id}")
        return 1
    video_path = dm.get_video_path(args.session_id)
    if not video_path:
        print(f"Video not found for session {args.session_id}")
        return 1
    out = dm.exports_dir / args.session_id / "annotated.mp4"
    stats = annotate_video(video_path, track_data, out, blur_lp=args.blur_lp)
    print(f"Wrote {stats['output_path']} ({stats['frames_written']} frames)")
    return 0


def cmd_heatmap(args):
    dm = DataManager()
    from cctv_yolo import analytics
    data = dm.load_session_data(args.session_id)
    if not data:
        print(f"No data for {args.session_id}")
        return 1
    video_path = dm.get_video_path(args.session_id)
    if not video_path:
        print(f"Video not found for session {args.session_id}")
        return 1
    out = dm.exports_dir / args.session_id / "heatmap.png"
    p = analytics.render_heatmap(video_path, data, out, sigma=args.sigma)
    print(f"Wrote {p}")
    return 0


def cmd_timeseries(args):
    dm = DataManager()
    from cctv_yolo import analytics
    data = dm.load_session_data(args.session_id)
    if not data:
        print(f"No data for {args.session_id}")
        return 1
    out = dm.exports_dir / args.session_id / "timeseries.csv"
    p = analytics.time_series_csv(data, out, bucket_seconds=args.bucket)
    print(f"Wrote {p}")
    return 0


def cmd_speeds(args):
    dm = DataManager()
    from cctv_yolo import analytics
    data = dm.load_session_data(args.session_id)
    if not data:
        print(f"No data for {args.session_id}")
        return 1
    out = dm.exports_dir / args.session_id / "speeds.csv"
    speeds = analytics.estimate_speeds(data, pixels_per_meter=args.ppm)
    analytics.write_speeds_csv(speeds, out)
    print(f"Wrote {out} ({len(speeds)} speeds)")
    return 0


def cmd_report(args):
    dm = DataManager()
    from cctv_yolo.report import render_html_report
    p = render_html_report(dm, args.session_id, embed_video=args.embed_video)
    print(f"Wrote {p}")
    return 0


def cmd_train(args):
    dm = DataManager()
    from cctv_yolo.training import build_yolo_dataset, TrainingWorker
    import datetime as dt
    out_root = dm.data_root / "training" / dt.datetime.now().strftime("ds_%Y%m%d_%H%M%S")
    print(f"[train] Building dataset at {out_root}")
    stats = build_yolo_dataset(dm, out_root, sample_every_n=args.sample_every)
    if stats["images"] == 0:
        print("No corrected sessions -- nothing to train on.")
        return 1
    print(f"[train] {stats['images']} images, {stats['labels']} labels, "
          f"classes={stats['classes']}")

    worker = TrainingWorker(
        data_yaml=stats["yaml_path"],
        base_model=args.base,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        models_dir=str(dm.models_dir),
    )
    # Write log lines to the REAL stdout captured HERE, before training starts.
    # TrainingWorker.run() swaps the process-global sys.stdout to an _EmitWriter
    # that re-emits log_line; this slot runs in the worker thread (a plain
    # callable connects as DirectConnection), so a bare print() would go back
    # into that writer and re-emit -> infinite recursion -> native STACK_OVERFLOW
    # (0xC00000FD). Writing to the captured console stream breaks the cycle.
    _console = sys.stdout

    def _print_log(line):
        try:
            _console.write(line + "\n")
            _console.flush()
        except Exception:
            pass

    worker.log_line.connect(_print_log)

    from PySide6.QtCore import QCoreApplication
    app = QCoreApplication.instance() or QCoreApplication(sys.argv)

    def _quit_with(msg):
        print(msg)
        app.quit()

    worker.finished_ok.connect(lambda p: _quit_with(f"[train] Saved {p}"))
    worker.failed.connect(lambda m: _quit_with(f"[train] FAILED: {m}"))
    worker.start()
    runner = getattr(app, "exec")
    runner()
    return 0


def cmd_list_sessions(args):
    dm = DataManager()
    sessions = dm.get_sessions()
    print(f"{'session_id':<60} {'tracks':>7} {'review':>7} {'corrected':>10}")
    for s in sessions:
        print(f"{s['id']:<60} {s['track_count']:>7} {s['needs_review']:>7} "
              f"{('yes' if s['has_corrections'] else 'no'):>10}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cctv-yolo", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("process", help="Run YOLO detection + tracking")
    p.add_argument("video", nargs="?")
    p.add_argument("--folder")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--model", default="yolov8m.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("annotate", help="Render annotated MP4")
    p.add_argument("session_id")
    p.add_argument("--blur-lp", action="store_true")
    p.set_defaults(func=cmd_annotate)

    p = sub.add_parser("heatmap", help="Render path-density heatmap PNG")
    p.add_argument("session_id")
    p.add_argument("--sigma", type=float, default=12.0)
    p.set_defaults(func=cmd_heatmap)

    p = sub.add_parser("timeseries", help="Export per-bucket time series CSV")
    p.add_argument("session_id")
    p.add_argument("--bucket", type=int, default=60)
    p.set_defaults(func=cmd_timeseries)

    p = sub.add_parser("speeds", help="Estimate speeds and export CSV")
    p.add_argument("session_id")
    p.add_argument("--ppm", type=float, required=True)
    p.set_defaults(func=cmd_speeds)

    p = sub.add_parser("report", help="Generate self-contained HTML report")
    p.add_argument("session_id")
    p.add_argument("--embed-video", action="store_true")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("train", help="Build dataset from corrections + train")
    p.add_argument("--base", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--sample-every", type=int, default=5)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("list-sessions", help="Print known sessions")
    p.set_defaults(func=cmd_list_sessions)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
