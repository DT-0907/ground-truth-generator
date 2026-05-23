"""
Timeline minimap widget — paints a thin strip showing where in the
video each track exists, with hot zones for low-confidence frames and
edited frames.

Click a column to seek to that frame.

Drop-in: ``minimap = TimelineMinimap(self); minimap.set_data(track_data,
total_frames); minimap.frame_clicked.connect(self._seek)``.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QColor, QPainter, QPen, QBrush
from PySide6.QtWidgets import QWidget


CLASS_COLORS = {
    "car": QColor(128, 255, 0),
    "truck": QColor(0, 128, 255),
    "bus": QColor(255, 128, 0),
    "motorcycle": QColor(0, 255, 255),
    "bicycle": QColor(255, 0, 128),
}


class TimelineMinimap(QWidget):
    """Strip showing track presence + flags. Click to seek."""

    frame_clicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setMaximumHeight(110)
        self._tracks: list[dict] = []
        self._total_frames: int = 1
        self._current_frame: int = 0
        self._hover_frame: int | None = None
        self.setMouseTracking(True)

    # ----- Public API -----
    def set_data(self, track_data: dict, total_frames: int):
        self._tracks = track_data.get("tracks", []) or []
        self._total_frames = max(1, int(total_frames))
        self._build_low_conf_density()
        self.update()

    def set_current_frame(self, frame: int):
        self._current_frame = max(0, int(frame))
        self.update()

    # ----- Internal -----
    def _build_low_conf_density(self):
        """Bucket low-confidence frames into per-pixel hits."""
        # We'll compute on-paint using current widget width
        pass

    def _frame_to_x(self, frame: int) -> float:
        return (frame / self._total_frames) * self.width()

    def _x_to_frame(self, x: float) -> int:
        return int((x / max(1, self.width())) * self._total_frames)

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)

        w = self.width()
        h = self.height()
        # Background
        p.fillRect(self.rect(), QColor("#15173D"))

        if not self._tracks or w < 4:
            self._draw_playhead(p)
            return

        # Layout: top half = track-presence rows, bottom row = density.
        track_band_h = h - 18
        density_y = h - 16

        # Render up to 40 tracks as horizontal lanes, oldest to newest
        sorted_tracks = sorted(
            self._tracks,
            key=lambda t: t.get("start_frame", 0),
        )[:40]
        n = max(1, len(sorted_tracks))
        lane_h = max(2.0, track_band_h / n)

        for i, tr in enumerate(sorted_tracks):
            y = i * lane_h
            cls = tr.get("class", "vehicle")
            color = CLASS_COLORS.get(cls, QColor(200, 200, 200))
            color.setAlpha(220 if tr.get("avg_confidence", 0) > 0.4 else 120)
            x1 = self._frame_to_x(tr.get("start_frame", 0))
            x2 = self._frame_to_x(tr.get("end_frame", 0))
            p.fillRect(QRectF(x1, y, max(1.0, x2 - x1), lane_h - 0.5), color)

        # Density of low-conf detections (per-pixel-column count)
        if w > 0:
            buckets = [0] * w
            max_bucket = 1
            for tr in self._tracks:
                if tr.get("avg_confidence", 0) >= 0.4:
                    continue
                for fd in tr.get("frames", []):
                    if fd.get("conf", 1.0) >= 0.4:
                        continue
                    px = int(self._frame_to_x(fd["frame"]))
                    if 0 <= px < w:
                        buckets[px] += 1
                        if buckets[px] > max_bucket:
                            max_bucket = buckets[px]
            for x, cnt in enumerate(buckets):
                if cnt == 0:
                    continue
                intensity = min(1.0, cnt / max_bucket)
                p.fillRect(
                    QRectF(x, density_y, 1, 14),
                    QColor(255, int(80 + 175 * intensity), 0,
                           int(80 + 175 * intensity)),
                )

        # Edited markers — separate styles for plain interpolation vs
        # occlusion-recovered frames so the reviewer can spot edges.
        interp_pen = QPen(QColor(78, 204, 163, 180))
        interp_pen.setWidth(1)
        occluded_pen = QPen(QColor(255, 100, 200, 230))
        occluded_pen.setWidth(2)
        for tr in self._tracks:
            for fd in tr.get("frames", []):
                if fd.get("occluded"):
                    p.setPen(occluded_pen)
                    px = int(self._frame_to_x(fd["frame"]))
                    p.drawLine(px, 0, px, track_band_h)
                elif fd.get("interpolated"):
                    p.setPen(interp_pen)
                    px = int(self._frame_to_x(fd["frame"]))
                    p.drawLine(px, 0, px, track_band_h)

        # Playhead
        self._draw_playhead(p)

        # Hover line
        if self._hover_frame is not None:
            hx = int(self._frame_to_x(self._hover_frame))
            p.setPen(QColor(255, 255, 255, 80))
            p.drawLine(hx, 0, hx, h)

    def _draw_playhead(self, p: QPainter):
        x = int(self._frame_to_x(self._current_frame))
        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(2)
        p.setPen(pen)
        p.drawLine(x, 0, x, self.height())

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            f = self._x_to_frame(ev.position().x())
            f = max(0, min(self._total_frames - 1, f))
            self._current_frame = f
            self.frame_clicked.emit(f)
            self.update()

    def mouseMoveEvent(self, ev):
        self._hover_frame = self._x_to_frame(ev.position().x())
        self.update()

    def leaveEvent(self, _ev):
        self._hover_frame = None
        self.update()
