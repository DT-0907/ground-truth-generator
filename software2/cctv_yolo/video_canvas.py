"""
Video canvas widget — displays video frames with bounding box and ROI overlays.
Uses QPainter for all rendering. No HTML/JS involved.
"""
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, Signal, QRect, QRectF, QPointF
from PySide6.QtGui import (QPainter, QPixmap, QImage, QPen, QColor, QBrush,
                            QFont, QPolygonF)
import cv2
import numpy as np


CLASS_COLORS = {
    "car": QColor("#3498db"),
    "truck": QColor("#e74c3c"),
    "bus": QColor("#9b59b6"),
    "motorcycle": QColor("#f39c12"),
    "bicycle": QColor("#1abc9c"),
    "unknown": QColor("#95a5a6"),
}


class VideoCanvas(QWidget):
    """Custom widget that displays video frames with bounding box and ROI overlays.

    Replaces the HTML canvas + img element from the web UI. Handles video frame
    extraction via cv2.VideoCapture, coordinate conversion between canvas and
    video spaces, and interactive drawing of bounding boxes and ROIs.
    """

    box_drawn = Signal(list)            # [x1, y1, x2, y2] in video coords
    roi_rect_drawn = Signal(dict, dict) # p1, p2 in video coords
    roi_polygon_drawn = Signal(list)    # list of {x, y} dicts in video coords
    frame_clicked = Signal(int, int)    # click position in video coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # Video state
        self._cap = None
        self._pixmap = None
        self._video_width = 0
        self._video_height = 0
        self._total_frames = 0
        self._last_frame_read = -1    # tracks cap position for sequential read optimization
        self._display_rect = QRect()  # where the frame is drawn on widget

        # Track overlay data (set from outside)
        self.tracks = []
        self.selected_track_id = None
        self.current_frame = 0
        self.rois = []

        # Drawing state
        self.drawing_mode = "select"  # "select", "draw_box", "roi_rect", "roi_polygon"
        self._is_drawing = False
        self._draw_start = None       # canvas coords tuple (cx, cy)
        self._draw_end = None         # canvas coords tuple (cx, cy)
        self._polygon_points = []     # canvas coords for in-progress polygon

        self.setStyleSheet("background-color: #000;")

    # ------------------------------------------------------------------
    # Video I/O
    # ------------------------------------------------------------------

    def open_video(self, video_path: str):
        """Open a video file for frame extraction."""
        if self._cap:
            self._cap.release()
        self._cap = cv2.VideoCapture(video_path)
        if self._cap.isOpened():
            self._video_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._video_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._last_frame_read = -1

    def close_video(self):
        """Release the underlying VideoCapture."""
        if self._cap:
            self._cap.release()
            self._cap = None
        self._pixmap = None
        self._last_frame_read = -1
        self.update()

    @property
    def video_width(self):
        return self._video_width

    @property
    def video_height(self):
        return self._video_height

    @property
    def total_frames(self):
        return self._total_frames

    def set_frame(self, frame_num: int):
        """Load and display a specific frame number.

        For sequential advancement (frame_num == last_frame_read + 1), we simply
        call cap.read() which is reliable across all codecs.  For arbitrary seeks
        we fall back to CAP_PROP_POS_FRAMES, and if that silently returns the
        wrong frame we verify and retry once.
        """
        if not self._cap or not self._cap.isOpened():
            return

        if frame_num == self._last_frame_read + 1:
            # Sequential read — the capture is already positioned at the next frame
            ret, frame = self._cap.read()
        else:
            # Non-sequential seek
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self._cap.read()

        if not ret:
            return

        self._last_frame_read = frame_num
        self.current_frame = frame_num

        # BGR -> RGB -> QImage -> QPixmap
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self._update_display_rect()
        self.repaint()  # force immediate repaint (update() can be coalesced/deferred)

    # ------------------------------------------------------------------
    # Coordinate mapping
    # ------------------------------------------------------------------

    def _update_display_rect(self):
        """Calculate where the frame should be drawn (maintaining aspect ratio)."""
        if not self._pixmap:
            return
        widget_w = self.width()
        widget_h = self.height()
        img_w = self._pixmap.width()
        img_h = self._pixmap.height()

        scale = min(widget_w / img_w, widget_h / img_h)
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)
        x = (widget_w - disp_w) // 2
        y = (widget_h - disp_h) // 2
        self._display_rect = QRect(x, y, disp_w, disp_h)

    def resizeEvent(self, event):
        self._update_display_rect()
        super().resizeEvent(event)

    def _canvas_to_video(self, cx, cy):
        """Convert canvas (widget) coordinates to video pixel coordinates."""
        dr = self._display_rect
        if dr.width() == 0 or dr.height() == 0:
            return 0, 0
        vx = (cx - dr.x()) / dr.width() * self._video_width
        vy = (cy - dr.y()) / dr.height() * self._video_height
        return vx, vy

    def _video_to_canvas(self, vx, vy):
        """Convert video pixel coordinates to canvas (widget) coordinates."""
        dr = self._display_rect
        if self._video_width == 0 or self._video_height == 0:
            return 0, 0
        cx = dr.x() + (vx / self._video_width) * dr.width()
        cy = dr.y() + (vy / self._video_height) * dr.height()
        return cx, cy

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. Draw the video frame (or placeholder text)
        if self._pixmap:
            painter.drawPixmap(self._display_rect, self._pixmap)
        else:
            painter.setPen(QColor("#888"))
            painter.drawText(self.rect(), Qt.AlignCenter, "No video loaded")
            painter.end()
            return

        dr = self._display_rect
        if dr.width() == 0:
            painter.end()
            return

        scale_x = dr.width() / self._video_width
        scale_y = dr.height() / self._video_height

        # Shared font for labels
        font = QFont("sans-serif", 10, QFont.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        text_h = fm.height() + 4

        # 2. Draw bounding boxes for visible tracks
        self._paint_tracks(painter, dr, scale_x, scale_y, fm, text_h)

        # 3. Draw ROIs
        self._paint_rois(painter, fm, text_h)

        # 4. Draw in-progress shapes
        self._paint_in_progress(painter)

        painter.end()

    def _paint_tracks(self, painter, dr, scale_x, scale_y, fm, text_h):
        """Draw bounding boxes and labels for all tracks visible at current_frame."""
        for track in self.tracks:
            frame_data = None
            for fd in track.get("frames", []):
                if fd["frame"] == self.current_frame:
                    frame_data = fd
                    break
            if not frame_data:
                continue

            x1, y1, x2, y2 = frame_data["bbox"]
            sx1 = dr.x() + x1 * scale_x
            sy1 = dr.y() + y1 * scale_y
            sw = (x2 - x1) * scale_x
            sh = (y2 - y1) * scale_y

            class_name = track.get("class", "unknown")
            color = CLASS_COLORS.get(class_name, CLASS_COLORS["unknown"])
            is_selected = track.get("track_id") == self.selected_track_id
            is_interpolated = frame_data.get("interpolated", False)

            # Bounding box outline
            pen = QPen(color, 4 if is_selected else 2)
            if is_interpolated:
                pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(sx1, sy1, sw, sh))

            # Semi-transparent fill for selected track
            if is_selected:
                fill_color = QColor(color)
                fill_color.setAlpha(50)
                painter.fillRect(QRectF(sx1, sy1, sw, sh), fill_color)

            # Label above the box
            interp_suffix = " [interp]" if is_interpolated else ""
            label = f"#{track.get('track_id', '?')} {class_name}{interp_suffix}"
            text_w = fm.horizontalAdvance(label) + 8

            label_bg = QColor(color)
            if is_interpolated:
                label_bg.setAlpha(170)
            painter.fillRect(QRectF(sx1, sy1 - text_h, text_w, text_h), label_bg)
            painter.setPen(QColor("#000"))
            painter.drawText(QPointF(sx1 + 4, sy1 - 4), label)

    def _paint_rois(self, painter, fm, text_h):
        """Draw all ROI overlays (rect and polygon)."""
        for roi in self.rois:
            roi_color = QColor(roi.get("color", "#ff6b6b"))
            pen = QPen(roi_color, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)

            name = roi.get("name", "ROI")
            tw = fm.horizontalAdvance(name) + 8

            if roi["type"] == "rect":
                p1 = roi["points"][0]
                p2 = roi["points"][1]
                cx1, cy1 = self._video_to_canvas(p1["x"], p1["y"])
                cx2, cy2 = self._video_to_canvas(p2["x"], p2["y"])
                rect = QRectF(cx1, cy1, cx2 - cx1, cy2 - cy1)
                painter.drawRect(rect)
                fill = QColor(roi_color)
                fill.setAlpha(20)
                painter.fillRect(rect, fill)
                # Label
                label_bg = QColor(roi_color)
                label_bg.setAlpha(200)
                painter.fillRect(QRectF(cx1, cy1 - text_h, tw, text_h), label_bg)
                painter.setPen(QColor("#000"))
                painter.drawText(QPointF(cx1 + 4, cy1 - 4), name)
                painter.setPen(pen)

            elif roi["type"] == "polygon" and len(roi["points"]) >= 3:
                poly = QPolygonF()
                first_cx, first_cy = 0, 0
                for i, pt in enumerate(roi["points"]):
                    cx, cy = self._video_to_canvas(pt["x"], pt["y"])
                    poly.append(QPointF(cx, cy))
                    if i == 0:
                        first_cx, first_cy = cx, cy
                # Outline
                painter.drawPolygon(poly)
                # Semi-transparent fill
                fill = QColor(roi_color)
                fill.setAlpha(20)
                painter.setBrush(QBrush(fill))
                painter.drawPolygon(poly)
                painter.setBrush(Qt.NoBrush)
                # Label at first vertex
                label_bg = QColor(roi_color)
                label_bg.setAlpha(200)
                painter.fillRect(QRectF(first_cx, first_cy - text_h, tw, text_h), label_bg)
                painter.setPen(QColor("#000"))
                painter.drawText(QPointF(first_cx + 4, first_cy - 4), name)
                painter.setPen(pen)

    def _paint_in_progress(self, painter):
        """Draw shapes currently being drawn by the user."""
        # Rectangle drag (draw_box or roi_rect)
        if self._is_drawing and self._draw_start and self._draw_end:
            pen = QPen(QColor("#fff"), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            x = min(self._draw_start[0], self._draw_end[0])
            y = min(self._draw_start[1], self._draw_end[1])
            w = abs(self._draw_end[0] - self._draw_start[0])
            h = abs(self._draw_end[1] - self._draw_start[1])
            painter.drawRect(QRectF(x, y, w, h))

        # Polygon in-progress
        if self.drawing_mode == "roi_polygon" and self._polygon_points:
            pen = QPen(QColor("#fff"), 2, Qt.DashLine)
            painter.setPen(pen)
            for i in range(len(self._polygon_points) - 1):
                p1 = self._polygon_points[i]
                p2 = self._polygon_points[i + 1]
                painter.drawLine(QPointF(p1[0], p1[1]), QPointF(p2[0], p2[1]))
            # Draw dots at vertices
            painter.setBrush(QColor("#fff"))
            for p in self._polygon_points:
                painter.drawEllipse(QPointF(p[0], p[1]), 4, 4)
            painter.setBrush(Qt.NoBrush)

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        pos = event.position()
        cx, cy = pos.x(), pos.y()

        if self.drawing_mode == "roi_polygon":
            self._polygon_points.append((cx, cy))
            self.update()
            return

        if self.drawing_mode in ("draw_box", "roi_rect"):
            self._is_drawing = True
            self._draw_start = (cx, cy)
            self._draw_end = (cx, cy)
            return

        # Select mode -- emit click in video coords
        if self.drawing_mode == "select":
            vx, vy = self._canvas_to_video(cx, cy)
            self.frame_clicked.emit(int(vx), int(vy))

    def mouseMoveEvent(self, event):
        if self._is_drawing and self.drawing_mode in ("draw_box", "roi_rect"):
            pos = event.position()
            self._draw_end = (pos.x(), pos.y())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        if self._is_drawing and self.drawing_mode in ("draw_box", "roi_rect"):
            self._is_drawing = False
            pos = event.position()
            self._draw_end = (pos.x(), pos.y())

            w = abs(self._draw_end[0] - self._draw_start[0])
            h = abs(self._draw_end[1] - self._draw_start[1])

            # Minimum drag threshold to avoid accidental clicks
            if w > 10 and h > 10:
                vx1, vy1 = self._canvas_to_video(
                    min(self._draw_start[0], self._draw_end[0]),
                    min(self._draw_start[1], self._draw_end[1]))
                vx2, vy2 = self._canvas_to_video(
                    max(self._draw_start[0], self._draw_end[0]),
                    max(self._draw_start[1], self._draw_end[1]))

                if self.drawing_mode == "draw_box":
                    self.box_drawn.emit([vx1, vy1, vx2, vy2])
                elif self.drawing_mode == "roi_rect":
                    self.roi_rect_drawn.emit(
                        {"x": vx1, "y": vy1},
                        {"x": vx2, "y": vy2})

            self._draw_start = None
            self._draw_end = None
            self.update()

    def mouseDoubleClickEvent(self, event):
        if self.drawing_mode == "roi_polygon" and len(self._polygon_points) >= 3:
            # Remove duplicate last point from the mousePress that fires before
            # the double-click event (proximity check like the JS version)
            if len(self._polygon_points) >= 2:
                last = self._polygon_points[-1]
                prev = self._polygon_points[-2]
                if abs(last[0] - prev[0]) < 5 and abs(last[1] - prev[1]) < 5:
                    self._polygon_points.pop()

            # Convert canvas coords to video coords
            video_points = []
            for cx, cy in self._polygon_points:
                vx, vy = self._canvas_to_video(cx, cy)
                video_points.append({"x": vx, "y": vy})

            self.roi_polygon_drawn.emit(video_points)
            self._polygon_points = []
            self.update()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def cancel_drawing(self):
        """Cancel any in-progress drawing operation."""
        self._is_drawing = False
        self._draw_start = None
        self._draw_end = None
        self._polygon_points = []
        self.update()

    def set_cursor_for_mode(self):
        """Update the mouse cursor shape based on the current drawing mode."""
        if self.drawing_mode in ("draw_box", "roi_rect", "roi_polygon"):
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
