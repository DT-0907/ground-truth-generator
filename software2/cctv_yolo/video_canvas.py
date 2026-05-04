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


HANDLE_SIZE = 8

HANDLE_CURSORS = {
    'tl': Qt.SizeFDiagCursor,
    'br': Qt.SizeFDiagCursor,
    'tr': Qt.SizeBDiagCursor,
    'bl': Qt.SizeBDiagCursor,
    'tc': Qt.SizeVerCursor,
    'bc': Qt.SizeVerCursor,
    'ml': Qt.SizeHorCursor,
    'mr': Qt.SizeHorCursor,
}

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
    bbox_resized = Signal(list)         # [x1, y1, x2, y2] in video coords
    bbox_moved = Signal(list)           # [x1, y1, x2, y2] in video coords (after drag)

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
        self._frame_cache = {}           # frame_num -> QPixmap
        self._frame_cache_order = []     # LRU order
        self._frame_cache_max = 60       # cache up to 60 frames
        self._display_rect = QRect()  # where the frame is drawn on widget

        # Track overlay data (set from outside)
        self.tracks = []
        self.selected_track_id = None
        self.current_frame = 0
        self.rois = []
        self.dimmed_track_ids = set()  # track IDs to draw at reduced opacity

        # Drawing state
        self.drawing_mode = "select"  # "select", "draw_box", "roi_rect", "roi_polygon"
        self._is_drawing = False
        self._draw_start = None       # canvas coords tuple (cx, cy)
        self._draw_end = None         # canvas coords tuple (cx, cy)
        self._polygon_points = []     # canvas coords for in-progress polygon

        # Resize state
        self._resizing = False
        self._resize_handle = None       # which handle is being dragged (e.g. 'tl')
        self._resize_orig_bbox = None    # original bbox in video coords [x1,y1,x2,y2]
        self._resize_preview_bbox = None # live preview bbox in video coords

        # Drag (move) state
        self._dragging = False
        self._drag_orig_bbox = None      # original bbox in video coords
        self._drag_start_video = None    # (vx, vy) start of drag in video coords
        self._drag_preview_bbox = None   # live preview bbox in video coords

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
        self.clear_frame_cache()
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
        """Load and display a specific frame number with caching."""
        if not self._cap or not self._cap.isOpened():
            return

        # Check cache first
        if frame_num in self._frame_cache:
            self._pixmap = self._frame_cache[frame_num]
            self._last_frame_read = frame_num
            self.current_frame = frame_num
            self._update_display_rect()
            self.update()
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

        # Store in cache
        self._cache_frame(frame_num, self._pixmap)

        self._update_display_rect()
        self.update()

    def _cache_frame(self, frame_num, pixmap):
        """Add a frame to the LRU cache."""
        if frame_num in self._frame_cache:
            self._frame_cache_order.remove(frame_num)
        self._frame_cache[frame_num] = pixmap
        self._frame_cache_order.append(frame_num)
        # Evict oldest if over limit
        while len(self._frame_cache_order) > self._frame_cache_max:
            oldest = self._frame_cache_order.pop(0)
            self._frame_cache.pop(oldest, None)

    def clear_frame_cache(self):
        """Clear the frame cache."""
        self._frame_cache.clear()
        self._frame_cache_order.clear()

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

    def showEvent(self, event):
        """Recalculate display rect when the widget becomes visible."""
        self._update_display_rect()
        super().showEvent(event)

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
    # Resize handles
    # ------------------------------------------------------------------

    def _get_selected_bbox_handles(self):
        """Get handle rects for the currently selected track's bbox on current frame."""
        if not self.selected_track_id or not self._display_rect:
            return {}
        for track in self.tracks:
            if track.get('track_id') == self.selected_track_id:
                for fd in track.get('frames', []):
                    if fd.get('frame') == self.current_frame:
                        bbox = fd['bbox']
                        # If we're actively resizing, use the preview bbox instead
                        if self._resizing and self._resize_preview_bbox is not None:
                            bbox = self._resize_preview_bbox
                        return self._compute_handles(bbox)
        return {}

    def _compute_handles(self, bbox):
        """Compute handle positions for a bbox in canvas coordinates."""
        dr = self._display_rect
        if dr.width() == 0 or self._video_width == 0:
            return {}
        scale_x = dr.width() / self._video_width
        scale_y = dr.height() / self._video_height
        x1c = dr.x() + bbox[0] * scale_x
        y1c = dr.y() + bbox[1] * scale_y
        x2c = dr.x() + bbox[2] * scale_x
        y2c = dr.y() + bbox[3] * scale_y
        mx = (x1c + x2c) / 2
        my = (y1c + y2c) / 2
        hs = HANDLE_SIZE / 2
        return {
            'tl': QRectF(x1c - hs, y1c - hs, HANDLE_SIZE, HANDLE_SIZE),
            'tc': QRectF(mx - hs, y1c - hs, HANDLE_SIZE, HANDLE_SIZE),
            'tr': QRectF(x2c - hs, y1c - hs, HANDLE_SIZE, HANDLE_SIZE),
            'ml': QRectF(x1c - hs, my - hs, HANDLE_SIZE, HANDLE_SIZE),
            'mr': QRectF(x2c - hs, my - hs, HANDLE_SIZE, HANDLE_SIZE),
            'bl': QRectF(x1c - hs, y2c - hs, HANDLE_SIZE, HANDLE_SIZE),
            'bc': QRectF(mx - hs, y2c - hs, HANDLE_SIZE, HANDLE_SIZE),
            'br': QRectF(x2c - hs, y2c - hs, HANDLE_SIZE, HANDLE_SIZE),
        }

    def _hit_test_handles(self, cx, cy):
        """Return the handle key ('tl', 'tc', ...) if (cx,cy) hits a handle, else None."""
        handles = self._get_selected_bbox_handles()
        pt = QPointF(cx, cy)
        for key, rect in handles.items():
            if rect.contains(pt):
                return key
        return None

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

        # 5. Draw resize handles for selected track's bbox
        if self.drawing_mode == "select":
            self._paint_resize_handles(painter, dr, scale_x, scale_y)

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

            track_id = track.get("track_id")
            is_dimmed = track_id in self.dimmed_track_ids

            # Use drag preview bbox if this track is being dragged
            if (self._dragging and self._drag_preview_bbox is not None
                    and track_id == self.selected_track_id):
                x1, y1, x2, y2 = self._drag_preview_bbox
            else:
                x1, y1, x2, y2 = frame_data["bbox"]
            sx1 = dr.x() + x1 * scale_x
            sy1 = dr.y() + y1 * scale_y
            sw = (x2 - x1) * scale_x
            sh = (y2 - y1) * scale_y

            class_name = track.get("class", "unknown")
            color = QColor(CLASS_COLORS.get(class_name, CLASS_COLORS["unknown"]))
            is_selected = track_id == self.selected_track_id
            is_interpolated = frame_data.get("interpolated", False)
            is_occluded = frame_data.get("occluded", False)

            # Reduce opacity for dimmed (outside ROI) tracks
            if is_dimmed:
                color.setAlpha(40)

            # Bounding box outline. Occluded segments use a thicker
            # dotted pen in pink so they're visually unmistakable.
            if is_occluded:
                occluded_color = QColor("#ff64c8")
                pen = QPen(occluded_color, 4 if is_selected else 3)
                pen.setStyle(Qt.DotLine)
            else:
                pen = QPen(color, 4 if is_selected else 2)
                if is_interpolated:
                    pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(sx1, sy1, sw, sh))

            # Semi-transparent fill for selected track
            if is_selected and not is_dimmed:
                fill_color = QColor(color)
                fill_color.setAlpha(50)
                painter.fillRect(QRectF(sx1, sy1, sw, sh), fill_color)

            # Label above the box
            if not is_dimmed:
                if is_occluded:
                    interp_suffix = " [occluded]"
                elif is_interpolated:
                    interp_suffix = " [interp]"
                else:
                    interp_suffix = ""
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

    def _paint_resize_handles(self, painter, dr, scale_x, scale_y):
        """Draw resize handles for the selected track's bbox, and a preview box if resizing."""
        # If resizing, draw the preview bbox
        if self._resizing and self._resize_preview_bbox is not None:
            bbox = self._resize_preview_bbox
            sx1 = dr.x() + bbox[0] * scale_x
            sy1 = dr.y() + bbox[1] * scale_y
            sw = (bbox[2] - bbox[0]) * scale_x
            sh = (bbox[3] - bbox[1]) * scale_y
            pen = QPen(QColor("#ffffff"), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(sx1, sy1, sw, sh))

        # Draw the 8 handles
        handles = self._get_selected_bbox_handles()
        if not handles:
            return
        for key, rect in handles.items():
            painter.setPen(QPen(QColor("#000000"), 1))
            painter.setBrush(QBrush(QColor("#ffffff")))
            painter.drawRect(rect)

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

        # Select mode -- check for resize handle hit first, then drag, then click
        if self.drawing_mode == "select":
            handle = self._hit_test_handles(cx, cy)
            if handle:
                # Start resizing
                self._resizing = True
                self._resize_handle = handle
                # Find the original bbox in video coords
                for track in self.tracks:
                    if track.get('track_id') == self.selected_track_id:
                        for fd in track.get('frames', []):
                            if fd.get('frame') == self.current_frame:
                                self._resize_orig_bbox = list(fd['bbox'])
                                self._resize_preview_bbox = list(fd['bbox'])
                                break
                        break
                return

            # Check if clicking inside selected bbox to drag-move
            vx, vy = self._canvas_to_video(cx, cy)
            if self.selected_track_id is not None:
                for track in self.tracks:
                    if track.get('track_id') == self.selected_track_id:
                        for fd in track.get('frames', []):
                            if fd.get('frame') == self.current_frame:
                                bbox = fd['bbox']
                                if bbox[0] <= vx <= bbox[2] and bbox[1] <= vy <= bbox[3]:
                                    self._dragging = True
                                    self._drag_orig_bbox = list(bbox)
                                    self._drag_start_video = (vx, vy)
                                    self._drag_preview_bbox = list(bbox)
                                    self.setCursor(Qt.ClosedHandCursor)
                                    return
                        break

            self.frame_clicked.emit(int(vx), int(vy))

    def mouseMoveEvent(self, event):
        pos = event.position()
        cx, cy = pos.x(), pos.y()

        # Resize dragging
        if self._resizing and self._resize_orig_bbox is not None:
            vx, vy = self._canvas_to_video(cx, cy)
            orig = self._resize_orig_bbox
            new_bbox = list(orig)
            h = self._resize_handle

            # Adjust edges based on which handle is being dragged
            if h in ('tl', 'ml', 'bl'):
                new_bbox[0] = vx  # adjust x1
            if h in ('tr', 'mr', 'br'):
                new_bbox[2] = vx  # adjust x2
            if h in ('tl', 'tc', 'tr'):
                new_bbox[1] = vy  # adjust y1
            if h in ('bl', 'bc', 'br'):
                new_bbox[3] = vy  # adjust y2

            # Ensure x1 < x2 and y1 < y2
            if new_bbox[0] > new_bbox[2]:
                new_bbox[0], new_bbox[2] = new_bbox[2], new_bbox[0]
            if new_bbox[1] > new_bbox[3]:
                new_bbox[1], new_bbox[3] = new_bbox[3], new_bbox[1]

            self._resize_preview_bbox = new_bbox
            self.update()
            return

        # Drag-move
        if self._dragging and self._drag_orig_bbox is not None:
            vx, vy = self._canvas_to_video(cx, cy)
            dx = vx - self._drag_start_video[0]
            dy = vy - self._drag_start_video[1]
            orig = self._drag_orig_bbox
            self._drag_preview_bbox = [
                orig[0] + dx, orig[1] + dy,
                orig[2] + dx, orig[3] + dy,
            ]
            self.update()
            return

        if self._is_drawing and self.drawing_mode in ("draw_box", "roi_rect"):
            self._draw_end = (cx, cy)
            self.update()
            return

        # Cursor changes for handle hover (select mode only)
        if self.drawing_mode == "select" and not self._is_drawing:
            handle = self._hit_test_handles(cx, cy)
            if handle:
                self.setCursor(HANDLE_CURSORS[handle])
            else:
                # Check if hovering inside selected bbox for move cursor
                if self.selected_track_id is not None:
                    vx, vy = self._canvas_to_video(cx, cy)
                    for track in self.tracks:
                        if track.get('track_id') == self.selected_track_id:
                            for fd in track.get('frames', []):
                                if fd.get('frame') == self.current_frame:
                                    bbox = fd['bbox']
                                    if bbox[0] <= vx <= bbox[2] and bbox[1] <= vy <= bbox[3]:
                                        self.setCursor(Qt.OpenHandCursor)
                                        return
                            break
                self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        # Finalize resize
        if self._resizing and self._resize_preview_bbox is not None:
            final_bbox = self._resize_preview_bbox
            self._resizing = False
            self._resize_handle = None
            self._resize_orig_bbox = None
            self._resize_preview_bbox = None
            self.bbox_resized.emit(final_bbox)
            self.update()
            return

        # Finalize drag-move
        if self._dragging and self._drag_preview_bbox is not None:
            final_bbox = self._drag_preview_bbox
            self._dragging = False
            self._drag_orig_bbox = None
            self._drag_start_video = None
            self._drag_preview_bbox = None
            self.setCursor(Qt.ArrowCursor)
            self.bbox_moved.emit(final_bbox)
            self.update()
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
