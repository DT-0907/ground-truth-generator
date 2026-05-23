"""Tiny QPainter line/bar/histogram widget for inline Analytics previews.

Modes:
- 'line'      — list of floats, drawn as a line plot
- 'bar'       — list of (label, value) tuples, drawn as labelled bars
- 'histogram' — list of floats, auto-bucketed into ~12 bins

Pure QPainter, no matplotlib dep. Theme-aware (uses cctv_yolo.theme).
Used by Analytics tab (PRD Part H).
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont
from PySide6.QtWidgets import QWidget

from cctv_yolo.theme import (
    PURPLE,
    PINK,
    BORDER,
    PANEL,
    OFFWHITE,
    TEXT_MUTED,
)


class MiniChart(QWidget):
    """Compact chart for inline previews next to analytics results."""

    def __init__(self, mode: str = "line", parent=None):
        super().__init__(parent)
        self._mode = mode  # 'line' | 'bar' | 'histogram'
        self._data = []
        self._title = ""
        self.setMinimumHeight(140)
        self.setStyleSheet(
            f"background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px;"
        )

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self.update()

    def set_title(self, title: str) -> None:
        self._title = title
        self.update()

    def set_data(self, data) -> None:
        """Set data.

        - line:      list[float]
        - bar:       list[(label, value)]
        - histogram: list[float]
        """
        self._data = list(data) if data else []
        self.update()

    # --- paint -----------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        # Outer fill is already from stylesheet; draw chart area inside padding.
        pad_l, pad_r, pad_t, pad_b = 36, 12, 18, 22
        chart = QRectF(pad_l, pad_t, max(1, w - pad_l - pad_r), max(1, h - pad_t - pad_b))

        # Title
        if self._title:
            painter.setPen(QPen(QColor(TEXT_MUTED)))
            f = QFont()
            f.setPointSize(9)
            painter.setFont(f)
            painter.drawText(
                QRectF(pad_l, 2, w - pad_l - pad_r, pad_t),
                Qt.AlignLeft | Qt.AlignVCenter,
                self._title,
            )

        # Empty state
        if not self._data:
            painter.setPen(QPen(QColor(TEXT_MUTED)))
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "no data",
            )
            painter.end()
            return

        # Axis lines
        axis_pen = QPen(QColor(BORDER))
        axis_pen.setWidth(1)
        painter.setPen(axis_pen)
        painter.drawLine(
            QPointF(chart.left(), chart.bottom()),
            QPointF(chart.right(), chart.bottom()),
        )
        painter.drawLine(
            QPointF(chart.left(), chart.top()),
            QPointF(chart.left(), chart.bottom()),
        )

        if self._mode == "line":
            self._paint_line(painter, chart)
        elif self._mode == "bar":
            self._paint_bar(painter, chart)
        elif self._mode == "histogram":
            self._paint_histogram(painter, chart)
        painter.end()

    def _paint_axis_labels(self, painter: QPainter, chart: QRectF, vmax: float):
        """Y-axis max + min labels and a faint mid-line."""
        f = QFont()
        f.setPointSize(8)
        painter.setFont(f)
        painter.setPen(QPen(QColor(TEXT_MUTED)))
        painter.drawText(
            QRectF(0, chart.top() - 8, chart.left() - 2, 16),
            Qt.AlignRight | Qt.AlignVCenter,
            f"{vmax:g}",
        )
        painter.drawText(
            QRectF(0, chart.bottom() - 8, chart.left() - 2, 16),
            Qt.AlignRight | Qt.AlignVCenter,
            "0",
        )

    def _paint_line(self, painter: QPainter, chart: QRectF):
        vals = [float(v) for v in self._data if v is not None]
        if not vals:
            return
        vmax = max(vals) if max(vals) > 0 else 1.0
        self._paint_axis_labels(painter, chart, vmax)

        # Fill area under line (subtle)
        n = len(vals)
        if n < 2:
            # Single value → draw a dot
            x = chart.left() + chart.width() / 2
            y = chart.bottom() - (vals[0] / vmax) * chart.height()
            painter.setBrush(QBrush(QColor(PURPLE)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(x, y), 3.0, 3.0)
            return

        step = chart.width() / (n - 1)
        points = []
        for i, v in enumerate(vals):
            x = chart.left() + i * step
            y = chart.bottom() - (v / vmax) * chart.height()
            points.append(QPointF(x, y))

        # Fill polygon
        fill_color = QColor(PURPLE)
        fill_color.setAlpha(60)
        painter.setBrush(QBrush(fill_color))
        painter.setPen(Qt.NoPen)
        from PySide6.QtGui import QPolygonF
        poly = QPolygonF(points + [QPointF(chart.right(), chart.bottom()),
                                   QPointF(chart.left(), chart.bottom())])
        painter.drawPolygon(poly)

        # Stroke
        pen = QPen(QColor(PURPLE))
        pen.setWidth(2)
        painter.setPen(pen)
        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])

    def _paint_bar(self, painter: QPainter, chart: QRectF):
        """data: list[(label, value)]"""
        pairs = []
        for item in self._data:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                pairs.append((str(item[0]), float(item[1])))
            else:
                pairs.append(("", float(item)))
        if not pairs:
            return
        vmax = max(v for _, v in pairs) if pairs else 1.0
        if vmax <= 0:
            vmax = 1.0
        self._paint_axis_labels(painter, chart, vmax)

        n = len(pairs)
        gap = 6
        bar_w = max(2.0, (chart.width() - gap * (n + 1)) / n)

        painter.setBrush(QBrush(QColor(PURPLE)))
        painter.setPen(Qt.NoPen)
        f = QFont()
        f.setPointSize(8)
        for i, (label, v) in enumerate(pairs):
            x = chart.left() + gap + i * (bar_w + gap)
            bh = (v / vmax) * chart.height()
            y = chart.bottom() - bh
            painter.setBrush(QBrush(QColor(PURPLE)))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(QRectF(x, y, bar_w, bh), 2.0, 2.0)

            # Value label on top
            painter.setPen(QPen(QColor(OFFWHITE)))
            painter.setFont(f)
            painter.drawText(
                QRectF(x - 4, y - 14, bar_w + 8, 14),
                Qt.AlignCenter,
                f"{int(v)}" if v == int(v) else f"{v:.1f}",
            )
            # X-axis label
            painter.setPen(QPen(QColor(TEXT_MUTED)))
            painter.drawText(
                QRectF(x - 4, chart.bottom() + 2, bar_w + 8, 16),
                Qt.AlignCenter,
                label,
            )

    def _paint_histogram(self, painter: QPainter, chart: QRectF):
        vals = [float(v) for v in self._data if v is not None]
        if not vals:
            return
        lo = min(vals)
        hi = max(vals)
        if hi <= lo:
            hi = lo + 1.0
        nbins = min(12, max(4, int(len(vals) ** 0.5)))
        bins = [0] * nbins
        width = (hi - lo) / nbins
        for v in vals:
            idx = min(nbins - 1, max(0, int((v - lo) / width)))
            bins[idx] += 1
        vmax = max(bins) if bins else 1
        self._paint_axis_labels(painter, chart, float(vmax))

        gap = 2
        bar_w = max(1.0, (chart.width() - gap * (nbins + 1)) / nbins)
        painter.setBrush(QBrush(QColor(PURPLE)))
        painter.setPen(Qt.NoPen)
        f = QFont()
        f.setPointSize(8)
        for i, c in enumerate(bins):
            x = chart.left() + gap + i * (bar_w + gap)
            bh = (c / vmax) * chart.height() if vmax else 0
            y = chart.bottom() - bh
            painter.setBrush(QBrush(QColor(PURPLE)))
            painter.setPen(Qt.NoPen)
            painter.drawRect(QRectF(x, y, bar_w, bh))

        # X range label
        painter.setPen(QPen(QColor(TEXT_MUTED)))
        painter.setFont(f)
        painter.drawText(
            QRectF(chart.left(), chart.bottom() + 2, chart.width() / 2, 16),
            Qt.AlignLeft | Qt.AlignVCenter,
            f"{lo:.1f}",
        )
        painter.drawText(
            QRectF(chart.left() + chart.width() / 2, chart.bottom() + 2,
                   chart.width() / 2, 16),
            Qt.AlignRight | Qt.AlignVCenter,
            f"{hi:.1f}",
        )
