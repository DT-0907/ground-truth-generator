"""Single source of truth for app colors + status semantics.

The whole UI reads from here. Adding a new color? Add a token here, not
inline. The PRD's acceptance test runs
``grep -rE '#[0-9a-fA-F]{6}' software2/cctv_yolo/``
and expects hits only in this file.

Palette comes from the user-supplied mandate:
- INDIGO   #15173D   darkest, main background
- PURPLE   #982598   action / primary accent
- PINK     #E491C9   highlight / completed
- OFFWHITE #F1E9E9   text

Derived tokens are matched-tone fillers for UI roles the four core colors
don't cover (panels above the bg, borders, muted text, error).
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Core palette
# ---------------------------------------------------------------------------
INDIGO    = "#15173D"   # background
PURPLE    = "#982598"   # action / primary
PINK      = "#E491C9"   # highlight / done
OFFWHITE  = "#F1E9E9"   # text

# ---------------------------------------------------------------------------
# Derived palette — matched tones for UI roles
# ---------------------------------------------------------------------------
PANEL       = "#1E2050"   # cards, group boxes — one tier above INDIGO
PANEL_HI    = "#2A2C66"   # hovered / raised panel
BORDER      = "#2D2F60"   # dividers, table grid, tree branches
TEXT_MUTED  = "#A89BA8"   # secondary text, hints, disabled labels
ERROR       = "#FF6B7A"   # warm red that harmonises with PINK
YELLOW      = "#F1C56B"   # warm yellow for class chips (e.g., motorcycle)

# ---------------------------------------------------------------------------
# Status-color mapping (consumed by every tab that shows item state)
# ---------------------------------------------------------------------------
# Each value is a (background, text) pair so callers can style consistently.
STATUS_DONE        = (PINK,    INDIGO)      # Processed / Completed
STATUS_PROCESSING  = (PURPLE,  OFFWHITE)    # In progress / Live
STATUS_QUEUED      = ("transparent", TEXT_MUTED)
STATUS_ERROR       = (ERROR,   OFFWHITE)
STATUS_CANCELLED   = (BORDER,  TEXT_MUTED)

# Overlays applied on top of the underlying status color
STATUS_SELECTED_OVERLAY = ("rgba(152, 37, 152, 0.30)", OFFWHITE)   # PURPLE @ 30%
STATUS_HOVER_OVERLAY    = ("rgba(228, 145, 201, 0.15)", OFFWHITE)  # PINK @ 15%

# ---------------------------------------------------------------------------
# Status label helpers (lookup by string state name)
# ---------------------------------------------------------------------------
STATUS_BY_NAME = {
    "done":        STATUS_DONE,
    "processed":   STATUS_DONE,
    "completed":   STATUS_DONE,
    "processing":  STATUS_PROCESSING,
    "in_progress": STATUS_PROCESSING,
    "running":     STATUS_PROCESSING,
    "live":        STATUS_PROCESSING,
    "queued":      STATUS_QUEUED,
    "pending":     STATUS_QUEUED,
    "idle":        STATUS_QUEUED,
    "error":       STATUS_ERROR,
    "failed":      STATUS_ERROR,
    "cancelled":   STATUS_CANCELLED,
    "canceled":    STATUS_CANCELLED,
    "paused":      STATUS_CANCELLED,
}


def status_colors(state: str) -> tuple[str, str]:
    """Return (background_color, text_color) for a status string.

    Defaults to STATUS_QUEUED for unknown states so the UI degrades gracefully.
    """
    return STATUS_BY_NAME.get(state.lower(), STATUS_QUEUED)


# ---------------------------------------------------------------------------
# Class / vehicle-type accent colors (used by Performance stat-card top borders
# and Analytics charts). These harmonise with the core palette.
# ---------------------------------------------------------------------------
CLASS_COLORS = {
    "total":      PURPLE,
    "car":        PINK,
    "truck":      "#C76EB1",      # peachier pink (PINK at ~70% sat)
    "bus":        "#7A2A7A",      # darker purple (PURPLE at ~60% lightness)
    "motorcycle": YELLOW,
    "bicycle":    ERROR,
    "person":     OFFWHITE,
}


# ---------------------------------------------------------------------------
# ROI color rotation (8 distinct, palette-harmonised colors for distinguishing
# multiple ROIs on the same canvas)
# ---------------------------------------------------------------------------
ROI_COLOR_ROTATION = [
    PINK,
    PURPLE,
    YELLOW,
    "#7AC4D4",   # teal-blue (added matching tone)
    "#C76EB1",
    "#9F6FC6",   # soft purple
    "#E07A5F",   # warm coral
    "#5DBB9B",   # soft mint
]


def roi_color(index: int) -> str:
    """Cycle through the ROI palette for the Nth ROI on a canvas."""
    return ROI_COLOR_ROTATION[index % len(ROI_COLOR_ROTATION)]


# ---------------------------------------------------------------------------
# UI metrics (kept here so every tab uses the same spacing scale)
# ---------------------------------------------------------------------------
TYPE_TITLE     = 22   # tab title
TYPE_SECTION   = 16   # section header
TYPE_BODY      = 14
TYPE_HINT      = 12   # muted hints, status text, captions

RADIUS         = 6    # all rounded corners (panels, buttons, cards)
GAP            = 12   # standard vertical spacing between groups
PAD            = 10   # internal padding inside cards/buttons
