/**
 * Correction UI JavaScript
 * Canvas-based frame playback, bbox overlay + resize, ROI filtering
 */

let currentSession = null;
let tracksData = null;
let selectedTrackId = null;
let currentFrame = 0;
let totalFrames = 0;
let videoFps = 30;
let videoWidth = 0;
let videoHeight = 0;

let frameCanvas = null;
let frameCtx = null;
let overlayCanvas = null;
let overlayCtx = null;

let isPlaying = false;
let playTimer = null;
let frameReady = true;

// Bbox resize state
let resizeHandle = null;   // which handle is being dragged
let resizeTrackId = null;
let resizeOrigBbox = null;
let resizeStart = null;     // {x, y} in video coords
let isDraggingBbox = false; // move whole bbox

// ROI filter
let roiFilterEnabled = false;
let rois = [];              // loaded from tracks data
let selectedRoiIds = new Set();

// Vehicle class colors
const CLASS_COLORS = {
    'car': '#FF6B6B',
    'truck': '#4ECDC4',
    'bus': '#45B7D1',
    'motorcycle': '#FFA07A',
    'bicycle': '#98D8C8',
    'person': '#F7DC6F'
};

const HANDLE_SIZE = 8;
const HANDLES = ['tl', 'tc', 'tr', 'ml', 'mr', 'bl', 'bc', 'br'];

// ---------------------------------------------------------------
// Init
// ---------------------------------------------------------------
window.initReviewApp = function(sessionId, videoPath, trackPath) {
    currentSession = sessionId;

    frameCanvas = document.getElementById('frameCanvas');
    overlayCanvas = document.getElementById('overlayCanvas');
    if (!frameCanvas || !overlayCanvas) { console.error('Canvas not found'); return; }
    frameCtx = frameCanvas.getContext('2d');
    overlayCtx = overlayCanvas.getContext('2d');

    // Load video info then tracks
    fetch(`/api/video-info/${sessionId}`)
        .then(r => r.json())
        .then(info => {
            videoFps = info.fps || 30;
            totalFrames = info.total_frames || 0;
            videoWidth = info.width || 1920;
            videoHeight = info.height || 1080;

            frameCanvas.width = videoWidth;
            frameCanvas.height = videoHeight;
            overlayCanvas.width = videoWidth;
            overlayCanvas.height = videoHeight;

            document.getElementById('totalFrames').textContent = totalFrames;
            const scrubber = document.getElementById('frameScrubber');
            if (scrubber) { scrubber.max = Math.max(0, totalFrames - 1); }

            loadFrame(0);
        })
        .catch(e => console.error('Error loading video info:', e));

    loadTracks();
    setupControls();
    setupKeyboardShortcuts();
    setupOverlayInteraction();
};

// ---------------------------------------------------------------
// Track loading
// ---------------------------------------------------------------
function loadTracks() {
    fetch(`/api/session/${currentSession}/tracks`)
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                showStatus('No tracks: ' + data.error, 'error');
                document.getElementById('tracksList').innerHTML = '<p class="loading">No tracks available</p>';
                return;
            }
            tracksData = data;
            if (data.fps) videoFps = data.fps;
            rois = data.rois || [];
            renderTracksList();
            updateOverlay();
        })
        .catch(err => showStatus('Error loading tracks: ' + err.message, 'error'));
}

// ---------------------------------------------------------------
// Frame loading (canvas-based)
// ---------------------------------------------------------------
function loadFrame(frameNum) {
    frameNum = Math.max(0, Math.min(frameNum, totalFrames - 1));
    currentFrame = frameNum;
    frameReady = false;
    updateFrameInfo();

    const img = new Image();
    img.onload = () => {
        frameReady = true;
        frameCtx.drawImage(img, 0, 0, frameCanvas.width, frameCanvas.height);
        updateOverlay();
    };
    img.onerror = () => {
        frameCtx.fillStyle = '#111';
        frameCtx.fillRect(0, 0, frameCanvas.width, frameCanvas.height);
        frameCtx.fillStyle = '#666';
        frameCtx.font = '24px sans-serif';
        frameCtx.fillText('Frame not available', 40, 40);
    };
    img.src = `/frame/${currentSession}/${frameNum}`;
}

function playVideo() {
    if (isPlaying) return;
    isPlaying = true;
    document.getElementById('playPause').textContent = 'Pause';

    function playStep() {
        if (!isPlaying) return;
        if (currentFrame >= totalFrames - 1) { pauseVideo(); return; }
        if (frameReady) {
            frameReady = false;
            loadFrame(currentFrame + 1);
        }
        playTimer = requestAnimationFrame(playStep);
    }
    playTimer = requestAnimationFrame(playStep);
}

function pauseVideo() {
    isPlaying = false;
    document.getElementById('playPause').textContent = 'Play';
    if (playTimer) { cancelAnimationFrame(playTimer); playTimer = null; }
}

// ---------------------------------------------------------------
// Overlay drawing
// ---------------------------------------------------------------
function updateOverlay() {
    if (!overlayCtx || !tracksData) return;
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    const tracks = getFilteredTracks();

    tracks.forEach(track => {
        const fd = track.frames?.find(f => f.frame === currentFrame);
        if (!fd || !fd.bbox) return;

        const [x1, y1, x2, y2] = fd.bbox;
        const isSelected = track.track_id === selectedTrackId;

        // Bbox
        overlayCtx.strokeStyle = isSelected ? '#FFFF00' : (CLASS_COLORS[track.class] || '#FFFFFF');
        overlayCtx.lineWidth = isSelected ? 3 : 2;
        overlayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Label
        const label = `#${track.track_id} ${track.class || ''} ${(fd.conf || 0).toFixed(2)}`;
        overlayCtx.font = '14px Arial';
        const lw = overlayCtx.measureText(label).width;
        overlayCtx.fillStyle = 'rgba(0,0,0,0.6)';
        overlayCtx.fillRect(x1, y1 - 20, lw + 6, 20);
        overlayCtx.fillStyle = isSelected ? '#FFFF00' : (CLASS_COLORS[track.class] || '#FFFFFF');
        overlayCtx.fillText(label, x1 + 3, y1 - 5);

        // Resize handles for selected track
        if (isSelected) {
            drawResizeHandles(x1, y1, x2, y2);
        }
    });

    // Draw ROIs
    rois.forEach(roi => {
        const isSelected = selectedRoiIds.has(roi.id || roi.name);
        overlayCtx.strokeStyle = isSelected ? 'rgba(78, 204, 163, 1.0)' : 'rgba(78, 204, 163, 0.4)';
        overlayCtx.lineWidth = isSelected ? 3 : 2;
        overlayCtx.setLineDash([6, 4]);
        const pts = roi.points || [];
        if (roi.type === 'rect' && pts.length >= 2) {
            const rx = Math.min(pts[0].x, pts[1].x);
            const ry = Math.min(pts[0].y, pts[1].y);
            const rw = Math.abs(pts[1].x - pts[0].x);
            const rh = Math.abs(pts[1].y - pts[0].y);
            overlayCtx.strokeRect(rx, ry, rw, rh);
        } else if (pts.length >= 3) {
            overlayCtx.beginPath();
            overlayCtx.moveTo(pts[0].x, pts[0].y);
            for (let i = 1; i < pts.length; i++) overlayCtx.lineTo(pts[i].x, pts[i].y);
            overlayCtx.closePath();
            overlayCtx.stroke();
        }
        overlayCtx.setLineDash([]);
    });
}

function drawResizeHandles(x1, y1, x2, y2) {
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    const positions = {
        tl: [x1, y1], tc: [mx, y1], tr: [x2, y1],
        ml: [x1, my],              mr: [x2, my],
        bl: [x1, y2], bc: [mx, y2], br: [x2, y2]
    };

    overlayCtx.fillStyle = '#FFFF00';
    overlayCtx.strokeStyle = '#000';
    overlayCtx.lineWidth = 1;
    for (const [, [hx, hy]] of Object.entries(positions)) {
        overlayCtx.fillRect(hx - HANDLE_SIZE / 2, hy - HANDLE_SIZE / 2, HANDLE_SIZE, HANDLE_SIZE);
        overlayCtx.strokeRect(hx - HANDLE_SIZE / 2, hy - HANDLE_SIZE / 2, HANDLE_SIZE, HANDLE_SIZE);
    }
}

// ---------------------------------------------------------------
// ROI filtering
// ---------------------------------------------------------------
function getFilteredTracks() {
    let tracks = tracksData?.tracks || [];
    if (selectedRoiIds.size > 0) {
        tracks = tracks.filter(track => trackInSelectedRois(track));
    } else if (roiFilterEnabled && rois.length > 0) {
        tracks = tracks.filter(track => {
            return track.frames?.some(fd => {
                if (!fd.bbox) return false;
                return rois.some(roi => bboxInRoi(fd.bbox, roi));
            });
        });
    }
    return tracks;
}

function bboxInRoi(bbox, roi) {
    const cx = (bbox[0] + bbox[2]) / 2;
    const cy = (bbox[1] + bbox[3]) / 2;
    const pts = roi.points || [];
    if (roi.type === 'rect' && pts.length >= 2) {
        const x1 = Math.min(pts[0].x, pts[1].x), y1 = Math.min(pts[0].y, pts[1].y);
        const x2 = Math.max(pts[0].x, pts[1].x), y2 = Math.max(pts[0].y, pts[1].y);
        return cx >= x1 && cx <= x2 && cy >= y1 && cy <= y2;
    } else if (pts.length >= 3) {
        let inside = false, j = pts.length - 1;
        for (let i = 0; i < pts.length; i++) {
            const xi = pts[i].x, yi = pts[i].y;
            const xj = pts[j].x, yj = pts[j].y;
            if ((yi > cy) !== (yj > cy) && (cx < (xj - xi) * (cy - yi) / (yj - yi) + xi))
                inside = !inside;
            j = i;
        }
        return inside;
    }
    return false;
}

function trackInSelectedRois(track) {
    if (selectedRoiIds.size === 0) return true;
    const activeRois = rois.filter(r => selectedRoiIds.has(r.id || r.name));
    return track.frames?.some(fd => {
        if (!fd.bbox) return false;
        return activeRois.some(roi => bboxInRoi(fd.bbox, roi));
    });
}

function distToSegment(p, a, b) {
    const dx = b.x - a.x, dy = b.y - a.y;
    const lenSq = dx * dx + dy * dy;
    if (lenSq === 0) return Math.sqrt((p.x - a.x) ** 2 + (p.y - a.y) ** 2);
    let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / lenSq;
    t = Math.max(0, Math.min(1, t));
    const projX = a.x + t * dx, projY = a.y + t * dy;
    return Math.sqrt((p.x - projX) ** 2 + (p.y - projY) ** 2);
}

// ---------------------------------------------------------------
// Overlay interaction: click to select, drag to resize bbox
// ---------------------------------------------------------------
function setupOverlayInteraction() {
    overlayCanvas.style.pointerEvents = 'auto';

    overlayCanvas.addEventListener('mousedown', e => {
        const pos = canvasCoords(e);
        if (!tracksData) return;

        // Check if we clicked on a resize handle of the selected track
        if (selectedTrackId !== null) {
            const handle = hitTestHandle(pos);
            if (handle) {
                const track = tracksData.tracks.find(t => t.track_id === selectedTrackId);
                const fd = track?.frames?.find(f => f.frame === currentFrame);
                if (fd && fd.bbox) {
                    resizeHandle = handle;
                    resizeTrackId = selectedTrackId;
                    resizeOrigBbox = [...fd.bbox];
                    resizeStart = pos;
                    e.preventDefault();
                    return;
                }
            }

            // Check if clicked inside selected bbox (drag move)
            const selTrack = tracksData.tracks.find(t => t.track_id === selectedTrackId);
            const selFd = selTrack?.frames?.find(f => f.frame === currentFrame);
            if (selFd && selFd.bbox) {
                const [bx1, by1, bx2, by2] = selFd.bbox;
                if (pos.x >= bx1 && pos.x <= bx2 && pos.y >= by1 && pos.y <= by2) {
                    isDraggingBbox = true;
                    resizeTrackId = selectedTrackId;
                    resizeOrigBbox = [...selFd.bbox];
                    resizeStart = pos;
                    e.preventDefault();
                    return;
                }
            }
        }

        // Check if clicked on an ROI to toggle selection
        for (const roi of rois) {
            const pts = roi.points || [];
            const roiId = roi.id || roi.name;
            if (roi.type === 'rect' && pts.length >= 2) {
                const x1 = Math.min(pts[0].x, pts[1].x), y1 = Math.min(pts[0].y, pts[1].y);
                const x2 = Math.max(pts[0].x, pts[1].x), y2 = Math.max(pts[0].y, pts[1].y);
                // Only check the border area (within 10px of the edge)
                const margin = 10;
                const onBorder = (pos.x >= x1 - margin && pos.x <= x2 + margin && pos.y >= y1 - margin && pos.y <= y2 + margin) &&
                    !(pos.x >= x1 + margin && pos.x <= x2 - margin && pos.y >= y1 + margin && pos.y <= y2 - margin);
                if (onBorder) {
                    if (selectedRoiIds.has(roiId)) {
                        selectedRoiIds.delete(roiId);
                    } else {
                        selectedRoiIds.add(roiId);
                    }
                    renderTracksList();
                    updateOverlay();
                    return;
                }
            } else if (pts.length >= 3) {
                // For polygons, check if near any edge
                for (let i = 0; i < pts.length; i++) {
                    const p1 = pts[i], p2 = pts[(i + 1) % pts.length];
                    const dist = distToSegment(pos, p1, p2);
                    if (dist < 10) {
                        if (selectedRoiIds.has(roiId)) {
                            selectedRoiIds.delete(roiId);
                        } else {
                            selectedRoiIds.add(roiId);
                        }
                        renderTracksList();
                        updateOverlay();
                        return;
                    }
                }
            }
        }

        // Click on a bbox to select it
        const tracks = getFilteredTracks();
        let clicked = null;
        for (const track of tracks) {
            const fd = track.frames?.find(f => f.frame === currentFrame);
            if (!fd || !fd.bbox) continue;
            const [bx1, by1, bx2, by2] = fd.bbox;
            if (pos.x >= bx1 && pos.x <= bx2 && pos.y >= by1 && pos.y <= by2) {
                clicked = track.track_id;
            }
        }
        if (clicked !== null) {
            selectTrack(clicked);
        } else {
            selectedTrackId = null;
            renderTracksList();
            updateSelectedTrackInfo();
            updateActionButtons();
            updateOverlay();
        }
    });

    overlayCanvas.addEventListener('mousemove', e => {
        const pos = canvasCoords(e);

        if (resizeHandle && resizeOrigBbox) {
            const dx = pos.x - resizeStart.x;
            const dy = pos.y - resizeStart.y;
            const newBbox = applyResize(resizeOrigBbox, resizeHandle, dx, dy);
            updateTrackBbox(resizeTrackId, currentFrame, newBbox);
            updateOverlay();
            return;
        }

        if (isDraggingBbox && resizeOrigBbox) {
            const dx = pos.x - resizeStart.x;
            const dy = pos.y - resizeStart.y;
            const [ox1, oy1, ox2, oy2] = resizeOrigBbox;
            updateTrackBbox(resizeTrackId, currentFrame, [ox1 + dx, oy1 + dy, ox2 + dx, oy2 + dy]);
            updateOverlay();
            return;
        }

        // Cursor hint
        if (selectedTrackId !== null) {
            const handle = hitTestHandle(pos);
            if (handle) {
                overlayCanvas.style.cursor = handleCursor(handle);
            } else {
                const t = tracksData?.tracks?.find(t => t.track_id === selectedTrackId);
                const fd = t?.frames?.find(f => f.frame === currentFrame);
                if (fd && fd.bbox) {
                    const [bx1, by1, bx2, by2] = fd.bbox;
                    overlayCanvas.style.cursor = (pos.x >= bx1 && pos.x <= bx2 && pos.y >= by1 && pos.y <= by2) ? 'move' : 'default';
                } else {
                    overlayCanvas.style.cursor = 'default';
                }
            }
        } else {
            overlayCanvas.style.cursor = 'default';
        }
    });

    overlayCanvas.addEventListener('mouseup', () => {
        if ((resizeHandle || isDraggingBbox) && resizeTrackId !== null) {
            // POST the bbox update to the server
            const track = tracksData?.tracks?.find(t => t.track_id === resizeTrackId);
            const fd = track?.frames?.find(f => f.frame === currentFrame);
            if (fd && fd.bbox) {
                fetch(`/api/session/${currentSession}/bbox`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        track_id: resizeTrackId,
                        frame: currentFrame,
                        bbox: fd.bbox.map(v => Math.round(v * 10) / 10)
                    })
                });
            }
        }
        resizeHandle = null;
        resizeTrackId = null;
        resizeOrigBbox = null;
        resizeStart = null;
        isDraggingBbox = false;
    });
}

function canvasCoords(e) {
    const rect = overlayCanvas.getBoundingClientRect();
    const sx = overlayCanvas.width / rect.width;
    const sy = overlayCanvas.height / rect.height;
    return {
        x: (e.clientX - rect.left) * sx,
        y: (e.clientY - rect.top) * sy
    };
}

function hitTestHandle(pos) {
    if (selectedTrackId === null || !tracksData) return null;
    const track = tracksData.tracks.find(t => t.track_id === selectedTrackId);
    const fd = track?.frames?.find(f => f.frame === currentFrame);
    if (!fd || !fd.bbox) return null;

    const [x1, y1, x2, y2] = fd.bbox;
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    const positions = {
        tl: [x1, y1], tc: [mx, y1], tr: [x2, y1],
        ml: [x1, my],              mr: [x2, my],
        bl: [x1, y2], bc: [mx, y2], br: [x2, y2]
    };

    const threshold = HANDLE_SIZE + 4;
    for (const [name, [hx, hy]] of Object.entries(positions)) {
        if (Math.abs(pos.x - hx) < threshold && Math.abs(pos.y - hy) < threshold) {
            return name;
        }
    }
    return null;
}

function applyResize(orig, handle, dx, dy) {
    let [x1, y1, x2, y2] = orig;
    switch (handle) {
        case 'tl': x1 += dx; y1 += dy; break;
        case 'tc': y1 += dy; break;
        case 'tr': x2 += dx; y1 += dy; break;
        case 'ml': x1 += dx; break;
        case 'mr': x2 += dx; break;
        case 'bl': x1 += dx; y2 += dy; break;
        case 'bc': y2 += dy; break;
        case 'br': x2 += dx; y2 += dy; break;
    }
    // Ensure min size
    if (x2 - x1 < 10) x2 = x1 + 10;
    if (y2 - y1 < 10) y2 = y1 + 10;
    return [x1, y1, x2, y2];
}

function handleCursor(handle) {
    const map = {
        tl: 'nw-resize', tc: 'n-resize', tr: 'ne-resize',
        ml: 'w-resize', mr: 'e-resize',
        bl: 'sw-resize', bc: 's-resize', br: 'se-resize'
    };
    return map[handle] || 'default';
}

function updateTrackBbox(trackId, frame, bbox) {
    if (!tracksData) return;
    const track = tracksData.tracks.find(t => t.track_id === trackId);
    if (!track) return;
    const fd = track.frames?.find(f => f.frame === frame);
    if (fd) fd.bbox = bbox;
}

// ---------------------------------------------------------------
// Controls setup
// ---------------------------------------------------------------
function setupControls() {
    const playPauseBtn = document.getElementById('playPause');
    const prevBtn = document.getElementById('prevFrame');
    const nextBtn = document.getElementById('nextFrame');
    const scrubber = document.getElementById('frameScrubber');
    const saveBtn = document.getElementById('saveBtn');
    const deleteBtn = document.getElementById('deleteTrack');
    const changeClassBtn = document.getElementById('changeClass');
    const splitBtn = document.getElementById('splitTrack');
    const mergeBtn = document.getElementById('mergeTrack');
    const roiCheck = document.getElementById('roiFilterCheck');

    if (playPauseBtn) playPauseBtn.addEventListener('click', () => { isPlaying ? pauseVideo() : playVideo(); });
    if (prevBtn) prevBtn.addEventListener('click', () => { pauseVideo(); loadFrame(currentFrame - 1); });
    if (nextBtn) nextBtn.addEventListener('click', () => { pauseVideo(); loadFrame(currentFrame + 1); });

    if (scrubber) {
        scrubber.addEventListener('input', () => {
            pauseVideo();
            loadFrame(parseInt(scrubber.value));
        });
    }

    if (saveBtn) saveBtn.addEventListener('click', saveAnnotations);
    if (deleteBtn) deleteBtn.addEventListener('click', deleteSelectedTrack);
    if (changeClassBtn) changeClassBtn.addEventListener('click', changeTrackClass);
    if (splitBtn) splitBtn.addEventListener('click', splitSelectedTrack);
    if (mergeBtn) mergeBtn.addEventListener('click', mergeSelectedTrack);

    if (roiCheck) {
        roiCheck.addEventListener('change', () => {
            roiFilterEnabled = roiCheck.checked;
            if (roiCheck.checked) {
                rois.forEach(r => selectedRoiIds.add(r.id || r.name));
            } else {
                selectedRoiIds.clear();
            }
            renderTracksList();
            updateOverlay();
        });
    }

    // Filter / sort
    const filterInput = document.getElementById('filterTracks');
    const sortSelect = document.getElementById('sortTracks');
    if (filterInput) filterInput.addEventListener('input', renderTracksList);
    if (sortSelect) sortSelect.addEventListener('change', renderTracksList);
}

function setupKeyboardShortcuts() {
    document.addEventListener('keydown', e => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

        // Ctrl+S to save
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            e.preventDefault();
            saveAnnotations();
            return;
        }

        switch (e.key.toLowerCase()) {
            case 'd': if (selectedTrackId) deleteSelectedTrack(); break;
            case 'c': if (selectedTrackId) changeTrackClass(); break;
            case 's': if (selectedTrackId) splitSelectedTrack(); break;
            case 'm': if (selectedTrackId) mergeSelectedTrack(); break;
            case ' ':
                e.preventDefault();
                isPlaying ? pauseVideo() : playVideo();
                break;
            case 'arrowleft':
                e.preventDefault();
                pauseVideo();
                loadFrame(currentFrame - 1);
                break;
            case 'arrowright':
                e.preventDefault();
                pauseVideo();
                loadFrame(currentFrame + 1);
                break;
        }
    });
}

// ---------------------------------------------------------------
// Track list rendering
// ---------------------------------------------------------------
function renderTracksList() {
    const container = document.getElementById('tracksList');
    let tracks = getFilteredTracks();

    if (tracks.length === 0) {
        container.innerHTML = '<p class="loading">No tracks found</p>';
        return;
    }

    const filterText = (document.getElementById('filterTracks')?.value || '').toLowerCase();
    const sortBy = document.getElementById('sortTracks')?.value || 'id';

    let filtered = tracks.filter(t => {
        const tid = String(t.track_id || '');
        const cls = (t.class || '').toLowerCase();
        return tid.includes(filterText) || cls.includes(filterText);
    });

    filtered.sort((a, b) => {
        switch (sortBy) {
            case 'id': return (a.track_id || 0) - (b.track_id || 0);
            case 'class': return (a.class || '').localeCompare(b.class || '');
            case 'confidence':
                return (b.avg_confidence || b.frames?.[0]?.conf || 0) - (a.avg_confidence || a.frames?.[0]?.conf || 0);
            case 'length': return (b.frames?.length || 0) - (a.frames?.length || 0);
            default: return 0;
        }
    });

    container.innerHTML = filtered.map(track => {
        const frames = track.frames || [];
        const avgConf = frames.length > 0
            ? (frames.reduce((s, f) => s + (f.conf || 0), 0) / frames.length).toFixed(2)
            : '0.00';
        return `
            <div class="track-item ${selectedTrackId === track.track_id ? 'selected' : ''}"
                 data-track-id="${track.track_id}"
                 onclick="selectTrack(${track.track_id})">
                <div class="track-item-header">
                    <span class="track-id">#${track.track_id}</span>
                    <span class="track-class">${track.class || 'unknown'}</span>
                </div>
                <div class="track-info">
                    Frames: ${frames.length} | Conf: ${avgConf} | ${track.start_frame || 0}-${track.end_frame || 0}
                    ${track.needs_review ? ' | <span style="color:#ffc107">Review</span>' : ''}
                </div>
            </div>`;
    }).join('');
}

window.selectTrack = function(trackId) {
    selectedTrackId = trackId;

    // Jump to track's first frame if not in range
    const track = tracksData?.tracks?.find(t => t.track_id === trackId);
    if (track) {
        const hasFrameNow = track.frames?.some(f => f.frame === currentFrame);
        if (!hasFrameNow && track.start_frame !== undefined) {
            loadFrame(track.start_frame);
        }
    }

    renderTracksList();
    updateSelectedTrackInfo();
    updateActionButtons();
    updateOverlay();
};

function updateSelectedTrackInfo() {
    const info = document.getElementById('selectedTrackInfo');
    if (!selectedTrackId || !tracksData) {
        info.innerHTML = '<p>No track selected. Click a track to select it.</p>';
        return;
    }
    const track = tracksData.tracks.find(t => t.track_id === selectedTrackId);
    if (!track) { info.innerHTML = '<p>Track not found</p>'; return; }
    const frames = track.frames || [];
    info.innerHTML = `
        <p><strong>Track #${track.track_id}</strong> &mdash; ${track.class || 'unknown'}</p>
        <p>Frames: ${frames.length} | Range: ${track.start_frame || 0} - ${track.end_frame || 0}</p>
        <p>Avg Confidence: ${track.avg_confidence || 'N/A'}</p>
    `;
}

function updateActionButtons() {
    const has = selectedTrackId !== null;
    document.getElementById('deleteTrack').disabled = !has;
    document.getElementById('changeClass').disabled = !has;
    document.getElementById('splitTrack').disabled = !has;
    document.getElementById('mergeTrack').disabled = !has;
}

function updateFrameInfo() {
    document.getElementById('currentFrame').textContent = currentFrame;
    const scrubber = document.getElementById('frameScrubber');
    if (scrubber) scrubber.value = currentFrame;

    const secs = videoFps > 0 ? currentFrame / videoFps : 0;
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    document.getElementById('currentTime').textContent = `${m}:${s.toString().padStart(2, '0')}`;
}

// ---------------------------------------------------------------
// Track actions
// ---------------------------------------------------------------
function deleteSelectedTrack() {
    if (!selectedTrackId) return;
    if (!confirm(`Delete track #${selectedTrackId}?`)) return;
    performAction('delete', selectedTrackId);
}

function changeTrackClass() {
    if (!selectedTrackId) return;
    const nc = prompt('Enter new class:', 'car');
    if (!nc) return;
    performAction('change_class', selectedTrackId, { new_class: nc });
}

function splitSelectedTrack() {
    if (!selectedTrackId) return;
    const frame = parseInt(prompt('Split at frame:', currentFrame));
    if (isNaN(frame)) return;
    performAction('split', selectedTrackId, { frame });
}

function mergeSelectedTrack() {
    if (!selectedTrackId) return;
    const tid = parseInt(prompt('Merge into track ID:', ''));
    if (isNaN(tid)) return;
    performAction('merge', selectedTrackId, { target_track_id: tid });
}

function performAction(action, trackId, data = {}) {
    fetch(`/api/session/${currentSession}/action`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ action, track_id: trackId, data })
    })
    .then(r => r.json())
    .then(result => {
        if (result.error) { showStatus('Error: ' + result.error, 'error'); return; }
        tracksData = result.tracks || tracksData;
        rois = tracksData.rois || [];
        selectedTrackId = null;
        renderTracksList();
        updateSelectedTrackInfo();
        updateActionButtons();
        updateOverlay();
        showStatus('Action completed', 'success');
    })
    .catch(err => showStatus('Error: ' + err.message, 'error'));
}

// ---------------------------------------------------------------
// Save
// ---------------------------------------------------------------
function saveAnnotations() {
    if (!tracksData) { showStatus('No tracks to save', 'error'); return; }

    // Include active ROI filter in saved data
    tracksData.active_roi_ids = [...selectedRoiIds];

    fetch(`/api/session/${currentSession}/tracks`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(tracksData)
    })
    .then(r => r.json())
    .then(result => {
        if (result.error) { showStatus('Error saving: ' + result.error, 'error'); return; }
        showStatus('Annotations saved!', 'success');
    })
    .catch(err => showStatus('Error saving: ' + err.message, 'error'));
}

// ---------------------------------------------------------------
// Status
// ---------------------------------------------------------------
function showStatus(message, type = 'success') {
    const el = document.getElementById('statusMessage');
    el.textContent = message;
    el.className = `status-message ${type} show`;
    setTimeout(() => el.classList.remove('show'), 3000);
}
