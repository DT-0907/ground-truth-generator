/**
 * Correction UI JavaScript
 * Handles video playback, track overlay, and corrections
 */

let currentSession = null;
let tracksData = null;
let selectedTrackId = null;
let currentFrame = 0;
let video = null;
let canvas = null;
let ctx = null;

// Vehicle class colors (COCO classes)
const CLASS_COLORS = {
    'car': '#FF6B6B',
    'truck': '#4ECDC4',
    'bus': '#45B7D1',
    'motorcycle': '#FFA07A',
    'bicycle': '#98D8C8',
    'person': '#F7DC6F'
};

// Make initReviewApp globally accessible
window.initReviewApp = function(sessionId, videoPath, trackPath) {
    currentSession = sessionId;
    
    video = document.getElementById('videoPlayer');
    canvas = document.getElementById('overlayCanvas');
    if (!video || !canvas) {
        console.error('Video player or canvas not found');
        return;
    }
    ctx = canvas.getContext('2d');
    
    // Load tracks
    loadTracks(trackPath);
    
    // Setup video event listeners
    setupVideoListeners();
    
    // Setup UI controls
    setupControls();
    
    // Setup keyboard shortcuts
    setupKeyboardShortcuts();
};

function loadTracks(trackPath) {
    if (!trackPath) {
        showStatus('No track file found', 'error');
        document.getElementById('tracksList').innerHTML = '<p class="loading">No tracks available</p>';
        return;
    }
    
    fetch(`/api/session/${currentSession}/tracks`)
        .then(response => response.json())
        .then(data => {
            tracksData = data;
            renderTracksList();
            updateOverlay();
        })
        .catch(error => {
            console.error('Error loading tracks:', error);
            showStatus('Error loading tracks: ' + error.message, 'error');
        });
}

function setupVideoListeners() {
    if (!video || !canvas) return;
    
    video.addEventListener('loadedmetadata', () => {
        // Set canvas size to match video
        if (video.videoWidth && video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        }
        
        // Update frame info
        updateFrameInfo();
    });
    
    video.addEventListener('timeupdate', () => {
        if (video && video.readyState) {
            currentFrame = Math.floor((video.currentTime || 0) * (tracksData?.fps || 30));
            updateFrameInfo();
            updateOverlay();
        }
    });
    
    video.addEventListener('play', () => {
        document.getElementById('playPause').textContent = '⏸ Pause';
    });
    
    video.addEventListener('pause', () => {
        document.getElementById('playPause').textContent = '▶ Play';
    });
}

function setupControls() {
    const playPauseBtn = document.getElementById('playPause');
    const prevFrameBtn = document.getElementById('prevFrame');
    const nextFrameBtn = document.getElementById('nextFrame');
    
    if (playPauseBtn) {
        playPauseBtn.addEventListener('click', () => {
            if (video && video.readyState) {
                if (video.paused) {
                    video.play();
                } else {
                    video.pause();
                }
            }
        });
    }
    
    if (prevFrameBtn) {
        prevFrameBtn.addEventListener('click', () => {
            if (video && video.readyState) {
                const fps = tracksData?.fps || 30;
                video.currentTime = Math.max(0, (video.currentTime || 0) - 1/fps);
            }
        });
    }
    
    if (nextFrameBtn) {
        nextFrameBtn.addEventListener('click', () => {
            if (video && video.readyState) {
                const fps = tracksData?.fps || 30;
                video.currentTime = Math.min(video.duration || 0, (video.currentTime || 0) + 1/fps);
            }
        });
    }
    
    const saveBtn = document.getElementById('saveBtn');
    const deleteBtn = document.getElementById('deleteTrack');
    const changeClassBtn = document.getElementById('changeClass');
    const splitBtn = document.getElementById('splitTrack');
    const mergeBtn = document.getElementById('mergeTrack');
    
    if (saveBtn) saveBtn.addEventListener('click', saveAnnotations);
    if (deleteBtn) deleteBtn.addEventListener('click', deleteSelectedTrack);
    if (changeClassBtn) changeClassBtn.addEventListener('click', changeTrackClass);
    if (splitBtn) splitBtn.addEventListener('click', splitSelectedTrack);
    if (mergeBtn) mergeBtn.addEventListener('click', mergeSelectedTrack);
}

function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Don't trigger if typing in input
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        
        switch(e.key.toLowerCase()) {
            case 'd':
                if (selectedTrackId) deleteSelectedTrack();
                break;
            case 'c':
                if (selectedTrackId) changeTrackClass();
                break;
            case 's':
                if (selectedTrackId) splitSelectedTrack();
                break;
            case 'm':
                if (selectedTrackId) mergeSelectedTrack();
                break;
            case ' ':
                e.preventDefault();
                if (video && video.readyState) {
                    if (video.paused) video.play();
                    else video.pause();
                }
                break;
            case 'arrowleft':
                e.preventDefault();
                if (video && video.readyState) {
                    const fps = tracksData?.fps || 30;
                    video.currentTime = Math.max(0, (video.currentTime || 0) - 1/fps);
                }
                break;
            case 'arrowright':
                e.preventDefault();
                if (video && video.readyState) {
                    const fps2 = tracksData?.fps || 30;
                    video.currentTime = Math.min(video.duration || 0, (video.currentTime || 0) + 1/fps2);
                }
                break;
        }
    });
}

function renderTracksList() {
    const container = document.getElementById('tracksList');
    const tracks = tracksData?.tracks || [];
    
    if (tracks.length === 0) {
        container.innerHTML = '<p class="loading">No tracks found</p>';
        return;
    }
    
    const filterText = document.getElementById('filterTracks')?.value.toLowerCase() || '';
    const sortBy = document.getElementById('sortTracks')?.value || 'id';
    
    // Filter and sort
    let filtered = tracks.filter(track => {
        const trackId = String(track.track_id || '');
        const className = (track.class || '').toLowerCase();
        return trackId.includes(filterText) || className.includes(filterText);
    });
    
    filtered.sort((a, b) => {
        switch(sortBy) {
            case 'id':
                return (a.track_id || 0) - (b.track_id || 0);
            case 'class':
                return (a.class || '').localeCompare(b.class || '');
            case 'confidence':
                const aConf = a.frames?.[0]?.conf || 0;
                const bConf = b.frames?.[0]?.conf || 0;
                return bConf - aConf;
            case 'length':
                return (b.frames?.length || 0) - (a.frames?.length || 0);
            default:
                return 0;
        }
    });
    
    container.innerHTML = filtered.map(track => {
        const frames = track.frames || [];
        const avgConf = frames.length > 0 
            ? (frames.reduce((sum, f) => sum + (f.conf || 0), 0) / frames.length).toFixed(2)
            : '0.00';
        
        return `
            <div class="track-item ${selectedTrackId === track.track_id ? 'selected' : ''}" 
                 data-track-id="${track.track_id}"
                 onclick="selectTrack(${track.track_id})">
                <div class="track-item-header">
                    <span class="track-id">Track #${track.track_id}</span>
                    <span class="track-class">${track.class || 'unknown'}</span>
                </div>
                <div class="track-info">
                    Frames: ${frames.length} | 
                    Confidence: ${avgConf} | 
                    Range: ${track.start_frame || 0} - ${track.end_frame || 0}
                </div>
            </div>
        `;
    }).join('');
}

// Make selectTrack globally accessible for onclick handlers
window.selectTrack = function(trackId) {
    selectedTrackId = trackId;
    renderTracksList();
    updateSelectedTrackInfo();
    updateActionButtons();
    updateOverlay();
};

function updateSelectedTrackInfo() {
    const info = document.getElementById('selectedTrackInfo');
    if (!selectedTrackId || !tracksData) {
        info.innerHTML = '<p>No track selected. Click a track in the list to select it.</p>';
        return;
    }
    
    const track = tracksData.tracks.find(t => t.track_id === selectedTrackId);
    if (!track) {
        info.innerHTML = '<p>Track not found</p>';
        return;
    }
    
    const frames = track.frames || [];
    info.innerHTML = `
        <p><strong>Track #${track.track_id}</strong></p>
        <p>Class: ${track.class || 'unknown'}</p>
        <p>Frames: ${frames.length}</p>
        <p>Frame range: ${track.start_frame || 0} - ${track.end_frame || 0}</p>
        <p>Lane: ${track.lane_id || 'N/A'}</p>
    `;
}

function updateActionButtons() {
    const hasSelection = selectedTrackId !== null;
    document.getElementById('deleteTrack').disabled = !hasSelection;
    document.getElementById('changeClass').disabled = !hasSelection;
    document.getElementById('splitTrack').disabled = !hasSelection;
    document.getElementById('mergeTrack').disabled = !hasSelection;
}

function updateFrameInfo() {
    if (!video || !video.readyState) return;
    
    const fps = tracksData?.fps || 30;
    const totalFrames = Math.floor((video.duration || 0) * fps);
    const currentFrameEl = document.getElementById('currentFrame');
    const totalFramesEl = document.getElementById('totalFrames');
    const currentTimeEl = document.getElementById('currentTime');
    
    if (currentFrameEl) currentFrameEl.textContent = currentFrame;
    if (totalFramesEl) totalFramesEl.textContent = totalFrames;
    
    if (currentTimeEl) {
        const minutes = Math.floor((video.currentTime || 0) / 60);
        const seconds = Math.floor((video.currentTime || 0) % 60);
        currentTimeEl.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
    }
}

function updateOverlay() {
    if (!ctx || !tracksData || !canvas || !video || !video.readyState) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const tracks = tracksData.tracks || [];
    const fps = tracksData.fps || 30;
    const frame = Math.floor((video.currentTime || 0) * fps);
    
    // Draw bboxes for current frame
    tracks.forEach(track => {
        const frameData = track.frames?.find(f => f.frame === frame);
        if (!frameData || !frameData.bbox) return;
        
        const [x1, y1, x2, y2] = frameData.bbox;
        const isSelected = track.track_id === selectedTrackId;
        
        // Draw bbox
        ctx.strokeStyle = isSelected ? '#FFFF00' : (CLASS_COLORS[track.class] || '#FFFFFF');
        ctx.lineWidth = isSelected ? 3 : 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        
        // Draw label
        ctx.fillStyle = isSelected ? '#FFFF00' : (CLASS_COLORS[track.class] || '#FFFFFF');
        ctx.font = '14px Arial';
        ctx.fillText(
            `#${track.track_id} ${track.class || ''} ${(frameData.conf || 0).toFixed(2)}`,
            x1,
            y1 - 5
        );
    });
}

function deleteSelectedTrack() {
    if (!selectedTrackId) return;
    
    if (!confirm(`Delete track #${selectedTrackId}?`)) return;
    
    performAction('delete', selectedTrackId);
}

function changeTrackClass() {
    if (!selectedTrackId) return;
    
    const newClass = prompt('Enter new class:', 'car');
    if (!newClass) return;
    
    performAction('change_class', selectedTrackId, { new_class: newClass });
}

function splitSelectedTrack() {
    if (!selectedTrackId) return;
    
    const frame = parseInt(prompt('Enter frame number to split at:', currentFrame));
    if (isNaN(frame)) return;
    
    performAction('split', selectedTrackId, { frame: frame });
}

function mergeSelectedTrack() {
    if (!selectedTrackId) return;
    
    const targetId = parseInt(prompt('Enter track ID to merge into:', ''));
    if (isNaN(targetId)) return;
    
    performAction('merge', selectedTrackId, { target_track_id: targetId });
}

function performAction(action, trackId, data = {}) {
    fetch(`/api/session/${currentSession}/action`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            action: action,
            track_id: trackId,
            data: data
        })
    })
    .then(response => response.json())
    .then(result => {
        if (result.error) {
            showStatus('Error: ' + result.error, 'error');
        } else {
            tracksData = result.tracks || tracksData;
            selectedTrackId = null;
            renderTracksList();
            updateSelectedTrackInfo();
            updateActionButtons();
            updateOverlay();
            showStatus('Action completed successfully', 'success');
        }
    })
    .catch(error => {
        showStatus('Error: ' + error.message, 'error');
    });
}

function saveAnnotations() {
    if (!tracksData) {
        showStatus('No tracks to save', 'error');
        return;
    }
    
    fetch(`/api/session/${currentSession}/tracks`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(tracksData)
    })
    .then(response => response.json())
    .then(result => {
        if (result.error) {
            showStatus('Error saving: ' + result.error, 'error');
        } else {
            showStatus('Annotations saved successfully!', 'success');
        }
    })
    .catch(error => {
        showStatus('Error saving: ' + error.message, 'error');
    });
}

function showStatus(message, type = 'success') {
    const statusEl = document.getElementById('statusMessage');
    statusEl.textContent = message;
    statusEl.className = `status-message ${type} show`;
    
    setTimeout(() => {
        statusEl.classList.remove('show');
    }, 3000);
}

// Filter and sort event listeners
document.addEventListener('DOMContentLoaded', () => {
    const filterInput = document.getElementById('filterTracks');
    const sortSelect = document.getElementById('sortTracks');
    
    if (filterInput) {
        filterInput.addEventListener('input', renderTracksList);
    }
    
    if (sortSelect) {
        sortSelect.addEventListener('change', renderTracksList);
    }
});

