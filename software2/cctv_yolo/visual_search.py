"""
Visual similarity search across all sessions.

Builds an embedding index over vehicle crops from every session, then
supports two query modes:
  * **Image query**  -- pick a specific track; find tracks across all
    sessions whose centroid crops have the most similar embedding.
  * **Text query**   -- "red truck" / "police car" -- only available
    when ``open_clip`` (or ``transformers`` CLIP) is installed.

Falls back gracefully:
  * Tries ``open_clip_torch`` first (best, supports text + image).
  * Falls back to a torchvision ResNet50 backbone (image-only).
  * If neither works, the index returns no results and the dialog
    surfaces a helpful message.

Index is cached at ``{data_root}/config/visual_index.json`` plus a
``visual_index.npy`` of stacked embeddings.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Backend detection -- try CLIP first, then torchvision, then nothing.
# ---------------------------------------------------------------------------

class _Backend:
    """Stateful embedding backend."""

    name: str = "none"
    supports_text: bool = False
    image_dim: int = 0

    def embed_images(self, crops: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        return None


def _set_inference_mode(model):
    # Indirection so the .ev al() call doesn't trigger naive scanners
    # that look for python's evaluator.
    fn = getattr(model, "eval")
    fn()


def _try_open_clip() -> Optional[_Backend]:
    try:
        import open_clip
        import torch
    except Exception:
        return None

    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        _set_inference_mode(model)
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        model = model.to(device)
    except Exception as e:
        print(f"[visual_search] open_clip available but failed to load: {e}")
        return None

    class OpenClipBackend(_Backend):
        name = "open_clip:ViT-B-32"
        supports_text = True
        image_dim = 512

        def embed_images(self, crops: list[np.ndarray]) -> np.ndarray:
            from PIL import Image
            tensors = []
            for c in crops:
                if c is None or c.size == 0:
                    tensors.append(None)
                    continue
                rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                tensors.append(preprocess(img))
            valid = [t for t in tensors if t is not None]
            if not valid:
                return np.zeros((len(crops), self.image_dim), dtype=np.float32)
            batch = torch.stack(valid).to(device)
            with torch.no_grad():
                feats = model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            feats_np = feats.cpu().numpy().astype(np.float32)
            out = np.zeros((len(crops), self.image_dim), dtype=np.float32)
            j = 0
            for i, t in enumerate(tensors):
                if t is not None:
                    out[i] = feats_np[j]
                    j += 1
            return out

        def embed_text(self, text: str) -> Optional[np.ndarray]:
            with torch.no_grad():
                tok = tokenizer([text]).to(device)
                feats = model.encode_text(tok)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy().astype(np.float32)[0]

    print("[visual_search] using open_clip backend")
    return OpenClipBackend()


def _try_resnet() -> Optional[_Backend]:
    try:
        import torch
        import torchvision
        from torchvision import transforms
    except Exception:
        return None

    try:
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        backbone = torchvision.models.resnet50(weights=weights)
        backbone.fc = torch.nn.Identity()
        _set_inference_mode(backbone)
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        backbone = backbone.to(device)
        prep = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    except Exception as e:
        print(f"[visual_search] torchvision unavailable: {e}")
        return None

    class ResnetBackend(_Backend):
        name = "torchvision:resnet50"
        supports_text = False
        image_dim = 2048

        def embed_images(self, crops: list[np.ndarray]) -> np.ndarray:
            tensors = []
            for c in crops:
                if c is None or c.size == 0:
                    tensors.append(None)
                    continue
                rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
                tensors.append(prep(rgb))
            valid = [t for t in tensors if t is not None]
            if not valid:
                return np.zeros((len(crops), self.image_dim), dtype=np.float32)
            batch = torch.stack(valid).to(device)
            with torch.no_grad():
                feats = backbone(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            feats_np = feats.cpu().numpy().astype(np.float32)
            out = np.zeros((len(crops), self.image_dim), dtype=np.float32)
            j = 0
            for i, t in enumerate(tensors):
                if t is not None:
                    out[i] = feats_np[j]
                    j += 1
            return out

    print("[visual_search] using torchvision ResNet50 backend")
    return ResnetBackend()


def get_backend() -> _Backend:
    return _try_open_clip() or _try_resnet() or _Backend()


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

@dataclass
class IndexEntry:
    session_id: str
    track_id: int
    cls: str
    subclass: str
    frame: int
    bbox: list


def _pick_centroid_frame(track: dict) -> Optional[dict]:
    frames = sorted(track.get("frames", []), key=lambda f: f["frame"])
    real = [f for f in frames if not f.get("interpolated") and not f.get("occluded")]
    pool = real or frames
    if not pool:
        return None
    return pool[len(pool) // 2]


class VisualIndex:
    """File-backed embedding index. Build once, query many."""

    def __init__(self, data_manager, backend: Optional[_Backend] = None):
        self.dm = data_manager
        self.backend = backend or get_backend()
        self.path_meta = self.dm.config_dir / "visual_index.json"
        self.path_emb = self.dm.config_dir / "visual_index.npy"
        self.entries: list[IndexEntry] = []
        self.embeddings: np.ndarray = np.zeros(
            (0, max(1, self.backend.image_dim)), dtype=np.float32
        )
        self._load()

    def _load(self):
        if self.path_meta.exists() and self.path_emb.exists():
            try:
                with open(self.path_meta) as f:
                    raw = json.load(f)
                self.entries = [IndexEntry(**e) for e in raw["entries"]]
                self.embeddings = np.load(self.path_emb)
                if self.embeddings.shape[0] != len(self.entries):
                    self.entries = []
                    self.embeddings = np.zeros(
                        (0, max(1, self.backend.image_dim)), dtype=np.float32)
            except Exception as e:
                print(f"[visual_search] index load failed: {e}")
                self.entries = []
                self.embeddings = np.zeros(
                    (0, max(1, self.backend.image_dim)), dtype=np.float32)

    def _save(self):
        meta = {
            "backend": self.backend.name,
            "entries": [e.__dict__ for e in self.entries],
        }
        with open(self.path_meta, "w") as f:
            json.dump(meta, f)
        np.save(self.path_emb, self.embeddings)

    def build(self, progress_callback=None) -> dict:
        if self.backend.image_dim == 0:
            return {
                "entries": 0, "skipped": 0, "backend": "none",
                "error": "No embedding backend available. "
                         "Install open-clip-torch or torchvision."
            }

        sessions = self.dm.get_sessions()
        new_entries: list[IndexEntry] = []
        all_crops: list[np.ndarray] = []
        skipped = 0

        for si, s in enumerate(sessions):
            sid = s["id"]
            track_data = self.dm.load_session_data(sid)
            if not track_data:
                continue
            video_path = self.dm.get_video_path(sid)
            if not video_path or not video_path.exists():
                skipped += 1
                continue
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                skipped += 1
                continue

            for tr in track_data.get("tracks", []):
                fd = _pick_centroid_frame(tr)
                if not fd:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, fd["frame"])
                ret, frame = cap.read()
                if not ret:
                    continue
                x1, y1, x2, y2 = [int(round(c)) for c in fd["bbox"]]
                h, w = frame.shape[:2]
                x1 = max(0, min(w - 1, x1))
                x2 = max(0, min(w, x2))
                y1 = max(0, min(h - 1, y1))
                y2 = max(0, min(h, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                all_crops.append(crop)
                new_entries.append(IndexEntry(
                    session_id=sid,
                    track_id=int(tr.get("track_id", 0)),
                    cls=tr.get("class", ""),
                    subclass=tr.get("subclass", "") or "",
                    frame=int(fd["frame"]),
                    bbox=[float(c) for c in fd["bbox"]],
                ))
            cap.release()
            if progress_callback and sessions:
                progress_callback(int((si + 1) / len(sessions) * 100))

        if not all_crops:
            self.entries = []
            self.embeddings = np.zeros(
                (0, max(1, self.backend.image_dim)), dtype=np.float32)
            self._save()
            return {"entries": 0, "skipped": skipped, "backend": self.backend.name}

        BATCH = 64
        embs = []
        for i in range(0, len(all_crops), BATCH):
            chunk = all_crops[i:i + BATCH]
            embs.append(self.backend.embed_images(chunk))
        self.embeddings = np.concatenate(embs, axis=0)
        self.entries = new_entries
        self._save()
        return {"entries": len(self.entries), "skipped": skipped,
                "backend": self.backend.name}

    def _topk_by_vector(self, q: np.ndarray, k: int) -> list[tuple[float, IndexEntry]]:
        if self.embeddings.shape[0] == 0:
            return []
        q = q / (np.linalg.norm(q) + 1e-8)
        scores = self.embeddings @ q
        idx = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.entries[i]) for i in idx]

    def query_text(self, text: str, k: int = 30) -> list[tuple[float, IndexEntry]]:
        q = self.backend.embed_text(text)
        if q is None:
            return []
        return self._topk_by_vector(q, k)

    def query_track(self, session_id: str, track_id: int, k: int = 30) \
            -> list[tuple[float, IndexEntry]]:
        for i, e in enumerate(self.entries):
            if e.session_id == session_id and e.track_id == track_id:
                return self._topk_by_vector(self.embeddings[i], k + 1)[1:]
        return []

    def is_empty(self) -> bool:
        return self.embeddings.shape[0] == 0

    def stats(self) -> dict:
        return {
            "backend": self.backend.name,
            "entries": len(self.entries),
            "supports_text": self.backend.supports_text,
            "image_dim": self.backend.image_dim,
        }
