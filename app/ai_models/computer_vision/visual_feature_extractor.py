"""
VisualFeatureExtractor
======================
Pure-Python visual-feature extractor designed to run with only Pillow + numpy.
Falls back gracefully if scikit-image / opencv are missing. Produces a
fixed-length feature vector per image so MultimodalDataScienceEngine can
treat images as just-another-modality.

Features extracted per image (default):
  * Color histogram   : 3 channels × 8 bins = 24 features
                        Swain & Ballard (1991) — Color Indexing.
  * Edge density      : Sobel magnitude mean + std + percentile-90
                        (3 features). Sobel & Feldman (1968).
  * Texture           : Local Binary Pattern bins (10 features)
                        Ojala et al. (2002) IEEE TPAMI 24(7):971–987.
                        Falls back to gradient histogram if scikit-image
                        is not installed.
  * Spatial moments   : Hu invariant moments (7 features)
                        Hu (1962) IRE Trans. Inf. Theory 8(2):179–187.

Total default vector length = 44.
"""
from __future__ import annotations

import io
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from skimage.feature import local_binary_pattern
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _load_image(image_bytes: bytes, size: Tuple[int, int] = (128, 128)) -> Optional[np.ndarray]:
    if not HAS_PIL:
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(size)
        return np.array(img)
    except Exception:
        return None


def _color_hist(rgb: np.ndarray, bins: int = 8) -> List[float]:
    feats: List[float] = []
    for c in range(3):
        h, _ = np.histogram(rgb[:, :, c], bins=bins, range=(0, 256), density=False)
        h = h / max(h.sum(), 1)
        feats.extend(h.tolist())
    return feats


def _sobel_stats(gray: np.ndarray) -> List[float]:
    gx = np.zeros_like(gray, dtype=float)
    gy = np.zeros_like(gray, dtype=float)
    gx[:, 1:-1] = gray[:, 2:].astype(float) - gray[:, :-2].astype(float)
    gy[1:-1, :] = gray[2:, :].astype(float) - gray[:-2, :].astype(float)
    mag = np.sqrt(gx * gx + gy * gy)
    return [float(mag.mean()) / 255.0,
            float(mag.std()) / 255.0,
            float(np.percentile(mag, 90)) / 255.0]


def _lbp(gray: np.ndarray, n_bins: int = 10) -> List[float]:
    if HAS_SKIMAGE:
        try:
            lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
            h, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=False)
            h = h / max(h.sum(), 1)
            return h.tolist()
        except Exception:
            pass
    # Fallback: gradient orientation histogram
    gx = np.zeros_like(gray, dtype=float); gy = np.zeros_like(gray, dtype=float)
    gx[:, 1:-1] = gray[:, 2:].astype(float) - gray[:, :-2].astype(float)
    gy[1:-1, :] = gray[2:, :].astype(float) - gray[:-2, :].astype(float)
    ang = np.degrees(np.arctan2(gy, gx)) % 180
    h, _ = np.histogram(ang, bins=n_bins, range=(0, 180), density=False)
    h = h / max(h.sum(), 1)
    return h.tolist()


def _hu_moments(gray: np.ndarray) -> List[float]:
    """Hu invariant moments via raw central moments. Pure numpy."""
    g = gray.astype(float)
    g = g / max(g.sum(), 1)
    y, x = np.indices(g.shape)
    m10 = (x * g).sum(); m01 = (y * g).sum()
    xc = m10; yc = m01  # centroid (since we normalized)
    def mu(p, q):
        return (((x - xc) ** p) * ((y - yc) ** q) * g).sum()
    mu20, mu02, mu11 = mu(2, 0), mu(0, 2), mu(1, 1)
    mu30, mu03, mu21, mu12 = mu(3, 0), mu(0, 3), mu(2, 1), mu(1, 2)
    eta = lambda p, q, m: mu(p, q) / (m ** ((p + q) / 2 + 1) + 1e-12)
    m00 = g.sum() + 1e-12
    n20 = mu20 / (m00 ** 2);  n02 = mu02 / (m00 ** 2);  n11 = mu11 / (m00 ** 2)
    n30 = mu30 / (m00 ** 2.5); n03 = mu03 / (m00 ** 2.5); n21 = mu21 / (m00 ** 2.5); n12 = mu12 / (m00 ** 2.5)
    h1 = n20 + n02
    h2 = (n20 - n02) ** 2 + 4 * n11 ** 2
    h3 = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    h4 = (n30 + n12) ** 2 + (n21 + n03) ** 2
    h5 = ((n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
          + (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2))
    h6 = (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + 4 * n11 * (n30 + n12) * (n21 + n03)
    h7 = ((3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
          - (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2))
    moments = [h1, h2, h3, h4, h5, h6, h7]
    return [float(np.sign(m) * np.log10(abs(m) + 1e-12)) / 10.0 for m in moments]


class VisualFeatureExtractor:
    """Returns a fixed-length feature vector per image."""

    name = "VisualFeatureExtractor"
    domain = "computer_vision"
    feature_length = 24 + 3 + 10 + 7  # 44
    citations = [
        "Swain & Ballard (1991) Color Indexing — color histogram.",
        "Sobel, Feldman (1968) — Sobel operator.",
        "Ojala et al. (2002) IEEE TPAMI 24(7):971–987 — LBP.",
        "Hu (1962) IRE Trans. IT 8(2):179–187 — invariant moments.",
    ]

    def extract_one(self, image_bytes: bytes) -> Dict[str, Any]:
        t0 = time.perf_counter()
        img = _load_image(image_bytes)
        if img is None:
            return {"status": "error", "message": "Pillow not available or image unreadable", "duration_ms": int((time.perf_counter() - t0) * 1000)}
        gray = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)
        ch = _color_hist(img)
        sb = _sobel_stats(gray)
        lb = _lbp(gray)
        hu = _hu_moments(gray)
        vec = ch + sb + lb + hu
        return {
            "status": "success",
            "model_name": self.name,
            "feature_length": len(vec),
            "feature_vector": vec,
            "feature_blocks": {"color_hist": len(ch), "sobel_stats": len(sb), "lbp": len(lb), "hu_moments": len(hu)},
            "duration_ms": int((time.perf_counter() - t0) * 1000),
            "method_monitor": {
                "method": "Color hist + Sobel edge stats + LBP texture + Hu invariant moments",
                "why_used": "Lightweight, well-cited handcrafted descriptors that do not require torch/tf.",
                "formulas": [
                    "Color hist (Swain & Ballard 1991): h_c[b] = |{(x,y): I_c(x,y) ∈ B_b}| / N",
                    "Sobel magnitude: |∇I| = √(Gx² + Gy²)",
                    "LBP_{P,R}(c) = Σ_p s(I_p − I_c)·2^p, s(x)=1 if x≥0 else 0",
                    "Hu moment 1: η₂₀ + η₀₂",
                ],
                "limitations": ["Discriminates texture/colour, not high-level semantics; install torch + open_clip for embeddings."],
                "citations": self.citations,
            },
        }
