"""
ImageDatasetBuilder
===================
Walk a folder of images and produce a clean dataset table:

  * scan recursively (folder name → label)
  * read width/height/mode/size_bytes via Pillow
  * detect duplicates by perceptual hash (Zauner, 2010 — pHash)
  * detect blurry images via Laplacian variance (Pertuz et al., 2013)
  * detect corrupted files (decode failures)
  * stratified random train/val/test split

No torch/tf required.
"""
from __future__ import annotations

import io
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def _phash_8x8(image_bytes: bytes) -> Optional[int]:
    """64-bit perceptual hash (Zauner 2010). Resize to 9×8, gradient sign."""
    if not HAS_PIL: return None
    try:
        im = Image.open(io.BytesIO(image_bytes)).convert("L").resize((9, 8))
        arr = np.array(im, dtype=int)
        diff = arr[:, 1:] > arr[:, :-1]   # 8×8 boolean
        bits = 0
        for v in diff.flatten():
            bits = (bits << 1) | int(bool(v))
        return int(bits)
    except Exception:
        return None


def _laplacian_var(image_bytes: bytes) -> Optional[float]:
    if not HAS_PIL: return None
    try:
        im = Image.open(io.BytesIO(image_bytes)).convert("L")
        a = np.array(im, dtype=float)
        L = -4 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
        return float(np.var(L))
    except Exception:
        return None


def _hamming_64(a: int, b: int) -> int:
    return bin((a ^ b) & ((1 << 64) - 1)).count("1")


class ImageDatasetBuilder:
    """Build a clean image dataset from a folder."""

    name = "ImageDatasetBuilder"
    domain = "computer_vision"
    citations = [
        "Zauner, C. (2010) — Implementation and Benchmarking of Perceptual Image Hash Functions.",
        "Pertuz et al. (2013) Pattern Recognition 46(5):1415–1432 — focus measure.",
    ]

    def build(self, root: str, blur_threshold: float = 100.0,
              duplicate_threshold: int = 6,
              splits: Tuple[float, float, float] = (0.70, 0.15, 0.15),
              seed: int = 42) -> Dict[str, Any]:
        t0 = time.perf_counter()
        root_path = Path(root)
        if not root_path.exists() or not root_path.is_dir():
            return {"status": "error", "model_name": self.name, "message": f"Folder not found: {root}"}

        rows: List[Dict[str, Any]] = []
        seen: List[Tuple[int, int]] = []  # (phash, row_index)
        labels: Dict[str, int] = {}
        ok_count = blur_count = corrupt_count = duplicate_count = 0

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        for p in sorted(root_path.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            try:
                blob = p.read_bytes()
            except Exception:
                continue
            label = p.parent.name if p.parent != root_path else "root"
            labels[label] = labels.get(label, 0) + 1
            row: Dict[str, Any] = {
                "path": str(p.relative_to(root_path)),
                "label": label,
                "size_bytes": len(blob),
                "is_corrupted": False,
                "is_blurry": False,
                "is_duplicate": False,
                "phash": None,
                "duplicate_of": None,
                "laplacian_focus": None,
                "width": None, "height": None,
            }
            if HAS_PIL:
                try:
                    with Image.open(io.BytesIO(blob)) as im:
                        row["width"], row["height"] = im.size
                        row["mode"] = im.mode
                except Exception:
                    row["is_corrupted"] = True
                    corrupt_count += 1
            ph = _phash_8x8(blob)
            row["phash"] = ph
            lv = _laplacian_var(blob)
            row["laplacian_focus"] = lv
            if lv is not None and lv < blur_threshold:
                row["is_blurry"] = True
                blur_count += 1
            if ph is not None:
                for prev_ph, prev_idx in seen:
                    if _hamming_64(prev_ph, ph) <= duplicate_threshold:
                        row["is_duplicate"] = True
                        row["duplicate_of"] = rows[prev_idx]["path"]
                        duplicate_count += 1
                        break
                seen.append((ph, len(rows)))
            if not row["is_corrupted"] and not row["is_blurry"] and not row["is_duplicate"]:
                ok_count += 1
            rows.append(row)

        # Stratified split
        rng = random.Random(seed)
        clean = [r for r in rows if not (r["is_corrupted"] or r["is_blurry"] or r["is_duplicate"])]
        by_label: Dict[str, List[Dict[str, Any]]] = {}
        for r in clean:
            by_label.setdefault(r["label"], []).append(r)
        train, val, test = [], [], []
        for lab, lst in by_label.items():
            rng.shuffle(lst)
            n = len(lst)
            n_tr = int(n * splits[0])
            n_va = int(n * splits[1])
            train.extend(lst[:n_tr])
            val.extend(lst[n_tr:n_tr + n_va])
            test.extend(lst[n_tr + n_va:])

        duration_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "status": "success",
            "model_name": self.name,
            "root": str(root),
            "n_total": len(rows),
            "n_ok": ok_count,
            "n_blurry": blur_count,
            "n_corrupted": corrupt_count,
            "n_duplicates": duplicate_count,
            "labels": labels,
            "rows_preview": rows[:30],
            "splits": {
                "train": len(train), "val": len(val), "test": len(test),
                "train_paths": [r["path"] for r in train[:50]],
                "val_paths":   [r["path"] for r in val[:50]],
                "test_paths":  [r["path"] for r in test[:50]],
            },
            "duration_ms": duration_ms,
            "method_monitor": {
                "method": "Folder walk + pHash dedup + Laplacian-variance blur + stratified random split",
                "formulas": [
                    "pHash bit i = 1 iff I(i_left) > I(i_right) on a resized 9×8 luminance grid",
                    "Hamming distance d_H = popcount(h_a XOR h_b)",
                    "Laplacian variance F = Var(∇²I)",
                ],
                "limitations": [
                    "pHash duplicate threshold defaults to 6/64 — tighten for visually-distinct duplicates.",
                    "Blur threshold (100) needs domain calibration; medical/satellite often need <50 or >200."
                ],
                "citations": self.citations,
            },
        }
