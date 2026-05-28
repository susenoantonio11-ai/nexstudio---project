"""
Sen1Floods11 Dataset Loader
============================
Sen1Floods11 (Bonafilia et al., 2020) adalah benchmark dataset untuk flood
mapping dengan paired Sentinel-1 SAR + Sentinel-2 optical + ground truth labels.

Total: 4,831 image patches (512×512 pixels) dari 11 flood events globally.

Dataset structure:
    /sen1floods11/
        S1Hand/      - Sentinel-1 SAR VV+VH (uint16, 2 bands)
        S2Hand/      - Sentinel-2 RGB+NIR (uint16, 4 bands)
        LabelHand/   - Ground truth flood mask (uint8, 0=non-flood, 1=flood, -1=invalid)

Reference:
    Bonafilia, D., et al. (2020). Sen1Floods11: a georeferenced dataset
    to train and test deep learning flood algorithms for Sentinel-1.
    CVPR Workshops.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import os


def _try_torch():
    try:
        import torch
        from torch.utils.data import Dataset
        return torch, Dataset
    except ImportError:
        return None, None


_TORCH, _TDataset = _try_torch()


# ==================================================================
# DATASET CLASS (only if torch available)
# ==================================================================
if _TORCH is not None:

    class Sen1Floods11Dataset(_TDataset):
        """
        PyTorch Dataset wrapper untuk Sen1Floods11.

        Args:
            root_dir: path ke folder berisi S1Hand/, S2Hand/, LabelHand/
            split: 'train' / 'val' / 'test' (kalau ada split file)
            modality: 's1_only' / 's2_only' / 'fusion' (s1+s2 stacked)
            patch_size: ukuran patch (default 512, native dataset size)
            transform: optional callable untuk augmentation
        """

        def __init__(
            self,
            root_dir: str,
            split: str = "train",
            modality: str = "fusion",
            patch_size: int = 512,
            transform=None,
            split_file: Optional[str] = None,
        ):
            self.root = Path(root_dir)
            self.split = split
            self.modality = modality
            self.patch_size = patch_size
            self.transform = transform

            self.s1_dir = self.root / "S1Hand"
            self.s2_dir = self.root / "S2Hand"
            self.label_dir = self.root / "LabelHand"

            # Discover patches
            if split_file and Path(split_file).exists():
                with open(split_file) as f:
                    self.patch_ids = [line.strip() for line in f if line.strip()]
            else:
                # Auto-discover: gunakan label files sebagai source of truth
                if self.label_dir.exists():
                    self.patch_ids = sorted([p.stem for p in self.label_dir.glob("*.tif")])
                else:
                    self.patch_ids = []

        def __len__(self) -> int:
            return len(self.patch_ids)

        def __getitem__(self, idx: int) -> Tuple:
            try:
                import rasterio
            except ImportError:
                raise RuntimeError("rasterio required: pip install rasterio")

            patch_id = self.patch_ids[idx]
            label_path = self.label_dir / f"{patch_id}.tif"
            with rasterio.open(label_path) as src:
                label = src.read(1)  # (H, W), values: 0/1/-1

            if self.modality in ("s1_only", "fusion"):
                with rasterio.open(self.s1_dir / f"{patch_id}.tif") as src:
                    s1 = src.read()  # (2, H, W) - VV, VH
            if self.modality in ("s2_only", "fusion"):
                with rasterio.open(self.s2_dir / f"{patch_id}.tif") as src:
                    s2 = src.read()  # (4 atau 13, H, W)

            if self.modality == "s1_only":
                image = s1
            elif self.modality == "s2_only":
                image = s2
            else:
                # Stack S1 (2 bands) + S2 (first 4: blue, green, red, nir) = 6 bands
                image = _np_stack_arrays(s1, s2[:4])

            # Convert to tensor
            image_tensor = _TORCH.from_numpy(image).float()
            label_tensor = _TORCH.from_numpy(label).long()
            # Mask invalid pixels (-1 in label)
            valid_mask = (label_tensor != -1)
            label_tensor = (label_tensor == 1).long()  # binary: 1=flood

            sample = {
                "image": image_tensor,
                "label": label_tensor,
                "valid_mask": valid_mask,
                "patch_id": patch_id,
            }
            if self.transform:
                sample = self.transform(sample)
            return sample


def _np_stack_arrays(*arrays):
    """Helper: stack arrays along axis 0."""
    import numpy as np
    return np.concatenate(arrays, axis=0)


# ==================================================================
# DATASET INFO + DOWNLOAD HELPERS
# ==================================================================
def get_dataset_info() -> Dict[str, Any]:
    """Return info tentang Sen1Floods11 dataset."""
    return {
        "name": "Sen1Floods11",
        "reference": "Bonafilia et al. (2020) — CVPR Workshops",
        "total_patches": 4831,
        "patch_size": 512,
        "modalities": {
            "s1_hand": "Sentinel-1 SAR (VV + VH polarization)",
            "s2_hand": "Sentinel-2 surface reflectance (4 bands: B,G,R,NIR)",
            "label_hand": "Ground truth: 0=non-flood, 1=flood, -1=invalid (cloud/no-data)",
        },
        "global_events": 11,
        "download_links": {
            "official": "https://github.com/cloudtostreet/Sen1Floods11",
            "weights_zenodo": "https://zenodo.org/record/4498086",
            "license": "CC-BY-4.0",
        },
        "recommended_use": (
            "Pretrain U-Net pada Sen1Floods11, lalu fine-tune pada AOI lokal. "
            "Transfer learning ini bisa mengurangi labeled data lokal dari 1000+ patches → 100-200 patches."
        ),
        "method_monitor": {
            "selected_method": "Sen1Floods11 pretraining",
            "why_chosen": (
                "Dataset terbesar dan paling beragam untuk flood mapping. "
                "Mencakup 11 event banjir global dengan kondisi yang bervariasi "
                "(kota, hutan, lahan basah). Pretrained features general → mudah di-fine-tune."
            ),
            "why_not_alternatives": [
                {"alternative": "WorldFloods", "reason_rejected": "Lebih kecil, hanya optical (Sentinel-2)"},
                {"alternative": "Train from scratch", "reason_rejected": "Membutuhkan 10x lebih banyak labeled data lokal"},
            ],
            "limitations": [
                "Bias geografis: 11 event tidak cover semua iklim",
                "Manual labels punya inherent uncertainty",
                "Dataset 24GB — perlu storage memadai",
            ],
            "reference_full": (
                "Bonafilia, D., Tellman, B., Anderson, T., & Issenberg, E. (2020). "
                "Sen1Floods11: a georeferenced dataset to train and test deep learning "
                "flood algorithms for Sentinel-1. In Proceedings of the IEEE/CVF Conference "
                "on Computer Vision and Pattern Recognition Workshops (pp. 210-211)."
            ),
        },
    }


def is_dataset_present(root_dir: str) -> Dict[str, Any]:
    """Cek apakah Sen1Floods11 dataset ada di lokasi tertentu."""
    root = Path(root_dir)
    if not root.exists():
        return {"present": False, "reason": f"Folder {root_dir} not found"}

    expected = ["S1Hand", "S2Hand", "LabelHand"]
    missing = [d for d in expected if not (root / d).exists()]
    if missing:
        return {
            "present": False,
            "reason": f"Missing subfolders: {missing}",
            "expected_structure": expected,
        }

    label_count = len(list((root / "LabelHand").glob("*.tif")))
    s1_count = len(list((root / "S1Hand").glob("*.tif")))
    s2_count = len(list((root / "S2Hand").glob("*.tif")))

    return {
        "present": True,
        "label_files": label_count,
        "s1_files": s1_count,
        "s2_files": s2_count,
        "consistency": label_count == s1_count == s2_count,
    }
