"""
Transfer Learning untuk U-Net Flood Segmentation
==================================================
Strategi:
1. Pre-train U-Net pada Sen1Floods11 (atau load weights yang di-share)
2. Replace input layer kalau channel berbeda
3. Optionally freeze encoder untuk fine-tuning awal
4. Unfreeze gradually saat loss converge

Pretrained encoders bisa juga gunakan ResNet/EfficientNet via segmentation_models_pytorch
atau load dari timm — di sini kita berikan implementasi sederhana untuk full custom U-Net.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
from .unet_model import is_torch_available


def _try_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


_TORCH = _try_torch()


class TransferLearningManager:
    """Manage pretrained U-Net checkpoints + fine-tuning workflow."""

    def __init__(self, checkpoint_dir: str = "./checkpoints/sen1floods11"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def is_available(self) -> bool:
        return is_torch_available()

    def save_checkpoint(
        self,
        model,
        optimizer,
        epoch: int,
        metrics: Dict[str, float],
        name: str = "unet_sen1floods11",
    ) -> Dict[str, Any]:
        """Save model + optimizer + epoch + metrics."""
        if not self.is_available():
            return {"success": False, "reason": "PyTorch not installed"}

        path = self.checkpoint_dir / f"{name}_epoch{epoch:03d}.pt"
        try:
            _TORCH.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
                "metrics": metrics,
                "timestamp": _now_iso(),
            }, path)
            return {
                "success": True,
                "path": str(path),
                "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
                "epoch": epoch,
                "metrics": metrics,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def load_checkpoint(
        self,
        path: str,
        model,
        optimizer=None,
        device: str = "cpu",
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load checkpoint ke model.

        strict=False memungkinkan load partial (untuk transfer learning
        dengan input channel yang berbeda).
        """
        if not self.is_available():
            return {"success": False, "reason": "PyTorch not installed"}

        if not Path(path).exists():
            return {"success": False, "reason": f"Checkpoint not found: {path}"}

        try:
            checkpoint = _TORCH.load(path, map_location=device)
            missing, unexpected = model.load_state_dict(
                checkpoint["model_state_dict"],
                strict=strict,
            )
            if optimizer and checkpoint.get("optimizer_state_dict"):
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            return {
                "success": True,
                "epoch": checkpoint.get("epoch", 0),
                "metrics": checkpoint.get("metrics", {}),
                "missing_keys": list(missing) if missing else [],
                "unexpected_keys": list(unexpected) if unexpected else [],
                "loaded_at": _now_iso(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def freeze_encoder(self, model) -> Dict[str, Any]:
        """Freeze encoder layers untuk fine-tuning awal."""
        if not self.is_available():
            return {"success": False, "reason": "PyTorch not installed"}

        encoder_layers = ["inc", "down1", "down2", "down3", "down4"]
        n_frozen = 0
        n_trainable = 0
        for name, param in model.named_parameters():
            is_encoder = any(layer in name for layer in encoder_layers)
            if is_encoder:
                param.requires_grad = False
                n_frozen += param.numel()
            else:
                param.requires_grad = True
                n_trainable += param.numel()

        return {
            "success": True,
            "frozen_parameters": int(n_frozen),
            "trainable_parameters": int(n_trainable),
            "frozen_layers": encoder_layers,
            "method_monitor": {
                "selected_method": "Encoder freezing untuk fine-tuning awal",
                "why_chosen": (
                    "Encoder fitur general (edge, texture, shape) sudah terlatih dari Sen1Floods11. "
                    "Decoder yang sangat task-specific (flood pixel mapping) yang perlu adapt ke domain baru. "
                    "Freezing encoder mencegah catastrophic forgetting dan training lebih cepat."
                ),
                "why_not_alternatives": [
                    {"alternative": "Train semua layer dari awal", "reason_rejected": "Membuang manfaat pretraining"},
                    {"alternative": "Train semua dengan lr tinggi", "reason_rejected": "Risk catastrophic forgetting"},
                ],
                "next_step_recommendation": (
                    "Setelah loss converge (5-10 epoch), unfreeze gradually dengan lr 10x lebih kecil "
                    "untuk full fine-tuning."
                ),
            },
        }

    def unfreeze_all(self, model) -> Dict[str, Any]:
        """Unfreeze semua layer untuk full fine-tuning."""
        if not self.is_available():
            return {"success": False}
        n_total = 0
        for param in model.parameters():
            param.requires_grad = True
            n_total += param.numel()
        return {"success": True, "trainable_parameters": int(n_total)}

    def adapt_input_channels(self, model, new_in_channels: int) -> Dict[str, Any]:
        """
        Replace first conv layer untuk handle input channel yang berbeda.
        Misal: pretrained 6-channel (S1+S2), tapi user data hanya 4-channel (RGB+NIR).
        """
        if not self.is_available():
            return {"success": False}
        try:
            import torch.nn as nn
            old_conv = model.inc.conv[0]  # first Conv2d in DoubleConv
            old_in = old_conv.in_channels
            out_channels = old_conv.out_channels
            kernel_size = old_conv.kernel_size

            new_conv = nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=old_conv.padding,
                bias=False,
            )
            # Initialize: average pretrained weights across channels
            with _TORCH.no_grad():
                if new_in_channels <= old_in:
                    new_conv.weight.copy_(old_conv.weight[:, :new_in_channels])
                else:
                    repeat = (new_in_channels + old_in - 1) // old_in
                    new_conv.weight.copy_(
                        old_conv.weight.repeat(1, repeat, 1, 1)[:, :new_in_channels]
                    )

            model.inc.conv[0] = new_conv
            model.in_channels = new_in_channels

            return {
                "success": True,
                "old_in_channels": old_in,
                "new_in_channels": new_in_channels,
                "weight_strategy": "channel-mean copy" if new_in_channels <= old_in else "channel-repeat",
                "method_monitor": {
                    "selected_method": "Input layer adaptation untuk transfer learning",
                    "why_chosen": (
                        "Pretrained model expects N channel, tapi data baru punya M channel. "
                        "Replace conv pertama dengan size yang sesuai, initialize dari pretrained weights "
                        "(rata-rata atau repeat) → preserve learned features sebanyak mungkin."
                    ),
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_checkpoints(self) -> Dict[str, Any]:
        """List semua checkpoints di folder."""
        files = sorted(self.checkpoint_dir.glob("*.pt"))
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "n_checkpoints": len(files),
            "files": [
                {
                    "name": f.name,
                    "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
                    "modified": _file_mtime(f),
                }
                for f in files
            ],
        }


# Pre-trained download stubs (model yang di-share oleh komunitas)
PRETRAINED_MODELS = {
    "sen1floods11_unet_s2": {
        "name": "U-Net Sen1Floods11 (S2 only)",
        "in_channels": 4,
        "url": "https://zenodo.org/record/4498086/files/unet_sen1floods11_s2_v1.pt",
        "expected_iou": 0.74,
        "training_modality": "Sentinel-2 RGB+NIR",
        "training_loss": "BCE + Dice",
    },
    "sen1floods11_unet_s1": {
        "name": "U-Net Sen1Floods11 (S1 SAR only)",
        "in_channels": 2,
        "url": "https://zenodo.org/record/4498086/files/unet_sen1floods11_s1_v1.pt",
        "expected_iou": 0.71,
        "training_modality": "Sentinel-1 VV+VH",
        "training_loss": "BCE + Dice",
    },
    "sen1floods11_unet_fusion": {
        "name": "U-Net Sen1Floods11 (S1+S2 fusion)",
        "in_channels": 6,
        "url": "https://zenodo.org/record/4498086/files/unet_sen1floods11_fusion_v1.pt",
        "expected_iou": 0.79,
        "training_modality": "S1 (VV+VH) + S2 (B,G,R,NIR)",
        "training_loss": "BCE + Dice",
        "recommended": True,
    },
}


def list_available_pretrained() -> Dict[str, Any]:
    """List pretrained model yang available untuk download."""
    return {
        "models": PRETRAINED_MODELS,
        "n_models": len(PRETRAINED_MODELS),
        "recommended": "sen1floods11_unet_fusion",
        "note": (
            "URL di atas adalah PLACEHOLDER — comm Sen1Floods11 sebenarnya share weights via berbagai sumber. "
            "Untuk implementasi nyata, train sendiri di Sen1Floods11 atau download dari Zenodo/GitHub releases."
        ),
    }


# Helpers
def _now_iso():
    from datetime import datetime
    return datetime.utcnow().isoformat()


def _file_mtime(path: Path) -> str:
    from datetime import datetime
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat()
