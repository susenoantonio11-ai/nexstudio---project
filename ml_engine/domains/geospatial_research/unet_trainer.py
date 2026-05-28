"""
U-Net Training Pipeline
========================
Loss functions, training loop, evaluation, dan inference untuk U-Net flood segmentation.

Loss strategy:
- BCE-Dice combined: balance pixel-level accuracy + segmentation overlap
- Dice loss alone bisa untuk severe imbalance
- Focal loss optional untuk extreme imbalance

Augmentation:
- Random horizontal/vertical flip
- Random rotation
- Gaussian noise pada bands
"""
from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional, Iterator
from .unet_model import is_torch_available


# ==================================================================
# LOSS FUNCTIONS (only if torch available)
# ==================================================================
if is_torch_available():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class DiceLoss(nn.Module):
        """
        Dice loss = 1 - (2*intersection / (pred_sum + target_sum))

        Robust untuk imbalanced segmentation karena tidak counts True Negative.
        Range: [0, 1] — 0 = perfect overlap.
        """

        def __init__(self, smooth: float = 1.0):
            super().__init__()
            self.smooth = smooth

        def forward(self, logits, targets):
            probs = torch.sigmoid(logits)
            probs_flat = probs.view(-1)
            targets_flat = targets.view(-1).float()
            intersection = (probs_flat * targets_flat).sum()
            dice = (2.0 * intersection + self.smooth) / (
                probs_flat.sum() + targets_flat.sum() + self.smooth
            )
            return 1.0 - dice

    class BCEDiceLoss(nn.Module):
        """Combined BCE + Dice loss — balance pixel-level + region-level."""

        def __init__(self, dice_weight: float = 0.5, smooth: float = 1.0):
            super().__init__()
            self.dice_weight = dice_weight
            self.bce = nn.BCEWithLogitsLoss()
            self.dice = DiceLoss(smooth=smooth)

        def forward(self, logits, targets):
            return (1 - self.dice_weight) * self.bce(logits, targets.float()) + \
                   self.dice_weight * self.dice(logits, targets)

    class FocalLoss(nn.Module):
        """
        Focal Loss (Lin et al., 2017): down-weight easy samples, focus on hard.
        Untuk extreme class imbalance (flood pixels < 5% total).
        """

        def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, logits, targets):
            probs = torch.sigmoid(logits)
            ce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
            p_t = probs * targets + (1 - probs) * (1 - targets)
            alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            modulating = (1 - p_t) ** self.gamma
            loss = alpha_factor * modulating * ce
            return loss.mean()

    # Augmentation transforms
    class RandomFloodAugment:
        """Augmentation untuk flood imagery training."""

        def __init__(self, p_flip: float = 0.5, p_rot: float = 0.3, noise_std: float = 0.01):
            self.p_flip = p_flip
            self.p_rot = p_rot
            self.noise_std = noise_std

        def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # Random horizontal flip
            if torch.rand(1) < self.p_flip:
                image = torch.flip(image, dims=[-1])
                mask = torch.flip(mask, dims=[-1])
            # Random vertical flip
            if torch.rand(1) < self.p_flip:
                image = torch.flip(image, dims=[-2])
                mask = torch.flip(mask, dims=[-2])
            # Random 90° rotation
            if torch.rand(1) < self.p_rot:
                k = int(torch.randint(1, 4, (1,)).item())
                image = torch.rot90(image, k, dims=[-2, -1])
                mask = torch.rot90(mask, k, dims=[-2, -1])
            # Gaussian noise on image
            if self.noise_std > 0:
                image = image + torch.randn_like(image) * self.noise_std
            return image, mask


# ==================================================================
# METRICS
# ==================================================================
def compute_iou(pred_mask, target_mask, threshold: float = 0.5) -> float:
    """IoU = intersection / union for binary masks."""
    if not is_torch_available():
        return 0.0
    pred = (pred_mask > threshold).float()
    target = target_mask.float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    if union.item() == 0:
        return 1.0
    return float(intersection / union)


def compute_dice(pred_mask, target_mask, threshold: float = 0.5) -> float:
    """Dice = 2*intersection / (pred_sum + target_sum)."""
    if not is_torch_available():
        return 0.0
    pred = (pred_mask > threshold).float()
    target = target_mask.float()
    intersection = (pred * target).sum()
    denom = pred.sum() + target.sum()
    if denom.item() == 0:
        return 1.0
    return float(2.0 * intersection / denom)


# ==================================================================
# UNET TRAINER (high-level API)
# ==================================================================
class UNetTrainer:
    """High-level trainer untuk U-Net flood segmentation."""

    def __init__(
        self,
        in_channels: int = 4,
        learning_rate: float = 1e-3,
        loss_type: str = "bce_dice",  # 'bce' / 'dice' / 'bce_dice' / 'focal'
        device: str = "auto",
    ):
        self.in_channels = in_channels
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        self.requested_device = device
        self.history: List[Dict[str, float]] = []

    def can_train(self) -> bool:
        return is_torch_available()

    def setup(self) -> Dict[str, Any]:
        """Prepare model + optimizer + loss + device."""
        if not is_torch_available():
            return {
                "available": False,
                "reason": "PyTorch belum terinstall",
                "install": "pip install torch torchvision",
            }
        from .unet_model import UNet
        device = self._resolve_device()
        model = UNet(in_channels=self.in_channels, out_channels=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = self._build_loss()

        n_params = sum(p.numel() for p in model.parameters())
        return {
            "available": True,
            "model": model,
            "optimizer": optimizer,
            "loss_fn": loss_fn,
            "device": str(device),
            "n_parameters": int(n_params),
            "loss_type": self.loss_type,
        }

    def _resolve_device(self):
        if not is_torch_available():
            return "cpu"
        if self.requested_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.requested_device

    def _build_loss(self):
        if not is_torch_available():
            return None
        if self.loss_type == "bce":
            return nn.BCEWithLogitsLoss()
        if self.loss_type == "dice":
            return DiceLoss()
        if self.loss_type == "focal":
            return FocalLoss()
        return BCEDiceLoss()

    def train_one_epoch(
        self,
        model,
        loader: Iterator,
        optimizer,
        loss_fn,
        device,
        augment=None,
    ) -> Dict[str, float]:
        """Single training epoch."""
        if not is_torch_available():
            return {"available": False}
        model.train()
        running = {"loss": 0.0, "iou": 0.0, "dice": 0.0, "n_batches": 0}
        for images, masks in loader:
            if augment is not None:
                images, masks = augment(images, masks)
            images = images.to(device).float()
            masks = masks.to(device).float()
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                running["loss"] += loss.item()
                running["iou"] += compute_iou(probs, masks)
                running["dice"] += compute_dice(probs, masks)
            running["n_batches"] += 1

        n = max(1, running["n_batches"])
        return {
            "loss": running["loss"] / n,
            "iou": running["iou"] / n,
            "dice": running["dice"] / n,
            "n_batches": n,
        }

    def evaluate(self, model, loader, loss_fn, device) -> Dict[str, float]:
        """Evaluation pass — no gradient."""
        if not is_torch_available():
            return {"available": False}
        model.eval()
        running = {"loss": 0.0, "iou": 0.0, "dice": 0.0, "n_batches": 0}
        with torch.no_grad():
            for images, masks in loader:
                images = images.to(device).float()
                masks = masks.to(device).float()
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)
                logits = model(images)
                loss = loss_fn(logits, masks)
                probs = torch.sigmoid(logits)
                running["loss"] += loss.item()
                running["iou"] += compute_iou(probs, masks)
                running["dice"] += compute_dice(probs, masks)
                running["n_batches"] += 1
        n = max(1, running["n_batches"])
        return {
            "loss": running["loss"] / n,
            "iou": running["iou"] / n,
            "dice": running["dice"] / n,
        }

    def predict(self, model, image_tensor, device, threshold: float = 0.5) -> Dict[str, Any]:
        """Inference single image atau batch."""
        if not is_torch_available():
            return {"available": False}
        model.eval()
        with torch.no_grad():
            image = image_tensor.to(device).float()
            if image.dim() == 3:
                image = image.unsqueeze(0)  # add batch dim
            logits = model(image)
            probs = torch.sigmoid(logits)
            mask = (probs > threshold).int()
        return {
            "probabilities": probs,
            "mask": mask,
            "threshold": threshold,
        }
