"""
U-Net Architecture untuk Flood Semantic Segmentation
====================================================
Implementasi U-Net (Ronneberger et al., 2015) khusus untuk binary flood mask
prediction dari multi-band raster.

Input shape: (batch, n_channels, H, W) - bisa multi-band (RGB+NIR+SWIR+VV)
Output shape: (batch, 1, H, W) - probability per pixel

Arsitektur:
- Encoder: 4 downsampling blocks (Conv-BN-ReLU x2, MaxPool)
- Bottleneck: 1 block dengan double conv
- Decoder: 4 upsampling blocks dengan skip connections
- Output: 1x1 conv ke 1 channel + sigmoid

Total params (default): ~7.7M

Reference: Ronneberger, O., Fischer, P., & Brox, T. (2015).
U-Net: Convolutional Networks for Biomedical Image Segmentation.
"""
from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional


def _try_import_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        return None, None


_TORCH, _NN = _try_import_torch()


def is_torch_available() -> bool:
    return _TORCH is not None


# ==================================================================
# U-Net Architecture (only defined if torch available)
# ==================================================================
if _TORCH is not None:

    class DoubleConv(_NN.Module):
        """(Conv2d → BatchNorm → ReLU) × 2 — basic building block U-Net."""

        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.conv = _NN.Sequential(
                _NN.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                _NN.BatchNorm2d(out_ch),
                _NN.ReLU(inplace=True),
                _NN.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                _NN.BatchNorm2d(out_ch),
                _NN.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.conv(x)

    class Down(_NN.Module):
        """Downsampling block: MaxPool → DoubleConv."""

        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.pool_conv = _NN.Sequential(
                _NN.MaxPool2d(2),
                DoubleConv(in_ch, out_ch),
            )

        def forward(self, x):
            return self.pool_conv(x)

    class Up(_NN.Module):
        """Upsampling block: ConvTranspose → concat skip → DoubleConv."""

        def __init__(self, in_ch: int, out_ch: int, use_bilinear: bool = True):
            super().__init__()
            if use_bilinear:
                self.up = _NN.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                self.conv = DoubleConv(in_ch, out_ch)
            else:
                self.up = _NN.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
                self.conv = DoubleConv(in_ch, out_ch)

        def forward(self, x_decode, x_skip):
            x_decode = self.up(x_decode)
            # Pad if size mismatch (caused by odd input sizes)
            diff_y = x_skip.size()[2] - x_decode.size()[2]
            diff_x = x_skip.size()[3] - x_decode.size()[3]
            if diff_y or diff_x:
                x_decode = _TORCH.nn.functional.pad(
                    x_decode,
                    [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
                )
            x = _TORCH.cat([x_skip, x_decode], dim=1)
            return self.conv(x)

    class UNet(_NN.Module):
        """
        U-Net untuk flood segmentation.

        Args:
            in_channels: jumlah band input (e.g., 4 untuk RGB+NIR, 6 untuk +SWIR+VV)
            out_channels: jumlah class output (1 untuk binary flood/non-flood)
            base_filters: filter count di layer pertama (32, 64=default)
            use_bilinear: True = bilinear upsampling (lebih sedikit param), False = transposed conv
        """

        def __init__(
            self,
            in_channels: int = 4,
            out_channels: int = 1,
            base_filters: int = 64,
            use_bilinear: bool = True,
        ):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels

            f = base_filters
            self.inc = DoubleConv(in_channels, f)
            self.down1 = Down(f, f * 2)
            self.down2 = Down(f * 2, f * 4)
            self.down3 = Down(f * 4, f * 8)
            self.down4 = Down(f * 8, f * 16)

            self.up1 = Up(f * 16 + f * 8, f * 8, use_bilinear)
            self.up2 = Up(f * 8 + f * 4, f * 4, use_bilinear)
            self.up3 = Up(f * 4 + f * 2, f * 2, use_bilinear)
            self.up4 = Up(f * 2 + f, f, use_bilinear)

            self.outc = _NN.Conv2d(f, out_channels, kernel_size=1)

        def forward(self, x):
            x1 = self.inc(x)         # (f,    H,    W)
            x2 = self.down1(x1)      # (2f,   H/2,  W/2)
            x3 = self.down2(x2)      # (4f,   H/4,  W/4)
            x4 = self.down3(x3)      # (8f,   H/8,  W/8)
            x5 = self.down4(x4)      # (16f,  H/16, W/16)

            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            return self.outc(x)


def build_unet(
    in_channels: int = 4,
    out_channels: int = 1,
    base_filters: int = 64,
) -> Dict[str, Any]:
    """
    Factory function — returns model + metadata.
    Graceful degradation if PyTorch not installed.
    """
    if not is_torch_available():
        return {
            "available": False,
            "reason": "PyTorch belum terinstall. Install: pip install torch",
            "expected_model": "U-Net dengan {} input channels".format(in_channels),
        }

    model = UNet(in_channels=in_channels, out_channels=out_channels, base_filters=base_filters)
    n_params = sum(p.numel() for p in model.parameters())

    return {
        "available": True,
        "model": model,
        "architecture": "U-Net (Ronneberger et al., 2015)",
        "in_channels": in_channels,
        "out_channels": out_channels,
        "base_filters": base_filters,
        "n_parameters": int(n_params),
        "n_parameters_human": f"{n_params/1e6:.2f}M",
        "device_recommended": "cuda" if (_TORCH and _TORCH.cuda.is_available()) else "cpu",
        "input_format": f"(batch, {in_channels}, H, W) — H dan W harus kelipatan 16",
        "output_format": f"(batch, {out_channels}, H, W) — apply sigmoid untuk binary, softmax untuk multi-class",
        "method_monitor": {
            "selected_method": "U-Net architecture",
            "why_chosen": (
                "U-Net mendominasi semantic segmentation karena: (1) skip connections "
                "memungkinkan localization presisi (penting untuk pixel-level flood mask), "
                "(2) encoder-decoder symmetry membuat training stabil dengan sedikit data, "
                "(3) terbukti SOTA di flood mapping (Bonafilia et al., 2020)."
            ),
            "why_not_alternatives": [
                {"alternative": "DeepLabV3+", "reason_rejected": "Lebih kompleks; butuh pretraining yang lebih banyak"},
                {"alternative": "FCN", "reason_rejected": "Tanpa skip connections, output less precise"},
                {"alternative": "Mask R-CNN", "reason_rejected": "Untuk instance segmentation; flood mapping hanya butuh semantic"},
                {"alternative": "Random Forest pixel-based", "reason_rejected": "Tidak memanfaatkan spatial context — U-Net jauh lebih akurat"},
            ],
            "limitations": [
                "Butuh banyak labeled training data (~1000 patches minimal)",
                "Training memakan GPU memory",
                "Sensitif terhadap class imbalance (flood biasanya minoritas) — pakai Dice loss",
            ],
            "reference": "Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.",
        },
    }
