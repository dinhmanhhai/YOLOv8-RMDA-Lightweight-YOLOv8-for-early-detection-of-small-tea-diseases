"""
Lightweight Dynamic Head implementation for improved YOLOv8.

This is a simplified DyHead that captures:
- Scale-aware attention: fuse neighboring pyramid levels (up/down)
- Spatial-aware attention: depthwise conv + sigmoid weighting
- Task-aware attention: pointwise conv refinement

No external dependencies (mmcv/DeformConv) are required.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv, DFL
from ultralytics.utils.tal import dist2bbox, make_anchors

__all__ = ["DyHeadBlockLite", "Detect_DyHead"]


class DyHeadBlockLite(nn.Module):
    """
    Simplified DyHead block:
    - Depthwise conv for spatial processing
    - Scale fusion: sum current, upsampled higher, and downsampled lower features
    - Scale-aware attention: sigmoid gate from pooled features
    - Task-aware attention: pointwise conv refinement
    """

    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(channels)
        self.task_attn = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )
        self.scale_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feats):
        outs = []
        n = len(feats)
        for i in range(n):
            cur = feats[i]
            cur = self.dw_bn(self.dw(cur))

            # scale-aware attention gate
            gate = self.scale_attn(cur)
            cur = cur * gate

            summed = cur
            count = 1

            # fuse lower level (i-1): downsample to current size
            if i > 0:
                low = feats[i - 1]
                low = F.max_pool2d(low, kernel_size=2)
                low = self.dw_bn(self.dw(low))
                low = low * self.scale_attn(low)
                summed = summed + low
                count += 1

            # fuse higher level (i+1): upsample to current size
            if i < n - 1:
                high = feats[i + 1]
                high = F.interpolate(high, size=cur.shape[-2:], mode="nearest")
                high = self.dw_bn(self.dw(high))
                high = high * self.scale_attn(high)
                summed = summed + high
                count += 1

            fused = summed / count
            fused = self.task_attn(fused)
            outs.append(fused)
        return outs


class Detect_DyHead(nn.Module):
    """
    Detect head with lightweight Dynamic Head blocks.
    Follows YOLOv8 Detect structure: per-level conv stem -> DyHead blocks -> cls/reg heads.
    """

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, hidc=256, block_num=2, ch=()):
        """
        Args:
            nc: number of classes
            hidc: hidden channels in the stem
            block_num: number of DyHead blocks
            ch: input channels list from neck (e.g., [256, 512, 1024])
        """
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], self.nc)

        # Per-scale stem conv
        self.conv = nn.ModuleList(nn.Sequential(Conv(x, hidc, 1)) for x in ch)
        # DyHead blocks
        self.dyhead = nn.Sequential(*[DyHeadBlockLite(hidc) for _ in range(block_num)])
        # Regression and classification heads
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(hidc, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1))
            for _ in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(hidc, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1))
            for _ in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        # Stem
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])

        # DyHead blocks
        x = self.dyhead(x)
        shape = x[0].shape

        # Heads
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        # Initialize cls/box biases similar to YOLOv8 Detect
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)

