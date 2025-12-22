"""
Backbone modules tùy chỉnh cho YOLO sample:
- RFCBAMConv: attention theo paper (channel + spatial với receptive field)
- C2f_RFCBAM: thay Bottleneck trong C2f bằng RFCBAM_Neck
- MixSPPF: thay SPPF bằng phiên bản kết hợp MaxPool + AvgPool

File này độc lập với improved_yolov8, chỉ dùng các lớp Conv/C2f chuẩn của Ultralytics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from ultralytics.nn.modules import Conv, C2f


class SE(nn.Module):
    """Squeeze-and-Excitation module cho channel attention."""

    def __init__(self, in_channel: int, ratio: int = 16) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channel, ratio, bias=False),
            nn.ReLU(),
            nn.Linear(ratio, in_channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[0:2]
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class RFCBAMConv(nn.Module):
    """
    RFCBAMConv: Receptive Field Concentration-Based Attention Module Convolution.
    Giao diện gần giống Conv của Ultralytics để YAML parse được.
    Trong YAML ta chỉ truyền [c2] nên ở đây c2 là số kênh output.
    """

    def __init__(self, c2: int, k: int = 3, s: int = 2, p=None, g: int = 1, d: int = 1, act=True) -> None:
        super().__init__()
        # Lưu hyper-params, build lazy khi biết c1
        self.out_channel = c2
        k = int(k)
        if k % 2 == 0:
            k = k + 1  # đảm bảo kernel odd
        self.kernel_size = k
        self.stride = max(int(s), 1)
        self._built = False

    def _build(self, c1: int) -> None:
        in_channel = c1
        k = self.kernel_size
        s = self.stride

        # Nhánh generate receptive field features
        self.generate = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (k**2), k, padding=k // 2, stride=s, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (k**2)),
            nn.ReLU(),
        )

        # Spatial attention trên max/mean
        self.get_weight = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

        # Channel attention
        self.se = SE(in_channel)

        # Conv cuối (chuẩn YOLO)
        self.conv = Conv(in_channel, self.out_channel, k=k, s=s, p=k // 2)
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[0:2]
        if not self._built:
            self._build(c)

        # Channel attention
        channel_attention = self.se(x)

        # Receptive field features
        generate_feature = self.generate(x)
        h, w = generate_feature.shape[2:]
        k = self.kernel_size
        generate_feature = generate_feature.view(b, c, k * k, h, w)
        generate_feature = rearrange(
            generate_feature, "b c (n1 n2) h w -> b c (h n1) (w n2)", n1=k, n2=k
        )

        # Áp dụng channel attention
        unfold_feature = generate_feature * channel_attention

        # Spatial attention
        max_feature, _ = torch.max(generate_feature, dim=1, keepdim=True)
        mean_feature = torch.mean(generate_feature, dim=1, keepdim=True)
        receptive_field_attention = self.get_weight(torch.cat((max_feature, mean_feature), dim=1))

        # Áp dụng spatial attention + Conv cuối
        conv_data = unfold_feature * receptive_field_attention
        return self.conv(conv_data)


class MixSPPF(nn.Module):
    """
    Mix Spatial Pyramid Pooling - Fast (MixSPPF).
    Kết hợp MaxPool và AvgPool như mô tả trong paper.
    """

    def __init__(self, c1: int, c2: int, k: int = 5) -> None:
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        # 4 nhánh max + 4 nhánh avg => 8 * c_
        self.cv2 = Conv(c_ * 8, c2, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.avgpool = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)

        # Max pooling branch
        y1_max = self.maxpool(x)
        y2_max = self.maxpool(y1_max)
        y3_max = self.maxpool(y2_max)
        y_max = torch.cat((x, y1_max, y2_max, y3_max), 1)

        # Avg pooling branch
        y1_avg = self.avgpool(x)
        y2_avg = self.avgpool(y1_avg)
        y3_avg = self.avgpool(y2_avg)
        y_avg = torch.cat((x, y1_avg, y2_avg, y3_avg), 1)

        return self.cv2(torch.cat((y_max, y_avg), 1))


class RFCBAM_Neck(nn.Module):
    """
    Block dùng trong C2f_RFCBAM: Conv -> RFCBAMConv (không residual).
    """

    def __init__(self, c1: int, c2: int, k=(3, 3)) -> None:
        super().__init__()
        self.cv1 = Conv(c1, c2, k[0], 1)
        self.cv2 = RFCBAMConv(c2, k=k[1], s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(self.cv1(x))


class C2f_RFCBAM(C2f):
    """
    Phiên bản C2f dùng RFCBAM_Neck thay cho Bottleneck chuẩn (theo Figure 4).
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5) -> None:
        # Không gọi init của C2f gốc, tự dựng lại theo logic C2f
        nn.Module.__init__(self)
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(RFCBAM_Neck(self.c, self.c, k=(3, 3)) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def register_yolo_sample_modules() -> None:
    """
    Đăng ký các module custom vào Ultralytics để parse từ YAML:
    - Thêm vào ultralytics.nn.tasks và parse_model.__globals__
    - Thêm vào ultralytics.nn.modules (cho tiện debug)
    Gọi hàm này TRƯỚC khi tạo YOLO(model_yaml).
    """

    import builtins
    import ultralytics.nn.tasks as tasks_module
    import ultralytics.nn.modules as modules

    custom_modules = {
        "RFCBAMConv": RFCBAMConv,
        "C2f_RFCBAM": C2f_RFCBAM,
        "MixSPPF": MixSPPF,
    }

    # Đăng ký trong tasks_module + builtins
    for name, cls in custom_modules.items():
        setattr(tasks_module, name, cls)
        setattr(builtins, name, cls)

    # Đăng ký trong modules (không bắt buộc nhưng tiện)
    for name, cls in custom_modules.items():
        setattr(modules, name, cls)

    # Patch globals của parse_model
    if hasattr(tasks_module, "parse_model"):
        g = tasks_module.parse_model.__globals__
        for name, cls in custom_modules.items():
            g[name] = cls


