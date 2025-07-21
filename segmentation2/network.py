# network.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    小卷积块：Conv2d -> BN -> ReLU (x2)
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class Down(nn.Module):
    """
    下采样：DoubleConv + MaxPool2d
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x_down = self.pool(x)
        return x, x_down


class Up(nn.Module):
    """
    上采样模块：
      1. 对 decoder 特征上采样（双线性 or 转置卷积）
      2. 与对应 encoder skip 特征拼接
      3. DoubleConv 融合
    """
    def __init__(self,
                 dec_channels: int,
                 skip_channels: int,
                 out_channels: int,
                 bilinear: bool = True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            in_conv_channels = dec_channels + skip_channels
        else:
            self.up = nn.ConvTranspose2d(
                dec_channels,
                dec_channels // 2,
                kernel_size=2,
                stride=2
            )
            in_conv_channels = dec_channels // 2 + skip_channels

        self.conv = DoubleConv(in_conv_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # 对齐大小（偶数尺寸防抖）
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class SketchSegmentUNet(nn.Module):
    """
    Arrow‑Sketch(3-ch) → Segmentation(1-ch) 的高性能 UNet
    （输出为 *raw logits*，不再做 sigmoid）
    """
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 1,
                 base_features: int = 128,
                 bilinear: bool = False):
        super(SketchSegmentUNet, self).__init__()

        # ---------- Encoder ----------
        self.down1 = Down(in_channels, base_features)
        self.down2 = Down(base_features, base_features * 2)
        self.down3 = Down(base_features * 2, base_features * 4)
        self.down4 = Down(base_features * 4, base_features * 8)

        # ---------- Bottom ----------
        self.bottom_conv = DoubleConv(base_features * 8, base_features * 16)

        # ---------- Decoder ----------
        self.up1 = Up(dec_channels=base_features * 16,
                      skip_channels=base_features * 8,
                      out_channels=base_features * 8,
                      bilinear=bilinear)

        self.up2 = Up(dec_channels=base_features * 8,
                      skip_channels=base_features * 4,
                      out_channels=base_features * 4,
                      bilinear=bilinear)

        self.up3 = Up(dec_channels=base_features * 4,
                      skip_channels=base_features * 2,
                      out_channels=base_features * 2,
                      bilinear=bilinear)

        self.up4 = Up(dec_channels=base_features * 2,
                      skip_channels=base_features,
                      out_channels=base_features,
                      bilinear=bilinear)

        # ---------- Output ----------
        self.out_conv = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def forward(self, x):
        x1, x1_down = self.down1(x)
        x2, x2_down = self.down2(x1_down)
        x3, x3_down = self.down3(x2_down)
        x4, x4_down = self.down4(x3_down)

        x_bottom = self.bottom_conv(x4_down)

        x_up1 = self.up1(x_bottom, x4)
        x_up2 = self.up2(x_up1, x3)
        x_up3 = self.up3(x_up2, x2)
        x_up4 = self.up4(x_up3, x1)

        logits = self.out_conv(x_up4)      # raw logits
        return logits


def test_network():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for size in [256, 512]:
        x = torch.randn(2, 3, size, size).to(device)
        model = SketchSegmentUNet().to(device)
        with torch.no_grad():
            y = model(x)
        y_prob = torch.sigmoid(y)
        print(f"[TEST] Input: {tuple(x.shape)} → Output: {tuple(y.shape)} | "
              f"Param #: {sum(p.numel() for p in model.parameters()):,} | "
              f"Val range: [{y_prob.min().item():.4f}, {y_prob.max().item():.4f}]")


if __name__ == "__main__":
    test_network()
