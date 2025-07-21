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
        x = self.conv(x)      # 先卷积
        x_down = self.pool(x) # 再池化
        return x, x_down      # x 用作 skip, x_down 送下一层


class Up(nn.Module):
    """
    上采样：先 Upsample(或转置卷积) -> 拼接 skip -> DoubleConv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            # 双线性插值
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 这里 in_channels = skip_ch + dec_ch
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # 转置卷积
            self.up = nn.ConvTranspose2d(
                in_channels // 2,  # 一般要跟 dec_ch 相符
                in_channels // 2,
                kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        """
        x: 来自上层 decoder 的特征
        skip: 来自对应 encoder 的特征
        """
        # 上采样
        x = self.up(x)
        # 大小对齐
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        # 拼接
        x = torch.cat([skip, x], dim=1)
        # 再卷积
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    可以处理不同输入通道数的UNet，兼容7通道和10通道输入
    """
    def __init__(self, in_channels=10, out_channels=1, base_features=64, bilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.base_features = base_features

        # --------------- Encoder ---------------
        self.down1 = Down(in_channels, base_features)            # (7/10 -> 64)
        self.down2 = Down(base_features, base_features * 2)      # (64 -> 128)
        self.down3 = Down(base_features * 2, base_features * 4)  # (128 -> 256)
        self.down4 = Down(base_features * 4, base_features * 8)  # (256 -> 512)

        # bottom
        self.bottom_conv = DoubleConv(base_features * 8, base_features * 16)  # (512 -> 1024)

        # --------------- Decoder ---------------
        # 注意 in_channels = skip通道数 + 上层decoder输出通道数
        # up1 拼接 bottom(1024) + x4(512) = 1536 -> out 512
        self.up1 = Up(in_channels=base_features*16 + base_features*8,
                      out_channels=base_features*8,
                      bilinear=bilinear)

        # up2 拼接 up1(512) + x3(256) = 768 -> out 256
        self.up2 = Up(in_channels=base_features*8 + base_features*4,
                      out_channels=base_features*4,
                      bilinear=bilinear)

        # up3 拼接 up2(256) + x2(128) = 384 -> out 128
        self.up3 = Up(in_channels=base_features*4 + base_features*2,
                      out_channels=base_features*2,
                      bilinear=bilinear)

        # up4 拼接 up3(128) + x1(64) = 192 -> out 64
        self.up4 = Up(in_channels=base_features*2 + base_features,
                      out_channels=base_features,
                      bilinear=bilinear)

        # 最终输出层
        self.out_conv = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def forward(self, x):
        # -------------- Encoder --------------
        x1, x1_down = self.down1(x)      #  => x1= [B,64,H,W], x1_down=[B,64,H/2,W/2]
        x2, x2_down = self.down2(x1_down)#  => x2= [B,128,H/4,W/4], ...
        x3, x3_down = self.down3(x2_down)#  => x3= [B,256,H/8,W/8]
        x4, x4_down = self.down4(x3_down)#  => x4= [B,512,H/16,W/16], x4_down=[B,512,H/32,W/32]

        # -------------- Bottom --------------
        x_bottom = self.bottom_conv(x4_down)  # => [B,1024,H/32,W/32]

        # -------------- Decoder --------------
        x_up1 = self.up1(x_bottom, x4)  # input通道=1024+512=1536 -> out=512
        x_up2 = self.up2(x_up1, x3)     # 512+256=768 -> 256
        x_up3 = self.up3(x_up2, x2)     # 256+128=384 -> 128
        x_up4 = self.up4(x_up3, x1)     # 128+64=192  -> 64

        logits = self.out_conv(x_up4)   # => [B,1,H,W]
        return logits


class FlexibleUNet(nn.Module):
    """
    灵活处理不同通道数输入的UNet，支持在同一批次中混合不同输入通道数
    """
    def __init__(self, base_features=64, out_channels=1, bilinear=True):
        super(FlexibleUNet, self).__init__()
        self.bilinear = bilinear
        
        # 不同输入通道数对应的两个UNet
        self.unet_7ch = UNet(in_channels=7, out_channels=out_channels, 
                             base_features=base_features, bilinear=bilinear)
        self.unet_10ch = UNet(in_channels=10, out_channels=out_channels, 
                              base_features=base_features, bilinear=bilinear)
    
    def forward(self, x_batch_or_list):
        """
        处理单一批次或输入列表
        Args:
            x_batch_or_list: 可以是单个张量 [B,C,H,W] 或 
                            包含两种不同通道数的张量列表 [(B1,C1,H,W), (B2,C2,H,W)]
        Returns:
            对应的预测张量或张量列表
        """
        if isinstance(x_batch_or_list, list):
            # 如果是列表，分别处理不同通道数的输入
            outputs = []
            for x_batch in x_batch_or_list:
                if x_batch.size(1) == 7:
                    outputs.append(self.unet_7ch(x_batch))
                elif x_batch.size(1) == 10:
                    outputs.append(self.unet_10ch(x_batch))
                else:
                    raise ValueError(f"Unsupported input channels: {x_batch.size(1)}")
            return outputs
        else:
            # 如果是单个张量，根据通道数选择对应的UNet
            x_batch = x_batch_or_list
            if x_batch.size(1) == 7:
                return self.unet_7ch(x_batch)
            elif x_batch.size(1) == 10:
                return self.unet_10ch(x_batch)
            else:
                raise ValueError(f"Unsupported input channels: {x_batch.size(1)}")


def test_forward_pass():
    """
    测试：构造不同通道数的随机输入，测试UNet和FlexibleUNet
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("测试普通UNet模型...")
    # 测试7通道输入
    unet_7ch = UNet(in_channels=7, out_channels=1, base_features=64, bilinear=True).to(device)
    x_7ch = torch.randn(2, 7, 512, 512).to(device)
    y_7ch = unet_7ch(x_7ch)
    print(f"UNet (7通道) - 输入: {x_7ch.shape}, 输出: {y_7ch.shape}")
    
    # 测试10通道输入
    unet_10ch = UNet(in_channels=10, out_channels=1, base_features=64, bilinear=True).to(device)
    x_10ch = torch.randn(2, 10, 512, 512).to(device)
    y_10ch = unet_10ch(x_10ch)
    print(f"UNet (10通道) - 输入: {x_10ch.shape}, 输出: {y_10ch.shape}")
    
    print("\n测试灵活的FlexibleUNet模型...")
    # 测试FlexibleUNet（可处理混合通道输入）
    flexible_unet = FlexibleUNet(base_features=64, out_channels=1, bilinear=True).to(device)
    
    # 单一批次测试
    y_7ch_flex = flexible_unet(x_7ch)
    print(f"FlexibleUNet (7通道) - 输入: {x_7ch.shape}, 输出: {y_7ch_flex.shape}")
    
    y_10ch_flex = flexible_unet(x_10ch)
    print(f"FlexibleUNet (10通道) - 输入: {x_10ch.shape}, 输出: {y_10ch_flex.shape}")
    
    # 混合批次测试
    y_mixed = flexible_unet([x_7ch, x_10ch])
    print(f"FlexibleUNet (混合通道) - 输出: [{y_mixed[0].shape}, {y_mixed[1].shape}]")


if __name__ == "__main__":
    test_forward_pass()
