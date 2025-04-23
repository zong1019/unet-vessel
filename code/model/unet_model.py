
import sys
import os
sys.path.append(os.path.abspath('/home/x4228/models/segment/unet_vessel/model'))
from .unet_parts import *
from .cbam import *
from .ScConv import *
import torch.nn as nn
from einops import rearrange
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        ## 实例化 Transformer 编码器层
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=512, nhead=8),
            num_layers=6
        )
        #实例化ssconv
        self.Ssconv1 = ScConv(64, 4, 0.5, 1/2, 2, 2, 3)
        self.Ssconv2 = ScConv(128, 4, 0.5, 1/2, 2, 2, 3)
        self.Ssconv3 = ScConv(256, 4, 0.5, 1/2, 2, 2, 3)
        self.Ssconv4 = ScConv(512, 4, 0.5, 1/2, 2, 2, 3)


    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.Ssconv4(x5) + x5

        b, c, h, w = x5.shape
        x5 = rearrange(x5, 'b c h w -> (h w) b c')
        x5 = self.transformer_encoder(x5)
        x5 = rearrange(x5, '(h w) b c -> b c h w', h=h, w=w)

        x4 = self.Ssconv4(x4) + x4
        x = self.up1(x5, x4)
        x3 = self.Ssconv3(x3) + x3
        x = self.up2(x, x3)
        x2 = self.Ssconv2(x2) + x2
        x = self.up3(x, x2)
        x1 = self.Ssconv1(x1) + x1
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits



if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=1)
    print(net)
