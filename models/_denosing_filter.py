"""
Unet model architecture.
Used in segmentation and image noising.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class _denoising_block(nn.Module): ### mean filter
    def __init__(self, in_channels, out_channels, head_size = 4, att = True):
        super(_denoising_block, self).__init__()
        self.isattention = att
        self.head_size = head_size
        if att :
            self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.query_conv =  nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)
        
    def multihead_transpose(self, x, HW):
        new_shape = x.size()[:-1] + (self.head_size, HW // self.head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def DenoisingBlock(self, x):
        B, C, H, W = x.shape

        if not self.isattention :
            x = x.reshape(B,C,H*W)
            out = torch.matmul(x.transpose(-2, -1), x)
            out = torch.matmul(x, out.transpose(-2,-1))

        else : 
            q_out = self.query_conv(x).reshape(B,C,H*W) ### B x C x H x W -> B x C x HW
            q_out = self.multihead_transpose(q_out, H*W).transpose(-1, -2) ### B x h x HW//h x C
            k_out = self.key_conv(x).reshape(B,C,H*W) ### B x C x H x W => B x C x HW
            k_out = self.multihead_transpose(k_out, H*W) 
            attention = self.softmax(torch.matmul(q_out, k_out) / math.sqrt(self.head_size))### B x h x HW//h x HW//h

            v_out = self.value_conv(x).reshape(B,C,H*W)
            v_out = self.multihead_transpose(v_out, H*W) ### B x h x C x HW//h
            out = torch.matmul(v_out, attention.permute(0,1,3,2)) ### B x h x C x HW//h

        return out.reshape((B, C, H, W))

    def forward(self, x):
        out = self.DenoisingBlock(x)
        out = self.conv1x1(out)
        out = x + out

        return out

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv = _conv_block(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class Denoising_UNet(nn.Module):
    """
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    https://arxiv.org/abs/1505.04597
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        depth (int): depth of network
        cf (int): Channel factor, number of channels for first layers output is 2 ** cf
    """

    def __init__(self, in_channels=1, out_channels=1, depth=5, cf=6):
        super(Denoising_UNet, self).__init__()
        self.depth = depth
        self.downs = nn.ModuleList(
            [
                _conv_block(
                    in_channels=(in_channels if i == 0 else 2 ** (cf + i - 1)),
                    out_channels=(2 ** (cf + i)),
                )
                for i in range(depth)
            ]
        )
        self.ups = nn.ModuleList(
            [
                Up(in_channels=(2 ** (cf + i + 1)), out_channels=(2 ** (cf + i)))
                for i in reversed(range(depth - 1))
            ]
        )
        self.denoising = nn.ModuleList(
            [
                _denoising_block(in_channels=(2 ** (cf + i)), out_channels=(2 ** (cf + i)))
                for i in reversed(range(depth - 1))
            ]
        )
        self.max = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1x1 = nn.Conv2d(2 ** cf, out_channels, kernel_size=1)
        self.n_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        
        outs = []
        for i, down in enumerate(self.downs):
            x = down(x)
            if i != (self.depth - 1):
                outs.append(x)
                x = self.max(x)

        for i, (denoise, up) in enumerate(zip(self.denoising, self.ups)):
            x = up(x, denoise(outs[-i - 1]))

        x = self.conv1x1(x)
        return x

class DUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth=5, cf=6):
        super(DUNet, self).__init__()
        self.depth = depth
        self.downs = nn.ModuleList(
            [
                _conv_block(
                    in_channels=(in_channels if i == 0 else 2 ** (cf + i - 1)),
                    out_channels=(2 ** (cf + i)),
                )
                for i in range(depth)
            ]
        )
        self.ups = nn.ModuleList(
            [
                Up(in_channels=(2 ** (cf + i + 1)), out_channels=(2 ** (cf + i)))
                for i in reversed(range(depth - 1))
            ]
        )

        self.max = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1x1 = nn.Conv2d(2 ** cf, out_channels, kernel_size=1)
        self.n_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        
        outs = []
        for i, down in enumerate(self.downs):
            x = down(x)
            
            if i != (self.depth - 1):
                outs.append(x)
                x = self.max(x)

        for i, up in enumerate(self.ups):
            x = up(x, outs[-i - 1])

        x = self.conv1x1(x)
        return x


def unet_denoising():
    return Denoising_UNet(in_channels=3, out_channels=3, depth=4, cf=6)

def dunet():
    return DUNet(in_channels=3, out_channels=3, depth=4, cf=6)