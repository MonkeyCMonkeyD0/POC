# Model replicated from https://github.com/nowtryz/SubUnet

import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

def ppm_conv(in_channels, module_count=4):
    out_channels = in_channels // module_count
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
        nn.ReLU(inplace=True),
    )



class DownBlock(nn.Module):
    """DownBlock: block_conv => MaxPooling"""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = double_conv(in_channels=in_channels, out_channels=out_channels)
        self.pool2d = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x_conv = self.conv(x)
        x_pool = self.pool2d(x_conv)
        return x_conv, x_pool

class UpBlock(nn.Module):
    """UpBlock: MaxUnpooling => block_conv => output_conv & next_conv"""
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(kernel_size=2, in_channels=in_channels, out_channels=in_channels//2, stride=2)
        self.conv = double_conv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x_previous, x_opposite):
        x_previous = self.upsample(x_previous)
        x = torch.cat((x_opposite, x_previous), dim=1)
        x = self.conv(x)
        return x

class PPMBlock(nn.Module):
    """PPMBlock: 1-conv => sigmoid => loss"""
    def __init__(self, in_channels, factor):
        super(PPMBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(factor, factor))
        self.conv = ppm_conv(in_channels=in_channels)

    def forward(self, x):
        img_size = x.shape[-2:]
        x = self.pool(x)
        x = self.conv(x)
        x = nn.functional.interpolate(x, size=img_size, mode='bilinear', align_corners=True)
        return x


class Encoder(nn.Module):
    """Encoder: 2 DownBlock(2) => 3 DownBlock(3)"""
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.block1 = DownBlock(in_channels=in_channels, out_channels=64)
        self.block2 = DownBlock(in_channels=64, out_channels=128)
        self.block3 = DownBlock(in_channels=128, out_channels=256)
        self.block4 = double_conv(in_channels=256, out_channels=512)

    def forward(self, img):
        output1, x = self.block1(img)
        output2, x = self.block2(x)
        output3, x = self.block3(x)
        x = self.block4(x)
        return x, (output1, output2, output3) #, (indice1, indice2, indice3)

class Decoder(nn.Module):
    """Decoder: 3 UpBlock(3) => 2 UpBlock(2)"""
    def __init__(self):
        super(Decoder, self).__init__()
        self.block1 = UpBlock(in_channels=128, out_channels=64)
        self.block2 = UpBlock(in_channels=256, out_channels=128)
        self.block3 = UpBlock(in_channels=512, out_channels=256)
        self.block4 = double_conv(in_channels=1024, out_channels=512)

    def forward(self, x, outputs):
        x = self.block4(x)
        x = self.block3(x, outputs[2])
        x = self.block2(x, outputs[1])
        x = self.block1(x, outputs[0])
        return x

class PPM(nn.Module):
    def __init__(self, in_channels):
        super(PPM, self).__init__()
        self.pool_mod1 = PPMBlock(in_channels=in_channels, factor=1)
        self.pool_mod2 = PPMBlock(in_channels=in_channels, factor=2)
        self.pool_mod3 = PPMBlock(in_channels=in_channels, factor=8)
        self.pool_mod4 = PPMBlock(in_channels=in_channels, factor=16)

    def forward(self, x):
        return torch.cat([
            x,
            self.pool_mod4(x),
            self.pool_mod3(x),
            self.pool_mod2(x),
            self.pool_mod1(x),
        ], dim=1)


class SubUNet(nn.Module):

    def __init__(self, n_channels, n_classes):
        super(SubUNet, self).__init__()
        self.encoder = Encoder(in_channels=n_channels)
        self.decoder = Decoder()
        self.ppm_mod = PPM(in_channels=512)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=3, padding='same')
        self.sigmoid = nn.Softmax(dim=1)

    def forward(self, img):
        x, outputs = self.encoder(img)
        x = self.ppm_mod(x)
        x = self.decoder(x, outputs)
        x = self.out_conv(x)
        result = self.sigmoid(x)
        return result

    def __repr__(self) -> str:
        return str(self.__class__.__name__)
