# Model inspired by https://github.com/milesial/Pytorch-UNet/

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms



def convolution(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class DownBlock(nn.Module):
    """DownBlock: block_conv => MaxPooling"""
    def __init__(self, in_channels, out_channels, n_conv=2):
        super(DownBlock, self).__init__()

        assert 2 <= n_conv <= 3
        self.block_conv = nn.Sequential(convolution(in_channels=in_channels, out_channels=out_channels))
        for i in range(n_conv - 1):
            self.block_conv.append(convolution(in_channels=out_channels, out_channels=out_channels))

        self.pool2d = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x):
        x_conv = self.block_conv(x)
        x_pool, indices = self.pool2d(x_conv)
        return x_conv, x_pool, indices

class UpBlock(nn.Module):
    """UpBlock: MaxUnpooling => block_conv => output_conv & next_conv"""
    def __init__(self, in_channels, out_channels, n_conv=2):
        super(UpBlock, self).__init__()

        self.upsample = nn.MaxUnpool2d(kernel_size=2)

        assert 2 <= n_conv <= 3
        self.block_conv = nn.Sequential(convolution(in_channels=in_channels, out_channels=in_channels))
        for i in range(n_conv - 2):
            self.block_conv.append(convolution(in_channels=in_channels, out_channels=in_channels))
        self.output_conv = convolution(in_channels=in_channels, out_channels=in_channels)
        self.next_conv = convolution(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, indices):
        x = self.upsample(x, indices)
        x = self.block_conv(x)
        output = self.output_conv(x)
        x = self.next_conv(x)
        return x, output

class OutputBlock(nn.Module):
    """OutputBlock: 1-conv => sigmoid => loss"""
    def __init__(self, in_channels, out_channels, scale_factor):
        super(OutputBlock, self).__init__()
        self.flat_conv = nn.Conv2d(2 * in_channels, out_channels=out_channels, kernel_size=3, padding='same')
        self.deconv = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=scale_factor, stride=scale_factor)

    def forward(self, x_encode, x_decode):
        x = torch.cat((x_encode, x_decode), dim=1)
        x = self.flat_conv(x)
        x = self.deconv(x)
        return x


class Encoder(nn.Module):
    """Encoder: 2 DownBlock(2) => 3 DownBlock(3)"""
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.block1 = DownBlock(in_channels=in_channels, out_channels=64, n_conv=2)
        self.block2 = DownBlock(in_channels=64, out_channels=128, n_conv=2)
        self.block3 = DownBlock(in_channels=128, out_channels=256, n_conv=3)
        self.block4 = DownBlock(in_channels=256, out_channels=512, n_conv=3)
        self.block5 = DownBlock(in_channels=512, out_channels=512, n_conv=3)

    def forward(self, img):
        output1, x, indice1 = self.block1(img)
        output2, x, indice2 = self.block2(x)
        output3, x, indice3 = self.block3(x)
        output4, x, indice4 = self.block4(x)
        output5, x, indice5 = self.block5(x)
        return x, (indice1, indice2, indice3, indice4, indice5), (output1, output2, output3, output4, output5)

class Decoder(nn.Module):
    """Decoder: 3 UpBlock(3) => 2 UpBlock(2)"""
    def __init__(self):
        super(Decoder, self).__init__()
        self.block1 = UpBlock(in_channels=512, out_channels=512, n_conv=3)
        self.block2 = UpBlock(in_channels=512, out_channels=256, n_conv=3)
        self.block3 = UpBlock(in_channels=256, out_channels=128, n_conv=3)
        self.block4 = UpBlock(in_channels=128, out_channels=64, n_conv=2)
        self.block5 = UpBlock(in_channels=64, out_channels=1, n_conv=2)

    def forward(self, x, indices):
        x, output5 = self.block1(x, indices[4])
        x, output4 = self.block2(x, indices[3])
        x, output3 = self.block3(x, indices[2])
        x, output2 = self.block4(x, indices[1])
        x, output1 = self.block5(x, indices[0])
        return output1, output2, output3, output4, output5


class DeepCrack(nn.Module):

    def __init__(self, n_channels, n_classes):
        super(DeepCrack, self).__init__()
        self.encoder = Encoder(in_channels=n_channels)
        self.decoder = Decoder()

        self.outBlock1 = OutputBlock(in_channels=64, out_channels=n_classes, scale_factor=1)
        self.outBlock2 = OutputBlock(in_channels=128, out_channels=n_classes, scale_factor=2)
        self.outBlock3 = OutputBlock(in_channels=256, out_channels=n_classes, scale_factor=4)
        self.outBlock4 = OutputBlock(in_channels=512, out_channels=n_classes, scale_factor=8)
        self.outBlock5 = OutputBlock(in_channels=512, out_channels=n_classes, scale_factor=16)

        self.outConv = nn.Conv2d(in_channels=5*n_classes, out_channels=n_classes, kernel_size=3, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        x, indices, outputs_encoder = self.encoder(img)
        outputs_decoder = self.decoder(x, indices)

        output1 = self.outBlock1(outputs_encoder[0], outputs_decoder[0])
        output2 = self.outBlock2(outputs_encoder[1], outputs_decoder[1])
        output3 = self.outBlock3(outputs_encoder[2], outputs_decoder[2])
        output4 = self.outBlock4(outputs_encoder[3], outputs_decoder[3])
        output5 = self.outBlock5(outputs_encoder[4], outputs_decoder[4])

        outputs = torch.cat((output1, output2, output3, output4, output5), dim=1)
        outputs = self.outConv(outputs)

        return self.sigmoid(outputs), self.sigmoid(output1), self.sigmoid(output2), self.sigmoid(output3), self.sigmoid(output4), self.sigmoid(output5)

    def __repr__(self) -> str:
        return str(self.__class__.__name__)

    def __str__(self) -> str:
        return super(DeepCrack, self).__str__()
