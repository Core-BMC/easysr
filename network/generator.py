import torch
import torch.nn as nn

# Resnet Block
class ResnetBlock(nn.Module):
    def __init__(self, inf, onf):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(inf, onf)

    def build_conv_block(self, inf, onf):
        conv_block = [
            nn.Conv3d(inf, onf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(onf), 
            nn.LeakyReLU(0.2)
        ]
        conv_block += [
            nn.Conv3d(onf, onf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(onf)
        ]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

# DeUpBlock for upsampling in the width dimension
class DeUpBlock(nn.Module):
    def __init__(self, inf, onf):
        super(DeUpBlock, self).__init__()
        self.deupblock = nn.Sequential(
            nn.ConvTranspose3d(inf, onf, kernel_size=(1, 3, 1), stride=(1, 3, 1), padding=(0, 0, 0)),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.deupblock(x)

# Resnet Generator
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, ngf=32, n_residual_blocks=2):
        super(ResnetGenerator, self).__init__()
        self.n_residual_blocks = n_residual_blocks

        self.conv_block1 = nn.Sequential(
            nn.Conv3d(input_nc, ngf, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )

        for i in range(n_residual_blocks):
            self.add_module(f'residual_block{i+1}', ResnetBlock(ngf, ngf))

        self.conv_block2 = nn.Sequential(
            nn.Conv3d(ngf, ngf, kernel_size=3, padding=1),
            nn.BatchNorm3d(ngf)
        )

        self.deup = DeUpBlock(ngf, ngf)
        self.conv3 = nn.Conv3d(ngf, output_nc, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_block1(x)
        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__(f'residual_block{i+1}')(y)
        x = self.conv_block2(y) + x
        x = self.deup(x)
        return self.conv3(x)
