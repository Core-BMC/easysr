import torch
import torch.nn as nn

def conv_block(ndf, in_channels, out_channels, kernel_size, stride, padding):
    """Defines a convolutional block with convolution, batch normalization, and LeakyReLU activation."""
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm3d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )

class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc=1, ndf=3, output_size=1):
        """Initializes the Patch Discriminator model.
        
        Args:
            input_nc (int): Number of input channels. Default is 1 (e.g., for grayscale images).
            ndf (int): Number of filters in the first convolution layer. Default is 64.
        """
        super(PatchDiscriminator, self).__init__()
        
        # Define convolutional blocks
        self.conv1 = conv_block(ndf, input_nc, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = conv_block(ndf, ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = conv_block(ndf * 2, ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = conv_block(ndf * 4, ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)

        # Final convolution layer
        self.conv5 = nn.Conv3d(ndf * 8, ndf * 8, kernel_size=4, padding=1)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layer
        self.fc = nn.Linear(ndf * 8 * 7 * 11 * 7, output_size)

        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)        
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
