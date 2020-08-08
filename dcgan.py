import torch
import torch.nn as nn

# Biases are disabled, as Batch Normalization effectively removes them.
class Generator(nn.Module):
    def __init__(self, num_noises=100, num_colors=3, depths=128, image_size=64):
        super(Generator, self).__init__()

        if image_size % 16 != 0:
            raise Exception("Size of the image must be divisible by 16")
        self.final_size = image_size // 16
        self.depths = depths

        self.lin = nn.Linear(num_noises, depths * 8 * self.final_size * self.final_size)
        # Some calculations behind deciding kernel size, stride, padding:
        # (x+2p-k)/s+1=x/2 <=> s != 0 or 2, k = ... or p = k/2 - 1, s = 2
        # => Choose k = 4, s = 2, p = 1
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(depths * 8, depths * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(depths * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(depths * 4, depths * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(depths * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(depths * 2, depths, 4, 2, 1, bias=False),
            nn.BatchNorm2d(depths),
            nn.ReLU(),
            nn.ConvTranspose2d(depths, num_colors, 4, 2, 1, bias=False),
            # Paper says: "... more quickly to sature and cover the color space ..."
            nn.Tanh()
        )
    
    def forward(self, x):
        lin_out = self.lin(x)
        conv_out = self.conv(lin_out.view(-1, self.depths * 8, self.final_size, self.final_size))
        return conv_out

class Discriminator(nn.Module):
    def __init__(self, num_colors=3, depths=128, image_size=64):
        super(Discriminator, self).__init__()

        if image_size % 16 != 0:
            raise Exception("Size of the image must be divisible by 16")
        
        final_size = image_size // 16
        self.conv = nn.Sequential(
            nn.Conv2d(num_colors, depths, 4, 2, 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(depths, depths * 2, 4, 2, 1),
            nn.BatchNorm2d(depths * 2),
            nn.LeakyReLU(),
            nn.Conv2d(depths * 2, depths * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(depths * 4),
            nn.LeakyReLU(),
            nn.Conv2d(depths * 4, depths * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(depths * 8),
            nn.LeakyReLU(),
            # Paper is unclear about what does "flattened and then fed
            # into a single sigmoid output" mean; I'll use nn.Linear()
            nn.Flatten(),
            nn.Linear(depths * 8 * final_size * final_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        pred = self.conv(x)
        return pred

class PrintShape(nn.Module):
    def __init__(self):
        super(PrintShape, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x
