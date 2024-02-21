import torch
from torch import nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):  # output_dim == img_dim
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, output_dim),
            nn.Tanh()  # ensure that output pixel values are between -1, 1 -> that's how we would normalize the data
        )

    def forward(self, z):
        return self.gen(z)


if __name__ == '__main__':
    print(torch.__version__)