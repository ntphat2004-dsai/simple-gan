import torch 
import torch.nn as nn


# Build Discriminator
class Discriminator(nn.Module):
  def __init__(self, image_dim):
    super(Discriminator, self).__init__()
    self.image_dim = image_dim

    self.main_block = nn.Sequential(
        nn.Linear(self.image_dim, 1024),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.3),

        nn.Linear(1024, 512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.3),

        nn.Linear(512, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.3),

        nn.Linear(256, 1),  # Output layer: fake or no fake => 0 or 1
        nn.Sigmoid()
    )

  def forward(self, x):
    return self.main_block(x)

# Build Generator
class Generator(nn.Module):
  def __init__(self, noise_dim, image_dim):
    super(Generator, self).__init__()
    self.noise_dim = noise_dim
    self.image_dim = image_dim

    self.main_block = nn.Sequential(
        nn.Linear(self.noise_dim, 256),
        nn.BatchNorm1d(256),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Linear(256, 512),
        nn.BatchNorm1d(512),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Linear(512, 1024),
        nn.BatchNorm1d(1024),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Linear(1024, self.image_dim),
        nn.Tanh()   # Output layer: -1 to 1 => image has value from -1 to 1 (normalized)
    )

  def forward(self, x):
    return self.main_block(x)