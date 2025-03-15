import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from simple_gan import *
from prepare_data import *
from train import *
from hyper_params import *
import webbrowser, time, subprocess


if __name__ == "__main__":
    # Prepare Data
    train_data, _ = prepare_data()
    train_dataloader = create_dataloader(train_data, BATCH_SIZE)

    # Initialize Models
    generator = Generator(NOISE_DIM, IMAGE_DIM).to(DEVICE)
    discriminator = Discriminator(IMAGE_DIM).to(DEVICE)

    # Initialize Optimizers
    gen_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # Loss Function
    criterion = nn.BCELoss()

    # Tensorboard
    writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
    writer_real = SummaryWriter(f"runs/GAN_MNIST/real")

    # Launch TensorBoard
    # tb_process = launch_tensorboard()

    # Print model architecture
    print(generator)
    print(discriminator)

    # Print device
    print(f"Device: {DEVICE}")

    # Open TensorBoard
    subprocess.Popen(["tensorboard", "--logdir", "runs/GAN_MNIST", "--host", "0.0.0.0"])
    time.sleep(5)
    print("TensorBoard URL: http://localhost:6006")
    webbrowser.open("http://localhost:6006")

    # Train
    train(generator, discriminator, gen_optimizer, dis_optimizer, 
          criterion, train_dataloader, writer_fake, writer_real)