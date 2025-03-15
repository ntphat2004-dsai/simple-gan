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
import webbrowser, time, subprocess, sys, os
from IPython.display import display, HTML


# def launch_tensorboard():
#     try:
#         if 'google.colab' in sys.modules:
#             from tensorboard import notebook
#             notebook.start("--logdir=runs/GAN_MNIST")
#             print("TensorBoard is running inline in Colab. You can access it below.")
#             return None
#         else:
#             tb_process = subprocess.Popen(
#                 ["tensorboard", "--logdir", "runs/GAN_MNIST", "--host", "0.0.0.0"],
#                 stdout=subprocess.PIPE, stderr=subprocess.PIPE
#             )
#             time.sleep(5)
#             url = "http://localhost:6006"
#             if "KAGGLE_URL_BASE" in os.environ:
#                 display(HTML(f'<a href="{url}" target="_blank">Click here to open TensorBoard</a>'))
#                 print("TensorBoard is running. Click the link above to open it.")
#             else:
#                 print("TensorBoard URL:", url)
#                 webbrowser.open(url)
#             return tb_process
#     except Exception as e:
#         print("Error launching TensorBoard:", e)
#         return None


if __name__ == "__main__":
    # Prepare Data
    train_data, _ = prepare_data()
    train_dataloader = create_dataloader(train_data, BATCH_SIZE)

    # Initialize Models
    generator = Generator(NOISE_DIM, IMAGE_DIM).to(DEVICE)
    discriminator = Discriminator(IMAGE_DIM).to(DEVICE)

    # Initialize Optimizers
    gen_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

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

    # Train
    train(generator, discriminator, gen_optimizer, dis_optimizer, 
          criterion, train_dataloader, writer_fake, writer_real)