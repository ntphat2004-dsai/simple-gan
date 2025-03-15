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
from hyper_params import *

def train(generator, discriminator, gen_optimizer, dis_optimizer, 
          criterion, train_dataloader, writer_fake, writer_real, num_epochs=50):
    
    global GLOBAL_STEP
    
    for epoch in range(num_epochs): 
        for batch_idx, (real_img, _) in enumerate(train_dataloader):
            real_img = real_img.view(-1, IMAGE_DIM).to(DEVICE)
            batch_size = real_img.shape[0]

            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z))) --> x is real, z is noise

            # Generate fake images
            noise = torch.randn(batch_size, NOISE_DIM).to(DEVICE)
            fake_img = generator(noise)

            # Compute loss for real images
            dis_real = discriminator(real_img).view(-1) # view(-1) means flatten
            loss_dis_real = criterion(dis_real, torch.ones_like(dis_real)) # real image should be classified as real (1)

            # Compute loss for fake images
            dis_fake = discriminator(fake_img.detach()).view(-1) # detach() to avoid backpropagation to generator
            loss_dis_fake = criterion(dis_fake, torch.zeros_like(dis_fake)) # fake image should be classified as fake (0)

            # Combine losses
            loss_dis = (loss_dis_real + loss_dis_fake) / 2

            # Update Discriminator
            discriminator.zero_grad()
            loss_dis.backward()
            dis_optimizer.step()

            ### Train Generator: min log(1 - D(G(z))) ~ max log(D(G(z))
            dis_pred = discriminator(fake_img).view(-1) # view(-1) means flatten

            # Compute loss
            loss_gen = criterion(dis_pred, torch.ones_like(dis_pred)) # fake image should be classified as real (1)

            # Update Generator
            generator.zero_grad()
            loss_gen.backward()
            gen_optimizer.step()

            # Print losses
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}]\n\
                        Batch {batch_idx}/{len(train_dataloader)}\n\
                        Discriminator Loss: {loss_dis:.4f}\n\
                        Generator Loss: {loss_gen:.4f}")
                
                with torch.no_grad():
                    fake = generator(FIXED_NOISE).reshape(-1, 1, 28, 28)
                    data = real_img.reshape(-1, 1, 28, 28)
                    img_grid_fake = make_grid(fake, normalize=True)
                    img_grid_real = make_grid(data, normalize=True)
                    writer_fake.add_image("Fake Images", img_grid_fake, global_step=GLOBAL_STEP)
                    writer_real.add_image("Real Images", img_grid_real, global_step=GLOBAL_STEP)

                    GLOBAL_STEP += 1
