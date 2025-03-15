import torch

# Hyperparameters
IMAGE_DIM = 28*28
NOISE_DIM = 100
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FIXED_NOISE = torch.randn((BATCH_SIZE, NOISE_DIM)).to(DEVICE) # BATCH_SIZE images to visualize generator's progress 
# Why need fixed noise? Because we want to see how the generator improves over time by generating images from the same noise
GLOBAL_STEP = 0 # Tensorboard global step for visualization purposes
