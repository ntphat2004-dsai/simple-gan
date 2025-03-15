# Simple GAN with MNIST

## Introduction

This project implements a simple Generative Adversarial Network (GAN) trained on the MNIST handwritten digits dataset from `torchvision.datasets`. The training process is monitored using TensorBoard for visualization.

### What is GAN?

A Generative Adversarial Network (GAN) consists of two neural networks:

- **Generator (G)**: Learns to generate realistic-looking images from random noise.
- **Discriminator (D)**: Tries to distinguish between real images (from the dataset) and fake images (produced by the generator).

The two networks are trained simultaneously in an adversarial manner, where the generator tries to fool the discriminator, and the discriminator tries to correctly classify real and fake images.

## Requirements

Ensure you have Python **3.12.4+** installed. Then install the required dependencies using:

```sh
pip install -r requirements.txt
```

Required libraries:

- `torch`
- `torchvision`
- `tensorboard`
- Other common dependencies (see `requirements.txt`)

## Usage

To train the GAN model, run the following command:

```sh
python main.py
```

## Improvements

This project includes several enhancements to improve performance and stability:

- **Batch Normalization**: Helps stabilize training by normalizing inputs to each layer.
- **LeakyReLU Activation**: Used in the discriminator to help with gradient flow.
- **Adam Optimizer**: Provides better optimization performance compared to standard SGD.
- **Label Smoothing**: Helps the discriminator generalize better and avoid overconfidence.
- **Feature Matching**: An alternative loss function for stabilizing training.

## Notes

- The model is a simple GAN designed for educational purposes and may require fine-tuning for better results.
- The architecture follows a basic fully connected neural network design for both the generator and discriminator.



 
