import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import math

from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms, datasets

import scipy.io
from torchvision import transforms
import tarfile

device = 'cuda' if torch.cuda.is_available() else 'cpu'



def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def sample_from_distribution(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std




@torch.no_grad()
def check_vae_quality(vae, dataloader, device):
    """
    Fixed version that works with MPS
    """
    # Get a batch of real images
    images = next(iter(dataloader))
    
    # Ensure images are float32 and on correct device
    images = images.to(device).float()
    
    # Ensure VAE is on correct device and in eval mode
    vae = vae.to(device).eval()
    
    # Pass through VAE
    recon, _, _ = vae(images)
    
    # Prepare for plotting (Denormalize and move to CPU)
    def to_img(x):
        x = x * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        return x.clamp(0, 1).cpu().permute(0, 2, 3, 1)
    
    real_imgs = to_img(images)
    recon_imgs = to_img(recon)
    
    # Plot top 4 results
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(4):
        axes[0, i].imshow(real_imgs[i])
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(recon_imgs[i])
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def check_single_image(vae, image_path, device="cuda", img_size=96):
    """
    Fixed version that works with MPS
    """
    # 1. Load and Preprocess the specific image
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Scale [0, 1] -> [-1, 1] to match training
    ])

    img_raw = Image.open(image_path).convert("RGB")
    img_tensor = transform(img_raw).unsqueeze(0)
    
    # Ensure images are float32 and on correct device
    img_tensor = img_tensor.to(device).float()
    
    # Ensure VAE is on correct device and in eval mode
    vae = vae.to(device).eval()

    # 2. Run through VAE
    # Based on your logs, your VAE returns (recon, mu, logvar)
    recon, _, _ = vae(img_tensor)

    # 3. Denormalize helper
    def to_img(x):
        x = x * 0.5 + 0.5 # [-1, 1] -> [0, 1]
        return x.clamp(0, 1).cpu().permute(0, 2, 3, 1).squeeze(0).numpy()

    real_plot = to_img(img_tensor)
    recon_plot = to_img(recon)

    # 4. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    axes[0].imshow(real_plot)
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(recon_plot)
    axes[1].set_title("VAE Reconstruction")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_noise_progression(dataloader, scheduler):
    """
    Selects a random image from the dataloader and plots its progression
    through the noise schedule at 5 evenly spaced timesteps.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch = next(iter(dataloader))
    if isinstance(batch, (list, tuple)):
        images = batch[0]
    else:
        images = batch

    idx = random.randint(0, images.size(0) - 1)
    z_0 = images[idx:idx+1].to(device)

    # Define 5 evenly spaced timesteps
    # For 1000 steps, this results in [0, 249, 499, 749, 999]
    num_plots = 5
    timesteps = torch.linspace(0, scheduler.num_timesteps - 1, num_plots).long().to(device)

    fig, axes = plt.subplots(1, num_plots, figsize=(15, 3))

    for i, t in enumerate(timesteps):
        # Sample noise for this specific timestep
        noise = torch.randn_like(z_0)

        with torch.no_grad():
            z_t = scheduler.add_noise(z_0, t.unsqueeze(0), noise)

        img_to_show = (z_t.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
        img_to_show = img_to_show.permute(1, 2, 0).cpu().numpy()

        axes[i].imshow(img_to_show)
        axes[i].set_title(f"Step {t.item()}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        # timesteps: (batch_size,)
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


@torch.no_grad()
def calculate_scaling_factor(vae, dataloader, num_batches=100, device=None):
    if device is None:
        # Auto-detect device from VAE or use global device
        device = next(vae.parameters()).device
    
    vae = vae.to(device).eval()
    latents = []

    for i, images in enumerate(dataloader):
        if i >= num_batches:
            break
        # Ensure float32 and correct device for MPS compatibility
        images = images.to(device).float()
        mu, logvar = vae.encoder(images)
        z = vae.reparameterize(mu, logvar)
        latents.append(z)

    latents = torch.cat(latents, dim=0)
    scale = 1.0 / latents.std()
    print(f"Calculated scaling factor: {scale:.5f}")
    return scale



def infer_latent_shape(encoder, image_size, in_channels=3, device=None):
    if device is None:
        device = next(encoder.parameters()).device

    encoder.eval()
    with torch.no_grad():
        x = torch.zeros(1, in_channels, image_size, image_size, device=device)
        mu, logvar = encoder(x)

    return mu.shape[1:]  # (C, H, W)
