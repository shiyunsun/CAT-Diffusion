import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass, field
from typing import List, Tuple
from tqdm import tqdm
from Diffusion import *
from torchvision.utils import make_grid
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from UNet import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_images(model):
    model.eval()
    generated_images = model.sample(36).cpu()

    fig, axes = plt.subplots(6, 6, figsize = (6, 6), gridspec_kw = {'hspace': 0, 'wspace': 0})
    plt.subplots_adjust(wspace=0, hspace=0)
    axes = axes.flatten()
    for i, _ in enumerate(generated_images):
        ax = axes[i]
        ax.imshow(generated_images[i].permute(1,2,0))
        ax.axis('off')
    fig.tight_layout()
    return fig

def train_helper(train_loader, model, optimizer):
    model.train()
    train_loss = 0
    i = 0
    for images, conditions in tqdm(train_loader, leave = False, desc = 'training'):
        images = images.to(device)
        noise_loss = model(images)
        train_loss = train_loss + noise_loss.item()
        i = i + 1
        optimizer.zero_grad()
        noise_loss.backward()
        optimizer.step()
    return train_loss/i


def train(train_loader, epochs):
    if not os.path.exists('./save'): 
        os.mkdir('./save')
    if not os.path.exists('./images'): 
        os.mkdir('./images')
    best_loss = 10
    #model = Diffusion().to(device)
    model = torch.load('save/model.pth').to(device)
    optimizer = optim.Adam(model.unet.parameters(), lr = 1e-4)    
    for i in range(epochs):
        epoch = i + 1
        print('epoch {}/{}'.format(epoch, epochs))
        train_noise_loss = train_helper(train_loader, model, optimizer)
        print('train: train_noise_loss = {:.4f}'.format(train_noise_loss))
        torch.save(model, 'save/model.pth')
        fig = plot_images(model)
        fig.savefig(os.path.join('./images', f'generate_epoch_{epoch}.png'))
        plt.close(fig)
        
        