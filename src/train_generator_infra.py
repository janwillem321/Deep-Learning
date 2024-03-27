import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
import torch.optim as optim
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, DatasetFolder
from torch.utils.data import DataLoader
import os
from PIL import Image

from discriminator import Discriminator

folder_dir_in_matthijs = r'C:\Users\matth\Documents\Master Nanobiology\Deep learning\github\train_folder\Input_images'
folder_dir_gt_matthijs = r'C:\Users\matth\Documents\Master Nanobiology\Deep learning\github\train_folder\ground_truth'

def train_generator(input_imgs, real_imgs):

    n_samples, in_channels, s_img, latent_dims = len(input_imgs), 6, 256, 512 # 6 for two images
    hdim = [64, 128, 256, 256, 512, 512, 1] #choose hidden dimension discriminator
    x = torch.empty((n_samples, in_channels, s_img, s_img))

    generator = Generator(PLACEHOLDER, PARAMS) # TODO:
    discriminator  = Discriminator(latent_dims, s_img, hdim)
    loss_L1 = nn.L1Loss()
    loss_BCE = nn.BSELoss()
    optimizer = optim.Adam(generator.parameters(), lr=2e-4)

    for i in range(len(input_imgs)):
        input_img = input_imgs[i]
        real_img = real_imgs[i]

        target = real_img
        enhanced_img = Generator(input_imgs)

        x[i, :, :, :] = torch.cat((input_img, enhanced_img), dim=0)

        # Right Column
        discriminator_out = discriminator.forward(x)

        loss_1 = loss_L1(enhanced_img, real_img)
        loss_2 = loss_BCE(discriminator_out, torch.ones(30, 30))

        total_loss = loss_1 + loss_2

        #Apply Gradients
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()



folder_dir_gt = folder_dir_in_matthijs
folder_dir_in = folder_dir_gt_matthijs

in_list = [os.path.join(folder_dir_in, filename) for filename in os.listdir(folder_dir_in)]
gt_list = [os.path.join(folder_dir_gt, filename) for filename in os.listdir(folder_dir_gt)]
enhanced_list = [os.path.join(folder_dir_in, filename) for filename in os.listdir(folder_dir_in)]
