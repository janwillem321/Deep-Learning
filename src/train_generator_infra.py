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

folder_dir_in_matthijs = r'C:\Users\matth\Documents\Master Nanobiology\Deep learning\github\train_folder\Input_images'
folder_dir_gt_matthijs = r'C:\Users\matth\Documents\Master Nanobiology\Deep learning\github\train_folder\ground_truth'

def train_generator(input_imgs, real_imgs):

    model = Generator(PLACEHOLDER, PARAMS) # TODO:

    loss = nn.L1Loss()
    loss2 = nn.BSELoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    for i in range(len(input_imgs)):
        input_img = input_imgs[i]
        real_img = real_imgs[i]

        target = real_img
        enhanced_img = Generator(input_imgs)

        # Left column
        output = loss(enhanced_img, real_img)

        # Right Column
        discriminator_out = Discriminator(enhanced_img, input_img)

        output2 = loss2(discriminator_out, torch.ones(30, 30))

        total_loss = output + output2

        #Apply Gradients
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()



folder_dir_gt = "Deep-Learning/train_folder/ground_truth"
folder_dir_in = "Deep-Learning/train_folder/input_images"

in_list = [os.path.join(folder_dir_in, filename) for filename in os.listdir(folder_dir_in)]
gt_list = [os.path.join(folder_dir_gt, filename) for filename in os.listdir(folder_dir_gt)]
enhanced_list = [os.path.join(folder_dir_in, filename) for filename in os.listdir(folder_dir_in)]
