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


# =============================================================================
# Define the Discriminator
# =============================================================================

class Discriminator(nn.Module):
    
    def __init__(self, latent_dims, s_img, hdim, kernel_size=(4, 4), stride=2):
        
        super(Discriminator, self).__init__()

        ########################################################################
        #    Create the necessary layers                                 #
        ########################################################################

        self.layers = nn.ModuleList()
        
        # Input layer dim -- down1
        self.layers.append(nn.Conv2d(in_channels=6, out_channels=64, kernel_size=kernel_size, stride=2, padding=1))

        # Hidden to hidden convolution -- down2 and down 3
        for i in range(0, 2):
            self.layers.append(nn.Conv2d(in_channels=hdim[i],
                                             out_channels=hdim[i + 1],
                                             kernel_size=kernel_size, stride = stride, padding=1))

        # Pad with zeroes
        self.layers.append(nn.ZeroPad2d(padding=(1,1,1,1)))

        # Conv2D
        self.layers.append(nn.Conv2d(in_channels=hdim[3],
                                             out_channels=hdim[4],
                                             kernel_size=kernel_size, stride = 1))

        # Batchnorm
        self.layers.append(nn.BatchNorm2d(hdim[4]))

        # Zeropad2
        self.layers.append(nn.ZeroPad2d(padding=(1,1,1,1)))

        #Conv2D 2
        self.layers.append(nn.Conv2d(in_channels=hdim[5],
                                             out_channels=hdim[6],
                                             kernel_size=kernel_size, stride = 1))

        self.Leakyrelu = nn.LeakyReLU(0.2)
        

    def forward(self, x):

        for n_layer, layer in enumerate(self.layers):
            ## The fourth layer first has a batchnorm and then a Leakyrelu
            if n_layer != 4:
                x = self.Leakyrelu(layer(x))
            else:
                x = layer(x)
        return x


def generate_fake_image():
    return torch.randn(in_channels//2, s_img, s_img)
    

# =============================================================================
# test auo-encoder
# =============================================================================
n_samples, in_channels, s_img, latent_dims = 10, 6, 256, 512 # 6 for two images
hdim = [64, 128, 256, 256, 512, 512, 1] #choose hidden dimension discriminator
output_shape = (n_samples, 1, 30, 30)


x = torch.empty((n_samples, in_channels, s_img, s_img))
labels = torch.empty(output_shape)

# =============================================================================
# Fill x with stacks of fake images (random) and real images (all ones)
# =============================================================================
# real_image = torch.ones(in_channels//2, s_img, s_img)

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the images to a consistent size
    transforms.ToTensor()           # Convert the images to PyTorch tensors
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
])


# get the path or directory
folder_dir_gt = "Deep-Learning/train_folder/ground_truth"
folder_dir_in = "Deep-Learning/train_folder/input_images"
# get the path or directory
# folder_dir_gt = "/Users/rexfleur/PycharmProjects/pix_gan/Deep-Learning/train_folder/ground_truth"
# folder_dir_in = "/Users/rexfleur/PycharmProjects/pix_gan/Deep-Learning/train_folder/input_images"

in_list = [os.path.join(folder_dir_in, filename) for filename in os.listdir(folder_dir_in)]
gt_list = [os.path.join(folder_dir_gt, filename) for filename in os.listdir(folder_dir_gt)]

print(gt_list)

for i in range(10):
    gt_img = transform(Image.open(gt_list.pop()))
    input_img = transform(Image.open(in_list.pop()))

    if i % 2 == 0:  # Even indices for real images
        x[i, :, :, :] = torch.cat((input_img, gt_img), dim=0)
        print(torch.cat((input_img, gt_img), dim=0).shape)
        labels[i, 0] = torch.ones(30, 30)

    else:  # Odd indices for fake images
        x[i, :, :, :] = torch.cat((input_img, input_img), dim=0)
        labels[i, 0] = torch.zeros(30, 30)

    i+=1

# =============================================================================
# initialize model
# =============================================================================
model = Discriminator(latent_dims, s_img, hdim)
summary(model, (in_channels, s_img, s_img), device='cpu') # (in_channels, height, width)

# =============================================================================
# training of the Discriminator
# =============================================================================
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-4)
n_epochs = 15
losses = []

# Training loop
for epoch in range(n_epochs):
    
    # Forward pass
    outputs = model(x)
    output_probabilities = torch.sigmoid(outputs)
   
    # Calculate loss
    loss = criterion(output_probabilities, labels)
    print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item()}')
    losses.append(loss.item())
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# =============================================================================
# plot the training Loss over epochs (on its own data,
# so will eventually convege to 0)
# =============================================================================

plt.plot(losses)
plt.ylim(0, .8)
plt.xlabel('epochs')
plt.ylabel('Loss (BCE)')
plt.show()