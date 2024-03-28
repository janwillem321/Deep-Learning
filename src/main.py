import torch
from generator import Generator
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataloader import PairedImageDataset
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from train_generator import train, try_gpu
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def show_images():    
   
    # Define your transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Initialize the dataset
    paired_dataset = PairedImageDataset(rootA=r'archive\EUVP\Paired\underwater_dark\trainA',
                                         rootB=r'archive\EUVP\Paired\underwater_dark\trainB',
                                           transform=transform)

    # Initialize DataLoader
    EUVP_data = DataLoader(paired_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

   # Fetch a single batch
    images_a, images_b = next(iter(EUVP_data))

    # Setup for displaying 15 pairs (30 images total)
    fig, axs = plt.subplots(15, 2, figsize=(10, 45))  # Adjust figure size as needed

    for i in range(15):
        # Display image from trainA
        img_a = images_a[i].permute(1, 2, 0)  # Convert to HxWxC format
        axs[i, 0].imshow(img_a.numpy())
        axs[i, 0].axis('off')
        axs[i, 0].set_title("TrainA")
        
        # Display corresponding image from trainB
        img_b = images_b[i].permute(1, 2, 0)  # Convert to HxWxC format
        axs[i, 1].imshow(img_b.numpy())
        axs[i, 1].axis('off')
        axs[i, 1].set_title("TrainB")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # show_images()
    num_workers = 4
    pin_memory = True
    batch_size = 128
    n_samples, in_channels, s_img, latent_dims, padding = 1, 3, 256, 512,1
    hdim_e = [3, 64, 128, 256, 512, 512, 512, 512, 512] #choose hidden dimension encoder
    hdim_d_input = [512, 1024, 1024, 1024, 1024, 512, 256, 128, 3] #choose hidden dimension decoder
    hdim_d_output = [512, 512, 512, 512, 512, 256, 128, 64, 3]
    kernel_size = (4,4)

        # Define your transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Initialize the dataset
    paired_dataset = PairedImageDataset(rootA=r'archive\EUVP\Paired\underwater_dark\trainA',
                                        rootB=r'archive\EUVP\Paired\underwater_dark\trainB',
                                        transform=transform)

    # Initialize DataLoader
    EUVP_data = DataLoader(paired_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

   # Fetch a single batch
    images_a, images_b = next(iter(EUVP_data))

    #Set the number of dimensions of the latent space
    # latent_dims = [2,3]
    s_img = np.size(images_a[1][0], axis = 1) #get image size (height = width) from a data sample
    # hdim = [100, 50]



    #initialize model
    model = Generator(latent_dims=latent_dims,
                        s_img=s_img,
                        hdim_e=hdim_e, 
                        hdim_d_input=hdim_d_input,
                        hdim_d_output=hdim_d_output, 
                        padding=padding,
                        kernel_size=kernel_size)

    # Create a writer to write to Tensorboard
    writer = SummaryWriter()

    #Create instance of Autoencoder
    device = try_gpu()
    print(device)
    if torch.cuda.is_available():
        model.cuda()
    # AE2 = Generator(latent_dims[0], s_img, hdim = hdim).to(device) #2-dimensional latent space
    # AE3 = Generator(latent_dims[1], s_img, hdim = hdim).to(device) #3-dimensional latent space

    # Create loss function and optimizer
    criterion = F.mse_loss

    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    # Set the number of epochs to for training
    epochs = 20

    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
        # Train on data
        train_loss = train(EUVP_data, model, optimizer, criterion, pin_memory, device)

        # Write metrics to Tensorboard
        writer.add_scalars("Loss", {'Train': train_loss}, epoch)



    # # Create a writer to write to Tensorboard
    # writer = SummaryWriter()

    # optimizer = optim.Adam(AE3.parameters(), lr=5e-4)

    # # Set the number of epochs to for training
    # epochs = 20

    # for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
    #     # Train on data
    #     train_loss = train(EUVP_data, AE3, optimizer, criterion, device)

    #     # Write metrics to Tensorboard
    #     writer.add_scalars("Loss", {'Train': train_loss}, epoch)






 