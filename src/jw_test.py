import torch
from torchinfo import summary

from gen import Generator, train
from torch.utils.data import  DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim

import os

from src.data_loader import PairedImageDataset, try_gpu

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def main():
    # show_images()

    # Define your transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Initialize the dataset
    paired_dataset = PairedImageDataset(rootA=r'../EUVP/Paired/underwater_dark/trainB',
                                        rootB=r'../EUVP/Paired/underwater_dark/trainB',
                                        transform=transform)

    # Initialize DataLoader
    EUVP_data = DataLoader(paired_dataset, batch_size=10, shuffle=False, num_workers=4)

    # Fetch a single batch
    images_a, images_b = next(iter(EUVP_data))

    # initialize model
    model = Generator()

    # Create a writer to write to Tensorboard
    writer = SummaryWriter()

    # Create instance of Autoencoder
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

    print("Start training procedur")
    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
        # Train on data
        train_loss = train(EUVP_data, model, optimizer, criterion, device)

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

def generator_test():
    # test auo-encoder
    n_samples, in_channels, s_img, latent_dims = 3, 3, 256, 512

    # generate random sample
    x = torch.randn((n_samples, in_channels, s_img, s_img))
    print(x.shape)

    # initialize model
    model = Generator()
    x_hat = model.forward(x)

    print('shape output encoder', x_hat.shape)

    # summary of auto-encoder
    summary(model, (3 ,in_channels, s_img, s_img), device='cpu')  # (in_channels, height, width)

if __name__ == "__main__":
    main()
