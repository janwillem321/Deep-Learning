import PIL.Image
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn
import torch
from torchinfo import summary
from torchvision.utils import save_image
from tqdm import tqdm

from gen import Generator, train
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

import os

from src.data_loader import PairedImageDataset, try_gpu

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class PairedImageDataset(Dataset):
    def __init__(self, rootA, rootB, transform=None):
        """
        Args:
            rootA (string): Directory with all the images in trainA.
            rootB (string): Directory with all the images in trainB.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.rootA = rootA
        self.rootB = rootB
        self.transform = transform

        # Assuming filenames in both folders are the same and in order
        self.filenames = sorted(os.listdir(rootA))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_nameA = os.path.join(self.rootA, self.filenames[idx])
        imageA = Image.open(img_nameA).convert('RGB')

        img_nameB = os.path.join(self.rootB, self.filenames[idx])
        imageB = Image.open(img_nameB).convert('RGB')

        if self.transform:
            imageA = self.transform(imageA)
            imageB = self.transform(imageB)

        return imageA, imageB


def try_gpu():
    """
    If GPU is available, return torch.device as cuda:0; else return torch.device
    as cpu.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # test
        dim_in_encoder = [3, 64, 128, 256, 512, 512, 512, 512, 512, 512]
        dim_in_decoder = [512, 512, 512, 512, 512, 256, 128, 64, 3]

        self.layers = nn.ModuleList()

        ####################################################################################
        # encoder
        ####################################################################################

        # layer 1
        self.layers.append(
            nn.Conv2d(
                in_channels=dim_in_encoder[0],
                out_channels=dim_in_encoder[1],
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            )
        )

        self.layers.append(nn.LeakyReLU(0.2))

        # the rest of the layers
        for in_out in range(1, (len(dim_in_encoder) - 1)):
            self.layers.append(
                nn.Conv2d(
                    in_channels=dim_in_encoder[in_out],
                    out_channels=dim_in_encoder[(in_out + 1)],
                    kernel_size=(4, 4),
                    stride=2,
                    padding=1,
                )
            )

            self.layers.append(nn.BatchNorm2d(dim_in_encoder[in_out + 1]))

            self.layers.append(nn.LeakyReLU(0.2))

        print("layer", len(self.layers))

        ####################################################################################
        # decoder
        ####################################################################################

        # layers 1 to 3 with dropout
        for in_out in range(0, 3):
            self.layers.append(
                nn.ConvTranspose2d(
                    in_channels=dim_in_decoder[in_out],
                    out_channels=dim_in_decoder[(in_out + 1)],
                    kernel_size=(4, 4),
                    stride=2,
                    padding=1,
                )
            )

            self.layers.append(nn.BatchNorm2d(dim_in_decoder[in_out + 1]))
            self.layers.append(nn.Dropout2d(0.5))
            self.layers.append(nn.LeakyReLU(0.2))

        # the rest of the layers without dropout
        for in_out in range(3, (len(dim_in_decoder) - 1)):
            self.layers.append(
                nn.ConvTranspose2d(
                    in_channels=dim_in_decoder[in_out],
                    out_channels=dim_in_decoder[(in_out + 1)],
                    kernel_size=(4, 4),
                    stride=2,
                    padding=1,
                )
            )

            self.layers.append(nn.BatchNorm2d(dim_in_decoder[in_out + 1]))
            self.layers.append(nn.Dropout2d(0.5))  # todo()
            self.layers.append(nn.LeakyReLU(0.2))

        print("layer", len(self.layers))

    def forward(self, x):
        encoder_out = []

        # forward on first layer
        x = self.forward_2_lay(x, 0)
        encoder_out.append(x)
        # print("first pass", x.shape)

        # forward encoder and save layers
        for i in range(0, 6):  # 1 - 8
            x = self.forward_3_lay(x, i, 2)
            encoder_out.append(x)
            # print("second loop", x.shape)

        # Do layer 512x1x1
        x = self.forward_3_lay(x, 6, 2)
        # Do layer 512x2x2
        x = self.forward_4_lay(x, 0, 26)

        # Do forward on decoder
        for i in range(0, 7):
            # print("encoder", encoder_out[6 - i].shape,"x", x.shape)
            x = self.forward_4_lay(torch.add(x, encoder_out[6 - i]), i, 30)
            # print("decoder for loop", x.shape)

        x = torch.tanh(x)
        x = x * 255

        return x

    def forward_2_lay(self, x, layer_num):

        for i in range(0, 2):
            x = self.layers[i + 2 * layer_num](x)

        return x

    def forward_3_lay(self, x, layer_count, start):

        for i in range(0, 3):
            x = self.layers[i + 3 * layer_count + start](x)

        return x

    def forward_4_lay(self, x, layer_num, start):
        # print(len(self.layers))
        for i in range(0, 4):
            # print(i + 4 * layer_num)
            x = self.layers[i + 4 * layer_num + start](x)

        return x


def train(train_loader, net, optimizer, criterion, device='cpu'):
    """
    Trains network for one epoch in batches.

    Args:
        train_loader: Data loader for training set.
        net: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        criterion: Loss function (e.g. cross-entropy loss).
        device: whether the network runs on cpu or gpu
    """

    avg_loss = 0

    # iterate through batches
    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, good_images = data

        # convert the inputs to run on GPU if set
        inputs, good_images = inputs.to(device), good_images.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        loss = criterion(outputs, good_images) # compare output with labels
        loss.backward()
        optimizer.step()

        # keep track of loss and accuracy
        avg_loss += loss

    return avg_loss / len(train_loader)



def main():
    # Define your transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Initialize the dataset
    paired_dataset = PairedImageDataset(rootA=r'../EUVP/Paired/underwater_dark/trainA',
                                        rootB=r'../EUVP/Paired/underwater_dark/trainB',
                                        transform=transform)

    # Initialize DataLoader
    EUVP_data = DataLoader(paired_dataset, batch_size=180, shuffle=False, num_workers=16)

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
    epochs = 0

    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
        # Train on data
        train_loss = train(EUVP_data, model, optimizer, criterion, device)

        # Write metrics to Tensorboard
        writer.add_scalars("Loss", {'Train': train_loss}, epoch)

    # make a folder
    count = 0
    image_counter = 10
    path = 'res_test'
    path_loop = None
    # Initialize DataLoader
    EUVP_data = DataLoader(paired_dataset, batch_size=5, shuffle=False, num_workers=16)

    print("start printing")
    # Store somep images
    for i, data in enumerate(EUVP_data):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        y_out = model.forward(inputs)

        for x in range(len(inputs) - 1):
            path_loop = f"{path}/test{count}"
            if not (os.path.exists(path_loop) and os.path.isdir(path_loop)):
                os.makedirs(path_loop)

            name = (f'{path_loop}/image_in{count}.jpg')
            save_image(inputs[x], name)
            name = (f'{path_loop}/image_out{count}.jpg')
            save_image(y_out[x], name)
            name = (f'{path_loop}/image_truth{count}.jpg')
            save_image(labels[x], name)
            count += 1

            if (count == image_counter):
                break
        break


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
    summary(model, (3, in_channels, s_img, s_img), device='cpu')  # (in_channels, height, width)


if __name__ == "__main__":
    main()


def train(train_loader, net, optimizer, criterion, device='cpu'):
    """
    Trains network for one epoch in batches.

    Args:
        train_loader: Data loader for training set.
        net: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        criterion: Loss function (e.g. cross-entropy loss).
        device: whether the network runs on cpu or gpu
    """

    avg_loss = 0

    # iterate through batches
    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, good_images = data

        # convert the inputs to run on GPU if set
        inputs, good_images = inputs.to(device), good_images.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        loss = criterion(outputs, good_images)  # compare output with labels
        loss.backward()
        optimizer.step()

        # keep track of loss and accuracy
        avg_loss += loss

    return avg_loss / len(train_loader)
