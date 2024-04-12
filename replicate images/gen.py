import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from IPython.display import clear_output


class Encoder(nn.Module):
    def __init__(self, latent_dims, s_img, hdim, kernel_size=(4, 4), stride=2, padding=1):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList()

        # layer 1
        self.layers.append(
            nn.Conv2d(
                in_channels=hdim[0],
                out_channels=hdim[1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )

        self.layers.append(nn.LeakyReLU(0.2))
        self.layers.append(nn.Identity())

        # the rest of the layers
        for in_out in range(1, (len(hdim) - 1)):
            self.layers.append(
                nn.Conv2d(
                    in_channels=hdim[in_out],
                    out_channels=hdim[(in_out + 1)],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )

            self.layers.append(nn.BatchNorm2d(hdim[in_out + 1]))

            self.layers.append(nn.LeakyReLU(0.2))

    def forward(self, x):

        SkipConnections = []

        for i, layer in enumerate(self.layers):
            # print(layer)
            x = layer.forward(x)
            if (i + 1) % 3 == 0 and i < 21:
                #                 print("Layer e: and shape ",i, x.shape)
                SkipConnections.append(x)
        return x, SkipConnections


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



class Decoder(nn.Module):
    def __init__(self, latent_dims, s_img, hdim_in, hdim_out, kernel_size=(4, 4), stride=2, padding=1):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList()

        # layers 1 to 3 with dropout
        for in_out in range(0, 3):
            self.layers.append(
                nn.ConvTranspose2d(
                    in_channels=hdim_in[in_out],
                    out_channels=hdim_out[(in_out + 1)],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )

            self.layers.append(nn.BatchNorm2d(hdim_out[in_out + 1]))
            self.layers.append(nn.Dropout2d(0.5))
            self.layers.append(nn.ReLU())

        # the rest of the layers
        for in_out in range(3, (len(hdim_in) - 2)):
            self.layers.append(
                nn.ConvTranspose2d(
                    in_channels=hdim_in[in_out],
                    out_channels=hdim_out[(in_out + 1)],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )

            self.layers.append(nn.BatchNorm2d(hdim_out[in_out + 1]))
            self.layers.append(nn.Identity())
            #             self.layers.append(nn.Dropout2d(0.5))
            self.layers.append(nn.ReLU())

        self.layers.append(
            nn.ConvTranspose2d(
                in_channels=hdim_in[len(hdim_in) - 2],
                out_channels=hdim_out[len(hdim_in) - 1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )
        self.layers.append(nn.Tanh())

    def forward(self, z, SkipConnections):

        EncoderIndex = 3 / 4
        SkipConnections.reverse()
        j = 0

        for i, layer in enumerate(self.layers):
            if (i + 1) % 4 == 0 and i < 28:
                #                 print("skipconn layer j:", j)
                #                 j = int(EncoderIndex * i)
                #                 z = layer.forward(torch.add(z, SkipConnections[j]))
                #                 print("layer z.shape", z.shape)
                #                 print("layer other.shape", SkipConnections[j].shape)
                z = layer.forward(torch.cat((z, SkipConnections[j]), 1))
                j += 1

            else:
                z = layer.forward(z)

        #         z = torch.tanh(z)
        #         z = z * 255
        return z


# Generator
class Generator(nn.Module):
    def __init__(self,
                 latent_dims,
                 s_img,
                 hdim_e,
                 hdim_d_input,
                 hdim_d_output,
                 kernel_size,
                 padding):
        super(Generator, self).__init__()

        self.encoder = Encoder(
            latent_dims=latent_dims,
            s_img=s_img, hdim=hdim_e,
            kernel_size=kernel_size,
            padding=padding)
        self.decoder = Decoder(
            latent_dims=latent_dims,
            s_img=s_img,
            hdim_in=hdim_d_input,
            hdim_out=hdim_d_output,
            kernel_size=kernel_size,
            padding=padding)

    def forward(self, x):
        z, skipConnections = self.encoder(x)
        # print(f"the shape of encoder is {z.shape}")
        y = self.decoder(z, skipConnections)

        return y

if __name__ == "__main__":
    num_workers = 4
    pin_memory = True
    batch_size = 64
    n_samples, in_channels, s_img, latent_dims, padding = 1, 3, 256, 512, 1
    hdim_e = [3, 64, 128, 256, 512, 512, 512, 512, 512]  # choose hidden dimension encoder
    hdim_d_output = [512, 512, 512, 512, 512, 256, 128, 64, 3]
    hdim_d_input = [512, 1024, 1024, 1024, 1024, 512, 256, 128, 3]  # choose hidden dimension decoder
    #     hdim_d_input = hdim_d_output
    in_channels_dis = 6  # 6 for two images
    hdim_dis = [64, 128, 256, 256, 512, 512, 1]  # choose hidden dimension discriminator
    output_shape = (n_samples, 1, 30, 30)

    kernel_size = (4, 4)

    # Define your transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    paired_dataset_dark = PairedImageDataset(rootA=r'/home/huts/Documents/tudelft/Q3/deep learning/Deep-Learning/replicate images/paper_images/trainA',
                                             rootB=r'/home/huts/Documents/tudelft/Q3/deep learning/Deep-Learning/replicate images/paper_images/trainB',
                                             transform=transform)

    EUVP_data = DataLoader(paired_dataset_dark, batch_size=100, shuffle=False, num_workers=4)

    model_ = Generator(latent_dims=latent_dims,
                        s_img=s_img,
                        hdim_e=hdim_e,
                        hdim_d_input=hdim_d_input,
                        hdim_d_output=hdim_d_output,
                        padding=padding,
                        kernel_size=kernel_size)

    device = torch.device("cpu")
    model_.load_state_dict(torch.load('model_dict.pth', map_location=torch.device('cpu')))
    # model_ = torch.load('model.pth', map_location=device)


    count = 0
    image_counter = 10
    path = 'test'
    path_loop = None

    print("start printing")
    # Store somep images
    for i, data in enumerate(EUVP_data):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        y_out = model_.forward(inputs)
        #     y_out = torch.tanh(y_out)

        for x in range(len(inputs)):
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

    print("Printing done")