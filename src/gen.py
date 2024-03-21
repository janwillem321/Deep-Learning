import torch
import torch.nn as nn
import numpy as np
from sympy import Range
from torchsummary import summary


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        #test
        dim_in = [3, 64, 128, 256, 512, 512, 512, 512, 512]

        self.layers = nn.ModuleList()

        # layer 1
        self.layers.append(
            nn.Conv2d(
                in_channels=dim_in[0],
                out_channels=dim_in[1],
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            )
        )

        self.layers.append(nn.LeakyReLU(0.2))

        # the rest of the layers
        for in_out in Range(1, (len(dim_in) - 1)):
            self.layers.append(
                nn.Conv2d(
                    in_channels=dim_in[in_out],
                    out_channels=dim_in[(in_out + 1)],
                    kernel_size=(4, 4),
                    stride=2,
                    padding=1,
                )
            )

            self.layers.append(nn.BatchNorm2d(dim_in[in_out + 1]))

            self.layers.append(nn.LeakyReLU(0.2))

        # Initialize params
        self.reset_params()

    def reset_params(self, std=1.):
        print("reset")



    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        dim_in = [512, 512 , 512, 512, 512, 256, 128, 64, 3]

        self.layers = nn.ModuleList()

        # layers 1 to 3 with dropout
        for in_out in Range(0, 3):
            self.layers.append(
                nn.ConvTranspose2d(
                    in_channels=dim_in[in_out],
                    out_channels=dim_in[(in_out + 1)],
                    kernel_size=(4, 4),
                    stride=2,
                    padding=1,
                )
            )

            self.layers.append(nn.BatchNorm2d(dim_in[in_out + 1]))
            self.layers.append(nn.Dropout2d(0.5))
            self.layers.append(nn.LeakyReLU(0.2))

        # the rest of the layers without dropout
        for in_out in Range(3, (len(dim_in) - 1)):
            self.layers.append(
                nn.ConvTranspose2d(
                    in_channels=dim_in[in_out],
                    out_channels=dim_in[(in_out + 1)],
                    kernel_size=(4, 4),
                    stride=2,
                    padding=1,
                )
            )

            self.layers.append(nn.BatchNorm2d(dim_in[in_out + 1]))
            self.layers.append(nn.LeakyReLU(0.2))

        #map back to an image with tangent or something

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x