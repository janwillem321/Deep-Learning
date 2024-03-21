import torch
import torch.nn as nn
import numpy as np
from sympy import Range
from torchsummary import summary


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.layers = []

        dim_in = [3, 64, 128, 256, 512, 512, 512, 512, 512]

        # layer 1
        self.layers.append(
            nn.Conv2d(
                in_channels=dim_in[0],
                out_channels=dim_in[1],
                kernel_size=(4, 4),
                stride=2,
                padding=2,
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
                    padding=2,
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
            x = layer.forward(x)

        return x
