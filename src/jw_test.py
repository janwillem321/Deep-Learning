import torch
from PIL import Image
from torchinfo import summary
from torchvision.transforms import transforms
import numpy as np
from torchvision import datasets
from gen import Encoder


def main():
    encoder = Encoder()
    print(encoder.layers)
    generator_test()


def generator_test():
    # test auo-encoder
    n_samples, in_channels, s_img, latent_dims = 3, 3, 256, 512

    # generate random sample
    x = torch.randn((n_samples, in_channels, s_img, s_img))
    print(x.shape)

    # initialize model
    model = Encoder()
    x_hat = model.forward(x)

    print('shape xhat', x_hat.shape)

    # summary of auto-encoder
    summary(model, (3 ,in_channels, s_img, s_img), device='cpu')  # (in_channels, height, width)


if __name__ == "__main__":
    main()
