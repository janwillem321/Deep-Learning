import torch
from PIL import Image
from torchinfo import summary
from torchvision.transforms import transforms
import numpy as np
from torchvision import datasets
from gen import Encoder
from gen import Decoder


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
    model2 = Decoder()
    x_hat = model.forward(x)
    x_hat2 = model2.forward(x_hat)

    print('shape output encoder', x_hat.shape, 'output decoder', x_hat2.shape)

    # summary of auto-encoder
    summary(model, (3 ,in_channels, s_img, s_img), device='cpu')  # (in_channels, height, width)
    summary(model2, (3, 512, 1, 1), device='cpu')

if __name__ == "__main__":
    main()
