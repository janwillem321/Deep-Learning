import torch
from PIL import Image
from torchsummary import summary
from torchvision.transforms import transforms
import numpy as np
from torchvision import datasets
from gen import Encoder

def main():
    encoder = Encoder()
    print(encoder.layers)
    summary(encoder, (3, 256, 256), device='cpu')


if __name__ == "__main__":
    main()


