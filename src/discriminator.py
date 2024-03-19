import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
class Discriminator(nn.Module):
    def __init__(self, latent_dims, s_img, hdim, kernel_size=(4, 4), stride=2):
        super(Discriminator, self).__init__()

        ########################################################################
        #    Create the necessary layers                                 #
        ########################################################################
        # self.layers = []
        batchnorms = []

        self.layers = nn.ModuleList()
        # Input layer dim -- down1
        self.layers.append(nn.Conv2d(in_channels=6, out_channels=64, kernel_size=kernel_size, stride=stride))

        # Hidden to hidden convolution -- down2 and down 3
        for i in range(0, 2):
            self.layers.append(nn.Conv2d(in_channels=hdim[i][2],
                                             out_channels=hdim[i + 1][2],
                                             kernel_size=kernel_size, stride = stride))

        # Pad with zeroes
        self.layers.append(nn.ZeroPad2d(padding=(1,1,1,1)))

        # Conv2D
        # self.layers.append(nn.Conv2d(hdim[3][0] * hdim[3][1] * hdim[3][2],
        #                                      hdim[3 + 1][0] * hdim[3 + 1][1] * hdim[3 + 1][2],
        #                                      kernel_size, stride=1))

        # # Zeropad2
        # self.layers.append(padding=(1,1,1,1))
        #
        # #Conv2D 2
        # self.layers.append(nn.Conv2d(hdim[5][0] * hdim[5][1] * hdim[5][2],
        #                                      hdim[5 + 1][0] * hdim[5 + 1][1] * hdim[5 + 1][2],
        #                                      kernel_size, stride=1))

        self.Leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):

        for layer in self.layers:
            x = self.Leakyrelu(layer(x))
        return x
def Discriminator_test():
    #test auo-encoder
    n_samples, in_channels, s_img, latent_dims = 1, 6, 256, 512 # 6 for two images
    hdim = [[128, 128,64], [64,64,128], [32,32,256],[34,34,256],[31,31,512],[33,33,512],[30,30,1]] #choose hidden dimension encoder

    #generate random sample
    x = torch.randn((in_channels, s_img, s_img))
    print(x.shape)

    #initialize model
    model = Discriminator(latent_dims, s_img, hdim)
    x_hat = model.forward(x)

    #compare input and output shape
    print('Output check:', x_hat.shape == x.shape)
    print('shape xhat', x_hat.shape)

    #summary of auto-encoder
    summary(model, (in_channels, s_img, s_img), device='cpu') # (in_channels, height, width)


Discriminator_test()