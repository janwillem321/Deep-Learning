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

        self.layers = nn.ModuleList()
        
        # Input layer dim -- down1
        self.layers.append(nn.Conv2d(in_channels=6, out_channels=64, kernel_size=kernel_size, stride=2, padding=1))

        # Hidden to hidden convolution -- down2 and down 3
        for i in range(0, 2):
            self.layers.append(nn.Conv2d(in_channels=hdim[i],
                                             out_channels=hdim[i + 1],
                                             kernel_size=kernel_size, stride = stride, padding=1))

        # Pad with zeroes
        self.layers.append(nn.ZeroPad2d(padding=(1,1,1,1)))

        # Conv2D
        self.layers.append(nn.Conv2d(in_channels=hdim[3],
                                             out_channels=hdim[4],
                                             kernel_size=kernel_size, stride = 1))

        # Batchnorm
        self.layers.append(nn.BatchNorm2d(hdim[4]))

        # Zeropad2
        self.layers.append(nn.ZeroPad2d(padding=(1,1,1,1)))

        #Conv2D 2
        self.layers.append(nn.Conv2d(in_channels=hdim[5],
                                             out_channels=hdim[6],
                                             kernel_size=kernel_size, stride = 1))

        self.Leakyrelu = nn.LeakyReLU(0.2)
        self.Batchnorm = None
        

    def forward(self, x):

        for n_layer, layer in enumerate(self.layers):
            ## The fourth layer first has a batchnorm and then a Leakyrelu
            if n_layer != 4:
                print(n_layer, str(layer))
                x = self.Leakyrelu(layer(x))
            else:
                x = layer(x)
        return x
    
    
def Discriminator_test():
    #test auo-encoder
    n_samples, in_channels, s_img, latent_dims = 1, 6, 256, 512 # 6 for two images
    hdim = [64, 128, 256, 256, 512, 512, 1] #choose hidden dimension discriminator

    #generate random sample
    x = torch.randn((n_samples, in_channels, s_img, s_img))
    print(x.shape)

    #initialize model
    model = Discriminator(latent_dims, s_img, hdim)
    x_hat = model.forward(x)

    #compare input and output shape
    print('shape xhat', x_hat.shape)

    #summary of auto-encoder
    summary(model, (in_channels, s_img, s_img), device='cpu') # (in_channels, height, width)


Discriminator_test()