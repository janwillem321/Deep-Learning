import torch
import torch.nn as nn
from torchinfo import summary

class Encoder(nn.Module):
    def __init__(self, latent_dims, s_img, hdim, kernel_size = (4,4), stride =2, padding = 1):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList()

        # layer 1
        self.layers.append(
            nn.Conv2d(
                in_channels=hdim[0],
                out_channels=hdim[1],
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            )
        )

        self.layers.append(nn.LeakyReLU(0.2))

        # the rest of the layers
        for in_out in range(1, (len(hdim) - 1)):
            self.layers.append(
                nn.Conv2d(
                    in_channels=hdim[in_out],
                    out_channels=hdim[(in_out + 1)],
                    kernel_size=(4, 4),
                    stride=2,
                    padding=1,
                )
            )

            self.layers.append(nn.BatchNorm2d(hdim[in_out + 1]))

            self.layers.append(nn.LeakyReLU(0.2))


    def forward(self, x):

        for layer in self.layers:
            x = layer.forward(x)
    

        return x

#decoder
class Decoder(nn.Module):
    def __init__(self, latent_dims, s_img, hdim, kernel_size = (4,4), stride =2, padding= 1):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList()

        for in_out in range(0, (len(hdim) - 1)):
            self.layers.append(
                nn.Conv2d(
                    in_channels=hdim[in_out],
                    out_channels=hdim[(in_out + 1)],
                    kernel_size=(4, 4),
                    stride=2,
                    padding=1,
                )
            )

            self.layers.append(nn.BatchNorm2d(hdim[in_out + 1]))

            self.layers.append(nn.LeakyReLU(0.2))

        # for i in range(len(hdim)-2):
        #     self.layers.append(
        #         nn.ConvTranspose2d(
        #             hdim[i],
        #             hdim[i+1],
        #             kernel_size, 
        #             stride,
        #             padding))

        #     self.layers.append(nn.BatchNorm2d(hdim[i+1]))

        #     # self.layers.append(nn.Dropout2d())

        #     self.layers.append(nn.LeakyReLU(0.2))

        # self.layers.append(nn.ConvTranspose2d(hdim[len(hdim)-2],
        #                                     hdim[len(hdim)-1], kernel_size, stride, padding))

        
    def forward(self, z):

        for layer in self.layers:
            z = layer.forward(z)

        return z

#Generator
class Generator(nn.Module):
    def __init__(self, latent_dims, s_img, hdim_e, hdim_d, padding):
        super(Generator, self).__init__()

        self.encoder = Encoder(latent_dims, s_img, hdim_e, padding)
        self.decoder = Decoder(latent_dims, s_img, hdim_d, padding)

    def forward(self, x):

        z = self.encoder(x)
        print(f"the shape of encoder is {z.shape}")
        y = self.decoder(z)

        return y
    
def generator_test():
    #test auo-encoder
    n_samples, in_channels, s_img, latent_dims, padding = 3, 3, 256, 512,1
    hdim_e = [3, 64, 128, 256, 512, 512, 512, 512, 512] #choose hidden dimension encoder
    hdim_d = [512, 512, 512, 512, 512, 256, 128, 64, 3] #choose hidden dimension encoder

    #generate random sample
    x = torch.randn((n_samples, in_channels, s_img, s_img))
    print(x.shape)

    #initialize model
    model = Generator(latent_dims, s_img, hdim_e, hdim_d, padding)
    x_hat = model(x)

    #compare input and output shape
    print('Output check:', x_hat.shape == x.shape)
    print('shape xhat', x_hat.shape)

    #summary of auto-encoder
    summary(model, (3 ,in_channels, s_img, s_img), device='cpu')  # (in_channels, height, width)


generator_test()
    

