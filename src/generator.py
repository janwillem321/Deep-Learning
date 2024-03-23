import torch
import torch.nn as nn
from torchinfo import summary

class Encoder(nn.Module):
    def __init__(self, latent_dims, s_img, hdim, kernel_size = (4,4), stride =2, padding = 0):
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

        for layer in self.layers:
            # print(layer)
            x = layer.forward(x)
            SkipConnections.append(x)

        return x, SkipConnections

#decoder
class Decoder(nn.Module):
    def __init__(self, latent_dims, s_img, hdim_in, hdim_out, kernel_size = (4,4), stride =2, padding= 0):
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
            self.layers.append(nn.LeakyReLU(0.2))

        # the rest of the layers without dropout
        for in_out in range(3, (len(hdim_in) - 1)):
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
            self.layers.append(nn.LeakyReLU(0.2))
            
    def forward(self, z, SkipConnections):
        
        EncoderIndex = 3/4
        SkipConnections.reverse()
        # for con in SkipConnections:
        #     print(con.shape)
        
        # for lay in self.layers:
        #     print(lay)

        for i, layer in enumerate(self.layers):
            if i % 4 == 0 and i != 0:
                j = int(EncoderIndex * i) 
                # print("index i= ", i)
                # print("index j = ",j)
                # print(f'layers shape{layer.parameters}')
                # print(f' z shape = {z.shape}')
                # print(f' x shape = {SkipConnections[j].shape}')
                # print(f'z concat x = {torch.cat((z,SkipConnections[j]), 1).shape}')
                z = layer.forward(torch.cat((z,SkipConnections[j]), 1)) 
            else:
                # print(i)
                # print(f' z shape = {z.shape}')
                # print(f' x shape = {SkipConnections[EncoderIndex * i].shape}')
                z = layer.forward(z)
        
        return z

#Generator
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
            s_img=s_img,hdim=hdim_e,
            kernel_size= kernel_size,
            padding= padding)
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


def generator_test():
    #test auo-encoder
    n_samples, in_channels, s_img, latent_dims, padding = 3, 3, 256, 512,1
    hdim_e = [3, 64, 128, 256, 512, 512, 512, 512, 512] #choose hidden dimension encoder
    hdim_d_input = [512, 1024, 1024, 1024, 1024, 512, 256, 128, 3] #choose hidden dimension decoder
    hdim_d_output = [512, 512, 512, 512, 512, 256, 128, 64, 3]
    kernel_size = (4,4)

    #generate random sample
    x = torch.randn((n_samples, in_channels, s_img, s_img))
    print(x.shape)

    #initialize model
    model = Generator(latent_dims=latent_dims,
                        s_img=s_img,
                        hdim_e=hdim_e, 
                        hdim_d_input=hdim_d_input,
                        hdim_d_output=hdim_d_output, 
                        padding=padding,
                        kernel_size=kernel_size)
    x_hat = model(x)

    #compare input and output shape
    print('Output check:', x_hat.shape == x.shape)
    print('shape xhat', x_hat.shape)

    #summary of auto-encoder
    summary(model, (3 ,in_channels, s_img, s_img), device='cpu', depth=5,)  # (in_channels, height, width)


# generator_test()
    

