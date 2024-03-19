import torch
import torch.nn as nn
from torchsummary import summary

class Encoder(nn.Module):
    def __init__(self, latent_dims, s_img, hdim, kernel_size = (4,4), stride =2):
        super(Encoder, self).__init__()

        ########################################################################
        #    Create the necessary layers                                 #
        ########################################################################
        layers = []
        batchnorms = []

        layers.append(nn.ConvTranspose2d(s_img * s_img * 3,
                                             hdim[0][0] * hdim[0][1] * hdim[0][2],
                                             kernel_size, stride))
        for i in range(1,7):
            layers.append(nn.ConvTranspose2d(hdim[i][0] * hdim[i][1] * hdim[i][2],
                                             hdim[i+1][0] * hdim[i+1][1] * hdim[i+1][2],
                                             kernel_size, stride))
        for i in range(7):    
            batchnorms.append(nn.BatchNorm2d(hdim[i-1][0] * hdim[i+1][1] * hdim[i+1][2]))
        
        batchnorms.append(nn.BatchNorm2d(latent_dims))
        
        self.Leakyrelu = nn.LeakyReLU(0.2)


        # self.batchnorm = nn.BatchNorm2d()
        # self.e1        = nn.ConvTranspose2d(s_img * s_img * 3,
        #                                      hdim[0][0] * hdim[0][1] * hdim[0][2],
        #                                      kernel_size, stride)
        # self.e2        = nn.ConvTranspose2d(hdim[0][0] * hdim[0][1] * hdim[0][2],
        #                                      hdim[1][0] * hdim[1][1] * hdim[1][2],
        #                                      kernel_size, stride)
        # self.e3        =  nn.ConvTranspose2d(hdim[1][0] * hdim[1][1] * hdim[1][2],
        #                                      hdim[2][0] * hdim[2][1] * hdim[2][2],
        #                                      kernel_size, stride)
        # self.e4        =  nn.ConvTranspose2d(hdim[2][0] * hdim[2][1] * hdim[2][2],
        #                                      hdim[3][0] * hdim[3][1] * hdim[3][2],
        #                                      kernel_size, stride)
        # self.e5        =  nn.ConvTranspose2d(hdim[3][0] * hdim[3][1] * hdim[3][2],
        #                                      hdim[4][0] * hdim[4][1] * hdim[4][2],
        #                                      kernel_size, stride)
        # self.e6        =  nn.ConvTranspose2d(hdim[4][0] * hdim[4][1] * hdim[4][2],
        #                                      hdim[5][0] * hdim[5][1] * hdim[5][2],
        #                                      kernel_size, stride)
        # self.e7        =  nn.ConvTranspose2d(hdim[5][0] * hdim[5][1] * hdim[5][2],
        #                                      hdim[6][0] * hdim[6][1] * hdim[6][2],
        #                                      kernel_size, stride)
        # self.e8        =  nn.ConvTranspose2d(hdim[6][0] * hdim[6][1] * hdim[6][2],
        #                                      latent_dims,kernel_size, stride)
        
        ########################################################################
        #                                                                      #
        ########################################################################


    def forward(self, x):

        x = torch.flatten(x, start_dim=0)
        x = self.Leakyrelu(self.batchnorm(self.e1(x)))
        x = self.Leakyrelu(self.batchnorm(self.e2(x)))
        x = self.Leakyrelu(self.batchnorm(self.e3(x)))
        x = self.Leakyrelu(self.batchnorm(self.e4(x)))
        x = self.Leakyrelu(self.batchnorm(self.e5(x)))
        x = self.Leakyrelu(self.batchnorm(self.e6(x)))
        x = self.Leakyrelu(self.batchnorm(self.e7(x)))
        x = self.Leakyrelu(self.batchnorm(self.e8(x)))        

        return x

#decoder
class Decoder(nn.Module):
    def __init__(self, latent_dims, s_img, hdim, kernel_size = (4,4), stride =2):
        super(Decoder, self).__init__()

        self.Leakyrelu = nn.LeakyReLU(0.2)
        self.batchnorm = nn.BatchNorm2d()
        self.dropout   = nn.Dropout2d()
        
        self.d1        = nn.ConvTranspose2d(hdim[0][0] * hdim[0][1] * hdim[0][2],
                                             latent_dims,kernel_size, stride)
        self.d2        = nn.ConvTranspose2d(hdim[0][0] * hdim[0][1] * hdim[0][2],
                                             hdim[1][0] * hdim[1][1] * hdim[1][2],
                                             kernel_size, stride)
        self.d3        =  nn.ConvTranspose2d(hdim[1][0] * hdim[1][1] * hdim[1][2],
                                             hdim[2][0] * hdim[2][1] * hdim[2][2],
                                             kernel_size, stride)
        self.d4        =  nn.ConvTranspose2d(hdim[2][0] * hdim[2][1] * hdim[2][2],
                                             hdim[3][0] * hdim[3][1] * hdim[3][2],
                                             kernel_size, stride)
        self.d5        =  nn.ConvTranspose2d(hdim[3][0] * hdim[3][1] * hdim[3][2],
                                             hdim[4][0] * hdim[4][1] * hdim[4][2],
                                             kernel_size, stride)
        self.d6        =  nn.ConvTranspose2d(hdim[0][0] * hdim[0][1] * hdim[0][2],
                                             s_img * s_img * 3, kernel_size, stride)
        
    def forward(self, z):

        ########################################################################
        #    TODO: Apply full forward function                                 #
        #    NOTE: Please have a close look at the forward function of the     #
        #    encoder                                                           #
        ########################################################################

        z = self.Leakyrelu(self.dropout(self.batchnorm(self.d1(z))))
        z = self.Leakyrelu(self.dropout(self.batchnorm(self.d2(z))))
        z = self.Leakyrelu(self.dropout(self.batchnorm(self.d3(z))))
        z = self.Leakyrelu(self.dropout(self.batchnorm(self.d4(z))))
        z = self.Leakyrelu(self.dropout(self.batchnorm(self.d5(z))))
        z = self.Leakyrelu(self.dropout(self.batchnorm(self.d6(z))))
        z = torch.unflatten(z,dim=0, sizes=(256, 256,3))

        ########################################################################
        #                         END OF YOUR CODE                             #
        ########################################################################

        return z

#Generator
class Generator(nn.Module):
    def __init__(self, latent_dims, s_img, hdim_e, hdim_d):
        super(Generator, self).__init__()

        self.encoder = Encoder(latent_dims, s_img, hdim_e)
        self.decoder = Decoder(latent_dims, s_img, hdim_d)

    def forward(self, x):

        ########################################################################
        #    TODO: concatanate encoder and decoder                             #
        ########################################################################

        z = self.encoder(x)
        y = self.decoder(z)

        ########################################################################
        #                         END OF YOUR CODE                             #
        ########################################################################

        return y
    
def generator_test():
    #test auo-encoder
    n_samples, in_channels, s_img, latent_dims = 3, 3, 256, 512
    hdim_e = [[128, 128,64], [64,64,128], [32,32,256],[16,16,512],[8,8,512],[4,4,512],[2,2,512]] #choose hidden dimension encoder
    hdim_d = [[2,2,512], [4,4,512], [8,8,512], [16,16,512], [32,32, 256], [64,64,128], [128,128,64]] #choose hidden dimension encoder

    #generate random sample
    x = torch.randn((n_samples, in_channels, s_img, s_img))
    print(x.shape)

    #initialize model
    model = Generator(latent_dims, s_img, hdim_e, hdim_d)
    x_hat = model(x)

    #compare input and output shape
    print('Output check:', x_hat.shape == x.shape)
    print('shape xhat', x_hat.shape)

    #summary of auto-encoder
    summary(model, (in_channels, s_img, s_img), device='cpu') # (in_channels, height, width)


generator_test()
    

