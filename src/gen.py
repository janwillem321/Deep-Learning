import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # test
        dim_in_encoder = [3, 64, 128, 256, 512, 512, 512, 512, 512, 512]
        dim_in_decoder = [512, 512, 512, 512, 512, 256, 128, 64, 3]

        self.layers = nn.ModuleList()

        ####################################################################################
        # encoder
        ####################################################################################

        # layer 1
        self.layers.append(
            nn.Conv2d(
                in_channels=dim_in_encoder[0],
                out_channels=dim_in_encoder[1],
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            )
        )

        self.layers.append(nn.LeakyReLU(0.2))

        # the rest of the layers
        for in_out in range(1, (len(dim_in_encoder) - 1)):
            self.layers.append(
                nn.Conv2d(
                    in_channels=dim_in_encoder[in_out],
                    out_channels=dim_in_encoder[(in_out + 1)],
                    kernel_size=(4, 4),
                    stride=2,
                    padding=1,
                )
            )

            self.layers.append(nn.BatchNorm2d(dim_in_encoder[in_out + 1]))

            self.layers.append(nn.LeakyReLU(0.2))

        print("layer", len(self.layers))

        ####################################################################################
        # decoder
        ####################################################################################

        # layers 1 to 3 with dropout
        for in_out in range(0, 3):
            self.layers.append(
                nn.ConvTranspose2d(
                    in_channels=dim_in_decoder[in_out],
                    out_channels=dim_in_decoder[(in_out + 1)],
                    kernel_size=(4, 4),
                    stride=2,
                    padding=1,
                )
            )

            self.layers.append(nn.BatchNorm2d(dim_in_decoder[in_out + 1]))
            self.layers.append(nn.Dropout2d(0.5))
            self.layers.append(nn.LeakyReLU(0.2))

        # the rest of the layers without dropout
        for in_out in range(3, (len(dim_in_decoder) - 1)):
            self.layers.append(
                nn.ConvTranspose2d(
                    in_channels=dim_in_decoder[in_out],
                    out_channels=dim_in_decoder[(in_out + 1)],
                    kernel_size=(4, 4),
                    stride=2,
                    padding=1,
                )
            )

            self.layers.append(nn.BatchNorm2d(dim_in_decoder[in_out + 1]))
            self.layers.append(nn.Dropout2d(0.5)) #todo()
            self.layers.append(nn.LeakyReLU(0.2))

        print("layer", len(self.layers))

    def forward(self, x):
        encoder_out = []

        #forward on first layer
        x = self.forward_2_lay(x, 0)
        encoder_out.append(x)
        # print("first pass", x.shape)

        #forward encoder and save layers
        for i in range(0, 6): # 1 - 8
            x = self.forward_3_lay(x, i, 2)
            encoder_out.append(x)
            # print("second loop", x.shape)

        #Do layer 512x1x1
        x = self.forward_3_lay(x, 6, 2)
        #Do layer 512x2x2
        x = self.forward_4_lay(x, 0, 26)

        #Do forward on decoder
        for i in range(0, 7):
            # print("encoder", encoder_out[6 - i].shape,"x", x.shape)
            x = self.forward_4_lay(torch.add(x, encoder_out[6 - i]), i, 30)
            # print("decoder for loop", x.shape)

        return x

    def forward_2_lay(self, x, layer_num):

        for i in range(0, 2):
            x = self.layers[i + 2 * layer_num](x)

        return x

    def forward_3_lay(self, x, layer_count, start):

        for i in range(0, 3):
            x = self.layers[i + 3 * layer_count + start](x)

        return x

    def forward_4_lay(self, x, layer_num, start):
        # print(len(self.layers))
        for i in range(0, 4):
            # print(i + 4 * layer_num)
            x = self.layers[i + 4 * layer_num + start](x)

        return x


def train(train_loader, encoder, decoder, optimizer, criterion, device='cpu'):
    """
    Trains network for one epoch in batches.

    Args:
        train_loader: Data loader for training set.
        net: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        criterion: Loss function (e.g. cross-entropy loss).
        device: whether the network runs on cpu or gpu
    """

    avg_loss = 0

    # iterate through batches
    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]

        # convert the inputs to run on GPU if set
        inputs = inputs.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = decoder(encoder(inputs))

        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        # keep track of loss and accuracy
        avg_loss += loss

    return avg_loss / len(train_loader)
