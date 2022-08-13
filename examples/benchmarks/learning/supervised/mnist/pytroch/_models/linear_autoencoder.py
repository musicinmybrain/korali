#!/usr/bin/env python
import torch
from torch import nn
# ## 2. Define Convolutional Autoencoder

def set_weights(l):
    if isinstance(l, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.xavier_uniform_(l.weight)
        l.bias.data.fill_(0.0)

class Encoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(28 * 28 * 1, encoded_space_dim),
        )
        self.encoder_lin.apply(set_weights)
        
    def forward(self, x):
        # Flatten
        x = self.flatten(x)
        # # Apply linear layers
        x = self.encoder_lin(x)
        return x

class Decoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()

        ### Linear section
        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(encoded_space_dim, 28 * 28 * 1),
        )
        self.decoder_lin.apply(set_weights)
        ### Unflatten
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(1, 28, 28))

    def forward(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Unflatten
        x = self.unflatten(x)
        # Apply transposed convolutions
        x = torch.sigmoid(x)
        return x
