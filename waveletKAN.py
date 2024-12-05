import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn
import numpy as np

class NaiveWaveletKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, initial_gridsize, addbias=True):
        super(NaiveWaveletKANLayer, self).__init__()
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        # Learnable gridsize parameter
        self.gridsize_param = nn.Parameter(torch.tensor(initial_gridsize, dtype=torch.float32))

        # Wavelet coefficients as a learnable parameter with Xavier initialization
        self.waveletcoeffs = nn.Parameter(torch.empty(2, outdim, inputdim, initial_gridsize))
        nn.init.xavier_uniform_(self.waveletcoeffs)

        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        gridsize = torch.clamp(self.gridsize_param, min=1).round().int()
        xshp = x.shape
        outshape = xshp[:-1] + (self.outdim,)
        x = torch.reshape(x, (-1, self.inputdim))

        # Create a range of scales and translations for the wavelet
        scales = torch.linspace(1, gridsize, gridsize, device=x.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        translations = torch.linspace(0, 1, gridsize, device=x.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # Morlet wavelet calculations
        xrshp = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))
        u = (xrshp - translations) * scales
        real = torch.cos(np.pi*u) * torch.exp(-u**2 /2.)
        imag = torch.sin(np.pi*u) * torch.exp(-u**2 /2.)

        # Apply wavelet coefficients to the wavelet transform outputs
        y_real = torch.sum(real * self.waveletcoeffs[0:1, :, :, :gridsize], (-2, -1))
        y_imag = torch.sum(imag * self.waveletcoeffs[1:2, :, :, :gridsize], (-2, -1))
        y = y_real + y_imag

        if self.addbias:
            y += self.bias
        y = torch.reshape(y, outshape)
        return y
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.waveletcoeffs.abs().mean(-1)  # (2, o, i, G) -> (2, o, i)
        regularization_loss_activation = l1_fake.sum()  
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

class WaveletKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
    ):
        super(WaveletKAN, self).__init__()

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                NaiveWaveletKANLayer(
                    in_features, 
                    out_features, 
                    grid_size, 
                    addbias=True, 
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )