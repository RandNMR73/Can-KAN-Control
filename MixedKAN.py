import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from efficient_kan.kan import KANLinear
from fftKAN import NaiveFourierKANLayer

class MixedKANLinear(torch.nn.Module):
    def __init__(self,
                in_features,
                out_features,
                grid_size=5, 
                spline_order=3,
                scale_noise=0.1,
                scale_base=1.0,
                scale_spline=1.0,
                enable_standalone_scale_spline=True,
                base_activation=torch.nn.SiLU,
                grid_eps=0.02,
                grid_range=[-1, 1],
                add_bias=True 
        ):
        super(MixedKANLinear, self).__init__()
        self.KAN_coeff = nn.Parameter(torch.randn(1))
        self.FFTKAN_coeff = nn.Parameter(torch.randn(1))
        self.KAN = KANLinear(
                in_features,
                out_features,
                grid_size=5, 
                spline_order=3,
                scale_noise=0.1,
                scale_base=1.0,
                scale_spline=1.0,
                enable_standalone_scale_spline=True,
                base_activation=torch.nn.SiLU,
                grid_eps=0.02,
                grid_range=[-1, 1],
        )
        self.FFTKAN = NaiveFourierKANLayer(in_features,
                                           out_features,
                                           grid_size,
                                           addbias=add_bias)
    def forward(self, x: torch.Tensor):
        return self.KAN_coeff * self.KAN(x) + self.FFTKAN_coeff * self.FFTKAN(x)

    def regularization_loss(self, regularize_activation_KAN=1.0, regularize_entropy_KAN=1.0, 
                            regularize_activation_FFTKAN=1.0, regularize_entropy_FFTKAN=1.0):
        return self.KAN.regularization_loss(regularize_activation_KAN, regularize_entropy_KAN) + self.FFTKAN.regularization_loss(regularize_activation_FFTKAN, regularize_entropy_FFTKAN)
    
class MixedKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        add_bias=True
    ):
        super(MixedKAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                MixedKANLinear(
                    in_features,
                    out_features,
                    grid_size, 
                    spline_order,
                    scale_noise,
                    scale_base,
                    scale_spline,
                    base_activation,
                    grid_eps,
                    grid_range,
                    add_bias 
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(self,  regularize_activation_KAN=1.0, regularize_entropy_KAN=1.0, 
                            regularize_activation_FFTKAN=1.0, regularize_entropy_FFTKAN=1.0):
        return sum(
            layer.regularization_loss( regularize_activation_KAN, regularize_entropy_KAN, 
                            regularize_activation_FFTKAN, regularize_entropy_FFTKAN)
            for layer in self.layers
        )