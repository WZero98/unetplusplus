import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from typing import Optional, Tuple, Literal
from collections import OrderedDict
from copy import deepcopy


class ConvNormNonlinearDropout(Module):
    """
    CNN base blocks in UNet series models
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            conv_ly=nn.Conv2d, conv_kwargs: Optional[dict] = None,
            norm_ly=nn.BatchNorm2d, norm_kwargs: Optional[dict] = None,
            nonlinear_ly=nn.ReLU, nonlinear_kwargs: Optional[dict] = None,
            dropout: Optional[float] = None
    ) -> None:
        super().__init__()
        # save two optional parameters
        self.norm_ly = norm_ly
        self.dropout = dropout

        # default conv_kwargs
        if conv_kwargs is None:
            conv_kwargs = {
                'kernel_size': 3,
                'stride': 1,
                'padding': 1
            }

        # default norm_kwargs
        if norm_kwargs is None:
            norm_kwargs = {
                'eps': 1e-05,
                'momentum': 0.1
            }

        # default nonlinear_kwargs
        if nonlinear_kwargs is None and nonlinear_ly == nn.ReLU:
            nonlinear_kwargs = {
                'inplace': True
            }

        self.conv = conv_ly(in_channels, out_channels, **conv_kwargs)
        if self.norm_ly is not None:
            self.norm = norm_ly(out_channels, **norm_kwargs)
        if nonlinear_kwargs is not None:
            self.nonlinear = nonlinear_ly(**nonlinear_kwargs)
        else:
            self.nonlinear = nonlinear_ly()
        if self.dropout is not None:
            self.dropout_ly = nn.Dropout2d(p=dropout)

    def forward(self, x_input):
        x = self.conv(x_input)
        if self.norm_ly is not None:
            x = self.norm(x)
        x = self.nonlinear(x)
        if self.dropout is not None:
            x = self.dropout_ly(x)
        return x


class Upsampling(Module):
    """
    upsampline blocks
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=2,
            stride=1,
            padding=1,
            size: int | Tuple[int] | Tuple[int, int] | Tuple[int, int, int] | None = None,
            scale_factor: float | Tuple[float] | None = None,
            mode: Literal[
                'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area', 'nearest-exact'] = 'bilinear',
            align_corners: Optional[bool] = False
    ):
        """
        the parameters from torch.nn.functional.interpolate
        :param size: output spatial size.
        :param mode: algorithm used for upsampling: 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact'.
        :param align_corners:
        """
        super().__init__()
        # upsampling parameters
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

        # up conv parameters
        self.up_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x_input):
        x = nn.functional.interpolate(
            x_input,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners
        )
        return self.up_conv(x)


class StackedConvLayers(Module):
    """
    stacks base blocks in UNet Series models.
    """

    def __init__(
            self,
            in_channels: int, out_channels: int,
            num_layers: int,
            conv_ly=nn.Conv2d, conv_kwargs: Optional[dict] = None,
            nonlinear_ly=nn.ReLU, nonlinear_kwargs: Optional[dict] = None,
            norm_ly=nn.BatchNorm2d, norm_kwargs: Optional[dict] = None,
            dropout: Optional[float] = None
    ):
        """

        :param in_channels:
        :param out_channels:
        :param num_layers:
        :param conv_ly:
        :param conv_kwargs:
        :param nonlinear_ly:
        :param nonlinear_kwargs:
        :param norm_ly:
        :param norm_kwargs:
        :param dropout:
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.conv_ly = conv_ly
        self.conv_kwargs = conv_kwargs
        self.nonlinear_ly = nonlinear_ly
        self.nonlinear_kwargs = nonlinear_kwargs
        self.norm_ly = norm_ly
        self.norm_kwargs = norm_kwargs
        self.dropout = dropout

        self.stackblocks = nn.Sequential(
            OrderedDict(
                [
                    ('cnn_block1', ConvNormNonlinearDropout(in_channels, out_channels, conv_ly, conv_kwargs,
                                                            norm_ly, norm_kwargs, nonlinear_ly, nonlinear_kwargs,
                                                            dropout))
                ] +
                [
                    (f'cnn_block{i + 1}', ConvNormNonlinearDropout(out_channels, out_channels, conv_ly, conv_kwargs,
                                                                   norm_ly, norm_kwargs, nonlinear_ly, nonlinear_kwargs,
                                                                   dropout))
                    for i in range(1, num_layers)
                ]
            )
        )

    def forward(self, x_input):
        return self.stackblocks(x_input)

