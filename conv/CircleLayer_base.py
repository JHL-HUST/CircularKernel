import math
from torch import nn
from torch.autograd import Function
import torch

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.optim as optim
from torch.nn.modules.utils import _pair
from abc import abstractmethod
import logging, sys

import numpy as np


class CircleLayerBase(nn.Module):
    """
        Input  [b, in_c,  h, w]
        Output [b, out_c, h, w]
        Note: Kernel_size must be 3 at present
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', version='CircleLayer base'):
        #print("Conv2D version: %s" % version)
        super(CircleLayerBase, self).__init__()
        self.version = version
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            if kernel_size[0] != kernel_size[1]:
                raise NotImplemented("Kernel_size h must be equal to w")
            kernel_size = kernel_size[0]
        if kernel_size % 2 != 1:
            print("Kernel_size must be even, %d was given" % kernel_size)
            raise NotImplemented("Kernel_size must be even")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = _pair(kernel_size)
        self.kernel_size = kernel_size
        self.padding_size = padding

        self.padding = _pair(padding)
        self.padding_mode = padding_mode
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.groups=groups
        self.in_channel_group=self.in_channels//groups
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    def init_weights(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def to_0_1(self, x, grid_x):

        if grid_x < 0:
            x = -x

        pos = x

        x = x - np.floor(x)  #relative position to down left
        # if x < 0:
        #     x += 1
        return x, pos

    def bilinear_interpolation(self, px, py):
        """
        :param px: x distance to ld(left down) point
        :param py: y distance to ld(left down) point
        :return: 4 weights for points [lt(left top), rt, ld, rd]
        """
        return (1 - px) * (py), (px) * (py), (1 - px) * (1 - py), (px) * (1 - py)

    def coordinate_to_index(self, x, y, center):
        return x + center + self.kernel_size * (-y + center)

    def append_a_weight(self, angle, grid_x, grid_y, center, select_x_indexes, weights, dist_to_center):
        radius = np.floor(dist_to_center)   #  1

        #down left x and y to calculate bilinear_interpolation
        x, posx = self.to_0_1(radius * np.cos(angle), grid_x)
        y, posy = self.to_0_1(radius * np.sin(angle), grid_y)
        w = self.bilinear_interpolation(x, y)

        # Top left(tl) x and y
        if grid_x > 0:
            tl_x = grid_x - 1  #tl_x top  left  x
        else:
            tl_x = grid_x
        if grid_y < 0:
            tl_y = grid_y + 1
        else:
            tl_y = grid_y
        select_x_indexes.append([self.coordinate_to_index(tl_x, tl_y, center),
                                 self.coordinate_to_index(tl_x + 1, tl_y, center),
                                 self.coordinate_to_index(tl_x, tl_y - 1, center),
                                 self.coordinate_to_index(tl_x + 1, tl_y - 1, center),
                                 ])
        weights.append(w)

    @abstractmethod
    def init_bilinear_weights(self) -> [torch.Tensor, [int]]:
        pass

    def get_w_transform_matrix(self, alpha=None, select_x_indexes=None):
        if alpha is None or select_x_indexes is None:
            alpha, select_x_indexes = self.init_bilinear_weights()
        # import code
        # code.interact(local=locals())
        w_transform_matrix = []
        alpha_index = 0
        for i in range(len(select_x_indexes)):
            cur_row = [0 for ii in range(self.kernel_size * self.kernel_size)]
            if len(select_x_indexes[i]) == 1:
                cur_row[select_x_indexes[i][0]] = 1
            else:
                for index, j in enumerate(select_x_indexes[i]):
                    # print(index, j)
                    cur_row[j] = alpha[alpha_index, index]
                alpha_index += 1
            w_transform_matrix.append(cur_row)
        # import code
        # code.interact(local=locals())
        w_transform_matrix = torch.tensor(w_transform_matrix, dtype=torch.float)
        #print(w_transform_matrix.shape)
        # w_transform_matrix = w_transform_matrix.expand(self.out_channels * self.in_channels, -1, -1)
        return w_transform_matrix

    def print_w_transform_matrix(self):
        print(self.w_transform_matrix)

