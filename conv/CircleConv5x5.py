import torch
from .CircleLayer_base import CircleLayerBase
import numpy as np
from torch.nn.parameter import Parameter
from torch import nn
import random

#non-perfectly symmetric structure
class CircleConv5x5(CircleLayerBase):
    """
        Input  [b, in_c,  h, w]
        Output [b, out_c, h, w]
        Note: Kernel_size must be 3 at present
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', version='CircleConv5x5'):
        super(CircleConv5x5, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                                groups, bias, padding_mode, version)
        if self.kernel_size != 5 and self.kernel_size != 1:
            print("Kernel_size must be 1 or 5, %d was given" % kernel_size)
            raise NotImplemented("Kernel_size must be 1 or 5")
        self.weight = Parameter(torch.zeros(out_channels, self.in_channel_group, self.kernel_size, self.kernel_size))
        if self.kernel_size != 1:
            w_transform_matrix = self.get_w_transform_matrix()
            self.register_buffer("w_transform_matrix", w_transform_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_size = self.weight.shape
        w = self.weight
        if self.kernel_size != 1:
            w = w.view(-1, self.kernel_size * self.kernel_size)
            w = w.matmul(self.w_transform_matrix)
        w = w.view(w_size[0], w_size[1], self.kernel_size, self.kernel_size)
        return nn.functional.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, groups=self.groups)

    def init_bilinear_weights(self) -> [torch.Tensor, [int]]:
        """
        :return:
            bilinear_weights: weights to get bilinear interpolation value
        """
        select_x_indexes = []
        weights = []
        center = self.kernel_size // 2  # 2
        for grid_y in range(center, -(center + 1), -1):
            for grid_x in range(-center, center + 1):
                if grid_y == 0 or grid_x == 0:
                    select_x_indexes.append([self.coordinate_to_index(grid_x, grid_y, center)])
                    continue
                dist_to_center = np.sqrt(np.power(grid_x, 2) + np.power(grid_y, 2))
                angle = np.arctan(np.abs(grid_y / grid_x))
                self.append_a_weight(angle, grid_x, grid_y, center, select_x_indexes, weights, dist_to_center)

        weights = np.array(weights)
        return weights, select_x_indexes

    def gt_y(self, x , w):
        w_size = w.shape
        if self.kernel_size != 1:
            w = w.view(-1, self.kernel_size * self.kernel_size)
            print(w.size(), self.w_transform_matrix.size())
            w = w.matmul(self.w_transform_matrix)
        w = w.view(w_size[0], w_size[1], self.kernel_size, self.kernel_size)
        return nn.functional.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation)

#completely symmetrical structure
class CircleConv5x5_original(CircleLayerBase):
    """
        Input  [b, in_c,  h, w]
        Output [b, out_c, h, w]
        Note: Kernel_size must be 3 at present
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', version='CircleConv5x5'):
        super(CircleConv5x5_original, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                                groups, bias, padding_mode, version)
        if self.kernel_size != 5 and self.kernel_size != 1:
            print("Kernel_size must be 1 or 5, %d was given" % kernel_size)
            raise NotImplemented("Kernel_size must be 1 or 5")
        self.weight = Parameter(torch.zeros(out_channels, self.in_channel_group, self.kernel_size, self.kernel_size))  # 必要性
        if self.kernel_size != 1:
            w_transform_matrix = self.get_w_transform_matrix()
            self.register_buffer("w_transform_matrix", w_transform_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_size = self.weight.shape
        w = self.weight
        if self.kernel_size != 1:
            w = w.view(-1, self.kernel_size * self.kernel_size)
            w = w.matmul(self.w_transform_matrix)
        w = w.view(w_size[0], w_size[1], self.kernel_size, self.kernel_size)
        return nn.functional.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, groups=self.groups)

    def init_bilinear_weights(self) -> [torch.Tensor, [int]]:
        """
        :return:
            bilinear_weights: weights to get bilinear interpolation value
        """
        pi=np.pi
        select_x_indexes = []
        weights = []
        center = self.kernel_size // 2  # 2
        for grid_y in range(center, -(center + 1), -1):
            for grid_x in range(-center, center + 1):
                if grid_y == 0 or grid_x == 0:
                    select_x_indexes.append([self.coordinate_to_index(grid_x, grid_y, center)])
                    continue
                elif np.abs(grid_y)==1 and np.abs(grid_x)==2:
                    angle = pi/8
                elif np.abs(grid_y) == 2 and np.abs(grid_x) == 1:
                    angle = 3* pi / 8
                else:
                    angle = np.arctan(np.abs(grid_y / grid_x))
                dist_to_center = np.sqrt(np.power(grid_x, 2) + np.power(grid_y, 2))

                self.append_a_weight(angle, grid_x, grid_y, center, select_x_indexes, weights, dist_to_center)

        weights = np.array(weights)
        return weights, select_x_indexes

    def gt_y(self, x , w):
        w_size = w.shape
        if self.kernel_size != 1:
            w = w.view(-1, self.kernel_size * self.kernel_size)
            print(w.size(), self.w_transform_matrix.size())
            w = w.matmul(self.w_transform_matrix)
        w = w.view(w_size[0], w_size[1], self.kernel_size, self.kernel_size)
        return nn.functional.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation)


if __name__ == '__main__':
    l = CircleConv5x5(1, 1)
    l.print_w_transform_matrix()
