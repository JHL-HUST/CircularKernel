import torch
from .CircleLayer_base import CircleLayerBase
import numpy as np
from torch.nn.parameter import Parameter
from torch import nn
import random

class MixConv7x7(CircleLayerBase):
    """
        Input  [b, in_c,  h, w]
        Output [b, out_c, h, w]
        Note: Kernel_size must be 3 at present
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', version='MixConv', probs=[0, 0, 0.5]):
        super(MixConv7x7, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                              groups, bias, padding_mode, version)
        if self.kernel_size != 7 and self.kernel_size != 1:
            print("Kernel_size must be 1 or 7, %d was given" % kernel_size)
            raise NotImplemented("Kernel_size must be 1 or 7")

        if kernel_size == 7:
            #self.probs=probs
            self.weight_dim = 49
            self.dense_count = 1
            self.Ts, self.probs = self.build_transforms(probs)
            self.length_T = len(self.Ts)
            self.cnts = [0 for _ in range(self.length_T)]
        else:
            self.weight_dim = 1
            self.dense_count = 1
        self.weight = Parameter(torch.zeros(out_channels, self.in_channel_group, self.kernel_size, self.kernel_size))

    def build_transforms(self, probstr):
        Ts = []
        Ts.append(self.get_w_transform_matrix(*self.init_bilinear_weights_L0()))
        # print("dist_to_center = %d, " % 0, Ts[-1])
        Ts.append(self.get_w_transform_matrix(*self.init_bilinear_weights(np.sqrt(2) / 2)))
        # print("dist_to_center = %d, " % 0, Ts[-1])
        Ts.append(self.get_w_transform_matrix(*self.init_bilinear_weights()))
        # print(Ts[-1])
        Ts.append(self.get_w_transform_matrix(*self.init_bilinear_weights(np.sqrt(2))))
        # print(Ts[-1])

        buffer_T = []

        probs = self.parse_prob(probstr, len(Ts)-1)

        for i, T in enumerate(Ts):
            self.register_buffer("Ts_%d" % i, T)
            buffer_T.append(getattr(self, "Ts_%d" % i))

        print("Center: {:.2f}, Diamond: {:.2f}, Circle: {:.2f}, Square: {:.2f}, ".format(*probs))
        eval_T = torch.zeros_like(Ts[0])
        for p, T in zip(probs, Ts):
            eval_T = eval_T + p * T
        self.register_buffer("eval_T", eval_T)

        return buffer_T, probs


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_size = self.weight.shape
        w = self.weight

        if self.kernel_size != 1:
            if self.training:
                index = np.random.choice(self.length_T, p=self.probs)
                T = getattr(self, 'Ts_%d' % index)
                self.cnts[index] += 1
            else:
                T = self.eval_T
            w = w.view(-1, self.weight_dim)
            w = w.matmul(T)
        w = w.view(w_size[0], w_size[1], self.kernel_size, self.kernel_size)

        # if sum(self.cnts) % 200 == 0:
        #     print(self.probs, [cnt / sum(self.cnts) for cnt in self.cnts] )

        return nn.functional.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, groups=self.groups)

    def init_bilinear_weights(self, dist_to_center=1) -> [np.array, [int]]:
        """
        :return:
            bilinear_weights: weights to get bilinear interpolation value
        """
        select_x_indexes = []
        weights = []
        center = self.kernel_size // 2
        for grid_y in range(center, -(center + 1), -1):
            for grid_x in range(-center, center + 1):
                if grid_y == 0 or grid_x == 0:  # Do not need to bilinear mapping
                    select_x_indexes.append([self.coordinate_to_index(grid_x, grid_y, center)])
                    continue
                self.append_a_weight(45 * np.pi / 180, grid_x, grid_y, center, select_x_indexes, weights, dist_to_center=dist_to_center)
        weights = np.array(weights)
        return weights, select_x_indexes

    def init_bilinear_weights_L0(self) -> [np.array, [int]]:
        """
        :return:
            bilinear_weights: weights to get bilinear interpolation value
        """
        select_x_indexes = []
        weights = []
        center = self.kernel_size // 2
        for grid_y in range(center, -(center + 1), -1):
            for grid_x in range(-center, center + 1):
                if grid_y == 0 or grid_x == 0:  # Do not need to bilinear mapping
                    select_x_indexes.append([self.coordinate_to_index(grid_x, grid_y, center)])
                else:
                    select_x_indexes.append([self.coordinate_to_index(0, 0, center)])
        weights = np.array(weights)
        return weights, select_x_indexes

    def to_0_1(self, x, grid_x):
        """
        Handle edge conditions, only support kernel_size=3
        :param x:
        :param grid_x:
        :return:
        """

        if grid_x < 0:
            x = -x

        pos = x

        if abs(x - 1) < 1e-5:
            x = 1
        elif abs(x + 1) < 1e-5:
            x = -1
        elif abs(x) < 1e-5:
            raise NotImplemented("x must not be close to center")
        else:
            x = x - np.floor(x)
        if x < 0:
            x += 1
        return x, pos

    def append_a_weight(self, angle, grid_x, grid_y, center, select_x_indexes, weights, dist_to_center=1.0):
        # print("dist_to_center=%f, grid_x=%d, grid_y=%d" % (dist_to_center, grid_x, grid_y))
        radius = dist_to_center
        x, _ = self.to_0_1(radius * np.cos(angle), grid_x)
        y, _ = self.to_0_1(radius * np.sin(angle), grid_y)
        w = self.bilinear_interpolation(x, y)

        # Top left(tl) x and y
        if grid_x > abs(1e-6):
            tl_x = grid_x - 1
        else:
            tl_x = grid_x
        if grid_y < abs(1e-6):
            tl_y = grid_y + 1
        else:
            tl_y = grid_y
        select_x_indexes.append([self.coordinate_to_index(tl_x, tl_y, center),
                                 self.coordinate_to_index(tl_x + 1, tl_y, center),
                                 self.coordinate_to_index(tl_x, tl_y - 1, center),
                                 self.coordinate_to_index(tl_x + 1, tl_y - 1, center),
                                 ])
        weights.append(w)

    def parse_prob(self, probs_arg, length=3):
        probs = []
        if len(probs_arg) != length:
            raise NotImplemented("Probabilities must have length {}".format(length))
        forth = 1
        for p in probs_arg:
            p = float(p)
            probs.append(p)
            forth -= p
        probs.append(forth)
        return probs




if __name__ == '__main__':
    l = MixConv7x7(1, 1)
