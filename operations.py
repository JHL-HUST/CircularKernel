import torch
import functools
import torch.nn as nn
from conv.CircleConv3x3 import CircleConv3x3
from conv.CircleConv5x5 import CircleConv5x5
from conv.CircleConv7x7 import CircleConv7x7


OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'sep_circle_conv_3x3' : lambda C, stride, affine: SepCircleConv(C, C, 3, stride, 1, affine=affine),
  'sep_circle_conv_5x5' : lambda C, stride, affine: SepCircleConv(C, C, 5, stride, 2, affine=affine),
  'sep_circle_conv_7x7' : lambda C, stride, affine: SepCircleConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'dil_conv_7x7' : lambda C, stride, affine: DilConv(C, C, 7, stride, 6, 2, affine=affine),
  'dil_circle_conv_3x3' : lambda C, stride, affine: DilCircleConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_circle_conv_5x5' : lambda C, stride, affine: DilCircleConv(C, C, 5, stride, 4, 2, affine=affine),
  'dil_circle_conv_7x7' : lambda C, stride, affine: DilCircleConv(C, C, 7, stride, 6, 2, affine=affine),
  "conv 1x1" : lambda C, stride, affine: ReLUConvBN(C, C, kernel_size=1, stride=stride, padding=0, affine=affine),
  "conv 3x3" : lambda C, stride, affine: ReLUConvBN(C, C, kernel_size=3, stride=stride, padding=1, affine=affine),
  "conv_3x1_1x3" : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,3), stride=(1, stride), padding=(0, 1), bias=False),
    nn.Conv2d(C, C, (3,1), stride=(stride, 1), padding=(1, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
  'max_pool_5x5' : lambda C, stride, affine: nn.MaxPool2d(5, stride=stride, padding=2),
  'max_pool_7x7' : lambda C, stride, affine: nn.MaxPool2d(7, stride=stride, padding=3)
}





class Sep_nx1_Conv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(Sep_nx1_Conv, self).__init__()
    pad=padding
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, (1, kernel_size), stride=(1, stride), padding=(0, pad), groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, (kernel_size, 1), stride=(stride, 1), padding=(pad, 0), groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)






class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_out, C_out, kernel_size=kernel_size, stride=1, padding=padding, groups=C_out, bias=False),
      nn.Conv2d(C_out, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)

class DilCircleConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilCircleConv, self).__init__()
    if kernel_size != 1 and kernel_size != 3 and kernel_size != 5 and kernel_size!=7:
      print("Kernel_size must be 1, 3, 5 or 7, %d was given" % kernel_size)
      raise NotImplemented("Kernel_size must be 1, 3, 5 or 7")
    if kernel_size == 3:
      conv2d = CircleConv3x3
    elif kernel_size == 5:
      conv2d = CircleConv5x5
    elif kernel_size == 7:
      conv2d = CircleConv7x7
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in,
                bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
    )

  def forward(self, x):
    return self.op(x)

class SepCircleConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepCircleConv, self).__init__()
    if kernel_size != 1 and kernel_size != 3 and kernel_size != 5 and kernel_size!=7:
      print("Kernel_size must be 1, 3, 5 or 7, %d was given" % kernel_size)
      raise NotImplemented("Kernel_size must be 1,5 or 7")
    if kernel_size == 3:
      conv2d = CircleConv3x3
    elif kernel_size == 5:
      conv2d = CircleConv5x5
    elif kernel_size == 7:
      conv2d = CircleConv7x7
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      nn.ReLU(inplace=False),
      conv2d(C_out, C_out, kernel_size=kernel_size, stride=1, padding=padding, groups=C_out, bias=False),
      nn.Conv2d(C_out, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
    )

  def forward(self, x):
    return self.op(x)




class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x




class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride
  def forward(self, x):
    n, c, h, w = x.size()
    h //= self.stride
    w //= self.stride
    if x.is_cuda:
      with torch.cuda.device(x.get_device()):
        padding = torch.cuda.FloatTensor(n, c, h, w).fill_(0)
    else:
      padding = torch.FloatTensor(n, c, h, w).fill_(0)
    return padding


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    # assert C_out % 2 != 0
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

