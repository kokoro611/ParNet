import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import AlexNet
from torchviz import make_dot
import torch
from torchvision.models import AlexNet

from tensorboardX import SummaryWriter

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    def __init__(self, inplace=True):
        super(Mish, self).__init__()
        inplace = True

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def get_activation(name="mish", inplace=True):
    if name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == 'mish':
        module = Mish(inplace=inplace)
    elif name == 'silu':
        module = nn.SiLU(inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> mish/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=True, act="mish"):
        super(BaseConv, self).__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Channel_attention(nn.Module):
    def __init__(self, in_channels, out_channels, strid):
        super(Channel_attention, self).__init__()
        self.Conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
            Swish(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
            )

        self.Conv_out = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=strid, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            )


    def forward(self, x_in):

        x_squeezed = F.adaptive_avg_pool2d(x_in, 1)
        x_squeezed = self.Conv_1(x_squeezed)
        x = torch.sigmoid(x_squeezed) * x_in
        x = self.Conv_out(x)
        return x


class Spatial_attention(nn.Module):
    def __init__(self, out_channels):
        super(Spatial_attention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)  # 输入两个通道，一个是maxpool 一个是avgpool的
        self.sigmoid = nn.Sigmoid()

        self.Conv_out = nn.Sequential(
            # nn.MaxPool2d(kernel_size=3, stride=strid, padding=1),
            nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 对池化完的数据cat 然后进行卷积
        x = self.sigmoid(x)
        x = self.Conv_out(x)
        return x





if __name__ == "__main__":
    x = torch.rand((1, 256, 56, 56))
    net = Spatial_attention(256)
    y = net(x)

    print(net)


    g = make_dot(y)
    g.render('espnet_model', view=False)


    with SummaryWriter(comment='resnet') as w:
        w.add_graph(net, x)
