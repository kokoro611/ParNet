import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
from basic_block import BaseConv
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


class BaseConv_no_act(nn.Module):
    """A Conv2d -> Batchnorm -> mish/leaky relu block"""
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False):
        super(BaseConv_no_act, self).__init__()
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

    def forward(self, x):
        return self.bn(self.conv(x))

class SSE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SSE,self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((None,None))
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.bn(x)
        x_res = x
        x = self.global_avg_pool(x)
        x = self.sigmoid(x)

        out = torch.mul(x_res, x)
        return out


class Par_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Par_block,self).__init__()

        self.Conv_1 = BaseConv_no_act(in_channels=in_channels, out_channels=out_channels,  ksize=1, stride=1)
        self.Conv_2 = BaseConv_no_act(in_channels=in_channels, out_channels=out_channels,  ksize=3, stride=1)

        self.SSE = SSE(in_channels=in_channels, out_channels=out_channels)

        self.act = get_activation('silu', inplace=True)
    def forward(self, x):
        x_1 = self.Conv_1(x)
        x_3 = self.Conv_2(x)
        x_se = self.SSE(x)

        out = x_1 + x_3 + x_se
        out = self.act(out)
        return out

class Down_sampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down_sampling,self).__init__()

        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.Conv_1_1 = BaseConv_no_act(in_channels=in_channels, out_channels=out_channels, ksize=1, stride=1)

        self.Conv_3 = BaseConv_no_act(in_channels=in_channels, out_channels=out_channels, ksize=3, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((None,None))
        self.Conv_1_2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2)
        self.act_1 = nn.Sigmoid()

        self.act_2 = get_activation('silu', inplace=True)
    def forward(self, x):
        x_1 = self.avg_pool(x)
        x_1 = self.Conv_1_1(x_1)

        x_3 = self.Conv_3(x)

        x_13 = x_1 + x_3

        x_se = self.global_avg_pool(x)
        x_se = self.Conv_1_2(x_se)
        x_se = self.act_1(x_se)

        out = torch.mul(x_13, x_se)
        out = self.act_2(out)
        return out

class Fusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Fusion,self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.Conv_1_1 = BaseConv_no_act(in_channels=in_channels*2, out_channels=out_channels, ksize=1, stride=1, groups=2)

        self.Conv_3 = BaseConv_no_act(in_channels=in_channels*2, out_channels=out_channels, ksize=3, stride=2, groups=2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((None,None))
        self.Conv_1_2 = nn.Conv2d(in_channels=in_channels*2, out_channels=out_channels, kernel_size=1, stride=2)
        self.act_1 = nn.Sigmoid()
        self.act_2 = get_activation('silu', inplace=True)

    def forward(self, x1, x2):
        x1 = self.bn1(x1)
        x2 = self.bn2(x2)
        x = torch.cat((x1,x2), dim=1)

        idx = torch.randperm(x.nelement())
        x = x.view(-1)[idx].view(x.size())

        x_1 = self.avg_pool(x)
        x_1 = self.Conv_1_1(x_1)

        x_3 = self.Conv_3(x)

        x_13 = x_1 + x_3

        x_se = self.global_avg_pool(x)
        x_se = self.Conv_1_2(x_se)
        x_se = self.act_1(x_se)

        out = torch.mul(x_13, x_se)
        out = self.act_2(out)
        return out


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class Parnet(nn.Module):
    def __init__(self,
                 cat='xl',
                 ):
        super(Parnet,self).__init__()
        deep_list = [64, 200, 400, 800, 3200]
        if cat == 'xl':
            deep_list = [64, 200, 400, 800, 3200]
        if cat == 's':
            deep_list = [64, 96, 192, 384, 1280]
        if cat == 'm':
            deep_list = [64, 128, 256, 512, 2048]
        if cat == 'l':
            deep_list = [64, 160, 320, 640, 2560]

        self.focus = Focus(in_channels=3, out_channels=deep_list[0], ksize=3)

        self.downsamping_2 = Down_sampling(in_channels=deep_list[0], out_channels=deep_list[1])
        self.downsamping_3 = Down_sampling(in_channels=deep_list[1], out_channels=deep_list[2])
        self.downsamping_4 = Down_sampling(in_channels=deep_list[2], out_channels=deep_list[3])

        self.Stream_1 = nn.Sequential(
                                    Par_block(in_channels=deep_list[1],out_channels=deep_list[1]),
                                    Par_block(in_channels=deep_list[1], out_channels=deep_list[1]),
                                    Par_block(in_channels=deep_list[1], out_channels=deep_list[1]),
                                    Par_block(in_channels=deep_list[1], out_channels=deep_list[1])
        )
        self.S1_downsamping = Down_sampling(in_channels=deep_list[1],out_channels=deep_list[2])

        self.Stream_2 = nn.Sequential(
                                    Par_block(in_channels=deep_list[2], out_channels=deep_list[2]),
                                    Par_block(in_channels=deep_list[2], out_channels=deep_list[2]),
                                    Par_block(in_channels=deep_list[2], out_channels=deep_list[2]),
                                    Par_block(in_channels=deep_list[2], out_channels=deep_list[2]),
                                    Par_block(in_channels=deep_list[2], out_channels=deep_list[2])
        )
        self.S2_Fusion = Fusion(in_channels=deep_list[2], out_channels=deep_list[3])

        self.Stream_3 = nn.Sequential(
                                    Par_block(in_channels=deep_list[3], out_channels=deep_list[3]),
                                    Par_block(in_channels=deep_list[3], out_channels=deep_list[3]),
                                    Par_block(in_channels=deep_list[3], out_channels=deep_list[3]),
                                    Par_block(in_channels=deep_list[3], out_channels=deep_list[3]),
                                    Par_block(in_channels=deep_list[3], out_channels=deep_list[3])
        )
        self.S3_Fusion = Fusion(in_channels=deep_list[3], out_channels=deep_list[3])

        self.conv_out1 = BaseConv(in_channels=deep_list[2],out_channels=256,ksize=1,stride=1,act='silu')
        self.conv_out2 = BaseConv(in_channels=deep_list[3],out_channels=512,ksize=1,stride=1,act='silu')

        self.SPP = SPPBottleneck(in_channels=deep_list[3], out_channels=deep_list[3])
        self.conv_out3 = BaseConv(in_channels=deep_list[3],out_channels=1024,ksize=1,stride=1,act='silu')


    def forward(self,x):
        x = self.focus(x)
        x = self.downsamping_2(x)

        x_s1 = self.S1_downsamping(self.Stream_1(x))

        x = self.downsamping_3(x)
        x_s2 = self.Stream_2(x)
        x_s12 = self.S2_Fusion(x_s1, x_s2)


        x = self.downsamping_4(x)
        x_s3 = self.Stream_3(x)
        x_s123 = self.S3_Fusion(x_s12,x_s3)

        x_out1 = self.conv_out1(x_s1)
        x_out2 = self.conv_out2(x_s12)

        x_out3 = self.SPP(x_s123)
        x_out3 = self.conv_out3(x_out3)
        return x_out1, x_out2, x_out3



if __name__ == "__main__":
    x = torch.rand((1, 3, 640, 640))

    net = Parnet(cat='l')
    y = net(x)

    print(net)


    g = make_dot(y)
    g.render('espnet_model', view=False)


    with SummaryWriter(comment='resnet') as w:
        w.add_graph(net, (x))



