import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out
    

class ResList(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale=1):
        super(ResList, self).__init__()
        self.num_res_blocks = num_res_blocks

        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats))

        self.conv_tail = conv3x3(n_feats, n_feats)

    def forward(self, x):
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x
    

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2, 1, 7, 1, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): 
        x_compress = self.compress(x)
        x_out = F.relu(self.spatial(x_compress))
        scale = self.sigmoid(x_out)
        return x * scale


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class Res_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1, SA=False, CA=False):
        super(Res_Attention, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.channel_attention = CALayer(out_channels, reduction=16)
        self.spatial_attention = SpatialGate()
        self.CA = CA
        self.SA = SA

    def forward(self, x):
        x1 = x
        out = self.relu(self.conv1(x))

        if self.SA:
            out = self.spatial_attention(out)

        if self.CA:
            out = self.channel_attention(out)

        out = self.relu(self.conv2(out))

        out = out * self.res_scale + x1
        return out


class Res_Attention_Conf(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1, SA=False, CA=False):
        super(Res_Attention_Conf, self).__init__()

        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.channel_attention = CALayer(out_channels, reduction=16)
        self.spatial_attention = SpatialGate()
        self.CA = CA
        self.SA = SA

    def forward(self, x):


        x1 = x
        out = self.relu(self.conv1(x))

        if self.SA:
            out = self.spatial_attention(out)
            out = out 

        if self.CA:
            out = self.channel_attention(out)

        out = self.relu(self.conv2(out))

        out = out * self.res_scale + x1
        return out


class Res_CA_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,  res_scale=1,  CA=False):
        super(Res_CA_Block, self).__init__()


        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.channel_attention = CALayer(out_channels, reduction=16)

        self.CA = CA


    def forward(self, x):
        x1 = x
        out = self.relu(self.conv1(x))
        if self.CA:
            out = self.channel_attention(out)

        out = self.relu(self.conv2(out))

        out = out * self.res_scale + x1
        return out


class Res_CA_List(nn.Module):
    def __init__(self, num_res_blocks, n_feats, res_scale=1):
        super(Res_CA_List, self).__init__()
        self.num_res_blocks = num_res_blocks

        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(Res_CA_Block(in_channels=n_feats, out_channels=n_feats))

        self.conv_tail = conv3x3(n_feats, n_feats)

    def forward(self, x):
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x


class PixelShufflePack(nn.Module):
    """ Pixel Shuffle upsample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)

    def forward(self, x):

        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)

        self.weight.requires_grad = False
        self.bias.requires_grad = False



