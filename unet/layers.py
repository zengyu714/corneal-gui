import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, relu=nn.ReLU()):
        """
        + Instantiate modules: conv-bn-relu
        + Assign them as member variables
        """
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = relu

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class UpConcat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConcat, self).__init__()
        # Right hand side needs `Upsample`
        self.rhs_up = nn.Upsample(scale_factor=2)
        self.conv_fit = ConvBNReLU(in_channels + out_channels, out_channels)
        self.conv = nn.Sequential(ConvBNReLU(out_channels, out_channels), ConvBNReLU(out_channels, out_channels))

    def forward(self, lhs, rhs):
        rhs = self.rhs_up(rhs)
        rhs = make_same(lhs, rhs)
        cat = torch.cat((lhs, rhs), dim=1)
        return self.conv(self.conv_fit(cat))


def make_same(good, evil):
    """
    good / evil could be 1-d, 2-d or 3-d Tensor, i.e., [batch_size, channels, (depth,) (height,) width]
    Implemented by tensor.narrow
    """
    # Make evil bigger
    g, e = good.size(), evil.size()
    ndim = len(e) - 2
    pad = int(max(np.subtract(g, e)))
    if pad > 0:
        pad = tuple([pad] * ndim * 2)
        evil = F.pad(evil, pad, mode='replicate')

    # evil > good:
    e = evil.size()  # update
    for i in range(2, len(e)):
        diff = (e[i] - g[i]) // 2
        evil = evil.narrow(i, diff, g[i])
    return evil
