# yolov4 common modules
import math
import torch
import torch.nn as nn

def autopad(k, p = None):
    """ implement "same" padding (formula: p = (f - 1) / 2)
    Arguments:
        k (int or list): kernel size
        p (None): padding number

    returns:
        p (int or list): padding
    """
    # "same" padding
    if  p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):

    def __init__(self, ch_in, ch_out, k = 1, s = 1, p = None, g = 1, act = True):
        """ implement standard convolution module

        Arguments:
            ch_in (int): number of input channels
            ch_out (int): number of output channels
            k (int): kernel size
            s (int): stride
            p (int or list): padding size
            g (int): number of blocked connections from input channels to output channels
            act (bool): whether or not apply activation function
        """
        super(conv_module, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, k, s, autopad(k, p), bias = False)
        self.bn = nn.BatchNormd2d(ch_out)
        self.act = nn.LeakyReLU(0.1) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def fuseforward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, ch_in, ch_out, shortcut = True, g = 1, e = 0.5):
        """ implement standard bottleneck module

        Arguments:
            ch_in (int): number of input channels
            ch_out (int): number of output channels
            shortcut (bool): whether or not apply shortcut (identity mapping)
            g (int): number of blocked connections from input channels to output channels
            e (float): expansion/compression coefficient
        """
        super(Bottleneck, self).__init__()
        ch_ = int(ch_out * e) # number of channels for hidden layer
        self.conv1 = Conv(ch_in, ch_, 1, 1)
        self.conv2 = Conv(ch_, ch_out, 3, 1, g = g)
        self.add = shortcut and ch_in == ch_out

    def forward(self, x):
        if self.add:
            x1 = self.conv1(x)
            x += self.conv2(x1)
        else:
            x1 = self.conv1(x)
            x = self.conv2(x)
        return x
