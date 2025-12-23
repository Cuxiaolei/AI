# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def init_layer(L):
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class Conv2d_fw(nn.Conv2d):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias=False):
        super(Conv2d_fw, self).__init__(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.weight.fast = None


    def forward(self, x):
        if self.weight.fast is not None:
            out = F.conv2d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
        else:
            out = super(Conv2d_fw, self).forward(x)

        return out

class BatchNorm2d_fw(nn.BatchNorm2d):
    def __init__(self, out_channel, momentum=0.1, track_running_stats=True):
        super(BatchNorm2d_fw, self).__init__(out_channel, momentum=momentum, track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(out_channel))
            self.register_buffer('running_var', torch.zeros(out_channel))
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias

        out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
        return out

class FeatureWiseTransformation2d_fw(nn.BatchNorm2d):
    Feature_Trans = False
    def __init__(self, out_channel, momentum=0.1, track_running_stats=True):
        super(FeatureWiseTransformation2d_fw, self).__init__(out_channel, momentum=momentum, track_running_stats=track_running_stats)
        self.out_channel = out_channel
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.out_channel))
            self.register_buffer('running_var', torch.zeros(self.out_channel))
            self.running_mean.zero_()
            self.running_var.fill_(1)
        if self.Feature_Trans:
            self.gamma = nn.Parameter(torch.ones(1, self.out_channel, 1, 1)*0.3)
            self.beta = nn.Parameter(torch.ones(1, self.out_channel, 1, 1)*0.5)

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
        if self.Feature_Trans and self.training:
            gamma = (1 + torch.randn(1, self.out_channel, 1, 1, dtype=self.gamma.dtype, device=self.gamma.device) * F.softplus(self.gamma, beta=100)).expand_as(out)
            beta = (torch.rand(1, self.out_channel, 1, 1, dtype=self.beta.dtype, device=self.beta.device) * F.softplus(self.beta, beta=100)).expand_as(out)
            out = gamma*out + beta
        return out

class ResidualBlock2d(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(ResidualBlock2d, self).__init__()
        self.C1 = Conv2d_fw(in_channel=in_channel, out_channel=out_channel, kernel_size=3, stride=stride, padding=1)
        self.B1 = BatchNorm2d_fw(out_channel=out_channel)
        self.C2 = Conv2d_fw(in_channel=out_channel, out_channel=out_channel, kernel_size=3, stride=1, padding=1)
        self.B2 = FeatureWiseTransformation2d_fw(out_channel=out_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.param_layer = [self.C1, self.B1, self.C2, self.B2]

        if in_channel != out_channel:
            self.shortcut = Conv2d_fw(in_channel=in_channel, out_channel=out_channel, kernel_size=1, stride=2, padding=0)
            self.BNshortcut = FeatureWiseTransformation2d_fw(out_channel=out_channel)
            self.channel = False
            self.param_layer.append(self.shortcut)
            self.param_layer.append(self.BNshortcut)
        else:
            self.channel = True

        for layer in self.param_layer:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.B1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.B2(out)
        if self.channel:
            short_out = x
        else:
            short_out = self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out

class Resnet2d(nn.Module):
    def __init__(self, blockdim, Feature_trans=False):
        super(Resnet2d, self).__init__()
        if Feature_trans:
            FeatureWiseTransformation2d_fw.Feature_Trans = True
        C1 = Conv2d_fw(in_channel=3, out_channel=blockdim[0], kernel_size=7, stride=2, padding=3)
        B1 = BatchNorm2d_fw(out_channel=blockdim[0])
        relu1 = nn.ReLU(inplace=True)
        maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        init_layer(C1)
        init_layer(B1)
        block1 = nn.Sequential(C1, B1, relu1, maxpool1)
        Resblock1 = ResidualBlock2d(in_channel=blockdim[0], out_channel=blockdim[1], stride=1)
        Resblock2 = ResidualBlock2d(in_channel=blockdim[1], out_channel=blockdim[2], stride=2)
        Resblock3 = ResidualBlock2d(in_channel=blockdim[2], out_channel=blockdim[3], stride=2)
        Resblock4 = ResidualBlock2d(in_channel=blockdim[3], out_channel=blockdim[4], stride=2)
        avgpool1 = nn.AvgPool2d(7)
        self.trunk = nn.Sequential(block1, Resblock1, Resblock2, Resblock3, Resblock4, avgpool1)

    def forward(self, x):
        out = self.trunk(x)
        out = out.view(out.size(0), -1)

        return out

class Conv1d_fw(nn.Conv1d):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias=False):
        super(Conv1d_fw, self).__init__(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.weight.fast = None

    def forward(self, x):
        if self.weight.fast is not None:
            out = F.conv1d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
        else:
            out = super(Conv1d_fw, self).forward(x)

        return out

class BatchNorm1d_fw(nn.BatchNorm1d):
    def __init__(self, out_channel, momentum=0.1, track_running_stats=True):
        super(BatchNorm1d_fw, self).__init__(out_channel, momentum=momentum, track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(out_channel))
            self.register_buffer('running_var', torch.zeros(out_channel))
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias

        out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
        return out

class FeatureWiseTransformation1d_fw(nn.BatchNorm1d):
    Feature_Trans = False
    def __init__(self, out_channel, momentum=0.1, track_running_stats=True):
        super(FeatureWiseTransformation1d_fw, self).__init__(out_channel, momentum=momentum, track_running_stats=track_running_stats)
        self.out_channel = out_channel
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.out_channel))
            self.register_buffer('running_var', torch.zeros(self.out_channel))
            self.running_mean.zero_()
            self.running_var.fill_(1)
        if self.Feature_Trans:
            self.gamma = nn.Parameter(torch.ones(1, self.out_channel, 1)*0.3)
            self.beta = nn.Parameter(torch.ones(1, self.out_channel, 1)*0.5)

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
        if self.Feature_Trans and self.training:
            gamma = (1 + torch.randn(1, self.out_channel, 1, dtype=self.gamma.dtype, device=self.gamma.device) * F.softplus(self.gamma, beta=100)).expand_as(out)
            beta = (torch.rand(1, self.out_channel, 1, dtype=self.beta.dtype, device=self.beta.device) * F.softplus(self.beta, beta=100)).expand_as(out)
            out = gamma*out + beta

        return out

class ResidualBlock1d(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(ResidualBlock1d, self).__init__()
        self.C1 = Conv1d_fw(in_channel=in_channel, out_channel=out_channel, kernel_size=3, stride=stride, padding=1)
        self.B1 = BatchNorm1d_fw(out_channel=out_channel)
        self.C2 = Conv1d_fw(in_channel=out_channel, out_channel=out_channel, kernel_size=3, stride=1, padding=1)
        self.B2 = FeatureWiseTransformation1d_fw(out_channel=out_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.param_layer = [self.C1, self.B1, self.C2, self.B2]

        if in_channel != out_channel:
            self.shortcut = Conv1d_fw(in_channel=in_channel, out_channel=out_channel, kernel_size=1, stride=2, padding=0)
            self.BNshortcut = FeatureWiseTransformation1d_fw(out_channel=out_channel)
            self.channel = False
            self.param_layer.append(self.shortcut)
            self.param_layer.append(self.BNshortcut)
        else:
            self.channel =True
        for layer in self.param_layer:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.B1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.B2(out)
        if self.channel:
            short_out = x
        else:
            short_out = self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out

class Resnet1d(nn.Module):
    def __init__(self, blockdim, Feature_trans=False):
        super(Resnet1d, self).__init__()
        if Feature_trans:
            FeatureWiseTransformation1d_fw.Feature_Trans = True

        C1 = Conv1d_fw(in_channel=1, out_channel=blockdim[0], kernel_size=7, stride=2, padding=3)
        B1 = BatchNorm1d_fw(out_channel=blockdim[0])
        relu1 = nn.ReLU(inplace=True)
        maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        init_layer(C1)
        init_layer(B1)
        block1 = nn.Sequential(C1, B1, relu1, maxpool1)
        Resblock1 = ResidualBlock1d(in_channel=blockdim[0], out_channel=blockdim[1], stride=1)
        Resblock2 = ResidualBlock1d(in_channel=blockdim[1], out_channel=blockdim[2], stride=2)
        Resblock3 = ResidualBlock1d(in_channel=blockdim[2], out_channel=blockdim[3], stride=2)
        Resblock4 = ResidualBlock1d(in_channel=blockdim[3], out_channel=blockdim[4], stride=2)
        adapool = nn.AdaptiveAvgPool1d(1)
        self.trunk = nn.Sequential(block1, Resblock1, Resblock2, Resblock3, Resblock4, adapool)

    def forward(self, x):
        out = self.trunk(x)
        out = out.view(x.size(0), -1)

        return out




