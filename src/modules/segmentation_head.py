# -*- coding: utf-8 -*-
# @Time    : 2019/12/4 14:54
# @Author  : zhoujun
import torch
from torch import nn


class ConvHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        return self.conv(x)


class DBHead(nn.Module):
    def __init__(self, in_channels, out_channels, k=50):
        super().__init__()
        self.k = k
        self.binarize = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2), nn.Sigmoid())
        self.binarize.apply(self.weights_init)

        self.thresh = self._init_thresh(in_channels)
        self.thresh.apply(self.weights_init)

    def forward(self, x):
        # prob map / threshold map / appro binary map
        shrink_maps = self.binarize(x)
        threshold_maps = self.thresh(x)
        if self.training:
            # appro binary map
            binary_maps = self.step_function(shrink_maps, threshold_maps)
            return torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
        else:
            return torch.cat((shrink_maps, threshold_maps), dim=1)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self,
                     inner_channels,
                     serial=False,
                     smooth=False,
                     bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels,
                      inner_channels // 4,
                      3,
                      padding=1,
                      bias=bias), nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4,
                                inner_channels // 4,
                                smooth=smooth,
                                bias=bias),
            nn.BatchNorm2d(inner_channels // 4), nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4,
                                1,
                                smooth=smooth,
                                bias=bias), nn.Sigmoid())
        return self.thresh

    def _init_upsample(self,
                       in_channels,
                       out_channels,
                       smooth=False,
                       bias=False):
        if not smooth:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        inter_out_channels = out_channels
        if out_channels == 1:
            inter_out_channels = in_channels
        module_list = [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)
        ]
        if out_channels == 1:
            module_list.append(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=1,
                          padding=1,
                          bias=True))
        return nn.Sequential(module_list)

    def step_function(self, x, y):
        # 1 / a
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
