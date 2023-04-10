# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 09:52:35 2020

@author: ASUS
"""

from __future__ import absolute_import
import math

import torch
import torch.nn as nn
from torch.autograd import Variable


__all__ = ['vggssd']

defaultcfg = {
    #16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    #16 : [61, 53, 'M', 118, 111, 'M', 247, 207, 254, 'M', 496, 459, 512, 'M', 274, 467, 280],
    #16 : [60, 37, 'M', 126, 102, 'M', 248, 146, 237, 'M', 453, 367, 512, 'M', 233, 463, 215], 
    #16 : [63, 53, 'M', 127, 83, 'M', 232, 108, 170, 'M', 341, 230, 461, 'M', 219, 377, 127], 
    16 : [50, 63, 'M', 109, 53, 'M', 152, 69, 84, 'M', 280, 82, 511, 'M', 184, 330, 67],
}                   


class vggssd(nn.Module):
    def __init__(self, dataset='cifar10', depth=16, init_weights=True, cfg=None):
        super(vggssd, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.cfg = cfg

        self.features = self.make_layers(cfg, True)

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        else:
            num_classes = 10
        self.classifier = nn.Linear(4096, num_classes)
        
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == 'C':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = x
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(67, 1024, kernel_size=3, padding=6, dilation=6)
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
