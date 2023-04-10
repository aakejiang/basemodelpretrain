# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 19:20:24 2019

@author: ASUS
"""

from __future__ import absolute_import
import math

import torch
import torch.nn as nn
from torch.autograd import Variable


__all__ = ['vggc']

defaultcfg = {
    9  : [128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    #16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    16 : [61, 53, 'M', 118, 111, 'M', 247, 207, 254, 'M', 496, 459, 512, 'M', 274, 467, 280, 'M'],  
    #19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], 
    19 : [64, 63, 'M', 113, 88, 'M', 225, 125, 109, 56, 'M', 266, 184, 61, 449, 'M', 235, 356, 336, 226, 'M'],
}

class vggc(nn.Module):
    def __init__(self, dataset='cifar10', depth=16, init_weights=True, cfg=None):
        super(vggc, self).__init__()
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
        self.classifier = nn.Linear(280, num_classes)
        
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
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

if __name__ == '__main__':
    net = vggc()
    x = Variable(torch.FloatTensor(16, 3, 32, 32))
    out = net(x)
    print(out.data.shape)