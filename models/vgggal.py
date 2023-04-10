import torch.nn as nn

defaultcfg = {
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGGAL(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGGGAL, self).__init__()
        cfg = defaultcfg
        if vgg_name == 'vgg16':
            self.features1 = self._make_layers1(vgg_name, cfg[vgg_name][:2])
            self.features2 = self._make_layers2(vgg_name, cfg[vgg_name][2:5])
            self.features3 = self._make_layers3(vgg_name, cfg[vgg_name][5:9])
            self.features4 = self._make_layers4(vgg_name, cfg[vgg_name][9:13])
            self.features5 = self._make_layers5(vgg_name, cfg[vgg_name][13:])
        else:
            self.features1 = self._make_layers1(vgg_name, cfg[vgg_name][:2])
            self.features2 = self._make_layers2(vgg_name, cfg[vgg_name][2:5])
            self.features3 = self._make_layers3(vgg_name, cfg[vgg_name][5:10])
            self.features4 = self._make_layers4(vgg_name, cfg[vgg_name][10:15])
            self.features5 = self._make_layers5(vgg_name, cfg[vgg_name][15:])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out1 = self.features1(x)
        out2 = self.features2(out1)
        out3 = self.features3(out2)
        out4 = self.features4(out3)
        out5 = self.features5(out4)
        out = out5.view(out5.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers1(self, vgg_name, cfg):
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
        return nn.Sequential(*layers)

    def _make_layers2(self, vgg_name, cfg):
        layers = []
        if vgg_name == 'vgg16':
            in_channels = defaultcfg[vgg_name][1]
        else:
            in_channels = defaultcfg[vgg_name][1]
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def _make_layers3(self, vgg_name, cfg):
        layers = []
        if vgg_name == 'vgg16':
            in_channels = defaultcfg[vgg_name][4]
        else:
            in_channels = defaultcfg[vgg_name][4]
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def _make_layers4(self, vgg_name, cfg):
        layers = []
        if vgg_name == 'vgg16':
            in_channels = defaultcfg[vgg_name][8]
        else:
            in_channels = defaultcfg[vgg_name][9]
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def _make_layers5(self, vgg_name, cfg):
        layers = []
        if vgg_name == 'vgg16':
            in_channels = defaultcfg[vgg_name][12]
        else:
            in_channels = defaultcfg[vgg_name][14]
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
