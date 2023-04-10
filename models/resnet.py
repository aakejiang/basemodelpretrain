import torch
import torch.nn as nn
import torch.nn.functional as F


conv_num_cfg = {
    'resnet18' : 8,
    'resnet34' : 16,
    'resnet50' : 16,
    'resnet101' : 33,
    'resnet152' : 50 
}

num_blocks = {
    'resnet18' : [2,2,2,2],
    'resnet34' : [3,4,6,3],
    'resnet50' : [3,4,6,3],
    'resnet101' : [3,4,23,3],
    'resnet152' : [3,8,36,3] 
}

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, food, index, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, food[index], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(food[index])
        self.conv2 = nn.Conv2d(food[index], planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, food, index, stride=1):
        super(Bottleneck, self).__init__()
        pr_channels = food[index]
        self.conv1 = nn.Conv2d(in_planes, pr_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(pr_channels)
        self.conv2 = nn.Conv2d(pr_channels , pr_channels,kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(pr_channels)
        self.conv3 = nn.Conv2d(pr_channels, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, food=None):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.food = food
        self.current_conv = 0

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,  bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layers(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layers(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layers(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layers(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.Sequential(nn.AvgPool2d(7))
        self.fc = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layers(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes,
                self.food, self.current_conv, stride))
            self.current_conv +=1
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


original_food_cfg = {
    #'resnet18': [64, 64, 128, 128, 256, 256, 512, 512],
    'resnet18': [1, 1, 1, 1, 1, 1, 1, 1],
    #'resnet34': [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512, 512],
    'resnet34': [5, 24, 37, 74, 50, 71, 32, 89, 107, 177, 132, 120, 106, 62, 330, 124],
    #'resnet50': [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512, 512],
    'resnet50': [16, 28, 28, 78, 33, 78, 49, 140, 85, 122, 114, 94, 85, 86, 42, 231],
    'resnet101': [64, 64, 64, 128, 128, 128, 128,
                  256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
                  512, 512, 512],
    'resnet152': [64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128,
                  256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
                  512, 512, 512],                  
    }

def resnet(cfg, food= None, num_classes = 1000):
    if food == None:
        food = original_food_cfg[cfg]
    if cfg == 'resnet18':
        return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, food=food)
    elif cfg == 'resnet34':
        return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes, food=food)
    elif cfg == 'resnet50':
        return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes, food=food)
    elif cfg == 'resnet101':
        return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes, food=food)
    elif cfg == 'resnet152':
        return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes, food=food)


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3], num_classes =1000, food = None)

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

'''
def test():
    food = [5,6,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6]
    model = resnet('resnet50', food)
    #print(model)

test()
'''


