import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1,
                                      stride=stride, bias=False)
        else:
            self.shortcut = None

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x

        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet18k(nn.Module):

    def __init__(self, k: int = 64, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.in_planes = k

        self.conv1 = nn.Conv2d(in_channels, k, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(PreActBlock, k,   num_blocks=2, stride=1)
        self.layer2 = self._make_layer(PreActBlock, 2*k, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(PreActBlock, 4*k, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(PreActBlock, 8*k, num_blocks=2, stride=2)

        self.bn_last = nn.BatchNorm2d(8 * k)
        self.fc = nn.Linear(8 * k, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, stride=s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)                
        out = self.layer1(out)             
        out = self.layer2(out)               
        out = self.layer3(out)              
        out = self.layer4(out)               
        out = F.relu(self.bn_last(out))
        out = F.adaptive_avg_pool2d(out, 1)  
        out = out.view(out.size(0), -1)      
        out = self.fc(out)                   
        return out