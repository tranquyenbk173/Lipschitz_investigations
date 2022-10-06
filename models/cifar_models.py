# ResNet20 #-----------------------------------------------------------------
import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, reg='none', p=0.25):
        super().__init__()
        self.reg = reg
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.do1 = nn.Dropout(min(p, 1))
        self.do2 = nn.Dropout(min(p, 1))

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or inplanes != (planes * self.expansion):
            if reg == 'bn':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(inplanes, planes * self.expansion, 1, stride=stride,
                              bias=False),
                    nn.BatchNorm2d(planes * self.expansion)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(inplanes, planes * self.expansion, 1, stride=stride,
                              bias=False)
                )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, inputs):
        if self.reg == 'bn':
            H = self.conv1(inputs)
            H = self.bn1(H)
            H = F.relu(H)
    
            H = self.conv2(H)
            H = self.bn2(H)
    
            H += self.shortcut(inputs)
            outputs = F.relu(H)
        
        elif self.reg == 'dropout':
            H = self.conv1(inputs)
            H = F.relu(H)
            H = self.do1(H)
    
            H = self.conv2(H)
    
            H += self.shortcut(inputs)
            outputs = self.do2(F.relu(H))
        else:
            H = self.conv1(inputs)
            H = F.relu(H)
            H = self.conv2(H)
    
            H += self.shortcut(inputs)
            outputs = F.relu(H)

        return outputs


class ResNet(nn.Module):

    def __init__(self, Block, layers, filters, num_classes=10, inplanes=None, reg='none', p=0.25):
        self.inplanes = inplanes or filters[0]
        super().__init__()
        self.reg = reg
        self.pre_act = 'Pre' in Block.__name__

        self.conv1 = nn.Conv2d(3, self.inplanes, 3, padding=1, bias=False)
        if not self.pre_act and reg == 'bn':
            self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.num_sections = len(layers)
        for section_index, (size, planes) in enumerate(zip(layers, filters)):
            section = []
            for layer_index in range(size):
                if section_index != 0 and layer_index == 0:
                    stride = 2
                else:
                    stride = 1
                section.append(Block(self.inplanes, planes, stride=stride, reg=reg, p=p))
                self.inplanes = planes * Block.expansion
            section = nn.Sequential(*section)
            setattr(self, f'section_{section_index}', section)

        if self.pre_act and reg=='bn':
            self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.fc = nn.Linear(filters[-1] * Block.expansion, num_classes)
        self.do1 = nn.Dropout(min(p, 1))
        self.do2 = nn.Dropout(min(p, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        if self.reg == 'bn':
            H = self.conv1(inputs)
    
            if not self.pre_act:
                H = self.bn1(H)
                H = F.relu(H)
    
            for section_index in range(self.num_sections):
                H = getattr(self, f'section_{section_index}')(H)
    
            if self.pre_act:
                H = self.bn1(H)
                H = F.relu(H)
    
            H = F.avg_pool2d(H, H.size()[2:])
            H = H.view(H.size(0), -1)
            
        elif self.reg == 'dropout':
            H = self.conv1(inputs)
    
            if not self.pre_act:
                # H = self.bn1(H)
                H = F.relu(H)
                H = self.do1(H)
    
            for section_index in range(self.num_sections):
                H = getattr(self, f'section_{section_index}')(H)
    
            if self.pre_act:
                # H = self.bn1(H)
                H = F.relu(H)
                H = self.do2(H)
                
    
            H = F.avg_pool2d(H, H.size()[2:])
            
            H = H.view(H.size(0), -1)
            
        else:
            H = self.conv1(inputs)
    
            if not self.pre_act:
                # H = self.bn1(H)
                H = F.relu(H)
    
            for section_index in range(self.num_sections):
                H = getattr(self, f'section_{section_index}')(H)
    
            if self.pre_act:
                # H = self.bn1(H)
                H = F.relu(H)
    
            H = F.avg_pool2d(H, H.size()[2:])
            H = H.view(H.size(0), -1)
        
        outputs = self.fc(H)

        return outputs

def ResNet20(num_classes=10, reg = 'none', p=0.25):
    return ResNet(BasicBlock, layers=[3] * 3, filters=[16, 32, 64],
                  num_classes=num_classes, inplanes=None, reg = reg, p=p)
    

# AlexNet ----------------------------------------------------------------------
class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000, reg = 'none', p=0.25) -> None:
        super(AlexNet, self).__init__()
        # self.reg = re
        if reg == 'bn':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.BatchNorm2d(),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.BatchNorm2d(),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.BatchNorm2d(),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.classifier = nn.Sequential(
                nn.Linear(256 * 6 * 6, 4096),
                nn.BatchNorm1d(),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 4096),
                nn.BatchNorm1d(),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        elif reg == 'dropout':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Dropout(min(p, 1)),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Dropout(min(p, 1)),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Dropout(min(p, 1)),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.classifier = nn.Sequential(
                nn.Dropout(min(1, p*1.5)),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(min(1, p*1.5)),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.classifier = nn.Sequential(
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resize = transforms.Resize(256)
        x = resize(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def AlexNett(num_classes=10, reg = 'none', p=0.25):
    return AlexNet(num_classes=num_classes, reg = reg, p=p)


# LeNet5 -----------------------------------------------------------------------
class LeNet(nn.Module):
    def __init__(self, num_classes=10, reg='none', p=0.25):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        self.do1 = nn.Dropout(min(p, 1))
        self.do2 = nn.Dropout(min(p, 1))
        self.do3 = nn.Dropout(min(1, p*1.5))
        self.do4 = nn.Dropout(min(1, p*1.5))
        
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm1d(120)
        self.bn4 = nn.BatchNorm1d(84)
        
        self.reg = reg
        

    def forward(self, x):
        if self.reg == 'bn':
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.max_pool2d(x, 2)
            x = x.view(x.shape[0], -1)
            x = F.relu(self.bn3(self.fc1(x)))
            x = F.relu(self.bn4(self.fc2(x)))
            x = self.fc3(x)
        elif self.reg == 'dropout':
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = self.do1(x)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = self.do2(x)
            x = x.view(x.shape[0], -1)
            x = F.relu(self.fc1(x))
            x = self.do3(x)
            x = F.relu(self.fc2(x))
            x = self.do4(x)
            x = self.fc3(x)
        else:
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.shape[0], -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        return x

def LeNett(num_classes=10, reg = 'none', p=0.25):
    return LeNet(num_classes=num_classes, reg = reg, p=p)

# LeNett()

# 3MLP -------------------------------------------------------------------------
class MLP_3(torch.nn.Module):
    def __init__(self, num_classes=10, reg='none', p=0.25):
        super(MLP_3, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1024)
        self.fc4 = nn.Linear(1024, num_classes)
        
        self.do1 = nn.Dropout(min(p, 1))
        self.do2 = nn.Dropout(min(p, 1))
        self.do3 = nn.Dropout(min(p, 1))
        
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.reg = reg

    def forward(self, x):
        bs = x.shape[0]
        x = x.view(-1, 32*32*3)
        if self.reg == 'bn':
            
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = F.relu(self.bn3(self.fc3(x)))
            x = self.fc4(x)
            
        elif self.reg == 'dropout':
            x = F.relu(self.fc1(x))
            x = self.do1(x)
            x = F.relu(self.fc2(x))
            x = self.do2(x)
            x = F.relu(self.fc3(x))
            x = self.do3(x)
            x = self.fc4(x)
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
        return x.reshape(bs, -1)

def MLP(num_classes=10, reg = 'none', p=0.25):
  return MLP_3(num_classes=num_classes, reg = reg, p=p)
  
# net = MLP()