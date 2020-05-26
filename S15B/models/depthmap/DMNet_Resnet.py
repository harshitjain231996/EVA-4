import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, inC, outC, addFlag=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inC, out_channels=outC, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outC)
        self.conv2 = nn.Conv2d(in_channels=outC, out_channels=outC, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outC)

        self.shortcut = nn.Sequential()
        if inC != outC:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=inC, out_channels=outC, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(outC)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        return out

class MaxBlock(nn.Module):
    def __init__(self, inC, outC):
        super(MaxBlock,self).__init__()

        self.conv = nn.Sequential(           
            nn.Conv2d(in_channels=inC, out_channels=outC, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2), 
            nn.BatchNorm2d(outC),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class ResnetDepthNet(nn.Module):
    def __init__(self):
        super(ResnetDepthNet, self).__init__()
        self.in_planes = 16
        self.in_planes_mask = 32
        self.in_planes_depth = 32

        # PrepLayer
        self.conv1 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(ResBlock, 32)
        self.layer2 = self._make_layer(ResBlock, 48)
        self.layer3 = self._make_layer(ResBlock, 64)
        
        self.combine = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), bias=False)
        )

        self.layer_m1 = self._make_layer_mask(ResBlock, 64)
        self.layer_m2 = self._make_layer_mask(ResBlock, 64)
        
        self.layer_d1 = self._make_layer_depth(ResBlock, 64)
        self.layer_d2 = self._make_layer_depth(ResBlock, 64)
        
        # Last layer for mask prediction
        self.last_layer_mask = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1), bias=False)
        )

        # Last layer for depth prediction
        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1), bias=False)
        )

    def _make_layer(self, r_block, planes):
        layers = []
        layers.append(r_block(self.in_planes, planes))
        self.in_planes = planes
        return nn.Sequential(*layers)
        

    def _make_layer_mask(self, r_block, planes):
        layers = []
        layers.append(r_block(self.in_planes_mask, planes))
        self.in_planes_mask = planes
        return nn.Sequential(*layers)

    def _make_layer_depth(self, r_block, planes):
        layers = []
        layers.append(r_block(self.in_planes_depth, planes))
        self.in_planes_depth = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # Here input is of 6 channels, mask and depth channels: 3+3=6

        out = F.relu(self.bn1(self.conv1(x))) # prep layer

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.combine(out)

        m1 = self.layer_m1(out)
        m2 = self.layer_m2(m1)
        m3 = self.last_layer_mask(m2) 
        
        d1 = self.layer_d1(out)
        d2 = self.layer_d2(d1)
        d3 = self.last_layer_depth(d2) 
        
        return m3, d3
