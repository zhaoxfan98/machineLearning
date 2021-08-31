"""
    > 若将输入设为X，将某一有参数网络层设为H，那么以X为输入的此层的输出将为H(X)。一般的
    CNN网络会直接通过训练学习出参数函数H的表达，从而直接学习X->H(X)。而残差学习则是致力于使用多个有参网络层来学习输入、输出之间的残差即H(X) - X即学习X -> (H(X) - X) + X。
    其中X这一部分为直接的identity mapping，而H(X)->X则为有参网络层要学习的输入输出间残差。
"""
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

from typing import BinaryIO
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules import padding

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """ 3x3 convolution with padding """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    # resnet 18/34
    # 控制是否对最后一个特征图的通道做扩展
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()
        # 用来针对不同特征图的相加操作  需要self.shortcut做个长度和channel的对齐
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
    
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x)+self.shortcut(x))


class BottleNeck(nn.Module):
    # resnet 50/101/152
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
            )
    
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
    # 通道数 BasicBlock或BottleNeck  block数量 分类的数目
    def __init__(self, in_chans, block, num_block, num_classes=1000):
        super().__init__()

        self.block = block
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # resnet50:[3,4,6,3]  和imgs图片中表格对应
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    # 
    def _make_layer(self, block, out_channels, num_blocks, stride):
        # strides=[stride,1,1,1,1,1]
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2_x(self.pool(f1))
        f3 = self.conv3_x(f2)
        f4 = self.conv4_x(f3)
        f5 = self.conv5_x(f4)
        output = self.avg_pool(f5)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return f1, f2, f3, f4, f5, output

def resnet18(in_chans):
    return ResNet(in_chans, BasicBlock, [2,2,2,2])

def resnet34(in_chans):
    return ResNet(in_chans, BasicBlock, [3,4,6,3])

def resnet50(in_chans):
    return ResNet(in_chans, BottleNeck, [3, 4, 6, 3])

def resnet101(in_chans):
    return ResNet(in_chans, BottleNeck, [3, 4, 23, 3])

def resnet152(in_chans):
    return ResNet(in_chans, BottleNeck, [3, 8, 36, 3])