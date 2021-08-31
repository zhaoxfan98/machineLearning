'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei 
'''
from torch._C import layout
import torch.nn as nn
from torch.nn.modules import conv
import torch.utils.model_zoo as model_zoo
import math

# 不同的vgg结构变种
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

# vgg预训练模型的下载地址
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        # 卷积部分
        self.features = features
        # 全连接部分
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    # 前向传播
    def forward(self, x):
        x = self.features(x)
        features = x.view(x.size(0), -1)
        x = self.classifier(features)
        return [x, features]

    # 对网络的一些权重进行初始化
    def _initialize_weights(self):
        for m in self.modules():
            # 使用normal进行卷积层初始化
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # 偏置初始化为0
                if m.bias is not None:
                    m.bias.data.zero_()
            # 批归一化层权重初始化为1
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # 全连接层权重初始化
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
# 循环产生卷积层
def make_layers(cfg, batch_norm=False):
    # 根据配置表，返回模型层列表
    layers = []
    in_channels = 3     # 输入3通道图像
    # 遍历配置列表
    for v in cfg:
        # M 代表Maxpooling 添加池化层
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:   # 添加卷积层
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:      # 卷积——>批归一化（可选）——>ReLU激活
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            # 通道数方面，下一层输入即为本层输出
            in_channels = v
    return nn.Sequential(*layers)
# 数字代表通道数  M代表最大池化操作
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg11(**kwargs):
    """Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg=['A']), **kwargs)
    return model