# encoding: utf-8
"""
@author: Jiahao LU
@contact: lujiahao8146@gmail.com
@file: Models.py
@time: 2020/4/10
@desc:
"""
from torch import nn
from torchvision import models, transforms


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class ResNet(nn.Module):
    """docstring for DeepPose"""
    def __init__(self, input_size, nJoints=16, modelName='resnet50', feature_extract=False):
        super(ResNet, self).__init__()
        self.nJoints = nJoints
        self.input_size = input_size
        self.inplanes = 64
        self.block = 'BottleNeck' if (int(modelName[6:]) > 34) else 'BasicBlock'
        self.resnet = getattr(models, modelName)(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.resnet.fc = nn.Linear(512 * (4 if self.block == 'BottleNeck' else 1), self.nJoints * 2)
        set_parameter_requires_grad(self.resnet, feature_extract)

    def forward(self, x):
        return self.resnet(x)