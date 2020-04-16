# encoding: utf-8
"""
@author: Jiahao LU
@contact: lujiahao8146@gmail.com
@file: Models.py
@time: 2020/4/10
@desc:
"""
import torch
from torch import nn
from torchvision import models, transforms
import Modules as M
import math


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class resnet50(nn.Module):
    """docstring for DeepPose"""
    def __init__(self, input_size=256, nJoints=16, modelName='resnet50', feature_extract=False):
        super(resnet50, self).__init__()
        self.nJoints = nJoints
        # self.input_size = input_size
        self.inplanes = 64
        self.block = 'BottleNeck' if (int(modelName[6:]) > 34) else 'BasicBlock'
        self.resnet = getattr(models, modelName)(pretrained=True)
        # # set_parameter_requires_grad(self.resnet, feature_extract)
        # self.resnet.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.resnet = ResNet(Bottleneck, [3, 4, 6, 3])
        # self.resnet.fc = nn.Linear(512 * 4, nJoints * 2)
        self.resnet.fc = nn.Linear(512 * (4 if self.block == 'BottleNeck' else 1), self.nJoints * 2)
        # self.resnet.fc = nn.Linear(512 * 4, self.nJoints * 2)

    def forward(self, x):
        return self.resnet(x)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=32):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# def resnet50(nJoints=16, pretrained=False, **kwargs):
#     """Constructs a ResNet-50 model.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#     model.fc = nn.Linear(512 * 4, nJoints * 2)
#     return model

class myUpsample(nn.Module):
     def __init__(self):
         super(myUpsample, self).__init__()
         pass
     def forward(self, x):
         return x[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(x.size(0), x.size(1), x.size(2)*2, x.size(3)*2)

class Hourglass(nn.Module):
    """docstring for Hourglass"""
    def __init__(self, nChannels = 256, numReductions = 4, nModules = 2, poolKernel = (2,2), poolStride = (2,2), upSampleKernel = 2):
        super(Hourglass, self).__init__()
        self.numReductions = numReductions
        self.nModules = nModules
        self.nChannels = nChannels
        self.poolKernel = poolKernel
        self.poolStride = poolStride
        self.upSampleKernel = upSampleKernel
        """
        For the skip connection, a residual module (or sequence of residuaql modules)
        """

        _skip = []
        for _ in range(self.nModules):
            _skip.append(M.Residual(self.nChannels, self.nChannels))

        self.skip = nn.Sequential(*_skip)

        """
        First pooling to go to smaller dimension then pass input through
        Residual Module or sequence of Modules then  and subsequent cases:
            either pass through Hourglass of numReductions-1
            or pass through M.Residual Module or sequence of Modules
        """

        self.mp = nn.MaxPool2d(self.poolKernel, self.poolStride)

        _afterpool = []
        for _ in range(self.nModules):
            _afterpool.append(M.Residual(self.nChannels, self.nChannels))

        self.afterpool = nn.Sequential(*_afterpool)

        if (numReductions > 1):
            self.hg = Hourglass(self.nChannels, self.numReductions-1, self.nModules, self.poolKernel, self.poolStride)
        else:
            _num1res = []
            for _ in range(self.nModules):
                _num1res.append(M.Residual(self.nChannels,self.nChannels))

            self.num1res = nn.Sequential(*_num1res)  # doesnt seem that important ?

        """
        Now another M.Residual Module or sequence of M.Residual Modules
        """

        _lowres = []
        for _ in range(self.nModules):
            _lowres.append(M.Residual(self.nChannels,self.nChannels))

        self.lowres = nn.Sequential(*_lowres)

        """
        Upsampling Layer (Can we change this??????)
        As per Newell's paper upsamping recommended
        """
        self.up = myUpsample()#nn.Upsample(scale_factor = self.upSampleKernel)


    def forward(self, x):
        out1 = x
        out1 = self.skip(out1)
        out2 = x
        out2 = self.mp(out2)
        out2 = self.afterpool(out2)
        if self.numReductions>1:
            out2 = self.hg(out2)
        else:
            out2 = self.num1res(out2)
        out2 = self.lowres(out2)
        out2 = self.up(out2)

        return out2 + out1


class StackedHourGlass(nn.Module):
    """docstring for StackedHourGlass"""
    def __init__(self, nChannels, nStack, nModules, numReductions, nJoints):
        super(StackedHourGlass, self).__init__()
        self.nChannels = nChannels
        self.nStack = nStack
        self.nModules = nModules
        self.numReductions = numReductions
        self.nJoints = nJoints

        self.start = M.BnReluConv(3, 64, kernelSize = 7, stride = 2, padding = 3)

        self.res1 = M.Residual(64, 128)
        self.mp = nn.MaxPool2d(2, 2)
        self.res2 = M.Residual(128, 128)
        self.res3 = M.Residual(128, self.nChannels)

        _hourglass, _Residual, _lin1, _chantojoints, _lin2, _jointstochan = [],[],[],[],[],[]

        for _ in range(self.nStack):
            _hourglass.append(Hourglass(self.nChannels, self.numReductions, self.nModules))
            _ResidualModules = []
            for _ in range(self.nModules):
                _ResidualModules.append(M.Residual(self.nChannels, self.nChannels))
            _ResidualModules = nn.Sequential(*_ResidualModules)
            _Residual.append(_ResidualModules)
            _lin1.append(M.BnReluConv(self.nChannels, self.nChannels))
            _chantojoints.append(nn.Conv2d(self.nChannels, self.nJoints,1))
            _lin2.append(nn.Conv2d(self.nChannels, self.nChannels,1))
            _jointstochan.append(nn.Conv2d(self.nJoints,self.nChannels,1))

        self.hourglass = nn.ModuleList(_hourglass)
        self.Residual = nn.ModuleList(_Residual)
        self.lin1 = nn.ModuleList(_lin1)
        self.chantojoints = nn.ModuleList(_chantojoints)
        self.lin2 = nn.ModuleList(_lin2)
        self.jointstochan = nn.ModuleList(_jointstochan)

    def forward(self, x):
        x = self.start(x)
        x = self.res1(x)
        x = self.mp(x)
        x = self.res2(x)
        x = self.res3(x)
        out = []

        for i in range(self.nStack):
            x1 = self.hourglass[i](x)
            x1 = self.Residual[i](x1)
            x1 = self.lin1[i](x1)
            out.append(self.chantojoints[i](x1))
            x1 = self.lin2[i](x1)
            x = x + x1 + self.jointstochan[i](out[i])

        return out