# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.)

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from pdb import set_trace as bp
affine_par = True



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
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


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()

        self.conv2d_list = nn.ModuleList([nn.Sequential(
            ASPP(2048, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )])

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=1000, num_seg_classes=19, pretrained=False, block=Bottleneck, layers=[3, 4, 23, 3]):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # , ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.fc_new = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], num_seg_classes)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if pretrained:
            model_urls = {
                'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
                'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
                'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
                'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
                'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
            }
            if layers == [3, 4, 6, 3]:
                saved_state_dict = torch.utils.model_zoo.load_url(model_urls['resnet50'])
            elif layers == [3, 4, 23, 3]:
                saved_state_dict = torch.utils.model_zoo.load_url(model_urls['resnet101'])
            new_params = self.state_dict().copy()
            for i in saved_state_dict:
                i_parts = str(i).split('.')
                if not i_parts[0] == 'fc':
                    assert '.'.join(i_parts[0:]) in new_params
                    new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
            self.load_state_dict(new_params)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, 0.01)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=(dilation//2) if dilation > 1 else dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series, num_classes):
        return block(dilation_series, padding_series, num_classes)

    def forward_fc(self, f4, task='old'):
        x = f4
        if task in ['old', 'new']:
            x = self.avgpool(x)
            x = x.reshape(x.size(0), -1)
        if task == 'old':
            x = self.fc(x)
        else:
            x = self.fc_new(x)
            x = nn.functional.interpolate(x, size=self.input_size, mode='bilinear', align_corners=True)
        return x

    def forward_partial(self, feature, stage):
        # stage: start forwarding **from** this stage (inclusive)
        if stage <= 1:
            feature = self.layer1(feature)
        if stage <= 2:
            feature = self.layer2(feature)
        if stage <= 3:
            feature = self.layer3(feature)
        if stage <= 4:
            feature = self.layer4(feature)
        return feature

    def forward_backbone(self, x, output_features=['layer4']):
        features = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if 'layer0' in output_features: features['layer0'] = f0
        x = self.maxpool(x)
        f1 = self.layer1(x)
        if 'layer1' in output_features: features['layer1'] = f1
        f2 = self.layer2(f1)
        if 'layer2' in output_features: features['layer2'] = f2
        f3 = self.layer3(f2)
        if 'layer3' in output_features: features['layer3'] = f3
        f4 = self.layer4(f3)
        if 'layer4' in output_features: features['layer4'] = f4
        if 'gap' in output_features:
            features['gap'] = self.avgpool(f4).view(f4.size(0), -1)
        return f4, features

    def forward(self, x, output_features=['layer4'], task='old'):
        '''
        task: 'old' | 'new' | 'new_seg'
        'old', 'new': classification tasks (ImageNet or Visda)
        'new_seg': segmentation head (convs)
        '''
        self.input_size = x.size()[2:]
        f4, features = self.forward_backbone(x, output_features)
        x = self.forward_fc(f4, task=task)
        return x, features
