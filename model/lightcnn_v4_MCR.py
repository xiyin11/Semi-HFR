# -*- coding: utf-8 -*-
# @Author: Alfred Xiang Wu
# @Date:   2022-02-09 14:45:31
# @Breif: 
# @Last Modified by:   Alfred Xiang Wu
# @Last Modified time: 2022-02-09 14:48:34

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def branchBottleNeck(channel_in, channel_out, kernel_size):
    middle_channel = channel_out//4
    return nn.Sequential(
        nn.Conv2d(channel_in, middle_channel, kernel_size=1, stride=1),
        nn.ReLU(),
        
        nn.Conv2d(middle_channel, middle_channel, kernel_size=kernel_size, stride=kernel_size),
        nn.ReLU(),
        
        nn.Conv2d(middle_channel, channel_out, kernel_size=1, stride=1),
        nn.ReLU(),
        )

class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):    
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class resblock_v1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock_v1, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out


class network(nn.Module):
    def __init__(self, block, layers, num_classes=300,dropout_rate=[0.6,0.7,0.8,0.9],classifier=False):
        super(network, self).__init__()

        self.conv1 = mfm(3, 48, 3, 1, 1)

        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.conv2  = mfm(48, 96, 3, 1, 1)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.conv3  = mfm(96, 192, 3, 1, 1)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.conv4  = mfm(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.conv5  = mfm(128, 128, 3, 1, 1)

        self.fc = nn.Linear(8*8*128, 256)
        self.fc2_ = nn.Linear(256, num_classes, bias=False)

        self.middle_conv_1 = branchBottleNeck(96, 128, kernel_size=7)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1,1))
        self.middle_conv_2 = branchBottleNeck(192, 256, kernel_size=5)
        self.avgpool2 = nn.AdaptiveAvgPool2d((1,1))

        self.middle_conv_3 = branchBottleNeck(128, 512, kernel_size=3)
        self.avgpool3 = nn.AdaptiveAvgPool2d((1,1))
        self.middle_conv_4 = nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1)
        self.middle_fc1_1 = nn.Linear(128,256)
        self.middle_fc2_1 = nn.Linear(256,256)
        self.middle_fc3_1 = nn.Linear(512,256)

        self.middle_fc1_4 = nn.Linear(256, num_classes, bias=False)
        self.middle_fc2_3 = nn.Linear(256, num_classes, bias=False)
        self.middle_fc3_3 = nn.Linear(256, num_classes, bias=False)


        self.dropout_rate=dropout_rate

        self.classifiar=classifier
        self.classifiar_fc1 = nn.Linear(8*8*128,256)
        self.classifiar_fc2 = nn.Linear(256,2)

            
    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, label=None):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
        middle1 = self.middle_conv_1(x)
        middle2 = self.avgpool1(middle1)
        middle_x1 = torch.flatten(middle2,1)
        middle_fc1_1 = self.middle_fc1_1(middle_x1)
        middle_fc1_1 = F.dropout(middle_fc1_1,self.dropout_rate[0])
        middle_out1 = self.middle_fc1_4(middle_fc1_1)

        x = self.block2(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
        middle3 = self.middle_conv_2(x)
        middle4 = self.avgpool2(middle3)
        middle_x2 = torch.flatten(middle4,1)
        middle_fc2_1 = self.middle_fc2_1(middle_x2)
        middle_fc2_1 = F.dropout(middle_fc2_1,self.dropout_rate[1])
        middle_out2 = self.middle_fc2_3(middle_fc2_1)

        x = self.block3(x)
        x = self.conv4(x)
        middle_x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
        middle5 = self.middle_conv_3(middle_x)
        middle6 = self.avgpool3(middle5)
        middle_x3 = torch.flatten(middle6,1)
        middle_fc3_1 = self.middle_fc3_1(middle_x3)
        middle_fc3_1 = F.dropout(middle_fc3_1,self.dropout_rate[2])
        middle_out3 = self.middle_fc3_3(middle_fc3_1)

        x = self.block4(x)
        x = self.conv5(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = torch.flatten(x, 1)
        if self.classifiar:
            y = self.classifiar_fc1(x)
            y = F.dropout(y, training=self.training,p=0.5)
            domain = self.classifiar_fc2(y)
        fc = self.fc(x)
        fc = F.dropout(fc, training=self.training,p=self.dropout_rate[3])
        out = self.fc2_(fc)
        if self.classifiar:
            # return out, fc, (domain,y) ,(middle_fc1_1,middle_fc2_1,middle_fc3_1) ,(middle_out1,middle_out2,middle_out3)
            return out, domain
        else:
            return out, fc ,(middle_fc1_1,middle_fc2_1,middle_fc3_1) ,(middle_out1,middle_out2,middle_out3)
            
def LightCNN_V4(**kwargs):
    model = network(resblock_v1, [1, 2, 3, 4], **kwargs)
    return model