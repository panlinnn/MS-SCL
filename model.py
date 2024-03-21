import os
import warnings
import numpy as np
import argparse
import math
import warnings
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils import data
import random
import cv2
import copy
from typing import Optional, List
import torch.nn.functional as F
from torch import nn, Tensor
from PIL import Image
import PIL.ImageOps
from tqdm import tqdm


class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        # print("input channel: ", input_dims)
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims)
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        # x:(4, 1, 32, 128)
        x = x.permute(0, 3, 2, 1)  # x:(4, 128, 32, 1)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)   # [4, 128, 32, 1]
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation)
            for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        #         print("before x:", x.shape)
        for conv in self.convs:
            x = conv(x)
        #         print("after x: ", x.shape)
        return x


class MAB(nn.Module):
    def __init__(self, K, d, input_dim, output_dim):
        super(MAB, self).__init__()
        D = K * d
        self.K = K
        self.d = d
        self.FC_q = FC(input_dims=input_dim, units=D, activations=F.relu)
        self.FC_k = FC(input_dims=input_dim, units=D, activations=F.relu)
        self.FC_v = FC(input_dims=input_dim, units=D, activations=F.relu)
        self.FC = FC(input_dims=D, units=output_dim, activations=F.relu)

    def forward(self, Q, K, batch_size):
        # torch.Size([4, 1, 32, 128]) torch.Size([4, 1, 1024, 128])

        query = self.FC_q(Q)  # [4, 1, 32, 128]
        key = self.FC_k(K)  # [4, 1, 1024, 128]
        value = self.FC_v(K)  # [4, 1, 1024, 128]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)  # [32, 1, 32, 16]
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)  # [32, 1, 1024, 16]
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)  # [32, 1, 1024, 16]

        attention = torch.matmul(query, key.transpose(2, 3))  # [32, 1, 32, 1024]
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        result = torch.matmul(attention, value)

        result = torch.cat(torch.split(result, batch_size, dim=0), dim=-1)  # orginal K, change to batch_size
        result = self.FC(result)  # [4, 1, 32, 128]

        return result


class BottleAttention(nn.Module):
    def __init__(self, K, d, set_dim):
        super(BottleAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.set_dim = set_dim
        self.I = nn.Parameter(torch.Tensor(1, 1, set_dim, D))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(K, d, D, D)
        self.mab1 = MAB(K, d, D, D)

    def forward(self, X):
        # x:[4, 128, 32, 32]
        batch_size = X.shape[0]
        X = X.flatten(2)  # [4, 1024, 128]
        X = X.unsqueeze(1).permute(0, 1, 3, 2)  # [4, 1, 1024, 128]

        # [batch_size, num_step, num_vertex, K * d]
        I = self.I.repeat(X.size(0), 1, 1, 1)
        H = self.mab0(I, X, batch_size)
        result = self.mab1(X, H, batch_size)  # [4, 1, 1024, 128]

        result = result.squeeze(1).permute(0, 2, 1).view(batch_size, 128, 32, -1)

        return result


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features),
                      nn.ReLU(),
                      nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features)
                      ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class LongResidualBlock7(nn.Module):
    def __init__(self, in_features):
        super(LongResidualBlock7, self).__init__()

        conv_block = [nn.Conv2d(in_features, in_features, 7, 1, 3),
                      nn.BatchNorm2d(in_features),
                      nn.ReLU(),

                      ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class LongResidualBlock9(nn.Module):
    def __init__(self, in_features):
        super(LongResidualBlock9, self).__init__()

        conv_block = [nn.Conv2d(in_features, in_features, 9, 1, 4),
                      nn.BatchNorm2d(in_features),
                      nn.ReLU(),
                      ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class MSSCL(nn.Module):
    def __init__(self, args):
        super(MSSCL, self).__init__()
        self.ext_flag = args.ext_flag
        self.map_width = args.map_width
        self.map_height = args.map_height
        self.in_channels = args.channels
        self.out_channels = args.channels

        if self.ext_flag and self.in_channels == 2:  # xian and chengdu
            self.embed_day = nn.Embedding(8, 2)  # Monday: 1, Sunday:7, ignore 0, thus use 8
            self.embed_hour = nn.Embedding(24, 3)  # hour range [0, 23], ignore 0, thus use 24
            self.embed_weather = nn.Embedding(15, 3)  # 14types: ignore 0, thus use 15

        if self.ext_flag and self.in_channels == 1:  # beijing
            self.embed_day = nn.Embedding(8, 2)  # Monday: 1, Sunday:7, ignore 0, thus use 8
            self.embed_hour = nn.Embedding(24, 3)  # hour range [0, 23], ignore 0, thus use 24
            self.embed_weather = nn.Embedding(18, 3)  # ignore 0, thus use 18

        self.ext2lr2 = nn.Sequential(
            nn.Linear(2, 128),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.map_width * self.map_height),
            nn.ReLU(inplace=True)
        )
        self.ext2lr3 = nn.Sequential(
            nn.Linear(3, 128),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.map_width * self.map_height),
            nn.ReLU(inplace=True)
        )
        self.ext2lr4 = nn.Sequential(
            nn.Linear(4, 128),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.map_width * self.map_height),
            nn.ReLU(inplace=True)
        )

        if self.ext_flag:
            conv1_in = self.in_channels + 4
        else:
            conv1_in = self.in_channels
        conv3_in = args.base_channels

        # input conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(conv1_in, args.base_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        res_blocks = []
        for _ in range(args.resnum):
            res_blocks.append(ResidualBlock(args.base_channels))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.down_pool = nn.AvgPool2d(2)

        long_trans = []
        for _ in range(args.attnum):
            long_trans.append(BottleAttention(16, 8, args.point))
        self.long_trans = nn.Sequential(*long_trans)
        self.long_scale9 = LongResidualBlock9(args.base_channels)
        self.long_scale7 = LongResidualBlock7(args.base_channels)

        # final conv
        self.conv4 = nn.Conv2d(args.base_channels, self.out_channels, 1)

        self.conv_trans = nn.Sequential(
            nn.Conv2d(args.base_channels, args.base_channels, 3, 1, 1),
            nn.BatchNorm2d(args.base_channels),
            nn.ReLU(inplace=True))
        self.loss = NTXentLoss(args.batch_size, 0.05, True)

    def forward(self, cmap, ext, roadmap):
        # camp:[4, 2, 64, 64]   ext:[4, 5]   roadmap:[4, 1, 128, 128])
        inp = cmap

        # external factor modeling
        if self.ext_flag and self.in_channels == 2:  # XiAn and ChengDu
            ext_out1 = self.embed_day(ext[:, 0].long().view(-1, 1)).view(-1, 2)
            out1 = self.ext2lr2(ext_out1).view(-1, 1, self.map_width, self.map_height)
            ext_out2 = self.embed_hour(ext[:, 1].long().view(-1, 1)).view(-1, 3)
            out2 = self.ext2lr3(ext_out2).view(-1, 1, self.map_width, self.map_height)
            ext_out3 = self.embed_weather(ext[:, 4].long().view(-1, 1)).view(-1, 3)
            out3 = self.ext2lr3(ext_out3).view(-1, 1, self.map_width, self.map_height)
            ext_out4 = ext[:, 2:4]
            out4 = self.ext2lr2(ext_out4).view(-1, 1, self.map_width, self.map_height)
            ext_out = torch.cat([out1, out2, out3, out4], dim=1)  # [4, 4, 64, 64]
            # [4, 2+4, 64, 64]
            inp = torch.cat([cmap, ext_out], dim=1)

        if self.ext_flag and self.in_channels == 1:  # TaxiBJ-P1
            ext_out1 = self.embed_day(ext[:, 4].long().view(-1, 1)).view(-1, 2)
            out1 = self.ext2lr2(ext_out1).view(-1, 1, self.map_width, self.map_height)
            ext_out2 = self.embed_hour(ext[:, 5].long().view(-1, 1)).view(-1, 3)
            out2 = self.ext2lr3(ext_out2).view(-1, 1, self.map_width, self.map_height)
            ext_out3 = self.embed_weather(ext[:, 6].long().view(-1, 1)).view(-1, 3)
            out3 = self.ext2lr3(ext_out3).view(-1, 1, self.map_width, self.map_height)
            ext_out4 = ext[:, :4]
            out4 = self.ext2lr4(ext_out4).view(-1, 1, self.map_width, self.map_height)
            ext_out = torch.cat([out1, out2, out3, out4], dim=1)
            inp = torch.cat([cmap, ext_out], dim=1)

        # input conv
        out1 = self.conv1(inp)  # [4, 2+4, 64, 64] -> [4, 128, 64, 64]

        short_out = self.res_blocks(out1)  # [4, 128, 64, 64]

        out_pool = self.down_pool(short_out)  # avgpooling   [4, 128, 32, 32]
        long_out = self.long_trans(out_pool)  # [4, 128, 32, 32]
        long_out1 = F.interpolate(long_out, size=list(short_out.shape[-2:]))

        scale_out7 = self.long_scale7(short_out)
        scale_out9 = self.long_scale9(short_out)
        scale_out = torch.add(scale_out7, scale_out9)
        long_out = torch.add(long_out1, scale_out)

        loss, loss1, loss2 = 0, 0, 0
        #         if self.training:
        #             loss1 = self.loss(short_out, long_out)
        #             loss2 = self.loss(long_out, short_out)
        #             loss = (loss1 + loss2) / 2

        out = torch.add(long_out, short_out)

        out = self.conv_trans(out)
        out = self.conv4(out)  # [4, 2, 64, 64]

        return out, loss