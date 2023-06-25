import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange
import util.util as util
from util.Selfpatch import Selfpatch


# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 15:47
# @Author  : ZiYe_
# @File    : undernet_include.py
# @Version : 0.1.0

###############################################################################
# Functions
###############################################################################


def get_nonlinearity_layer(activation_type='PReLU'):
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU(True)
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU(True)
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.2, True)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.uniform_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def nor_conv(x, in_channel, out_channel, ks=3, std=1, padding=1):
    # conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=ks, stride=std,
    #                  padding=padding).cuda()
    model = nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=ks, stride=std,
                  padding=padding),
        nn.BatchNorm2d(out_channel),
        nn.PReLU()).cuda()
    return model(x)


def get_residue(tensor, r_dim=1):
    max_channel = torch.max(tensor, dim=r_dim, keepdim=True)  # keepdim
    min_channel = torch.min(tensor, dim=r_dim, keepdim=True)
    res_channel = max_channel[0] - min_channel[0]
    return res_channel


def kernel_conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
              padding=padding, dilation=dilation, groups=groups, bias=False).cuda()
    return


######################################################################################
# Basic Operation
######################################################################################
class KernelConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU()):
        super(KernelConv, self).__init__()

        model = [
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=False),
            norm_layer(out_planes),
            nonlinearity,
        ]
        self.model = nn.Sequential(*model).cuda()

    def forward(self, x):
        return self.model(x)


class SKConv(nn.Module):
    def __init__(self, num_features, ratio):  # num_features = 64->128     radio = 8
        super(SKConv, self).__init__()
        self.out_channels = num_features
        self.conv_init = nn.Conv2d(num_features * 2, num_features, kernel_size=1, padding=0,
                                   stride=1).cuda()  # 256->128
        self.conv_dc = nn.Conv2d(num_features, num_features // ratio, kernel_size=1, padding=0,
                                 stride=1).cuda()  # 128->16
        self.conv_ic = nn.Conv2d(num_features // ratio, num_features * 2, kernel_size=1, padding=0,
                                 stride=1).cuda()  # 16->256
        self.act = nn.ReLU(inplace=True).cuda()
        self.avg_pool = nn.AdaptiveAvgPool2d(1).cuda()
        self.softmax = nn.Softmax(dim=1).cuda()

    def forward(self, x1, x2):  # x1 = [4, 128, 112, 112]    x2 = [4, 128, 112, 112]
        batch_size = x1.size(0)  # 4

        feat_init = torch.cat((x1, x2), dim=1)  # feat_init = [4, 256, 112, 112]
        feat_init = self.conv_init(feat_init)  # feat_init = [4, 128, 112, 112]
        fea_avg = self.avg_pool(feat_init)  # fea_avg = [4, 128, 1, 1]
        feat_ca = self.conv_dc(fea_avg)  # feat_ca = [4, 42, 1, 1]
        feat_ca = self.conv_ic(self.act(feat_ca))  # feat_ca = [4, 256, 1, 1]

        a_b = feat_ca.reshape(batch_size, 2, self.out_channels, -1)

        a_b = self.softmax(a_b)
        a_b = list(a_b.chunk(2, dim=1))  # split to a and b
        a_b = list(map(lambda x1: x1.reshape(batch_size, self.out_channels, 1, 1), a_b))  # [4, 128, 1, 1] * 2
        V1 = a_b[0] * x1
        V2 = a_b[1] * x2
        V = V1 + V2
        return V  # [4, 128, 112, 112]


###############################################################################################################
# TODO: RCP

class Convd(nn.Module):
    def __init__(self, inputchannel, outchannel, kernel_size, stride):
        super(Convd, self).__init__()
        self.relu = nn.ReLU().cuda()
        self.padding = nn.ReflectionPad2d(kernel_size // 2).cuda()
        self.conv = nn.Conv2d(inputchannel, outchannel, kernel_size, stride).cuda()
        self.ins = nn.InstanceNorm2d(outchannel, affine=True).cuda()

    def forward(self, x):
        x = self.conv(self.padding(x))
        x = self.relu(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        ).cuda()

    def forward(self, x):  # x = [4, 32, 112, 112]
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # avg_pool = [4, 32, 1, 1]  y = [4, 32]
        y = self.fc(y).view(b, c, 1, 1)  # fc = [4, 32]  y = [4, 32, 1, 1]
        return x * y.expand_as(x)  # [4, 32, 112, 112]


class RB(nn.Module):
    def __init__(self, n_feats):  # n_feats = 32
        super(RB, self).__init__()
        module_body = []
        for i in range(2):
            module_body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))
            module_body.append(nn.ReLU())
        self.module_body = nn.Sequential(*module_body).cuda()
        self.relu = nn.ReLU().cuda()
        self.se = SELayer(n_feats, 1).cuda()

    def forward(self, x):
        res = self.module_body(x)  # res = [4, 32, 112, 112]
        res = self.se(res)  # res = [4, 32, 112, 112]
        res += x
        return res  # [4, 32, 112, 112]


class RIR(nn.Module):
    def __init__(self, n_feats, n_blocks):
        super(RIR, self).__init__()
        module_body = [
            RB(n_feats) for _ in range(n_blocks)
        ]
        module_body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))
        self.module_body = nn.Sequential(*module_body).cuda()
        self.relu = nn.ReLU().cuda()

    def forward(self, x):
        res = self.module_body(x)
        res += x
        return self.relu(res)  # [4, 32, 112, 112]


class Res_ch(nn.Module):
    def __init__(self, n_feats, blocks=2):
        super(Res_ch, self).__init__()
        self.conv_init1 = Convd(3, n_feats // 2, 3, 1).cuda()
        self.conv_init2 = Convd(n_feats // 2, n_feats, 3, 1).cuda()
        self.extra = RIR(n_feats, n_blocks=blocks).cuda()

    def forward(self, x):
        x = self.conv_init2(self.conv_init1(x))  # x = [4, 32, 112, 112]
        x = self.extra(x)  # [4, 32, 112, 112]
        return x  # [4, 32, 112, 112]


class RCP(nn.Module):
    def __init__(self, n_feats, blocks=2):  # n_feats = 32
        super(RCP, self).__init__()
        self.rcp_result = Res_ch(n_feats, blocks).cuda()

    def forward(self, x):
        res_channel = get_residue(tensor=x, r_dim=1)  # res_channel = [4, 1, 112, 112]
        return self.rcp_result(torch.cat((res_channel, res_channel, res_channel), dim=1))
        # cat = [4, 3, 112, 112]    return = [4, 32, 112, 112]


class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):   # [4, 3, 112, 112]
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

class Weight_Prediction_Network(nn.Module):
    def __init__(self,n_feats=128):
        super(Weight_Prediction_Network, self).__init__()
        f = n_feats // 4
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.conv_dilation = nn.Conv2d(f, f, kernel_size=3, padding=1,
                                        stride=3, dilation=2)
    def forward(self, x): # x is the input feature
        x = self.conv1(x)
        shortCut = x
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=7, stride=3)
        x = self.relu(self.conv_max(x))
        x = self.relu(self.conv3(x))
        x = self.conv3_(x)
        x = F.interpolate(x, (shortCut.size(2), shortCut.size(3)),
                          mode='bilinear', align_corners=False)
        shortCut = self.conv_f(shortCut)
        x = self.conv4(x+shortCut)
        x = self.sigmoid(x)
        return x


class PyConv4(nn.Module):
    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 2, 4, 8]):
        super(PyConv4, self).__init__()
        self.conv1 = KernelConv(inplans, planes // 4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                                stride=stride, groups=pyconv_groups[0])
        self.conv2 = KernelConv(inplans, planes // 4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                                stride=stride, groups=pyconv_groups[1])
        self.conv3 = KernelConv(inplans, planes // 4, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2] // 2,
                                stride=stride, groups=pyconv_groups[2])
        self.conv4 = KernelConv(inplans, planes // 4, kernel_size=pyconv_kernels[3], padding=pyconv_kernels[3] // 2,
                                stride=stride, groups=pyconv_groups[3])

    def forward(self, x):
        return torch.cat((self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)), dim=1)


class GlobalPyConvBlock(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm=nn.BatchNorm2d):
        super(GlobalPyConvBlock, self).__init__()
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(bins),
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            BatchNorm(reduction_dim),
            nn.ReLU(inplace=True),
            PyConv4(reduction_dim, reduction_dim),
            BatchNorm(reduction_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_dim, reduction_dim, kernel_size=1, bias=False),
            BatchNorm(reduction_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_size = x.size()
        x = F.interpolate(self.features(x), x_size[2:], mode='bilinear', align_corners=True)
        return x


class LocalPyConvBlock(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm=nn.BatchNorm2d, reduction1=4):
        super(LocalPyConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inplanes, inplanes // reduction1, kernel_size=1, bias=False),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True),
            PyConv4(inplanes // reduction1, inplanes // reduction1),
            BatchNorm(inplanes // reduction1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // reduction1, planes, kernel_size=1, bias=False),
            BatchNorm(planes),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        return self.layers(x)


class MergeLocalGlobal(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm=nn.BatchNorm2d):
        super(MergeLocalGlobal, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, groups=1, bias=False),
            BatchNorm(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, local_context, global_context):
        x = torch.cat((local_context, global_context), dim=1)
        x = self.features(x)
        return x


###############################################################################################################
# TODO: Residue Channel Prior Guidance(RCP) -> Preserving Structure
class PreservingStructure(nn.Module):
    def __init__(self, num_features=128, ratio=8, n_feats=32):
        super(PreservingStructure, self).__init__()
        self.sk_conv = SKConv(num_features, ratio).cuda()
        self.rcp_extra = RCP(n_feats).cuda()

    def forward(self, x_ST):
        x_ST_rcp_pre = nor_conv(x=x_ST, in_channel=x_ST.shape[1], out_channel=3)  # x_ST_conv_pre = [4, 3, 112, 112]
        x_ST_sk_pre = nor_conv(x=x_ST, in_channel=x_ST.shape[1],
                               out_channel=x_ST.shape[1])  # x_ST_sk_pre = [4, 128, 112, 112]
        x_ST_rcp = self.rcp_extra(x_ST_rcp_pre)  # x_ST = [4, 3, 112, 112]    x_ST_rcp = [4, 32, 112, 112]
        x_ST_conv = nor_conv(x=x_ST_rcp, in_channel=x_ST_rcp.shape[1],
                             out_channel=x_ST.shape[1])  # x_ST_conv = [4, 128, 112, 112]
        return self.sk_conv(x_ST_sk_pre, x_ST_conv)


class PreservingStructure_wskconv(nn.Module):
    def __init__(self, num_features=128, ratio=8, n_feats=32):
        super(PreservingStructure_wskconv, self).__init__()
        #self.sk_conv = SKConv(num_features, ratio).cuda()
        self.rcp_extra = RCP(n_feats).cuda()

    def forward(self, x_ST):
        x_ST_rcp_pre = nor_conv(x=x_ST, in_channel=x_ST.shape[1], out_channel=3)  # x_ST_conv_pre = [4, 3, 112, 112]
       # x_ST_sk_pre = nor_conv(x=x_ST, in_channel=x_ST.shape[1],
        #                       out_channel=x_ST.shape[1])  # x_ST_sk_pre = [4, 128, 112, 112]
        x_ST_rcp = self.rcp_extra(x_ST_rcp_pre)  # x_ST = [4, 3, 112, 112]    x_ST_rcp = [4, 32, 112, 112]
        x_ST_conv = nor_conv(x=x_ST_rcp, in_channel=x_ST_rcp.shape[1],
                             out_channel=x_ST.shape[1])  # x_ST_conv = [4, 128, 112, 112]
        return x_ST_conv

class PreservingStructure_wrcp(nn.Module):
    def __init__(self, num_features=128, ratio=8, n_feats=32):
        super(PreservingStructure_wrcp, self).__init__()
        self.sk_conv = SKConv(num_features, ratio).cuda()
        #self.rcp_extra = RCP(n_feats).cuda()

    def forward(self, x_ST):
        x_ST_rcp_pre = nor_conv(x=x_ST, in_channel=x_ST.shape[1], out_channel=3)  # x_ST_conv_pre = [4, 3, 112, 112]
        x_ST_sk_pre = nor_conv(x=x_ST, in_channel=x_ST.shape[1],
                               out_channel=x_ST.shape[1])  # x_ST_sk_pre = [4, 128, 112, 112]
        #x_ST_rcp = self.rcp_extra(x_ST_rcp_pre)  # x_ST = [4, 3, 112, 112]    x_ST_rcp = [4, 32, 112, 112]
        x_ST_conv = nor_conv(x=x_ST_rcp_pre, in_channel=x_ST_rcp_pre.shape[1],
                             out_channel=x_ST.shape[1])  # x_ST_conv = [4, 128, 112, 112]
        return self.sk_conv(x_ST_sk_pre, x_ST_conv)
###############################################################################################################
# # TODO: Residue Gradient Prior Guidance(RGP) -> Sharping Details
# class SharpingDetails(nn.Module):
#     def __init__(self, num_features=128, ratio=8):
#         super(SharpingDetails, self).__init__()
#         self.grad_operation = Get_gradient().cuda()
#         self.sk_conv = SKConv(num_features, ratio).cuda()
#
#     def forward(self, x_DE):    # [4, 128, 112, 112]
#         x_DE_pre = nor_conv(x=x_DE, in_channel=x_DE.shape[1], out_channel=3)    # [4, 3, 112, 112]
#         x_DE = nor_conv(x=x_DE, in_channel=x_DE.shape[1], out_channel=x_DE.shape[1])    # [4, 128, 112, 112]
#         x_DE_grad = self.grad_operation(x_DE_pre)   # [4, 3, 112, 112]
#         x_DE_conv = nor_conv(x=x_DE_grad, in_channel=x_DE_grad.shape[1], out_channel=x_DE.shape[1])
#         return x_DE_grad, self.sk_conv(x_DE_conv, x_DE)  # [4, 3, 112, 112] []


###############################################################################################################

#############################################################################################################

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

#############################################################################################################
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv)
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

##############################################################################################################

class PyConv4(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 2, 4, 8]):
        super(PyConv4, self).__init__()

        self.conv2_1 = nn.Conv2d(inplans, planes // 4, kernel_size=pyconv_kernels[0], stride=stride,
                                 padding=pyconv_kernels[0]//2, dilation=1, groups=pyconv_groups[0], bias=False)
        self.conv2_2 = nn.Conv2d(inplans, planes // 4, kernel_size=pyconv_kernels[1], stride=stride,
                                 padding=pyconv_kernels[1] // 2, dilation=1, groups=pyconv_groups[1], bias=False)
        self.conv2_3 = nn.Conv2d(inplans, planes // 4, kernel_size=pyconv_kernels[2], stride=stride,
                                 padding=pyconv_kernels[2] // 2, dilation=1, groups=pyconv_groups[2], bias=False)
        self.conv2_4 = nn.Conv2d(inplans, planes // 4, kernel_size=pyconv_kernels[3], stride=stride,
                                 padding=pyconv_kernels[3] // 2, dilation=1, groups=pyconv_groups[3], bias=False)

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)

class LocalPyConv(nn.Module):
    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 2, 4, 8]):
        super(LocalPyConv, self).__init__()

        self.conv2_1 = nn.Conv2d(inplans, planes // 4, kernel_size=pyconv_kernels[0], stride=stride,
                                 padding=pyconv_kernels[0]//2, dilation=1, groups=pyconv_groups[0], bias=False)
        self.conv2_2 = nn.Conv2d(inplans, planes // 4, kernel_size=pyconv_kernels[1], stride=stride,
                                 padding=pyconv_kernels[1] // 2, dilation=1, groups=pyconv_groups[1], bias=False)
        self.conv2_3 = nn.Conv2d(inplans, planes // 4, kernel_size=pyconv_kernels[2], stride=stride,
                                 padding=pyconv_kernels[2] // 2, dilation=1, groups=pyconv_groups[2], bias=False)
        self.conv2_4 = nn.Conv2d(inplans, planes // 4, kernel_size=pyconv_kernels[3], stride=stride,
                                 padding=pyconv_kernels[3] // 2, dilation=1, groups=pyconv_groups[3], bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)))

class PyConvDecode(nn.Module):
    def __init__(self, inplanes, planes, BatchNorm=nn.BatchNorm2d):
        super(PyConvDecode, self).__init__()

        out_size_local_context = 512
        out_size_global_context = 512

        self.local_context = LocalPyConvBlock(inplanes, out_size_local_context, BatchNorm, reduction1=4)
        self.global_context = GlobalPyConvBlock(inplanes, out_size_global_context, 9, BatchNorm)

        self.merge_context = MergeLocalGlobal(out_size_local_context + out_size_global_context, planes, BatchNorm)

    def forward(self, x):
        x = self.merge_context(self.local_context(x), self.global_context(x))
        return x


class InceptionBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), width=1, drop_rate=0,
                 use_bias=False):
        super(InceptionBlock, self).__init__()

        self.width = width
        self.drop_rate = drop_rate

        for i in range(width):
            layer = nn.Sequential(
                nn.ReflectionPad2d(i * 2 + 1),
                nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=0, dilation=i * 2 + 1, bias=use_bias)
            )
            setattr(self, 'layer' + str(i), layer)

        self.norm1 = norm_layer(output_nc * width)
        self.norm2 = norm_layer(output_nc)
        self.nonlinearity = nonlinearity
        self.branch1x1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(output_nc * width, output_nc, kernel_size=3, padding=0, bias=use_bias)
        )

    def forward(self, x):
        result = []
        for i in range(self.width):
            layer = getattr(self, 'layer' + str(i))
            result.append(layer(x))
        output = torch.cat(result, 1)
        output = self.nonlinearity(self.norm1(output))
        output = self.norm2(self.branch1x1(output))
        if self.drop_rate > 0:
            output = F.dropout(output, p=self.drop_rate, training=self.training)

        return self.nonlinearity(output + x)


class DecoderUpBlock(nn.Module):
    def __init__(self, input_nc, middle_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(),
                 use_bias=False):
        super(DecoderUpBlock, self).__init__()

        model = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc, middle_nc, kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(middle_nc),
            nonlinearity,
            nn.ConvTranspose2d(middle_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(output_nc),
            nonlinearity
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class OutputBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=3, use_bias=False):
        super(OutputBlock, self).__init__()

        model = [
            nn.ReflectionPad2d(int(kernel_size / 2)),
            nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, padding=0, bias=use_bias),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class BasicBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), use_bias=False):
        super(BasicBlock, self).__init__()

        model = [
            nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nonlinearity,
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FeaBlock(nn.Module):
    def __init__(self, C_in, C_out):
        super(FeaBlock, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_out, 1, 1, 0)
        self.bn = nn.BatchNorm2d(C_out)
        self.act = nn.PReLU()
        self.attention = SE_Block(C_out, reduction=8)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.attention(out)
        return out


class Upblock(nn.Module):
    def __init__(self, C_in, C_out,stride=2, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(),output_padding=1):
        super(Upblock, self).__init__()

        model = [
            nn.ConvTranspose2d(C_in, C_out, kernel_size=3, stride=stride, padding=1, output_padding=output_padding),
            norm_layer(C_out),
            nonlinearity
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ConvDown(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, padding=0, dilation=1, groups=1, bias=False, layers=1, activ=True):
        super().__init__()
        nf_mult = 1
        nums = out_c / 64
        sequence = []

        for i in range(1, layers + 1):
            nf_mult_prev = nf_mult
            if nums == 8:
                if in_c == 512:

                    nfmult = 1
                else:
                    nf_mult = 2

            else:
                nf_mult = min(2 ** i, 8)
            if kernel != 1:

                if activ == False and layers == 1:
                    sequence += [
                        nn.Conv2d(nf_mult_prev * in_c, nf_mult * in_c,
                                  kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                        nn.InstanceNorm2d(nf_mult * in_c)
                    ]
                else:
                    sequence += [
                        nn.Conv2d(nf_mult_prev * in_c, nf_mult * in_c,
                                  kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                        nn.InstanceNorm2d(nf_mult * in_c),
                        nn.LeakyReLU(0.2, True)
                    ]

            else:

                sequence += [
                    nn.Conv2d(in_c, out_c,
                              kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                    nn.InstanceNorm2d(out_c),
                    nn.LeakyReLU(0.2, True)
                ]

            if activ == False:
                if i + 1 == layers:
                    if layers == 2:
                        sequence += [
                            nn.Conv2d(nf_mult * in_c, nf_mult * in_c,
                                      kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                            nn.InstanceNorm2d(nf_mult * in_c)
                        ]
                    else:
                        sequence += [
                            nn.Conv2d(nf_mult_prev * in_c, nf_mult * in_c,
                                      kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                            nn.InstanceNorm2d(nf_mult * in_c)
                        ]
                    break

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)

    def forward(self, inputt):
        input = inputt

        output = self.input_conv(input)
        out = output
        return out


class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='leaky',
                 conv_bias=False, innorm=False, inner=False, outer=False):
        super().__init__()
        if sample == 'same-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 1, 2, bias=conv_bias)
        elif sample == 'same-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 1, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.InstanceNorm2d(out_ch, affine=True)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.innorm = innorm
        self.inner = inner
        self.outer = outer

    def forward(self, input):
        out = input
        if self.inner:
            out = self.bn(out)
            out = self.activation(out)
            out = self.conv(out)
            out = self.bn(out)
            out = self.activation(out)

        elif self.innorm:
            out = self.conv(out)
            out = self.bn(out)
            out = self.activation(out)
        elif self.outer:
            out = self.conv(out)
            out = self.bn(out)
        else:
            out = self.conv(out)
            out = self.bn(out)
            if hasattr(self, 'activation'):
                out = self.activation(out)
        return out


class BASE(nn.Module):
    def __init__(self, inner_nc):
        super(BASE, self).__init__()
        se = SELayer(inner_nc, 16)
        model = [se]
        gus = util.gussin(1.5).cuda()
        self.gus = torch.unsqueeze(gus, 1).double()
        self.model = nn.Sequential(*model)
        self.down = nn.Sequential(
            nn.Conv2d(1024, 512, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        Nonparm = Selfpatch()
        out_32 = self.model(x)
        b, c, h, w = out_32.size()
        gus = self.gus.float()
        gus_out = out_32[0].expand(h * w, c, h, w)
        gus_out = gus * gus_out
        gus_out = torch.sum(gus_out, -1)
        gus_out = torch.sum(gus_out, -1)
        gus_out = gus_out.contiguous().view(b, c, h, w)
        csa2_in = F.sigmoid(out_32)
        csa2_f = torch.nn.functional.pad(csa2_in, (1, 1, 1, 1))
        csa2_ff = torch.nn.functional.pad(out_32, (1, 1, 1, 1))
        csa2_fff, csa2_f, csa2_conv = Nonparm.buildAutoencoder(csa2_f[0], csa2_in[0], csa2_ff[0], 3, 1)
        csa2_conv = csa2_conv.expand_as(csa2_f)
        csa_a = csa2_conv * csa2_f
        csa_a = torch.mean(csa_a, 1)
        a_c, a_h, a_w = csa_a.size()
        csa_a = csa_a.contiguous().view(a_c, -1)
        csa_a = F.softmax(csa_a, dim=1)
        csa_a = csa_a.contiguous().view(a_c, 1, a_h, a_h)
        out = csa_a * csa2_fff
        out = torch.sum(out, -1)
        out = torch.sum(out, -1)
        out_csa = out.contiguous().view(b, c, h, w)
        out_32 = torch.cat([gus_out, out_csa], 1)
        out_32 = self.down(out_32)
        return out_32
