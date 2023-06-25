import functools

import numpy as np
import torch
import torch.nn as nn
from common.undernet_include import *
from common.aab import *

'''
Dual-Branch Squeeze-and-Excitation Attention Module 
'''
class DuSEAttention(nn.Module):
    def __init__(self, n_channels_extract=32):
        super(DuSEAttention, self).__init__()
        # (1) Spatial-Squeeze + Channel-Excitation
        self.avg_pool_ch1 = nn.AdaptiveAvgPool2d((1,1))
        self.avg_pool_ch2 = nn.AdaptiveAvgPool2d((1,1))
        self.fc_comb = nn.Linear(n_channels_extract * 2, n_channels_extract, bias=True)
        self.fc_ch1  = nn.Linear(n_channels_extract, n_channels_extract, bias=True)
        self.fc_ch2  = nn.Linear(n_channels_extract, n_channels_extract, bias=True)

        # (2) Channel-Squeeze + Spatial-Excitation
        self.conv_squeeze_ch1 = nn.Conv2d(n_channels_extract, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_squeeze_ch2 = nn.Conv2d(n_channels_extract, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_comb = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_adjust_ch1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_adjust_ch2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=True)

        # (3) Concatenation + Feature Fusion
        self.conv_fuse_ch1 = nn.Conv2d(n_channels_extract*3, n_channels_extract, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_fuse_ch1 = nn.BatchNorm2d(n_channels_extract)
        self.conv_fuse_ch2 = nn.Conv2d(n_channels_extract*3, n_channels_extract, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_fuse_ch2 = nn.BatchNorm2d(n_channels_extract)


    def forward(self, inp_ch1, inp_ch2):
        # Basic Information
        batch_size, n_channels, H, W = inp_ch1.size()

        # (1) Spatial-Squeeze + Channel-Excitation
        squeeze_ch1 = self.avg_pool_ch1(inp_ch1).view(batch_size, n_channels)  # [B, C,1,1] to [B, C]
        squeeze_ch2 = self.avg_pool_ch2(inp_ch2).view(batch_size, n_channels)  # [B, C,1,1] to [B, C]
        squeeze_comb = torch.cat((squeeze_ch1, squeeze_ch2), 1)  # [B, C*2]

        # Fully connected layers
        fc_comb = self.fc_comb(squeeze_comb)
        # fc_ch1 = self.fc_ch1(fc_comb)
        # fc_ch2 = self.fc_ch2(fc_comb)
        fc_ch1 = torch.sigmoid(self.fc_ch1(fc_comb))
        fc_ch2 = torch.sigmoid(self.fc_ch2(fc_comb))

        # Multiplication
        inp_ch1_scSE = torch.mul(inp_ch1, fc_ch1.view(batch_size, n_channels, 1, 1))
        inp_ch2_scSE = torch.mul(inp_ch2, fc_ch2.view(batch_size, n_channels, 1, 1))  # [B, C, D,H,W]


        # (2) Channel-Squeeze + Spatial-Excitation
        squeeze_volume_ch1 = self.conv_squeeze_ch1(inp_ch1)
        squeeze_volume_ch2 = self.conv_squeeze_ch2(inp_ch2)  # [B, 1, H,W]
        squeeze_volume_comb = torch.cat((squeeze_volume_ch1, squeeze_volume_ch2), 1)  # [B, 2, D,H,W]

        # Fusion Layer
        conv_comb = self.conv_comb(squeeze_volume_comb)  # [B, 1, D,H,W]
        # conv_adjust_ch1 = self.conv_adjust_ch1(conv_comb)
        # conv_adjust_ch2 = self.conv_adjust_ch2(conv_comb)  # [B, 1, D,H,W]
        conv_adjust_ch1 = torch.sigmoid(self.conv_adjust_ch1(conv_comb))
        conv_adjust_ch2 = torch.sigmoid(self.conv_adjust_ch2(conv_comb))  # [B, 1, D,H,W]

        # Multiplication
        inp_ch1_csSE = torch.mul(inp_ch1, conv_adjust_ch1.view(batch_size, 1, H, W))
        inp_ch2_csSE = torch.mul(inp_ch2, conv_adjust_ch2.view(batch_size, 1, H, W))  # [B, C, D,H,W]


        # (3) Concatenation + Feature Fusion
        inp_ch1_fuse = self.bn_fuse_ch1(inp_ch1 + inp_ch1_scSE + inp_ch1_csSE)
        inp_ch2_fuse = self.bn_fuse_ch2(inp_ch2 + inp_ch2_scSE + inp_ch2_csSE)

        return inp_ch1_fuse, inp_ch2_fuse

class UnderNet(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, center_num=3, norm='batch', activation='PReLU', drop_rate=0,
                 gpu_ids=[], weight=1.0):
        super(UnderNet, self).__init__()

        self.weight = weight
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            nonlinearity
        )

        center = []
        for i in range(center_num):
            center += [
                InceptionBlock(ngf * 8, ngf * 8, norm_layer, nonlinearity, center_num, drop_rate, use_bias)
            ]
        self.center = nn.Sequential(*center)

        """
        TODO：这里面的up应该是pooling
        """
        self.up4 = Upblock(ngf * 8, ngf * 8)
        self.de4 = LocalPyConv(ngf * 24, ngf * 8)  # TODO: 768 -> 256
        self.up3 = Upblock(ngf * 8, ngf * 4)
        self.de3 = LocalPyConv(ngf * 16, ngf * 4)  # TODO: 512 -> 128
        self.up2 = Upblock(ngf * 4, ngf * 2)
        self.de2 = LocalPyConv(ngf * 8, ngf * 2)  # TODO: 256 -> 64
        self.up1 = Upblock(ngf * 2, ngf)
        self.de1 = LocalPyConv(ngf * 4, ngf)  # TODO: 128 -> 32

        self.out = OutputBlock(ngf, output_nc, 7, use_bias)

        # self.fea1 = FeaBlock(64, ngf)
        # self.fea2 = FeaBlock(128, ngf * 2)
        # self.fea3 = FeaBlock(256, ngf * 4)
        # self.fea4 = FeaBlock(512, ngf * 8)

        self.fea1_1 = Upblock(128, 32)
        self.fea2_1=Upblock(128,64,1,output_padding=0)
        self.fea3_1 = ConvDown(128, 256, 4, 2, padding=1)
        self.fea3_2=ConvDown(256,128,1,1)
        self.fea4_1 = ConvDown(128, 256, 4, 2, padding=1)
        self.fea4_2 = ConvDown(256, 512, 4, 2, padding=1)
        self.fea4_3 = ConvDown(512, 256, 1, 1)

        # AAB
        self.aab1 = AAB(nf=ngf)
        self.aab2 = AAB(nf=ngf*2)
        self.aab3 = AAB(nf=ngf*4)
        self.aab4 = AAB(nf=ngf*8)

        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.down_1 = ConvDown(64, 128, 4, 2, padding=1)
        self.down_2 = ConvDown(128, 128, 1, 1)

        self.up_2 = Upblock(256, 128)
        self.up_3 = Upblock(512, 256)
        self.up_4 = Upblock(256, 128)

        self.down_DE = ConvDown(256, 128, 1, 1)
        self.down_ST = ConvDown(256, 128, 1, 1)
        self.down_fus_DE = ConvDown(384, 128, 1, 1)
        self.fuse = ConvDown(256, 128, 1, 1)

        self.rcp = PreservingStructure()
        # self.rgp = SharpingDetails()
        # self.attn = Attention(128, 2,False )

        self.attn_duse = DuSEAttention(128)

        # self.conv = default_conv(128,128,3)
        # self.resblock1 = ResBlock(self.conv,128,3,bias=True,bn=True)
        # self.resblock2 = ResBlock(self.conv, 128, 3,bias=True, bn=True)

        seuqence_3 = []
        seuqence_5 = []
        seuqence_7 = []
        for i in range(3):
            seuqence_3 += [PCBActiv(128, 128, innorm=True)]
            seuqence_5 += [PCBActiv(128, 128, sample='same-5', innorm=True)]
            seuqence_7 += [PCBActiv(128, 128, sample='same-7', innorm=True)]

        self.cov_3 = nn.Sequential(*seuqence_3)
        self.cov_5 = nn.Sequential(*seuqence_5)
        self.cov_7 = nn.Sequential(*seuqence_7)

        self.base = BASE(256)
        self.act = nn.ReLU()

    def forward(self, x, fea_vgg):
        # TODO: vgg每层的特征抽取出来（1.2、细节信息 3.4、结构信息）
        # TODO: ##################################################################
        x1 = self.activation(fea_vgg[0])  # torch.Size([4, 64, 224, 224])
        x2 = self.activation(fea_vgg[1])  # torch.Size([4, 128, 112, 112])
        x3 = self.activation(fea_vgg[2])  # torch.Size([4, 256, 56, 56])
        x4 = self.activation(fea_vgg[3])  # torch.Size([4, 512, 28, 28])

        x1_1 = self.down_1(x1)  # torch.Size([4, 128, 112, 112])
        x2_1 = self.down_2(x2)  # torch.Size([4, 128, 112, 112])
        x3_1 = self.up_2(x3)  # torch.Size([4, 128, 112, 112])
        x4_1 = self.up_4(self.up_3(x4))  # torch.Size([4, 128, 112, 112])

        # TODO: 特征选择器
        x1_1, x2_1 = self.attn_duse(x1_1, x2_1)
        x_DE = torch.cat([x1_1, x2_1], 1)  # torch.Size([4, 256, 112, 112])
        x_ST = torch.cat([x3_1, x4_1], 1)  # torch.Size([4, 256, 112, 112])

        x_DE = self.down_DE(x_DE)  # torch.Size([4, 128, 112, 112])
        x_ST = self.down_ST(x_ST)  # torch.Size([4, 128, 112, 112])

        # 多路径 CFRM 了解详情
        x_DE_3 = self.cov_3(x_DE)  # torch.Size([4, 128, 112, 112])
        x_DE_5 = self.cov_5(x_DE)  # torch.Size([4, 128, 112, 112])
        x_DE_7 = self.cov_7(x_DE)  # torch.Size([4, 128, 112, 112])
        x_DE_fuse = torch.cat([x_DE_3, x_DE_5, x_DE_7], 1)  # torch.Size([4, 384, 112, 112])
        x_DE_fi = self.down_fus_DE(x_DE_fuse)  # torch.Size([4, 128, 112, 112])
        #
        # # 结构的多路径 CFRM
        # x_ST_3 = self.cov_3(x_ST)  # torch.Size([8, 128, 112, 112])
        # x_ST_5 = self.cov_5(x_ST)  # torch.Size([4, 128, 112, 112])
        # x_ST_7 = self.cov_7(x_ST)  # torch.Size([4, 128, 112, 112])
        # x_ST_fuse = torch.cat([x_ST_3, x_ST_5, x_ST_7], 1)  # torch.Size([4, 384, 112, 112])
        # x_ST_fi = self.down_fus_DE(x_ST_fuse)  # torch.Size([4, 128, 112, 112])

        # TODO：1、RCP+SKConv 2、GT、Sobel+Sobel+SKConv
        # 1 PreservingStructure \ 2 SharpingDetails
        x_ST_fi = self.rcp(x_ST)

        # TODO: 细节与结构的融合
        x_cat = torch.cat([x_ST_fi, x_DE_fi], 1)  # torch.Size([4, 256, 112, 112])
        x_cat_fuse = self.fuse(x_cat)  # torch.Size([4, 128, 112, 112])

        # TODO: 特征主导子网  ##################################################################
        add_fea_1 = self.aab1(self.fea1_1(x_cat_fuse))  # torch.Size([4, 32, 224, 224])
        add_fea_2 = self.aab2(self.fea2_1(x_cat_fuse))  # torch.Size([4, 64, 112, 112])
        add_fea_3 = self.aab3(self.fea3_2(self.fea3_1(x_cat_fuse)))  # torch.Size([4, 128, 56, 56])
        add_fea_4 = self.aab4(self.fea4_3(self.fea4_2(self.fea4_1(x_cat_fuse))))  # torch.Size([4, 256, 28, 28])

        # TODO: Decoder
        # level 4
        deco4 = self.de4(
            torch.cat([add_fea_4, x4], dim=1))  # torch.Size([4, 256, 28, 28])

        # level 3
        deco3 = self.de3(
            torch.cat([self.up3(deco4), add_fea_3, x3], dim=1))  # torch.Size([4, 128, 56, 56])

        # level 2
        deco2 = self.de2(
            torch.cat([self.up2(deco3), add_fea_2, x2], dim=1))  # torch.Size([4, 64, 112, 112])

        # level 1
        deco1 = self.de1(
            torch.cat([self.up1(deco2), add_fea_1, x1], dim=1))  # torch.Size([4, 32, 224, 224])

        out = self.out(deco1)

        # x_DE_grad change size
        # conv_transpose = nn.ConvTranspose2d(in_channels=x_DE_grad.shape[1], out_channels=out.shape[1], kernel_size=4, stride=2, padding=1).cuda()
        # x_DE_grad = conv_transpose(x_DE_grad)

        # return out, x_DE_fi, x_ST_fi, x_DE_grad
        return out, x_DE_fi, x_ST_fi


if __name__ == '__main__':
    # def params_count(model):
    #     """
    #     Compute the number of parameters.
    #     Args:
    #         model (model): model to count the number of parameters.
    #     """
    #     return np.sum([p.numel() for p in model.parameters()]).item()
    #
    #
    # net = UnderNet(32, 32).cuda()
    # print(params_count(net) / (1000 ** 2))

    from model.my_vgg import VGGFeature
    from model.RCP_RGP_AAB_PYC import UnderNet
    import time
    x = torch.Tensor(1, 3, 512, 512).cuda()
    vgg = VGGFeature().cuda()
    gen = UnderNet(3, 3, ngf=32, weight=0.5).cuda()
    N = 10
    with torch.no_grad():
        for _ in range(N):
            fea_input = vgg(x)
            out, _, _ = gen(x, fea_input)

        result = []
        for _ in range(N):
            torch.cuda.synchronize()
            st = time.time()
            for _ in range(N):
                fea_input = vgg(x)
                out, _, _ = gen(x, fea_input)
            torch.cuda.synchronize()
            result.append((time.time() - st)/N)
        print("Running Time: {:.3f}s\n".format(np.mean(result)))
