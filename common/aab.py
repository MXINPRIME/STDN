import torch
from torch import nn
import torch.nn.functional as F

class AttentionBranch(nn.Module):

    def __init__(self, nf, k_size=3):
        super(AttentionBranch, self).__init__()
        self.k1 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.k2 = nn.Conv2d(nf, nf, 1)  # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution

    def forward(self, x):
        y = self.k1(x)
        y = self.lrelu(y)
        y = self.k2(y)
        y = self.sigmoid(y)

        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out

class AAB(nn.Module):

    def __init__(self, nf, reduction=4, K=2, t=30):
        super(AAB, self).__init__()
        self.t = t
        self.K = K

        self.conv_first = nn.Conv2d(nf, nf, kernel_size=1, bias=False)
        self.conv_last = nn.Conv2d(nf, nf, kernel_size=1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Attention Dropout Module
        self.ADM = nn.Sequential(
            nn.Linear(nf, nf // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nf // reduction, self.K, bias=False),
        )

        # attention branch
        self.attention = AttentionBranch(nf)

        # non-attention branch
        # 3x3 conv for A2N
        self.non_attention = nn.Conv2d(nf, nf, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        # 1x1 conv for A2N-M (Recommended, fewer parameters)
        # self.non_attention = nn.Conv2d(nf, nf, kernel_size=1, bias=False)

    def forward(self, x):
        residual = x
        a, b, c, d = x.shape

        x = self.conv_first(x)
        x = self.lrelu(x)

        # Attention Dropout
        y = self.avg_pool(x).view(a, b)
        y = self.ADM(y)
        ax = F.softmax(y / self.t, dim=1)

        attention = self.attention(x)
        non_attention = self.non_attention(x)

        x = attention * ax[:, 0].view(a, 1, 1, 1) + non_attention * ax[:, 1].view(a, 1, 1, 1)
        x = self.lrelu(x)

        out = self.conv_last(x)
        out += residual

        return out