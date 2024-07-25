import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import numpy as np
from ..modules.conv import Conv, DSConv

__all__ = ['HLFFF','GSConv','SCDown']

class GSConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, Conv.default_act)
        self.cv2 = Conv(c_, c_, 5, 1, p, c_, d, Conv.default_act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])

        b, n, h, w = x2.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)


class EFF(nn.Module):
    def __init__(self, in_C, out_C):
        super(EFF, self).__init__()
        self.RGB_K = DSConv(out_C, out_C, 3)
        self.RGB_V = DSConv(out_C, out_C, 3)
        self.Q = DSConv(in_C, out_C, 3)
        self.INF_K = DSConv(out_C, out_C, 3)
        self.INF_V = DSConv(out_C, out_C, 3)
        self.Second_reduce = DSConv(in_C, out_C, 3)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        Q = self.Q(torch.cat([x, y], dim=1))
        RGB_K = self.RGB_K(x)
        RGB_V = self.RGB_V(x)
        m_batchsize, C, height, width = RGB_V.size()
        RGB_V = RGB_V.view(m_batchsize, -1, width * height)
        RGB_K = RGB_K.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        RGB_Q = Q.view(m_batchsize, -1, width * height)
        RGB_mask = torch.bmm(RGB_K, RGB_Q)
        RGB_mask = self.softmax(RGB_mask)
        RGB_refine = torch.bmm(RGB_V, RGB_mask.permute(0, 2, 1))
        RGB_refine = RGB_refine.view(m_batchsize, -1, height, width)
        RGB_refine = self.gamma1 * RGB_refine + y

        INF_K = self.INF_K(y)
        INF_V = self.INF_V(y)
        INF_V = INF_V.view(m_batchsize, -1, width * height)
        INF_K = INF_K.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        INF_Q = Q.view(m_batchsize, -1, width * height)
        INF_mask = torch.bmm(INF_K, INF_Q)
        INF_mask = self.softmax(INF_mask)
        INF_refine = torch.bmm(INF_V, INF_mask.permute(0, 2, 1))
        INF_refine = INF_refine.view(m_batchsize, -1, height, width)
        INF_refine = self.gamma2 * INF_refine + x

        out = self.Second_reduce(torch.cat([RGB_refine, INF_refine], dim=1))
        return out


class DenseLayer(nn.Module):
    def __init__(self, in_C, out_C, down_factor=4, k=2):
        super(DenseLayer, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(DSConv(mid_C * i, mid_C, 3))

        self.fuse = DSConv(in_C + mid_C, out_C, 3)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        out_feats = []
        for i in self.denseblock:
            feats = i(torch.cat((*out_feats, down_feats), dim=1))
            out_feats.append(feats)

        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)


class HLFFF(nn.Module):
    def __init__(self, Channel):
        super(HLFFF, self).__init__()
        self.RGBobj = DenseLayer(Channel, Channel)
        self.Infobj = DenseLayer(Channel, Channel)
        self.obj_fuse = EFF(Channel * 2, Channel)

    def forward(self, data):
        rgb, depth = data
        rgb_sum = self.RGBobj(rgb)
        Inf_sum = self.Infobj(depth)
        out = self.obj_fuse(rgb_sum, Inf_sum)
        return out

class SCDown(nn.Module):
    default_act = nn.SiLU()
    def __init__(self,c1,c2,k,s,act=True):
        super().__init__()
        self.cv1 = Conv(c1,c2,1,1)
        self.cv2 = Conv(c2,c2,k=k,s=s,g=c2)
        self.act = self.default_act if act is True else act if isinstance(act,nn.Module) else nn.Identity()

    def forward(self,x):
        return self.act(self.cv2(self.cv1(x)))