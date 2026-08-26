import math

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from ultralytics.nn.modules import Conv, DFL
from ultralytics.nn.modules.conv import autopad
from ultralytics.utils.tal import dist2bbox, make_anchors


class Scale(nn.Module):
    """Learnable scalar multiplier."""

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale


class Conv_GN(nn.Module):
    """Conv + GroupNorm + activation."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.gn = nn.GroupNorm(16, c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def get_weight(self):
        weight = self.conv.weight
        shape = weight.shape
        weight = Rearrange("c_in c_out k1 k2 -> c_in c_out (k1 k2)")(weight)
        cd = torch.zeros(shape[0], shape[1], 9, device=weight.device, dtype=weight.dtype)
        cd[:, :, :] = weight[:, :, :]
        cd[:, :, 4] = weight[:, :, 4] - weight[:, :, :].sum(2)
        cd = Rearrange("c_in c_out (k1 k2) -> c_in c_out k1 k2", k1=shape[2], k2=shape[3])(cd)
        return cd, self.conv.bias


class Conv2d_ad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.theta = theta

    def get_weight(self):
        weight = self.conv.weight
        shape = weight.shape
        weight = Rearrange("c_in c_out k1 k2 -> c_in c_out (k1 k2)")(weight)
        ad = weight - self.theta * weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        ad = Rearrange("c_in c_out (k1 k2) -> c_in c_out k1 k2", k1=shape[2], k2=shape[3])(ad)
        return ad, self.conv.bias


class Conv2d_hd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def get_weight(self):
        weight = self.conv.weight
        shape = weight.shape
        hd = torch.zeros(shape[0], shape[1], 9, device=weight.device, dtype=weight.dtype)
        hd[:, :, [0, 3, 6]] = weight[:, :, :]
        hd[:, :, [2, 5, 8]] = -weight[:, :, :]
        hd = Rearrange("c_in c_out (k1 k2) -> c_in c_out k1 k2", k1=shape[2], k2=shape[2])(hd)
        return hd, self.conv.bias


class Conv2d_vd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def get_weight(self):
        weight = self.conv.weight
        shape = weight.shape
        vd = torch.zeros(shape[0], shape[1], 9, device=weight.device, dtype=weight.dtype)
        vd[:, :, [0, 1, 2]] = weight[:, :, :]
        vd[:, :, [6, 7, 8]] = -weight[:, :, :]
        vd = Rearrange("c_in c_out (k1 k2) -> c_in c_out k1 k2", k1=shape[2], k2=shape[2])(vd)
        return vd, self.conv.bias


class DEConv(nn.Module):
    """Detail-enhanced convolution."""

    def __init__(self, dim):
        super().__init__()
        self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)
        self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True)
        self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True)
        self.conv1_4 = Conv2d_ad(dim, dim, 3, bias=True)
        self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(dim)
        self.act = Conv.default_act

    def forward(self, x):
        if hasattr(self, "conv1_1"):
            w1, b1 = self.conv1_1.get_weight()
            w2, b2 = self.conv1_2.get_weight()
            w3, b3 = self.conv1_3.get_weight()
            w4, b4 = self.conv1_4.get_weight()
            w5, b5 = self.conv1_5.weight, self.conv1_5.bias
            x = nn.functional.conv2d(x, w1 + w2 + w3 + w4 + w5, b1 + b2 + b3 + b4 + b5, stride=1, padding=1)
        else:
            x = self.conv1_5(x)

        if hasattr(self, "bn"):
            x = self.bn(x)
        return self.act(x)


class DEConv_GN(DEConv):
    """Detail-enhanced convolution with GroupNorm."""

    def __init__(self, dim):
        super().__init__(dim)
        self.bn = nn.GroupNorm(16, dim)


class Detect_LSDECD(nn.Module):
    """Lightweight shared detail-enhanced convolutional detection head."""

    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, hidc=256, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        self.conv = nn.ModuleList(nn.Sequential(Conv_GN(x, hidc, 1)) for x in ch)
        self.share_conv = nn.Sequential(DEConv_GN(hidc), DEConv_GN(hidc))
        self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(hidc, self.nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for _ in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])
            x[i] = self.share_conv(x[i])
            x[i] = torch.cat((self.scale[i](self.cv2(x[i])), self.cv3(x[i])), 1)

        if self.training:
            return x

        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        dbox = self.decode_bboxes(box)
        if self.export and self.format in ("tflite", "edgetpu"):
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        self.cv2.bias.data[:] = 1.0
        self.cv3.bias.data[: self.nc] = math.log(5 / self.nc / (640 / 16) ** 2)

    def decode_bboxes(self, bboxes):
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
