import torch
import torch.nn as nn
import torch.nn.functional as F


class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = HSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CA_HSFPN(nn.Module):
    """Coordinate attention used in the HSFPN/HAFPN feature fusion path."""

    def __init__(self, inp, reduction=8, flag=True):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = HSwish()
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.flag = flag

    def forward(self, x):
        _, _, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        out = self.conv_w(x_w).sigmoid() * self.conv_h(x_h).sigmoid()
        return x * out if self.flag else out


class Multiply(nn.Module):
    def forward(self, x):
        return x[0] * x[1]


class Add(nn.Module):
    def forward(self, x):
        if len(x) == 0:
            return x

        target_size = x[0].shape[-2:]
        aligned = [
            F.interpolate(t, size=target_size, mode="bilinear", align_corners=False)
            if t.shape[-2:] != target_size
            else t
            for t in x
        ]
        return torch.sum(torch.stack(aligned, dim=0), dim=0)
