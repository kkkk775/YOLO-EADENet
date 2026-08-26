import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import Conv


class SobelConv(nn.Module):
    """Depthwise Sobel operator for extracting edge responses."""

    def __init__(self, channel):
        super().__init__()
        sobel = torch.tensor(
            [
                [1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1],
            ],
            dtype=torch.float32,
        )
        sobel_kernel_y = sobel.unsqueeze(0).expand(channel, 1, 1, 3, 3)
        sobel_kernel_x = sobel.T.unsqueeze(0).expand(channel, 1, 1, 3, 3)

        self.sobel_kernel_x_conv3d = nn.Conv3d(
            channel, channel, kernel_size=3, padding=1, groups=channel, bias=False
        )
        self.sobel_kernel_y_conv3d = nn.Conv3d(
            channel, channel, kernel_size=3, padding=1, groups=channel, bias=False
        )
        self.sobel_kernel_x_conv3d.weight.data = sobel_kernel_x.clone()
        self.sobel_kernel_y_conv3d.weight.data = sobel_kernel_y.clone()
        self.sobel_kernel_x_conv3d.requires_grad_(False)
        self.sobel_kernel_y_conv3d.requires_grad_(False)

    def forward(self, x):
        x = x[:, :, None, :, :]
        edge_x = self.sobel_kernel_x_conv3d(x)
        edge_y = self.sobel_kernel_y_conv3d(x)
        return (edge_x + edge_y)[:, :, 0]


class MutilScaleEdgeInfoGenetator(nn.Module):
    """Generate multi-scale edge information for P3/P4/P5 fusion."""

    def __init__(self, inc, oucs):
        super().__init__()
        self.sc = SobelConv(inc)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_1x1s = nn.ModuleList(Conv(inc, ouc, 1) for ouc in oucs)

    def forward(self, x):
        outputs = [self.sc(x)]
        outputs.extend(self.maxpool(outputs[-1]) for _ in self.conv_1x1s)
        outputs = outputs[1:]
        for i, conv in enumerate(self.conv_1x1s):
            outputs[i] = conv(outputs[i])
        return outputs


class ConvEdgeFusion(nn.Module):
    """Fuse backbone features with edge information."""

    def __init__(self, inc, ouc):
        super().__init__()
        self.conv_channel_fusion = Conv(sum(inc), ouc // 2, k=1)
        self.conv_3x3_feature_extract = Conv(ouc // 2, ouc // 2, 3)
        self.conv_1x1 = Conv(ouc // 2, ouc, 1)

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
        x = torch.cat(aligned, dim=1)
        return self.conv_1x1(self.conv_3x3_feature_extract(self.conv_channel_fusion(x)))
