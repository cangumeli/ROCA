from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Adapted from
# https://github.com/JunjH/Revisiting_Single_Depth_Estimation/blob/master/models/modules.py
class UpProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=5, stride=1, padding=2
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=5, stride=1, padding=2
        )
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] != size:
            x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        x_conv1 = self.relu1(self.conv1(x))

        bran1 = self.conv1_2(x_conv1)
        bran2 = self.conv2(x)
        return self.relu2(bran1 + bran2)


class DepthFeatures(nn.Module):
    def __init__(
        self,
        size: Tuple[int, int] = (120, 160),
        num_levels: int = 4,
        in_channels: int = 256,
        out_channels_per_level: int = 32
    ):
        super().__init__()
        self.size = size
        self.ups = nn.ModuleList([
            UpProjection(in_channels, out_channels_per_level)
            for _ in range(num_levels)
        ])
        self.out_channels = out_channels_per_level * num_levels

    def forward(self, features: List[torch.Tensor]):
        return torch.cat([
            up(x, self.size) for x, up in zip(features, self.ups)
        ], dim=1)


class DepthOutput(nn.Module):
    def __init__(
        self,
        in_channels: int,
        up_ratio: Union[int, float],
        num_hiddens: int = 2,
        hidden_channels: int = 128
    ):
        super().__init__()

        assert up_ratio == int(up_ratio)
        up_ratio = int(up_ratio)

        convs = []
        for i in range(num_hiddens):
            convs.append(nn.Conv2d(
                in_channels, hidden_channels, kernel_size=5, padding=2
            ))
            convs.append(nn.ReLU(True))
            in_channels = hidden_channels
        self.convs = nn.Sequential(*convs)

        if up_ratio > 1:
            # import pdb; pdb.set_trace()
            self.output = nn.Sequential(
                nn.Conv2d(hidden_channels, up_ratio**2, kernel_size=1),
                nn.PixelShuffle(up_ratio)
            )
        else:
            self.output = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(
        self,
        features: torch.Tensor,
        output_size: Optional[Tuple[int, int]]=None
    ) -> torch.Tensor:
        features = self.convs(features)
        depth = self.output(features)
        if output_size is not None and output_size != depth.shape[-2:]:
            # import pdb; pdb.set_trace()
            depth = F.interpolate(
                depth, output_size, mode='bilinear', align_corners=True
            )
        return depth


# https://github.com/JunjH/Revisiting_Single_Depth_Estimation/blob/master/sobel.py
class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_conv = nn.Conv2d(
            1, 2, kernel_size=3, stride=1, padding=1, bias=False
        )
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
        return out
