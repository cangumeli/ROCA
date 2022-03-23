import abc

import torch
import torch.nn as nn
import torch.nn.functional as F

from roca.modeling.retrieval_head.retrieval_ops import mask_point_features


class STN3d(nn.Module):
    def __init__(self, k=3):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k**2)

        self.k = k

    def forward(self, x, mask=None):
        x = F.relu_(self.conv1(x))
        x = F.relu_(self.conv2(x))
        x = F.relu_(self.conv3(x))

        if mask is not None:
            x = x * mask

        x = torch.max(x, 2).values
        x = F.relu_(self.fc1(x))
        x = F.relu_(self.fc2(x))
        x = self.fc3(x)

        iden = torch.eye(self.k, device=x.device).reshape(1, -1)
        x = x + iden
        x = x.view(-1, self.k, self.k)

        return x


class ResNetBlock(nn.Module):
    def __init__(
        self,
        num_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: nn.Module = nn.ReLU
    ):
        super().__init__()

        padding = kernel_size // 2

        self.weight_block_0 = nn.Sequential(
            activation(inplace=True),
            nn.Conv3d(
                num_channels,
                num_channels,
                kernel_size,
                stride=stride,
                padding=padding
            )
        )
        self.weight_block_1 = nn.Sequential(
            activation(inplace=True),
            nn.Conv3d(
                num_channels,
                num_channels,
                kernel_size,
                stride=stride,
                padding=padding
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.weight_block_0(x)
        out = self.weight_block_1(out)
        return x + out


class _RetrievalNetwork(nn.Module, abc.ABC):
    @abc.abstractproperty
    def embedding_dim(self):
        pass


class PointNet(_RetrievalNetwork):
    def __init__(self, relu_out=True, feat_trs=True, ret_trs=False):
        super().__init__()

        self.relu_out = relu_out
        self.feat_trs = feat_trs
        self.ret_trs = ret_trs

        self.stn = STN3d()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
    
        if self.feat_trs:
            self.fstn = STN3d(k=64)

    @property
    def embedding_dim(self):
        return self.conv3.out_channels

    def forward(self, x, mask=None):
        if x.numel() == 0:
            return self._empty_output(x)

        x = x.flatten(2)  # points are channels
        if mask is not None:
            mask = mask.flatten(2)  # num-points is the spatial dim

        trs = self.stn(x, mask)
        x = trs @ x

        x = F.relu_(self.conv1(x))
        if self.feat_trs:
            ftrs = self.fstn(x, mask)
            x = ftrs @ x
        else:
            ftrs = None
        
        x = F.relu_(self.conv2(x))
        x = self.conv3(x)

        x = mask_point_features(x, mask)
        x = torch.max(x, 2).values

        if self.relu_out:
            x = F.relu_(x)

        return (x, trs, ftrs) if self.ret_trs else x

    def _empty_output(self, x):
        with torch.device(x.device):
            x = torch.zeros(0, 1024)
            trs = torch.zeros(0, 3, 3)
            ftrs = torch.zeros(0, 64, 64) if self.feat_trs else None
        return (x, trs, ftrs) if self.ret_trs else x


class ResNetEncoder(_RetrievalNetwork):
    def __init__(self, relu_out: bool = False, embedding_dim: int = 256):
        super().__init__()

        self.feats = feats = [1, 8, 16, 32, 64, embedding_dim]

        self.network = nn.Sequential(
            # 32 x 32 x 32
            nn.Conv3d(feats[0], feats[1], 7, stride=1, padding=3),
            nn.ReLU(inplace=True),

            # 32 x 32 x 32
            nn.Conv3d(feats[1], feats[1], 4, stride=2, padding=1),
            ResNetBlock(feats[1]),

            # 16 x 16 x 16
            nn.ReLU(inplace=True),
            nn.Conv3d(feats[1], feats[2], 4, stride=2, padding=1),
            ResNetBlock(feats[2]),

            # 8 x 8 x 8
            nn.ReLU(inplace=True),
            nn.Conv3d(feats[2], feats[3], 4, stride=2, padding=1),
            ResNetBlock(feats[3]),

            # 4 x 4 x 4
            nn.ReLU(inplace=True),
            nn.Conv3d(feats[3], feats[4], 4, stride=2, padding=1),
            ResNetBlock(feats[4]),

            # 2 x 2 x 2
            nn.ReLU(inplace=True),
            nn.Conv3d(feats[4], feats[5], 2, stride=1),

            # Flatten to a vector
            nn.Flatten(1),

            # Relu out or not
            nn.ReLU(inplace=True) if relu_out else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    @property
    def embedding_dim(self):
        return self.feats[-1]


class ResNetDecoder(nn.Module):
    def __init__(self, feats: list, out: int = 1, relu_in: bool = False):
        super().__init__()

        self.feats = feats = [out, *feats[1:]]

        self.network = nn.Sequential(
            nn.ReLU() if relu_in else nn.Identity(),
            nn.ConvTranspose3d(feats[5], feats[4], 2),

            # 4 x 4 x 4
            ResNetBlock(feats[4]),
            nn.ConvTranspose3d(feats[4], feats[3], 4, stride=2, padding=1),

            # 8 x 8 x 8
            ResNetBlock(feats[3]),
            nn.ConvTranspose3d(feats[3], feats[2], 4, stride=2, padding=1),

            # 16 x 16 x 16
            ResNetBlock(feats[2]),
            nn.ConvTranspose3d(feats[2], feats[1], 4, stride=2, padding=1),

            # 32 x 32 x 32
            ResNetBlock(feats[1]),
            nn.ConvTranspose3d(feats[1], feats[1], 4, stride=2, padding=1),

            # 32 x 32 x 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(feats[1], feats[0], 7, stride=1, padding=3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 5:
            x = x.reshape(*x.size(), *(1 for _ in range(5 - x.ndim)))
        return self.network(x)
