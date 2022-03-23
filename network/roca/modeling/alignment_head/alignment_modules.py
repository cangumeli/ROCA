from typing import Optional, Type

import torch
import torch.nn as nn
import detectron2.layers as L


class SharedMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: Optional[int] = None,
        hidden_size: int = 256,
        num_hiddens: int = 1,
        activation: Type[nn.Module] = nn.ReLU,
        output_activation: Type[nn.Module] = nn.Identity
    ):
        super().__init__()

        assert num_hiddens > 0
        self.hiddens = nn.Sequential(*[
            L.Conv2d(
                in_channels=input_size if i == 0 else hidden_size,
                out_channels=hidden_size,
                kernel_size=(1, 1),
                activation=activation(inplace=True)
            ) for i in range(num_hiddens)
        ])

        if output_size is not None:
            self.output = L.Conv2d(
                in_channels=hidden_size,
                out_channels=output_size,
                kernel_size=(1, 1)
            )
            self.out_channels = output_size
        else:
            self.output = nn.Identity()
            self.out_channels = hidden_size

        self.output_activation = output_activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_activation(self.output(self.hiddens(x)))


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: Optional[int] = None,
        hidden_size: int = 256,
        num_hiddens: int = 1,
        activation: Type[nn.Module] = nn.ReLU,
        output_activation: Type[nn.Module] = nn.Identity
    ):
        super().__init__()

        assert num_hiddens > 0
        self.hiddens = nn.Sequential(*[
            L.Linear(
                in_features=input_size if i == 0 else hidden_size,
                out_features=hidden_size
            ) if i % 2 == 0 else activation(inplace=True)
            for i in range(2 * num_hiddens)
        ])

        if output_size is not None:
            self.output = L.Linear(
                in_features=hidden_size,
                out_features=output_size,
            )
            self.out_features = output_size
        else:
            self.output = nn.Identity()
            self.out_features = hidden_size
        self.output_activation = output_activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_activation(self.output(self.hiddens(x)))


class Aggregator(nn.Module):
    def __init__(
        self,
        shared_net: nn.Module = nn.Identity(),
        global_net: nn.Module = nn.Identity()
    ):
        super().__init__()
        self.shared_net = shared_net
        self.global_pool = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(1)
        )
        self.global_net = global_net

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        features = self.shared_net(features)
        if features.numel() > 0:
            features = self.global_pool(features * mask)
        else:
            features = features.view(0, features.size(1))
        return self.global_net(features)
