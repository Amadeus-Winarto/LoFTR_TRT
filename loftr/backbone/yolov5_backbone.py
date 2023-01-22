from typing import Tuple
import torch
import torch.nn as nn

from .common import SPPF, Conv, C3, get_activation


class CSPNet(nn.Module):
    def __init__(
        self,
        initial_dim,
        block_dims=[128, 196, 256],
        num_layers=[3, 6, 9],
        activation="silu",
    ) -> None:
        super().__init__()
        block_dims.append(block_dims[-1] * 2)  # P5/32

        act = get_activation(activation)
        self.layer1 = Conv(c1=1, c2=initial_dim, k=6, s=2, p=2, act=act)  # 1/2
        self.layer2 = Conv(c1=initial_dim, c2=block_dims[0], k=3, s=2, act=act)  # 1/4

        # P3/8
        layer3s = [
            C3(c1=block_dims[0], c2=block_dims[0], n=3, act=act)
            for _ in range(num_layers[0])
        ]
        self.layer3 = nn.Sequential(*layer3s)

        # P4/16
        layer4s = [Conv(c1=block_dims[0], c2=block_dims[1], k=3, s=2, act=act)] + [
            C3(c1=block_dims[1], c2=block_dims[1], n=3, act=act)
            for _ in range(num_layers[1])
        ]
        self.layer4 = nn.Sequential(*layer4s)

        # P5/32
        layer5s = [Conv(c1=block_dims[1], c2=block_dims[2], k=3, s=2, act=act)] + [
            C3(c1=block_dims[2], c2=block_dims[2], n=3, act=act)
            for _ in range(num_layers[2])
        ]
        self.layer5 = nn.Sequential(*layer5s)

        # P6/32
        self.layer6 = nn.Sequential(
            Conv(c1=block_dims[2], c2=block_dims[3], k=3, s=2, act=act),
            C3(c1=block_dims[3], c2=block_dims[3], n=3, act=act),
            C3(c1=block_dims[3], c2=block_dims[3], n=3, act=act),
            C3(c1=block_dims[3], c2=block_dims[3], n=3, act=act),
            SPPF(c1=block_dims[3], c2=block_dims[3], k=5, act=act),  # 1/32
        )

    def forward(self, image: torch.Tensor):
        x = self.layer1(image)
        x = self.layer2(x)
        P3 = self.layer3(x)
        P4 = self.layer4(P3)
        P5 = self.layer5(P4)
        P6 = self.layer6(P5)
        return P3, P4, P5, P6


class PANet(nn.Module):
    def __init__(
        self, initial_dim, block_dims=[256, 196, 128], activation="silu"
    ) -> None:
        super().__init__()

        assert len(block_dims) >= 3

        act = get_activation(activation)
        self.layer1 = nn.Sequential(
            Conv(c1=initial_dim, c2=block_dims[0], k=1, s=1, act=act),
            nn.Upsample(scale_factor=2, mode="nearest"),
            C3(c1=block_dims[0], c2=block_dims[0], n=3, shortcut=False, act=act),
            C3(c1=block_dims[0], c2=block_dims[0], n=3, shortcut=False, act=act),
            C3(c1=block_dims[0], c2=block_dims[0], n=3, shortcut=False, act=act),
        )

        self.layer2 = nn.Sequential(
            Conv(c1=block_dims[0] * 2, c2=block_dims[1], k=1, s=1, act=act),
            nn.Upsample(scale_factor=2, mode="nearest"),
            C3(c1=block_dims[1], c2=block_dims[1], n=3, shortcut=False, act=act),
            C3(c1=block_dims[1], c2=block_dims[1], n=3, shortcut=False, act=act),
            C3(c1=block_dims[1], c2=block_dims[1], n=3, shortcut=False, act=act),
        )

        self.layer3 = nn.Sequential(
            Conv(c1=block_dims[1] * 2, c2=block_dims[2], k=1, s=1, act=act),
            nn.Upsample(scale_factor=2, mode="nearest"),
            C3(c1=block_dims[2], c2=block_dims[2], n=3, shortcut=False, act=act),
            C3(c1=block_dims[2], c2=block_dims[2], n=3, shortcut=False, act=act),
            C3(c1=block_dims[2], c2=block_dims[2], n=3, shortcut=False, act=act),
        )

        self.layer3 = nn.Sequential(
            Conv(c1=block_dims[1] * 2, c2=block_dims[2], k=1, s=1, act=act),
            nn.Upsample(scale_factor=2, mode="nearest"),
            C3(c1=block_dims[2], c2=block_dims[2], n=3, shortcut=False, act=act),
            C3(c1=block_dims[2], c2=block_dims[2], n=3, shortcut=False, act=act),
            C3(c1=block_dims[2], c2=block_dims[2], n=3, shortcut=False, act=act),
        )
        self.high_res = False

        if len(block_dims) > 3:
            self.high_res = True
            self.layer4 = nn.Sequential(
                Conv(c1=block_dims[2] * 2, c2=block_dims[3], k=1, s=1, act=act),
                nn.Upsample(scale_factor=2, mode="nearest"),
                C3(c1=block_dims[3], c2=block_dims[3], n=3, shortcut=False, act=act),
                C3(c1=block_dims[3], c2=block_dims[3], n=3, shortcut=False, act=act),
                C3(c1=block_dims[3], c2=block_dims[3], n=3, shortcut=False, act=act),
            )

    def forward(
        self, P3: torch.Tensor, P4: torch.Tensor, P5: torch.Tensor, P6: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _P5 = self.layer1(P6)
        _P4 = self.layer2(torch.cat([P5, _P5], 1))
        _P3 = self.layer3(torch.cat([P4, _P4], 1))

        if not self.high_res:
            return _P3, _P5
        else:
            _P2 = self.layer4(torch.cat([_P3, P3], 1))
            return _P2, _P4


class CSPNet_PANet_8_2(nn.Module):
    def __init__(self, config, activation="silu") -> None:
        super().__init__()
        self.backbone = CSPNet(
            config["initial_dim"], config["block_dims"], activation=activation
        )
        self.neck = PANet(512, [256, 196, 128, 64], activation=activation)

    def forward(self, image: torch.Tensor):
        P3, P4, P5, P6 = self.backbone(image)
        P2, P4 = self.neck(P3, P4, P5, P6)
        return P4, P2


class CSPNet_PANet_16_4(nn.Module):
    def __init__(self, config, activation="silu") -> None:
        super().__init__()
        self.backbone = CSPNet(128, [128, 196, 256, 512], activation=activation)
        self.neck = PANet(512, [256, 196, 128], activation=activation)

    def forward(self, image: torch.Tensor):
        P3, P4, P5, P6 = self.backbone(image)
        P3, P5 = self.neck(P3, P4, P5, P6)
        return P5, P3
