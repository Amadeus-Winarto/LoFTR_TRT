import torch.nn as nn
import torch.nn.functional as F
from .common import get_activation


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
    )


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class ConvNextBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, activation="gelu"):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv1x1(planes, 4 * planes)
        self.conv3 = conv1x1(4 * planes, planes)
        self.ln = nn.LayerNorm(planes)
        self.activation = get_activation(activation)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride), nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.ln(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        y = self.conv3(self.activation(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + y


class ConvNeXtFPN_8_2(nn.Module):
    """
    ConvNext+FPN, output resolution are 1/8 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self, config, activation="relu"):
        super().__init__()
        # Config
        block = ConvNextBlock
        initial_dim = config["initial_dim"]
        block_dims = config["block_dims"]

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(
            1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.activation = get_activation(activation)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8

        # 3. FPN upsample
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            get_activation("leaky_relu"),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            get_activation("leaky_relu"),
            conv3x3(block_dims[1], block_dims[0]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) and activation in ["relu", "leaky_relu"]:
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity=activation
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ConvNext Backbone
        x0 = self.activation(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8

        # FPN
        x3_out = self.layer3_outconv(x3)

        x2_out = self.layer2_outconv(x2)

        x1_out = self.layer1_outconv(x1)

        return self.complete_result(x3_out, x2_out, x1_out)

    def complete_result(self, x3_out, x2_out, x1_out):
        x3_out_2x = F.interpolate(
            x3_out, scale_factor=[2.0, 2.0], mode="bilinear", align_corners=True
        )
        x2_out = self.layer2_outconv2(x2_out + x3_out_2x)

        x2_out_2x = F.interpolate(
            x2_out, scale_factor=[2.0, 2.0], mode="bilinear", align_corners=True
        )
        x1_out = self.layer1_outconv2(x1_out + x2_out_2x)

        return x3_out, x1_out


class ConvNeXtFPN_16_4(nn.Module):
    """
    ConvNext+FPN, output resolution are 1/16 and 1/4.
    Each block has 2 layers.
    """

    def __init__(self, config, activation="relu"):
        super().__init__()
        # Config
        block = ConvNextBlock
        initial_dim = config["initial_dim"]
        block_dims = config["block_dims"]

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(
            1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.activation = get_activation(activation)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8
        self.layer4 = self._make_layer(block, block_dims[3], stride=2)  # 1/16

        # 3. FPN upsample
        self.layer4_outconv = conv1x1(block_dims[3], block_dims[3])
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[3])
        self.layer3_outconv2 = nn.Sequential(
            conv3x3(block_dims[3], block_dims[3]),
            nn.BatchNorm2d(block_dims[3]),
            get_activation("leaky_relu"),
            conv3x3(block_dims[3], block_dims[2]),
        )

        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            get_activation("leaky_relu"),
            conv3x3(block_dims[2], block_dims[1]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) and activation in ["relu", "leaky_relu"]:
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity=activation
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ConvNext Backbone
        x0 = self.activation(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8
        x4 = self.layer4(x3)  # 1/16

        # FPN
        x4_out = self.layer4_outconv(x4)

        x4_out_2x = F.interpolate(
            x4_out, scale_factor=2.0, mode="bilinear", align_corners=True
        )
        x3_out = self.layer3_outconv(x3)
        x3_out = self.layer3_outconv2(x3_out + x4_out_2x)

        x3_out_2x = F.interpolate(
            x3_out, scale_factor=2.0, mode="bilinear", align_corners=True
        )
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out + x3_out_2x)

        return x4_out, x2_out