import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock1D(nn.Module):
    """1D ResNet基本块"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # 第一个卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个卷积
        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return


class Bottleneck1D(nn.Module):
    """1D ResNet瓶颈块"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck1D, self).__init__()
        width = out_channels

        self.conv1 = nn.Conv1d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width)

        self.conv2 = nn.Conv1d(width, width, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(width)

        self.conv3 = nn.Conv1d(width, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # 1x1卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3x3卷积
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1x1卷积
        out = self.conv3(out)
        out = self.bn3(out)

        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    """1D ResNet模型"""

    def __init__(self, block, layers, in_channels=1, num_classes=3,
                 feature_dim=512):
        super(ResNet1D, self).__init__()
        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 全局平均池化和分类头
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = 512 * block.expansion

        # 用于分类的头
        self.fc = nn.Linear(self.feature_dim, num_classes)

        # 初始化权重
        self._initialize_weights()

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features=False):
        """
        前向传播
        Args:
            x: 输入 [B, 1, L]
            return_features: 是否返回特征
        """
        # 初始层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局平均池化
        x = self.avgpool(x)  # [B, C, 1]
        features = torch.flatten(x, 1)  # [B, C]

        # 分类
        logits = self.fc(features)

        if return_features:
            return logits, features
        return logits


def resnet18_1d(in_channels=1, num_classes=3, feature_dim=512):
    """构建ResNet-18 1D模型"""
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2], in_channels, num_classes, feature_dim)


def resnet34_1d(in_channels=1, num_classes=3, feature_dim=512):
    """构建ResNet-34 1D模型"""
    return ResNet1D(BasicBlock1D, [3, 4, 6, 3], in_channels, num_classes, feature_dim)


def resnet50_1d(in_channels=1, num_classes=3, feature_dim=512):
    """构建ResNet-50 1D模型"""
    return ResNet1D(Bottleneck1D, [3, 4, 6, 3], in_channels, num_classes, feature_dim)


# 测试
if __name__ == "__main__":
    model = resnet18_1d()
    x = torch.randn(2, 1, 2048)
    logits, features = model(x, return_features=True)
    print(f"输入形状: {x.shape}")
    print(f"特征形状: {features.shape}")
    print(f"输出形状: {logits.shape}")