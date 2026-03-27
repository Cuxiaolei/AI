import torch
import torch.nn as nn
from dataclasses import dataclass
from src.backbones.resnet1d import resnet1d18, resnet1d34, resnet1d50



@dataclass
class BaseDGConfig:
    """配置类（仅保留freq相关）"""
    num_classes: int = 2
    freq_backbone_name: str = "resnet1d18"
    freq_in_channels: int = 1
    freq_pretrained: bool = False
    classifier_dropout: float = 0.1
    backbone_kwargs: dict = None

    def __post_init__(self):
        if self.backbone_kwargs is None:
            self.backbone_kwargs = {}


class LinearClassifier(nn.Module):
    """简单分类头"""
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(self.dropout(x))


class BaseDGClassifier(nn.Module):
    """仅支持频域的DG分类基类"""
    def __init__(self, config: BaseDGConfig):
        super().__init__()
        self.config = config
        self.num_classes = int(config.num_classes)

        self.freq_backbone = self._build_backbone(
            backbone_name=config.freq_backbone_name,
            in_channels=config.freq_in_channels,
            pretrained=config.freq_pretrained,
            **config.backbone_kwargs
        )

        self.feat_dim = int(self.freq_backbone.out_dim)

        self.classifier = LinearClassifier(
            in_dim=self.feat_dim,
            num_classes=self.num_classes,
            dropout=config.classifier_dropout
        )

    def _build_backbone(self, backbone_name, in_channels=1, pretrained=False, **kwargs):
        # 根据名称构建1D backbone
        backbones = {
            "resnet1d18": resnet1d18,
            "resnet1d34": resnet1d34,
            "resnet1d50": resnet1d50
        }
        return backbones[backbone_name](in_channels=in_channels, **kwargs)

    def extract_freq_feature(self, x_freq: torch.Tensor) -> torch.Tensor:
        # 提取频域特征
        return self.freq_backbone(x_freq)

    def extract_features(self, batch):
        # 对外接口：仅提取频域特征
        x_freq = batch["x_freq"]
        feat = self.extract_freq_feature(x_freq)
        return {"feature": feat}

    def forward_logits(self, feature):
        # 特征 -> 分类结果
        return self.classifier(feature)

    def forward(self, batch):
        # 完整前向
        feats = self.extract_features(batch)
        logits = self.forward_logits(feats["feature"])
        return {**feats, "logits": logits}

    def compute_loss(self, outputs, batch):
        # 子类实现损失
        raise NotImplementedError