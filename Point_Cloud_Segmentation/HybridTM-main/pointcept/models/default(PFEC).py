import torch
import torch.nn as nn
import torch_scatter

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        # 验证 input_dict 中的 coord（保留日志，便于调试）
        # if "coord" in input_dict:
        #     print(f"[Segmentor] input_dict 有 coord，shape: {input_dict['coord'].shape}")
        # else:
        #     print(f"[Segmentor] input_dict 无 coord！当前键: {list(input_dict.keys())}")
        #
        # # 验证 criteria 是否需要坐标（保留日志）
        # if hasattr(self.criteria, 'requires_coords'):
        #     print(f"[Segmentor] criteria.requires_coords: {self.criteria.requires_coords}")
        # else:
        #     print(f"[Segmentor] criteria 无 requires_coords 属性！")

        if "condition" in input_dict.keys():
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)

        # train 阶段：修改传参，第三个参数传 input_dict（完整字典）
        if self.training:
            if hasattr(self.criteria, 'requires_coords') and self.criteria.requires_coords:
                # 正确传参：seg_logits（预测）、input_dict["segment"]（标签）、input_dict（完整字典）
                loss = self.criteria(seg_logits, input_dict["segment"], input_dict)
            else:
                loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)

        # eval 阶段：同样修改传参，传递完整 input_dict
        elif "segment" in input_dict.keys():
            if hasattr(self.criteria, 'requires_coords') and self.criteria.requires_coords:
                # 正确传参：第三个参数是 input_dict（字典），而非 input_dict["coord"]（Tensor）
                loss = self.criteria(seg_logits, input_dict["segment"], input_dict)
            else:
                loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)

        # test 阶段：无标签，无需计算损失
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
            self,
            num_classes,
            backbone_out_channels,
            backbone=None,
            criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)  # 此处 criteria 是你修改后的 Criteria 实例

    def forward(self, input_dict):


        point = Point(input_dict)
        # if hasattr(point, "coord"):
        #     print(f"[Segmentor] Point.coord shape: {point.coord.shape}")
        # else:
        #     print(f"[Segmentor] Point.coord NOT found!")
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)

        # -------------------------- 核心修改部分 --------------------------
        # 1. 训练模式：传递 input_dict 给 Criteria，支持多损失（含 PLCCLoss）
        if self.training:
            # 直接传递 seg_logits + 标签 + 完整 input_dict（让 Criteria 内部判断是否需要坐标）
            loss = self.criteria(
                pred=seg_logits,
                target=input_dict["segment"],
                input_dict=input_dict  # 关键：传递完整输入字典，包含 coord 等数据
            )
            loss = torch.nan_to_num(loss)
            return dict(loss=loss)

        # 2. 验证模式：同训练模式，需计算损失用于评估
        elif "segment" in input_dict.keys():
            loss = self.criteria(
                pred=seg_logits,
                target=input_dict["segment"],
                input_dict=input_dict
            )
            loss = torch.nan_to_num(loss)
            return dict(loss=loss, seg_logits=seg_logits)

        # 3. 测试模式：无标签，不计算损失
        else:
            return dict(seg_logits=seg_logits)

@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
