# -*- coding: utf-8 -*-
from __future__ import annotations

from .base import BaseDGClassifier, BaseDGConfig
from .erm import ERMClassifier, ERMConfig
from .mixstyle import MixStyleClassifier, MixStyleConfig
from .irm import IRMClassifier, IRMConfig
from .vrex import VRExClassifier, VRExConfig
from .groupdro import GroupDROClassifier, GroupDROConfig
from .mldg import MLDGClassifier, MLDGConfig
from .darm import DARMClassifier, DARMConfig
from .dpjdg import DPJDGClassifier, DPJDGConfig
from .mcpdg import MCPDGClassifier, MCPDGConfig


def _common_cfg(cfg: dict):
    model_cfg = cfg['model']
    return dict(
        num_classes=int(cfg['data']['num_classes']),
        freq_backbone_name=model_cfg.get('freq_backbone_name', 'resnet1d18'),
        freq_in_channels=int(model_cfg.get('freq_in_channels', 1)),
        freq_pretrained=bool(model_cfg.get('freq_pretrained', False)),
        classifier_dropout=float(model_cfg.get('classifier_dropout', 0.0)),
        backbone_kwargs=model_cfg.get('backbone_kwargs', {}),
    )

def build_method(cfg: dict):
    method_name = str(cfg['method']['name']).lower()
    model_cfg = cfg['model']
    common = _common_cfg(cfg)

    if method_name == 'erm':
        return ERMClassifier(ERMConfig(**common))
    if method_name == 'mixstyle':
        return MixStyleClassifier(MixStyleConfig(
            **common,
            mix_p=float(model_cfg.get('mix_p', 0.5)),
            mix_alpha=float(model_cfg.get('mix_alpha', 0.1)),
            mix_layer=str(model_cfg.get('mix_layer', 'layer1')),
        ))
    if method_name == 'irm':
        return IRMClassifier(IRMConfig(
            **common,
            irm_lambda=float(model_cfg.get('irm_lambda', 1.0)),
            irm_penalty_anneal_iters=int(model_cfg.get('irm_penalty_anneal_iters', 0)),
        ))
    if method_name == 'vrex':
        return VRExClassifier(VRExConfig(
            **common,
            vrex_lambda=float(model_cfg.get('vrex_lambda', 1.0)),
            vrex_penalty_anneal_iters=int(model_cfg.get('vrex_penalty_anneal_iters', 0)),
        ))
    if method_name == 'groupdro':
        return GroupDROClassifier(GroupDROConfig(
            **common,
            groupdro_eta=float(model_cfg.get('groupdro_eta', 0.01)),
        ))
    if method_name == 'mldg':
        return MLDGClassifier(MLDGConfig(
            **common,
            mldg_beta=float(model_cfg.get('mldg_beta', 1.0)),
            mldg_inner_lr=float(model_cfg.get('mldg_inner_lr', 1e-2)),
            mldg_meta_test_domains=int(model_cfg.get('mldg_meta_test_domains', 1)),
            mldg_first_order=bool(model_cfg.get('mldg_first_order', False)),
            mldg_split_seed=int(model_cfg.get('mldg_split_seed', 42)),
        ))
    if method_name == 'darm':
        return DARMClassifier(DARMConfig(
            **common,
            darm_iti_weight=float(model_cfg.get('darm_iti_weight', 0.1)),
            darm_ptp_weight=float(model_cfg.get('darm_ptp_weight', 0.1)),
            darm_margin=float(model_cfg.get('darm_margin', 1.0)),
            darm_feature_normalize=bool(model_cfg.get('darm_feature_normalize', True)),
        ))
    if method_name == 'dpjdg':
        return DPJDGClassifier(DPJDGConfig(
            **common,
            dpjdg_consistency_weight=float(model_cfg.get('dpjdg_consistency_weight', 0.5)),
            dpjdg_mmd_weight=float(model_cfg.get('dpjdg_mmd_weight', 0.5)),
            dpjdg_aug_noise_std=float(model_cfg.get('dpjdg_aug_noise_std', 0.02)),
            dpjdg_mask_ratio=float(model_cfg.get('dpjdg_mask_ratio', 0.05)),
            dpjdg_rbf_gamma=float(model_cfg.get('dpjdg_rbf_gamma', 1.0)),
        ))

    if method_name == 'mcpdg':
        return MCPDGClassifier(MCPDGConfig(
            **common,
            cond_dim=int(model_cfg.get('cond_dim', 3)),
            cond_hidden_dim=int(model_cfg.get('cond_hidden_dim', 64)),
            proto_hidden_dim=int(model_cfg.get('proto_hidden_dim', 256)),

            use_linear_head=bool(model_cfg.get('use_linear_head', True)),
            use_dynamic_proto=bool(model_cfg.get('use_dynamic_proto', True)),
            use_proto_cls=bool(model_cfg.get('use_proto_cls', True)),
            use_align_loss=bool(model_cfg.get('use_align_loss', True)),
            use_pcl_loss=bool(model_cfg.get('use_pcl_loss', True)),

            proto_residual_alpha=float(model_cfg.get('proto_residual_alpha', 0.2)),
            proto_cls_weight=float(model_cfg.get('proto_cls_weight', 0.5)),
            eval_proto_weight=float(model_cfg.get('eval_proto_weight', 0.5)),
            align_weight=float(model_cfg.get('align_weight', 1.0)),
            pcl_weight=float(model_cfg.get('pcl_weight', 0.1)),
            pcl_temperature=float(model_cfg.get('pcl_temperature', 0.1)),
            imbalance_power=float(model_cfg.get('imbalance_power', 0.5)),

            # meta split
            meta_test_domains=int(model_cfg.get('meta_test_domains')),
            meta_randomize=bool(model_cfg.get('meta_randomize', True)),
            meta_split_seed=int(model_cfg.get('meta_split_seed'), 42),
            meta_debug=bool(model_cfg.get('meta_debug')),
            meta_debug_max_steps = int(model_cfg.get('meta_debug_max_steps'), 20),
        ))
    raise ValueError(f'Unsupported method: {method_name}')

__all__ = [
    'BaseDGClassifier', 'BaseDGConfig',
    'ERMClassifier', 'ERMConfig',
    'MixStyleClassifier', 'MixStyleConfig',
    'IRMClassifier', 'IRMConfig',
    'VRExClassifier', 'VRExConfig',
    'GroupDROClassifier', 'GroupDROConfig',
    'MLDGClassifier', 'MLDGConfig',
    'DARMClassifier', 'DARMConfig',
    'DPJDGClassifier', 'DPJDGConfig',
    'MCPDGClassifier', 'MCPDGConfig',
    'build_method',
]