# -*- coding: utf-8 -*-
from __future__ import annotations

from .base import BaseDGClassifier, BaseDGConfig
from .erm import ERMClassifier, ERMConfig
from .vrex import VRExClassifier, VRExConfig
from .dfdn import DFDNClassifier, DFDNConfig
from .mldg import MLDGClassifier, MLDGConfig
from .darm import DARMClassifier, DARMConfig
from .sdagn import SDAGNClassifier, SDAGNConfig
from .dpjdg import DPJDGClassifier, DPJDGConfig
from .masfd import MASFDClassifier, MASFDConfig
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
    if method_name == 'mldg':
        return MLDGClassifier(MLDGConfig(
            **common,
            mldg_beta=float(model_cfg.get('mldg_beta', 1.0)),
            mldg_inner_lr=float(model_cfg.get('mldg_inner_lr', 1e-2)),
            mldg_meta_test_domains=int(model_cfg.get('mldg_meta_test_domains', 1)),
            mldg_first_order=bool(model_cfg.get('mldg_first_order', False)),
        ))
    if method_name == 'vrex':
        return VRExClassifier(VRExConfig(
            **common,
            vrex_lambda=float(model_cfg.get('vrex_lambda', 1.0)),
            vrex_penalty_anneal_iters=int(model_cfg.get('vrex_penalty_anneal_iters', 0)),
        ))

    if method_name == 'dfdn':
        return DFDNClassifier(DFDNConfig(
            **common,
            num_domains=int(cfg['data']['num_domains']),
            decouple_hidden_dim=int(model_cfg.get('decouple_hidden_dim', 512)),
            fault_feat_dim=int(model_cfg.get('fault_feat_dim', 256)),
            domain_feat_dim=int(model_cfg.get('domain_feat_dim', 256)),
            integrator_hidden_dim=int(model_cfg.get('integrator_hidden_dim', 256)),
            domain_disc_hidden_dim=int(model_cfg.get('domain_disc_hidden_dim', 256)),
            disc_dropout=float(model_cfg.get('disc_dropout', 0.1)),
            lambda_fault_cls=float(model_cfg.get('lambda_fault_cls', 1.0)),
            lambda_aux_cls=float(model_cfg.get('lambda_aux_cls', 0.5)),
            lambda_domain_cls=float(model_cfg.get('lambda_domain_cls', 1.0)),
            lambda_adv_domain=float(model_cfg.get('lambda_adv_domain', 0.1)),
            lambda_orth=float(model_cfg.get('lambda_orth', 0.05)),
            lambda_fused_align=float(model_cfg.get('lambda_fused_align', 0.0)),
            grl_lambda=float(model_cfg.get('grl_lambda', 1.0)),
            use_grl_schedule=bool(model_cfg.get('use_grl_schedule', True)),
            grl_warmup_steps=int(model_cfg.get('grl_warmup_steps', 1000)),
        ))


    if method_name == 'darm':
        return DARMClassifier(DARMConfig(
            **common,
            darm_iti_weight=float(model_cfg.get('darm_iti_weight', 0.1)),
            darm_ptp_weight=float(model_cfg.get('darm_ptp_weight', 0.1)),
            darm_margin=float(model_cfg.get('darm_margin', 1.0)),
            darm_feature_normalize=bool(model_cfg.get('darm_feature_normalize', True)),
        ))

    if method_name == 'sdagn':
        return SDAGNClassifier(SDAGNConfig(
            **common,
            sdagn_mixup_alpha=float(model_cfg.get('sdagn_mixup_alpha', 0.4)),
            sdagn_mixup_mode=str(model_cfg.get('sdagn_mixup_mode', 'beta')),
            sdagn_cls_weight=float(model_cfg.get('sdagn_cls_weight', 1.0)),
            sdagn_aug_cls_weight=float(model_cfg.get('sdagn_aug_cls_weight', 1.0)),
            sdagn_semantic_weight=float(model_cfg.get('sdagn_semantic_weight', 1.0)),
            sdagn_triplet_weight=float(model_cfg.get('sdagn_triplet_weight', 0.1)),
            sdagn_triplet_margin=float(model_cfg.get('sdagn_triplet_margin', 1.0)),
            sdagn_mmd_gamma=float(model_cfg.get('sdagn_mmd_gamma', 1.0)),
            sdagn_mmd_num_kernels=int(model_cfg.get('sdagn_mmd_num_kernels', 1)),
            sdagn_normalize_triplet_feat=bool(model_cfg.get('sdagn_normalize_triplet_feat', True)),
            sdagn_max_aug_per_class=int(model_cfg.get('sdagn_max_aug_per_class', 64)),
            sdagn_balance_to_max=bool(model_cfg.get('sdagn_balance_to_max', True)),
            sdagn_min_samples_per_class_to_mix=int(model_cfg.get('sdagn_min_samples_per_class_to_mix', 2)),
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

    if method_name == 'masfd':
        return MASFDClassifier(MASFDConfig(
            **common,
            num_domains=int(cfg['data'].get('num_domains', 0)),
            num_modes=int(model_cfg.get('num_modes', 3)),
            mode_channels=int(model_cfg.get('mode_channels', 32)),
            mode_feat_dim=int(model_cfg.get('mode_feat_dim', 128)),
            fusion_hidden_dim=int(model_cfg.get('fusion_hidden_dim', 128)),
            cls_weight=float(model_cfg.get('cls_weight', 1.0)),
            aux_cls_weight=float(model_cfg.get('aux_cls_weight', 0.5)),
            domain_spec_weight=float(model_cfg.get('domain_spec_weight', 0.5)),
            domain_inv_weight=float(model_cfg.get('domain_inv_weight', 0.2)),
            ortho_weight=float(model_cfg.get('ortho_weight', 0.05)),
            meta_weight=float(model_cfg.get('meta_weight', 0.5)),
            grl_lambda=float(model_cfg.get('grl_lambda', 1.0)),
            eval_aux_weight=float(model_cfg.get('eval_aux_weight', 0.25)),
            meta_min_samples=int(model_cfg.get('meta_min_samples', 2)),
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


            meta_split_seed=int(model_cfg.get('meta_split_seed', 42)),
            meta_debug=bool(model_cfg.get('meta_debug', False)),
            meta_debug_max_steps = int(model_cfg.get('meta_debug_max_steps', 20)),
        ))
    raise ValueError(f'Unsupported method: {method_name}')

__all__ = [
    'BaseDGClassifier', 'BaseDGConfig',
    'ERMClassifier', 'ERMConfig',
    'VRExClassifier', 'VRExConfig',
    'DFDNClassifier', 'DFDNConfig',
    'MLDGClassifier', 'MLDGConfig',
    'DARMClassifier', 'DARMConfig',
    'SDAGNClassifier', 'SDAGNConfig',
    'DPJDGClassifier', 'DPJDGConfig',
    'MASFDClassifier', 'MASFDConfig',
    'MCPDGClassifier', 'MCPDGConfig',
    'build_method',
]