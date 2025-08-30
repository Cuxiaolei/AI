_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 3  # bs: total bs in all gpus
mix_prob = 0.8
empty_cache = True
enable_amp = True  # 混合精度训练，加速训练且节省显存
sync_bn = True     # 同步BN，适合多GPU训练，提升稳定性
num_worker = 0     # 根据GPU内存调整，内存不足时设为0

# model settings: 适配改进版模型（电力塔→背景→电力线顺序）
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="OACNNs_PFEC",  # 改为改进版模型类名（对应之前的新模型代码）
        in_channels=9,          # 输入9通道（coord3 + color3 + normal3），与数据集一致
        num_classes=3,          # 3类：电力塔、背景、电力线
        embed_channels=64,      #  stem输出通道数，保持原配置
        enc_channels=[64, 64, 128, 256],  # Encoder各阶段输出通道，保持原配置
        groups=[4, 4, 8, 16],              # 分组卷积参数，适配高通道数时的效率
        enc_depth=[3, 3, 9, 8],            # Encoder各阶段BasicBlock数量，保持原配置
        dec_channels=[256, 256, 256, 256], # Decoder各阶段输出通道，保持原配置
        # 关键调整：point_grid_size改为[电力塔, 背景, 电力线]顺序，适配三类设施特性
        # 格式：[[stage1电力塔, stage1背景, stage1电力线], [stage2...], [stage3...], [stage4...]]
        # 数值逻辑：电力塔（中等尺度3×3×3）、背景（大尺度10×10×10）、电力线（小尺度1×1×5，z轴拉长保连续性）
        # 策略1: PFAS模块控制
        use_pfas=True,  # 是否启用PFAS
        pfas=dict(
            K=16,  # 近邻数
            grid_size_options=[  # 塔/背景/线的基础网格
                [[3, 3, 3], [10, 10, 10], [1, 1, 5]],
                [[3, 3, 3], [10, 10, 10], [1, 1, 5]],
                [[3, 3, 3], [10, 10, 10], [1, 1, 5]],
                [[3, 3, 3], [10, 10, 10], [1, 1, 5]]
            ],
            max_points_per_batch=20000,
            knn_batch_size=512
        ),
        # 策略2: CMPFE模块控制
        use_cmpfe=True,  # 是否启用CMPFE
        cmpfe=dict(
            proj_dim=6,  # 特征投影维度
            attn_hidden_dim=16  # 注意力隐藏层维度
        ),
        dec_depth=[2, 2, 2, 2],  # Decoder各阶段UpBlock数量，保持原配置
        enc_num_ref=[16, 16, 16, 16],  # 原模型参考点数量，保持兼容
        # 新增：PLCCL损失参数（对应改进点3，强化电力线连续性）

    ),
    # 关键调整：新增PLCCL损失，与交叉熵损失加权融合（多损失协同训练）
    # 损失函数配置
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        # dict(
        #     type="PLCCLoss",
        #     loss_weight=0.3,
        #     ignore_index=-1,
        #     temperature=0.1,
        #     gamma=0.5,
        #     max_batch_size=2000,  # 进一步减小批次
        #     pos_dist_thresh=1.0,
        #     neg_sample_ratio=1.5,  # 降低负样本比例
        #     max_neg_samples=1000,  # 限制负样本数量
        #     memory_safe_mode=False  # 启用内存安全模式
        # )
    ],
)

# 训练参数：保持原配置逻辑，适配改进模型复杂度
epoch = 100  # 总训练轮次，可根据收敛情况调整（改进模型可能需120-150轮）
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.02)  # AdamW优化器，抑制过拟合
scheduler = dict(
    type="OneCycleLR",  # OneCycle学习率调度，快速收敛且泛化性好
    max_lr=optimizer["lr"],  # 最大学习率（与optimizer.lr一致）
    pct_start=0.05,          # 前5%轮次线性升温到max_lr
    anneal_strategy="cos",   # 余弦退火降温
    div_factor=10.0,         # 初始lr = max_lr / 10
    final_div_factor=1000.0  # 最终lr = max_lr / 1000
)

# dataset settings: 明确类别映射，与模型顺序一致
dataset_type = "ScanNetDataset"  # 数据集类（若为自定义电力数据集，需改为对应类名）
data_root = "/root/autodl-tmp/data/data_scannet_tower"  # 数据路径，保持原配置

data = dict(
    num_classes=3,          # 3类，与模型一致
    ignore_index=-1,        # 忽略无效标签（如未标注点）
    # 关键调整：names与模型类别顺序严格一致（电力塔→背景→电力线）
    names=["power_tower", "background", "power_line"],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        # 数据增强：保留原配置，新增"NormalizeNormal"（法向量归一化，适配CMPFE模块）
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1/64, 1/64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1/64, 1/64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_min_coord=True,
            ),
            dict(type="SphereCrop", sample_rate=0.8, mode="random"),
            dict(type="SphereCrop", point_max=10000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),       # 颜色归一化（适配CMPFE颜色注意力）
            # dict(type="NormalizeNormal"),      # 新增：法向量归一化（CMPFE需稳定的法向量输入）
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            # 关键调整：feat_keys顺序与输入通道一致（coord3 + normal3 + color3）
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),  # segment为标签（电力塔0/背景1/电力线2）
                feat_keys=("coord", "normal", "color"),   # 输入特征顺序，与模型in_channels对应
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_min_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            # 新增：验证阶段裁剪点云（与训练保持相似规模）
            # dict(type='SphereCrop', point_max=5000, mode='center'),  # 限制最大5万个点
            # dict(type="NormalizeNormal"),  # 新增：法向量归一化
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "normal", "color"),  # 与训练一致的特征顺序
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
            # dict(type="NormalizeNormal"),  # 新增：法向量归一化
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                keys=("coord", "normal", "color"),  # 测试时特征顺序与训练一致
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "normal", "color"),
                ),
            ],
            # 测试增强：保持原配置（仅z轴0角度旋转，可根据需求增加其他角度）
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0,0,0], p=1)]
            ],
        ),
    ),
)

# # 可选：添加评估指标细化（重点关注电力线连续性与电力塔完整性）
# evaluation = dict(
#     metric=["mIoU", "Acc"],  # 基础指标：平均IoU、准确率
#     # 自定义指标：电力线断裂率、电力塔结构完整性（需数据集支持）
#     custom_metrics=[
#         dict(type="PowerLineBreakRate", threshold=2.0),  # 电力线片段长度<2m视为断裂
#         dict(type="PowerTowerCompleteness", iou_threshold=0.7)  # 电力塔IoU>0.7视为完整
#     ]
# )
