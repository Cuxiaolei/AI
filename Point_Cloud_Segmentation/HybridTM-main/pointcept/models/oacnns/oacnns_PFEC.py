from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from timm.models.layers import trunc_normal_
from ..builder import MODELS
from ..utils import offset2batch
from torch_geometric.nn.pool import voxel_grid
from torch_geometric.utils import scatter
import logging
import time

# 配置调试日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# 策略1: 电力设施自适应尺度模块（PFAS）- 优化版
class PFASModule(nn.Module):
    def __init__(self, in_channels, grid_size_options, K=16):
        super().__init__()
        self.grid_size_options = grid_size_options  # [电力塔, 背景, 电力线]
        self.K = K  # 保持小近邻数
        self.feature_judge = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 输出三类概率
        )

    def forward(self, feat, coord, batch):
        start_time = time.time()
        logger.debug(
            f"PFAS forward start - feat shape: {feat.shape}, coord shape: {coord.shape}, batch shape: {batch.shape}")
        N = coord.shape[0]
        B = batch.max().item() + 1 if batch.numel() > 0 else 0
        if B == 0:
            logger.debug("PFAS: Empty batch, returning default grid")
            return torch.ones_like(coord) * self.grid_size_options[1][0]

        # -------------------------- 核心优化：分批次KNN（无[N,N]矩阵） --------------------------
        # 1. 按batch分组处理，避免跨样本计算
        dynamic_grid_sizes = torch.zeros_like(coord, device=coord.device)
        for b in range(B):
            # 提取当前batch的点（局部化，避免全量操作）
            batch_mask = (batch == b)
            coord_b = coord[batch_mask]  # [N_b, 3]，N_b是当前batch的点数
            feat_b = feat[batch_mask]    # [N_b, C]
            N_b = coord_b.shape[0]
            if N_b < self.K:  # 点数不足K时，直接用背景网格
                dynamic_grid_sizes[batch_mask] = torch.tensor(self.grid_size_options[1], device=coord.device)
                continue

            # 2. 分批次KNN（避免一次性计算全量距离）
            # 批次大小根据GPU内存调整（如1024，可根据实际情况修改）
            batch_size_knn = 1024
            idx_b = torch.zeros((N_b, self.K), dtype=torch.long, device=coord.device)
            for i in range(0, N_b, batch_size_knn):
                # 当前批次的坐标：[batch_size_knn, 3]
                coord_batch = coord_b[i:i+batch_size_knn]
                # 计算当前批次与所有batch内点的距离：[batch_size_knn, N_b]（而非[N,N]）
                dist_batch = torch.cdist(coord_batch, coord_b)
                # 屏蔽自身（对角线）
                dist_batch[:, i:i+batch_size_knn] = torch.inf  # 只屏蔽当前批次的自身
                # 取Top-K近邻索引
                _, idx_batch = torch.topk(dist_batch, self.K, largest=False)
                idx_b[i:i+batch_size_knn] = idx_batch

            # -------------------------- 后续逻辑保持原功能（局部化处理） --------------------------
            # 3. 近邻特征计算（基于局部索引）
            neighbor_coords = coord_b[idx_b]  # [N_b, K, 3]
            neighbor_centered = neighbor_coords - neighbor_coords.mean(dim=1, keepdim=True)  # 中心化

            # 4. 批量协方差与PCA（保持向量化，无循环）
            cov = torch.einsum('nkd,nke->nde', neighbor_centered, neighbor_centered) / (self.K - 1)  # [N_b, 3, 3]
            U, S, V = torch.svd(cov.float())  # 用float降低精度，省内存
            S = S.to(feat.dtype)
            S_norm = S / (S.sum(dim=1, keepdim=True) + 1e-6)  # [N_b, 3]

            # 线性度计算
            linearity = S_norm[:, 0].unsqueeze(1) - (S_norm[:, 1] + S_norm[:, 2]).unsqueeze(1)  # [N_b, 1]

            # 5. 密度计算（基于局部距离）
            # 提取当前batch内的近邻距离（避免全量dist矩阵）
            neighbor_dists = torch.gather(
                torch.cdist(coord_b, coord_b),  # 仅当前batch的距离矩阵：[N_b, N_b]（比全量小B倍）
                dim=1,
                index=idx_b
            )  # [N_b, K]
            mean_dist = neighbor_dists.mean(dim=1, keepdim=True)
            density = 1.0 / (mean_dist + 1e-6)  # [N_b, 1]

            # 6. 特征判断与网格生成（局部化）
            feat_logits = self.feature_judge(feat_b)  # [N_b, 3]
            feat_probs = F.softmax(feat_logits, dim=1)  # [N_b, 3]

            tower_prob = (density * 2.0 + feat_probs[:, 0:1]) / 3.0
            background_prob = (torch.maximum(1.0 - linearity, 1.0 - density) + feat_probs[:, 1:2]) / 3.0
            line_prob = (linearity * 2.0 + feat_probs[:, 2:3]) / 3.0

            # 生成当前batch的动态网格
            for i_axis in range(3):  # x/y/z轴
                tower_grid = self.grid_size_options[0][i_axis]
                background_grid = self.grid_size_options[1][i_axis]
                line_grid = self.grid_size_options[2][i_axis] if i_axis < 2 else self.grid_size_options[2][i_axis] * 5
                dynamic_grid_sizes[batch_mask, i_axis] = (
                    tower_prob[:, 0] * tower_grid +
                    background_prob[:, 0] * background_grid +
                    line_prob[:, 0] * line_grid + 1e-6
                )

        logger.debug(
            f"PFAS forward complete - total time: {time.time() - start_time:.4f}s - output shape: {dynamic_grid_sizes.shape}")
        return dynamic_grid_sizes



# 策略2: 跨模态电力特征增强（CMPFE）- 保持原逻辑，并行高效
class CMPFEModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.feature_projection = nn.Sequential(
            nn.Linear(in_channels, 9),
            nn.BatchNorm1d(9),
            nn.ReLU()
        )
        self.color_attention = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Sigmoid()
        )
        self.normal_attention = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Sigmoid()
        )
        self.feature_fusion = nn.Sequential(
            nn.Linear(9, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels)
        )
        self.semantic_attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        start_time = time.time()
        logger.debug(f"CMPFE forward start - input shape: {x.shape}")
        # 所有操作均为并行批量计算，无优化空间（已最优）
        projected_feat = self.feature_projection(x)
        coord_feat = projected_feat[:, :3]
        color_feat = projected_feat[:, 3:6]
        normal_feat = projected_feat[:, 6:9]

        enhanced_color = color_feat * self.color_attention(color_feat)
        enhanced_normal = normal_feat * self.normal_attention(normal_feat)

        enhanced_feat = torch.cat([coord_feat, enhanced_color, enhanced_normal], dim=1)
        fused_feat = self.feature_fusion(enhanced_feat)

        sem_att = self.semantic_attention(fused_feat)
        final_feat = fused_feat * sem_att + x * (1 - sem_att)

        logger.debug(
            f"CMPFE forward complete - total time: {time.time() - start_time:.4f}s - output shape: {final_feat.shape}")
        return final_feat


class BasicBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            embed_channels,
            norm_fn=None,
            indice_key=None,
            depth=4,
            groups=None,
            grid_size=None,
            bias=False,
    ):
        super().__init__()
        assert embed_channels % groups == 0
        self.groups = groups
        self.embed_channels = embed_channels
        self.proj = nn.ModuleList()
        self.grid_size = grid_size
        self.weight = nn.ModuleList()
        self.l_w = nn.ModuleList()
        self.cmpfe = CMPFEModule(embed_channels)  # 调用优化后的CMPFE（无变化）
        self.proj.append(
            nn.Sequential(
                nn.Linear(embed_channels, embed_channels, bias=False),
                norm_fn(embed_channels),
                nn.ReLU(),
            )
        )
        for _ in range(depth - 1):
            self.proj.append(
                nn.Sequential(
                    nn.Linear(embed_channels, embed_channels, bias=False),
                    norm_fn(embed_channels),
                    nn.ReLU(),
                )
            )
            self.l_w.append(
                nn.Sequential(
                    nn.Linear(embed_channels, embed_channels, bias=False),
                    norm_fn(embed_channels),
                    nn.ReLU(),
                )
            )
            self.weight.append(nn.Linear(embed_channels, embed_channels, bias=False))
        self.adaptive = nn.Linear(embed_channels, depth - 1, bias=False)
        self.fuse = nn.Sequential(
            nn.Linear(embed_channels * 2, embed_channels, bias=False),
            norm_fn(embed_channels),
            nn.ReLU(),
        )
        self.voxel_block = spconv.SparseSequential(
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                indice_key=indice_key,
                bias=bias,
            ),
            norm_fn(embed_channels),
            nn.ReLU(),
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                indice_key=indice_key,
                bias=bias,
            ),
            norm_fn(embed_channels),
        )
        self.act = nn.ReLU()
        self.pfas = PFASModule(embed_channels, grid_size)  # 调用优化后的PFAS

    def forward(self, x, clusters=None):
        block_id = id(self) % 1000
        start_time = time.time()
        logger.debug(
            f"BasicBlock {block_id} forward start - features shape: {x.features.shape}, spatial shape: {x.spatial_shape}")

        # 调用优化后的CMPFE和PFAS
        enhanced_feat = self.cmpfe(x.features)
        x = x.replace_feature(enhanced_feat)
        feat = x.features

        coord = x.indices[:, 1:].float()
        batch = x.indices[:, 0]
        dynamic_grid_sizes = self.pfas(feat, coord, batch)  # 优化后的PFAS

        if dynamic_grid_sizes.numel() == 0:
            logger.debug(f"BasicBlock {block_id} empty input, returning early")
            return x

        # 网格生成逻辑简化（保留原逻辑，无优化）
        grid_mean = dynamic_grid_sizes.mean(dim=1)
        representative_grids = []
        if dynamic_grid_sizes.shape[0] > 200:
            representative_grids.append(torch.mean(dynamic_grid_sizes[torch.argsort(grid_mean)[100:200]], dim=0))
        else:
            representative_grids.append(grid_mean.mean() * 0.8)
        if dynamic_grid_sizes.shape[0] > 100:
            representative_grids.append(
                torch.mean(dynamic_grid_sizes[torch.argsort(grid_mean, descending=True)[:100]], dim=0))
            representative_grids.append(torch.mean(dynamic_grid_sizes[torch.argsort(grid_mean)[:100]], dim=0))
        else:
            representative_grids.append(grid_mean.mean() * 1.2)
            representative_grids.append(grid_mean.mean() * 0.5)

        # 聚类和融合逻辑（保留原逻辑，无优化）
        clusters = []
        for i, grid_size in enumerate(representative_grids):
            cluster = voxel_grid(pos=coord, size=torch.clamp(grid_size, min=1e-6).tolist(), batch=batch)
            _, cluster = torch.unique(cluster, return_inverse=True)
            clusters.append(cluster)

        feats = []
        valid_depth = min(len(self.l_w), len(clusters))
        for i in range(valid_depth):
            cluster = clusters[i]
            pw = self.l_w[i](feat)
            pw = pw - scatter(pw, cluster, reduce="mean")[cluster]
            pw = self.weight[i](pw)
            pw = torch.exp(pw - pw.max())
            pw = pw / (scatter(pw, cluster, reduce="sum", dim=0)[cluster] + 1e-6)
            pfeat = self.proj[i](feat) * pw
            pfeat = scatter(pfeat, cluster, reduce="sum")[cluster]
            feats.append(pfeat)

        if not feats:
            logger.debug(f"BasicBlock {block_id} no features for fusion, returning early")
            return x

        adp = self.adaptive(feat)
        adp = torch.softmax(adp, dim=1)
        feats = torch.stack(feats, dim=1)
        feats = torch.einsum("nc, ncd -> nd", adp, feats)
        feat = self.proj[-1](feat)
        feat = torch.cat([feat, feats], dim=1)
        feat = self.fuse(feat) + x.features
        res = feat

        x = x.replace_feature(feat)
        x = self.voxel_block(x)
        x = x.replace_feature(self.act(x.features + res))

        logger.debug(f"BasicBlock {block_id} forward complete - total time: {time.time() - start_time:.4f}s")
        return x


class DownBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            embed_channels,
            depth,
            sp_indice_key,
            point_grid_size,
            num_ref=16,
            groups=None,
            norm_fn=None,
            sub_indice_key=None,
    ):
        super().__init__()
        self.num_ref = num_ref
        self.depth = depth
        self.point_grid_size = point_grid_size
        self.down = spconv.SparseSequential(
            spconv.SparseConv3d(
                in_channels,
                embed_channels,
                kernel_size=2,
                stride=2,
                indice_key=sp_indice_key,
                bias=False,
            ),
            norm_fn(embed_channels),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                BasicBlock(
                    in_channels=embed_channels,
                    embed_channels=embed_channels,
                    depth=len(point_grid_size) + 1,
                    groups=groups,
                    grid_size=point_grid_size,
                    norm_fn=norm_fn,
                    indice_key=sub_indice_key,
                )
            )

    def forward(self, x):
        block_id = id(self) % 1000
        start_time = time.time()
        logger.debug(f"DownBlock {block_id} forward start - features shape: {x.features.shape}")

        # 下采样
        down_start = time.time()
        x = self.down(x)
        logger.debug(
            f"DownBlock {block_id} downsampling took {time.time() - down_start:.4f}s - new shape: {x.features.shape}")

        # 处理每个BasicBlock
        for i, block in enumerate(self.blocks):
            block_start = time.time()
            x = block(x)
            logger.debug(f"DownBlock {block_id} block {i} took {time.time() - block_start:.4f}s")

        logger.debug(f"DownBlock {block_id} forward complete - total time: {time.time() - start_time:.4f}s")
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        embed_channels,
        depth,
        sp_indice_key,
        norm_fn=None,
        down_ratio=2,
        sub_indice_key=None,
    ):
        super().__init__()
        assert depth > 0
        self.up = spconv.SparseSequential(
            spconv.SparseInverseConv3d(
                in_channels,
                embed_channels,
                kernel_size=down_ratio,
                indice_key=sp_indice_key,
                bias=False,
            ),
            norm_fn(embed_channels),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList()
        self.fuse = nn.Sequential(
            nn.Linear(skip_channels + embed_channels, embed_channels),
            norm_fn(embed_channels),
            nn.ReLU(),
            nn.Linear(embed_channels, embed_channels),
            norm_fn(embed_channels),
            nn.ReLU(),
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        x = x.replace_feature(
            self.fuse(torch.cat([x.features, skip_x.features], dim=1)) + x.features
        )
        return x


@MODELS.register_module()
class OACNNs_PFEC(nn.Module):
    def __init__(
            self,
            in_channels,
            num_classes,
            embed_channels=64,
            enc_num_ref=[16, 16, 16, 16],
            enc_channels=[64, 64, 128, 256],
            groups=[2, 4, 8, 16],
            enc_depth=[2, 3, 6, 4],
            down_ratio=[2, 2, 2, 2],
            dec_channels=[96, 96, 128, 256],
            point_grid_size=[[[3, 3, 3], [10, 10, 10], [1, 1, 5]],
                             [[3, 3, 3], [10, 10, 10], [1, 1, 5]],
                             [[3, 3, 3], [10, 10, 10], [1, 1, 5]],
                             [[3, 3, 3], [10, 10, 10], [1, 1, 5]]],
            dec_depth=[2, 2, 2, 2],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(enc_channels)
        self.embed_channels = embed_channels
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=3,
                padding=1,
                indice_key="stem",
                bias=False,
            ),
            norm_fn(embed_channels),
            nn.ReLU(),
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                padding=1,
                indice_key="stem",
                bias=False,
            ),
            norm_fn(embed_channels),
            nn.ReLU(),
        )

        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(self.num_stages):
            self.enc.append(
                DownBlock(
                    in_channels=embed_channels if i == 0 else enc_channels[i - 1],
                    embed_channels=enc_channels[i],
                    depth=enc_depth[i],
                    norm_fn=norm_fn,
                    groups=groups[i],
                    point_grid_size=point_grid_size[i],
                    num_ref=enc_num_ref[i],
                    sp_indice_key=f"spconv{i}",
                    sub_indice_key=f"subm{i + 1}",
                )
            )
            self.dec.append(
                UpBlock(
                    in_channels=enc_channels[-1] if i == self.num_stages - 1 else dec_channels[i + 1],
                    skip_channels=embed_channels if i == 0 else enc_channels[i - 1],
                    embed_channels=dec_channels[i],
                    depth=dec_depth[i],
                    norm_fn=norm_fn,
                    sp_indice_key=f"spconv{i}",
                    sub_indice_key=f"subm{i}",
                )
            )

        self.final = spconv.SubMConv3d(dec_channels[0], num_classes, kernel_size=1)
        self.apply(self._init_weights)

    def forward(self, input_dict):
        discrete_coord = input_dict["grid_coord"]
        feat = input_dict["feat"]
        offset = input_dict["offset"]
        batch = offset2batch(offset)
        x = spconv.SparseConvTensor(
            features=feat,
            indices=torch.cat([batch.unsqueeze(-1), discrete_coord], dim=1)
            .int()
            .contiguous(),
            spatial_shape=torch.add(
                torch.max(discrete_coord, dim=0).values, 1
            ).tolist(),
            batch_size=batch[-1].tolist() + 1,
        )

        x = self.stem(x)
        skips = [x]
        for i in range(self.num_stages):
            x = self.enc[i](x)
            skips.append(x)
        x = skips.pop(-1)
        for i in reversed(range(self.num_stages)):
            skip = skips.pop(-1)
            x = self.dec[i](x, skip)
        x = self.final(x)
        # 仅返回预测结果（logits），损失由框架外部根据配置计算
        return x.features

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
