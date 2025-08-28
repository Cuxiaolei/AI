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


# 策略1: 电力设施自适应尺度模块（PFAS）
class PFASModule(nn.Module):
    def __init__(self, in_channels, grid_size_options):
        super().__init__()
        self.grid_size_options = grid_size_options  # [电力塔, 背景, 电力线]
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

        K = 32
        B = batch.max().item() + 1
        linearity = []
        density = []

        logger.debug(f"PFAS processing {B} batches with K={K}")

        for b in range(B):
            batch_start = time.time()
            mask = batch == b
            mask_sum = mask.sum().item()
            logger.debug(f"Processing batch {b}/{B} - {mask_sum} points")

            if mask_sum < K:
                logger.debug(f"Batch {b} has fewer points than K={K}, skipping PCA")
                linearity.append(torch.zeros(mask_sum, device=feat.device))
                density.append(torch.zeros(mask_sum, device=feat.device))
                continue

            points_b = coord[mask]
            dist_start = time.time()
            dist = torch.cdist(points_b, points_b)
            logger.debug(f"Batch {b} cdist took {time.time() - dist_start:.4f}s - dist shape: {dist.shape}")

            topk_start = time.time()
            _, idx = torch.topk(dist, K + 1, largest=False)
            idx = idx[:, 1:K + 1]  # 排除自身
            logger.debug(f"Batch {b} topk took {time.time() - topk_start:.4f}s - idx shape: {idx.shape}")

            pca_features = []
            pca_total = 0
            for i in range(points_b.shape[0]):
                pca_start = time.time()
                neighbors = points_b[idx[i]]
                centered = neighbors - neighbors.mean(dim=0)

                # SVD计算
                cov = torch.matmul(centered.T, centered) / (K - 1)
                cov_float = cov.to(torch.float32)
                eigenvalues = torch.svd(cov_float).S
                eigenvalues = eigenvalues.to(feat.dtype)
                eigenvalues = eigenvalues / eigenvalues.sum()
                pca_features.append(eigenvalues)

                pca_total += time.time() - pca_start
                # 每1000个点打印一次进度
                if (i + 1) % 1000 == 0:
                    logger.debug(f"Batch {b} PCA progress: {i + 1}/{points_b.shape[0]} points")

            logger.debug(
                f"Batch {b} PCA total time: {pca_total:.4f}s - avg per point: {pca_total / points_b.shape[0]:.6f}s")

            pca_features = torch.stack(pca_features)
            linearness = pca_features[:, 0] - (pca_features[:, 1] + pca_features[:, 2])
            linearity.append(linearness)

            # 计算密度
            density_start = time.time()
            row_indices = torch.arange(points_b.shape[0], device=dist.device).unsqueeze(1)
            neighbor_dists = dist[row_indices, idx]
            mean_dist = neighbor_dists.mean(dim=1)
            density_val = 1.0 / (mean_dist + 1e-6)
            density.append(density_val)
            logger.debug(f"Batch {b} density calculation took {time.time() - density_start:.4f}s")

            logger.debug(f"Batch {b} total time: {time.time() - batch_start:.4f}s")

        # 合并结果
        merge_start = time.time()
        linearity = torch.cat(linearity, dim=0)
        density = torch.cat(density, dim=0)
        logger.debug(f"Merging results took {time.time() - merge_start:.4f}s")

        # 特征判断
        judge_start = time.time()
        feat_logits = self.feature_judge(feat)
        feat_probs = F.softmax(feat_logits, dim=1)
        logger.debug(f"Feature judgment took {time.time() - judge_start:.4f}s")

        # 计算概率
        prob_start = time.time()
        tower_prob = (density.unsqueeze(1) * 2.0 + feat_probs[:, 0:1]) / 3.0
        background_prob = (torch.maximum(1.0 - linearity.unsqueeze(1), 1.0 - density.unsqueeze(1)) + feat_probs[:,
                                                                                                     1:2]) / 3.0
        line_prob = (linearity.unsqueeze(1) * 2.0 + feat_probs[:, 2:3]) / 3.0
        logger.debug(f"Probability calculation took {time.time() - prob_start:.4f}s")

        # 生成网格大小
        grid_start = time.time()
        grid_sizes = torch.zeros_like(coord)
        for i in range(3):
            tower_grid = self.grid_size_options[0][i]
            background_grid = self.grid_size_options[1][i]
            line_grid = self.grid_size_options[2][i] if i < 2 else self.grid_size_options[2][i] * 5

            grid_sizes[:, i] = (
                    tower_prob[:, 0] * tower_grid +
                    background_prob[:, 0] * background_grid +
                    line_prob[:, 0] * line_grid + 1e-6
            )
        logger.debug(f"Grid size calculation took {time.time() - grid_start:.4f}s")

        logger.debug(
            f"PFAS forward complete - total time: {time.time() - start_time:.4f}s - output shape: {grid_sizes.shape}")
        return grid_sizes


# 策略2: 跨模态电力特征增强（CMPFE）
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

        # 特征投影
        proj_start = time.time()
        projected_feat = self.feature_projection(x)
        logger.debug(f"Feature projection took {time.time() - proj_start:.4f}s - shape: {projected_feat.shape}")

        # 特征分离
        split_start = time.time()
        coord_feat = projected_feat[:, :3]
        color_feat = projected_feat[:, 3:6]
        normal_feat = projected_feat[:, 6:9]
        logger.debug(f"Feature splitting took {time.time() - split_start:.4f}s")

        # 注意力计算
        att_start = time.time()
        enhanced_color = color_feat * self.color_attention(color_feat)
        enhanced_normal = normal_feat * self.normal_attention(normal_feat)
        logger.debug(f"Attention calculation took {time.time() - att_start:.4f}s")

        # 特征融合
        fuse_start = time.time()
        enhanced_feat = torch.cat([coord_feat, enhanced_color, enhanced_normal], dim=1)
        fused_feat = self.feature_fusion(enhanced_feat)
        logger.debug(f"Feature fusion took {time.time() - fuse_start:.4f}s")

        # 语义注意力
        sem_start = time.time()
        sem_att = self.semantic_attention(fused_feat)
        final_feat = fused_feat * sem_att + x * (1 - sem_att)
        logger.debug(f"Semantic attention took {time.time() - sem_start:.4f}s")

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

        self.cmpfe = CMPFEModule(embed_channels)

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

        self.pfas = PFASModule(embed_channels, grid_size)

    def forward(self, x, clusters=None):
        block_id = id(self) % 1000  # 简单的块标识
        start_time = time.time()
        logger.debug(
            f"BasicBlock {block_id} forward start - features shape: {x.features.shape}, spatial shape: {x.spatial_shape}")

        # CMPFE特征增强
        cmpfe_start = time.time()
        enhanced_feat = self.cmpfe(x.features)
        x = x.replace_feature(enhanced_feat)
        feat = x.features
        logger.debug(f"BasicBlock {block_id} CMPFE took {time.time() - cmpfe_start:.4f}s")

        # PFAS动态网格生成
        pfas_start = time.time()
        coord = x.indices[:, 1:].float()
        batch = x.indices[:, 0]
        dynamic_grid_sizes = self.pfas(feat, coord, batch)
        logger.debug(f"BasicBlock {block_id} PFAS took {time.time() - pfas_start:.4f}s")

        if dynamic_grid_sizes.numel() == 0:
            logger.debug(f"BasicBlock {block_id} empty input, returning early")
            return x

        # 生成代表性网格
        grid_start = time.time()
        grid_mean = dynamic_grid_sizes.mean(dim=1)
        representative_grids = [
            torch.mean(dynamic_grid_sizes[torch.argsort(grid_mean)[100:200]], dim=0)
            if dynamic_grid_sizes.shape[0] > 200 else grid_mean.mean() * 0.8,
            torch.mean(dynamic_grid_sizes[torch.argsort(grid_mean, descending=True)[:100]], dim=0)
            if dynamic_grid_sizes.shape[0] > 100 else grid_mean.mean() * 1.2,
            torch.mean(dynamic_grid_sizes[torch.argsort(grid_mean)[:100]], dim=0)
            if dynamic_grid_sizes.shape[0] > 100 else grid_mean.mean() * 0.5
        ]
        logger.debug(f"BasicBlock {block_id} representative grids took {time.time() - grid_start:.4f}s")

        # 生成多尺度聚类
        cluster_start = time.time()
        clusters = []
        for i, grid_size in enumerate(representative_grids):
            cluster = voxel_grid(pos=coord, size=torch.clamp(grid_size, min=1e-6).tolist(), batch=batch)
            _, cluster = torch.unique(cluster, return_inverse=True)
            clusters.append(cluster)
            logger.debug(f"BasicBlock {block_id} cluster {i} shape: {cluster.shape}")
        logger.debug(f"BasicBlock {block_id} clustering took {time.time() - cluster_start:.4f}s")

        # 多尺度特征融合
        fusion_start = time.time()
        feats = []
        valid_depth = min(len(self.l_w), len(clusters))
        logger.debug(f"BasicBlock {block_id} starting fusion with valid_depth={valid_depth}")

        for i in range(valid_depth):
            fuse_i_start = time.time()
            cluster = clusters[i]
            pw = self.l_w[i](feat)
            pw = pw - scatter(pw, cluster, reduce="mean")[cluster]
            pw = self.weight[i](pw)
            pw = torch.exp(pw - pw.max())
            pw = pw / (scatter(pw, cluster, reduce="sum", dim=0)[cluster] + 1e-6)
            pfeat = self.proj[i](feat) * pw
            pfeat = scatter(pfeat, cluster, reduce="sum")[cluster]
            feats.append(pfeat)
            logger.debug(f"BasicBlock {block_id} fusion step {i} took {time.time() - fuse_i_start:.4f}s")

        if not feats:
            logger.debug(f"BasicBlock {block_id} no features for fusion, returning early")
            return x

        # 自适应融合
        adp_start = time.time()
        adp = self.adaptive(feat)
        adp = torch.softmax(adp, dim=1)
        feats = torch.stack(feats, dim=1)
        feats = torch.einsum("nc, ncd -> nd", adp, feats)
        feat = self.proj[-1](feat)
        feat = torch.cat([feat, feats], dim=1)
        feat = self.fuse(feat) + x.features
        res = feat
        logger.debug(f"BasicBlock {block_id} adaptive fusion took {time.time() - adp_start:.4f}s")

        # 稀疏卷积处理
        conv_start = time.time()
        x = x.replace_feature(feat)
        x = self.voxel_block(x)
        x = x.replace_feature(self.act(x.features + res))
        logger.debug(f"BasicBlock {block_id} convolution took {time.time() - conv_start:.4f}s")

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
