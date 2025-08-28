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
    def __init__(self, in_channels, grid_size_options, K=16):  # 优化：K从32→16
        super().__init__()
        self.grid_size_options = grid_size_options  # [电力塔, 背景, 电力线]
        self.K = K  # 近邻数减少，降低计算量
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
        N = coord.shape[0]  # 总点数
        B = batch.max().item() + 1 if batch.numel() > 0 else 0
        if B == 0:
            logger.debug("PFAS: Empty batch, returning default grid")
            return torch.ones_like(coord) * self.grid_size_options[1][0]

        # 修正1：构造 [N, N] 的同一样本掩码矩阵
        batch_mask = (batch.unsqueeze(0) == batch.unsqueeze(1)).float()  # [N, N]

        # 计算距离矩阵（N x N），并过滤跨样本点对
        dist = torch.cdist(coord, coord)  # [N, N]
        # 修正2：只保留同一样本内的点对距离，跨样本点对距离设为无穷大
        dist = dist * batch_mask  # 非本样本点对距离置0
        dist = dist + (1 - batch_mask) * torch.inf  # 非本样本点对距离设为inf（屏蔽）
        dist[torch.eye(N, device=dist.device).bool()] = torch.inf  # 屏蔽自身距离（对角线）

        # 后续KNN等操作保持不变
        _, idx = torch.topk(dist, self.K, largest=False)  # [N, K]

        # 3. 批量提取近邻坐标（N x K x 3）
        neighbor_coords = coord[idx]  # 利用索引批量获取，替代逐点循环
        # 中心化（N x K x 3）
        neighbor_centered = neighbor_coords - neighbor_coords.mean(dim=1, keepdim=True)

        # 优化2：批量计算协方差和PCA（向量化操作，无Python循环）
        # 协方差矩阵：N x 3 x 3（替代逐点计算）
        cov = torch.einsum('nkd,nke->nde', neighbor_centered, neighbor_centered) / (self.K - 1)
        # SVD分解（批量处理，N x 3 x 3）
        U, S, V = torch.svd(cov.float())  # 用float降低计算精度，加速且省内存
        S = S.to(feat.dtype)
        # 归一化特征值（N x 3）
        S_norm = S / (S.sum(dim=1, keepdim=True) + 1e-6)

        # 计算线性度（N x 1）
        linearity = S_norm[:, 0].unsqueeze(1) - (S_norm[:, 1] + S_norm[:, 2]).unsqueeze(1)

        # 4. 批量计算密度（N x 1）
        neighbor_dists = dist.gather(1, idx)  # N x K
        mean_dist = neighbor_dists.mean(dim=1, keepdim=True)
        density = 1.0 / (mean_dist + 1e-6)

        logger.debug(f"PFAS: PCA and density calculation took {time.time() - start_time:.4f}s")

        # 特征判断（与原逻辑一致，并行操作）
        judge_start = time.time()
        feat_logits = self.feature_judge(feat)
        feat_probs = F.softmax(feat_logits, dim=1)  # N x 3

        # 计算类别概率（批量操作）
        tower_prob = (density * 2.0 + feat_probs[:, 0:1]) / 3.0
        background_prob = (torch.maximum(1.0 - linearity, 1.0 - density) + feat_probs[:, 1:2]) / 3.0
        line_prob = (linearity * 2.0 + feat_probs[:, 2:3]) / 3.0

        # 生成动态网格（批量计算，N x 3）
        grid_start = time.time()
        grid_sizes = torch.zeros_like(coord, device=coord.device)
        for i in range(3):  # 遍历x/y/z轴
            tower_grid = self.grid_size_options[0][i]
            background_grid = self.grid_size_options[1][i]
            line_grid = self.grid_size_options[2][i] if i < 2 else self.grid_size_options[2][i] * 5
            grid_sizes[:, i] = (
                tower_prob[:, 0] * tower_grid +
                background_prob[:, 0] * background_grid +
                line_prob[:, 0] * line_grid + 1e-6
            )

        logger.debug(
            f"PFAS forward complete - total time: {time.time() - start_time:.4f}s - output shape: {grid_sizes.shape}")
        return grid_sizes


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
