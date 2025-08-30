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

# 配置调试日志（减少冗余输出）
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# 策略1: 电力设施自适应尺度模块（PFAS）- 深度内存优化版
class PFASModule(nn.Module):
    def __init__(self, in_channels, grid_size_options, K=16, max_points_per_batch=20000, knn_batch_size=512):
        super().__init__()
        self.grid_size_options = grid_size_options  # [电力塔, 背景, 电力线]
        self.K = K
        self.max_points_per_batch = max_points_per_batch
        self.knn_batch_size = knn_batch_size

        # 缩减特征判断器维度（64→32）
        self.feature_judge = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 输出三类概率
        )

    def forward(self, feat, coord, batch):
        start_time = time.time()
        # if logger.isEnabledFor(logging.INFO):
        #     logger.info(
        #         f"PFAS forward start - feat shape: {feat.shape}, coord shape: {coord.shape}, batch num: {batch.max().item() + 1 if batch.numel() > 0 else 0}"
        #     )

        N = coord.shape[0]
        B = batch.max().item() + 1 if batch.numel() > 0 else 0
        if B == 0:
            logger.warning("PFAS: Empty batch, returning default background grid")
            return torch.ones_like(coord) * self.grid_size_options[1][0]

        dynamic_grid_sizes = torch.zeros_like(coord, device=coord.device)
        max_points_per_batch = self.max_points_per_batch
        knn_batch_size = self.knn_batch_size

        for b in range(B):
            batch_mask = (batch == b)
            coord_b = coord[batch_mask]  # [N_b, 3]
            feat_b = feat[batch_mask]  # [N_b, C]
            N_b = coord_b.shape[0]

            if N_b < self.K:
                dynamic_grid_sizes[batch_mask] = torch.tensor(self.grid_size_options[1], device=coord.device)
                continue

            # 拆分超大数据集（避免[N_b, N_b]全量矩阵）
            sub_batch_indices = torch.split(torch.arange(N_b, device=coord.device), max_points_per_batch)
            all_idx = torch.zeros((N_b, self.K), dtype=torch.long, device=coord.device)
            all_dist = torch.zeros((N_b, self.K), dtype=coord.dtype, device=coord.device)

            for sub_idx in sub_batch_indices:
                sub_size = len(sub_idx)
                coord_sub = coord_b[sub_idx]
                sub_idx_batch = torch.zeros((sub_size, self.K), dtype=torch.long, device=coord.device)
                sub_dist_batch = torch.zeros((sub_size, self.K), dtype=coord.dtype, device=coord.device)

                # 分批次计算KNN（降低内存峰值）
                for i in range(0, sub_size, knn_batch_size):
                    current_end = min(i + knn_batch_size, sub_size)
                    current_coord = coord_sub[i:current_end]
                    current_original_idx = sub_idx[i:current_end]

                    dist = torch.cdist(current_coord, coord_b)
                    dist[torch.arange(len(current_original_idx)), current_original_idx] = torch.inf
                    min_dist, min_idx = torch.topk(dist, self.K, largest=False)

                    sub_dist_batch[i:current_end] = min_dist
                    sub_idx_batch[i:current_end] = min_idx

                all_idx[sub_idx] = sub_idx_batch
                all_dist[sub_idx] = sub_dist_batch

            # 近邻特征计算（复用索引，减少内存碎片）
            neighbor_coords = torch.index_select(coord_b, dim=0, index=all_idx.view(-1)).view(N_b, self.K, 3)
            neighbor_centered = neighbor_coords - neighbor_coords.mean(dim=1, keepdim=True)

            # PCA计算精度优化（float类型）
            with torch.cuda.amp.autocast(enabled=False):
                cov = torch.einsum('nkd,nke->nde', neighbor_centered.float(), neighbor_centered.float()) / (self.K - 1)
                U, S, V = torch.svd(cov)
            S = S.to(feat.dtype)
            S_norm = S / (S.sum(dim=1, keepdim=True) + 1e-6)

            # 特征计算（复用已有距离数据）
            linearity = S_norm[:, 0].unsqueeze(1) - (S_norm[:, 1] + S_norm[:, 2]).unsqueeze(1)
            mean_dist = all_dist.mean(dim=1, keepdim=True)
            density = 1.0 / (mean_dist + 1e-6)

            # 动态网格生成
            feat_logits = self.feature_judge(feat_b)
            feat_probs = F.softmax(feat_logits, dim=1)
            tower_prob = (density * 2.0 + feat_probs[:, 0:1]) / 3.0
            background_prob = (torch.maximum(1.0 - linearity, 1.0 - density) + feat_probs[:, 1:2]) / 3.0
            line_prob = (linearity * 2.0 + feat_probs[:, 2:3]) / 3.0

            for i_axis in range(3):
                tower_grid = self.grid_size_options[0][i_axis]
                background_grid = self.grid_size_options[1][i_axis]
                line_grid = self.grid_size_options[2][i_axis] if i_axis < 2 else self.grid_size_options[2][i_axis] * 5
                dynamic_grid_sizes[batch_mask, i_axis] = (
                        tower_prob[:, 0] * tower_grid +
                        background_prob[:, 0] * background_grid +
                        line_prob[:, 0] * line_grid + 1e-6
                )

        # if logger.isEnabledFor(logging.INFO):
        #     logger.info(
        #         f"PFAS forward complete - time: {time.time() - start_time:.4f}s - output shape: {dynamic_grid_sizes.shape}"
        #     )
        return dynamic_grid_sizes


# 策略2: 跨模态电力特征增强（CMPFE）- 内存精简版
class CMPFEModule(nn.Module):
    def __init__(self, in_channels, proj_dim=6, attn_hidden_dim=16):
        super().__init__()
        self.in_channels = in_channels
        self.proj_dim = proj_dim
        self.attn_hidden_dim = attn_hidden_dim

        # 精简特征投影
        self.feature_projection = nn.Sequential(
            nn.Linear(in_channels, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
        )

        # 缩减注意力层中间维度
        self.color_attention = nn.Sequential(
            nn.Linear(2, attn_hidden_dim),
            nn.ReLU(),
            nn.Linear(attn_hidden_dim, 2),
            nn.Sigmoid()
        )
        self.normal_attention = nn.Sequential(
            nn.Linear(2, attn_hidden_dim),
            nn.ReLU(),
            nn.Linear(attn_hidden_dim, 2),
            nn.Sigmoid()
        )

        # 精简特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(proj_dim, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels)
        )

        # 语义注意力层维度匹配
        self.semantic_attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        start_time = time.time()
        # if logger.isEnabledFor(logging.INFO):
        #     logger.info(f"CMPFE forward start - input shape: {x.shape}")

        # 特征投影
        projected_feat = self.feature_projection(x)  # [N, proj_dim]

        # 特征拆分（带边界检查）
        coord_feat = projected_feat[:, :2]
        color_feat = projected_feat[:, 2:4]
        normal_feat = projected_feat[:, 4:min(self.proj_dim, projected_feat.shape[1])]
        if normal_feat.shape[1] < 2:
            normal_feat = F.pad(normal_feat, (0, 2 - normal_feat.shape[1]), mode='constant', value=0)

        # 跨模态注意力增强
        enhanced_color = color_feat * self.color_attention(color_feat)
        enhanced_normal = normal_feat * self.normal_attention(normal_feat)

        # 特征拼接与融合
        enhanced_feat = torch.cat([coord_feat, enhanced_color, enhanced_normal], dim=1)
        fused_feat = self.feature_fusion(enhanced_feat)  # [N, in_channels]

        # 语义注意力加权（合并残差计算）
        sem_att = self.semantic_attention(fused_feat)
        final_feat = fused_feat * sem_att + x * (1 - sem_att)

        # if logger.isEnabledFor(logging.INFO):
        #     logger.info(
        #         f"CMPFE forward complete - time: {time.time() - start_time:.4f}s - output shape: {final_feat.shape}"
        #     )
        return final_feat


class BasicBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            embed_channels,
            norm_fn=None,
            indice_key=None,
            depth=3,  # 减少深度降低内存
            groups=None,
            grid_size=None,
            bias=False,
            use_pfas=True,
            use_cmpfe=True,
            cmpfe_params=None,
            pfas_params=None,
    ):
        super().__init__()
        assert embed_channels % groups == 0
        self.groups = groups
        self.embed_channels = embed_channels
        self.proj = nn.ModuleList()
        self.grid_size = grid_size
        self.weight = nn.ModuleList()
        self.l_w = nn.ModuleList()
        self.use_cmpfe = use_cmpfe
        self.use_pfas = use_pfas
        self.pfas_params = pfas_params or {}

        # 实例化优化后的CMPFE模块
        if self.use_cmpfe:
            self.cmpfe = CMPFEModule(
                in_channels=embed_channels,
                proj_dim=cmpfe_params.get('proj_dim', 6),
                attn_hidden_dim=cmpfe_params.get('attn_hidden_dim', 16)
            )
        else:
            self.cmpfe = None

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

        # 实例化优化后的PFAS模块
        if self.use_pfas and grid_size is not None:
            self.pfas = PFASModule(
                in_channels=embed_channels,
                grid_size_options=grid_size,
                K=self.pfas_params.get('K', 16),
                max_points_per_batch=self.pfas_params.get('max_points_per_batch', 20000),
                knn_batch_size=self.pfas_params.get('knn_batch_size', 512)
            )
        else:
            self.pfas = None

    def forward(self, x, clusters=None):
        block_id = id(self) % 1000
        start_time = time.time()
        # if logger.isEnabledFor(logging.INFO):
        #     logger.info(
        #         f"BasicBlock {block_id} start - features shape: {x.features.shape}, spatial shape: {x.spatial_shape}"
        #     )

        # CMPFE特征增强
        if self.use_cmpfe and self.cmpfe is not None:
            enhanced_feat = self.cmpfe(x.features)
            x = x.replace_feature(enhanced_feat)

        feat = x.features
        coord = x.indices[:, 1:].float()
        batch = x.indices[:, 0]
        representative_grids = []

        # PFAS动态网格生成
        if self.use_pfas and self.pfas is not None:
            dynamic_grid_sizes = self.pfas(feat, coord, batch)
            if dynamic_grid_sizes is None or dynamic_grid_sizes.numel() == 0:
                logger.warning(f"BasicBlock {block_id} empty input, returning early")
                return x
            grid_mean = dynamic_grid_sizes.mean(dim=1)
            if dynamic_grid_sizes.shape[0] > 100:
                representative_grids.append(torch.mean(dynamic_grid_sizes[torch.argsort(grid_mean)[50:100]], dim=0))
                representative_grids.append(
                    torch.mean(dynamic_grid_sizes[torch.argsort(grid_mean, descending=True)[:50]], dim=0))
            else:
                representative_grids.append(grid_mean.mean() * 1.0)
                representative_grids.append(grid_mean.mean() * 1.2)
        else:
            if self.grid_size is not None and len(self.grid_size) > 0:
                representative_grids = [torch.tensor(gs, device=coord.device) for gs in self.grid_size]
            else:
                representative_grids = [
                    torch.tensor([8, 12, 16, 16], device=coord.device),
                    torch.tensor([6, 9, 12, 12], device=coord.device)
                ]

        # 生成聚类
        clusters = []
        for grid_size in representative_grids:
            cluster = voxel_grid(pos=coord, size=torch.clamp(grid_size, min=1e-6).tolist(), batch=batch)
            _, cluster = torch.unique(cluster, return_inverse=True)
            clusters.append(cluster)

        # 特征融合
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
            logger.warning(f"BasicBlock {block_id} no features for fusion, returning early")
            return x

        # 自适应权重与特征融合
        adp = self.adaptive(feat)
        adp = adp[:, :len(feats)]
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

        # if logger.isEnabledFor(logging.INFO):
        #     logger.info(f"BasicBlock {block_id} complete - time: {time.time() - start_time:.4f}s")
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
            use_pfas=True,
            use_cmpfe=True,
            cmpfe_params=None,
            pfas_params=None,
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
                    use_pfas=use_pfas,
                    use_cmpfe=use_cmpfe,
                    cmpfe_params=cmpfe_params,
                    pfas_params=pfas_params,
                )
            )

    def forward(self, x):
        block_id = id(self) % 1000
        start_time = time.time()
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"DownBlock {block_id} start - features shape: {x.features.shape}")

        # 下采样
        x = self.down(x)
        # if logger.isEnabledFor(logging.INFO):
        #     logger.info(f"DownBlock {block_id} downsampled - shape: {x.features.shape}")

        # 处理每个BasicBlock
        for i, block in enumerate(self.blocks):
            x = block(x)

        # if logger.isEnabledFor(logging.INFO):
        #     logger.info(f"DownBlock {block_id} complete - time: {time.time() - start_time:.4f}s")
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
            embed_channels=32,  # 降低嵌入维度
            enc_num_ref=[16, 16, 16, 16],
            enc_channels=[32, 32, 64, 128],  # 降低通道数
            groups=[2, 4, 8, 16],
            enc_depth=[2, 2, 4, 3],  # 减少深度
            down_ratio=[2, 2, 2, 2],
            dec_channels=[96, 96, 128, 256],
            orig_point_grid_size=[[16, 32, 64], [8, 16, 24], [4, 8, 12], [2, 4, 6]],
            point_grid_size=[[[3, 3, 3], [10, 10, 10], [1, 1, 5]],
                             [[3, 3, 3], [10, 10, 10], [1, 1, 5]],
                             [[3, 3, 3], [10, 10, 10], [1, 1, 5]],
                             [[3, 3, 3], [10, 10, 10], [1, 1, 5]]],
            dec_depth=[2, 2, 2, 2],
            use_pfas=True,
            pfas=None,
            use_cmpfe=True,
            cmpfe=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(enc_channels)
        self.embed_channels = embed_channels
        self.use_pfas = use_pfas
        self.use_cmpfe = use_cmpfe
        self.pfas_params = pfas or {}
        self.cmpfe_params = cmpfe or {}
        self.orig_point_grid_size = orig_point_grid_size
        self.point_grid_size = point_grid_size
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
            current_grid = self.pfas_params["grid_size_options"][i] if self.use_pfas else self.orig_point_grid_size[i]

            self.enc.append(
                DownBlock(
                    in_channels=embed_channels if i == 0 else enc_channels[i - 1],
                    embed_channels=enc_channels[i],
                    depth=enc_depth[i],
                    norm_fn=norm_fn,
                    groups=groups[i],
                    point_grid_size=current_grid,
                    num_ref=enc_num_ref[i],
                    sp_indice_key=f"spconv{i}",
                    sub_indice_key=f"subm{i + 1}",
                    use_pfas=self.use_pfas,
                    use_cmpfe=self.use_cmpfe,
                    cmpfe_params=self.cmpfe_params,
                    pfas_params=self.pfas_params,
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
