import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import knn
import spconv.pytorch as spconv
from functools import partial
from timm.models.layers import trunc_normal_
from pointcept.utils.logger import get_root_logger
from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch
from torch_geometric.nn.pool import voxel_grid
from torch_geometric.utils import scatter


# --- 原始基础模块定义 ---
class OriginalBasicBlock(nn.Module):
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

    def forward(self, x, clusters):
        feat = x.features
        feats = []
        for i, cluster in enumerate(clusters):
            pw = self.l_w[i](feat)
            pw = pw - scatter(pw, cluster, reduce="mean")[cluster]
            pw = self.weight[i](pw)
            pw = torch.exp(pw - pw.max())
            pw = pw / (scatter(pw, cluster, reduce="sum", dim=0)[cluster] + 1e-6)
            pfeat = self.proj[i](feat) * pw
            pfeat = scatter(pfeat, cluster, reduce="sum")[cluster]
            feats.append(pfeat)
        adp = self.adaptive(feat)
        adp = torch.softmax(adp, dim=1)
        feats = torch.stack(feats, dim=1)
        feats = torch.einsum("l n, l n c -> l c", adp, feats)
        feat = self.proj[-1](feat)
        feat = torch.cat([feat, feats], dim=1)
        feat = self.fuse(feat) + x.features
        res = feat
        x = x.replace_feature(feat)
        x = self.voxel_block(x)
        x = x.replace_feature(self.act(x.features + res))
        return x


class OriginalDownBlock(nn.Module):
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
                OriginalBasicBlock(
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
        x = self.down(x)
        coord = x.indices[:, 1:].float()
        batch = x.indices[:, 0]
        clusters = []
        for grid_size in self.point_grid_size:
            cluster = voxel_grid(pos=coord, size=grid_size, batch=batch)
            _, cluster = torch.unique(cluster, return_inverse=True)
            clusters.append(cluster)
        for block in self.blocks:
            x = block(x, clusters)
        return x


class OriginalUpBlock(nn.Module):
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


# --- 创新点1: 拓扑感知图卷积（PL-TopoConv）实现 ---
def compute_curvature(normals, k=16):
    """计算点云的曲率"""
    # print(f"[PL-TopoConv] compute_curvature - 输入法向量形状: {normals.shape}, k={k}")
    if normals.size(0) < k:
        # print(f"[PL-TopoConv] 点数量不足({normals.size(0)} < {k})，返回零曲率")
        return torch.zeros(normals.size(0), device=normals.device)

    # 计算KNN邻接
    edge_index = knn(normals, normals, k=k)  # [2, E]
    # print(f"[PL-TopoConv] KNN邻接边数量: {edge_index.size(1)}")
    row, col = edge_index[0], edge_index[1]

    # 计算法向量夹角的曲率近似
    dot_product = torch.sum(normals[row] * normals[col], dim=1)
    cos_theta = torch.clamp(dot_product, -1.0, 1.0)
    curvature = 1.0 - cos_theta
    # print(
        # f"[PL-TopoConv] 原始曲率统计 - min: {curvature.min():.4f}, max: {curvature.max():.4f}, mean: {curvature.mean():.4f}")

    # 聚合邻域曲率
    N = normals.size(0)
    curvature_per_point = torch.zeros(N, device=normals.device)
    if row.numel() > 0:
        curvature_per_point.scatter_add_(0, row, curvature)

    count = torch.full((N,), k, device=normals.device, dtype=torch.float)
    avg_curvature = curvature_per_point / count
    # print(
        # f"[PL-TopoConv] 平均曲率统计 - min: {avg_curvature.min():.4f}, max: {avg_curvature.max():.4f}, mean: {avg_curvature.mean():.4f}")
    return avg_curvature


def topo_aware_knn_weight(coords, normals, k=16, angle_weight=1.0, curvature_weight=1.0):
    """基于法向一致性和曲率生成拓扑感知的KNN邻域权重"""
    N = coords.size(0)
    # print(f"[PL-TopoConv] topo_aware_knn_weight - 坐标形状: {coords.shape}, 法向量形状: {normals.shape}, k={k}")
    if N < 2:
        # print(f"[PL-TopoConv] 点数量过少({N})，返回空权重")
        return torch.empty((2, 0), dtype=torch.long, device=coords.device), torch.tensor([], device=coords.device)

    # 标准KNN获取邻接
    edge_index_full = knn(coords, coords, k=min(k, N - 1))
    if edge_index_full.size(1) == 0:
        # print(f"[PL-TopoConv] 未生成邻接边")
        return edge_index_full, torch.tensor([], device=coords.device)

    row_full, col_full = edge_index_full[0], edge_index_full[1]
    # print(f"[PL-TopoConv] 邻接边覆盖点数量 - 行: {row_full.unique().numel()}, 列: {col_full.unique().numel()}")

    # 计算法向夹角权重
    norm_row = F.normalize(normals[row_full], dim=-1)
    norm_col = F.normalize(normals[col_full], dim=-1)
    cos_angle = torch.sum(norm_row * norm_col, dim=1).clamp(-1, 1)
    angle_w = (1.0 + cos_angle) / 2.0
    # print(
        # f"[PL-TopoConv] 角度权重统计 - min: {angle_w.min():.4f}, max: {angle_w.max():.4f}, mean: {angle_w.mean():.4f}")

    # 计算曲率权重
    curvature = compute_curvature(normals, k=k)
    curv_row = curvature[row_full]
    curv_col = curvature[col_full]
    avg_curv = (curv_row + curv_col) / 2.0
    curv_w = torch.exp(-curvature_weight * avg_curv)
    # print(f"[PL-TopoConv] 曲率权重统计 - min: {curv_w.min():.4f}, max: {curv_w.max():.4f}, mean: {curv_w.mean():.4f}")

    # 综合权重
    combined_weight = (angle_weight * angle_w) * (curv_w)
    # print(
        # f"[PL-TopoConv] 综合权重统计 - min: {combined_weight.min():.4f}, max: {combined_weight.max():.4f}, mean: {combined_weight.mean():.4f}")

    return edge_index_full, combined_weight


# --- 创新点2: 多模态通道注意力机制（MMCA）实现 ---
class MMCAModule(nn.Module):
    def __init__(self, in_channels,
                 coord_channels=3,
                 color_channels=3,
                 normal_channels=3,
                 attn_hidden_dim=16):
        super().__init__()
        self.in_channels = in_channels
        # 保持模态特征提取器不变
        self.coord_mlp = nn.Sequential(
            nn.Linear(coord_channels, attn_hidden_dim),
            nn.BatchNorm1d(attn_hidden_dim),
            nn.ReLU(),
            nn.Linear(attn_hidden_dim, 1)
        )
        self.color_mlp = nn.Sequential(
            nn.Linear(color_channels, attn_hidden_dim),
            nn.BatchNorm1d(attn_hidden_dim),
            nn.ReLU(),
            nn.Linear(attn_hidden_dim, 1)
        )
        self.normal_mlp = nn.Sequential(
            nn.Linear(normal_channels, attn_hidden_dim),
            nn.BatchNorm1d(attn_hidden_dim),
            nn.ReLU(),
            nn.Linear(attn_hidden_dim, 1)
        )
        # 特征融合改为作用于整个输入特征
        self.fusion = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )

    def forward(self, x, coords, colors, normals):
        # x: [N, C],  coords/colors/normals: [N, 3]
        N, C = x.shape
        # print(f"[MMCA] 前向传播 - 输入特征形状: {x.shape}, 坐标/颜色/法向量形状: {coords.shape}")

        # 计算各模态注意力权重（无需批处理repeat，直接对每个点计算）
        coord_attn = self.coord_mlp(coords).sigmoid()  # [N, 1]
        color_attn = self.color_mlp(colors).sigmoid()  # [N, 1]
        normal_attn = self.normal_mlp(normals).sigmoid()  # [N, 1]

        # 注意力权重形状调整为 [N, 1]，与特征x [N, C] 广播匹配
        coord_attn = coord_attn  # 无需额外维度
        color_attn = color_attn
        normal_attn = normal_attn

        # 融合注意力权重（而非分离特征）
        combined_attn = (coord_attn + color_attn + normal_attn) / 3.0  # 简单平均融合
        enhanced_feat = x * combined_attn  # [N, C] * [N, 1] → [N, C]

        # 特征融合与残差连接
        fused_feat = self.fusion(enhanced_feat)
        return fused_feat + x  # 残差连接



# --- 改进的基础模块 ---
class BasicBlock(OriginalBasicBlock):
    def __init__(self,
                 in_channels,
                 embed_channels,
                 norm_fn=None,
                 indice_key=None,
                 depth=4,
                 groups=None,
                 grid_size=None,
                 bias=False,
                 use_pl_topoconv=False,
                 pl_topoconv_kwargs=None):
        super().__init__(in_channels, embed_channels, norm_fn, indice_key,
                         depth, groups, grid_size, bias)
        self.use_pl_topoconv = use_pl_topoconv
        self.pl_topoconv_kwargs = pl_topoconv_kwargs or {}

        # 初始化拓扑卷积层
        if self.use_pl_topoconv:
            self.topo_conv = nn.Linear(embed_channels, embed_channels, bias=False)
            # print(f"[BasicBlock] PL-TopoConv已启用 - 拓扑卷积层初始化 (输入: {embed_channels}, 输出: {embed_channels})")
        else:
            print(f"[BasicBlock] PL-TopoConv未启用")

    def forward(self, x, clusters, normals=None):
        feat = x.features
        feats = []
        # print(f"[BasicBlock] 前向传播 - 初始特征形状: {feat.shape}, 聚类数量: {len(clusters)}")

        for i, cluster in enumerate(clusters):
            pw = self.l_w[i](feat)
            pw = pw - scatter(pw, cluster, reduce="mean")[cluster]
            pw = self.weight[i](pw)
            pw = torch.exp(pw - pw.max())
            pw = pw / (scatter(pw, cluster, reduce="sum", dim=0)[cluster] + 1e-6)

            # 应用PL-TopoConv
            if self.use_pl_topoconv and normals is not None:
                coords = x.indices[:, 1:].float()  # 获取坐标
                # print(f"[BasicBlock] 应用PL-TopoConv - 坐标形状: {coords.shape}, 法向量形状: {normals.shape}")
                edge_index, topo_weights = topo_aware_knn_weight(
                    coords, normals,
                    k=self.pl_topoconv_kwargs.get('k', 16),
                    angle_weight=self.pl_topoconv_kwargs.get('angle_weight', 1.0),
                    curvature_weight=self.pl_topoconv_kwargs.get('curvature_weight', 1.0)
                )

                if edge_index.numel() > 0:
                    row, col = edge_index
                    topo_feat = scatter(feat[col] * topo_weights.unsqueeze(1), row, reduce='mean')
                    # print(f"[BasicBlock] 拓扑特征形状: {topo_feat.shape}, 原始特征形状: {feat.shape}")
                    feat = feat + self.topo_conv(topo_feat)
                    # print(f"[BasicBlock] 拓扑特征融合完成 - 新特征形状: {feat.shape}")
                else:
                    print(f"[BasicBlock] 无有效邻接边，跳过PL-TopoConv")
            elif self.use_pl_topoconv:
                print(f"[BasicBlock] PL-TopoConv已启用但法向量为None，无法应用")

            pfeat = self.proj[i](feat) * pw
            pfeat = scatter(pfeat, cluster, reduce="sum")[cluster]
            feats.append(pfeat)

        # 自适应融合
        adp = self.adaptive(feat)
        adp = torch.softmax(adp, dim=1)
        feats = torch.stack(feats, dim=1)
        feats = torch.einsum("l n, l n c -> l c", adp, feats)
        feat = self.proj[-1](feat)
        feat = torch.cat([feat, feats], dim=1)
        feat = self.fuse(feat) + x.features
        res = feat

        # 体素卷积
        x = x.replace_feature(feat)
        x = self.voxel_block(x)
        x = x.replace_feature(self.act(x.features + res))

        # 计算并返回曲率
        curvature = compute_curvature(normals, k=self.pl_topoconv_kwargs.get('k', 16)) if (
                self.use_pl_topoconv and normals is not None) else None
        # print(
            # f"[BasicBlock] 输出特征形状: {x.features.shape}, 曲率形状: {curvature.shape if curvature is not None else 'None'}")
        return x, curvature


class DownBlock(OriginalDownBlock):
    def __init__(self,
                 in_channels,
                 embed_channels,
                 depth,
                 sp_indice_key,
                 point_grid_size,
                 num_ref=16,
                 groups=None,
                 norm_fn=None,
                 sub_indice_key=None,
                 use_pl_topoconv=False,
                 pl_topoconv_kwargs=None,
                 use_mmca=False,
                 mmca_kwargs=None):
        super().__init__(in_channels, embed_channels, depth, sp_indice_key,
                         point_grid_size, num_ref, groups, norm_fn, sub_indice_key)
        self.use_pl_topoconv = use_pl_topoconv
        self.use_mmca = use_mmca

        # 初始化基础块
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
                    use_pl_topoconv=use_pl_topoconv,
                    pl_topoconv_kwargs=pl_topoconv_kwargs
                )
            )

        # 初始化MMCA模块
        if self.use_mmca:
            self.mmca = MMCAModule(
                in_channels=embed_channels,
                **(mmca_kwargs or {})
            )
            # print(f"[DownBlock] MMCA已启用 - 输入通道: {embed_channels}")
        else:
            print(f"[DownBlock] MMCA未启用")

    def forward(self, x, normals, coords, colors):
        # 1. 执行下采样
        x_down = self.down(x)  # 下采样后的稀疏张量（特征数：N_new）
        N_new = x_down.features.shape[0]  # 下采样后的特征数量
        # print(f"[DownBlock] 下采样后特征数: {N_new}")

        # 2. 生成与下采样特征匹配的多模态特征（关键修复：同步更新coords/colors/normals）
        # 2.1 提取当前阶段的坐标和批次信息（基于下采样后的x_down）
        current_coord = x_down.indices[:, 1:].float()  # 下采样后的坐标 [N_new, 3]
        current_batch = x_down.indices[:, 0]  # 下采样后的批次索引 [N_new]
        batch_size = current_batch.max().item() + 1

        # 2.2 为当前阶段坐标生成聚类（用于聚合原始多模态特征）
        # 注意：这里的coords仍是上一阶段的，需要基于当前坐标聚类
        voxel_size = self.calculate_matching_voxel_size(
            current_coord, current_batch, target_num_clusters=N_new
        )
        # print(f"[DownBlock] 动态计算的网格大小: {voxel_size}")

        # 2.3 为原始多模态特征生成聚类索引（长度与原始coords一致）
        # 生成与原始coords匹配的批次索引（扩展当前批次索引以匹配原始长度）
        original_batch = self.expand_batch(current_batch, x.indices[:, 0], coords.shape[0])
        cluster = voxel_grid(pos=coords, size=voxel_size, batch=original_batch)
        _, cluster = torch.unique(cluster, return_inverse=True)
        # print(f"[DownBlock] 聚类数量: {cluster.max().item() + 1}")

        # 3. 聚合多模态特征（使长度与N_new一致）
        coords_down = scatter(coords, cluster, reduce="mean")  # [N_cluster, 3]
        colors_down = scatter(colors, cluster, reduce="mean")  # [N_cluster, 3]
        normals_down = scatter(normals, cluster, reduce="mean")  # [N_cluster, 3]

        # 4. 强制聚类数量与下采样特征数匹配
        if coords_down.shape[0] != N_new:
            # print(f"[DownBlock] 警告: 聚类数量({coords_down.shape[0]})与下采样特征数({N_new})不匹配，进行调整...")
            if coords_down.shape[0] < N_new:
                repeat_times = (N_new // coords_down.shape[0]) + 1
                coords_down = coords_down.repeat(repeat_times, 1)[:N_new]
                colors_down = colors_down.repeat(repeat_times, 1)[:N_new]
                normals_down = normals_down.repeat(repeat_times, 1)[:N_new]
            else:
                coords_down = coords_down[:N_new]
                colors_down = colors_down[:N_new]
                normals_down = normals_down[:N_new]

        # 5. 应用MMCA
        if self.use_mmca and hasattr(self, 'mmca'):
            feat = x_down.features
            assert coords_down.shape[0] == feat.shape[0], \
                f"MMCA输入不匹配: 多模态特征数 {coords_down.shape[0]} vs 下采样特征数 {feat.shape[0]}"
            feat = self.mmca(feat, coords_down, colors_down, normals_down)
            x_down = x_down.replace_feature(feat)

        # 6. 生成用于BasicBlock的聚类（基于下采样后的坐标）
        clusters = []
        for grid_size in self.point_grid_size:
            cluster_down = voxel_grid(pos=current_coord, size=grid_size, batch=current_batch)
            _, cluster_down = torch.unique(cluster_down, return_inverse=True)
            clusters.append(cluster_down)

        # 7. 处理BasicBlock
        curvature = None
        for block in self.blocks:
            x_down, block_curvature = block(x_down, clusters, normals=normals_down)
            if block_curvature is not None:
                curvature = block_curvature

        # 8. 返回更新后的多模态特征（供下一阶段使用）
        return x_down, curvature, coords_down, colors_down, normals_down

    def calculate_matching_voxel_size(self, coords, batch, target_num_clusters):
        """动态计算网格大小，使聚类数量接近目标值"""
        voxel_size = 0.1  # 初始值
        min_size, max_size = 0.001, 1.0  # 网格大小范围

        for _ in range(10):  # 迭代10次足够收敛
            cluster = voxel_grid(pos=coords, size=voxel_size, batch=batch)
            current_clusters = cluster.unique().numel()

            if current_clusters < target_num_clusters * 0.9:
                voxel_size *= 0.8  # 聚类太少，减小网格
            elif current_clusters > target_num_clusters * 1.1:
                voxel_size *= 1.2  # 聚类太多，增大网格
            else:
                break

            voxel_size = max(min_size, min(voxel_size, max_size))

        return voxel_size

    def expand_batch(self, current_batch, prev_batch, target_length):
        """扩展批次索引以匹配目标长度（原始coords长度）"""
        # 简单实现：通过最近邻映射将当前批次索引扩展到原始长度
        # 更优实现可基于坐标距离进行映射
        expanded_batch = torch.zeros(target_length, dtype=torch.long, device=current_batch.device)
        # 假设prev_batch是上一阶段的批次索引（与原始coords长度匹配）
        for b in range(current_batch.max().item() + 1):
            mask = prev_batch == b
            expanded_batch[mask] = b
        return expanded_batch


# --- 改进的主模型 ---
@MODELS.register_module()
class OACNNs_TopoFusion(nn.Module):
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
            point_grid_size=[[16, 32, 64], [8, 16, 24], [4, 8, 12], [2, 4, 6]],
            dec_depth=[2, 2, 2, 2],
            # 创新点开关和参数
            use_pl_topoconv=False,
            pl_topoconv_kwargs=None,
            use_mmca=False,
            mmca_kwargs=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(enc_channels)
        self.embed_channels = embed_channels

        # 创新点配置日志
        print(f"\n===== 模型初始化 - 创新点配置 =====")
        print(f"PL-TopoConv 启用状态: {use_pl_topoconv}")
        print(f"MMCA 启用状态: {use_mmca}")
        if use_pl_topoconv:
            print(f"PL-TopoConv 参数: {pl_topoconv_kwargs}")
        if use_mmca:
            print(f"MMCA 参数: {mmca_kwargs}")
        print(f"=================================\n")

        self.use_pl_topoconv = use_pl_topoconv
        self.use_mmca = use_mmca
        self.pl_topoconv_kwargs = pl_topoconv_kwargs or {}
        self.mmca_kwargs = mmca_kwargs or {}

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        # 初始卷积层
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

        # 编码器和解码器
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
                    use_pl_topoconv=use_pl_topoconv,
                    pl_topoconv_kwargs=pl_topoconv_kwargs,
                    use_mmca=use_mmca,
                    mmca_kwargs=mmca_kwargs,
                )
            )
            self.dec.append(
                OriginalUpBlock(
                    in_channels=(
                        enc_channels[-1]
                        if i == self.num_stages - 1
                        else dec_channels[i + 1]
                    ),
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
        # print(f"\n===== OACNNs_TopoFusion 前向传播开始 =====")
        # 解析输入
        discrete_coord = input_dict["grid_coord"]
        feat = input_dict["feat"]
        offset = input_dict["offset"]
        batch = offset2batch(offset)
        # print(
            # f"[Model] 输入特征形状: {feat.shape}, 离散坐标形状: {discrete_coord.shape}, 批次大小: {batch[-1].item() + 1}")

        # 提取多模态特征
        coords = feat[..., 0:3]  # 前3通道：坐标
        colors = feat[..., 3:6]  # 中间3通道：法向量
        normals = feat[..., 6:9]  # 后3通道：颜色
        # 计算并备份原始曲率（30000点）
        if self.use_pl_topoconv and normals is not None and normals.numel() > 0:
            original_curvature = compute_curvature(
                normals,
                k=self.pl_topoconv_kwargs.get('k', 16)
            )
            input_dict["original_curvature"] = original_curvature  # 用于损失计算
        # print(f"[Model] 多模态特征 - 坐标: {coords.shape if coords is not None else 'None'}, "
        #       f"颜色: {colors.shape if colors is not None else 'None'}, "
        #       f"法向量: {normals.shape if normals is not None else 'None'}")

        # 构建稀疏张量
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
        # print(f"[Model] 稀疏张量构建完成 - 空间形状: {x.spatial_shape}, 批次大小: {x.batch_size}")

        # 初始卷积
        x = self.stem(x)
        # print(f"[Model] 初始卷积后特征形状: {x.features.shape}")
        skips = [x]
        curvatures = []

        # 编码器传播
        for i in range(self.num_stages):
            # print(f"\n[Model] 编码器阶段 {i + 1}/{self.num_stages}")
            x, curvature, coords, colors, normals = self.enc[i](x, normals, coords, colors)
            # x, curvature = self.enc[i](x, normals, coords, colors)
            skips.append(x)
            if curvature is not None:
                curvatures.append(curvature)
                # print(f"[Model] 编码器阶段 {i + 1} 曲率形状: {curvature.shape}")
            # print(f"[Model] 编码器阶段 {i + 1} 输出特征形状: {x.features.shape}")

        # 解码器传播
        x = skips.pop(-1)
        # print(f"\n[Model] 解码器开始 - 初始特征形状: {x.features.shape}")
        for i in reversed(range(self.num_stages)):
            skip = skips.pop(-1)
            # print(f"[Model] 解码器阶段 {i + 1}/{self.num_stages} - 跳跃连接特征形状: {skip.features.shape}")
            x = self.dec[i](x, skip)
            # print(f"[Model] 解码器阶段 {i + 1} 输出特征形状: {x.features.shape}")

        # 最终预测
        x = self.final(x)
        seg_logits = x.features  # 预测结果
        # print(f"\n[Model] 最终预测形状: {seg_logits.shape}")

        # 保存曲率用于损失计算（创新点3: PL-PLE Loss依赖）
        # if curvatures:
            # input_dict["original_curvature"] = curvatures[-1]
            # print(f"[Model] 曲率已存入input_dict - 形状: {curvatures[-1].shape}")
        # else:
        #     print(f"[Model] 未生成曲率数据 - 可能PL-TopoConv未启用或未正确运行")

        # print(f"===== OACNNs_TopoFusion 前向传播结束 =====\n")
        return seg_logits

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
