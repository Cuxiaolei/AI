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

# 导入原始模块
from .oacnns_v1m1_base import BasicBlock as OriginalBasicBlock, DonwBlock as OriginalDownBlock, \
    UpBlock as OriginalUpBlock


# --- 创新点1: 拓扑感知图卷积（PL-TopoConv）实现 ---
def compute_curvature(normals, k=16):
    """计算点云的曲率"""
    # 调试：输入信息
    print(f"[PL-TopoConv] compute_curvature - normals shape: {normals.shape}, k: {k}")
    if normals.size(0) < k:
        print(f"[PL-TopoConv] Not enough points ({normals.size(0)} < {k}), return zero curvature")
        return torch.zeros(normals.size(0), device=normals.device)

    edge_index = knn(normals, normals, k=k)  # [2, E]
    # 调试：KNN邻接信息
    print(f"[PL-TopoConv] KNN edge index shape: {edge_index.shape}, total edges: {edge_index.size(1)}")
    row, col = edge_index[0], edge_index[1]

    # 计算法向量之间的夹角 (1 - cos(theta)) 作为曲率近似
    dot_product = torch.sum(normals[row] * normals[col], dim=1)
    cos_theta = torch.clamp(dot_product, -1.0, 1.0)
    curvature = 1.0 - cos_theta

    # 调试：曲率基本统计
    print(
        f"[PL-TopoConv] Raw curvature - min: {curvature.min():.4f}, max: {curvature.max():.4f}, mean: {curvature.mean():.4f}")

    # 聚合邻域曲率
    N = normals.size(0)
    curvature_per_point = torch.zeros(N, device=normals.device)
    if row.numel() > 0:
        curvature_per_point.scatter_add_(0, row, curvature)

    count = torch.full((N,), k, device=normals.device, dtype=torch.float)
    avg_curvature = curvature_per_point / count
    # 调试：聚合后曲率统计
    print(
        f"[PL-TopoConv] Avg curvature - min: {avg_curvature.min():.4f}, max: {avg_curvature.max():.4f}, mean: {avg_curvature.mean():.4f}")
    return avg_curvature


def topo_aware_knn_weight(coords, normals, k=16, angle_weight=1.0, curvature_weight=1.0):
    """基于法向一致性和曲率生成拓扑感知的KNN邻域权重"""
    N = coords.size(0)
    print(f"[PL-TopoConv] topo_aware_knn_weight - coords shape: {coords.shape}, normals shape: {normals.shape}, k: {k}")
    if N < 2:
        print(f"[PL-TopoConv] Too few points ({N}), return empty edges")
        return torch.empty((2, 0), dtype=torch.long, device=coords.device), torch.tensor([], device=coords.device)

    # 标准KNN获取邻接
    edge_index_full = knn(coords, coords, k=min(k, N - 1))
    if edge_index_full.size(1) == 0:
        print(f"[PL-TopoConv] No edges generated")
        return edge_index_full, torch.tensor([], device=coords.device)

    row_full, col_full = edge_index_full[0], edge_index_full[1]
    print(f"[PL-TopoConv] Full edge index - rows: {row_full.unique().numel()}, cols: {col_full.unique().numel()}")

    # 计算法向夹角权重 (1 + cos(theta)) / 2
    norm_row = F.normalize(normals[row_full], dim=-1)
    norm_col = F.normalize(normals[col_full], dim=-1)
    cos_angle = torch.sum(norm_row * norm_col, dim=1).clamp(-1, 1)
    angle_w = (1.0 + cos_angle) / 2.0
    # 调试：角度权重统计
    print(
        f"[PL-TopoConv] Angle weights - min: {angle_w.min():.4f}, max: {angle_w.max():.4f}, mean: {angle_w.mean():.4f}")

    # 计算曲率权重 exp(-curvature)
    curvature = compute_curvature(normals, k=k)
    curv_row = curvature[row_full]
    curv_col = curvature[col_full]
    avg_curv = (curv_row + curv_col) / 2.0
    curv_w = torch.exp(-curvature_weight * avg_curv)
    # 调试：曲率权重统计
    print(
        f"[PL-TopoConv] Curvature weights - min: {curv_w.min():.4f}, max: {curv_w.max():.4f}, mean: {curv_w.mean():.4f}")

    # 综合权重
    combined_weight = (angle_weight * angle_w) * (curv_w)
    print(
        f"[PL-TopoConv] Combined weights - min: {combined_weight.min():.4f}, max: {combined_weight.max():.4f}, mean: {combined_weight.mean():.4f}")

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
        self.coord_channels = coord_channels
        self.color_channels = color_channels
        self.normal_channels = normal_channels

        # 模态特征提取
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

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )

        # 调试：初始化信息
        print(f"[MMCA] Initialized - in_channels: {in_channels}, coord_channels: {coord_channels}, "
              f"color_channels: {color_channels}, normal_channels: {normal_channels}, "
              f"attn_hidden_dim: {attn_hidden_dim}")

    def forward(self, x, coords, colors, normals):
        # 调试：输入形状
        print(f"[MMCA] Forward - x shape: {x.shape}, coords shape: {coords.shape}, "
              f"colors shape: {colors.shape}, normals shape: {normals.shape}")

        # 计算各模态注意力权重
        batch_size = x.size(0) // coords.size(0) if len(x.shape) == 3 else 1
        print(f"[MMCA] Batch size inferred: {batch_size}")

        if batch_size > 1:
            # 处理批处理情况
            coords = coords.repeat(batch_size, 1)
            colors = colors.repeat(batch_size, 1)
            normals = normals.repeat(batch_size, 1)
            print(
                f"[MMCA] Repeated modalities - coords: {coords.shape}, colors: {colors.shape}, normals: {normals.shape}")

        coord_attn = self.coord_mlp(coords).sigmoid()
        color_attn = self.color_mlp(colors).sigmoid()
        normal_attn = self.normal_mlp(normals).sigmoid()

        # 调试：注意力权重统计
        print(
            f"[MMCA] Coord attention - min: {coord_attn.min():.4f}, max: {coord_attn.max():.4f}, mean: {coord_attn.mean():.4f}")
        print(
            f"[MMCA] Color attention - min: {color_attn.min():.4f}, max: {color_attn.max():.4f}, mean: {color_attn.mean():.4f}")
        print(
            f"[MMCA] Normal attention - min: {normal_attn.min():.4f}, max: {normal_attn.max():.4f}, mean: {normal_attn.mean():.4f}")

        # 调整注意力权重形状以匹配输入特征
        if len(x.shape) == 3:  # [B, N, C]
            coord_attn = coord_attn.unsqueeze(0).unsqueeze(-1)
            color_attn = color_attn.unsqueeze(0).unsqueeze(-1)
            normal_attn = normal_attn.unsqueeze(0).unsqueeze(-1)
        else:  # [N, C]
            coord_attn = coord_attn.unsqueeze(-1)
            color_attn = color_attn.unsqueeze(-1)
            normal_attn = normal_attn.unsqueeze(-1)
        print(
            f"[MMCA] Adjusted attention shapes - coord: {coord_attn.shape}, color: {color_attn.shape}, normal: {normal_attn.shape}")

        # 分离坐标、颜色和法向量特征通道
        coord_feat = x[:, :self.coord_channels] * coord_attn
        color_feat = x[:, self.coord_channels:self.coord_channels + self.color_channels] * color_attn
        normal_feat = x[:, self.coord_channels + self.color_channels:] * normal_attn
        print(
            f"[MMCA] Modality features - coord: {coord_feat.shape}, color: {color_feat.shape}, normal: {normal_feat.shape}")

        # 融合增强后的特征
        enhanced_feat = torch.cat([coord_feat, color_feat, normal_feat], dim=-1)
        fused_feat = self.fusion(enhanced_feat)
        print(f"[MMCA] Fused feature shape: {fused_feat.shape}")

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

        # 如果使用PL-TopoConv，初始化额外的卷积层
        if self.use_pl_topoconv:
            self.topo_conv = nn.Linear(embed_channels, embed_channels, bias=False)
            print(
                f"[BasicBlock] PL-TopoConv enabled - topo_conv initialized (in: {embed_channels}, out: {embed_channels})")
        else:
            print(f"[BasicBlock] PL-TopoConv disabled")

    def forward(self, x, clusters, normals=None):
        feat = x.features
        feats = []
        print(f"[BasicBlock] Forward - initial feat shape: {feat.shape}, num clusters: {len(clusters)}")

        for i, cluster in enumerate(clusters):
            pw = self.l_w[i](feat)
            pw = pw - scatter(pw, cluster, reduce="mean")[cluster]
            pw = self.weight[i](pw)
            pw = torch.exp(pw - pw.max())
            pw = pw / (scatter(pw, cluster, reduce="sum", dim=0)[cluster] + 1e-6)

            # 如果启用PL-TopoConv，应用拓扑感知权重
            if self.use_pl_topoconv and normals is not None:
                coords = x.indices[:, 1:].float()  # 获取坐标
                print(
                    f"[BasicBlock] Applying PL-TopoConv - coords shape: {coords.shape}, normals shape: {normals.shape}")
                edge_index, topo_weights = topo_aware_knn_weight(
                    coords, normals,
                    k=self.pl_topoconv_kwargs.get('k', 16),
                    angle_weight=self.pl_topoconv_kwargs.get('angle_weight', 1.0),
                    curvature_weight=self.pl_topoconv_kwargs.get('curvature_weight', 1.0)
                )

                if edge_index.numel() > 0:
                    row, col = edge_index
                    topo_feat = scatter(feat[col] * topo_weights.unsqueeze(1), row, reduce='mean')
                    print(f"[BasicBlock] Topo feature shape: {topo_feat.shape}, original feat shape: {feat.shape}")
                    feat = feat + self.topo_conv(topo_feat)
                    print(f"[BasicBlock] Feat updated with topo_conv - new shape: {feat.shape}")
                else:
                    print(f"[BasicBlock] No valid edges for PL-TopoConv")
            elif self.use_pl_topoconv:
                print(f"[BasicBlock] PL-TopoConv enabled but normals is None, skipping")

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

        # 返回曲率用于损失计算
        curvature = compute_curvature(normals, k=self.pl_topoconv_kwargs.get('k', 16)) if (
                self.use_pl_topoconv and normals is not None) else None
        print(
            f"[BasicBlock] Output x features shape: {x.features.shape}, curvature shape: {curvature.shape if curvature is not None else 'None'}")
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

        # 重新初始化blocks以支持新参数
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
                in_channels=in_channels, **(mmca_kwargs or {})
            )
            print(f"[DownBlock] MMCA enabled - in_channels: {in_channels}")
        else:
            print(f"[DownBlock] MMCA disabled")

    def forward(self, x, normals=None, coords=None, colors=None):
        print(f"[DownBlock] Forward - input features shape: {x.features.shape}")
        # 如果启用MMCA，先处理特征
        if self.use_mmca and coords is not None and colors is not None and normals is not None:
            print(
                f"[DownBlock] Applying MMCA - coords: {coords.shape}, colors: {colors.shape}, normals: {normals.shape}")
            x = x.replace_feature(self.mmca(x.features, coords, colors, normals))
            print(f"[DownBlock] After MMCA - features shape: {x.features.shape}")
        elif self.use_mmca:
            print(f"[DownBlock] MMCA enabled but missing modalities (coords/colors/normals), skipping")

        x = self.down(x)
        print(f"[DownBlock] After downsampling - features shape: {x.features.shape}")
        coord = x.indices[:, 1:].float()
        batch = x.indices[:, 0]
        clusters = []
        for grid_size in self.point_grid_size:
            cluster = voxel_grid(pos=coord, size=grid_size, batch=batch)
            _, cluster = torch.unique(cluster, return_inverse=True)
            clusters.append(cluster)
            print(f"[DownBlock] Cluster (grid {grid_size}) - unique clusters: {cluster.unique().numel()}")

        curvatures = []
        for i, block in enumerate(self.blocks):
            print(f"[DownBlock] Processing block {i + 1}/{len(self.blocks)}")
            x, curvature = block(x, clusters, normals)
            if curvature is not None:
                curvatures.append(curvature)
                print(f"[DownBlock] Block {i + 1} curvature shape: {curvature.shape}")

        # 返回最后一个曲率值用于损失计算
        final_curvature = curvatures[-1] if curvatures else None
        print(
            f"[DownBlock] Output features shape: {x.features.shape}, final curvature: {final_curvature.shape if final_curvature is not None else 'None'}")
        return x, final_curvature


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
        print(f"[Model] PL-TopoConv enabled: {use_pl_topoconv}")
        print(f"[Model] MMCA enabled: {use_mmca}")
        if use_pl_topoconv:
            print(f"[Model] PL-TopoConv params: {pl_topoconv_kwargs}")
        if use_mmca:
            print(f"[Model] MMCA params: {mmca_kwargs}")

        self.use_pl_topoconv = use_pl_topoconv
        self.use_mmca = use_mmca
        self.pl_topoconv_kwargs = pl_topoconv_kwargs or {}
        self.mmca_kwargs = mmca_kwargs or {}

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
        print(f"\n[Model] Starting forward pass - input keys: {input_dict.keys()}")
        discrete_coord = input_dict["grid_coord"]
        feat = input_dict["feat"]
        offset = input_dict["offset"]
        batch = offset2batch(offset)
        print(
            f"[Model] Input feat shape: {feat.shape}, discrete_coord shape: {discrete_coord.shape}, batch size: {batch[-1].item() + 1}")

        # 提取多模态特征（坐标、颜色、法向量）
        coords = input_dict.get("coord", None)
        colors = input_dict.get("color", None)
        normals = input_dict.get("normal", None)
        print(f"[Model] Modalities - coords: {coords.shape if coords is not None else 'None'}, "
              f"colors: {colors.shape if colors is not None else 'None'}, "
              f"normals: {normals.shape if normals is not None else 'None'}")

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
        print(f"[Model] Initial SparseConvTensor - spatial shape: {x.spatial_shape}, batch size: {x.batch_size}")

        x = self.stem(x)
        print(f"[Model] After stem - features shape: {x.features.shape}")
        skips = [x]
        curvatures = []

        # 编码器前向传播
        for i in range(self.num_stages):
            print(f"\n[Model] Encoder stage {i + 1}/{self.num_stages}")
            x, curvature = self.enc[i](x, normals, coords, colors)
            skips.append(x)
            if curvature is not None:
                curvatures.append(curvature)
                print(f"[Model] Encoder stage {i + 1} curvature shape: {curvature.shape}")
            print(f"[Model] Encoder stage {i + 1} output features: {x.features.shape}")

        # 解码器前向传播
        x = skips.pop(-1)
        print(f"\n[Model] Starting decoder - initial feature shape: {x.features.shape}")
        for i in reversed(range(self.num_stages)):
            skip = skips.pop(-1)
            print(f"[Model] Decoder stage {i + 1}/{self.num_stages} - skip feat shape: {skip.features.shape}")
            x = self.dec[i](x, skip)
            print(f"[Model] Decoder stage {i + 1} output feat shape: {x.features.shape}")

        x = self.final(x)
        seg_logits = x.features  # 预测结果张量（核心输出）
        print(f"\n[Model] Final seg logits shape: {seg_logits.shape}")

        # 附加信息（曲率）放在字典中，通过input_dict传递给损失函数
        if curvatures:
            input_dict["curvatures"] = curvatures[-1]  # 将曲率存入输入字典
            print(f"[Model] Saved final curvature to input_dict - shape: {curvatures[-1].shape}")

        return seg_logits  # 主输出为预测张量，符合损失函数要求

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
