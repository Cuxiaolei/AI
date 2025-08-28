from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from timm.models.layers import trunc_normal_
from ..builder import MODELS, LOSSES
from ..utils import offset2batch
from torch_geometric.nn.pool import voxel_grid
from torch_geometric.utils import scatter


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
        K = 32
        B = batch.max().item() + 1
        linearity = []
        density = []

        for b in range(B):
            mask = batch == b
            if mask.sum() < K:
                linearity.append(torch.zeros(mask.sum(), device=feat.device))
                density.append(torch.zeros(mask.sum(), device=feat.device))
                continue

            points_b = coord[mask]
            dist = torch.cdist(points_b, points_b)
            _, idx = torch.topk(dist, K + 1, largest=False)
            idx = idx[:, 1:K + 1]

            pca_features = []
            for i in range(points_b.shape[0]):
                neighbors = points_b[idx[i]]
                centered = neighbors - neighbors.mean(dim=0)
                cov = torch.matmul(centered.T, centered) / (K - 1)
                cov_float = cov.to(torch.float32)  # 修复半精度SVD问题
                eigenvalues = torch.svd(cov_float).S
                eigenvalues = eigenvalues.to(feat.dtype)
                eigenvalues = eigenvalues / eigenvalues.sum()
                pca_features.append(eigenvalues)

            pca_features = torch.stack(pca_features)
            linearness = pca_features[:, 0] - (pca_features[:, 1] + pca_features[:, 2])
            linearity.append(linearness)

            # 修复索引不匹配问题
            row_indices = torch.arange(points_b.shape[0], device=dist.device).unsqueeze(1)
            neighbor_dists = dist[row_indices, idx]
            mean_dist = neighbor_dists.mean(dim=1)
            density_val = 1.0 / (mean_dist + 1e-6)
            density.append(density_val)

        linearity = torch.cat(linearity, dim=0)
        density = torch.cat(density, dim=0)

        feat_logits = self.feature_judge(feat)
        feat_probs = F.softmax(feat_logits, dim=1)

        # 计算三类别的概率
        tower_prob = (density.unsqueeze(1) * 2.0 + feat_probs[:, 0:1]) / 3.0
        background_prob = (torch.maximum(1.0 - linearity.unsqueeze(1), 1.0 - density.unsqueeze(1)) + feat_probs[:,
                                                                                                     1:2]) / 3.0
        line_prob = (linearity.unsqueeze(1) * 2.0 + feat_probs[:, 2:3]) / 3.0

        # 生成动态网格大小
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

        return grid_sizes


# 策略2: 跨模态电力特征增强（CMPFE）
class CMPFEModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # 特征投影到9通道（匹配坐标3+颜色3+法向量3）
        self.feature_projection = nn.Sequential(
            nn.Linear(in_channels, 9),
            nn.BatchNorm1d(9),
            nn.ReLU()
        )

        # 颜色特征注意力机制
        self.color_attention = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Sigmoid()
        )

        # 法向量特征注意力机制
        self.normal_attention = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Sigmoid()
        )

        # 特征融合模块
        self.feature_fusion = nn.Sequential(
            nn.Linear(9, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels)
        )

        # 电力设施语义注意力
        self.semantic_attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 投影到9通道特征
        projected_feat = self.feature_projection(x)

        # 按实际数据顺序分离特征（坐标→颜色→法向量）
        coord_feat = projected_feat[:, :3]
        color_feat = projected_feat[:, 3:6]  # 处理颜色特征（原数据3-5通道）
        normal_feat = projected_feat[:, 6:9]  # 处理法向量特征（原数据6-8通道）

        # 特征增强
        enhanced_color = color_feat * self.color_attention(color_feat)
        enhanced_normal = normal_feat * self.normal_attention(normal_feat)

        # 特征融合与注意力加权
        enhanced_feat = torch.cat([coord_feat, enhanced_color, enhanced_normal], dim=1)
        fused_feat = self.feature_fusion(enhanced_feat)
        sem_att = self.semantic_attention(fused_feat)
        final_feat = fused_feat * sem_att + x * (1 - sem_att)  # 残差连接

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

        # 集成CMPFE模块
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

        # 集成PFAS模块
        self.pfas = PFASModule(embed_channels, grid_size)

    def forward(self, x, clusters=None):
        # 应用CMPFE特征增强
        enhanced_feat = self.cmpfe(x.features)
        x = x.replace_feature(enhanced_feat)
        feat = x.features

        # 应用PFAS生成动态网格
        coord = x.indices[:, 1:].float()
        batch = x.indices[:, 0]
        dynamic_grid_sizes = self.pfas(feat, coord, batch)

        if dynamic_grid_sizes.numel() == 0:
            return x

        # 生成代表性网格
        grid_mean = dynamic_grid_sizes.mean(dim=1)
        representative_grids = [
            torch.mean(dynamic_grid_sizes[torch.argsort(grid_mean)[100:200]], dim=0)
            if dynamic_grid_sizes.shape[0] > 200 else grid_mean.mean() * 0.8,
            torch.mean(dynamic_grid_sizes[torch.argsort(grid_mean, descending=True)[:100]], dim=0)
            if dynamic_grid_sizes.shape[0] > 100 else grid_mean.mean() * 1.2,
            torch.mean(dynamic_grid_sizes[torch.argsort(grid_mean)[:100]], dim=0)
            if dynamic_grid_sizes.shape[0] > 100 else grid_mean.mean() * 0.5
        ]

        # 生成多尺度聚类
        clusters = []
        for grid_size in representative_grids:
            cluster = voxel_grid(pos=coord, size=torch.clamp(grid_size, min=1e-6).tolist(), batch=batch)
            _, cluster = torch.unique(cluster, return_inverse=True)
            clusters.append(cluster)

        # 多尺度特征融合
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
        x = self.down(x)
        for block in self.blocks:
            x = block(x)
        return x


# 策略3: 电力线连续性约束对比学习（PLCCL）
@LOSSES.register_module()
class PLCCLoss(nn.Module):
    def __init__(self, temperature=0.1, gamma=0.5, loss_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma
        self.loss_weight = loss_weight

    def forward(self, features, labels, coords):
        # 电力线标签为第2类（索引2）
        line_mask = labels == 2
        if line_mask.sum() < 2:
            return torch.tensor(0.0, device=features.device) * self.loss_weight

        line_feats = features[line_mask]
        line_coords = coords[line_mask]

        # 特征相似度计算
        feat_sim = F.cosine_similarity(line_feats.unsqueeze(1), line_feats.unsqueeze(0), dim=2)

        # 空间距离计算
        coord_dist = torch.cdist(line_coords, line_coords)
        pos_mask = (coord_dist < 1.0) & (coord_dist > 1e-6)  # 近邻点作为正样本
        neg_mask = ~pos_mask

        # InfoNCE损失
        exp_sim = torch.exp(feat_sim / self.temperature)
        sum_exp = exp_sim * neg_mask.float()
        sum_exp = sum_exp.sum(dim=1, keepdim=True)
        pos_exp = exp_sim * pos_mask.float()
        pos_sum = pos_exp.sum(dim=1)
        info_nce_loss = -torch.log(pos_sum / (sum_exp.squeeze() + pos_sum + 1e-6)).mean()

        # 连续性约束损失
        feat_dist = 1 - feat_sim
        continuity_loss = torch.mean(torch.abs(feat_dist - coord_dist) * pos_mask.float())

        return (info_nce_loss + self.gamma * continuity_loss) * self.loss_weight


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
            # 电力设施网格配置：[电力塔, 背景, 电力线]
            point_grid_size=[[[3, 3, 3], [10, 10, 10], [1, 1, 5]],
                             [[3, 3, 3], [10, 10, 10], [1, 1, 5]],
                             [[3, 3, 3], [10, 10, 10], [1, 1, 5]],
                             [[3, 3, 3], [10, 10, 10], [1, 1, 5]]],
            dec_depth=[2, 2, 2, 2],
            # PLCCL参数
            plccl_temperature=0.1,
            plccl_gamma=0.5,
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
        self.plccl_loss = PLCCLoss(temperature=plccl_temperature, gamma=plccl_gamma)

        self.apply(self._init_weights)

    def forward(self, input_dict, labels=None):
        discrete_coord = input_dict["grid_coord"]
        feat = input_dict["feat"]  # 10通道输入：坐标3+颜色3+法向量3+标签1
        offset = input_dict["offset"]
        batch = offset2batch(offset)

        x = spconv.SparseConvTensor(
            features=feat,
            indices=torch.cat([batch.unsqueeze(-1), discrete_coord], dim=1).int().contiguous(),
            spatial_shape=torch.add(torch.max(discrete_coord, dim=0).values, 1).tolist(),
            batch_size=batch[-1].tolist() + 1,
        )

        x = self.stem(x)
        skips = [x]
        intermediate_features = [x.features.detach()]

        for i in range(self.num_stages):
            x = self.enc[i](x)
            skips.append(x)
            intermediate_features.append(x.features.detach())

        x = skips.pop(-1)
        for i in reversed(range(self.num_stages)):
            skip = skips.pop(-1)
            x = self.dec[i](x, skip)

        x = self.final(x)
        output = x.features

        # 训练时计算PLCCL损失
        if self.training and labels is not None:
            coord = discrete_coord[:labels.shape[0]] if discrete_coord.shape[0] != labels.shape[0] else discrete_coord
            plccl_loss = self.plccl_loss(intermediate_features[-2], labels, coord)
            return output, plccl_loss
        else:
            return output

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
