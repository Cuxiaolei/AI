import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from ..build import DATASETS
from ..data_util import crop_pc, voxelize
from ...transforms.point_transform_cpu import PointsToTensor
import glob
from tqdm import tqdm
import logging
import pickle


# 引入与S3DISTower一致的辅助函数
def _ensure_2d(arr, name):
    """确保数组是二维的"""
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    return arr


def _sanitize_numeric(coord, color, normal):
    """修改：支持同时清洗坐标、颜色、法向量，去掉 NaN/Inf 并填 0"""
    mask_c = np.isfinite(coord).all(axis=1)  # 坐标有效性掩码
    mask_col = np.isfinite(color).all(axis=1) if color is not None else np.ones(len(coord), dtype=bool)  # 颜色有效性掩码
    mask_norm = np.isfinite(normal).all(axis=1) if normal is not None else np.ones(len(coord), dtype=bool)  # 法向量有效性掩码
    mask = mask_c & mask_col & mask_norm  # 总有效掩码

    if not mask.all():
        coord = coord[mask]
        color = color[mask] if color is not None else None
        normal = normal[mask] if normal is not None else None

    # 填充无效值为0
    coord = np.nan_to_num(coord, nan=0.0, posinf=0.0, neginf=0.0)
    if color is not None:
        color = np.nan_to_num(color, nan=0.0, posinf=0.0, neginf=0.0)
    if normal is not None:
        normal = np.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)

    return coord, color, normal, mask


def _limit_points(coord, color, normal, label, voxel_max, split, sample_name):
    """修改：支持同时限制坐标、颜色、法向量、标签的点数到 voxel_max"""
    if voxel_max is None or voxel_max <= 0:
        return coord, color, normal, label

    n = coord.shape[0]
    if n > voxel_max:
        choice = np.random.choice(n, voxel_max, replace=False)
        coord = coord[choice]
        if color is not None:
            color = color[choice]
        if normal is not None:
            normal = normal[choice]
        if label is not None:
            label = label[choice]

        if split != 'train':
            print(f"[{split}] limit points: {sample_name} {n} -> {voxel_max}")

    return coord, color, normal, label


@DATASETS.register_module()
class ScanNet(Dataset):
    classes = ['Tower_Insulator', 'Background', 'Conductor']
    num_classes = 3
    num_per_class = np.array([0, 0, 0], dtype=np.int32)
    class2color = {
        'Tower_Insulator': [0, 255, 0],
        'Background': [0, 0, 255],
        'Conductor': [255, 255, 0]
    }
    cmap = [*class2color.values()]
    gravity_dim = 2

    def __init__(self,
                 data_root='/root/autodl-tmp/data/data_scannet_tower',
                 split='train',
                 voxel_size=0.04,
                 voxel_max=None,
                 transform=None,
                 loop=1,
                 presample=False,
                 variable=False,
                 shuffle: bool = True,
                 n_shifted=1
                 ):
        super().__init__()
        self.split = split
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.transform = transform
        self.presample = presample
        self.variable = variable
        self.loop = loop
        self.n_shifted = n_shifted
        self.shuffle = shuffle
        self.pipe_transform = PointsToTensor()

        # 加载数据列表（保持多文件加载方式）
        if split in ["train", "val", "test"]:
            self.data_list = glob.glob(os.path.join(data_root, split, "scene*"))
        elif split == 'trainval':
            self.data_list = glob.glob(os.path.join(data_root, "train", "scene*")) + \
                             glob.glob(os.path.join(data_root, "val", "scene*"))
        else:
            raise ValueError("no such split: {}".format(split))

        # 提取样本名
        self.sample_names = [osp.basename(path) for path in self.data_list]
        logging.info(
            f"[{split}] Found {len(self.data_list)} scenes: {self.sample_names[:5]}{'...' if len(self.data_list) > 5 else ''}")

        processed_root = os.path.join(data_root, 'processed')
        os.makedirs(processed_root, exist_ok=True)
        self.pkl_path = os.path.join(
            processed_root, f'scannet_{split}_{voxel_size:.3f}_with_normal.pkl'
        )
        logging.info(f"[{split}] Cache pkl path: {self.pkl_path}")

        # 预采样逻辑（缓存结构不变，仅加载时拆分）
        if self.presample and not os.path.exists(self.pkl_path):
            np.random.seed(0)
            self.data = []
            for idx, scene_path in enumerate(tqdm(self.data_list, desc=f'Loading ScanNet {split} split with normal')):
                sample_name = self.sample_names[idx]
                # 多文件加载
                coord = np.load(os.path.join(scene_path, 'coord.npy')).astype(np.float32)
                feat_color = np.load(os.path.join(scene_path, 'color.npy')).astype(np.float32)
                feat_normal = np.load(os.path.join(scene_path, 'normal.npy')).astype(np.float32)
                label = np.load(os.path.join(scene_path, 'segment20.npy')).astype(np.float32)

                # 坐标原点对齐
                coord -= np.min(coord, 0)

                # 体素下采样
                if voxel_size:
                    uniq_idx = voxelize(coord, voxel_size)
                    coord, feat_color, feat_normal, label = coord[uniq_idx], feat_color[uniq_idx], feat_normal[
                        uniq_idx], label[uniq_idx]

                # RGB归一化
                if feat_color.max() > 1.0:
                    feat_color = feat_color / 255.0
                feat_color = np.clip(feat_color, 0.0, 1.0)

                # 合并特征用于缓存（保持原缓存结构，避免重新生成）
                combined_feat = np.hstack([feat_color, feat_normal])
                cdata = np.hstack((coord, combined_feat, np.expand_dims(label, -1))).astype(np.float32)
                self.data.append(cdata)

            # 统计点数
            npoints = np.array([len(arr) for arr in self.data])
            logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' % (
                self.split, np.median(npoints), np.average(npoints), np.std(npoints)))
            with open(self.pkl_path, 'wb') as f:
                pickle.dump(self.data, f)
                print(f"{self.pkl_path} saved successfully")
        elif self.presample:
            with open(self.pkl_path, 'rb') as f:
                self.data = pickle.load(f)
                print(f"{self.pkl_path} load successfully")

        # 索引与长度
        self.data_idx = np.arange(len(self.data_list))
        assert len(self.data_idx) > 0, f"No samples found for split={split}. Check data root."
        logging.info(f"Totally {len(self.data_idx)} samples in {split} set")

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        sample_name = self.sample_names[data_idx]

        if self.presample:
            # 加载缓存数据：拆分出coord、color、normal、label（不合并特征）
            cdata = self.data[data_idx]
            # 缓存结构：3(coord) + 3(color) + 3(normal) + 1(label)
            coord = cdata[:, :3].astype(np.float32)
            feat_color = cdata[:, 3:6].astype(np.float32)  # 单独的颜色特征（3维）
            feat_normal = cdata[:, 6:9].astype(np.float32)  # 单独的法向量（3维）
            label = cdata[:, 9:].astype(np.float32)
        else:
            # 直接加载多文件
            scene_path = self.data_list[data_idx]
            coord = np.load(os.path.join(scene_path, 'coord.npy')).astype(np.float32)
            feat_color = np.load(os.path.join(scene_path, 'color.npy')).astype(np.float32)
            feat_normal = np.load(os.path.join(scene_path, 'normal.npy')).astype(np.float32)
            label = np.load(os.path.join(scene_path, 'segment20.npy')).astype(np.float32)

            # 坐标原点对齐
            coord -= np.min(coord, 0)

            # 合并用于crop（保持点对应关系，crop后立即拆分）
            temp_feat = np.hstack([feat_color, feat_normal])
            coord, temp_feat, label = crop_pc(
                coord, temp_feat, label, self.split, self.voxel_size, self.voxel_max,
                downsample=not self.presample, variable=self.variable, shuffle=self.shuffle)
            # crop后拆分回单独的color和normal
            feat_color = temp_feat[:, :3]
            feat_normal = temp_feat[:, 3:6]

        # RGB归一化（确保范围正确）
        if feat_color.max() > 1.0:
            feat_color = feat_color / 255.0
        feat_color = np.clip(feat_color, 0.0, 1.0)

        # 法向量可选归一化（保持原注释逻辑）
        # feat_normal = np.clip((feat_normal + 1) / 2.0, 0.0, 1.0)

        # -------------------------- 关键修改1：单独校验各特征维度 --------------------------
        # 坐标校验（3列）
        coord = _ensure_2d(coord, "coord")
        if coord.shape[1] != 3:
            raise ValueError(f"[{self.split}] {sample_name}: coord must have 3 columns, got {coord.shape[1]}")

        # 颜色特征校验（3列：RGB）
        feat_color = _ensure_2d(feat_color, "feat_color")
        if feat_color.shape[1] != 3:
            raise ValueError(
                f"[{self.split}] {sample_name}: feat_color must have 3 columns (RGB), got {feat_color.shape[1]}")

        # 法向量校验（3列）
        feat_normal = _ensure_2d(feat_normal, "feat_normal")
        if feat_normal.shape[1] != 3:
            raise ValueError(
                f"[{self.split}] {sample_name}: feat_normal must have 3 columns, got {feat_normal.shape[1]}")

        # 标签校验（1列）
        if label is not None:
            label = _ensure_2d(label, "label")
            if label.shape[1] != 1:
                raise ValueError(f"[{self.split}] {sample_name}: label must have 1 column, got {label.shape[1]}")

        # RGB通道调试信息（保留原逻辑）
        for ch_idx, ch_name in enumerate(["R", "G", "B"]):
            unique_vals = np.unique(feat_color[:, ch_idx])
            if len(unique_vals) == 1:
                print(f"[Debug][{self.split}] {sample_name} {ch_name} 通道恒为 {unique_vals[0]}，可能缺失原始值")

        # -------------------------- 关键修改2：单独清洗color和normal --------------------------
        # 数值清洗（去掉NaN/Inf）
        coord, feat_color, feat_normal, mask = _sanitize_numeric(coord, feat_color, feat_normal)
        if label is not None:
            label = label[mask]

        # -------------------------- 关键修改3：单独限制color和normal点数 --------------------------
        # 限制点数
        coord, feat_color, feat_normal, label = _limit_points(
            coord, feat_color, feat_normal, label, self.voxel_max, self.split, sample_name)

        # 安全检查（补充normal和color的校验）
        if coord.shape[0] < 1:
            raise ValueError(f"[{self.split}] sample {sample_name} has no points after processing")
        if not (np.isfinite(coord).all() and np.isfinite(feat_color).all() and np.isfinite(feat_normal).all()):
            raise ValueError(f"[{self.split}] sample {sample_name} contains NaN/Inf values")

        # -------------------------- 关键修改4：transform前拆分存储（x=color，normal单独） --------------------------
        # 构建数据字典：transform前不合并特征，color作为x，normal单独存储
        data = {
            'pos': coord.astype(np.float32),  # 坐标
            'x': feat_color.astype(np.float32),  # x仅为颜色特征（3维）
            'normal': feat_normal.astype(np.float32),  # 法向量单独存储（3维）
            'y': label.squeeze(-1).astype(np.long)  # y保持为标签（不混淆特征与标签）
        }

        # -------------------------- 应用transform --------------------------
        if self.transform is not None:
            data = self.transform(data)

        # -------------------------- 关键修改5：transform后合并color和normal --------------------------
        # 合并特征（3维color + 3维normal = 6维特征）
        # 注意：原需求提到“合并为y”，但y是标签，此处调整为合并为特征x（避免覆盖标签）
        # 若需强制合并到y，请将'data["x"]'改为'data["y"]'并删除原y，但会导致标签丢失（不推荐）
        data['x'] = np.hstack([data['x'], data['normal']])
        # 移除单独的normal键（特征已合并，无需保留）
        del data['normal']

        # 转换为tensor（保持原逻辑）
        data = self.pipe_transform(data)

        return data

    def __len__(self):
        return len(self.data_idx) * self.loop