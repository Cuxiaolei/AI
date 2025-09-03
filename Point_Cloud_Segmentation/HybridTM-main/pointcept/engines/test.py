"""
Tester

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import time
import numpy as np
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
import csv

from .defaults import create_ddp_model
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
from pointcept.utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
)
from pointcept.utils.visualization import save_point_cloud
import open3d as o3d



TESTERS = Registry("testers")


class TesterBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.verbose = verbose
        if self.verbose:
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader

    def test(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)


@TESTERS.register_module()
class SemSegTester(TesterBase):
    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        # 新增：确保laspy库可用
        try:
            import laspy
        except ImportError:
            logger.error("Please install laspy first: pip install laspy")
            raise

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        # 新增：创建las文件保存目录
        las_save_path = os.path.join(save_path, "las")
        make_dirs(las_save_path)
        make_dirs(save_path)

        # 原有代码：创建提交文件夹（仅主进程）
        if (
                self.cfg.data.test.type in ["ScanNetDataset", "ScanNet200Dataset", "ScanNetPPDataset"]
                and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json
            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                    os.path.join(save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)
        comm.synchronize()

        record = {}
        num_classes = self.cfg.data.num_classes
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

        # 新增：用于存储点云坐标（根据不同数据集可能需要调整键名）
        coords_all = None

        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            if isinstance(data_dict, list):
                data_dict = data_dict[0]

            # 提取fragment_list并添加日志
            fragment_list = data_dict.pop("fragment_list")
            # 日志：fragment_list基本信息
            logger.debug(f"[{data_name}] fragment_list 长度: {len(fragment_list)}")
            if len(fragment_list) > 0:
                # 日志：第一个元素的类型和关键键名
                first_fragment = fragment_list[0]
                logger.debug(f"[{data_name}] 第一个fragment类型: {type(first_fragment)}")
                if isinstance(first_fragment, dict):
                    logger.debug(f"[{data_name}] 第一个fragment包含键: {list(first_fragment.keys())}")
                    # 检查是否有坐标相关的键
                    coord_keys = [k for k in first_fragment.keys() if k in ["coord", "points", "xyz"]]
                    logger.debug(f"[{data_name}] fragment中可能的坐标键: {coord_keys}")
                # 日志：最后一个元素的基本信息（避免过长）
                last_fragment = fragment_list[-1]
                logger.debug(f"[{data_name}] 最后一个fragment类型: {type(last_fragment)}")

            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))

            # 新增：初始化坐标存储
            coords_all = np.zeros((segment.size, 3), dtype=np.float32)

            if os.path.isfile(pred_save_path):
                logger.info(
                    f"{idx + 1}/{len(self.test_loader)}: {data_name}, loaded pred and label."
                )
                pred = np.load(pred_save_path)
                # 新增：从数据字典加载坐标（根据实际数据集键名调整）
                if "coord" in data_dict:
                    coords_all = data_dict["coord"].cpu().numpy()
                    logger.debug(f"[{data_name}] 从data_dict['coord']加载坐标，前5个点: {coords_all[:5]}")
                elif "points" in data_dict:  # 有些数据集可能用points存储坐标
                    coords_all = data_dict["points"][:, :3].cpu().numpy()
                    logger.debug(f"[{data_name}] 从data_dict['points']加载坐标，前5个点: {coords_all[:5]}")
                else:
                    # 新增：如果没有找到坐标键，明确输出日志
                    logger.warning(f"[{data_name}] data_dict中未找到'coord'或'points'键，无法加载坐标")

                if "origin_segment" in data_dict.keys():
                    segment = data_dict["origin_segment"]

            else:
                pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
                # 新增：初始化GPU坐标存储
                coords_gpu = torch.zeros((segment.size, 3), dtype=torch.float32).cuda()

                for i in range(len(fragment_list)):
                    fragment_batch_size = 1
                    s_i, e_i = i * fragment_batch_size, min(
                        (i + 1) * fragment_batch_size, len(fragment_list)
                    )
                    input_dict = self.__class__.collate_fn(fragment_list[s_i:e_i])
                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    idx_part = input_dict["index"]

                    # 新增：收集片段坐标
                    if "coord" in input_dict:
                        coords_part = input_dict["coord"]
                    elif "points" in input_dict:
                        coords_part = input_dict["points"][:, :3]
                    else:
                        raise ValueError("Could not find coordinate data in input_dict")

                    with torch.no_grad():
                        pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                        pred_part = F.softmax(pred_part, -1)
                        if self.cfg.empty_cache:
                            torch.cuda.empty_cache()
                        bs = 0
                        for be in input_dict["offset"]:
                            pred[idx_part[bs:be], :] += pred_part[bs:be]
                            # 新增：累加片段坐标
                            coords_gpu[idx_part[bs:be]] = coords_part[bs:be]
                            bs = be

                    logger.info(
                        f"Test: {idx + 1}/{len(self.test_loader)}-{data_name}, Batch: {i}/{len(fragment_list)}"
                    )

                # 新增：将GPU坐标转换为CPU并存储
                coords_all = coords_gpu.cpu().numpy()

                if self.cfg.data.test.type == "ScanNetPPDataset":
                    pred = pred.topk(3, dim=1)[1].data.cpu().numpy()
                else:
                    pred = pred.max(1)[1].data.cpu().numpy()

                if "origin_segment" in data_dict.keys():
                    assert "inverse" in data_dict.keys()
                    pred = pred[data_dict["inverse"]]
                    # 新增：同步坐标的inverse转换
                    coords_all = coords_all[data_dict["inverse"]]
                    segment = data_dict["origin_segment"]

                np.save(pred_save_path, pred)

            # 原有代码：生成提交文件
            if self.cfg.data.test.type in ["ScanNetDataset", "ScanNet200Dataset"]:
                np.savetxt(
                    os.path.join(save_path, "submit", f"{data_name}.txt"),
                    self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
                    fmt="%d",
                )
            elif self.cfg.data.test.type == "ScanNetPPDataset":
                np.savetxt(
                    os.path.join(save_path, "submit", f"{data_name}.txt"),
                    pred.astype(np.int32),
                    delimiter=",",
                    fmt="%d",
                )
                pred = pred[:, 0]  # 用于mIoU计算
            elif self.cfg.data.test.type == "SemanticKITTIDataset":
                sequence_name, frame_name = data_name.split("_")
                os.makedirs(
                    os.path.join(
                        save_path, "submit", "sequences", sequence_name, "predictions"
                    ),
                    exist_ok=True,
                )
                submit = pred.astype(np.uint32)
                submit = np.vectorize(
                    self.test_loader.dataset.learning_map_inv.__getitem__
                )(submit).astype(np.uint32)
                submit.tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "sequences",
                        sequence_name,
                        "predictions",
                        f"{frame_name}.label",
                    )
                )
            elif self.cfg.data.test.type == "NuScenesDataset":
                np.array(pred + 1).astype(np.uint8).tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "lidarseg",
                        "test",
                        f"{data_name}_lidarseg.bin",
                    )
                )

            # 新增：生成LAS文件（仅主进程）
            # 新增：生成LAS文件（仅主进程）
            if comm.is_main_process():
                las_filename = os.path.join(las_save_path, f"{data_name}.las")
                try:
                    # 1. 验证坐标数据有效性
                    if coords_all is None or len(coords_all) == 0:
                        logger.warning(f"点云坐标数据为空，无法生成LAS文件: {data_name}")
                        continue  # 跳过空数据

                    # 记录点云数量
                    point_count = len(coords_all)
                    logger.info(f"准备生成LAS文件: {las_filename}，包含 {point_count} 个点")

                    # 2. 计算坐标范围并记录
                    min_x, max_x = np.min(coords_all[:, 0]), np.max(coords_all[:, 0])
                    min_y, max_y = np.min(coords_all[:, 1]), np.max(coords_all[:, 1])
                    min_z, max_z = np.min(coords_all[:, 2]), np.max(coords_all[:, 2])

                    # 检查坐标是否有效（非NaN/Inf）
                    if np.isnan([min_x, max_x, min_y, max_y, min_z, max_z]).any():
                        logger.error(f"坐标数据包含无效值(NaN/Inf)，文件生成失败: {data_name}")
                        continue

                    # 记录坐标范围日志
                    logger.info(
                        f"点云坐标范围 - X: [{min_x:.2f}, {max_x:.2f}], Y: [{min_y:.2f}, {max_y:.2f}], Z: [{min_z:.2f}, {max_z:.2f}]")

                    # 3. 验证分类数据有效性
                    if pred is None or len(pred) != point_count:
                        logger.error(
                            f"分类数据与点云数量不匹配（点云: {point_count}, 分类: {len(pred) if pred is not None else 0}），文件生成失败: {data_name}")
                        continue

                    # 记录分类范围
                    min_class = np.min(pred)
                    max_class = np.max(pred)
                    logger.info(f"分类标签范围: [{min_class}, {max_class}]，共 {len(np.unique(pred))} 种类别")

                    # 4. 创建LAS文件并写入数据
                    header = laspy.LasHeader(point_format=3, version="1.2")
                    las = laspy.LasData(header)

                    las.x = coords_all[:, 0]
                    las.y = coords_all[:, 1]
                    las.z = coords_all[:, 2]

                    # 设置边界框元数据
                    header.min_x = min_x
                    header.max_x = max_x
                    header.min_y = min_y
                    header.max_y = max_y
                    header.min_z = min_z
                    header.max_z = max_z

                    # 设置分类信息
                    las.classification = pred.astype(np.uint8)

                    # 5. 保存文件并验证文件大小
                    las.write(las_filename)

                    # 检查文件是否生成成功
                    if os.path.exists(las_filename):
                        file_size = os.path.getsize(las_filename) / 1024  # 转换为KB
                        logger.info(f"LAS文件生成成功 - 大小: {file_size:.2f} KB，路径: {las_filename}")

                        # 验证文件点数量是否匹配
                        if len(las.points) == point_count:
                            logger.debug(f"文件点数量验证通过（预期: {point_count}, 实际: {len(las.points)}）")
                        else:
                            logger.warning(f"文件点数量不匹配（预期: {point_count}, 实际: {len(las.points)}）")
                    else:
                        logger.error(f"LAS文件生成失败，文件不存在: {las_filename}")

                except Exception as e:
                    logger.error(f"生成LAS文件时发生错误: {str(e)}", exc_info=True)  # 记录详细异常堆栈

            # 原有代码：计算评估指标
            intersection, union, target = intersection_and_union(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            # 原有代码：更新混淆矩阵
            valid_mask = segment != self.cfg.data.ignore_index
            valid_pred = pred[valid_mask]
            valid_segment = segment[valid_mask]
            indices = valid_segment * num_classes + valid_pred
            counts = np.bincount(indices, minlength=num_classes ** 2)
            confusion_matrix += counts.reshape(num_classes, num_classes)

            # 原有代码：计算并打印指标
            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask]) if mask.any() else 0.0
            acc = sum(intersection) / (sum(target) + 1e-10)
            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))
            batch_time.update(time.time() - end)

            logger.info(
                f"Test: {data_name} [{idx + 1}/{len(self.test_loader)}]-{segment.size} "
                f"Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                f"Accuracy {acc:.4f} ({m_acc:.4f}) "
                f"mIoU {iou:.4f} ({m_iou:.4f})"
            )

        # 以下为原有代码，保持不变
        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)
        confusion_matrix_list = comm.gather(confusion_matrix, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r

            total_confusion = np.sum(confusion_matrix_list, axis=0)
            confusion_norm = total_confusion / (total_confusion.sum(axis=1, keepdims=True) + 1e-10)

            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            if self.cfg.data.test.type == "S3DISDataset":
                torch.save(
                    dict(intersection=intersection, union=union, target=target),
                    os.path.join(save_path, f"{self.test_loader.dataset.split}.pth"),
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            csv_path = os.path.join(save_path, "test_results.csv")
            num_classes = self.cfg.data.num_classes
            class_names = self.cfg.data.names if hasattr(self.cfg.data, 'names') else [f"Class_{i}" for i in
                                                                                       range(num_classes)]
            if len(class_names) != num_classes:
                class_names = [f"Class_{i}" for i in range(num_classes)]
                logger.warning("Class names not properly defined in config, using default names")

            with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                header = ["Scene_Name"]
                for i in range(num_classes):
                    header.append(f"Class_{i}_IOU({class_names[i]})")
                for i in range(num_classes):
                    header.append(f"Class_{i}_ACC({class_names[i]})")
                header.extend(["Scene_mIoU", "Scene_OA"])
                writer.writerow(header)

                for scene_name, metrics in record.items():
                    inter = metrics["intersection"]
                    union = metrics["union"]
                    target = metrics["target"]
                    scene_iou = inter / (union + 1e-10)
                    scene_acc = inter / (target + 1e-10)
                    valid_mask = union != 0
                    valid_iou = scene_iou[valid_mask]
                    scene_miou = np.mean(valid_iou) if valid_iou.size > 0 else 0.0
                    scene_oa = np.sum(inter) / (np.sum(target) + 1e-10)
                    row = [scene_name]
                    row.extend([f"{x:.4f}" for x in scene_iou])
                    row.extend([f"{x:.4f}" for x in scene_acc])
                    row.extend([f"{scene_miou:.4f}", f"{scene_oa:.4f}"])
                    writer.writerow(row)

                avg_row = ["Global_Average"]
                avg_row.extend([f"{x:.4f}" for x in iou_class])
                avg_row.extend([f"{x:.4f}" for x in accuracy_class])
                avg_row.extend([f"{mIoU:.4f}", f"{allAcc:.4f}"])
                writer.writerow(avg_row)

            conf_matrix_path = os.path.join(save_path, "confusion_matrix.csv")
            with open(conf_matrix_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                header = ["Actual/Predicted"] + [f"Class_{i}({class_names[i]})" for i in range(num_classes)]
                writer.writerow(header)
                for i in range(num_classes):
                    row = [f"Class_{i}({class_names[i]})"] + total_confusion[i].tolist()
                    writer.writerow(row)

            conf_norm_path = os.path.join(save_path, "confusion_matrix_normalized.csv")
            with open(conf_norm_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                header = ["Actual/Predicted"] + [f"Class_{i}({class_names[i]})" for i in range(num_classes)]
                writer.writerow(header)
                for i in range(num_classes):
                    row = [f"Class_{i}({class_names[i]})"] + [f"{x:.4f}" for x in confusion_norm[i]]
                    writer.writerow(row)

            logger.info(f"Test results saved to CSV: {csv_path}")
            logger.info(f"Confusion matrix saved to: {conf_matrix_path}")
            logger.info(f"Normalized confusion matrix (proportion) saved to: {conf_norm_path}")
            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            for i in range(num_classes):
                logger.info(
                    f"Class_{i} - {class_names[i]} Result: iou/accuracy {iou_class[i]:.4f}/{accuracy_class[i]:.4f}"
                )
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        if len(batch) == 1 and isinstance(batch[0], dict) and "fragment_list" not in batch[0]:
            return batch[0]
        else:
            return batch


@TESTERS.register_module()
class ClsTester(TesterBase):
    def test(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        for i, input_dict in enumerate(self.test_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            end = time.time()
            with torch.no_grad():
                output_dict = self.model(input_dict)
            output = output_dict["cls_logits"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred, label, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            intersection_meter.update(intersection), union_meter.update(
                union
            ), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)

            logger.info(
                "Test: [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {accuracy:.4f} ".format(
                    i + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    accuracy=accuracy,
                )
            )

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU, mAcc, allAcc
            )
        )

        for i in range(self.cfg.data.num_classes):
            logger.info(
                "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=accuracy_class[i],
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


@TESTERS.register_module()
class ClsVotingTester(TesterBase):
    def __init__(
        self,
        num_repeat=100,
        metric="allAcc",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_repeat = num_repeat
        self.metric = metric
        self.best_idx = 0
        self.best_record = None
        self.best_metric = 0

    def test(self):
        for i in range(self.num_repeat):
            logger = get_root_logger()
            logger.info(f">>>>>>>>>>>>>>>> Start Evaluation {i + 1} >>>>>>>>>>>>>>>>")
            record = self.test_once()
            if comm.is_main_process():
                if record[self.metric] > self.best_metric:
                    self.best_record = record
                    self.best_idx = i
                    self.best_metric = record[self.metric]
                info = f"Current best record is Evaluation {i + 1}: "
                for m in self.best_record.keys():
                    info += f"{m}: {self.best_record[m]:.4f} "
                logger.info(info)

    def test_once(self):
        logger = get_root_logger()
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        target_meter = AverageMeter()
        record = {}
        self.model.eval()

        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            voting_list = data_dict.pop("voting_list")
            category = data_dict.pop("category")
            data_name = data_dict.pop("name")
            # pred = torch.zeros([1, self.cfg.data.num_classes]).cuda()
            # for i in range(len(voting_list)):
            #     input_dict = voting_list[i]
            #     for key in input_dict.keys():
            #         if isinstance(input_dict[key], torch.Tensor):
            #             input_dict[key] = input_dict[key].cuda(non_blocking=True)
            #     with torch.no_grad():
            #         pred += F.softmax(self.model(input_dict)["cls_logits"], -1)
            input_dict = collate_fn(voting_list)
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                pred = F.softmax(self.model(input_dict)["cls_logits"], -1).sum(
                    0, keepdim=True
                )
            pred = pred.max(1)[1].cpu().numpy()
            intersection, union, target = intersection_and_union(
                pred, category, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            target_meter.update(target)
            record[data_name] = dict(intersection=intersection, target=target)
            acc = sum(intersection) / (sum(target) + 1e-10)
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))
            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) ".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                )
            )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)
            accuracy_class = intersection / (target + 1e-10)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info("Val result: mAcc/allAcc {:.4f}/{:.4f}".format(mAcc, allAcc))
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        accuracy=accuracy_class[i],
                    )
                )
            return dict(mAcc=mAcc, allAcc=allAcc)

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class PartSegTester(TesterBase):
    def test(self):
        test_dataset = self.test_loader.dataset
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()

        num_categories = len(self.test_loader.dataset.categories)
        iou_category, iou_count = np.zeros(num_categories), np.zeros(num_categories)
        self.model.eval()

        save_path = os.path.join(
            self.cfg.save_path, "result", "test_epoch{}".format(self.cfg.test_epoch)
        )
        make_dirs(save_path)

        for idx in range(len(test_dataset)):
            end = time.time()
            data_name = test_dataset.get_data_name(idx)

            data_dict_list, label = test_dataset[idx]
            pred = torch.zeros((label.size, self.cfg.data.num_classes)).cuda()
            batch_num = int(np.ceil(len(data_dict_list) / self.cfg.batch_size_test))
            for i in range(batch_num):
                s_i, e_i = i * self.cfg.batch_size_test, min(
                    (i + 1) * self.cfg.batch_size_test, len(data_dict_list)
                )
                input_dict = collate_fn(data_dict_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                with torch.no_grad():
                    pred_part = self.model(input_dict)["cls_logits"]
                    pred_part = F.softmax(pred_part, -1)
                if self.cfg.empty_cache:
                    torch.cuda.empty_cache()
                pred_part = pred_part.reshape(-1, label.size, self.cfg.data.num_classes)
                pred = pred + pred_part.total(dim=0)
                logger.info(
                    "Test: {} {}/{}, Batch: {batch_idx}/{batch_num}".format(
                        data_name,
                        idx + 1,
                        len(test_dataset),
                        batch_idx=i,
                        batch_num=batch_num,
                    )
                )
            pred = pred.max(1)[1].data.cpu().numpy()

            category_index = data_dict_list[0]["cls_token"]
            category = self.test_loader.dataset.categories[category_index]
            parts_idx = self.test_loader.dataset.category2part[category]
            parts_iou = np.zeros(len(parts_idx))
            for j, part in enumerate(parts_idx):
                if (np.sum(label == part) == 0) and (np.sum(pred == part) == 0):
                    parts_iou[j] = 1.0
                else:
                    i = (label == part) & (pred == part)
                    u = (label == part) | (pred == part)
                    parts_iou[j] = np.sum(i) / (np.sum(u) + 1e-10)
            iou_category[category_index] += parts_iou.mean()
            iou_count[category_index] += 1

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} "
                "({batch_time.avg:.3f}) ".format(
                    data_name, idx + 1, len(self.test_loader), batch_time=batch_time
                )
            )

        ins_mIoU = iou_category.sum() / (iou_count.sum() + 1e-10)
        cat_mIoU = (iou_category / (iou_count + 1e-10)).mean()
        logger.info(
            "Val result: ins.mIoU/cat.mIoU {:.4f}/{:.4f}.".format(ins_mIoU, cat_mIoU)
        )
        for i in range(num_categories):
            logger.info(
                "Class_{idx}-{name} Result: iou_cat/num_sample {iou_cat:.4f}/{iou_count:.4f}".format(
                    idx=i,
                    name=self.test_loader.dataset.categories[i],
                    iou_cat=iou_category[i] / (iou_count[i] + 1e-10),
                    iou_count=int(iou_count[i]),
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)

@TESTERS.register_module()
class SemSegVisualization(TesterBase):
    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Visualization >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        # self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "visual_result")
        make_dirs(save_path)
        comm.is_main_process()
        assert self.cfg.data.test.type == "ScanNetDataset" or "NuScenesDataset", "Wrong Dataset"

        comm.synchronize()
        record = {}
        for idx, data_dict in enumerate(self.test_loader.dataset):
            end = time.time()

            # data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")

            pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
            if os.path.isfile("sdm" + pred_save_path):
                logger.info(
                    "{}/{}: {}, loaded pred and label.".format(
                        idx + 1, len(self.test_loader), data_name
                    )
                )
                pred = np.load(pred_save_path)
                if "origin_segment" in data_dict.keys():
                    segment = data_dict["origin_segment"]
            else:
                pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
                coords_all = torch.zeros((segment.size, 3)).cuda()
                for i in range(len(fragment_list)):
                    fragment_batch_size = 1
                    s_i, e_i = i * fragment_batch_size, min(
                        (i + 1) * fragment_batch_size, len(fragment_list)
                    )
                    input_dict = collate_fn(fragment_list[s_i:e_i])
                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    idx_part = input_dict["index"]
                    with torch.no_grad():
                        pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                        pred_part = F.softmax(pred_part, -1)
                        if self.cfg.empty_cache:
                            torch.cuda.empty_cache()
                        bs = 0
                        for be in input_dict["offset"]:
                            pred[idx_part[bs:be], :] += pred_part[bs:be]
                            coords_all[[idx_part[bs:be]]] = input_dict['coord']
                            bs = be

                    logger.info(
                        "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                            idx + 1,
                            len(self.test_loader),
                            data_name=data_name,
                            batch_idx=i,
                            batch_num=len(fragment_list),
                        )
                    )
                if self.cfg.data.test.type == "ScanNetPPDataset":
                    pred = pred.topk(3, dim=1)[1].data.cpu().numpy()
                else:
                    pred = pred.max(1)[1].data.cpu().numpy()
                if "origin_segment" in data_dict.keys():
                    assert "inverse" in data_dict.keys()
                    pred = pred[data_dict["inverse"]]
                    coords_all = coords_all[data_dict["inverse"]]
                    segment = data_dict["origin_segment"]
                # np.save(pred_save_path, pred)

            # import ipdb; ipdb.set_trace()
            # save_point_cloud(coords_all, pred, file_path=save_path + "pred.ply")
            # save_point_cloud(segment, file_path=save_path + "pred.ply")
            colors = [
                [0.1, 0.2, 0.3],   # 蓝色调
                [0.3, 0.1, 0.4],   # 紫色调
                [0.4, 0.5, 0.1],   # 黄色调
                [0.6, 0.3, 0.2],   # 红色调
                [0.2, 0.8, 0.4],   # 绿调
                [0.7, 0.1, 0.6],   # 粉色调
                [0.1, 0.7, 0.8],   # 天蓝色调
                [0.9, 0.9, 0.1],   # 浅黄色
                [0.3, 0.7, 0.2],   # 草绿色
                [0.6, 0.1, 0.1],   # 深红色
                [0.5, 0.5, 0.5],   # 灰色
                [0.8, 0.6, 0.3],   # 沙棕色
                [0.2, 0.3, 0.9],   # 深蓝色
                [0.7, 0.9, 0.1],   # 黄绿色
                [0.3, 0.5, 0.9],   # 海洋蓝
                [0.4, 0.7, 0.5],   # 绿松石色
                [0.8, 0.3, 0.5],   # 玫瑰色
                [0.9, 0.4, 0.1],   # 橙色
                [0.1, 0.9, 0.7],   # 明亮的薄荷绿
                [0.3, 0.3, 0.6],   # 暗紫色
            ]
            mask = np.where(segment != -1)[0]

            colors_pred = np.array(colors)[pred[mask]]
            colors_gt = np.array(colors)[segment[mask]]

            # import ipdb; ipdb.set_trace()
            if True:
                points = coords_all.cpu().numpy()[mask]
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(points)
                point_cloud.paint_uniform_color([0.5, 0.5, 0.5])
                point_cloud.colors = o3d.utility.Vector3dVector(colors_gt)
                # vis = o3d.visualization.Visualizer()
                # vis.create_window()
                # for point_cloud in point_clouds:
                #     vis.add_geometry(point_cloud)
                # vis.get_render_option().point_size = 2
                # vis.run()
                # vis.destroy_window()
                o3d.io.write_point_cloud(save_path + f"/pred_{idx}.ply", point_cloud, write_ascii=False, compressed=False, print_progress=False)
                # o3d.visualization.draw_geometries([point_cloud],
                #                   zoom=0.3412,
                #                   front=[0.4257, -0.2125, -0.8795],
                #                   lookat=[2.6172, 2.0475, 1.532],
                #                   up=[-0.0694, -0.9768, 0.2024])
            if idx > 100:
                exit()


@TESTERS.register_module()
class SemSegVisualizationLoad(TesterBase):
    def test(self):
        assert self.cfg.data.test.type == "NuScenesDataset"
        pass