import __init__
import argparse, yaml, os, logging, numpy as np, csv, wandb, glob
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port, \
    load_checkpoint_inv
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys, get_class_weights
from openpoints.dataset.data_util import voxelize
from openpoints.dataset.semantic_kitti.semantickitti import load_label_kitti, load_pc_kitti, remap_lut_read, \
    remap_lut_write, get_semantickitti_file_list
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def check_class_distribution(loader, split_name, num_classes):
    """统计 dataloader 内的标签分布并打印百分比（适配配置文件的类别数）"""
    all_labels = []
    for batch in loader:
        y = batch['y'].cpu().numpy().ravel()
        all_labels.append(y)
    if len(all_labels) == 0:
        print(f"[Debug][{split_name}] 没有样本")
        return

    all_labels = np.concatenate(all_labels)
    uniq, cnts = np.unique(all_labels, return_counts=True)
    total = all_labels.shape[0]

    # 打印数量
    dist = dict(zip(uniq.tolist(), cnts.tolist()))
    print(f"\n[Debug][{split_name}] 标签分布(数量): {dist}")

    # 打印百分比
    perc = {cls: round(100.0 * dist.get(cls, 0) / total, 2) for cls in range(num_classes)}
    print(f"[Debug][{split_name}] 标签分布(百分比): {perc} (总点数={total})")


def write_to_csv(oa, macc, miou, ious, best_epoch, cfg, write_header=True, area=5):
    ious_table = [f'{item:.2f}' for item in ious]
    header = ['method', 'Area', 'OA', 'mACC', 'mIoU'] + [f'cls_{i}' for i in range(cfg.num_classes)] + ['best_epoch',
                                                                                                        'log_path',
                                                                                                        'wandb link']
    data = [cfg.cfg_basename, str(area), f'{oa:.2f}', f'{macc:.2f}',
            f'{miou:.2f}'] + ious_table + [str(best_epoch), cfg.run_dir,
                                           wandb.run.get_url() if cfg.wandb.use_wandb else '-']
    with open(cfg.csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)
        f.close()


def save_epoch_results(epoch, train_metrics, val_metrics, csv_path):
    """保存每个epoch的训练和验证指标到CSV"""
    header = [
                 'epoch', 'train_loss', 'train_miou', 'train_macc', 'train_oa',
                 'val_miou', 'val_macc', 'val_oa'
             ] + [f'val_iou_cls_{i}' for i in range(len(val_metrics.get("ious", [])))]

    row = [
        epoch,
        f"{train_metrics['loss']:.4f}",
        f"{train_metrics['miou']:.4f}",
        f"{train_metrics['macc']:.4f}",
        f"{train_metrics['oa']:.4f}",
        f"{val_metrics['miou']:.4f}" if val_metrics else '',
        f"{val_metrics['macc']:.4f}" if val_metrics else '',
        f"{val_metrics['oa']:.4f}" if val_metrics else ''
    ]
    if val_metrics and 'ious' in val_metrics:
        row += [f"{iou:.4f}" for iou in val_metrics['ious']]

    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def generate_data_list(cfg):
    """适配ScanNet数据集的数据列表生成"""
    if 's3dis' in cfg.dataset.common.NAME.lower():
        list_file = os.path.join(cfg.dataset.common.data_root, 'test_scenes.txt')
        if not os.path.isfile(list_file):
            raise FileNotFoundError(f"Missing test scene file: {list_file}")
        logging.info(f"[test] Using test scene list file: {list_file}")

        data_list = []
        with open(list_file, "r") as f:
            for line in f.readlines():
                name = line.strip()
                if not name:
                    continue
                if name.startswith("merged/"):
                    name = name.split("/")[-1]
                if name.endswith(".npy"):
                    name = name[:-4]
                npy_path = os.path.join(cfg.dataset.common.data_root, 'merged', name + ".npy")
                data_list.append(npy_path)

        logging.info(f"[test] Found {len(data_list)} test scenes: {data_list[:5]}{'...' if len(data_list) > 5 else ''}")


    elif 'scannet' in cfg.dataset.common.NAME.lower():
        # 适配配置文件中的数据路径和split
        split = cfg.dataset.test.split
        data_list = glob.glob(os.path.join(cfg.dataset.common.data_root, split, "*.pth"))
        logging.info(
            f"[test] Found {len(data_list)} ScanNet {split} scenes in {os.path.join(cfg.dataset.common.data_root, split)}")
        # 新增：打印验证集数据列表前5个，确认路径正确
        logging.debug(f"[Debug][generate_data_list] 验证集前5个文件: {data_list[:5]}")


    elif 'semantickitti' in cfg.dataset.common.NAME.lower():
        if cfg.dataset.test.split == 'val':
            split_no = 1
        else:
            split_no = 2
        data_list = get_semantickitti_file_list(os.path.join(cfg.dataset.common.data_root, 'sequences'),
                                                str(cfg.dataset.test.test_id + 11))[split_no]
    else:
        raise Exception(f'dataset {cfg.dataset.common.NAME} not supported yet')
    return data_list


def load_data(data_path, cfg):
    """适配ScanNet的数据加载格式，增强调试日志"""
    label, feat = None, None
    dataset_name = cfg.dataset.common.NAME.lower()
    logging.debug(f"[Debug][load_data] 加载数据: {os.path.basename(data_path)} (数据集: {dataset_name})")

    if 's3dis' in dataset_name:
        data = np.load(data_path)  # xyzrgbl, N*7
        coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]
        feat = np.clip(feat / 255., 0, 1).astype(np.float32)
    elif 'scannet' in dataset_name:
        # 新增：验证数据加载是否成功
        if not os.path.exists(data_path):
            logging.error(f"[Error][load_data] 数据文件不存在: {data_path}")
        try:
            data = torch.load(data_path)  # (coord, feat, label)
        except Exception as e:
            logging.error(f"[Error][load_data] 加载数据失败: {data_path}, 错误: {str(e)}")
            raise

        # 新增：检查数据结构
        if len(data) != 3:
            logging.error(f"[Error][load_data] 数据结构错误，预期(coord, feat, label)，实际长度: {len(data)}")

        coord = data[0]  # 坐标 (3D)
        rgb = data[1][:, :3]  # 颜色 (3D)
        normal = data[1][:, 3:6]  # 法向量 (3D)

        # 新增：特征统计信息
        logging.debug(f"[Debug][load_data] RGB范围: min={rgb.min()}, max={rgb.max()}, mean={rgb.mean()}")
        logging.debug(f"[Debug][load_data] 法向量范围: min={normal.min()}, max={normal.max()}, mean={normal.mean()}")

        # 适配配置文件中的特征预处理
        rgb = np.clip(rgb / 255., 0, 1).astype(np.float32)  # 假设RGB范围[0,255]
        feat = np.hstack([rgb, normal])  # 颜色+法向量，共6维
        label = data[2]  # 新增：显式加载标签

        # 新增：验证集标签分布详细日志
        if 'val' in cfg.mode or 'test' in cfg.mode:
            uniq, cnts = np.unique(label, return_counts=True)
            logging.debug(f"[Debug][load_data][验证集] 标签值: {uniq}, 数量: {cnts}, 总点数: {label.shape[0]}")

    elif 'semantickitti' in dataset_name:
        coord = load_pc_kitti(data_path[0])
        if cfg.dataset.test.split != 'test':
            label = load_label_kitti(data_path[1], remap_lut_read)

    # 新增：坐标归一化前后检查
    coord_orig = coord.copy()
    coord -= coord.min(0)
    logging.debug(
        f"[Debug][load_data] 坐标归一化: 原始范围={[coord_orig.min(0), coord_orig.max(0)]}, 归一化后范围={[coord.min(0), coord.max(0)]}")

    idx_points = []
    voxel_idx, reverse_idx_part, reverse_idx_sort = None, None, None
    voxel_size = cfg.dataset.common.get('voxel_size', None)

    if voxel_size is not None:
        # 适配配置文件中的体素化参数
        idx_sort, voxel_idx, count = voxelize(coord, voxel_size, mode=1)
        if cfg.get('test_mode', 'multi_voxel') == 'nearest_neighbor':
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
            idx_part = idx_sort[idx_select]
            npoints_subcloud = voxel_idx.max() + 1
            idx_shuffle = np.random.permutation(npoints_subcloud)
            idx_part = idx_part[idx_shuffle]
            reverse_idx_part = np.argsort(idx_shuffle, axis=0)
            idx_points.append(idx_part)
            reverse_idx_sort = np.argsort(idx_sort, axis=0)
        else:
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                np.random.shuffle(idx_part)
                idx_points.append(idx_part)
    else:
        idx_points.append(np.arange(coord.shape[0]))

    # 调试标签分布
    if label is not None:
        uniq, cnts = np.unique(label, return_counts=True)
        logging.debug(
            f"[Debug][load_data] {os.path.basename(data_path)} 标签分布: {dict(zip(uniq.tolist(), cnts.tolist()))}")

    # 新增：特征和坐标维度检查
    logging.debug(f"[Debug][load_data] 特征维度: {feat.shape if feat is not None else 'None'}, 坐标维度: {coord.shape}")
    return coord, feat, label, idx_points, voxel_idx, reverse_idx_part, reverse_idx_sort


def main(gpu, cfg):
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier()

    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0:
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        writer = SummaryWriter(log_dir=cfg.run_dir) if cfg.is_training else None
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    # 适配模型输入通道配置
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')

    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    # 构建验证集数据加载器（适配配置文件）
    val_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=cfg.distributed
                                           )

    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
    num_classes = val_loader.dataset.num_classes if hasattr(val_loader.dataset, 'num_classes') else cfg.num_classes
    assert cfg.num_classes == num_classes, f"配置文件num_classes({cfg.num_classes})与数据集不匹配配"
    logging.info(f"number of classes of the dataset: {num_classes}")
    cfg.classes = val_loader.dataset.classes if hasattr(val_loader.dataset, 'classes') else np.arange(num_classes)
    cfg.cmap = np.array(val_loader.dataset.cmap) if hasattr(val_loader.dataset, 'cmap') else None
    validate_fn = validate if 'sphere' not in cfg.dataset.common.NAME.lower() else validate_sphere

    # 新增：验证集整体标签分布检查
    logging.info("[Debug][main] 开始统计验证集标签分布...")
    check_class_distribution(val_loader, "val", num_classes=num_classes)

    model_module = model.module if hasattr(model, 'module') else model

    # 模型加载逻辑
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler, pretrained_path=cfg.pretrained_path)
        else:
            if cfg.mode == 'val':
                best_epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
                val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(model, val_loader, cfg, num_votes=1,
                                                                             epoch=0)
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f'Best ckpt @E{best_epoch},  val_oa , val_macc, val_miou: {val_oa:.2f} {val_macc:.2f} {val_miou:.2f}, '
                        f'\niou per cls is: {val_ious}')
                return val_miou
            elif cfg.mode == 'test':
                best_epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
                data_list = generate_data_list(cfg)
                logging.info(f"length of test dataset: {len(data_list)}")
                test_miou, test_macc, test_oa, test_ious, test_accs, _ = test(model, data_list, cfg)

                if test_miou is not None:
                    with np.printoptions(precision=2, suppress=True):
                        logging.info(
                            f'Best ckpt @E{best_epoch},  test_oa , test_macc, test_miou: {test_oa:.2f} {test_macc:.2f} {test_miou:.2f}, '
                            f'\niou per cls is: {test_ious}')
                    cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + '_test.csv')
                    write_to_csv(test_oa, test_macc, test_miou, test_ious, best_epoch, cfg)
                return test_miou
            elif 'encoder' in cfg.mode:
                if 'inv' in cfg.mode:
                    logging.info(f'Finetuning from {cfg.pretrained_path}')
                    load_checkpoint_inv(model.encoder, cfg.pretrained_path)
                else:
                    logging.info(f'Finetuning from {cfg.pretrained_path}')
                    load_checkpoint(model_module.encoder, cfg.pretrained_path, cfg.get('pretrained_module', None))
            else:
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model, cfg.pretrained_path, cfg.get('pretrained_module', None))
    else:
        logging.info('Training from scratch')

    if 'freeze_blocks' in cfg.mode:
        for p in model_module.encoder.blocks.parameters():
            p.requires_grad = False

    # 构建训练集数据加载器（适配配置文件）
    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             )
    check_class_distribution(train_loader, "train", num_classes=cfg.num_classes)  # 适配配置的类别数
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")

    # 构建损失函数（适配配置文件）
    cfg.criterion_args.weight = None
    if cfg.get('cls_weighed_loss', False):
        if hasattr(train_loader.dataset, 'num_per_class'):
            cfg.criterion_args.weight = get_class_weights(train_loader.dataset.num_per_class, normalize=True)
        else:
            logging.info('`num_per_class` attribute is not founded in dataset')
    criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()

    # 初始化每个epoch的结果保存路径
    epoch_csv_path = os.path.join(cfg.run_dir, f'{cfg.run_name}_epoch_results.csv')

    if cfg.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    val_miou, val_macc, val_oa, val_ious, val_accs = 0., 0., 0., [], []
    best_val, macc_when_best, oa_when_best, ious_when_best, best_epoch = 0., 0., 0., [], 0
    total_iter = 0
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):
            train_loader.dataset.epoch = epoch - 1

        # 训练一个epoch
        train_loss, train_miou, train_macc, train_oa, _, _, total_iter = \
            train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, total_iter, cfg)

        # 收集训练指标
        train_metrics = {
            'loss': train_loss,
            'miou': train_miou,
            'macc': train_macc,
            'oa': train_oa
        }

        # 验证并收集验证指标
        val_metrics = None
        is_best = False
        if epoch % cfg.val_freq == 0:
            val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(model, val_loader, cfg, epoch=epoch,
                                                                         total_iter=total_iter)
            val_metrics = {
                'loss': 0.0,  # 验证阶段无损失，可设为0
                'oa': val_oa,
                'macc': val_macc,
                'miou': val_miou,
                'ious': val_ious,
                'accs': val_accs
            }
            # 保存验证指标到CSV
            if cfg.rank == 0:
                epoch_csv_path = os.path.join(cfg.run_dir, f'{cfg.run_name}_epoch_metrics.csv')
                save_epoch_metrics(epoch, val_metrics, epoch_csv_path, is_train=False)

            if val_miou > best_val:
                is_best = True
                best_val = val_miou
                macc_when_best = val_macc
                oa_when_best = val_oa
                ious_when_best = val_ious
                best_epoch = epoch
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f'Find a better ckpt @E{epoch}, val_miou {val_miou:.2f} val_macc {macc_when_best:.2f}, val_oa {oa_when_best:.2f}'
                        f'\nmious: {val_ious}')

        # 保存当前epoch的指标到CSV
        if cfg.rank == 0:  # 只在主进程保存
            save_epoch_results(epoch, train_metrics, val_metrics, epoch_csv_path)

        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train_miou {train_miou:.2f}, val_miou {val_miou:.2f}, best val miou {best_val:.2f}')
        if writer is not None:
            writer.add_scalar('best_val', best_val, epoch)
            writer.add_scalar('val_miou', val_miou, epoch)
            writer.add_scalar('macc_when_best', macc_when_best, epoch)
            writer.add_scalar('oa_when_best', oa_when_best, epoch)
            writer.add_scalar('val_macc', val_macc, epoch)
            writer.add_scalar('val_oa', val_oa, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_miou', train_miou, epoch)
            writer.add_scalar('train_macc', train_macc, epoch)
            writer.add_scalar('lr', lr, epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)

        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val': best_val},
                            is_best=is_best
                            )
            is_best = False

    with np.printoptions(precision=2, suppress=True):
        logging.info(
            f'Best ckpt @E{best_epoch},  val_oa {oa_when_best:.2f}, val_macc {macc_when_best:.2f}, val_miou {best_val:.2f}, '
            f'\niou per cls is: {ious_when_best}')

    if cfg.world_size < 2:
        load_checkpoint(model, pretrained_path=os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
        cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + f'.csv')
        if 'sphere' in cfg.dataset.common.NAME.lower():
            test_miou, test_macc, test_oa, test_ious, test_accs = validate_sphere(model, val_loader, cfg, epoch=0)
        else:
            data_list = generate_data_list(cfg)
            test_miou, test_macc, test_oa, test_ious, test_accs, _ = test(model, data_list, cfg)

        with np.printoptions(precision=2, suppress=True):
            logging.info(
                f'Best ckpt @E{best_epoch},  test_oa {test_oa:.2f}, test_macc {test_macc:.2f}, test_miou {test_miou:.2f}, '
                f'\niou per cls is: {test_ious}')
        if writer is not None:
            writer.add_scalar('test_miou', test_miou, 0)
            writer.add_scalar('test_macc', test_macc, 0)
            writer.add_scalar('test_oa', test_oa, 0)
        write_to_csv(test_oa, test_macc, test_miou, test_ious, best_epoch, cfg, write_header=True)
        logging.info(f'save results in {cfg.csv_path}')
        if cfg.use_voting:
            load_checkpoint(model, pretrained_path=os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
            set_random_seed(cfg.seed)
            val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(model, val_loader, cfg, num_votes=20,
                                                                         data_transform=None, epoch=0)
            if writer is not None:
                writer.add_scalar('val_miou20', val_miou, cfg.epochs + 50)

            ious_table = [f'{item:.2f}' for item in val_ious]
            data = [cfg.cfg_basename, 'True', f'{val_oa:.2f}', f'{val_macc:.2f}', f'{val_miou:.2f}'] + ious_table + [
                str(best_epoch), cfg.run_dir]

            with open(cfg.csv_path, 'a', encoding='UT8') as f:  # 改为追加模式避免覆盖
                writer = csv.writer(f)
                writer.writerow(data)
    else:
        logging.warning(
            'Testing using multiple GPUs is not allowed for now. Running testing after this training is required.')
    if writer is not None:
        writer.close()
    wandb.finish(exit_code=True)


# 配置日志
logging.basicConfig(
    level=logging.DEBUG,  # 改为DEBUG级别，输出更详细日志
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def save_epoch_metrics(epoch, metrics, csv_path, is_train=True):
    """
    保存单轮训练/验证的所有指标到CSV
    :param epoch: 轮次
    :param metrics: 包含各类指标的字典
    :param csv_path: 保存路径
    :param is_train: 是否为训练指标
    """
    # 构建表头（首次写入时）
    header = ['epoch', 'type']  # type区分train/val
    # 添加整体指标
    header.extend(['loss', 'OA', 'mACC', 'mIoU'])
    # 添加每个类别的IoU和Acc
    num_classes = len(metrics['ious'])
    for i in range(num_classes):
        header.extend([f'cls_{i}_IoU', f'cls_{i}_Acc'])

    # 构建数据行
    row = [epoch, 'train' if is_train else 'val']
    row.extend([
        f"{metrics['loss']:.4f}",
        f"{metrics['oa']:.4f}",
        f"{metrics['macc']:.4f}",
        f"{metrics['miou']:.4f}"
    ])
    # 添加每个类别的指标
    for i in range(num_classes):
        row.extend([
            f"{metrics['ious'][i]:.4f}",
            f"{metrics['accs'][i]:.4f}"
        ])

    # 写入CSV
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def log_metrics(epoch, metrics, is_train=True):
    """打印指标到日志"""
    phase = "Training" if is_train else "Validation"
    logger.info(f"\n===== {phase} Epoch {epoch} Metrics =====")
    logger.info(f"Loss: {metrics['loss']:.4f}")
    logger.info(f"Overall Accuracy (OA): {metrics['oa']:.4f}")
    logger.info(f"Mean Accuracy (mACC): {metrics['macc']:.4f}")
    logger.info(f"Mean IoU (mIoU): {metrics['miou']:.4f}")

    # 打印每个类别的指标
    logger.info("\nPer-class metrics:")
    logger.info(f"{'Class':<6} {'IoU':<10} {'Accuracy':<10}")
    logger.info("-" * 28)
    for i, (iou, acc) in enumerate(zip(metrics['ious'], metrics['accs'])):
        logger.info(f"Class {i:<2} {iou:.4f}      {acc:.4f}")
    logger.info("=====================================\n")


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, total_iter, cfg):
    loss_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    model.train()
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0

    # 关键：确认feature_keys配置（与数据加载器一致）
    assert cfg.feature_keys in [['pos', 'x'], 'pos,x'], \
        f"feature_keys配置错误，需为'pos,x'，当前为{cfg.feature_keys}"

    for idx, data in pbar:
        # 1. 数据转移到GPU（覆盖所有必要键：pos、x、y，可选heights）
        keys = data.keys() if callable(data.keys) else data.keys()
        for key in keys:
            if key in ['pos', 'x', 'y', 'heights']:  # 确保关键特征转移
                data[key] = data[key].cuda(non_blocking=True)
        num_iter += 1
        target = data['y'].squeeze(-1)  # 标签形状：(B*N,)

        # 新增：训练集标签分布调试
        target_np = target.cpu().numpy().ravel()
        target_uniq, target_cnt = np.unique(target_np, return_counts=True)
        logging.debug(f"[Debug][train][Batch {idx}] 标签分布: {dict(zip(target_uniq.tolist(), target_cnt.tolist()))}")

        # 2. 特征提取：按feature_keys拼接pos和x（生成9维输入）
        data['x'] = get_features_by_keys(data, cfg.feature_keys)

        # 3. 维度强制校验（确保输入是9维）
        assert data['x'].shape[1] == 9, \
            f"训练阶段输入特征维度错误！需为9维（pos3+x6），当前为{data['x'].shape[1]}维"

        # 新增：特征统计信息
        logging.debug(
            f"[Debug][train][Batch {idx}] 特征统计: 均值={data['x'].mean().item():.4f}, 标准差={data['x'].std().item():.4f}, 最小值={data['x'].min().item():.4f}, 最大值={data['x'].max().item():.4f}")

        # 4. 传递训练元信息（供数据增强或模型使用）
        data['epoch'] = epoch
        total_iter += 1
        data['iter'] = total_iter

        # 5. 模型推理与损失计算
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            logits = model(data)
            # 调试：打印当前batch的输入形状和预测分布
            logging.debug(
                f"[Train][Epoch {epoch}][Batch {idx}] "
                f"输入特征 shape: {tuple(data['x'].shape)} (B, C, N) | "
                f"logits shape: {tuple(logits.shape)} | "
                f"target shape: {tuple(target.shape)}"
            )
            # 预测分布调试
            pred = logits.argmax(dim=1)
            pred_hist = torch.bincount(pred.view(-1), minlength=cfg.num_classes).cpu().numpy()
            logging.debug(f"[Debug][train] 当前 batch 预测分布: {dict(enumerate(pred_hist))}")

            # 损失计算（与原逻辑一致）
            loss = criterion(logits, target) if 'mask' not in cfg.criterion_args.NAME.lower() \
                else criterion(logits, target, data.get('mask', None))

        # 6. 梯度反向传播与优化（原逻辑不变）
        if cfg.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0

            if cfg.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        # 7. 指标更新（原逻辑不变）
        cm.update(logits.argmax(dim=1), target)
        loss_meter.update(loss.item())

        if idx % cfg.print_freq == 0:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} Acc {cm.overall_accuray:.2f}")

    miou, macc, oa, ious, accs = cm.all_metrics()

    # 构建训练指标字典
    train_metrics = {
        'loss': loss_meter.avg,
        'oa': oa,
        'macc': macc,
        'miou': miou,
        'ious': ious,
        'accs': accs
    }

    # 输出指标到日志
    log_metrics(epoch, train_metrics, is_train=True)

    # 保存训练指标到CSV
    if cfg.rank == 0:  # 只在主进程保存
        epoch_csv_path = os.path.join(cfg.run_dir, f'{cfg.run_name}_epoch_metrics.csv')
        save_epoch_metrics(epoch, train_metrics, epoch_csv_path, is_train=True)

    return loss_meter.avg, miou, macc, oa, ious, accs, total_iter


@torch.no_grad()
def validate(model, val_loader, cfg, num_votes=1, data_transform=None, epoch=-1, total_iter=-1):
    model.eval()
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__(), desc='Val')

    # 关键：确认feature_keys配置
    assert cfg.feature_keys in [['pos', 'x'], 'pos,x'], \
        f"feature_keys配置错误，需为'pos,x'，当前为{cfg.feature_keys}"

    for idx, data in pbar:
        # 1. 数据转移到GPU（与训练一致，覆盖关键键）
        keys = data.keys() if callable(data.keys) else data.keys()
        for key in keys:
            if key in ['pos', 'x', 'y', 'heights']:
                data[key] = data[key].cuda(non_blocking=True)
        target = data['y'].squeeze(-1)

        # 新增：验证集标签分布调试
        target_np = target.cpu().numpy().ravel()
        target_uniq, target_cnt = np.unique(target_np, return_counts=True)
        logging.debug(f"[Debug][val][Batch {idx}] 标签分布: {dict(zip(target_uniq.tolist(), target_cnt.tolist()))}")

        # 2. 特征提取：按feature_keys拼接pos和x（生成9维输入）
        data['x'] = get_features_by_keys(data, cfg.feature_keys)

        # 3. 维度强制校验
        assert data['x'].shape[1] == 9, \
            f"验证阶段输入特征维度错误！需为9维（pos3+x6），当前为{data['x'].shape[1]}维"

        # 新增：验证集特征统计
        logging.debug(
            f"[Debug][val][Batch {idx}] 特征统计: 均值={data['x'].mean().item():.4f}, 标准差={data['x'].std().item():.4f}, 最小值={data['x'].min().item():.4f}, 最大值={data['x'].max().item():.4f}")

        # 4. 传递元信息（供模型使用）
        data['epoch'] = epoch
        data['iter'] = total_iter

        # 5. 模型推理（无梯度计算）
        logits = model(data)
        # 调试：打印验证阶段输入形状和logits统计
        logging.debug(
            f"[Val][Batch {idx}] "
            f"输入特征 shape: {tuple(data['x'].shape)} (B, C, N) | "
            f"logits shape: {tuple(logits.shape)} | "
            f"logits统计: 均值={logits.mean().item():.4f}, 最大值={logits.max().item():.4f}"
        )

        # 新增：验证集预测分布调试
        pred = logits.argmax(dim=1)
        pred_hist = torch.bincount(pred.view(-1), minlength=cfg.num_classes).cpu().numpy()
        logging.debug(f"[Debug][val][Batch {idx}] 预测分布: {dict(enumerate(pred_hist))}")

        # 6. 混淆矩阵更新（原逻辑不变）
        if 'mask' not in cfg.criterion_args.NAME or cfg.get('use_maks', False):
            cm.update(logits.argmax(dim=1), target)
        else:
            mask = data['mask'].bool()
            cm.update(logits.argmax(dim=1)[mask], target[mask])

    # 7. 指标计算（原逻辑不变）
    tp, union, count = cm.tp, cm.union, cm.count
    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)
    val_miou, val_macc, val_oa, val_ious, val_accs = get_mious(tp, union, count)

    val_metrics = {
        'loss': 0.0,  # 验证阶段无损失，可设为0或实际计算的验证损失
        'oa': val_oa,
        'macc': val_macc,
        'miou': val_miou,
        'ious': val_ious,
        'accs': val_accs
    }

    # 输出验证指标到日志
    log_metrics(epoch, val_metrics, is_train=False)

    return val_miou, val_macc, val_oa, val_ious, val_accs


@torch.no_grad()
def validate_sphere(model, val_loader, cfg, num_votes=1, data_transform=None, epoch=-1, total_iter=-1):
    model.eval()
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    if cfg.get('visualize', False):
        from openpoints.dataset.vis3d import write_obj
        cfg.vis_dir = os.path.join(cfg.run_dir, 'visualization')
        os.makedirs(cfg.vis_dir, exist_ok=True)
        cfg.cmap = cfg.cmap.astype(np.float32) / 255.

    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    all_logits, idx_points = [], []

    # 关键：确认feature_keys配置
    assert cfg.feature_keys in [['pos', 'x'], 'pos,x'], \
        f"feature_keys配置错误，需为'pos,x'，当前为{cfg.feature_keys}"

    for idx, data in pbar:
        # 1. 数据转移到GPU
        for key in data.keys():
            if key in ['pos', 'x', 'y', 'heights', 'input_inds']:  # 包含球形采样所需的input_inds
                data[key] = data[key].cuda(non_blocking=True)

        # 新增：球形验证标签分布
        target = data['y'].squeeze(-1)
        target_np = target.cpu().numpy().ravel()
        target_uniq, target_cnt = np.unique(target_np, return_counts=True)
        logging.debug(
            f"[Debug][validate_sphere][Batch {idx}] 标签分布: {dict(zip(target_uniq.tolist(), target_cnt.tolist()))}")

        # 2. 特征提取：按feature_keys拼接pos和x
        data['x'] = get_features_by_keys(data, cfg.feature_keys)
        # 3. 维度校验
        assert data['x'].shape[1] == 9, \
            f"球形验证输入特征维度错误！需为9维（pos3+x6），当前为{data['x'].shape[1]}维"

        # 新增：球形验证特征统计
        logging.debug(
            f"[Debug][validate_sphere][Batch {idx}] 特征统计: 均值={data['x'].mean().item():.4f}, 标准差={data['x'].std().item():.4f}")

        # 4. 模型推理
        data['epoch'] = epoch
        data['iter'] = total_iter
        logits = model(data)

        # 新增：球形验证预测分布
        pred = logits.argmax(dim=1)
        pred_hist = torch.bincount(pred.view(-1), minlength=cfg.num_classes).cpu().numpy()
        logging.debug(f"[Debug][validate_sphere][Batch {idx}] 预测分布: {dict(enumerate(pred_hist))}")

        all_logits.append(logits)
        idx_points.append(data['input_inds'])

    # 后续合并预测、计算指标的逻辑不变（仅确保特征维度正确）
    all_logits = torch.cat(all_logits, dim=0).transpose(1, 2).reshape(-1, cfg.num_classes)
    idx_points = torch.cat(idx_points, dim=0).flatten()

    if cfg.distributed:
        dist.all_reduce(all_logits), dist.all_reduce(idx_points)

    all_logits = scatter(all_logits, idx_points, dim=0, reduce='mean')
    all_logits = all_logits.argmax(dim=1)
    val_points_labels = torch.from_numpy(val_loader.dataset.clouds_points_labels[0]).squeeze(-1).to(all_logits.device)
    val_points_projections = torch.from_numpy(val_loader.dataset.projections[0]).to(all_logits.device).long()
    val_points_preds = all_logits[val_points_projections]

    # 新增：合并后预测分布
    pred_hist = torch.bincount(val_points_preds.view(-1), minlength=cfg.num_classes).cpu().numpy()
    logging.debug(f"[Debug][validate_sphere] 合并后预测分布: {dict(enumerate(pred_hist))}")
    # 新增：合并后标签分布
    label_hist = torch.bincount(val_points_labels.view(-1), minlength=cfg.num_classes).cpu().numpy()
    logging.debug(f"[Debug][validate_sphere] 合并后标签分布: {dict(enumerate(label_hist))}")

    del all_logits, idx_points
    torch.cuda.empty_cache()

    cm.update(val_points_preds, val_points_labels)
    miou, macc, oa, ious, accs = cm.all_metrics()

    if cfg.get('visualize', False):
        dataset_name = cfg.dataset.common.NAME.lower()
        coord = val_loader.dataset.clouds_points[0]
        colors = val_loader.dataset.clouds_points_colors[0].astype(np.float32)
        gt = val_points_labels.cpu().numpy().squeeze()
        pred = val_points_preds.cpu().numpy().squeeze()
        gt = cfg.cmap[gt, :] if gt is not None else None
        pred = cfg.cmap[pred, :]
        rooms = val_loader.dataset.clouds_rooms[0]

        for idx in tqdm(range(len(rooms) - 1), desc='save visualization'):
            start_idx, end_idx = rooms[idx], rooms[idx + 1]
            write_obj(coord[start_idx:end_idx], colors[start_idx:end_idx],
                      os.path.join(cfg.vis_dir, f'input-{dataset_name}-{idx}.obj'))
            if gt is not None:
                write_obj(coord[start_idx:end_idx], gt[start_idx:end_idx],
                          os.path.join(cfg.vis_dir, f'gt-{dataset_name}-{idx}.obj'))
            write_obj(coord[start_idx:end_idx], pred[start_idx:end_idx],
                      os.path.join(cfg.vis_dir, f'{cfg.cfg_basename}-{dataset_name}-{idx}.obj'))
    return miou, macc, oa, ious, accs


@torch.no_grad()
def test(model, data_list, cfg, num_votes=1):
    """适配ScanNet修改后数据加载器的测试逻辑（feature_keys: pos,x）"""
    model.eval()
    all_cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    set_random_seed(0)
    cfg.visualize = cfg.get('visualize', False)
    if cfg.visualize:
        from openpoints.dataset.vis3d import write_obj
        cfg.vis_dir = os.path.join(cfg.run_dir, 'visualization')
        os.makedirs(cfg.vis_dir, exist_ok=True)
        cfg.cmap = cfg.cmap.astype(np.float32) / 255.

    # 适配配置文件中的数据变换（保持与训练一致的预处理）
    trans_split = 'val' if cfg.datatransforms.get('test', None) is None else 'test'
    pipe_transform = build_transforms_from_cfg(trans_split, cfg.datatransforms)
    logging.debug(f"[Debug][test] 使用的数据变换: {trans_split}, 变换管道: {pipe_transform}")

    dataset_name = cfg.dataset.common.NAME.lower()
    len_data = len(data_list)

    # 适配保存路径
    cfg.save_path = cfg.get('save_path', f'results/{cfg.task_name}/{cfg.dataset.test.split}/{cfg.cfg_basename}')
    if 'semantickitti' in dataset_name:
        cfg.save_path = os.path.join(cfg.save_path, str(cfg.dataset.test.test_id + 11), 'predictions')
    os.makedirs(cfg.save_path, exist_ok=True)

    gravity_dim = cfg.datatransforms.kwargs.get('gravity_dim', 2)  # 从配置获取重力方向维度
    nearest_neighbor = cfg.get('test_mode', 'multi_voxel') == 'nearest_neighbor'

    # 关键：确认feature_keys配置（确保是pos,x）
    assert cfg.feature_keys == ['pos', 'x'] or cfg.feature_keys == 'pos,x', \
        f"feature_keys配置错误，需为'pos,x'，当前为{cfg.feature_keys}"

    for cloud_idx, data_path in enumerate(data_list):
        logging.info(f'Test [{cloud_idx}]/[{len_data}] cloud: {os.path.basename(data_path)}')
        cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
        all_logits = []

        # 1. 加载数据（调用修改后的数据加载逻辑，确保pos和x的正确提取）
        # （剩余代码保持不变，此处省略）
        # 1. 加载数据（调用修改后的数据加载逻辑，确保pos和x的正确提取）
        coord, feat, label, idx_points, voxel_idx, reverse_idx_part, reverse_idx = load_data(data_path, cfg)

        # 数据合法性校验（确保pos和x的维度正确）
        assert coord.ndim == 2 and coord.shape[1] == 3, f"坐标维度错误，需为(N,3)，当前为{coord.shape}"
        assert feat.ndim == 2 and feat.shape[1] == 6, f"特征x维度错误，需为(N,6)（RGB+法向量），当前为{feat.shape}"
        if label is not None:
            label = torch.from_numpy(label.astype(np.int).squeeze()).cuda(non_blocking=True)
            gt_unique, gt_counts = np.unique(label.cpu().numpy(), return_counts=True)
            logging.info(
                f"[Debug][test][cloud {cloud_idx}] GT 分布: {dict(zip(gt_unique.tolist(), gt_counts.tolist()))}")

        len_part = len(idx_points)
        nearest_neighbor = len_part == 1
        pbar = tqdm(range(len(idx_points)), desc=f"Cloud {cloud_idx} subclouds")
        for idx_subcloud in pbar:
            pbar.set_description(f"Test on {cloud_idx}-th cloud [{idx_subcloud}]/[{len_part}]")
            if not (nearest_neighbor and idx_subcloud > 0):
                # 2. 提取子点云的pos和x特征
                idx_part = idx_points[idx_subcloud]
                pos_part = coord[idx_part].astype(np.float32)  # pos: (N_sub, 3)
                x_part = feat[idx_part].astype(np.float32)  # x: (N_sub, 6)（RGB+法向量）

                # 坐标原点对齐（与训练数据加载器保持一致：减去子点云最小值）
                pos_part -= pos_part.min(0)

                # 3. 构建数据字典（包含pos和x，与训练时的输入结构完全匹配）
                data = {
                    'pos': pos_part,  # 子点云坐标
                    'x': x_part  # 子点云特征（RGB+法向量）
                }

                # 4. 应用测试阶段的数据变换（如颜色归一化、Tensor转换）
                if pipe_transform is not None:
                    data = pipe_transform(data)

                # 处理高度特征（如果配置需要，保持与训练一致）
                if 'heights' in cfg.feature_keys and 'heights' not in data.keys():
                    if 'semantickitti' in dataset_name:
                        data['heights'] = torch.from_numpy(
                            (pos_part[:, gravity_dim:gravity_dim + 1] - pos_part[:, gravity_dim:gravity_dim + 1].min())
                            .astype(np.float32)
                        ).unsqueeze(0)
                    else:
                        data['heights'] = torch.from_numpy(
                            pos_part[:, gravity_dim:gravity_dim + 1].astype(np.float32)
                        ).unsqueeze(0)

                # 5. 调整数据形状（适配非variable模式：添加batch维度）
                if not cfg.dataset.common.get('variable', False):
                    data['pos'] = data['pos'].unsqueeze(0)  # (1, N_sub, 3)
                    data['x'] = data['x'].unsqueeze(0)  # (1, N_sub, 6)
                    if 'heights' in data:
                        data['heights'] = data['heights'].unsqueeze(0)
                else:
                    # variable模式：添加点云长度和batch标识
                    data['o'] = torch.IntTensor([len(pos_part)])
                    data['batch'] = torch.LongTensor([0] * len(pos_part))

                # 6. 数据转移到GPU（非阻塞加载）
                for key in data.keys():
                    data[key] = data[key].cuda(non_blocking=True)

                # 7. 关键：根据feature_keys提取并拼接特征（pos+x，共9维）
                # get_features_by_keys会自动按cfg.feature_keys顺序拼接pos和x
                data['x'] = get_features_by_keys(data, cfg.feature_keys)
                # 特征维度校验（确保是9维：pos(3)+x(6)）
                assert data['x'].shape[1] == 9, \
                    f"模型输入特征维度错误，需为9维（pos3+x6），当前为{data['x'].shape[1]}维"

                # 8. 模型推理
                logits = model(data)

                # 9. 调试信息：输入特征形状和预测分布
                logging.info(
                    f"[Debug][test][cloud {cloud_idx} sub {idx_subcloud}] "
                    f"输入特征 shape: {tuple(data['x'].shape)} (B, C, N) | "
                    f"logits shape: {tuple(logits.shape)}"
                )
                # 数值健康检查
                has_nan = torch.isnan(logits).any().item()
                logit_min = logits.min().item()
                logit_max = logits.max().item()
                logging.info(
                    f"[Debug][test][cloud {cloud_idx} sub {idx_subcloud}] "
                    f"logits nan: {has_nan}, min: {logit_min:.4f}, max: {logit_max:.4f}"
                )
                # 子点云预测分布
                pred_sub = logits.argmax(dim=1).squeeze().detach().cpu().numpy()
                u_pred, c_pred = np.unique(pred_sub, return_counts=True)
                logging.info(
                    f"[Debug][test][cloud {cloud_idx} sub {idx_subcloud}] "
                    f"预测分布: {dict(zip(u_pred.tolist(), c_pred.tolist()))}"
                )
                # GT子片分布（如有标签）
                if label is not None:
                    gt_sub = label[idx_part].detach().cpu().numpy()
                    u_gt, c_gt = np.unique(gt_sub, return_counts=True)
                    logging.info(
                        f"[Debug][test][cloud {cloud_idx} sub {idx_subcloud}] "
                        f"GT分布: {dict(zip(u_gt.tolist(), c_gt.tolist()))}"
                    )

            all_logits.append(logits)

        # 10. 合并多子点云的预测结果
        all_logits = torch.cat(all_logits, dim=0)
        # 调整logits形状（适配模型输出：(B*N_sub, C) 或 (B, C, N_sub)）
        if not cfg.dataset.common.get('variable', False):
            # 模型输出为(B, C, N_sub)，转为(B*N_sub, C)用于后续scatter合并
            all_logits = all_logits.transpose(1, 2).reshape(-1, cfg.num_classes)

        # 11. 聚合预测结果（多子点云投票合并到原始点云）
        if not nearest_neighbor:
            # 拼接所有子点云的索引，用于scatter平均
            idx_points_flat = torch.from_numpy(np.hstack(idx_points)).cuda(non_blocking=True)
            # 按索引平均合并重叠点的预测
            all_logits = scatter(all_logits, idx_points_flat, dim=0, reduce='mean')
        else:
            # 最近邻插值（适配nearest_neighbor模式）
            all_logits = all_logits[reverse_idx_part][voxel_idx][reverse_idx]

        # 12. 生成最终预测标签
        pred = all_logits.argmax(dim=1)

        # 13. 整云预测分布调试
        u_pred_all, c_pred_all = np.unique(pred.cpu().numpy(), return_counts=True)
        logging.info(
            f"[Debug][test][cloud {cloud_idx}] "
            f"聚合后预测分布: {dict(zip(u_pred_all.tolist(), c_pred_all.tolist()))}"
        )
        if label is not None:
            u_gt_all, c_gt_all = np.unique(label.cpu().numpy(), return_counts=True)
            logging.info(
                f"[Debug][test][cloud {cloud_idx}] "
                f"聚合后GT分布: {dict(zip(u_gt_all.tolist(), c_gt_all.tolist()))}"
            )

        # 14. 更新混淆矩阵（计算指标）
        if label is not None:
            cm.update(pred, label)
            tp, union, count = cm.tp, cm.union, cm.count
            miou, macc, oa, ious, accs = get_mious(tp, union, count)
            with np.printoptions(precision=2, suppress=True):
                logging.info(
                    f"[{cloud_idx}]/[{len_data}] cloud | "
                    f"test_oa: {oa:.2f}, test_macc: {macc:.2f}, test_miou: {miou:.2f} | "
                    f"iou per cls: {ious}"
                )
            all_cm.value += cm.value

        # 15. 可视化（如配置需要）
        if cfg.visualize:
            gt_vis = label.cpu().numpy().squeeze() if label is not None else None
            pred_vis = pred.cpu().numpy().squeeze()
            # 映射标签到颜色
            if gt_vis is not None:
                gt_vis = cfg.cmap[gt_vis, :]
            pred_vis = cfg.cmap[pred_vis, :]
            # 生成文件名
            if 's3dis' in dataset_name:
                file_name = f'{dataset_name}-Area{cfg.dataset.common.test_area}-{cloud_idx}'
            else:
                file_name = f'{dataset_name}-{cloud_idx}'
            # 保存可视化结果（输入点云、GT、预测）
            write_obj(coord, feat[:, :3],  # 输入点云用RGB颜色
                      os.path.join(cfg.vis_dir, f'input-{file_name}.obj'))
            if gt_vis is not None:
                write_obj(coord, gt_vis,
                          os.path.join(cfg.vis_dir, f'gt-{file_name}.obj'))
            write_obj(coord, pred_vis,
                      os.path.join(cfg.vis_dir, f'pred-{cfg.cfg_basename}-{file_name}.obj'))

        # 16. 保存预测结果（如配置需要）
        if cfg.get('save_pred', False):
            pred_out = pred.cpu().numpy().squeeze()
            if 'scannet' in dataset_name:
                # 适配ScanNet标签格式（按需调整映射）
                save_file_name = os.path.basename(data_path).split('.')[0] + '.txt'
                save_file_path = os.path.join(cfg.save_path, save_file_name)
                np.savetxt(save_file_path, pred_out, fmt="%d")
                logging.info(f"预测结果保存到: {save_file_path}")
            elif 'semantickitti' in dataset_name:
                # 适配SemanticKITTI标签格式
                pred_out = pred_out + 1  # KITTI标签从1开始
                pred_out = pred_out.astype(np.uint32)
                upper_half = pred_out >> 16
                lower_half = pred_out & 0xFFFF
                lower_half = remap_lut_write[lower_half]
                pred_out = (upper_half << 16) + lower_half
                frame_id = os.path.basename(data_path[0]).split('.')[0]
                save_file_path = os.path.join(cfg.save_path, f'{frame_id}.label')
                pred_out.tofile(save_file_path)

    # 17. 计算全量测试集指标
    if label is not None:
        tp_total, union_total, count_total = all_cm.tp, all_cm.union, all_cm.count
        if cfg.distributed:
            # 分布式环境下聚合多GPU的混淆矩阵
            dist.all_reduce(tp_total), dist.all_reduce(union_total), dist.all_reduce(count_total)
        miou_total, macc_total, oa_total, ious_total, accs_total = get_mious(tp_total, union_total, count_total)
        logging.info("=" * 80)
        with np.printoptions(precision=2, suppress=True):
            logging.info(
                f"Test Set Total Metrics | "
                f"OA: {oa_total:.2f}, mACC: {macc_total:.2f}, mIoU: {miou_total:.2f} | "
                f"IoU per cls: {ious_total}"
            )
        logging.info("=" * 80)
        return miou_total, macc_total, oa_total, ious_total, accs_total, all_cm
    else:
        logging.info("Test Set No Label Data, Skip Metric Calculation")
        return None, None, None, None, None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Scene segmentation training/testing')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)  # 覆盖配置文件中的默认参数

    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    # 初始化分布式环境
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # 初始化日志目录
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]
    cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]
    tags = [
        cfg.task_name,
        cfg.mode,
        cfg.cfg_basename,
        f'ngpus{cfg.world_size}',
    ]
    opt_list = []
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            opt_list.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    cfg.opts = '-'.join(opt_list)

    cfg.is_training = cfg.mode not in ['test', 'testing', 'val', 'eval', 'evaluation']
    if cfg.mode in ['resume', 'val', 'test']:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = [cfg.mode]
    else:
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
        cfg.wandb.tags = tags
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path

    # wandb配置
    cfg.wandb.name = cfg.run_name

    # 多进程启动
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print('using mp spawn for distributed training')
        mp.spawn(main, nprocs=cfg.world_size, args=(cfg,))
    else:
        main(0, cfg)
