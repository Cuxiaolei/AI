from pointcept.utils.registry import Registry
import logging
# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
LOSSES = Registry("losses")


class Criteria(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        self.needs_coords = []  # 记录每个损失是否需要坐标
        self.requires_coords = False  # 全局标记：是否有任何损失需要坐标（供Segmentor判断）

        for loss_cfg in self.cfg:
            # 实例化损失函数（如CrossEntropyLoss、PLCCLoss）
            crit = LOSSES.build(cfg=loss_cfg)
            self.criteria.append(crit)

            # 判断当前损失是否需要坐标（检查是否有requires_coords属性且为True）
            need_coord = hasattr(crit, "requires_coords") and crit.requires_coords
            self.needs_coords.append(need_coord)

            # 只要有一个损失需要坐标，全局标记就设为True
            if need_coord:
                self.requires_coords = True

        # 初始化日志：打印所有损失是否需要坐标（便于调试）
        logging.info("初始化损失函数列表：")
        for i, (crit, need_coord) in enumerate(zip(self.criteria, self.needs_coords)):
            logging.info(f"  损失 {i}：{type(crit).__name__}，需要坐标：{need_coord}")

    def __call__(self, pred, target, input_dict=None):
        """
        Args:
            pred: 模型预测结果（logits），shape [N, C]
            target: 真实标签，shape [N]
            input_dict: 额外输入数据（包含 coords 等）
        """
        if len(self.criteria) == 0:
            return pred

        total_loss = 0.0
        for i, (crit, need_coord) in enumerate(zip(self.criteria, self.needs_coords)):
            loss_weight = self.cfg[i].get("loss_weight", 1.0)

            if need_coord:
                # 校验坐标输入字典或坐标，跳过当前损失
                if input_dict is None:
                    logging.warning(f"Loss {type(crit).__name__} needs input_dict but not provided, skip")
                    continue
                if "coord" not in input_dict:
                    logging.warning(f"Loss {type(crit).__name__} needs coords but not found in input_dict, skip")
                    continue

                # 提取坐标并验证形状（确保是[N, 3]）
                coords = input_dict["coord"]
                if coords.ndim != 2 or coords.shape[1] != 3:
                    logging.warning(
                        f"Loss {type(crit).__name__} needs coords with shape [N, 3], "
                        f"but got {coords.shape}, skip"
                    )
                    continue

                # 计算需要坐标的损失（如PLCCLoss）
                loss = crit(pred, target, coords)

            else:
                # 计算不需要坐标的损失（如CrossEntropyLoss）
                loss = crit(pred, target)

            # 累加加权损失
            total_loss += loss * loss_weight
            # 打印损失详情（保留4位小数）
            logging.info(
                f"[Loss {i}] {type(crit).__name__}: "
                f"raw={loss.item():.4f}, weighted={loss.item() * loss_weight:.4f}"
            )

        return total_loss


def build_criteria(cfg):
    """构建损失函数容器"""
    return Criteria(cfg)
