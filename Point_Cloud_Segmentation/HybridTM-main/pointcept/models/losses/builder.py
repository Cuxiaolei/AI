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
        # 记录每个损失是否需要坐标（提前解析配置，避免每次调用时重复判断）
        self.needs_coords = []
        for loss_cfg in self.cfg:
            # 实例化损失函数
            crit = LOSSES.build(cfg=loss_cfg)
            self.criteria.append(crit)
            # 判断该损失是否需要坐标（检查是否有 requires_coords 属性且为 True）
            need_coord = hasattr(crit, "requires_coords") and crit.requires_coords

            self.needs_coords.append(need_coord)

    def __call__(self, pred, target, input_dict=None):
        """
        Args:
            pred: 模型预测结果（logits），shape [N, C]
            target: 真实标签，shape [N]
            input_dict: 额外输入数据（包含 coords 等），仅当损失需要坐标时使用
        """
        if len(self.criteria) == 0:
            return pred
        total_loss = 0.0
        for i, (crit, need_coord) in enumerate(zip(self.criteria, self.needs_coords)):
            # 1. 处理损失权重（从配置中获取，默认1.0）
            loss_weight = self.cfg[i].get("loss_weight", 1.0)
            # 2. 根据是否需要坐标，传递不同参数
            if need_coord:
                # 校验 input_dict 和 coords 是否存在
                if input_dict is None or "coord" not in input_dict:
                    logging.warning(f"Loss {type(crit).__name__} needs coords but not provided, skip")
                    continue
                # 传递 3 个参数（pred, target, coords），适配 PLCCLoss
                coords = input_dict["coord"]
                loss = crit(pred, target, coords)
            else:
                # 传递 2 个参数（pred, target），适配 CrossEntropyLoss/LovaszLoss
                loss = crit(pred, target)
            # 3. 应用损失权重并累加
            # 打印每个损失的原始值和加权后的值
            print(
                f"[Loss {i}] {type(crit).__name__}: raw={loss.item():.4f}, weighted={loss.item() * loss_weight:.4f}")
            total_loss += loss * loss_weight
        return total_loss


def build_criteria(cfg):
    return Criteria(cfg)