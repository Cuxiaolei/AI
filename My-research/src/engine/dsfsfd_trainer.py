# src/engine/dsfsfd_trainer.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
import time
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from ..models.backbones.dsfsfd.blocks import Resnet1d, Resnet2d
from ..models.backbones.dsfsfd.net import MahFusion_Network
from ..models.backbones.dsfsfd.loss import Fusion_loss
from ..data.samplers.Data_Sampler_dsfsfd import TrainSampler, TestSampler




# -------------------------
# tqdm-friendly logging
# -------------------------
class TqdmLoggingHandler(logging.Handler):
    """让 logger 输出不破坏 tqdm 进度条"""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"dsfsfd_{log_file.parent.parent.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(message)s")

    sh = TqdmLoggingHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    return logger



# -------------------------
# small utils
# -------------------------
class EMA:
    def __init__(self, momentum: float = 0.95):
        self.m = float(momentum)
        self.v: Optional[float] = None

    def update(self, x: float) -> float:
        x = float(x)
        if self.v is None:
            self.v = x
        else:
            self.v = self.m * self.v + (1.0 - self.m) * x
        return self.v


def make_unique_exp_dir(base_dir: Path, exp_name: str) -> Path:
    """
    base_dir/exp_name 若存在，则返回 base_dir/exp_name_1, _2, ...
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    cand = base_dir / exp_name
    if not cand.exists():
        return cand
    for i in range(1, 10000):
        cand_i = base_dir / f"{exp_name}_{i}"
        if not cand_i.exists():
            return cand_i
    raise RuntimeError(f"Too many existing experiment folders for name={exp_name}")


def _append_csv_row(csv_path: Path, header: List[str], row: List[Any]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow(row)


def split_model_param(model: torch.nn.Module):
    """gamma/beta -> ft_params，其它 -> model_params"""
    model_params, ft_params = [], []
    for name, p in model.named_parameters():
        last = name.split(".")[-1]
        if last in ("gamma", "beta"):
            ft_params.append(p)
        else:
            model_params.append(p)
    return model_params, ft_params


def _ensure_cuda_ready(device: str) -> torch.device:
    # 你的 sampler/loss 内部存在 .cuda() 写死，当前实现要求 GPU 才能跑
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    raise RuntimeError(
        "DSFSFD 当前实现（sampler/loss）存在 .cuda() 写死，必须在 GPU 环境运行。"
        "如果你想支持 CPU，我可以给你一版把所有 .cuda() 改成 .to(device) 的补丁。"
    )


def _set_seed(seed: int):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _stats_line(pbar: tqdm, metrics: dict) -> str:
    fd = getattr(pbar, "format_dict", {})
    elapsed = tqdm.format_interval(fd.get("elapsed", 0.0))

    remaining = fd.get("remaining", None)
    remaining_str = tqdm.format_interval(remaining) if remaining is not None else "??:??"

    rate = fd.get("rate", None)
    rate_str = f"{rate:5.2f}it/s" if rate else "  ?.?it/s"

    m = ", ".join([f"{k}={v}" for k, v in metrics.items()])
    return f"[{elapsed}<{remaining_str}, {rate_str}, {m}]"


# -------------------------
# train / test
# -------------------------
def train_one_run(opt: SimpleNamespace, model: torch.nn.Module, run_dir: Path, logger: logging.Logger):
    Tsampler = TrainSampler(opt=opt)
    scaler = Tsampler.scaler
    Titer = iter(Tsampler)

    loss_fn = Fusion_loss(opt=opt)
    model_params, ft_params = split_model_param(model)

    model_optim = torch.optim.Adam(model_params, lr=opt.lr)
    ft_optim = torch.optim.Adam(ft_params, weight_decay=1e-8, lr=opt.ft_lr)

    ckpt_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "metrics"
    log_csv = metrics_dir / "train_metrics.csv"
    header = ["epoch", "ps_loss", "ps_acc", "pu_loss", "pu_acc", "epoch_sec"]

    logger.info(f"[Train] epochs={opt.epochs} episodes={opt.episodes} log_every={opt.log_every} save_every={opt.save_every}")

    # Epoch 总进度条（第1行）
    epoch_bar = tqdm(
        range(1, opt.epochs + 1),
        desc="Train",
        dynamic_ncols=True,
        leave=True,
        position=0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]"
    )

    for epoch in epoch_bar:
        t0 = time.time()
        ps_loss = ps_acc = 0.0
        pu_loss = pu_acc = 0.0

        ema_ps_loss = EMA(0.95)
        ema_ps_acc = EMA(0.95)
        ema_pu_loss = EMA(0.95)
        ema_pu_acc = EMA(0.95)

        # Episode 进度条（第2行，只显示 bar，不显示 stats）
        epi_bar = tqdm(
            range(1, opt.episodes + 1),
            desc=f"E{epoch}/{opt.epochs}",
            dynamic_ncols=True,
            leave=False,
            position=1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]"
        )

        # 第3行：专门放 stats（只显示 desc）
        info_bar = tqdm(
            total=0,
            position=2,
            leave=False,
            dynamic_ncols=True,
            bar_format="{desc}"
        )
        info_bar.set_description_str("[00:00<??:??,  ?.?it/s, ps_loss=..., ps_acc=..., pu_loss=..., pu_acc=...]")

        for step in epi_bar:
            (ps_TFs, ps_DEs, ps_FFTs, ps_TFq, ps_DEq, ps_FFTq,
             pu_TFs, pu_DEs, pu_FFTs, pu_TFq, pu_DEq, pu_FFTq) = next(Titer)

            for w in model_params:
                w.fast = None

            # support
            model.train()
            out = model(TFs=ps_TFs, TFq=ps_TFq, DEs=ps_DEs, DEq=ps_DEq, FFTs=ps_FFTs, FFTq=ps_FFTq)
            psloss, psacc = loss_fn(*out)

            meta_grad = torch.autograd.grad(psloss, model_params, create_graph=True)
            for k, w in enumerate(model_params):
                w.fast = w - opt.lr * meta_grad[k]
            meta_grad = [g.detach() for g in meta_grad]

            # query
            model.eval()
            out = model(TFs=pu_TFs, TFq=pu_TFq, DEs=pu_DEs, DEq=pu_DEq, FFTs=pu_FFTs, FFTq=pu_FFTq)
            puloss, puacc = loss_fn(*out)

            # update model params
            model_optim.zero_grad()
            for k, w in enumerate(model_params):
                w.grad = meta_grad[k]
            model_optim.step()

            # update ft params（不要 detach）
            ft_optim.zero_grad()
            puloss.backward()
            ft_optim.step()

            ps_loss += float(psloss.item()); ps_acc += float(psacc.item())
            pu_loss += float(puloss.item()); pu_acc += float(puacc.item())

            # 每 log_every 更新一次 stats 行（第3行）
            if step == 1 or step == opt.episodes or (step % opt.log_every == 0):
                ps_loss_s = ema_ps_loss.update(ps_loss / step)
                ps_acc_s = ema_ps_acc.update(ps_acc / step)
                pu_loss_s = ema_pu_loss.update(pu_loss / step)
                pu_acc_s = ema_pu_acc.update(pu_acc / step)

                metrics = {
                    "ps_loss": f"{ps_loss_s:.3f}",
                    "ps_acc":  f"{ps_acc_s:.3f}",
                    "pu_loss": f"{pu_loss_s:.3f}",
                    "pu_acc":  f"{pu_acc_s:.3f}",
                }

                if getattr(opt, "show_mem", False) and torch.cuda.is_available():
                    mem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                    metrics["memGB"] = f"{mem_gb:.2f}"

                info_bar.set_description_str(_stats_line(epi_bar, metrics))

        info_bar.close()

        epoch_sec = time.time() - t0
        ps_loss /= opt.episodes; ps_acc /= opt.episodes
        pu_loss /= opt.episodes; pu_acc /= opt.episodes

        epoch_bar.set_postfix({"ps_acc": f"{ps_acc:.3f}", "pu_acc": f"{pu_acc:.3f}"})

        logger.info(
            f"[Train][E{epoch:03d}] DONE "
            f"ps_loss={ps_loss:.6f} ps_acc={ps_acc:.4f} | "
            f"pu_loss={pu_loss:.6f} pu_acc={pu_acc:.4f} | "
            f"time={epoch_sec:.1f}s"
        )

        _append_csv_row(
            log_csv,
            header,
            [epoch, f"{ps_loss:.6f}", f"{ps_acc:.6f}", f"{pu_loss:.6f}", f"{pu_acc:.6f}", f"{epoch_sec:.3f}"]
        )

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if epoch % opt.save_every == 0:
            torch.save(model.state_dict(), ckpt_dir / f"{epoch}.pth")
        torch.save(model.state_dict(), ckpt_dir / "last.pth")

    logger.info(f"[Saved] {log_csv}")
    return scaler, log_csv



@torch.no_grad()
def test_one_run(opt: SimpleNamespace, model: torch.nn.Module, scaler, ckpt_path: Path, run_dir: Path, logger: logging.Logger):
    model.load_state_dict(torch.load(ckpt_path, map_location=opt.device), strict=False)
    model.eval()

    Vsampler = TestSampler(opt=opt, scaler=scaler)
    Viter = iter(Vsampler)
    loss_fn = Fusion_loss(opt=opt)

    test_acc = 0.0
    result = []

    pbar = tqdm(
        range(1, opt.test_iters + 1),
        desc="Test",
        dynamic_ncols=True,
        leave=True,
        position=0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]"
    )

    info_bar = tqdm(
        total=0,
        position=1,
        leave=False,
        dynamic_ncols=True,
        bar_format="{desc}"
    )
    info_bar.set_description_str("[00:00<??:??,  ?.?it/s, avg_acc=...]")

    for i in pbar:
        TFs, DEs, FFTs, TFq, DEq, FFTq = next(Viter)
        out = model(TFs=TFs, TFq=TFq, DEs=DEs, DEq=DEq, FFTs=FFTs, FFTq=FFTq)
        _, acc = loss_fn(*out)
        test_acc += float(acc.item())

        if i % 100 == 0:
            avg_acc = test_acc / i
            result.append(avg_acc)
            info_bar.set_description_str(_stats_line(pbar, {"avg_acc": f"{avg_acc:.4f}"}))

    info_bar.close()

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_csv = metrics_dir / "test_curve.csv"
    np.savetxt(out_csv, np.array(result), fmt="%.6f")
    logger.info(f"[Saved] {out_csv}")

    return result


# -------------------------
# cfg -> opt
# -------------------------
def build_opt_from_cfg(cfg: Dict[str, Any], project_root: Path) -> SimpleNamespace:
    opt = SimpleNamespace()

    # seed/device
    opt.seed = int(cfg.get("seed", 2025))
    opt.device = _ensure_cuda_ready(cfg.get("device", "cuda"))

    # data
    data_root = cfg["data"]["root"]
    data_root = Path(data_root)
    if not data_root.is_absolute():
        data_root = (project_root / data_root).resolve()
    opt.data_dir = str(data_root)
    opt.train_domain = cfg["data"]["train_domain"]
    opt.test_domain = cfg["data"]["test_domain"]

    # task
    opt.k_train = cfg["task"]["k_train"]
    opt.n_train = cfg["task"]["n_train"]
    opt.q_train = cfg["task"]["q_train"]
    opt.k_val = cfg["task"]["k_val"]
    opt.n_val = cfg["task"]["n_val"]
    opt.q_val = cfg["task"]["q_val"]

    # train
    opt.repeat = int(cfg.get("train", {}).get("repeat", 1))
    opt.epochs = int(cfg["train"]["epochs"])
    opt.episodes = int(cfg["train"]["episodes"])
    opt.lr = float(cfg["train"]["lr"])
    opt.ft_lr = float(cfg["train"]["ft_lr"])
    opt.save_every = int(cfg.get("train", {}).get("save_every", 10))
    opt.log_every = int(cfg.get("train", {}).get("log_every", 10))
    opt.show_mem = bool(cfg.get("train", {}).get("show_mem", False))

    # test
    opt.test_iters = int(cfg.get("test", {}).get("iters", 500))

    # model loss weights
    opt.TF_weight = cfg["model"]["loss"]["TF_weight"]
    opt.DE_weight = cfg["model"]["loss"]["DE_weight"]
    opt.FFT_weight = cfg["model"]["loss"]["FFT_weight"]

    # encoder params
    opt.tf_blockdim = cfg["model"]["encoders"]["tf"]["blockdim"]
    opt.de_blockdim = cfg["model"]["encoders"]["de"]["blockdim"]
    opt.fft_blockdim = cfg["model"]["encoders"]["fft"]["blockdim"]

    opt.tf_feature_trans = bool(cfg["model"]["encoders"]["tf"].get("feature_trans", True))
    opt.de_feature_trans = bool(cfg["model"]["encoders"]["de"].get("feature_trans", True))
    opt.fft_feature_trans = bool(cfg["model"]["encoders"]["fft"].get("feature_trans", True))

    # output
    out_cfg = cfg.get("output", {})
    base_dir = Path(out_cfg.get("base_dir", "outputs"))
    if not base_dir.is_absolute():
        base_dir = (project_root / base_dir).resolve()

    exp_name = out_cfg.get("exp_name", "default_exp")
    auto_inc = bool(out_cfg.get("auto_increment", True))

    exp_dir = make_unique_exp_dir(base_dir, exp_name) if auto_inc else (base_dir / exp_name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    opt.base_dir = str(exp_dir)
    opt.exp_dir_name = exp_dir.name

    return opt


def run(cfg: Dict[str, Any], project_root: Path):
    opt = build_opt_from_cfg(cfg, project_root)

    # seed
    _set_seed(opt.seed)

    base_dir = Path(opt.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # 保存最终合并后的 cfg，方便复现实验
    (base_dir / "meta").mkdir(exist_ok=True)
    with open(base_dir / "meta" / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # 总览输出目录
    tqdm.write(f"[Output] Using experiment dir: {base_dir}")

    for r in range(opt.repeat):
        run_dir = base_dir / f"run{r}"
        run_dir.mkdir(parents=True, exist_ok=True)

        logger = setup_logger(run_dir / "logs" / "train.log")

        per_name = f"{opt.k_val}way{opt.n_val}shot-[{opt.train_domain}--{opt.test_domain}]-Exp{r}"
        logger.info(f"\n==== {per_name} ====")
        logger.info(f"[Paths] data_dir={opt.data_dir} | out_dir={run_dir}")

        # build model
        train_model = MahFusion_Network(
            TF_encoder=Resnet2d(blockdim=opt.tf_blockdim, Feature_trans=opt.tf_feature_trans),
            DE_encoder=Resnet1d(blockdim=opt.de_blockdim, Feature_trans=opt.de_feature_trans),
            FFT_encoder=Resnet1d(blockdim=opt.fft_blockdim, Feature_trans=opt.fft_feature_trans),
        ).to(opt.device)

        scaler, _ = train_one_run(opt, train_model, run_dir, logger)

        # test with last.pth
        ckpt_path = run_dir / "checkpoints" / "last.pth"
        test_model = MahFusion_Network(
            TF_encoder=Resnet2d(blockdim=opt.tf_blockdim, Feature_trans=opt.tf_feature_trans),
            DE_encoder=Resnet1d(blockdim=opt.de_blockdim, Feature_trans=opt.de_feature_trans),
            FFT_encoder=Resnet1d(blockdim=opt.fft_blockdim, Feature_trans=opt.fft_feature_trans),
        ).to(opt.device)

        _ = test_one_run(opt, test_model, scaler, ckpt_path, run_dir, logger)

        logger.info(f"[DONE] run_dir={run_dir}")
