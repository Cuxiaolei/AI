# src/engine/dsfsfd_trainer.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
import time
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ..models.backbones.dsfsfd.blocks import Resnet1d, Resnet2d
from ..models.backbones.dsfsfd.net import MahFusion_Network
from ..models.backbones.dsfsfd.loss import Fusion_loss
from ..data.samplers.Data_Sampler_dsfsfd import TrainSampler, TestSampler


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

def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"dsfsfd_{log_file.parent.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    return logger


def split_model_param(model: torch.nn.Module):
    model_params, ft_params = [], []
    for name, p in model.named_parameters():
        last = name.split(".")[-1]
        if last in ("gamma", "beta"):
            ft_params.append(p)
        else:
            model_params.append(p)
    return model_params, ft_params


def _ensure_cuda_ready(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    raise RuntimeError(
        "DSFSFD 当前实现（sampler/loss）存在 .cuda() 写死，必须在 GPU 环境运行。"
    )


def _append_csv_row(csv_path: Path, header: list, row: list):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow(row)


def train_one_run(opt: SimpleNamespace, model: torch.nn.Module, run_dir: Path, logger: logging.Logger):
    """
    返回 scaler + 训练日志 CSV 路径
    """
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
    logger.info(f"[Train] episodes={opt.episodes} log_every={opt.log_every} save_every={opt.save_every}")

    for epoch in range(1, opt.epochs + 1):
        t0 = time.time()
        ps_loss = ps_acc = 0.0
        pu_loss = pu_acc = 0.0

        for step in range(1, opt.episodes + 1):
            (ps_TFs, ps_DEs, ps_FFTs, ps_TFq, ps_DEq, ps_FFTq,
             pu_TFs, pu_DEs, pu_FFTs, pu_TFq, pu_DEq, pu_FFTq) = next(Titer)

            # reset fast weights
            for w in model_params:
                w.fast = None

            # -------- support (inner) --------
            model.train()
            out = model(
                TFs=ps_TFs, TFq=ps_TFq,
                DEs=ps_DEs, DEq=ps_DEq,
                FFTs=ps_FFTs, FFTq=ps_FFTq
            )
            psloss, psacc = loss_fn(*out)

            meta_grad = torch.autograd.grad(psloss, model_params, create_graph=True)
            for k, w in enumerate(model_params):
                w.fast = w - opt.lr * meta_grad[k]
            meta_grad = [g.detach() for g in meta_grad]

            # -------- query (outer) --------
            model.eval()
            out = model(
                TFs=pu_TFs, TFq=pu_TFq,
                DEs=pu_DEs, DEq=pu_DEq,
                FFTs=pu_FFTs, FFTq=pu_FFTq
            )
            puloss, puacc = loss_fn(*out)

            # update model params
            model_optim.zero_grad()
            for k, w in enumerate(model_params):
                w.grad = meta_grad[k]
            model_optim.step()

            # update ft params (关键：不要 detach)
            ft_optim.zero_grad()
            puloss.backward()
            ft_optim.step()

            ps_loss += float(psloss.item()); ps_acc += float(psacc.item())
            pu_loss += float(puloss.item()); pu_acc += float(puacc.item())

            # ---- 进度日志（每 log_every 打一次）----
            if (step % opt.log_every) == 0:
                logger.info(
                    f"[Train][E{epoch:03d}][{step:04d}/{opt.episodes}] "
                    f"ps_loss={ps_loss/step:.4f} ps_acc={ps_acc/step:.4f} | "
                    f"pu_loss={pu_loss/step:.4f} pu_acc={pu_acc/step:.4f}"
                )

        epoch_sec = time.time() - t0
        ps_loss /= opt.episodes; ps_acc /= opt.episodes
        pu_loss /= opt.episodes; pu_acc /= opt.episodes

        logger.info(
            f"[Train][E{epoch:03d}] DONE "
            f"ps_loss={ps_loss:.6f} ps_acc={ps_acc:.4f} | "
            f"pu_loss={pu_loss:.6f} pu_acc={pu_acc:.4f} | "
            f"time={epoch_sec:.1f}s"
        )

        # 写训练曲线 CSV
        _append_csv_row(
            log_csv,
            header,
            [epoch, f"{ps_loss:.6f}", f"{ps_acc:.6f}", f"{pu_loss:.6f}", f"{pu_acc:.6f}", f"{epoch_sec:.3f}"]
        )

        # 保存 checkpoint
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if epoch % opt.save_every == 0:
            torch.save(model.state_dict(), ckpt_dir / f"{epoch}.pth")
        # 始终写 last
        torch.save(model.state_dict(), ckpt_dir / "last.pth")

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
    for i in range(1, opt.test_iters + 1):
        TFs, DEs, FFTs, TFq, DEq, FFTq = next(Viter)
        out = model(TFs=TFs, TFq=TFq, DEs=DEs, DEq=DEq, FFTs=FFTs, FFTq=FFTq)
        _, acc = loss_fn(*out)
        test_acc += float(acc.item())

        if i % 100 == 0:
            avg_acc = test_acc / i
            result.append(avg_acc)
            logger.info(f"[Test] iter={i} avg_acc={avg_acc:.4f}")

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(metrics_dir / "test_curve.csv", np.array(result), fmt="%.6f")
    logger.info(f"[Saved] {metrics_dir / 'test_curve.csv'}")

    return result


def build_opt_from_cfg(cfg: dict, project_root: Path) -> SimpleNamespace:
    opt = SimpleNamespace()

    data_root = (project_root / cfg["data"]["root"]).resolve()
    opt.data_dir = str(data_root)
    opt.train_domain = cfg["data"]["train_domain"]
    opt.test_domain = cfg["data"]["test_domain"]

    opt.k_train = cfg["task"]["k_train"]
    opt.n_train = cfg["task"]["n_train"]
    opt.q_train = cfg["task"]["q_train"]
    opt.k_val = cfg["task"]["k_val"]
    opt.n_val = cfg["task"]["n_val"]
    opt.q_val = cfg["task"]["q_val"]

    opt.epochs = cfg["train"]["epochs"]
    opt.episodes = cfg["train"]["episodes"]
    opt.lr = cfg["train"]["lr"]
    opt.ft_lr = cfg["train"]["ft_lr"]
    opt.save_every = cfg["train"].get("save_every", 10)
    opt.log_every = cfg["train"].get("log_every", 10)  # 新增：每多少 episode 打一次日志

    opt.test_iters = cfg.get("test", {}).get("iters", 500)

    # cfg 期望结构：cfg["model"]["loss"] / cfg["model"]["encoders"]
    opt.TF_weight = cfg["model"]["loss"]["TF_weight"]
    opt.DE_weight = cfg["model"]["loss"]["DE_weight"]
    opt.FFT_weight = cfg["model"]["loss"]["FFT_weight"]

    opt.tf_blockdim = cfg["model"]["encoders"]["tf"]["blockdim"]
    opt.de_blockdim = cfg["model"]["encoders"]["de"]["blockdim"]
    opt.fft_blockdim = cfg["model"]["encoders"]["fft"]["blockdim"]

    opt.device = _ensure_cuda_ready(cfg.get("device", "cuda"))

    out_base = (project_root / cfg["output"]["base_dir"]).resolve()
    exp_name = cfg["output"]["exp_name"]
    auto_inc = bool(cfg.get("output", {}).get("auto_increment", True))
    exp_dir = (out_base / exp_name)
    if auto_inc:
        exp_dir = make_unique_exp_dir(out_base, exp_name)
    else:
        exp_dir.mkdir(parents=True, exist_ok=True)
    opt.base_dir = str(exp_dir)
    # 可选：把最终实际使用的 exp_name 记录下来（日志/保存 config 时更清楚）
    opt.exp_dir_name = exp_dir.name


    opt.repeat = int(cfg["train"].get("repeat", 1))
    opt.seed = int(cfg.get("seed", 2025))

    return opt


def run(cfg: dict, project_root: Path):
    opt = build_opt_from_cfg(cfg, project_root)

    base_dir = Path(opt.base_dir)
    print(f"[Output] Using experiment dir: {base_dir}")
    base_dir.mkdir(parents=True, exist_ok=True)

    # 记录 cfg
    (base_dir / "meta").mkdir(exist_ok=True)
    with open(base_dir / "meta" / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    for r in range(opt.repeat):
        run_dir = base_dir / f"run{r}"
        run_dir.mkdir(parents=True, exist_ok=True)

        logger = setup_logger(run_dir / "logs" / "train.log")

        per_name = f"{opt.k_val}way{opt.n_val}shot-[{opt.train_domain}--{opt.test_domain}]-Exp{r}"
        logger.info(f"\n==== {per_name} ====")

        # build model
        train_model = MahFusion_Network(
            TF_encoder=Resnet2d(blockdim=opt.tf_blockdim, Feature_trans=True),
            DE_encoder=Resnet1d(blockdim=opt.de_blockdim, Feature_trans=True),
            FFT_encoder=Resnet1d(blockdim=opt.fft_blockdim, Feature_trans=True),
        ).to(opt.device)

        scaler, train_csv = train_one_run(opt, train_model, run_dir, logger)
        logger.info(f"[Saved] {train_csv}")

        # test: 用 last.pth（不等 epochs.pth）
        ckpt_path = run_dir / "checkpoints" / "last.pth"
        test_model = MahFusion_Network(
            TF_encoder=Resnet2d(blockdim=opt.tf_blockdim, Feature_trans=True),
            DE_encoder=Resnet1d(blockdim=opt.de_blockdim, Feature_trans=True),
            FFT_encoder=Resnet1d(blockdim=opt.fft_blockdim, Feature_trans=True),
        ).to(opt.device)

        _ = test_one_run(opt, test_model, scaler, ckpt_path, run_dir, logger)

        logger.info(f"[DONE] run_dir={run_dir}")
