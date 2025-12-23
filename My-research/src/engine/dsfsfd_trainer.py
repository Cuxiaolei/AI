# src/engine/dsfsfd_trainer.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import torch
from tqdm import tqdm

from ..engine.common_tools import (
    EMA,
    append_csv_row,
    ensure_device,
    make_unique_exp_dir,
    resolve_path,
    save_json,
    set_seed,
    setup_logger,
    stats_line,
)

from ..models.backbones.dsfsfd.blocks import Resnet1d, Resnet2d
from ..models.backbones.dsfsfd.net import MahFusion_Network
from ..models.backbones.dsfsfd.loss import Fusion_loss

# 你当前目录是 src/data/samplers/Data_Sampler_dsfsfd.py
from ..data.samplers.Data_Sampler_dsfsfd import TrainSampler, TestSampler


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


def train_one_run(opt: SimpleNamespace, model: torch.nn.Module, run_dir: Path, logger):
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

    # 第1行：Epoch 总进度条
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

        # 第2行：Episode bar（不显示 stats）
        epi_bar = tqdm(
            range(1, opt.episodes + 1),
            desc=f"E{epoch}/{opt.epochs}",
            dynamic_ncols=True,
            leave=False,
            position=1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]"
        )

        # 第3行：专门显示 stats
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

            # reset fast weights
            for w in model_params:
                w.fast = None

            # -------- support (inner) --------
            model.train()
            out = model(TFs=ps_TFs, TFq=ps_TFq, DEs=ps_DEs, DEq=ps_DEq, FFTs=ps_FFTs, FFTq=ps_FFTq)
            psloss, psacc = loss_fn(*out)

            meta_grad = torch.autograd.grad(psloss, model_params, create_graph=True)
            for k, w in enumerate(model_params):
                w.fast = w - opt.lr * meta_grad[k]
            meta_grad = [g.detach() for g in meta_grad]

            # -------- query (outer) --------
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

            # 每 log_every 更新一次第3行 stats
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

                info_bar.set_description_str(stats_line(epi_bar, metrics))

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

        append_csv_row(
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
def test_one_run(opt: SimpleNamespace, model: torch.nn.Module, scaler, ckpt_path: Path, run_dir: Path, logger):
    model.load_state_dict(torch.load(ckpt_path, map_location=opt.device), strict=False)
    model.eval()

    Vsampler = TestSampler(opt=opt, scaler=scaler)
    Viter = iter(Vsampler)
    loss_fn = Fusion_loss(opt=opt)

    test_acc = 0.0
    result = []

    # 第1行：Test bar
    pbar = tqdm(
        range(1, opt.test_iters + 1),
        desc="Test",
        dynamic_ncols=True,
        leave=True,
        position=0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]"
    )

    # 第2行：Test stats
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
            info_bar.set_description_str(stats_line(pbar, {"avg_acc": f"{avg_acc:.4f}"}))

    info_bar.close()

    out_csv = (run_dir / "metrics" / "test_curve.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_csv, np.array(result), fmt="%.6f")
    logger.info(f"[Saved] {out_csv}")
    return result


def build_opt_from_cfg(cfg: Dict[str, Any], project_root: Path) -> SimpleNamespace:
    opt = SimpleNamespace()

    # seed/device
    opt.seed = int(cfg.get("seed", 2025))
    opt.device = ensure_device(cfg.get("device", "cuda"))

    # data
    data_root = resolve_path(project_root, cfg["data"]["root"])
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

    # model
    opt.TF_weight = cfg["model"]["loss"]["TF_weight"]
    opt.DE_weight = cfg["model"]["loss"]["DE_weight"]
    opt.FFT_weight = cfg["model"]["loss"]["FFT_weight"]

    opt.tf_blockdim = cfg["model"]["encoders"]["tf"]["blockdim"]
    opt.de_blockdim = cfg["model"]["encoders"]["de"]["blockdim"]
    opt.fft_blockdim = cfg["model"]["encoders"]["fft"]["blockdim"]

    opt.tf_feature_trans = bool(cfg["model"]["encoders"]["tf"].get("feature_trans", True))
    opt.de_feature_trans = bool(cfg["model"]["encoders"]["de"].get("feature_trans", True))
    opt.fft_feature_trans = bool(cfg["model"]["encoders"]["fft"].get("feature_trans", True))

    # output: base_dir/exp_name，并且同名自动加 _1/_2...
    out_cfg = cfg.get("output", {})
    base_dir = resolve_path(project_root, out_cfg.get("base_dir", "outputs"))
    exp_name = out_cfg.get("exp_name", "default_exp")
    auto_inc = bool(out_cfg.get("auto_increment", True))

    exp_dir = make_unique_exp_dir(base_dir, exp_name) if auto_inc else (base_dir / exp_name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    opt.base_dir = str(exp_dir)
    opt.exp_dir_name = exp_dir.name
    return opt


def run(cfg: Dict[str, Any], project_root: Path):
    opt = build_opt_from_cfg(cfg, project_root)
    set_seed(opt.seed)

    base_dir = Path(opt.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # 保存最终合并 cfg
    save_json(base_dir / "meta" / "config.json", cfg)

    tqdm.write(f"[Output] Using experiment dir: {base_dir}")

    for r in range(opt.repeat):
        run_dir = base_dir / f"run{r}"
        run_dir.mkdir(parents=True, exist_ok=True)

        logger = setup_logger(run_dir / "logs" / "train.log", name="dsfsfd")

        per_name = f"{opt.k_val}way{opt.n_val}shot-[{opt.train_domain}--{opt.test_domain}]-Exp{r}"
        logger.info(f"\n==== {per_name} ====")
        logger.info(f"[Paths] data_dir={opt.data_dir} | out_dir={run_dir}")

        train_model = MahFusion_Network(
            TF_encoder=Resnet2d(blockdim=opt.tf_blockdim, Feature_trans=opt.tf_feature_trans),
            DE_encoder=Resnet1d(blockdim=opt.de_blockdim, Feature_trans=opt.de_feature_trans),
            FFT_encoder=Resnet1d(blockdim=opt.fft_blockdim, Feature_trans=opt.fft_feature_trans),
        ).to(opt.device)

        scaler, _ = train_one_run(opt, train_model, run_dir, logger)

        ckpt_path = run_dir / "checkpoints" / "last.pth"
        test_model = MahFusion_Network(
            TF_encoder=Resnet2d(blockdim=opt.tf_blockdim, Feature_trans=opt.tf_feature_trans),
            DE_encoder=Resnet1d(blockdim=opt.de_blockdim, Feature_trans=opt.de_feature_trans),
            FFT_encoder=Resnet1d(blockdim=opt.fft_blockdim, Feature_trans=opt.fft_feature_trans),
        ).to(opt.device)

        _ = test_one_run(opt, test_model, scaler, ckpt_path, run_dir, logger)
        logger.info(f"[DONE] run_dir={run_dir}")
