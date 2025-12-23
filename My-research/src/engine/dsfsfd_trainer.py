# src/engine/dsfsfd_trainer.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# 你需要把下面这些模块按我上面说的“搬家路径”放好
from ..models.backbones.dsfsfd.blocks import Resnet1d, Resnet2d
from ..models.backbones.dsfsfd.net import MahFusion_Network
from ..models.backbones.dsfsfd.loss import Fusion_loss
from ..data.samplers.Data_Sampler_dsfsfd import TrainSampler, TestSampler


def split_model_param(model: torch.nn.Module):
    """完全复用你原来的分组策略：gamma/beta -> ft_params，其它 -> model_params。"""
    model_params, ft_params = [], []
    for name, p in model.named_parameters():
        last = name.split(".")[-1]
        if last in ("gamma", "beta"):
            ft_params.append(p)
        else:
            model_params.append(p)
    return model_params, ft_params


def _ensure_cuda_ready(device: str):
    # 你的 sampler / loss 里有 hardcode .cuda()（比如 y = ... .cuda()）
    # 所以这里先明确要求 cuda（想支持 CPU 再去把这些 .cuda() 改成 .to(device)）
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    raise RuntimeError(
        "DSFSFD 当前实现（sampler/loss）默认使用 .cuda()，需要 GPU 才能跑。"
        "如果你想支持 CPU，我可以给你一版把所有 .cuda() 改成 .to(device) 的补丁。"
    )


def train_one_run(opt: SimpleNamespace, model: torch.nn.Module, result_dir: Path):
    """
    训练逻辑基本等价于你原 Train(opt, model, result_path)。:contentReference[oaicite:8]{index=8}
    """
    Tsampler = TrainSampler(opt=opt)
    scaler = Tsampler.scaler
    Titer = iter(Tsampler)

    loss_fn = Fusion_loss(opt=opt)

    model_params, ft_params = split_model_param(model)

    model_optim = torch.optim.Adam(model_params, lr=opt.lr)
    ft_optim = torch.optim.Adam(ft_params, weight_decay=1e-8, lr=opt.ft_lr)

    for epoch in range(1, opt.epochs + 1):
        ps_loss = ps_acc = 0.0
        pu_loss = pu_acc = 0.0

        for _ in range(opt.episodes):
            (ps_TFs, ps_DEs, ps_FFTs, ps_TFq, ps_DEq, ps_FFTq,
             pu_TFs, pu_DEs, pu_FFTs, pu_TFq, pu_DEq, pu_FFTq) = next(Titer)

            # reset fast weights
            for w in model_params:
                w.fast = None

            # ========= inner loop (support) =========
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

            # ========= outer loop (query) =========
            model.eval()
            out = model(
                TFs=pu_TFs, TFq=pu_TFq,
                DEs=pu_DEs, DEq=pu_DEq,
                FFTs=pu_FFTs, FFTq=pu_FFTq
            )
            puloss, puacc = loss_fn(*out)

            # update model params with meta_grad
            model_optim.zero_grad()
            for k, w in enumerate(model_params):
                w.grad = meta_grad[k]
            model_optim.step()

            # update ft params with puloss (你已修复 detach 的版本):contentReference[oaicite:9]{index=9}
            ft_optim.zero_grad()
            puloss.backward()
            ft_optim.step()

            ps_loss += float(psloss.item())
            ps_acc += float(psacc.item())
            pu_loss += float(puloss.item())
            pu_acc += float(puacc.item())

        ps_loss /= opt.episodes
        ps_acc /= opt.episodes
        pu_loss /= opt.episodes
        pu_acc /= opt.episodes

        print(
            f"[Train][E{epoch:03d}] "
            f"ps_loss={ps_loss:.6f} ps_acc={ps_acc:.4f} | "
            f"pu_loss={pu_loss:.6f} pu_acc={pu_acc:.4f}"
        )

        if epoch % opt.save_every == 0:
            ckpt = result_dir / "checkpoints" / f"{epoch}.pth"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt)

    return scaler


@torch.no_grad()
def test_one_run(opt: SimpleNamespace, model: torch.nn.Module, scaler, ckpt_path: Path):
    """
    测试逻辑等价于你原 Test(opt, model, scaler)，默认 500 iter。:contentReference[oaicite:10]{index=10}
    """
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
        avg_acc = test_acc / i
        if i % 100 == 0:
            result.append(avg_acc)
            print(f"[Test] iter={i} avg_acc={avg_acc:.4f}")

    return result


def build_opt_from_cfg(cfg: dict, project_root: Path) -> SimpleNamespace:
    """
    把 YAML cfg 转成原代码期望的 opt 字段。
    你原始 sampler 需要：data_dir/train_domain/test_domain、k/n/q、lr/ft_lr、权重等。
    """
    data_root = (project_root / cfg["data"]["root"]).resolve()
    # data_root 目录下应包含 N15_M07_F10 / N09_M07_F10 ... 这些文件夹
    opt = SimpleNamespace()

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

    opt.TF_weight = cfg["model"]["loss"]["TF_weight"]
    opt.DE_weight = cfg["model"]["loss"]["DE_weight"]
    opt.FFT_weight = cfg["model"]["loss"]["FFT_weight"]

    # 这些 blockdim 直接从 cfg/model 读（你不想拆通用就别动）
    opt.tf_blockdim = cfg["model"]["encoders"]["tf"]["blockdim"]
    opt.de_blockdim = cfg["model"]["encoders"]["de"]["blockdim"]
    opt.fft_blockdim = cfg["model"]["encoders"]["fft"]["blockdim"]

    opt.test_iters = cfg["test"].get("iters", 500)

    # device: DSFSFD 当前要求 cuda（sampler/loss 里 hardcode .cuda()）
    opt.device = _ensure_cuda_ready(cfg.get("device", "cuda"))

    # 输出目录
    out_base = (project_root / cfg["output"]["base_dir"]).resolve()
    exp_name = cfg["output"]["exp_name"]
    opt.base_dir = str(out_base / exp_name)

    # repeat
    opt.repeat = int(cfg["train"].get("repeat", 3))
    opt.seed = int(cfg.get("seed", 2025))

    return opt


def run(cfg: dict, project_root: Path):
    """
    被 src/main.py 调用的统一入口。
    """
    opt = build_opt_from_cfg(cfg, project_root)

    # 记录 cfg 到输出目录，方便复现实验
    base_dir = Path(opt.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "meta").mkdir(exist_ok=True)
    with open(base_dir / "meta" / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    for r in range(opt.repeat):
        run_dir = base_dir / f"run{r}"
        run_dir.mkdir(parents=True, exist_ok=True)

        per_name = f"{opt.k_val}way{opt.n_val}shot-[{opt.train_domain}--{opt.test_domain}]-Exp{r}"
        print(f"\n==== {per_name} ====")

        # build model (结构等价于你原 main 里创建的)
        train_model = MahFusion_Network(
            TF_encoder=Resnet2d(blockdim=opt.tf_blockdim, Feature_trans=True),
            DE_encoder=Resnet1d(blockdim=opt.de_blockdim, Feature_trans=True),
            FFT_encoder=Resnet1d(blockdim=opt.fft_blockdim, Feature_trans=True),
        ).to(opt.device)

        scaler = train_one_run(opt, train_model, run_dir)

        # test with last epoch ckpt (默认用 epochs.pth)
        ckpt_path = run_dir / "checkpoints" / f"{opt.epochs}.pth"
        test_model = MahFusion_Network(
            TF_encoder=Resnet2d(blockdim=opt.tf_blockdim, Feature_trans=True),
            DE_encoder=Resnet1d(blockdim=opt.de_blockdim, Feature_trans=True),
            FFT_encoder=Resnet1d(blockdim=opt.fft_blockdim, Feature_trans=True),
        ).to(opt.device)

        result = test_one_run(opt, test_model, scaler, ckpt_path)

        # save metrics
        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(metrics_dir / f"result_{r}.csv", np.array(result), fmt="%.6f")
        print(f"[Saved] {metrics_dir / f'result_{r}.csv'}")
