# run_dsfsfd_test.py
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Union

import numpy as np
import torch
from tqdm import tqdm

try:
    import yaml
except Exception as e:
    raise RuntimeError("Missing dependency: pyyaml. Please `pip install pyyaml`") from e

from src.engine.common_tools import (
    ensure_device,
    resolve_path,
    save_json,
    set_seed,
    setup_logger,
    stats_line,
)

from src.models.backbones.dsfsfd.blocks import Resnet1d, Resnet2d
from src.models.backbones.dsfsfd.net import MahFusion_Network
from src.models.backbones.dsfsfd.loss import Fusion_loss
from src.data.samplers.Data_Sampler_dsfsfd import TrainSampler, TestSampler


def load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def as_list(x: Union[str, List[str]]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [str(x)]


def build_opt(
    train_cfg: Dict[str, Any],
    project_root: Path,
    train_domain_for_scaler: str,
    test_domain: str,
    iters: int,
    overrides: Dict[str, Any],
) -> SimpleNamespace:
    opt = SimpleNamespace()

    # seed/device: allow override
    device = overrides.get("device", train_cfg.get("device", "cuda"))
    seed = overrides.get("seed", train_cfg.get("seed", 2025))
    opt.seed = int(seed)
    opt.device = ensure_device(str(device))

    # data.root comes from training config.json
    data_root = resolve_path(project_root, train_cfg["data"]["root"])
    opt.data_dir = str(data_root)

    # domains
    opt.train_domain = train_domain_for_scaler
    opt.test_domain = test_domain

    # task (val for test)
    task_override = overrides.get("task_override", {}) or {}
    opt.k_val = int(task_override.get("k_val", train_cfg["task"]["k_val"]))
    opt.n_val = int(task_override.get("n_val", train_cfg["task"]["n_val"]))
    opt.q_val = int(task_override.get("q_val", train_cfg["task"]["q_val"]))

    # train task (TrainSampler needs these to init; keep from train cfg)
    opt.k_train = int(train_cfg["task"]["k_train"])
    opt.n_train = int(train_cfg["task"]["n_train"])
    opt.q_train = int(train_cfg["task"]["q_train"])

    # iters
    opt.test_iters = int(iters)

    # model params
    opt.TF_weight = train_cfg["model"]["loss"]["TF_weight"]
    opt.DE_weight = train_cfg["model"]["loss"]["DE_weight"]
    opt.FFT_weight = train_cfg["model"]["loss"]["FFT_weight"]

    opt.tf_blockdim = train_cfg["model"]["encoders"]["tf"]["blockdim"]
    opt.de_blockdim = train_cfg["model"]["encoders"]["de"]["blockdim"]
    opt.fft_blockdim = train_cfg["model"]["encoders"]["fft"]["blockdim"]

    opt.tf_feature_trans = bool(train_cfg["model"]["encoders"]["tf"].get("feature_trans", True))
    opt.de_feature_trans = bool(train_cfg["model"]["encoders"]["de"].get("feature_trans", True))
    opt.fft_feature_trans = bool(train_cfg["model"]["encoders"]["fft"].get("feature_trans", True))

    return opt


def build_model(opt: SimpleNamespace) -> torch.nn.Module:
    return MahFusion_Network(
        TF_encoder=Resnet2d(blockdim=opt.tf_blockdim, Feature_trans=opt.tf_feature_trans),
        DE_encoder=Resnet1d(blockdim=opt.de_blockdim, Feature_trans=opt.de_feature_trans),
        FFT_encoder=Resnet1d(blockdim=opt.fft_blockdim, Feature_trans=opt.fft_feature_trans),
    ).to(opt.device)


@torch.no_grad()
def test_one_domain(opt: SimpleNamespace, ckpt_path: Path, out_dir: Path, logger):
    # 1) build scaler from train_domain_for_scaler
    logger.info(f"[Scaler] build from domain={opt.train_domain}")
    ts = TrainSampler(opt=opt)
    scaler = ts.scaler

    # 2) build model + load ckpt
    model = build_model(opt)
    sd = torch.load(str(ckpt_path), map_location=opt.device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(f"[CKPT] {ckpt_path}")
    logger.info(f"[CKPT] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    if missing:
        logger.info(f"[CKPT] missing sample: {missing[:10]}")
    if unexpected:
        logger.info(f"[CKPT] unexpected sample: {unexpected[:10]}")
    model.eval()

    loss_fn = Fusion_loss(opt=opt)

    # 3) sampler
    logger.info(f"[Test] domain={opt.test_domain} iters={opt.test_iters} k={opt.k_val} n={opt.n_val} q={opt.q_val}")
    vs = TestSampler(opt=opt, scaler=scaler)
    it = iter(vs)

    test_acc_sum = 0.0
    curve = []

    pbar = tqdm(
        range(1, opt.test_iters + 1),
        desc=f"Test({opt.test_domain})",
        dynamic_ncols=True,
        leave=True,
        position=0,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]"
    )
    info_bar = tqdm(total=0, position=1, leave=False, dynamic_ncols=True, bar_format="{desc}")
    info_bar.set_description_str("[00:00<??:??,  ?.?it/s, avg_acc=...]")

    for i in pbar:
        TFs, DEs, FFTs, TFq, DEq, FFTq = next(it)
        out = model(TFs=TFs, TFq=TFq, DEs=DEs, DEq=DEq, FFTs=FFTs, FFTq=FFTq)
        _, acc = loss_fn(*out)
        test_acc_sum += float(acc.item())

        if i % 100 == 0:
            avg_acc = test_acc_sum / i
            curve.append(avg_acc)
            info_bar.set_description_str(stats_line(pbar, {"avg_acc": f"{avg_acc:.4f}"}))

    info_bar.close()

    final_acc = test_acc_sum / opt.test_iters
    logger.info(f"[Test] FINAL avg_acc={final_acc:.6f}")

    # 4) save
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    ckpt_tag = ckpt_path.stem
    dom_tag = opt.test_domain

    curve_path = metrics_dir / f"test_curve_{dom_tag}_{ckpt_tag}.csv"
    np.savetxt(curve_path, np.array(curve, dtype=np.float32), fmt="%.6f")

    summary = {
        "ckpt": str(ckpt_path),
        "train_domain_for_scaler": opt.train_domain,
        "test_domain": opt.test_domain,
        "iters": opt.test_iters,
        "k": opt.k_val,
        "n": opt.n_val,
        "q": opt.q_val,
        "final_avg_acc": float(final_acc),
        "missing_keys": int(len(missing)),
        "unexpected_keys": int(len(unexpected)),
    }
    summary_path = metrics_dir / f"test_summary_{dom_tag}_{ckpt_tag}.json"
    save_json(summary_path, summary)

    logger.info(f"[Saved] {curve_path}")
    logger.info(f"[Saved] {summary_path}")

    return final_acc, curve_path, summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="test yaml config path")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    test_cfg_path = resolve_path(project_root, args.config)
    test_cfg = load_yaml(test_cfg_path)

    # exp_dir + training config
    exp_dir = resolve_path(project_root, test_cfg["exp_dir"])
    if not exp_dir.exists():
        raise FileNotFoundError(f"exp_dir not found: {exp_dir}")

    train_cfg_path = exp_dir / "meta" / "config.json"
    if not train_cfg_path.exists():
        raise FileNotFoundError(f"Cannot find: {train_cfg_path}")

    train_cfg = load_json(train_cfg_path)

    # ckpt
    ckpt_path = Path(test_cfg.get("ckpt", "checkpoints/last.pth"))
    ckpt_path = ckpt_path if ckpt_path.is_absolute() else (exp_dir / ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    # domains
    test_domains = as_list(test_cfg.get("test_domains"))
    if not test_domains:
        raise ValueError("test_domains is empty in test yaml")

    train_domain_for_scaler = test_cfg.get("train_domain_for_scaler", train_cfg["data"]["train_domain"])

    # iters
    iters = int(test_cfg.get("iters", train_cfg.get("test", {}).get("iters", 500)))

    # output_dir (default exp_dir)
    out_dir = resolve_path(project_root, test_cfg.get("output_dir", str(exp_dir)))
    out_dir.mkdir(parents=True, exist_ok=True)

    # logger
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"test_{ckpt_path.stem}.log"
    logger = setup_logger(log_file, name="dsfsfd_test")

    logger.info(f"[TestConfig] {test_cfg_path}")
    logger.info(f"[TrainConfig] {train_cfg_path}")
    logger.info(f"[ExpDir] {exp_dir}")
    logger.info(f"[OutDir] {out_dir}")
    logger.info(f"[CKPT] {ckpt_path}")
    logger.info(f"[ScalerDomain] {train_domain_for_scaler}")
    logger.info(f"[Iters] {iters}")
    logger.info(f"[TestDomains] {test_domains}")

    # 记录这次测试的配置（便于复现）
    save_json(out_dir / "meta" / f"test_config_{ckpt_path.stem}.json", {
        "test_yaml": str(test_cfg_path),
        "exp_dir": str(exp_dir),
        "out_dir": str(out_dir),
        "ckpt": str(ckpt_path),
        "train_domain_for_scaler": train_domain_for_scaler,
        "iters": iters,
        "test_domains": test_domains,
        "task_override": test_cfg.get("task_override", None),
        "device": test_cfg.get("device", None),
        "seed": test_cfg.get("seed", None),
    })

    # build opt + loop domains
    overrides = {
        "device": test_cfg.get("device", None),
        "seed": test_cfg.get("seed", None),
        "task_override": test_cfg.get("task_override", None),
    }
    # remove Nones
    overrides = {k: v for k, v in overrides.items() if v is not None}

    # seed once (before loop)
    seed = overrides.get("seed", train_cfg.get("seed", 2025))
    set_seed(int(seed))

    results = []
    for dom in test_domains:
        opt = build_opt(
            train_cfg=train_cfg,
            project_root=project_root,
            train_domain_for_scaler=train_domain_for_scaler,
            test_domain=dom,
            iters=iters,
            overrides=overrides,
        )
        acc, curve_path, summary_path = test_one_domain(opt, ckpt_path, out_dir, logger)
        results.append({"domain": dom, "acc": float(acc), "curve": str(curve_path), "summary": str(summary_path)})

    # summary table
    summary_all = out_dir / "metrics" / f"test_all_{ckpt_path.stem}.json"
    save_json(summary_all, {"ckpt": str(ckpt_path), "results": results})
    logger.info(f"[Saved] {summary_all}")

    print("\n[Test DONE]")
    for r in results:
        print(f"  {r['domain']}: acc={r['acc']:.6f}")


if __name__ == "__main__":
    main()
