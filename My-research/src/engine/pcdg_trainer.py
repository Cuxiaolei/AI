# src/engine/pcdg_trainer.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from tqdm import tqdm

from ..engine.common_tools import (
    EMA,
    append_csv_dict_row,
    ensure_device_any,
    make_unique_exp_dir,
    resolve_path,
    save_json,
    set_seed,
    setup_logger,
    stats_line,
    safe_name,
)

from ..data.samplers.pcdg_episode_sampler import PCDGTrainSampler, PCDGTestSampler


def _build_optimizer(cfg: dict, model: torch.nn.Module):
    opt_cfg = cfg.get("optim", {})
    name = str(opt_cfg.get("name", "adamw")).lower()
    lr = float(opt_cfg.get("lr", 1e-3))
    wd = float(opt_cfg.get("weight_decay", 1e-4))

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        mom = float(opt_cfg.get("momentum", 0.9))
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
    raise ValueError(f"Unknown optim name: {name}")


@torch.no_grad()
def _eval(model, memory, cfg: Dict[str, Any], test_domain: str, device: torch.device, eval_episodes: int, logger) -> float:
    model.eval()
    sampler = PCDGTestSampler(cfg, test_domain=test_domain)
    accs = []
    for _ in range(eval_episodes):
        epi = sampler.sample()
        support = {k: v.to(device) for k, v in epi["support"].items()}
        query = {k: v.to(device) for k, v in epi["query"].items()}
        class_ids = epi["class_ids"].to(device)
        out = model.forward_episode(support=support, query=query, class_ids=class_ids, memory=memory)
        accs.append(float(out["acc"]))
    return float(sum(accs) / max(1, len(accs)))


def run(cfg: Dict[str, Any], project_root: Path):
    """
    复用 DSFSFD 的输出目录与日志工具：
      outputs/<exp_name[_i]>/meta/config.json
      outputs/<exp_name[_i]>/run0/logs/train.log
      outputs/<exp_name[_i]>/run0/metrics/train_metrics.csv
      outputs/<exp_name[_i]>/run0/checkpoints/*.pth
    """
    # ---------- seed/device ----------
    seed = int(cfg.get("seed", 2025))
    set_seed(seed)

    device_str = cfg.get("train", {}).get("device", "cuda")
    device = ensure_device_any(device_str)

    # ---------- output dirs ----------
    out_cfg = cfg.get("output", {})
    base_dir = resolve_path(project_root, out_cfg.get("base_dir", "outputs"))
    exp_name = out_cfg.get("exp_name", "pcdg_exp")
    auto_inc = bool(out_cfg.get("auto_increment", True))

    exp_dir = make_unique_exp_dir(base_dir, exp_name) if auto_inc else (base_dir / exp_name)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 保存最终合并 cfg
    save_json(exp_dir / "meta" / "config.json", cfg)
    tqdm.write(f"[Output] Using experiment dir: {exp_dir}")

    # ---------- build model/loss ----------
    from ..models.backbones.pcdg.net import PCDGNet
    from ..models.backbones.pcdg.loss import PCDGLoss, PrototypeMemory

    model = PCDGNet(cfg["model"]).to(device)
    loss_fn = PCDGLoss(cfg["model"]).to(device)

    cont_cfg = cfg["model"].get("continual", {})
    memory = None
    if bool(cont_cfg.get("enable", True)):
        num_classes = int(cont_cfg["num_classes"])
        emb_dim = int(cfg["model"]["emb_dim"])
        momentum = float(cont_cfg.get("momentum", 0.9))
        memory = PrototypeMemory(num_classes=num_classes, emb_dim=emb_dim, momentum=momentum, device=str(device))
    # ---------- optimizer ----------
    optimizer = _build_optimizer(cfg, model)

    # ---------- experiment plan ----------
    exp_cfg = cfg.get("experiment", {})
    mode = str(exp_cfg.get("mode", "lodo")).lower()
    repeats = int(exp_cfg.get("repeats", 1))

    domains = list(cfg["data"]["domains"])
    runs: List[Tuple[List[str], str]] = []
    if mode == "lodo":
        for td in domains:
            tr = [d for d in domains if d != td]
            runs.append((tr, td))
    else:
        tr = list(cfg["data"]["train_domains"])
        td = str(cfg["data"]["test_domain"])
        runs.append((tr, td))

    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 50))
    save_every = int(train_cfg.get("save_every", 10))
    log_every = int(train_cfg.get("log_every", 10))
    show_mem = bool(train_cfg.get("show_mem", False))
    grad_clip = float(train_cfg.get("grad_clip", 5.0))

    ep_cfg = cfg["episode"]
    episodes_per_epoch = int(ep_cfg.get("episodes_per_epoch", 100))
    eval_episodes = int(ep_cfg.get("eval_episodes", 200))

    # ---------- metrics csv ----------
    header = [
        "rep", "run_name", "epoch",
        "train_loss", "train_acc",
        "ce", "supcon", "drift",
        "test_domain", "test_acc",
        "epoch_sec",
        "best_acc_sofar",
        "best_ckpt",
    ]

    best_acc_global = -1.0
    best_ckpt_global = ""

    # ---------- loop ----------
    for rep in range(repeats):
        run_dir = exp_dir / f"run{rep}"
        run_dir.mkdir(parents=True, exist_ok=True)

        logger = setup_logger(run_dir / "logs" / "train.log", name="pcdg")
        ckpt_dir = run_dir / "checkpoints"
        metrics_dir = run_dir / "metrics"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        csv_path = metrics_dir / "train_metrics.csv"

        logger.info(f"[Init] seed={seed} device={device} exp_dir={exp_dir} run_dir={run_dir}")
        logger.info(f"[Init] modalities={cfg['model'].get('modalities', {})}")

        for train_domains, test_domain in runs:
            # 写回 cfg 便于 sampler 使用
            cfg["data"]["train_domains"] = train_domains
            cfg["data"]["test_domain"] = test_domain

            run_name = f"{cfg['episode']['k']}way{cfg['episode']['n']}shot-[{'+'.join(train_domains)}--{test_domain}]"
            logger.info("\n" + "=" * 80)
            logger.info(f"[Run] {run_name}")
            logger.info(f"[Domains] train={train_domains} test={test_domain}")

            train_sampler = PCDGTrainSampler(cfg)

            # Epoch 总进度条（第1行）
            epoch_bar = tqdm(
                range(1, epochs + 1),
                desc="Train",
                dynamic_ncols=True,
                leave=True,
                position=0,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]"
            )

            best_acc_local = -1.0
            best_ckpt_local = ""

            for epoch in epoch_bar:
                t0 = time.time()
                model.train()

                # EMA smoother
                ema_loss = EMA(0.95)
                ema_acc = EMA(0.95)
                ema_ce = EMA(0.95)
                ema_supcon = EMA(0.95)
                ema_drift = EMA(0.95)

                loss_sum = 0.0
                acc_sum = 0.0
                ce_sum = 0.0
                supcon_sum = 0.0
                drift_sum = 0.0

                # Episode bar（第2行）
                epi_bar = tqdm(
                    range(1, episodes_per_epoch + 1),
                    desc=f"E{epoch}/{epochs}",
                    dynamic_ncols=True,
                    leave=False,
                    position=1,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]"
                )

                # stats（第3行）
                info_bar = tqdm(
                    total=0,
                    position=2,
                    leave=False,
                    dynamic_ncols=True,
                    bar_format="{desc}"
                )
                info_bar.set_description_str("[00:00<??:??,  ?.?it/s, loss=..., acc=..., ce=..., supcon=..., drift=...]")

                for step in epi_bar:
                    epi = next(train_sampler)
                    support = {k: v.to(device) for k, v in epi["support"].items()}
                    query = {k: v.to(device) for k, v in epi["query"].items()}
                    class_ids = epi["class_ids"].to(device)

                    out = model.forward_episode(support=support, query=query, class_ids=class_ids, memory=memory)
                    loss_dict = loss_fn(out)
                    loss = loss_dict["total"]

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                    if memory is not None:
                        memory.update(class_ids=class_ids, protos=out["protos"].detach())

                    loss_sum += float(loss.item())
                    acc_sum += float(out["acc"])
                    ce_sum += float(loss_dict["ce"].item())
                    supcon_sum += float(loss_dict["supcon"].item())
                    drift_sum += float(loss_dict["drift"].item())

                    # 每 log_every 更新一次第3行 stats
                    if step == 1 or step == episodes_per_epoch or (step % log_every == 0):
                        loss_s = ema_loss.update(loss_sum / step)
                        acc_s = ema_acc.update(acc_sum / step)
                        ce_s = ema_ce.update(ce_sum / step)
                        supcon_s = ema_supcon.update(supcon_sum / step)
                        drift_s = ema_drift.update(drift_sum / step)

                        metrics = {
                            "loss": f"{loss_s:.4f}",
                            "acc": f"{acc_s:.3f}",
                            "ce": f"{ce_s:.4f}",
                            "supcon": f"{supcon_s:.4f}",
                            "drift": f"{drift_s:.4f}",
                        }
                        if show_mem and torch.cuda.is_available():
                            mem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                            metrics["memGB"] = f"{mem_gb:.2f}"

                        info_bar.set_description_str(stats_line(epi_bar, metrics))

                info_bar.close()

                # epoch summary
                epoch_sec = time.time() - t0
                train_loss = loss_sum / episodes_per_epoch
                train_acc = acc_sum / episodes_per_epoch
                ce_avg = ce_sum / episodes_per_epoch
                supcon_avg = supcon_sum / episodes_per_epoch
                drift_avg = drift_sum / episodes_per_epoch

                # eval
                test_acc = _eval(model, memory, cfg, test_domain, device, eval_episodes, logger)

                epoch_bar.set_postfix({"tr_acc": f"{train_acc:.3f}", "te_acc": f"{test_acc:.3f}"})

                logger.info(
                    f"[Train][{run_name}][E{epoch:03d}] DONE "
                    f"loss={train_loss:.6f} acc={train_acc:.4f} | "
                    f"ce={ce_avg:.6f} supcon={supcon_avg:.6f} drift={drift_avg:.6f} | "
                    f"test({test_domain})={test_acc:.4f} | "
                    f"time={epoch_sec:.1f}s"
                )

                # checkpoint
                torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_dir / "last.pth")
                if (epoch % save_every) == 0:
                    torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_dir / f"epoch_{epoch}.pth")

                # best local & global
                best_path = ""
                if test_acc > best_acc_local:
                    best_acc_local = test_acc
                    best_path = ckpt_dir / f"best_{safe_name(run_name)}.pth"
                    torch.save({"model": model.state_dict(), "cfg": cfg}, best_path)
                    best_ckpt_local = str(best_path)

                if test_acc > best_acc_global:
                    best_acc_global = test_acc
                    best_ckpt_global = str(best_path) if best_path else best_ckpt_local

                # csv append
                append_csv_dict_row(
                    csv_path,
                    fieldnames=header,
                    row_dict={
                        "rep": rep,
                        "run_name": run_name,
                        "epoch": epoch,
                        "train_loss": f"{train_loss:.6f}",
                        "train_acc": f"{train_acc:.6f}",
                        "ce": f"{ce_avg:.6f}",
                        "supcon": f"{supcon_avg:.6f}",
                        "drift": f"{drift_avg:.6f}",
                        "test_domain": test_domain,
                        "test_acc": f"{test_acc:.6f}",
                        "epoch_sec": f"{epoch_sec:.3f}",
                        "best_acc_sofar": f"{best_acc_global:.6f}",
                        "best_ckpt": best_ckpt_global,
                    }
                )

        logger.info(f"[DONE] exp_dir={exp_dir} best_acc={best_acc_global:.4f} best_ckpt={best_ckpt_global}")
