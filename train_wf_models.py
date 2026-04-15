# -*- coding: utf-8 -*-
"""
作用：统一训练入口，读取 Dataset 中 A 部分 npz，按 WFlib 各模型脚本中的 feature / 超参 训练对应网络。

输入 npz：labels、signed_sizes；DT/DT2/TAM 需 times。TIKTOK 用 DT、VARCNN 用 DT2（与 WFlib load_data 一致，见 wf/features.py 说明）。

特征缓存：指定 --feature-cache 路径时，若缓存与当前源文件 mtime/大小及特征参数一致则直接加载；否则现算并写入该文件。加 --force-rebuild-features 可强制重算并覆盖。

输出：
  - 控制台与 ./Logs/train_<模型>_<时间戳>.log 双写日志。
  - 训练结束后保存最后一轮权重至 ./Models/<model>_last.pt（可用 --output-model 覆盖）。
  - 结束时打印验证集 Accuracy / Precision / Recall / F1-score。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from wf.data_utils import evaluate, make_loader, set_seed, split_train_val
from wf.features import load_or_build_features
from wf.models import AWF, BAPM, DF, DFsimCLR, NetCLR, RF, TF, TikTok, TMWF, VarCNN
from fs_net.train import train_fsnet
from laserbeak_wf.train import train_laserbeak
from stmwf_wf.train import train_stmwf

# 与 WFlib scripts/*.sh 对齐：feature、序列长度、batch、优化器、学习率、调度器
MODEL_CONFIG: dict[str, dict[str, Any]] = {
    "LASERBEAK": {
        # Laserbeak 原版默认 6 通道（benchmark.py 中的配置）
        "feature_list": [
            "time_dirs",
            "times_norm",
            "cumul_norm",
            "iat_dirs",
            "inv_iat_log_dirs",
            "running_rates",
        ],
        # 原版 input_size 默认 10000
        "seq_len": 10000,
        "batch_size": 64,
        "optimizer": "AdamW",
        "lr": 1e-3,
        "weight_decay": 0.001,
        "label_smoothing": 0.0,
        "grad_clip_norm": 1.0,
        "default_epochs": 30,
        "warmup_epochs": 5,
    },
    "DF": {
        "feature": "DIR",
        "seq_len": 5000,
        "mtaf_align_len": None,
        "batch_size": 128,
        "optimizer": "Adamax",
        "lr": 2e-3,
        "scheduler": None,
    },
    "AWF": {
        "feature": "DIR",
        "seq_len": 3000,
        "mtaf_align_len": None,
        "batch_size": 256,
        "optimizer": "RMSprop",
        "lr": 8e-4,
        "scheduler": None,
    },
    "BAPM": {
        "feature": "DIR",
        "seq_len": 8500,
        "mtaf_align_len": None,
        "batch_size": 128,
        "optimizer": "Adam",
        "lr": 5e-4,
        "scheduler": None,
        "num_tab": 1,
    },
    "RF": {
        "feature": "TAM",
        "seq_len": 1800,
        "mtaf_align_len": None,
        "batch_size": 200,
        "optimizer": "Adam",
        "lr": 5e-4,
        "scheduler": None,
    },
    "NETCLR": {
        "feature": "DIR",
        "seq_len": 5000,
        "mtaf_align_len": None,
        "batch_size": 256,
        "optimizer": "Adam",
        "lr": 3e-4,
        "scheduler": None,
        "pretrain_epochs": 25,
        "finetune_epochs": 25,
        "proj_dim": 128,
    },
    "TF": {
        "feature": "DIR",
        "seq_len": 5000,
        "mtaf_align_len": None,
        "batch_size": 512,
        "optimizer": "Adam",
        "lr": 1e-4,
        "scheduler": None,
    },
    "TMWF": {
        "feature": "DIR",
        "seq_len": 30720,
        "mtaf_align_len": None,
        "batch_size": 80,
        "optimizer": "Adam",
        "lr": 5e-4,
        "scheduler": None,
        "num_tab": 1,
    },
    "TIKTOK": {
        "feature": "DT",
        "seq_len": 5000,
        "mtaf_align_len": None,
        "batch_size": 128,
        "optimizer": "Adamax",
        "lr": 2e-3,
        "scheduler": None,
    },
    "VARCNN": {
        "feature": "DT2",
        "seq_len": 5000,
        "mtaf_align_len": None,
        "batch_size": 50,
        "optimizer": "Adam",
        "lr": 1e-3,
        "scheduler": None,
    },
    "FSNET": {
        "max_flow_length": 200,
        "length_block": 1,
        "max_packet_length": 5000,
        "min_length": 2,
        "sample_mode": "first",
        "batch_size": 64,
        "lr": 1e-3,
        "hidden": 128,
        "layer": 2,
        "keep_prob": 0.8,
        "rec_loss_weight": 0.5,
        "epochs": 20,
    },
    "STMWF": {
        # STMWF 复现：输入会被构建成固定长度 10000 的单通道序列
        "seq_len": 10000,
        "batch_size": 32,
        "optimizer": "Adam",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "label_smoothing": None,  # STMWF 使用 BCEWithLogitsLoss，不用 label smoothing
        "grad_clip_norm": 1.0,
        "default_epochs": 30,
        "num_hidden": 250,
    },
}


class _TeeIO:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, s: str) -> None:
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self) -> None:
        for st in self.streams:
            st.flush()


def _setup_logging(model_name: str) -> tuple[Path, TextIO]:
    root = Path(__file__).resolve().parent
    log_dir = root / "Logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_{model_name}_{ts}.log"
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _TeeIO(sys.__stdout__, log_file)
    sys.stderr = _TeeIO(sys.__stderr__, log_file)
    return log_path, log_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统一模型训练入口（DF / AWF / BAPM / RF / NETCLR / TF / TMWF / TIKTOK / VARCNN / FSNET / LASERBEAK）"
    )
    parser.add_argument("--model", type=str, default="BAPM", choices=list(MODEL_CONFIG.keys()))
    parser.add_argument(
        "--input-a",
        type=Path,
        default=Path(__file__).resolve().parent / "Dataset" / "DF" / "raw-data-50-1000-A.npz",
        help="A 部分 npz",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=None,
        help="权重保存路径，默认 ./Models/<model>_last.pt",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练总轮数；NETCLR 默认 50；其它模型默认 30（若 MODEL_CONFIG 含 default_epochs 则以其为准）",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="默认随模型与 WFlib 脚本一致")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=None,
        help="特征缓存 .npz：存在且与源数据/参数一致则加载；否则现算并保存到此路径。不传则每次现算不写盘",
    )
    parser.add_argument(
        "--force-rebuild-features",
        action="store_true",
        help="忽略已有缓存，重新提取特征并覆盖 --feature-cache",
    )
    return parser.parse_args()


def _build_model(name: str, num_classes: int, cfg: dict[str, Any]) -> nn.Module:
    if name == "DF":
        return DF(num_classes)
    if name == "AWF":
        return AWF(num_classes)
    if name == "BAPM":
        return BAPM(num_classes, num_tab=int(cfg.get("num_tab", 1)))
    if name == "RF":
        return RF(num_classes)
    if name == "NETCLR":
        return NetCLR(num_classes)
    if name == "TF":
        return TF(num_classes)
    if name == "TMWF":
        return TMWF(num_classes, num_tab=int(cfg.get("num_tab", 1)))
    if name == "TIKTOK":
        return TikTok(num_classes)
    if name == "VARCNN":
        return VarCNN(num_classes)
    raise SystemExit(f"未知模型: {name}")


def _build_optimizer(
    name: str, model: nn.Module, lr: float, *, weight_decay: float = 0.0
) -> torch.optim.Optimizer:
    cls = getattr(torch.optim, name)
    if weight_decay > 0.0:
        return cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    return cls(model.parameters(), lr=lr)


def _augment_dir_batch(x: torch.Tensor) -> torch.Tensor:
    """轻量 NetCLR 增强：随机零化+随机局部平移+轻噪声。"""
    out = x.clone()
    b, _, l = out.shape
    keep_mask = (torch.rand((b, 1, l), device=out.device) > 0.05).float()
    out = out * keep_mask
    shifts = torch.randint(low=-20, high=21, size=(b,), device=out.device)
    for i in range(b):
        out[i] = torch.roll(out[i], shifts=int(shifts[i].item()), dims=-1)
    out = out + 0.01 * torch.randn_like(out)
    return out.clamp(min=-1.0, max=1.0)


def _info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    n = z1.size(0)
    sim = torch.matmul(z, z.t()) / temperature
    mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -1e9)
    pos = torch.cat([torch.arange(n, 2 * n, device=z.device), torch.arange(0, n, device=z.device)])
    return F.cross_entropy(sim, pos)


def _train_netclr_two_stage(
    model: NetCLR,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: dict[str, Any],
    total_epochs: int,
) -> tuple[NetCLR, dict[str, float]]:
    pre_epochs_cfg = int(cfg.get("pretrain_epochs", 25))
    ft_epochs_cfg = int(cfg.get("finetune_epochs", 25))
    total_cfg = pre_epochs_cfg + ft_epochs_cfg
    if total_epochs <= 0:
        raise SystemExit("--epochs 必须 > 0")
    pre_epochs = max(1, round(total_epochs * pre_epochs_cfg / total_cfg))
    ft_epochs = max(1, total_epochs - pre_epochs)

    proj_dim = int(cfg.get("proj_dim", 128))
    simclr = DFsimCLR(NetCLR(512), out_dim=proj_dim).to(device)
    optim_pre = _build_optimizer(str(cfg["optimizer"]), simclr, float(cfg["lr"]))

    print(f"[NetCLR] 预训练阶段: epochs={pre_epochs}")
    for epoch in range(1, pre_epochs + 1):
        simclr.train()
        sum_loss = 0.0
        seen = 0
        pbar = tqdm(train_loader, desc=f"NetCLR-Pretrain {epoch}/{pre_epochs}", unit="batch")
        for batch_x, _ in pbar:
            batch_x = batch_x.to(device, non_blocking=True)
            v1 = _augment_dir_batch(batch_x)
            v2 = _augment_dir_batch(batch_x)
            optim_pre.zero_grad(set_to_none=True)
            z1 = simclr(v1)
            z2 = simclr(v2)
            loss = _info_nce_loss(z1, z2, temperature=0.5)
            loss.backward()
            optim_pre.step()
            bs = batch_x.size(0)
            sum_loss += loss.item() * bs
            seen += bs
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"pretrain epoch {epoch}: loss={sum_loss / max(1, seen):.4f}")

    # 只加载 backbone（卷积特征），跳过 projector 与分类头，避免 fc 键不匹配
    ckpt = simclr.state_dict()
    backbone_sd: dict[str, torch.Tensor] = {}
    for k, v in ckpt.items():
        if k.startswith("backbone.fc."):
            continue
        if k.startswith("backbone."):
            backbone_sd[k[len("backbone."):]] = v
    log = model.load_state_dict(backbone_sd, strict=False)
    print(f"[NetCLR] 载入预训练backbone完成，missing={len(log.missing_keys)}, unexpected={len(log.unexpected_keys)}")

    optim_ft = _build_optimizer(str(cfg["optimizer"]), model, float(cfg["lr"]))
    criterion = nn.CrossEntropyLoss()
    print(f"[NetCLR] 微调阶段: epochs={ft_epochs}")
    for epoch in range(1, ft_epochs + 1):
        model.train()
        seen = 0
        correct = 0
        sum_loss = 0.0
        pbar = tqdm(train_loader, desc=f"NetCLR-Finetune {epoch}/{ft_epochs}", unit="batch")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optim_ft.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optim_ft.step()
            pred = torch.argmax(logits, dim=1)
            bs = batch_y.size(0)
            correct += (pred == batch_y).sum().item()
            seen += bs
            sum_loss += loss.item() * bs
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_acc = correct / max(1, seen)
        val_acc, _ = evaluate(model, val_loader, device)
        print(f"finetune epoch {epoch}: loss={sum_loss / max(1, seen):.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    _, final_metrics = evaluate(model, val_loader, device)
    return model, final_metrics


def main() -> None:
    args = parse_args()
    cfg = MODEL_CONFIG[args.model]
    log_path, _log_f = _setup_logging(args.model)
    print(f"日志文件: {log_path}")

    set_seed(args.seed)
    root = Path(__file__).resolve().parent
    if args.output_model is None:
        args.output_model = root / "Models" / f"{args.model.lower()}_last.pt"

    if not args.input_a.is_file():
        raise SystemExit(f"A 部分数据不存在: {args.input_a}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if args.model == "FSNET":
        total_epochs = int(args.epochs) if args.epochs is not None else int(cfg.get("epochs", 20))
        batch_size = int(args.batch_size) if args.batch_size is not None else int(cfg.get("batch_size", 64))
        metrics = train_fsnet(
            input_a_npz=args.input_a,
            output_model_path=args.output_model,
            val_ratio=float(args.val_ratio),
            seed=int(args.seed),
            batch_size=batch_size,
            epochs=total_epochs,
            lr=float(cfg.get("lr", 1e-3)),
            max_flow_length=int(cfg.get("max_flow_length", 200)),
            length_block=int(cfg.get("length_block", 1)),
            max_packet_length=int(cfg.get("max_packet_length", 5000)),
            min_length=int(cfg.get("min_length", 2)),
            keep_prob=float(cfg.get("keep_prob", 0.8)),
            hidden=int(cfg.get("hidden", 128)),
            layer=int(cfg.get("layer", 2)),
            rec_loss_weight=float(cfg.get("rec_loss_weight", 0.5)),
            sample_mode=str(cfg.get("sample_mode", "first")),
            num_workers=int(args.num_workers),
            device=str(device),
        )
        print("FSNET 训练结束，返回指标：")
        print(metrics)
        return

    if args.model == "LASERBEAK":
        total_epochs = int(args.epochs) if args.epochs is not None else int(cfg.get("default_epochs", 30))
        batch_size = int(args.batch_size) if args.batch_size is not None else int(cfg.get("batch_size", 64))
        metrics = train_laserbeak(
            input_a_npz=args.input_a,
            output_model_path=args.output_model,
            val_ratio=float(args.val_ratio),
            seed=int(args.seed),
            batch_size=batch_size,
            epochs=total_epochs,
            lr=float(cfg.get("lr", 1e-3)),
            optimizer_name=str(cfg.get("optimizer", "AdamW")),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
            feature_cache=args.feature_cache,
            force_rebuild_features=bool(args.force_rebuild_features),
            num_workers=int(args.num_workers),
            device=str(device),
            feature_list=list(cfg["feature_list"]),
            seq_len=int(cfg["seq_len"]),
            input_size=int(cfg.get("seq_len", 2000)),
            label_smoothing=float(cfg.get("label_smoothing", 0.1)),
            grad_clip_norm=cfg.get("grad_clip_norm", 1.0),
            warmup_epochs=int(cfg.get("warmup_epochs", 5)),
        )
        print("LASERBEAK 训练结束，返回指标：")
        print(metrics)
        return

    if args.model == "STMWF":
        total_epochs = int(args.epochs) if args.epochs is not None else int(cfg.get("default_epochs", 30))
        batch_size = int(args.batch_size) if args.batch_size is not None else int(cfg.get("batch_size", 32))
        metrics = train_stmwf(
            input_a_npz=args.input_a,
            output_model_path=args.output_model,
            val_ratio=float(args.val_ratio),
            seed=int(args.seed),
            batch_size=batch_size,
            epochs=total_epochs,
            lr=float(cfg.get("lr", 1e-3)),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
            feature_cache=args.feature_cache,
            force_rebuild_features=bool(args.force_rebuild_features),
            num_workers=int(args.num_workers),
            device=str(device),
            label_size=None,  # 自动检测
            out_len=int(cfg.get("seq_len", 10000)),
            num_hidden=int(cfg.get("num_hidden", 250)),
            grad_clip_norm=float(cfg.get("grad_clip_norm", 1.0)),
        )
        print("STMWF 训练结束，返回指标：")
        print(metrics)
        return

    mtaf_align = cfg.get("mtaf_align_len")
    mtaf_opt = int(mtaf_align) if mtaf_align is not None else None
    feature_name = str(cfg["feature"]).upper()
    if feature_name in {"DT", "DT2", "TAM"}:
        with np.load(args.input_a, allow_pickle=True) as d:
            if "times" not in d:
                raise SystemExit(f"{args.model} 使用特征 {feature_name}，输入 npz 必须包含 times 字段")
    X, y = load_or_build_features(
        args.input_a,
        feature=feature_name,
        seq_len=int(cfg["seq_len"]),
        mtaf_align_len=mtaf_opt,
        cache_path=args.feature_cache,
        force_rebuild=args.force_rebuild_features,
    )
    if args.feature_cache is not None:
        print(f"特征缓存路径: {args.feature_cache}")
    print(f"feature={cfg['feature']}, X.shape={X.shape}, seq_len={cfg['seq_len']}")

    classes = np.unique(y)
    if classes.size == 0:
        raise SystemExit("数据为空")
    num_classes = int(classes.max() + 1)
    if num_classes != classes.size:
        raise SystemExit("labels 须为 0..C-1 连续整数")

    X_train, y_train, X_val, y_val = split_train_val(X, y, args.val_ratio, args.seed)
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Valid: X={X_val.shape}, y={y_val.shape}, num_classes={num_classes}")

    batch_size = args.batch_size if args.batch_size is not None else int(cfg["batch_size"])
    train_loader = make_loader(X_train, y_train, batch_size, True, args.num_workers)
    val_loader = make_loader(X_val, y_val, batch_size, False, args.num_workers)

    model = _build_model(args.model, num_classes, cfg)
    if args.epochs is not None:
        total_epochs = int(args.epochs)
    elif args.model == "NETCLR":
        total_epochs = 50
    else:
        total_epochs = int(cfg.get("default_epochs", 30))
    wd = float(cfg["weight_decay"]) if cfg.get("weight_decay") is not None else 0.0
    optimizer = _build_optimizer(str(cfg["optimizer"]), model, float(cfg["lr"]), weight_decay=wd)

    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if cfg.get("scheduler") == "StepLR":
        step_sz = int(cfg.get("step_lr_step_size", 30))
        gamma = float(cfg.get("step_lr_gamma", 0.74))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_sz, gamma=gamma)
    elif cfg.get("scheduler") == "CosineAnnealingLR":
        eta_min = float(cfg.get("cosine_eta_min", 1e-6))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=eta_min
        )

    ls = float(cfg["label_smoothing"]) if cfg.get("label_smoothing") is not None else 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=ls)
    grad_clip = cfg.get("grad_clip_norm")
    model.to(device)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)

    if args.model == "NETCLR":
        model.to(device)
        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        model, final_metrics = _train_netclr_two_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            cfg=cfg,
            total_epochs=total_epochs,
        )
        torch.save(model.state_dict(), args.output_model)
        print(f"已保存最终模型 -> {args.output_model}")
        print("训练结束，验证集评估指标：")
        print(final_metrics)
        return

    for epoch in range(1, total_epochs + 1):
        model.train()
        seen = 0
        correct = 0
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}", unit="batch")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

            pred = torch.argmax(logits, dim=1)
            correct += (pred == batch_y).sum().item()
            seen += batch_y.shape[0]
            epoch_loss += loss.item() * batch_y.shape[0]
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_acc = correct / max(1, seen)
        train_loss = epoch_loss / max(1, seen)
        val_acc, _ = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
        )
        if scheduler is not None:
            scheduler.step()

    torch.save(model.state_dict(), args.output_model)
    print(f"已保存最终模型 -> {args.output_model}")

    _, final_metrics = evaluate(model, val_loader, device)
    print("训练结束，验证集评估指标：")
    print(final_metrics)


if __name__ == "__main__":
    main()
