from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from fs_net.dataset import FSNetDataConfig, FSNetNpzDataset
from fs_net.model import FSNetConfig, FSNetTorch

from train_wf_models import MODEL_CONFIG
from wf.features import load_or_build_features
from wf.models import AWF, BAPM, DF, DFsimCLR, NetCLR, RF, TF, TikTok, TMWF, VarCNN
from laserbeak_wf.features import LaserbeakDataConfig, LaserbeakNpzDataset, collate_and_pad
from laserbeak_wf.model import DFNet
from stmwf_wf.features import STMWFNpzDataset, load_or_build_stmwf_features
from stmwf_wf.model import BertExtractSTMWF


def _macro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "F1-score": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def _infer_wf_model(
    model_name: str,
    input_b_npz: Path,
    model_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    feature_cache: Path | None,
    force_rebuild_features: bool,
) -> tuple[dict[str, float], int]:
    cfg = MODEL_CONFIG[model_name]
    feature_name = str(cfg["feature"]).upper()
    if feature_name in {"DT", "DT2", "TAM"}:
        with np.load(input_b_npz, allow_pickle=True) as d_check:
            if "times" not in d_check:
                raise ValueError(f"{model_name} 使用特征 {feature_name}，输入 npz 必须包含 times 字段")

    X, y = load_or_build_features(
        input_b_npz,
        feature=feature_name,
        seq_len=int(cfg["seq_len"]),
        mtaf_align_len=None,
        cache_path=feature_cache,
        force_rebuild=force_rebuild_features,
    )
    classes = np.unique(y)
    if classes.size == 0:
        raise ValueError("B labels empty")
    num_classes = int(classes.max() + 1)
    if num_classes != classes.size:
        raise ValueError("B labels must be continuous 0..C-1")

    # Build model to load weights.
    if model_name == "DF":
        model = DF(num_classes)
    elif model_name == "AWF":
        model = AWF(num_classes)
    elif model_name == "BAPM":
        model = BAPM(num_classes, num_tab=int(cfg.get("num_tab", 1)))
    elif model_name == "RF":
        model = RF(num_classes)
    elif model_name == "NETCLR":
        model = NetCLR(num_classes)
    elif model_name == "TF":
        model = TF(num_classes)
    elif model_name == "TMWF":
        model = TMWF(num_classes, num_tab=int(cfg.get("num_tab", 1)))
    elif model_name == "TIKTOK":
        model = TikTok(num_classes)
    elif model_name == "VARCNN":
        model = VarCNN(num_classes)
    else:
        raise ValueError(f"Unknown wf model: {model_name}")

    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    model.to(device)
    model.eval()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    preds: list[int] = []
    trues: list[int] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            logits = model(batch_x)
            pred = torch.argmax(logits, dim=1).cpu().numpy().astype(int)
            preds.extend(pred.tolist())
            trues.extend(batch_y.numpy().astype(int).tolist())

    y_pred = np.asarray(preds, dtype=np.int64)
    y_true = np.asarray(trues, dtype=np.int64)
    return _macro_metrics(y_true, y_pred), int(y_true.shape[0])


def _infer_fsnet(
    input_b_npz: Path,
    model_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    fsnet_cfg: dict[str, Any],
) -> tuple[dict[str, float], int]:
    # labels_num
    d = np.load(input_b_npz, allow_pickle=True)
    labels = d["labels"].astype(np.int64)
    classes = np.unique(labels)
    if classes.size == 0:
        raise ValueError("B labels empty")
    class_num = int(classes.max() + 1)

    data_cfg = FSNetDataConfig(
        max_flow_length=int(fsnet_cfg.get("max_flow_length", 200)),
        min_length=int(fsnet_cfg.get("min_length", 2)),
        length_block=int(fsnet_cfg.get("length_block", 1)),
        max_packet_length=int(fsnet_cfg.get("max_packet_length", 5000)),
        sample_mode=str(fsnet_cfg.get("sample_mode", "first")),
    )
    ds = FSNetNpzDataset(input_b_npz, data_cfg=data_cfg, labels_num=class_num)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    length_num = int(fsnet_cfg.get("max_packet_length", 5000)) // int(fsnet_cfg.get("length_block", 1)) + 4
    model = FSNetTorch(
        FSNetConfig(
            class_num=class_num,
            length_num=length_num,
            length_dim=16,
            hidden=int(fsnet_cfg.get("hidden", 128)),
            layer=int(fsnet_cfg.get("layer", 2)),
            keep_prob=float(fsnet_cfg.get("keep_prob", 0.8)),
            rec_loss_weight=float(fsnet_cfg.get("rec_loss_weight", 0.5)),
            grad_clip_norm=5.0,
        )
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    model.to(device)
    model.eval()

    preds: list[int] = []
    trues: list[int] = []
    with torch.no_grad():
        for flow, y in loader:
            flow = flow.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            _, pred = model(flow, y)
            preds.extend(pred.cpu().numpy().astype(int).tolist())
            trues.extend(y.cpu().numpy().astype(int).tolist())

    y_pred = np.asarray(preds, dtype=np.int64)
    y_true = np.asarray(trues, dtype=np.int64)
    return _macro_metrics(y_true, y_pred), int(y_true.shape[0])


def _infer_laserbeak(
    input_b_npz: Path,
    model_path: Path,
    model_cfg: dict[str, Any],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    feature_cache: Path | None,
    force_rebuild_features: bool,
) -> tuple[dict[str, float], int]:
    feature_list = list(model_cfg["feature_list"])
    input_size = int(model_cfg.get("seq_len", 10000))

    # labels -> class num
    d = np.load(input_b_npz, allow_pickle=True)
    y = d["labels"].astype(np.int64, copy=False)
    classes = np.unique(y)
    if classes.size == 0:
        raise ValueError("B labels empty")
    num_classes = int(classes.max() + 1)
    if num_classes != classes.size:
        raise ValueError("B labels must be continuous 0..C-1")

    model = DFNet(num_classes, input_channels=len(feature_list), input_size=input_size)
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    model.to(device)
    model.eval()

    ds = LaserbeakNpzDataset(input_b_npz, cfg=LaserbeakDataConfig(feature_list=tuple(feature_list), input_size=input_size))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_and_pad,
    )

    preds: list[int] = []
    trues: list[int] = []
    with torch.no_grad():
        for batch_x, batch_y, sample_sizes in loader:
            batch_x = batch_x.to(device, non_blocking=True).float()
            logits = model(batch_x, sample_sizes=sample_sizes)
            pred = torch.argmax(logits, dim=1).cpu().numpy().astype(int)
            preds.extend(pred.tolist())
            trues.extend(batch_y.numpy().astype(int).tolist())

    y_pred = np.asarray(preds, dtype=np.int64)
    y_true = np.asarray(trues, dtype=np.int64)
    return _macro_metrics(y_true, y_pred), int(y_true.shape[0])


def _infer_stmwf(
    input_b_npz: Path,
    model_path: Path,
    model_cfg: dict[str, Any],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    feature_cache: Path | None,
    force_rebuild_features: bool,
) -> tuple[dict[str, float], int]:
    # auto-detect num_classes
    d = np.load(input_b_npz, allow_pickle=True)
    if "labels" not in d:
        raise ValueError("STMWF 输入 npz 缺少 labels")
    labels = d["labels"].astype(np.int64, copy=False)
    classes = np.unique(labels)
    if classes.size == 0:
        raise ValueError("B labels empty")
    num_classes = int(classes.max() + 1)
    if num_classes != classes.size:
        raise ValueError("B labels must be continuous 0..C-1")

    model = BertExtractSTMWF(
        num_hidden=int(model_cfg.get("num_hidden", 250)),
        label_size=num_classes,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    model.to(device)
    model.eval()

    if feature_cache is not None:
        X, y_idx = load_or_build_stmwf_features(
            input_b_npz,
            cache_path=feature_cache,
            force_rebuild=force_rebuild_features,
            out_len=int(model_cfg.get("seq_len", 10000)),
            dtype=np.float16,
        )
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(y_idx)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        ds = STMWFNpzDataset(input_b_npz, out_len=int(model_cfg.get("seq_len", 10000)))
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )

    preds: list[int] = []
    trues: list[int] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True).float()
            logits, _ = model(batch_x)
            pred = torch.argmax(logits, dim=1).cpu().numpy().astype(int)
            preds.extend(pred.tolist())
            trues.extend(batch_y.numpy().astype(int).tolist())

    y_pred = np.asarray(preds, dtype=np.int64)
    y_true = np.asarray(trues, dtype=np.int64)
    return _macro_metrics(y_true, y_pred), int(y_true.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="对 B 部分数据进行分类，并输出各项 metric。")
    root = Path(__file__).resolve().parent

    parser.add_argument("--model", type=str, default="FSNET", choices=list(MODEL_CONFIG.keys()))
    parser.add_argument("--input-b", type=Path, default=root / "Dataset" / "DF" / "raw-data-50-1000-B.npz")
    parser.add_argument("--model-path", type=Path, default=None, help="权重路径；默认按 model 推断")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--feature-cache", type=Path, default=None, help="用于 wf 模型构建特征的缓存路径")
    parser.add_argument("--force-rebuild-features", action="store_true")

    # FSNET only
    parser.add_argument("--fsnet-max-flow-length", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = MODEL_CONFIG[args.model]

    if args.model_path is not None:
        model_path = args.model_path
    else:
        if args.model == "FSNET":
            model_path = root / "Models" / "fsnet_last.pt"
        else:
            # wf models
            model_path = root / "Models" / f"{args.model.lower()}_last.pt"

    if not model_path.is_file():
        raise SystemExit(f"model 权重不存在: {model_path}")

    if args.model == "FSNET":
        if args.fsnet_max_flow_length is not None:
            cfg = dict(cfg)
            cfg["max_flow_length"] = int(args.fsnet_max_flow_length)
        metrics, n_samples = _infer_fsnet(
            input_b_npz=args.input_b,
            model_path=model_path,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            fsnet_cfg=cfg,
        )
    else:
        if args.model == "LASERBEAK":
            metrics, n_samples = _infer_laserbeak(
                input_b_npz=args.input_b,
                model_path=model_path,
                model_cfg=cfg,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                feature_cache=args.feature_cache,
                force_rebuild_features=args.force_rebuild_features,
            )
        elif args.model == "STMWF":
            metrics, n_samples = _infer_stmwf(
                input_b_npz=args.input_b,
                model_path=model_path,
                model_cfg=cfg,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                feature_cache=args.feature_cache,
                force_rebuild_features=args.force_rebuild_features,
            )
        else:
            metrics, n_samples = _infer_wf_model(
                model_name=args.model,
                input_b_npz=args.input_b,
                model_path=model_path,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                feature_cache=args.feature_cache,
                force_rebuild_features=args.force_rebuild_features,
            )

    print(f"[{args.model}] B metrics (n_samples={n_samples}):")
    print(metrics)


if __name__ == "__main__":
    main()

