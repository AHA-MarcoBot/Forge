from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from fs_net.dataset import FSNetDataConfig, FSNetNpzDataset
from fs_net.model import FSNetConfig, FSNetTorch
from laserbeak_wf.features import LaserbeakDataConfig, LaserbeakNpzDataset, collate_and_pad
from laserbeak_wf.model import DFNet as LaserbeakDFNet
from stmwf_wf.model import BertExtractSTMWF
from stmwf_wf.features import STMWFNpzDataset
from attack_plot_utils import save_label_agm_weight_figure, save_label_packet_distribution_figure
from train_wf_models import MODEL_CONFIG
from wf.features import load_or_build_features
from wf.models import AWF, BAPM, DF, NetCLR, RF, TF, TikTok, TMWF, VarCNN


WF_MODELS = {"DF", "AWF", "BAPM", "RF", "NETCLR", "TF", "TMWF", "TIKTOK", "VARCNN"}
ALL_MODELS = tuple(MODEL_CONFIG.keys())
DEFAULT_DUMMY_LEN_MAX = 1024


def _macro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "F1-score": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def _aggregate_victim_metrics(
    class_reports: list[dict[str, Any]],
    *,
    exclude_victims: set[str] | None = None,
) -> dict[str, Any]:
    exclude = exclude_victims or set()
    metric_keys = ("Accuracy", "Precision", "Recall", "F1-score")
    clean_vals: dict[str, list[float]] = {k: [] for k in metric_keys}
    adv_vals: dict[str, list[float]] = {k: [] for k in metric_keys}
    acc_drop_vals: list[float] = []
    used_pairs = 0
    used_labels: set[int] = set()
    used_victims: set[str] = set()

    for cr in class_reports:
        label = int(cr.get("label", -1))
        victims = cr.get("victims", {})
        if not isinstance(victims, dict):
            continue
        for vm, vm_res in victims.items():
            vm_upper = str(vm).upper()
            if vm_upper in exclude:
                continue
            if not isinstance(vm_res, dict) or ("error" in vm_res):
                continue
            clean = vm_res.get("clean", {})
            adv = vm_res.get("adv", {})
            if not isinstance(clean, dict) or not isinstance(adv, dict):
                continue
            if any(k not in clean or k not in adv for k in metric_keys):
                continue
            for k in metric_keys:
                clean_vals[k].append(float(clean[k]))
                adv_vals[k].append(float(adv[k]))
            acc_drop_vals.append(float(vm_res.get("acc_drop", 0.0)))
            used_pairs += 1
            used_labels.add(label)
            used_victims.add(vm_upper)

    if used_pairs == 0:
        return {
            "pair_count": 0,
            "label_count": 0,
            "victim_count": 0,
            "mean_clean": {k: 0.0 for k in metric_keys},
            "mean_adv": {k: 0.0 for k in metric_keys},
            "mean_acc_drop": 0.0,
        }

    return {
        "pair_count": int(used_pairs),
        "label_count": int(len(used_labels)),
        "victim_count": int(len(used_victims)),
        "mean_clean": {k: round(float(np.mean(v)), 4) for k, v in clean_vals.items()},
        "mean_adv": {k: round(float(np.mean(v)), 4) for k, v in adv_vals.items()},
        "mean_acc_drop": round(float(np.mean(acc_drop_vals)), 4),
    }


def _model_default_path(root: Path, model: str) -> Path:
    if model == "FSNET":
        return root / "Models" / "fsnet_last.pt"
    return root / "Models" / f"{model.lower()}_last.pt"


def _build_wf_model(name: str, num_classes: int, cfg: dict[str, Any]) -> torch.nn.Module:
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
    raise ValueError(f"Unsupported wf model: {name}")


def _smooth_sign(x: torch.Tensor, k: float = 0.1) -> torch.Tensor:
    return torch.tanh(k * x)


def _extract_logits(model_output: Any) -> torch.Tensor:
    """Unify model forward outputs: logits or (logits, ...)."""
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


def _align_1d_torch(x: torch.Tensor, L: int) -> torch.Tensor:
    if x.shape[-1] >= L:
        return x[..., :L]
    pad = L - x.shape[-1]
    return F.pad(x, (0, pad))


def _build_feature_torch(
    model_name: str,
    cfg: dict[str, Any],
    signed_sizes: torch.Tensor,  # (B,S)
    times: torch.Tensor,  # (B,S)
    *,
    exact_eval: bool = False,
) -> torch.Tensor:
    feature = str(cfg.get("feature", "DT")).upper()
    seq_len = int(cfg.get("seq_len", signed_sizes.shape[-1]))
    x = signed_sizes
    if feature == "DIR":
        out = (torch.sign(x) if exact_eval else _smooth_sign(x)).unsqueeze(1)
        return _align_1d_torch(out, seq_len)
    if feature == "DT":
        # DT: directional timestamp sequence sign(signed_sizes) * abs(times)
        dir_part = torch.sign(x) if exact_eval else _smooth_sign(x)
        out = (dir_part * torch.abs(times)).unsqueeze(1)
        return _align_1d_torch(out, seq_len)
    if feature == "DT2":
        # DT2: dir=sign(signed_sizes), time=diff(abs(times)) clipped at 0
        x_dir = torch.sign(x) if exact_eval else _smooth_sign(x)
        t_abs = torch.abs(times)
        x_time = t_abs[:, 1:] - t_abs[:, :-1]
        x_time = torch.relu(x_time)
        x_time = F.pad(x_time, (0, 1))
        out = torch.stack([x_dir, x_time], dim=1)
        return _align_1d_torch(out, seq_len)
    if feature == "TAM":
        bins = 1800
        max_time = 80.0
        B, S = times.shape
        rel_t = torch.clamp(times - times[:, :1], min=0.0, max=max_time)
        idx = torch.clamp((rel_t * (bins - 1) / max_time).long(), 0, bins - 1)
        out = torch.zeros((B, 1, 2, bins), dtype=times.dtype, device=times.device)
        if exact_eval:
            pos = (x > 0).to(times.dtype)
            neg = (x < 0).to(times.dtype)
            pos_hist = torch.zeros((B, bins), dtype=times.dtype, device=times.device)
            neg_hist = torch.zeros((B, bins), dtype=times.dtype, device=times.device)
            pos_hist.scatter_add_(1, idx, pos)
            neg_hist.scatter_add_(1, idx, neg)
            out[:, 0, 0, :] = pos_hist
            out[:, 0, 1, :] = neg_hist
        else:
            # Soft TAM (BPDA-friendly): linearly splat timestamps to nearest bins
            u = rel_t * (bins - 1) / max_time  # (B,S)
            l = torch.floor(u).long().clamp(0, bins - 1)
            r = torch.clamp(l + 1, max=bins - 1)
            wr = (u - l.float())
            wl = 1.0 - wr
            # Use a smooth sign split instead of ReLU around zero to avoid dead gradients
            # when dummy amplitudes start at (or are rounded to) zero.
            sgn_soft = _smooth_sign(x)
            pos = 0.5 * (1.0 + sgn_soft)
            neg = 0.5 * (1.0 - sgn_soft)
            pos_hist = torch.zeros((B, bins), dtype=times.dtype, device=times.device)
            neg_hist = torch.zeros((B, bins), dtype=times.dtype, device=times.device)
            pos_hist.scatter_add_(1, l, wl * pos)
            pos_hist.scatter_add_(1, r, wr * pos)
            neg_hist.scatter_add_(1, l, wl * neg)
            neg_hist.scatter_add_(1, r, wr * neg)
            out[:, 0, 0, :] = pos_hist
            out[:, 0, 1, :] = neg_hist
        return out
    raise ValueError(f"Unsupported feature for surrogate: {feature}")


def _build_laserbeak_feature_torch(
    signed_sizes: torch.Tensor, times: torch.Tensor, feature_list: list[str], seq_len: int
) -> torch.Tensor:
    B, S = signed_sizes.shape
    x = signed_sizes
    t = times
    dirs = _smooth_sign(x)
    sizes = torch.abs(x)
    iats = torch.diff(t, dim=1, prepend=torch.zeros_like(t[:, :1]))
    channels = []
    for feat in feature_list:
        f = feat.lower()
        if f == "time_dirs":
            c = t * dirs
        elif f == "times_norm":
            c = t - t.mean(dim=1, keepdim=True)
            c = c / (torch.amax(torch.abs(c), dim=1, keepdim=True) + 1e-8)
        elif f == "cumul_norm":
            cumul = torch.cumsum(sizes * dirs, dim=1)
            c = cumul - cumul.mean(dim=1, keepdim=True)
            c = c / (torch.amax(torch.abs(c), dim=1, keepdim=True) + 1e-8)
        elif f == "iat_dirs":
            c = (1.0 + iats) * dirs
        elif f == "inv_iat_log_dirs":
            inv = torch.log(torch.nan_to_num((1.0 / (iats + 1e-8)) + 1.0, nan=1e4, posinf=1e4, neginf=1e4))
            c = inv * dirs
        elif f == "running_rates":
            tt = torch.cumsum(iats, dim=1)
            ss = torch.cumsum(sizes, dim=1)
            c = torch.where(tt != 0, ss / tt, torch.ones_like(tt))
        else:
            raise ValueError(f"Unsupported Laserbeak feature: {feat}")
        channels.append(c)
    out = torch.stack(channels, dim=1)  # B,C,S
    return _align_1d_torch(out, seq_len)


def _build_stmwf_feature_torch(signed_sizes: torch.Tensor, times: torch.Tensor, out_len: int = 10000) -> torch.Tensor:
    dirs = _smooth_sign(signed_sizes)
    iats = torch.diff(times, dim=1, prepend=torch.zeros_like(times[:, :1]))
    seq = (1.0 + torch.relu(iats)) * dirs
    return _align_1d_torch(seq.unsqueeze(1), out_len)


def _build_surrogate_for_grad(
    model_name: str,
    model_path: Path,
    num_classes: int,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, str]:
    """
    Returns:
      - model used in backward
      - grad_mode: "native" or "bpda_df_proxy"
    """
    if model_name in WF_MODELS:
        m = _build_wf_model(model_name, num_classes, cfg)
        m.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
        return m.to(device).eval(), "native"
    if model_name == "LASERBEAK":
        m = LaserbeakDFNet(num_classes, input_channels=len(cfg["feature_list"]), input_size=int(cfg["seq_len"]))
        m.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
        return m.to(device).eval(), "native"
    if model_name == "STMWF":
        m = BertExtractSTMWF(num_hidden=int(cfg.get("num_hidden", 250)), label_size=num_classes)
        m.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
        return m.to(device).eval(), "native"

    # FSNET uses discrete token IDs -> use BPDA proxy (DF on DT)
    proxy_cfg = MODEL_CONFIG["DF"]
    proxy_path = model_path.parent / "df_last.pt"
    if not proxy_path.is_file():
        raise FileNotFoundError(f"FSNET surrogate needs BPDA proxy df_last.pt, not found: {proxy_path}")
    proxy = DF(num_classes)
    proxy.load_state_dict(torch.load(proxy_path, map_location="cpu"), strict=True)
    return proxy.to(device).eval(), "bpda_df_proxy"


def _forward_surrogate_logits(
    *,
    name: str,
    model: torch.nn.Module,
    grad_mode: str,
    cfg: dict[str, Any],
    signed_adv: torch.Tensor,
    times_adv: torch.Tensor,
    exact_eval: bool = False,
) -> torch.Tensor:
    if grad_mode == "native":
        if name in WF_MODELS:
            feat = _build_feature_torch(name, cfg, signed_adv, times_adv, exact_eval=exact_eval)
            return _extract_logits(model(feat))
        if name == "LASERBEAK":
            feat = _build_laserbeak_feature_torch(signed_adv, times_adv, list(cfg["feature_list"]), int(cfg["seq_len"]))
            return _extract_logits(model(feat))
        if name == "STMWF":
            feat = _build_stmwf_feature_torch(signed_adv, times_adv, out_len=int(cfg.get("seq_len", 10000)))
            return _extract_logits(model(feat))
        feat = _build_feature_torch("DF", MODEL_CONFIG["DF"], signed_adv, times_adv, exact_eval=exact_eval)
        return _extract_logits(model(feat))
    # BPDA proxy (DF on DT-like)
    feat = _build_feature_torch("DF", MODEL_CONFIG["DF"], signed_adv, times_adv, exact_eval=exact_eval)
    return _extract_logits(model(feat))


def _compose_adv_flow(
    times: np.ndarray,
    signed_sizes: np.ndarray,
    delay_vec: np.ndarray,
    dummy_count_vec: np.ndarray,
    dummy_len_max: int = DEFAULT_DUMMY_LEN_MAX,
    delta_ins: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=np.float64)
    s = np.asarray(signed_sizes, dtype=np.float64)
    d = np.maximum(np.asarray(delay_vec, dtype=np.float64), 0.0)
    u_cnt = np.maximum(np.asarray(dummy_count_vec, dtype=np.float64), 0.0)

    n = len(s)
    if n == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int8),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )

    delay_lim = min(n, d.size, 500)
    cnt_lim = min(n, u_cnt.size, 500)

    adv_t: list[float] = []
    adv_s: list[float] = []
    adv_is_dummy: list[int] = []
    inserted_dummy_abs: list[float] = []
    inserted_after_positions: list[int] = []
    for i in range(len(s)):
        if s[i] == 0:
            continue
        delay_i = float(d[i]) if i < delay_lim else 0.0
        t_i = float(t[i] + delay_i)
        adv_t.append(t_i)
        adv_s.append(float(s[i]))
        adv_is_dummy.append(0)
        if i < cnt_lim:
            k = int(np.round(u_cnt[i]))
            if k > 0:
                lens = np.random.randint(0, int(dummy_len_max) + 1, size=k, dtype=np.int32)
                lens = lens[lens > 0]
                for j, dummy_len in enumerate(lens.tolist()):
                    adv_t.append(t_i + delta_ins * float(j + 1))
                    adv_s.append(float(np.sign(s[i]) * float(dummy_len)))
                    adv_is_dummy.append(1)
                    inserted_dummy_abs.append(float(dummy_len))
                    inserted_after_positions.append(int(i))

    if len(adv_t) == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int8),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )

    order = np.argsort(np.asarray(adv_t))
    adv_t_arr = np.asarray(adv_t, dtype=np.float32)[order]
    adv_s_arr = np.asarray(adv_s, dtype=np.float32)[order]
    adv_is_dummy_arr = np.asarray(adv_is_dummy, dtype=np.int8)[order]
    # enforce non-decreasing timestamps
    adv_t_arr = np.maximum.accumulate(adv_t_arr)

    return (
        adv_t_arr,
        adv_s_arr,
        adv_is_dummy_arr,
        np.asarray(inserted_dummy_abs, dtype=np.float32),
        np.asarray(inserted_after_positions, dtype=np.int32),
    )


def _evaluate_with_existing_infer(
    model_name: str,
    npz_path: Path,
    model_path: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[dict[str, float], int]:
    cfg = MODEL_CONFIG[model_name]
    sd = torch.load(model_path, map_location="cpu")

    def infer_class_num() -> int:
        if model_name in {"DF", "TIKTOK"}:
            return int(sd["classifier.9.weight"].shape[0])
        if model_name == "AWF":
            return int(sd["classifier.1.weight"].shape[0])
        if model_name == "BAPM":
            return int(sd["fc.0.weight"].shape[0])
        if model_name == "RF":
            return int(sd["features.20.weight"].shape[0])
        if model_name == "NETCLR":
            return int(sd["fc.weight"].shape[0])
        if model_name == "TF":
            return int(sd["cls_head.weight"].shape[0])
        if model_name == "TMWF":
            return int(sd["fc.weight"].shape[0])
        if model_name == "VARCNN":
            return int(sd["classifier.4.weight"].shape[0])
        if model_name == "FSNET":
            return int(sd["classifier.weight"].shape[0])
        if model_name == "LASERBEAK":
            return int(sd["pred.0.weight"].shape[0])
        if model_name == "STMWF":
            return int(sd["MLP_att.0.weight"].shape[0])
        raise ValueError(f"Unsupported victim model: {model_name}")

    num_classes = infer_class_num()

    # ---- WF models ----
    if model_name in WF_MODELS:
        X, y = load_or_build_features(
            npz_path,
            feature=str(cfg["feature"]),
            seq_len=int(cfg["seq_len"]),
            mtaf_align_len=None,
            cache_path=None,
            force_rebuild=False,
            show_progress=False,
        )
        model = _build_wf_model(model_name, num_classes, cfg)
        model.load_state_dict(sd, strict=True)
        model.to(device).eval()
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
            for bx, by in loader:
                bx = bx.to(device, non_blocking=True)
                logits = model(bx)
                pred = torch.argmax(logits, dim=1).cpu().numpy().astype(int)
                preds.extend(pred.tolist())
                trues.extend(by.numpy().astype(int).tolist())
        y_pred = np.asarray(preds, dtype=np.int64)
        y_true = np.asarray(trues, dtype=np.int64)
        return _macro_metrics(y_true, y_pred), int(y_true.shape[0])

    # ---- FSNET ----
    if model_name == "FSNET":
        data_cfg = FSNetDataConfig(
            max_flow_length=int(cfg.get("max_flow_length", 200)),
            min_length=int(cfg.get("min_length", 2)),
            length_block=int(cfg.get("length_block", 1)),
            max_packet_length=int(cfg.get("max_packet_length", 5000)),
            sample_mode=str(cfg.get("sample_mode", "first")),
        )
        ds = FSNetNpzDataset(npz_path, data_cfg=data_cfg, labels_num=num_classes)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
        length_num = int(cfg.get("max_packet_length", 5000)) // int(cfg.get("length_block", 1)) + 4
        model = FSNetTorch(
            FSNetConfig(
                class_num=num_classes,
                length_num=length_num,
                length_dim=16,
                hidden=int(cfg.get("hidden", 128)),
                layer=int(cfg.get("layer", 2)),
                keep_prob=float(cfg.get("keep_prob", 0.8)),
                rec_loss_weight=float(cfg.get("rec_loss_weight", 0.5)),
                grad_clip_norm=5.0,
            )
        )
        model.load_state_dict(sd, strict=True)
        model.to(device).eval()
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

    # ---- LASERBEAK ----
    if model_name == "LASERBEAK":
        input_size = int(cfg.get("seq_len", 10000))
        ds = LaserbeakNpzDataset(npz_path, cfg=LaserbeakDataConfig(feature_list=tuple(cfg["feature_list"]), input_size=input_size))
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_and_pad,
        )
        model = LaserbeakDFNet(num_classes, input_channels=len(cfg["feature_list"]), input_size=input_size)
        model.load_state_dict(sd, strict=True)
        model.to(device).eval()
        preds: list[int] = []
        trues: list[int] = []
        with torch.no_grad():
            for bx, by, sample_sizes in loader:
                bx = bx.to(device, non_blocking=True).float()
                logits = model(bx, sample_sizes=sample_sizes)
                pred = torch.argmax(logits, dim=1).cpu().numpy().astype(int)
                preds.extend(pred.tolist())
                trues.extend(by.numpy().astype(int).tolist())
        y_pred = np.asarray(preds, dtype=np.int64)
        y_true = np.asarray(trues, dtype=np.int64)
        return _macro_metrics(y_true, y_pred), int(y_true.shape[0])

    # ---- STMWF ----
    if model_name == "STMWF":
        out_len = int(cfg.get("seq_len", 10000))
        ds = STMWFNpzDataset(npz_path, out_len=out_len)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
        model = BertExtractSTMWF(num_hidden=int(cfg.get("num_hidden", 250)), label_size=num_classes)
        model.load_state_dict(sd, strict=True)
        model.to(device).eval()
        preds: list[int] = []
        trues: list[int] = []
        with torch.no_grad():
            for bx, by in loader:
                bx = bx.to(device, non_blocking=True).float()
                logits, _ = model(bx)
                pred = torch.argmax(logits, dim=1).cpu().numpy().astype(int)
                preds.extend(pred.tolist())
                trues.extend(by.numpy().astype(int).tolist())
        y_pred = np.asarray(preds, dtype=np.int64)
        y_true = np.asarray(trues, dtype=np.int64)
        return _macro_metrics(y_true, y_pred), int(y_true.shape[0])

    raise ValueError(f"Unsupported victim model: {model_name}")


def _victim_seq_len(model_name: str) -> int | None:
    cfg = MODEL_CONFIG[model_name]
    if model_name in WF_MODELS:
        return int(cfg.get("seq_len", 0)) or None
    if model_name == "LASERBEAK":
        return int(cfg.get("seq_len", 0)) or None
    if model_name == "STMWF":
        return int(cfg.get("seq_len", 0)) or None
    if model_name == "FSNET":
        return int(cfg.get("seq_len", 0)) or None
    return None


def _crop_object_sequences(arr_obj: np.ndarray, max_len: int | None) -> np.ndarray:
    if max_len is None or max_len <= 0:
        return np.asarray(arr_obj, dtype=object)
    out = []
    for x in arr_obj:
        x_np = np.asarray(x)
        out.append(x_np[:max_len] if x_np.shape[0] > max_len else x_np)
    return np.asarray(out, dtype=object)


def _pad_time_signed_batch(
    times_list: list[np.ndarray],
    signed_list: list[np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = len(times_list)
    lengths = [int(min(len(np.asarray(t)), len(np.asarray(s)))) for t, s in zip(times_list, signed_list)]
    s_max = max(lengths) if lengths else 0
    if s_max <= 0:
        return (
            torch.zeros((bsz, 1), dtype=torch.float32, device=device),
            torch.zeros((bsz, 1), dtype=torch.float32, device=device),
            torch.zeros((bsz,), dtype=torch.long, device=device),
        )
    times_pad = np.zeros((bsz, s_max), dtype=np.float32)
    signed_pad = np.zeros((bsz, s_max), dtype=np.float32)
    for bi, (t_raw, s_raw) in enumerate(zip(times_list, signed_list)):
        t = np.asarray(t_raw, dtype=np.float32)
        s = np.asarray(s_raw, dtype=np.float32)
        m = lengths[bi]
        if m <= 0:
            continue
        t = t[:m]
        s = s[:m]
        times_pad[bi, :m] = t
        signed_pad[bi, :m] = s
        if m < s_max:
            times_pad[bi, m:] = float(t[m - 1])
    return (
        torch.tensor(times_pad, dtype=torch.float32, device=device),
        torch.tensor(signed_pad, dtype=torch.float32, device=device),
        torch.tensor(lengths, dtype=torch.long, device=device),
    )


@dataclass
class AttackConfig:
    steps: int = 20
    alpha_delay: float = 2e-3
    alpha_dummy: float = 8.0
    mu: float = 0.9
    lambda_bw: float = 0.5      ##MODIFIED
    lambda_time: float = 0.5        ##MODIFIED
    overhead_threshold: float = 0.05
    perturb_len: int = 1000     ##MODIFIED
    dummy_len_max: int = DEFAULT_DUMMY_LEN_MAX
    proxy_dummy_len_min: float = 0.1
    proxy_dummy_len_max: float = 1.0
    overhead_escalate_delta: float = 0.2
    overhead_escalate_factor: float = 20.0
    agm_beta: float = 1.0
    agm_temp: float = 1.0
    agm_eps: float = 1e-8
    warmup_steps: int = 3
    delta_ins: float = 1e-6


def _attack_batch(
    times_list: list[np.ndarray],
    signed_list: list[np.ndarray],
    y_list: np.ndarray,
    surrogate_bundles: list[dict[str, Any]],
    device: torch.device,
    atk: AttackConfig,
    *,
    record_agm_weights: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]], np.ndarray | None]:
    times, signed, lengths_t = _pad_time_signed_batch(times_list, signed_list, device)
    y_t = torch.tensor(y_list.astype(np.int64, copy=False), dtype=torch.long, device=device)

    B, S = signed.shape
    # Random initialization (seeded by global torch.manual_seed in main) avoids
    # zero-gradient stagnation around all-zero perturbations while remaining reproducible.
    # Larger init helps the optimizer "start" before projection constraints kick in.
    delay_init = 1e-3 * torch.rand((B, atk.perturb_len), dtype=torch.float32, device=device)
    dummy_count_init = 0.5 * torch.rand((B, atk.perturb_len), dtype=torch.float32, device=device)

    delay = delay_init.clone().detach().requires_grad_(True)
    dummy_count = dummy_count_init.clone().detach().requires_grad_(True)
    m_delay = torch.zeros_like(delay)
    m_dummy_count = torch.zeros_like(dummy_count)
    K = len(surrogate_bundles)
    agm_step_weights = np.zeros((atk.steps, K), dtype=np.float32) if record_agm_weights else None

    col_idx = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
    valid_mask = (col_idx < lengths_t.unsqueeze(1)).to(torch.float32)
    base_bytes = torch.sum(torch.abs(signed) * valid_mask, dim=1).detach()
    last_idx = torch.clamp(lengths_t - 1, min=0)
    last_t = times.gather(1, last_idx.unsqueeze(1)).squeeze(1)
    first_t = times[:, 0]
    base_time = torch.clamp(last_t - first_t, min=1e-6).detach()
    host_len = min(S, atk.perturb_len)
    if host_len > 0:
        host_cols = torch.arange(host_len, device=device).unsqueeze(0).expand(B, host_len)
        host_mask = (host_cols < lengths_t.unsqueeze(1)).to(torch.float32)
        host_mask_full = torch.zeros((B, atk.perturb_len), dtype=torch.float32, device=device)
        host_mask_full[:, :host_len] = host_mask
    else:
        host_mask = torch.zeros((B, 0), dtype=torch.float32, device=device)
        host_mask_full = torch.zeros((B, atk.perturb_len), dtype=torch.float32, device=device)

    def _build_soft_adv_from_vars(
        delay_var: torch.Tensor,
        dummy_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        d_pos_local = torch.relu(delay_var) * host_mask_full
        u_cnt_local = torch.relu(dummy_var) * host_mask_full
        mean_dummy_len_proxy_local = 0.5 * float(atk.proxy_dummy_len_min + atk.proxy_dummy_len_max)
        # STE rounded insertion count to better match discrete insertion behavior.
        u_cnt_ste_local = (torch.round(u_cnt_local) - u_cnt_local).detach() + u_cnt_local
        delay_full_local = torch.zeros((B, S), dtype=torch.float32, device=device)
        if host_len > 0:
            delay_full_local[:, :host_len] = d_pos_local[:, :host_len]
        orig_times_local = times + delay_full_local  # (B,S)
        dummy_signed_local = (
            torch.sign(signed[:, :host_len]) * (u_cnt_ste_local[:, :host_len] * mean_dummy_len_proxy_local)
        )  # (B,host_len)
        dummy_times_local = orig_times_local[:, :host_len] + atk.delta_ins
        times_mix_local = torch.cat([orig_times_local, dummy_times_local], dim=1)  # (B,S+K)
        signed_mix_local = torch.cat([signed, dummy_signed_local], dim=1)  # (B,S+K)
        order_local = torch.argsort(times_mix_local, dim=1)
        times_adv_local = torch.gather(times_mix_local, 1, order_local)
        signed_adv_local = torch.gather(signed_mix_local, 1, order_local)
        return times_adv_local, signed_adv_local, u_cnt_ste_local, d_pos_local

    for step in range(atk.steps):
        mean_dummy_len_real = 0.5 * float(atk.dummy_len_max)

        per_model_gd: list[torch.Tensor] = []
        per_model_gu: list[torch.Tensor] = []
        for b in surrogate_bundles:
            if delay.grad is not None:
                delay.grad.zero_()
            if dummy_count.grad is not None:
                dummy_count.grad.zero_()

            times_adv, signed_adv, u_cnt_ste, d_pos = _build_soft_adv_from_vars(delay, dummy_count)
            logits_i = _forward_surrogate_logits(
                name=b["name"],
                model=b["model"],
                grad_mode=b["grad_mode"],
                cfg=b["cfg"],
                signed_adv=signed_adv,
                times_adv=times_adv,
                exact_eval=False,
            )

            true_logit_i = logits_i.gather(1, y_t.view(-1, 1)).squeeze(1)
            other_i = logits_i.clone()
            other_i.scatter_(1, y_t.view(-1, 1), float("-inf"))
            other_max_i = torch.max(other_i, dim=1).values
            margin_i = other_max_i - true_logit_i

            if host_len > 0:
                bw_num_i = torch.sum(u_cnt_ste[:, :host_len], dim=1) * mean_dummy_len_real
                time_num_i = torch.sum(d_pos[:, :host_len], dim=1)
            else:
                bw_num_i = torch.zeros((B,), dtype=delay.dtype, device=device)
                time_num_i = torch.zeros((B,), dtype=delay.dtype, device=device)
            bw_cost_i = bw_num_i / (base_bytes + 1e-8)
            time_cost_i = time_num_i / (base_time + 1e-8)
            if step < atk.warmup_steps:
                attack_obj_i = margin_i.mean()
            else:
                bw_excess_i = torch.relu(bw_cost_i - atk.overhead_threshold)
                time_excess_i = torch.relu(time_cost_i - atk.overhead_threshold)
                bw_scale_i = torch.where(
                    bw_excess_i > atk.overhead_escalate_delta,
                    torch.as_tensor(atk.overhead_escalate_factor, dtype=bw_excess_i.dtype, device=bw_excess_i.device),
                    torch.as_tensor(1.0, dtype=bw_excess_i.dtype, device=bw_excess_i.device),
                )
                time_scale_i = torch.where(
                    time_excess_i > atk.overhead_escalate_delta,
                    torch.as_tensor(atk.overhead_escalate_factor, dtype=time_excess_i.dtype, device=time_excess_i.device),
                    torch.as_tensor(1.0, dtype=time_excess_i.dtype, device=time_excess_i.device),
                )
                attack_obj_i = (
                    margin_i.mean()
                    - torch.mean(atk.lambda_bw * bw_scale_i * bw_excess_i)
                    - torch.mean(atk.lambda_time * time_scale_i * time_excess_i)
                )

            attack_obj_i.backward()
            g_d_i = delay.grad.detach() if delay.grad is not None else torch.zeros_like(delay)
            g_u_i = dummy_count.grad.detach() if dummy_count.grad is not None else torch.zeros_like(dummy_count)
            g_d_i = g_d_i / (g_d_i.abs().mean(dim=1, keepdim=True) + 1e-8)
            g_u_i = g_u_i / (g_u_i.abs().mean(dim=1, keepdim=True) + 1e-8)
            per_model_gd.append(g_d_i)
            per_model_gu.append(g_u_i)

        if K == 1:
            weights = torch.ones((1,), dtype=delay.dtype, device=device)
        else:
            # AGM weighting from cross-model transferability of single-step probe perturbations.
            s_mat = torch.zeros((K, K), dtype=delay.dtype, device=device)  # row k, col i -> s_{k,i}
            with torch.no_grad():
                for i, g_i in enumerate(per_model_gd):
                    gu_i = per_model_gu[i]
                    delay_probe = torch.clamp(delay + atk.alpha_delay * g_i.sign(), min=0.0)
                    dummy_probe = torch.clamp(dummy_count + atk.alpha_dummy * gu_i.sign(), min=0.0)
                    t_probe, s_probe, _, _ = _build_soft_adv_from_vars(delay_probe, dummy_probe)
                    for k, b_k in enumerate(surrogate_bundles):
                        logits_ki = _forward_surrogate_logits(
                            name=b_k["name"],
                            model=b_k["model"],
                            grad_mode=b_k["grad_mode"],
                            cfg=b_k["cfg"],
                            signed_adv=s_probe,
                            times_adv=t_probe,
                            exact_eval=False,
                        )
                        s_mat[k, i] = F.cross_entropy(logits_ki, y_t)

                rho = torch.zeros((K,), dtype=delay.dtype, device=device)
                for i in range(K):
                    ratios = []
                    for k in range(K):
                        if k == i:
                            continue
                        ratios.append(s_mat[k, i] / (s_mat[k, k] + atk.agm_eps))
                    if ratios:
                        rho[i] = atk.agm_beta * torch.stack(ratios).mean()
                weights = torch.softmax(rho / max(atk.agm_temp, 1e-8), dim=0)
        if agm_step_weights is not None:
            agm_step_weights[step, :] = weights.detach().cpu().numpy().astype(np.float32)

        g_d = torch.zeros_like(delay)
        g_u_cnt = torch.zeros_like(dummy_count)
        for i in range(K):
            g_d = g_d + weights[i] * per_model_gd[i]
            g_u_cnt = g_u_cnt + weights[i] * per_model_gu[i]

        m_delay = atk.mu * m_delay + g_d
        m_dummy_count = atk.mu * m_dummy_count + g_u_cnt

        with torch.no_grad():
            # MI-FGSM style update with sign(momentum).
            delay += atk.alpha_delay * m_delay.sign()
            dummy_count += atk.alpha_dummy * m_dummy_count.sign()

            delay.clamp_(min=0.0)
            dummy_count.clamp_(min=0.0)

    delay_np_all = torch.relu(delay).detach().cpu().numpy()
    dummy_count_np_all = torch.relu(dummy_count).detach().cpu().numpy()
    adv_times_list: list[np.ndarray] = []
    adv_signed_list: list[np.ndarray] = []
    adv_is_dummy_list: list[np.ndarray] = []
    inserted_abs_list: list[np.ndarray] = []
    inserted_pos_list: list[np.ndarray] = []
    stats_list: list[dict[str, Any]] = []

    for bi in range(B):
        t_np = np.asarray(times_list[bi], dtype=np.float32)
        s_np = np.asarray(signed_list[bi], dtype=np.float32)
        delay_np = delay_np_all[bi]
        dummy_count_np = dummy_count_np_all[bi]
        adv_t, adv_s, adv_is_dummy, inserted_dummy_abs, inserted_after_positions = _compose_adv_flow(
            t_np,
            s_np,
            delay_np,
            dummy_count_np,
            dummy_len_max=atk.dummy_len_max,
            delta_ins=atk.delta_ins,
        )
        adv_times_list.append(adv_t)
        adv_signed_list.append(adv_s)
        adv_is_dummy_list.append(adv_is_dummy)
        inserted_abs_list.append(inserted_dummy_abs)
        inserted_pos_list.append(inserted_after_positions)
        valid_len = min(len(t_np), atk.perturb_len)
        delay_eff = np.maximum(delay_np[:valid_len], 0.0)
        base_t = max(1e-8, float(t_np[-1] - t_np[0])) if t_np.shape[0] > 1 else 1e-8
        stats_list.append(
            {
                "bw_overhead": float(np.sum(inserted_dummy_abs) / (np.sum(np.abs(s_np)) + 1e-8)),
                "time_overhead": float(np.sum(delay_eff) / base_t),
                "insert_count_selected": float(np.asarray(inserted_dummy_abs).shape[0]),
                "orig_len": int(s_np.shape[0]),
                "orig_nonzero_len": int(np.count_nonzero(s_np != 0)),
                "adv_len": int(np.asarray(adv_s).shape[0]),
                "dummy_inserted_lengths": np.asarray(inserted_dummy_abs, dtype=np.float32),
                "inserted_after_positions": inserted_after_positions,
                "adv_is_dummy": adv_is_dummy,
                "adv_signed_for_plot": adv_s,
                "inserted_dummy_abs": inserted_dummy_abs,
            }
        )

    # Re-evaluate surrogate on discretized final adversarial samples in batch.
    with torch.no_grad():
        t_eval, s_eval, _ = _pad_time_signed_batch(adv_times_list, adv_signed_list, device)
        logits_eval_list = [
            _forward_surrogate_logits(
                name=b["name"],
                model=b["model"],
                grad_mode=b["grad_mode"],
                cfg=b["cfg"],
                signed_adv=s_eval,
                times_adv=t_eval,
                exact_eval=True,
            )
            for b in surrogate_bundles
        ]
        logits_eval = torch.stack(logits_eval_list, dim=0).mean(dim=0)
        probs_eval = torch.softmax(logits_eval, dim=1)
        pred_eval = torch.argmax(logits_eval, dim=1)
        sur_conf_eval = torch.max(probs_eval, dim=1).values.detach().cpu().numpy()
        sur_true_prob_eval = probs_eval.gather(1, y_t.view(-1, 1)).squeeze(1).detach().cpu().numpy()
        sur_success_eval = (pred_eval != y_t).float().detach().cpu().numpy()

    for bi in range(B):
        stats_list[bi]["surrogate_confidence"] = float(sur_conf_eval[bi])
        stats_list[bi]["surrogate_true_prob"] = float(sur_true_prob_eval[bi])
        stats_list[bi]["surrogate_attack_success"] = float(sur_success_eval[bi])

    return adv_times_list, adv_signed_list, stats_list, agm_step_weights


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Forge adversarial traffic generation (MI-FGSM + BPDA)")
    p.add_argument("--input-b", type=Path, default=root / "Dataset" / "DF" / "raw-data-50-1000-B.npz")
    p.add_argument("--surrogate", type=str, required=True, help="one or more model names, comma-separated")
    p.add_argument("--surrogate-model-path", type=Path, default=None)
    p.add_argument("--victims", type=str, required=True, help="comma-separated model names from implemented 12 models")
    p.add_argument("--attack-batch-size", type=int, default=16, help="batch size for adversarial optimization")
    p.add_argument("--enable-packet-figure", action="store_true", default=False, help="save per-label packet distribution figure")
    p.add_argument("--enable-agm-weight-figure", action="store_true", default=False, help="save per-label AGM weight stacked bars")
    p.add_argument("--batch-size-eval", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--label-start", type=int, default=0)
    p.add_argument("--label-end", type=int, default=94)
    p.add_argument("--save-adv-path", type=Path, default=root / "Dataset" / "DF" / "raw-data-50-1000-B-adv.npz")
    p.add_argument("--save-report-path", type=Path, default=root / "Logs" / "attack_report_apply_forge.json")
    p.add_argument("--seed", type=int, default=2024)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(__file__).resolve().parent
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    data = np.load(args.input_b, allow_pickle=True)
    labels = data["labels"].astype(np.int64, copy=False)
    times_all = data["times"]
    signed_all = data["signed_sizes"]
    class_num = int(np.max(labels) + 1)

    surrogate_names = [s.strip().upper() for s in str(args.surrogate).split(",") if s.strip()]
    if not surrogate_names:
        raise SystemExit("empty --surrogate list")
    bad_sur = [s for s in surrogate_names if s not in MODEL_CONFIG]
    if bad_sur:
        raise SystemExit(f"invalid surrogate models: {bad_sur}")
    if args.surrogate_model_path is not None and len(surrogate_names) > 1:
        raise SystemExit("--surrogate-model-path only supports single surrogate; remove it for multi-surrogate mode")

    surrogate_bundles: list[dict[str, Any]] = []
    for sname in surrogate_names:
        scfg = MODEL_CONFIG[sname]
        spath = args.surrogate_model_path if args.surrogate_model_path is not None else _model_default_path(root, sname)
        if not spath.is_file():
            raise SystemExit(f"surrogate model path not found: {spath}")
        smodel, sgrad_mode = _build_surrogate_for_grad(sname, spath, class_num, scfg, device)
        surrogate_bundles.append({"name": sname, "cfg": scfg, "model": smodel, "grad_mode": sgrad_mode})

    atk = AttackConfig()

    victims = [m.strip().upper() for m in args.victims.split(",") if m.strip()]
    bad = [m for m in victims if m not in MODEL_CONFIG]
    if bad:
        raise SystemExit(f"invalid victims: {bad}")

    adv_times = np.asarray(times_all, dtype=object).copy()
    adv_signed = np.asarray(signed_all, dtype=object).copy()

    class_reports: list[dict[str, Any]] = []
    selected_labels = list(range(args.label_start, args.label_end + 1))
    for c in selected_labels:
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        stats_bw = []
        stats_time = []
        stats_conf = []
        stats_true_prob = []
        stats_succ = []
        first_case_adv_signed: np.ndarray | None = None
        first_case_adv_is_dummy: np.ndarray | None = None
        agm_weights_sum: np.ndarray | None = None
        agm_weight_batches = 0
        attack_bs = max(1, int(args.attack_batch_size))
        sample_bar = tqdm(total=int(idx.size), desc=f"Attack label={c}", unit="sample", leave=False)
        for start in range(0, int(idx.size), attack_bs):
            sub = idx[start : start + attack_bs]
            times_batch = [np.asarray(times_all[i], dtype=np.float32) for i in sub.tolist()]
            signed_batch = [np.asarray(signed_all[i], dtype=np.float32) for i in sub.tolist()]
            y_batch = labels[sub].astype(np.int64, copy=False)
            adv_t_list, adv_s_list, st_list, agm_step_weights = _attack_batch(
                times_batch,
                signed_batch,
                y_batch,
                surrogate_bundles,
                device,
                atk,
                record_agm_weights=bool(args.enable_agm_weight_figure),
            )
            if args.enable_agm_weight_figure and agm_step_weights is not None:
                if agm_weights_sum is None:
                    agm_weights_sum = agm_step_weights.astype(np.float64, copy=False)
                else:
                    agm_weights_sum = agm_weights_sum + agm_step_weights
                agm_weight_batches += 1
            for local_j, i in enumerate(sub.tolist()):
                st = st_list[local_j]
                adv_times[i] = adv_t_list[local_j]
                adv_signed[i] = adv_s_list[local_j]
                stats_bw.append(st["bw_overhead"])
                stats_time.append(st["time_overhead"])
                stats_conf.append(float(st.get("surrogate_confidence", 0.0)))
                stats_true_prob.append(float(st.get("surrogate_true_prob", 0.0)))
                stats_succ.append(float(st.get("surrogate_attack_success", 0.0)))
                if first_case_adv_signed is None:
                    first_case_adv_signed = np.asarray(st.get("adv_signed_for_plot", np.zeros((0,), dtype=np.float32)))
                    first_case_adv_is_dummy = np.asarray(st.get("adv_is_dummy", np.zeros((0,), dtype=np.int8)))
            sample_bar.update(len(sub))
            if stats_bw and stats_time:
                sample_bar.set_postfix(
                    bw=f"{float(np.mean(stats_bw)):.4f}",
                    time=f"{float(np.mean(stats_time)):.4f}",
                    conf=f"{float(np.mean(stats_conf)):.4f}",
                    p_true=f"{float(np.mean(stats_true_prob)):.4f}",
                    succ=f"{float(np.mean(stats_succ)):.4f}",
                )
        sample_bar.close()

        # per-class evaluate with victims
        with tempfile.TemporaryDirectory() as td:
            victim_metrics: dict[str, Any] = {}
            for vm in victims:
                vm_path = _model_default_path(root, vm)
                if not vm_path.is_file():
                    victim_metrics[vm] = {"error": f"missing model {vm_path}"}
                    continue
                vlen = _victim_seq_len(vm)
                clean_vm_path = Path(td) / f"class_{c}_clean_{vm}.npz"
                adv_vm_path = Path(td) / f"class_{c}_adv_{vm}.npz"
                np.savez_compressed(
                    clean_vm_path,
                    labels=labels[idx],
                    times=_crop_object_sequences(np.asarray(times_all[idx], dtype=object), vlen),
                    signed_sizes=_crop_object_sequences(np.asarray(signed_all[idx], dtype=object), vlen),
                )
                np.savez_compressed(
                    adv_vm_path,
                    labels=labels[idx],
                    times=_crop_object_sequences(np.asarray(adv_times[idx], dtype=object), vlen),
                    signed_sizes=_crop_object_sequences(np.asarray(adv_signed[idx], dtype=object), vlen),
                )
                clean_m, _ = _evaluate_with_existing_infer(vm, clean_vm_path, vm_path, args.batch_size_eval, args.num_workers, device)
                adv_m, _ = _evaluate_with_existing_infer(vm, adv_vm_path, vm_path, args.batch_size_eval, args.num_workers, device)
                victim_metrics[vm] = {
                    "clean": clean_m,
                    "adv": adv_m,
                    "acc_drop": round(clean_m["Accuracy"] - adv_m["Accuracy"], 4),
                }

        class_reports.append(
            {
                "label": int(c),
                "n_samples": int(idx.size),
                "avg_bw_overhead": float(np.mean(stats_bw)) if stats_bw else 0.0,
                "avg_time_overhead": float(np.mean(stats_time)) if stats_time else 0.0,
                "victims": victim_metrics,
            }
        )
        # print per-label summary immediately after victim evaluation
        print(
            f"[label {c}] done: n={int(idx.size)}, "
            f"avg_bw_overhead={float(np.mean(stats_bw)) if stats_bw else 0.0:.4f}, "
            f"avg_time_overhead={float(np.mean(stats_time)) if stats_time else 0.0:.4f}"
        )
        for vm, vm_res in victim_metrics.items():
            if "error" in vm_res:
                print(f"  - victim={vm}: error={vm_res['error']}")
                continue
            clean_acc = vm_res["clean"]["Accuracy"]
            adv_acc = vm_res["adv"]["Accuracy"]
            acc_drop = vm_res["acc_drop"]
            print(f"  - victim={vm}: clean_acc={clean_acc:.4f}, adv_acc={adv_acc:.4f}, acc_drop={acc_drop:.4f}")

        if args.enable_agm_weight_figure and agm_weights_sum is not None and agm_weight_batches > 0:
            mean_agm_weights = (agm_weights_sum / float(agm_weight_batches)).astype(np.float32, copy=False)
            save_label_agm_weight_figure(
                root=root,
                label=int(c),
                surrogate_names=[b["name"] for b in surrogate_bundles],
                step_weights=mean_agm_weights,
            )

        # Save one illustrative distribution chart per label using the first sample.
        if args.enable_packet_figure and idx.size > 0:
            save_label_packet_distribution_figure(
                root=root,
                label=int(c),
                adv_signed=(
                    first_case_adv_signed
                    if first_case_adv_signed is not None
                    else np.zeros((0,), dtype=np.float32)
                ),
                adv_is_dummy=(
                    first_case_adv_is_dummy
                    if first_case_adv_is_dummy is not None
                    else np.zeros((0,), dtype=np.int8)
                ),
            )

    # Save all adv samples
    print(f"[final] saving adversarial samples -> {args.save_adv_path} (this may take time)")
    args.save_adv_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.save_adv_path,
        labels=labels,
        times=np.asarray(adv_times, dtype=object),
        signed_sizes=np.asarray(adv_signed, dtype=object),
    )
    print(f"[final] adversarial samples saved")

    # Aggregate report
    serializable_attack_cfg = dict(vars(atk))

    agg: dict[str, Any] = {
        "surrogate": surrogate_names if len(surrogate_names) > 1 else surrogate_names[0],
        "surrogate_grad_mode": {b["name"]: b["grad_mode"] for b in surrogate_bundles},
        "victims": victims,
        "label_range": [args.label_start, args.label_end],
        "attack_cfg": serializable_attack_cfg,
        "class_reports": class_reports,
    }
    if class_reports:
        agg["avg_bw_overhead"] = float(np.mean([r["avg_bw_overhead"] for r in class_reports]))
        agg["avg_time_overhead"] = float(np.mean([r["avg_time_overhead"] for r in class_reports]))
        surrogate_set = {s.upper() for s in surrogate_names}
        agg["global_victim_stats_all"] = _aggregate_victim_metrics(class_reports, exclude_victims=set())
        agg["global_victim_stats_exclude_surrogate"] = _aggregate_victim_metrics(
            class_reports, exclude_victims=surrogate_set
        )

    first_sur = (surrogate_names[0] if surrogate_names else "NA").lower()
    sur_cnt = len(surrogate_names) if surrogate_names else 0
    first_vic = (victims[0] if victims else "NA").lower()
    vic_cnt = len(victims)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = root / "Logs" / f"attack_report_{first_sur}_s{sur_cnt}_{first_vic}_v{vic_cnt}_{ts}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[final] saving report -> {report_path}")
    report_path.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[final] report saved")

    print(f"saved adv npz -> {args.save_adv_path}")
    print(f"saved report -> {report_path}")
    print(f"surrogate_grad_mode={ {b['name']: b['grad_mode'] for b in surrogate_bundles} }")


if __name__ == "__main__":
    main()

