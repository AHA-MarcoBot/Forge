from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def save_label_packet_distribution_figure(
    root: Path,
    label: int,
    adv_signed: np.ndarray,
    adv_is_dummy: np.ndarray,
) -> Path:
    import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]

    figs_dir = root / "Figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    out_path = figs_dir / f"label_{label}_case0_packet_distribution.png"

    signed = np.asarray(adv_signed, dtype=np.float32)
    is_dummy = np.asarray(adv_is_dummy, dtype=np.int8)
    x = np.arange(signed.shape[0], dtype=np.int32)
    y = np.abs(signed)
    orig_mask = is_dummy == 0
    dummy_mask = is_dummy == 1

    plt.figure(figsize=(12, 4.8))
    if np.any(orig_mask):
        plt.bar(x[orig_mask], y[orig_mask], width=1.0, color="#1f77b4", label="original packets")
    if np.any(dummy_mask):
        plt.bar(x[dummy_mask], y[dummy_mask], width=1.0, color="#ff7f0e", label="inserted dummy packets")
    plt.xlabel("Packet index in total sequence")
    plt.ylabel("Packet size magnitude")
    plt.title(f"Label {label} injected-vs-original packet sequence (case-0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def save_label_agm_weight_figure(
    root: Path,
    label: int,
    surrogate_names: Sequence[str],
    step_weights: np.ndarray,
) -> Path:
    import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]

    weights = np.asarray(step_weights, dtype=np.float32)
    if weights.ndim != 2:
        raise ValueError(f"step_weights must be 2D, got shape={weights.shape}")
    steps, k = weights.shape
    if steps == 0 or k == 0:
        raise ValueError("step_weights is empty")

    names = [str(s).upper() for s in surrogate_names]
    if len(names) != k:
        names = [f"S{i}" for i in range(k)]

    row_sum = np.sum(weights, axis=1, keepdims=True)
    row_sum = np.maximum(row_sum, 1e-8)
    norm_w = weights / row_sum

    figs_dir = root / "Figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    out_path = figs_dir / f"label_{label}_agm_weights.png"

    x = np.arange(steps, dtype=np.int32)
    bottom = np.zeros((steps,), dtype=np.float32)
    cmap = plt.get_cmap("tab20")

    plt.figure(figsize=(max(10, steps * 0.35), 4.8))
    for i in range(k):
        color = cmap(i % 20)
        plt.bar(
            x,
            norm_w[:, i],
            bottom=bottom,
            width=0.85,
            color=color,
            edgecolor="none",
            label=names[i],
        )
        bottom = bottom + norm_w[:, i]

    plt.ylim(0.0, 1.0)
    plt.xlabel("Attack optimization step")
    plt.ylabel("Surrogate weight proportion")
    plt.title(f"Label {label} AGM surrogate weights by step")
    plt.legend(ncol=min(4, k), fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

