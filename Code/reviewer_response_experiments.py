#!/usr/bin/env python3
"""
Reviewer-response experiments for the Psi-HDL / Psi-NN manuscript.

This script generates *paper-ready artifacts* addressing common reviewer asks:
  - Clustering threshold sensitivity (robustness vs. distance threshold)
  - Reduced-precision / fixed-point quantization sensitivity (bit-width sweeps)
  - tanh approximation error (ideal tanh vs. lightweight approximations)
  - Training curves (loss vs. iteration) for the memristor PINN example

Outputs default to: Code/output/reviewer_response/
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    import torch
    import torch.nn as nn
except Exception as e:  # pragma: no cover
    raise RuntimeError("This script requires PyTorch to run.") from e


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_DIR / "output" / "reviewer_response"
DEFAULT_MEMRISTOR_CSV = REPO_DIR / "output" / "memristor" / "memristor_training_data.csv"
DEFAULT_MEMRISTOR_MODEL = REPO_DIR / "output" / "memristor" / "memristor_pinn.pth"


# -----------------------------------------------------------------------------
# Models (kept local to avoid import side effects)
# -----------------------------------------------------------------------------


class MemristorPINN(nn.Module):
    def __init__(self, hidden_dims: List[int] | Tuple[int, ...] = (2, 40, 40, 40, 2)):
        super().__init__()
        dims = list(hidden_dims)
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, V: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat([V, x], dim=1)
        outputs = self.network(inputs)
        I = outputs[:, 0:1]
        x_new = outputs[:, 1:2]
        return I, x_new


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path: Path, header: List[str], rows: Iterable[Iterable]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(list(row))


@dataclass(frozen=True)
class Metrics:
    mae: float
    rmse: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return Metrics(mae=mae, rmse=rmse)


def load_memristor_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Memristor CSV not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        V: List[float] = []
        I: List[float] = []
        x: List[float] = []
        for row in reader:
            V.append(float(row["Voltage_V"]))
            I.append(float(row["Current_A"]))
            x.append(float(row["State_x"]))
    V_arr = np.array(V, dtype=np.float32).reshape(-1, 1)
    I_arr = np.array(I, dtype=np.float32).reshape(-1, 1)
    x_arr = np.array(x, dtype=np.float32).reshape(-1, 1)
    return V_arr, I_arr, x_arr


def train_test_split(
    V: np.ndarray, I: np.ndarray, x: np.ndarray, test_ratio: float, seed: int
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be in (0, 1)")
    n = len(V)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_test = int(round(n * test_ratio))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return (V[train_idx], I[train_idx], x[train_idx]), (V[test_idx], I[test_idx], x[test_idx])


def load_memristor_model(model_path: Path, device: str) -> nn.Module:
    """
    Tries to load either:
      - a full torch.nn.Module saved via torch.save(model)
      - a state_dict saved via torch.save(model.state_dict())
      - a dict with a "state_dict" key
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Memristor model not found: {model_path}")

    obj = torch.load(model_path, map_location=device)

    if isinstance(obj, nn.Module):
        model = obj
    else:
        model = MemristorPINN()
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state = obj["state_dict"]
        elif isinstance(obj, dict):
            state = obj
        else:
            raise TypeError(f"Unsupported torch.load payload type: {type(obj)}")
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


def predict_memristor_current(model: nn.Module, V: np.ndarray, x: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        V_t = torch.tensor(V, dtype=torch.float32, device=device)
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        I_pred, _ = model(V_t, x_t)
        return I_pred.detach().cpu().numpy()


# -----------------------------------------------------------------------------
# Threshold clustering / compression sensitivity
# -----------------------------------------------------------------------------


def _cluster_centers_by_threshold(sorted_vals: np.ndarray, threshold: float) -> List[Tuple[int, int, float]]:
    """
    Returns segments as (start_idx_in_sorted, end_idx_exclusive, center_value).
    """
    if threshold <= 0:
        raise ValueError("threshold must be > 0")
    if sorted_vals.size == 0:
        return []

    diffs = np.diff(sorted_vals)
    split_points = np.where(diffs > threshold)[0]

    segments: List[Tuple[int, int]] = []
    start = 0
    for sp in split_points:
        end = int(sp) + 1
        segments.append((start, end))
        start = end
    segments.append((start, int(sorted_vals.size)))

    out: List[Tuple[int, int, float]] = []
    for s, e in segments:
        center = float(np.mean(sorted_vals[s:e]))
        out.append((s, e, center))
    return out


def quantize_array_by_threshold(values: np.ndarray, threshold: float) -> Tuple[np.ndarray, int]:
    """
    A deterministic, threshold-based 1D clustering quantizer:
      - sort values
      - split where adjacent diffs exceed threshold
      - replace each value by its segment mean

    Returns: (quantized_values, num_clusters)
    """
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return values, 0

    order = np.argsort(flat)
    sorted_vals = flat[order]

    segments = _cluster_centers_by_threshold(sorted_vals, threshold=threshold)
    centers = np.array([c for _, _, c in segments], dtype=np.float64)

    # Build bin edges as midpoints between adjacent segment centers in the sorted domain
    # (assignment is based on value, not original index).
    seg_max_vals = np.array([sorted_vals[e - 1] for _, e, _ in segments], dtype=np.float64)
    edges = (seg_max_vals[:-1] + seg_max_vals[1:]) / 2.0

    # Assign each original value to a segment by value.
    seg_idx = np.searchsorted(edges, flat, side="right")
    quantized = centers[seg_idx]

    return quantized.reshape(values.shape).astype(values.dtype, copy=False), int(len(centers))


def apply_threshold_quantization_to_model(model: nn.Module, threshold: float) -> Tuple[nn.Module, Dict[str, Dict[str, float]]]:
    model_q = copy.deepcopy(model)
    per_tensor: Dict[str, Dict[str, float]] = {}

    total_params = 0
    total_clusters = 0

    with torch.no_grad():
        for name, param in model_q.named_parameters():
            arr = param.detach().cpu().numpy()
            q_arr, n_clusters = quantize_array_by_threshold(arr, threshold=threshold)
            param.copy_(torch.tensor(q_arr, dtype=param.dtype, device=param.device))

            n_params = int(arr.size)
            total_params += n_params
            total_clusters += n_clusters
            per_tensor[name] = {
                "params": float(n_params),
                "clusters": float(n_clusters),
                "compression_pct": float(100.0 * (1.0 - (n_clusters / max(1, n_params)))),
            }

    per_tensor["_overall"] = {
        "params": float(total_params),
        "clusters": float(total_clusters),
        "compression_pct": float(100.0 * (1.0 - (total_clusters / max(1, total_params)))),
    }
    return model_q, per_tensor


def run_threshold_sweep(
    output_dir: Path,
    model_path: Path,
    data_csv: Path,
    thresholds: List[float],
    test_ratio: float,
    seed: int,
    device: str,
) -> None:
    out_dir = ensure_dir(output_dir / "threshold_sweep")

    V, I, x = load_memristor_csv(data_csv)
    (V_tr, I_tr, x_tr), (V_te, I_te, x_te) = train_test_split(V, I, x, test_ratio=test_ratio, seed=seed)

    base_model = load_memristor_model(model_path, device=device)
    base_pred = predict_memristor_current(base_model, V_te, x_te, device=device)
    base_metrics = compute_metrics(I_te, base_pred)

    rows = []
    summary = {
        "model_path": str(model_path),
        "data_csv": str(data_csv),
        "test_ratio": test_ratio,
        "seed": seed,
        "device": device,
        "baseline": {"mae": base_metrics.mae, "rmse": base_metrics.rmse},
        "thresholds": [],
    }

    for th in thresholds:
        model_q, clustering_info = apply_threshold_quantization_to_model(base_model, threshold=th)
        pred = predict_memristor_current(model_q, V_te, x_te, device=device)
        m = compute_metrics(I_te, pred)
        overall = clustering_info["_overall"]
        rows.append([th, overall["clusters"], overall["compression_pct"], m.mae, m.rmse])
        summary["thresholds"].append(
            {
                "threshold": th,
                "overall": overall,
                "test": {"mae": m.mae, "rmse": m.rmse},
            }
        )

    write_csv(
        out_dir / "threshold_sweep_results.csv",
        ["threshold", "clusters_total", "compression_pct", "test_mae", "test_rmse"],
        rows,
    )
    save_json(out_dir / "threshold_sweep_results.json", summary)

    if plt is not None:
        ths = [r[0] for r in rows]
        comp = [r[2] for r in rows]
        maes = [r[3] for r in rows]
        rmses = [r[4] for r in rows]

        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(ths, comp, marker="o", label="Compression (%)")
        ax1.set_xlabel("Clustering threshold")
        ax1.set_ylabel("Compression (%)")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(ths, maes, marker="s", color="tab:red", label="Test MAE")
        ax2.set_ylabel("Test MAE")

        lines = ax1.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "threshold_sweep_plot.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ths, rmses, marker="o")
        ax.set_xlabel("Clustering threshold")
        ax.set_ylabel("Test RMSE")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "threshold_sweep_rmse.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


# -----------------------------------------------------------------------------
# Reduced precision / fixed-point sensitivity
# -----------------------------------------------------------------------------


def quantize_tensor_uniform(t: torch.Tensor, bits: int, symmetric: bool = True) -> torch.Tensor:
    if bits < 2:
        raise ValueError("bits must be >= 2")
    if t.numel() == 0:
        return t

    if symmetric:
        qmax = (2 ** (bits - 1)) - 1
        max_abs = torch.max(torch.abs(t)).clamp(min=1e-12)
        scale = max_abs / qmax
        q = torch.clamp(torch.round(t / scale), -qmax, qmax)
        return q * scale

    qmin = 0
    qmax = (2**bits) - 1
    t_min = torch.min(t)
    t_max = torch.max(t)
    if (t_max - t_min).abs() < 1e-12:
        return t.clone()
    scale = (t_max - t_min) / float(qmax - qmin)
    zero_point = torch.round(qmin - t_min / scale)
    q = torch.clamp(torch.round(t / scale + zero_point), qmin, qmax)
    return (q - zero_point) * scale


def apply_uniform_quantization_to_model(model: nn.Module, bits: int, symmetric: bool = True) -> nn.Module:
    model_q = copy.deepcopy(model)
    with torch.no_grad():
        for _, param in model_q.named_parameters():
            param.copy_(quantize_tensor_uniform(param, bits=bits, symmetric=symmetric))
    return model_q


def run_bitwidth_sweep(
    output_dir: Path,
    model_path: Path,
    data_csv: Path,
    bits_list: List[int],
    test_ratio: float,
    seed: int,
    symmetric: bool,
    device: str,
) -> None:
    out_dir = ensure_dir(output_dir / "bitwidth_sweep")

    V, I, x = load_memristor_csv(data_csv)
    (_, _, _), (V_te, I_te, x_te) = train_test_split(V, I, x, test_ratio=test_ratio, seed=seed)

    base_model = load_memristor_model(model_path, device=device)
    base_pred = predict_memristor_current(base_model, V_te, x_te, device=device)
    base_metrics = compute_metrics(I_te, base_pred)

    rows = []
    summary = {
        "model_path": str(model_path),
        "data_csv": str(data_csv),
        "test_ratio": test_ratio,
        "seed": seed,
        "device": device,
        "baseline": {"mae": base_metrics.mae, "rmse": base_metrics.rmse},
        "bits": [],
        "quantization": {"scheme": "uniform", "symmetric": symmetric},
    }

    for b in bits_list:
        model_q = apply_uniform_quantization_to_model(base_model, bits=b, symmetric=symmetric)
        pred = predict_memristor_current(model_q, V_te, x_te, device=device)
        m = compute_metrics(I_te, pred)
        rows.append([b, m.mae, m.rmse])
        summary["bits"].append({"bits": b, "test": {"mae": m.mae, "rmse": m.rmse}})

    write_csv(out_dir / "bitwidth_sweep_results.csv", ["bits", "test_mae", "test_rmse"], rows)
    save_json(out_dir / "bitwidth_sweep_results.json", summary)

    if plt is not None:
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("Quantization bit-width (weights/biases)")
        ax.set_ylabel("Test MAE")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "bitwidth_sweep_plot.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


# -----------------------------------------------------------------------------
# tanh approximation error
# -----------------------------------------------------------------------------


def tanh_rational(x: np.ndarray) -> np.ndarray:
    """
    A common tanh rational approximation:
      tanh(x) ≈ x * (27 + x^2) / (27 + 9 x^2)
    """
    x2 = x * x
    return x * (27.0 + x2) / (27.0 + 9.0 * x2)


def tanh_pwl(x: np.ndarray, x_max: float = 3.0, n_segments: int = 8) -> np.ndarray:
    """
    Piecewise-linear approximation of tanh on [-x_max, x_max], saturated outside.
    """
    if n_segments < 2:
        raise ValueError("n_segments must be >= 2")
    x = np.asarray(x, dtype=np.float64)
    y = np.tanh(np.clip(x, -x_max, x_max))
    grid_x = np.linspace(-x_max, x_max, n_segments + 1)
    grid_y = np.tanh(grid_x)
    return np.interp(np.clip(x, -x_max, x_max), grid_x, grid_y, left=-1.0, right=1.0)


def run_tanh_approx(
    output_dir: Path,
    x_max: float,
    n_points: int,
    n_segments: int,
) -> None:
    out_dir = ensure_dir(output_dir / "tanh_approx")

    x = np.linspace(-x_max, x_max, n_points, dtype=np.float64)
    y_true = np.tanh(x)
    y_rat = tanh_rational(x)
    y_pwl = tanh_pwl(x, x_max=min(3.0, x_max), n_segments=n_segments)

    rat = compute_metrics(y_true, y_rat)
    pwl = compute_metrics(y_true, y_pwl)
    max_err_rat = float(np.max(np.abs(y_true - y_rat)))
    max_err_pwl = float(np.max(np.abs(y_true - y_pwl)))

    summary = {
        "range": {"x_max": x_max, "n_points": n_points},
        "rational": {"rmse": rat.rmse, "mae": rat.mae, "max_abs_err": max_err_rat},
        "pwl": {"rmse": pwl.rmse, "mae": pwl.mae, "max_abs_err": max_err_pwl, "n_segments": n_segments},
    }
    save_json(out_dir / "tanh_approx_results.json", summary)

    write_csv(
        out_dir / "tanh_approx_results.csv",
        ["method", "rmse", "mae", "max_abs_err", "n_segments"],
        [
            ["rational", rat.rmse, rat.mae, max_err_rat, ""],
            ["pwl", pwl.rmse, pwl.mae, max_err_pwl, n_segments],
        ],
    )

    if plt is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, y_true, label="tanh (ideal)", linewidth=2)
        ax.plot(x, y_rat, label="rational approx", linestyle="--")
        ax.plot(x, y_pwl, label=f"PWL approx ({n_segments} seg)", linestyle=":")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "tanh_approx_plot.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, np.abs(y_true - y_rat), label="|err| rational", linestyle="--")
        ax.plot(x, np.abs(y_true - y_pwl), label="|err| PWL", linestyle=":")
        ax.set_xlabel("x")
        ax.set_ylabel("Absolute error")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "tanh_approx_error.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


# -----------------------------------------------------------------------------
# Training curves (loss vs iteration)
# -----------------------------------------------------------------------------


def train_memristor_pinn_with_logging(
    model: nn.Module,
    V_data: np.ndarray,
    I_data: np.ndarray,
    x_data: np.ndarray,
    epochs: int,
    lr: float,
    lambda_physics: float,
    seed: int,
    device: str,
) -> Dict[str, List[float]]:
    set_seed(seed)

    model = model.to(device)
    model.train()

    V = torch.tensor(V_data, dtype=torch.float32, device=device, requires_grad=True)
    I_true = torch.tensor(I_data, dtype=torch.float32, device=device)
    x = torch.tensor(x_data, dtype=torch.float32, device=device, requires_grad=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)

    history: Dict[str, List[float]] = {"loss": [], "loss_data": [], "loss_physics": [], "loss_smooth": []}

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        I_pred, x_new = model(V, x)

        loss_data = torch.mean((I_pred - I_true) ** 2)
        loss_physics = torch.mean(torch.relu(-x_new) + torch.relu(x_new - 1))
        dI_dV = torch.autograd.grad(I_pred.sum(), V, create_graph=True, retain_graph=True)[0]
        loss_smooth = torch.mean(dI_dV**2) * 1e-6

        loss = loss_data + lambda_physics * loss_physics + loss_smooth
        loss.backward()
        optimizer.step()
        scheduler.step()

        history["loss"].append(float(loss.detach().cpu().item()))
        history["loss_data"].append(float(loss_data.detach().cpu().item()))
        history["loss_physics"].append(float(loss_physics.detach().cpu().item()))
        history["loss_smooth"].append(float(loss_smooth.detach().cpu().item()))

    return history


def run_training_curves(
    output_dir: Path,
    data_csv: Path,
    epochs: int,
    lr: float,
    lambda_physics: float,
    seed: int,
    device: str,
) -> None:
    out_dir = ensure_dir(output_dir / "training_curves")
    V, I, x = load_memristor_csv(data_csv)

    model = MemristorPINN()
    start = time.time()
    history = train_memristor_pinn_with_logging(
        model=model,
        V_data=V,
        I_data=I,
        x_data=x,
        epochs=epochs,
        lr=lr,
        lambda_physics=lambda_physics,
        seed=seed,
        device=device,
    )
    duration_s = time.time() - start

    # Save raw history
    hist_csv = out_dir / "memristor_training_loss.csv"
    rows = []
    for i in range(epochs):
        rows.append([i + 1, history["loss"][i], history["loss_data"][i], history["loss_physics"][i], history["loss_smooth"][i]])
    write_csv(hist_csv, ["epoch", "loss", "loss_data", "loss_physics", "loss_smooth"], rows)

    summary = {
        "data_csv": str(data_csv),
        "epochs": epochs,
        "lr": lr,
        "lambda_physics": lambda_physics,
        "seed": seed,
        "device": device,
        "wall_time_s": duration_s,
        "final": {k: history[k][-1] for k in history},
    }
    save_json(out_dir / "memristor_training_loss.json", summary)

    if plt is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history["loss"], label="total")
        ax.plot(history["loss_data"], label="data", alpha=0.9)
        ax.plot(history["loss_physics"], label="physics", alpha=0.9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "memristor_training_loss.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("threshold-sweep", help="Clustering threshold sensitivity on a trained memristor model")
    s1.add_argument("--model", type=Path, default=DEFAULT_MEMRISTOR_MODEL)
    s1.add_argument("--data", type=Path, default=DEFAULT_MEMRISTOR_CSV)
    s1.add_argument("--thresholds", type=str, default="0.001,0.005,0.01,0.02,0.05,0.1")
    s1.add_argument("--test-ratio", type=float, default=0.2)
    s1.add_argument("--seed", type=int, default=42)

    s2 = sub.add_parser("bitwidth-sweep", help="Reduced-precision weight sensitivity sweep")
    s2.add_argument("--model", type=Path, default=DEFAULT_MEMRISTOR_MODEL)
    s2.add_argument("--data", type=Path, default=DEFAULT_MEMRISTOR_CSV)
    s2.add_argument("--bits", type=str, default="4,6,8,10,12,16")
    s2.add_argument("--test-ratio", type=float, default=0.2)
    s2.add_argument("--seed", type=int, default=42)
    s2.add_argument("--asymmetric", action="store_true", help="Use asymmetric min/max quantization (default: symmetric)")

    s3 = sub.add_parser("tanh-approx", help="Quantify tanh approximation error")
    s3.add_argument("--x-max", type=float, default=4.0)
    s3.add_argument("--n-points", type=int, default=2001)
    s3.add_argument("--pwl-segments", type=int, default=8)

    s4 = sub.add_parser("training-curves", help="Generate loss-vs-epoch curves for memristor PINN")
    s4.add_argument("--data", type=Path, default=DEFAULT_MEMRISTOR_CSV)
    s4.add_argument("--epochs", type=int, default=1500)
    s4.add_argument("--lr", type=float, default=1e-3)
    s4.add_argument("--lambda-physics", type=float, default=0.1)
    s4.add_argument("--seed", type=int, default=42)

    s5 = sub.add_parser("all", help="Run all reviewer-response artifact generators")
    s5.add_argument("--model", type=Path, default=DEFAULT_MEMRISTOR_MODEL)
    s5.add_argument("--data", type=Path, default=DEFAULT_MEMRISTOR_CSV)
    s5.add_argument("--thresholds", type=str, default="0.001,0.005,0.01,0.02,0.05,0.1")
    s5.add_argument("--bits", type=str, default="4,6,8,10,12,16")
    s5.add_argument("--test-ratio", type=float, default=0.2)
    s5.add_argument("--seed", type=int, default=42)
    s5.add_argument("--epochs", type=int, default=1500)
    s5.add_argument("--lr", type=float, default=1e-3)
    s5.add_argument("--lambda-physics", type=float, default=0.1)

    return p.parse_args()


def _parse_float_list(s: str) -> List[float]:
    out: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    if plt is None:
        print("[WARN] matplotlib not available; plots will be skipped (CSV/JSON still produced).")

    if args.cmd == "threshold-sweep":
        run_threshold_sweep(
            output_dir=output_dir,
            model_path=args.model,
            data_csv=args.data,
            thresholds=_parse_float_list(args.thresholds),
            test_ratio=args.test_ratio,
            seed=args.seed,
            device=args.device,
        )
        return 0

    if args.cmd == "bitwidth-sweep":
        run_bitwidth_sweep(
            output_dir=output_dir,
            model_path=args.model,
            data_csv=args.data,
            bits_list=_parse_int_list(args.bits),
            test_ratio=args.test_ratio,
            seed=args.seed,
            symmetric=not args.asymmetric,
            device=args.device,
        )
        return 0

    if args.cmd == "tanh-approx":
        run_tanh_approx(
            output_dir=output_dir,
            x_max=args.x_max,
            n_points=args.n_points,
            n_segments=args.pwl_segments,
        )
        return 0

    if args.cmd == "training-curves":
        run_training_curves(
            output_dir=output_dir,
            data_csv=args.data,
            epochs=args.epochs,
            lr=args.lr,
            lambda_physics=args.lambda_physics,
            seed=args.seed,
            device=args.device,
        )
        return 0

    if args.cmd == "all":
        thresholds = _parse_float_list(args.thresholds)
        bits_list = _parse_int_list(args.bits)

        run_threshold_sweep(
            output_dir=output_dir,
            model_path=args.model,
            data_csv=args.data,
            thresholds=thresholds,
            test_ratio=args.test_ratio,
            seed=args.seed,
            device=args.device,
        )
        run_bitwidth_sweep(
            output_dir=output_dir,
            model_path=args.model,
            data_csv=args.data,
            bits_list=bits_list,
            test_ratio=args.test_ratio,
            seed=args.seed,
            symmetric=True,
            device=args.device,
        )
        run_tanh_approx(output_dir=output_dir, x_max=4.0, n_points=2001, n_segments=8)
        run_training_curves(
            output_dir=output_dir,
            data_csv=args.data,
            epochs=args.epochs,
            lr=args.lr,
            lambda_physics=args.lambda_physics,
            seed=args.seed,
            device=args.device,
        )
        return 0

    raise AssertionError(f"Unhandled cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())

