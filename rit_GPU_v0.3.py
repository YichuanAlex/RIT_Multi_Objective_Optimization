#!/usr/bin/env python3
# DDP-enabled grid search + top10 retrain distributed among ranks (each rank handles subset of top10)
# torchrun --nproc_per_node=4 --master_port=29547 rit_GPU_v0.3.py --csv AI_workload.csv --out outputs --seed 42
# python rit_GPU_v0.3.py --csv AI_workload.csv --out outputs --seed 42

import argparse
import json
import os
import random
import time
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# -------------------------
# Utilities / Data handling
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def minmax_normalize(values: np.ndarray):
    arr = np.asarray(values, dtype=float)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmax == vmin:
        return np.zeros_like(arr, dtype=float), vmin, vmax
    return (arr - vmin) / (vmax - vmin), vmin, vmax

def make_sliding_windows(values: np.ndarray, window_len: int = 24, step: int = 2) -> np.ndarray:
    xs = []
    i = 0
    while i + window_len <= len(values):
        xs.append(values[i:i + window_len])
        i += step
    return np.array(xs)

def split_windows(windows: np.ndarray, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    n = len(windows)
    n_train = int(n * train_ratio)
    return windows[:n_train], windows[n_train:]

def train_val_split(windows: np.ndarray, val_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    n = len(windows)
    n_val = max(1, int(n * val_ratio))
    return windows[:-n_val], windows[-n_val:]

def safe_makedirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# -------------------------
# Model: LSTM action predictor
# -------------------------
class LSTMAction(nn.Module):
    def __init__(self, hidden_size: int = 256, num_layers: int = 2, dropout_rate: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size + 1, 1)

    def forward_once(self, y_seq: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
        # y_seq: (t,)
        feat = y_seq.unsqueeze(0).unsqueeze(-1).contiguous()  # (1,t,1)
        _, (h_T, _) = self.lstm(feat)  # h_T: (num_layers, 1, hidden)
        h = h_T[-1, 0, :]  # (hidden,)
        h = self.dropout(h)
        x_prev_ = x_prev.unsqueeze(0)  # (1,)
        hcat = torch.cat([h, x_prev_], dim=-1)  # (hidden+1,)
        out = self.fc(hcat)  # (1,)
        return out.squeeze()

# unwrap DDP/DP
def unwrap_model(m: nn.Module) -> nn.Module:
    if isinstance(m, torch.nn.parallel.DistributedDataParallel) or isinstance(m, torch.nn.DataParallel):
        return m.module
    return m

# -------------------------
# E2E Trainer with closed-form MLA-ROBD calibrator
# -------------------------
class E2ETrainer:
    def __init__(self, base_model: nn.Module, m: float,
                 lambda1: float, lambda2: float, lambda3: float, mu: float,
                 device: torch.device):
        self.base = base_model.to(device)
        self.m = float(m)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.lambda3 = float(lambda3)
        self.mu = float(mu)
        self.device = device
        self.eps = 1e-9

    def forward_window(self, y_seq: torch.Tensor, x0: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        T = y_seq.shape[0]
        prev_cal = torch.tensor(x0, dtype=torch.float32, device=self.device)
        x_mls = []
        x_cals = []
        pre_sq = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        post_cost = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        denom = (self.m + self.lambda1 + self.lambda2 + self.lambda3) + self.eps

        base_model = unwrap_model(self.base)

        for t in range(1, T + 1):
            y_hist = y_seq[:t]
            y_t = y_seq[t - 1]
            x_ml = base_model.forward_once(y_hist, prev_cal)
            v_t = y_t
            numerator = (self.m * y_t) + (self.lambda1 * prev_cal) + (self.lambda2 * v_t) + (self.lambda3 * x_ml)
            x_cal = numerator / denom
            pre_sq = pre_sq + (x_ml - v_t) ** 2
            hit = (self.m / 2.0) * ((x_cal - y_t) ** 2)
            sw = (1.0 / 2.0) * ((x_cal - prev_cal) ** 2)
            post_cost = post_cost + (hit + sw)
            x_mls.append(x_ml)
            x_cals.append(x_cal)
            prev_cal = x_cal

        pre_mse = pre_sq / float(T)
        post_cost_avg = post_cost / float(T)
        x_ml_seq = torch.stack(x_mls)
        x_cal_seq = torch.stack(x_cals)
        return pre_mse, post_cost_avg, x_ml_seq, x_cal_seq

# -------------------------
# Train for single hyperparam combo
# -------------------------
def train_for_combo(lambda1, lambda2, lambda3, mu,
                    train_windows, val_windows,
                    args, device, ddp_use=False, world_size=1, rank=0, combo_dir: Optional[Path]=None) -> Tuple[dict, nn.Module, list]:
    """
    Train base model for given hyperparams.
    If ddp_use True, wrap model in DDP on this process (for gradient sync during tuning).
    Returns metrics, model_for_train (possibly DDP wrapped), train_log.
    """
    base = LSTMAction(hidden_size=args.hidden, num_layers=args.layers, dropout_rate=args.dropout).to(device)

    model_for_train = base
    ddp_model = None
    if ddp_use and torch.distributed.is_initialized() and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        try:
            ddp_model = torch.nn.parallel.DistributedDataParallel(base, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
            model_for_train = ddp_model
        except Exception:
            # fallback: leave base unwrapped
            model_for_train = base

    trainer = E2ETrainer(model_for_train, m=args.m, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, mu=mu, device=device)
    optimizer = optim.Adam(model_for_train.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model_for_train.train()
    train_log = []
    best_val_post = float('inf')
    best_epoch = -1
    patience = args.patience
    no_improve = 0
    best_state = None

    for ep in range(1, args.tune_epochs + 1):
        total_loss = 0.0
        # training over windows
        for w in train_windows:
            y_w = torch.tensor(w, dtype=torch.float32, device=device)
            optimizer.zero_grad()
            pre_mse, post_cost_avg, _, _ = trainer.forward_window(y_w, x0=0.0)
            loss = mu * pre_mse + (1.0 - mu) * post_cost_avg
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        avg_train_loss = total_loss / float(max(1, len(train_windows)))

        # validation
        model_for_train.eval()
        val_post_list = []
        val_pre_list = []
        with torch.no_grad():
            for vw in val_windows:
                y_v = torch.tensor(vw, dtype=torch.float32, device=device)
                pre_mse_v, post_cost_v, _, _ = trainer.forward_window(y_v, x0=0.0)
                val_post_list.append(float(post_cost_v.item()))
                val_pre_list.append(float(pre_mse_v.item()))
        val_post_avg = float(np.mean(val_post_list)) if len(val_post_list) > 0 else float('inf')
        val_pre_avg = float(np.mean(val_pre_list)) if len(val_pre_list) > 0 else float('inf')
        model_for_train.train()

        train_log.append({"epoch": ep, "train_loss": avg_train_loss, "val_post_avg": val_post_avg, "val_pre_avg": val_pre_avg})

        # early stopping
        if val_post_avg + 1e-12 < best_val_post:
            best_val_post = val_post_avg
            best_epoch = ep
            no_improve = 0
            # save cpu copy of unwrapped state_dict
            try:
                cpu_state = {k: v.cpu().clone() for k, v in unwrap_model(model_for_train).state_dict().items()}
                best_state = cpu_state
            except Exception:
                best_state = None
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # reload best state if present
    if best_state is not None:
        try:
            unwrap_model(model_for_train).load_state_dict(best_state)
        except Exception:
            pass

    metrics = {
        "val_post_avg": best_val_post,
        "val_pre_avg": val_pre_avg,
        "best_epoch": best_epoch,
        "epochs_ran": ep
    }

    # save model state for combo_dir if provided (save unwrapped module state)
    if combo_dir is not None:
        try:
            torch.save(unwrap_model(model_for_train).state_dict(), combo_dir / "model_state.pt")
        except Exception as e:
            print(f"Warning: failed to save model_state for {combo_dir}: {e}")

    return metrics, model_for_train, train_log

# -------------------------
# Evaluation helpers
# -------------------------
def evaluate_post_calibrated(base_model: nn.Module, windows: List[np.ndarray],
                             m: float, lambda1: float, lambda2: float, lambda3: float, device: torch.device):
    trainer = E2ETrainer(base_model, m=m, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, mu=0.0, device=device)
    base_model.eval()
    losses = []
    seqs = []
    with torch.no_grad():
        for w in windows:
            y_w = torch.tensor(w, dtype=torch.float32, device=device)
            _, post_cost_avg, _, x_cal_seq = trainer.forward_window(y_w, x0=0.0)
            losses.append(float(post_cost_avg.item()))
            seqs.append(x_cal_seq.cpu().numpy())
    return losses, seqs

def evaluate_ml_only(base_model: nn.Module, windows: List[np.ndarray], m: float, device: torch.device):
    base_model.eval()
    losses = []
    seqs = []
    with torch.no_grad():
        for w in windows:
            y_w = torch.tensor(w, dtype=torch.float32, device=device)
            T = y_w.shape[0]
            prev = torch.tensor(0.0, dtype=torch.float32, device=device)
            total = torch.tensor(0.0, dtype=torch.float32, device=device)
            xs = []
            for t in range(1, T + 1):
                y_hist = y_w[:t]
                x_ml = unwrap_model(base_model).forward_once(y_hist, prev)
                hit = (m / 2.0) * ((x_ml - y_w[t - 1]) ** 2)
                sw = (1.0 / 2.0) * ((x_ml - prev) ** 2)
                total = total + (hit + sw)
                xs.append(float(x_ml.cpu().numpy()))
                prev = x_ml
            losses.append(float((total / float(T)).cpu().numpy()))
            seqs.append(np.array(xs))
    return losses, seqs

def robd_baseline_np(windows: List[np.ndarray], m: float):
    losses = []
    seqs = []
    for w in windows:
        xs = []
        prev = 0.0
        total = 0.0
        for t in range(len(w)):
            y_t = float(w[t])
            x_t = (m * y_t + prev) / (1.0 + m)
            xs.append(x_t)
            total += (m / 2.0) * ((x_t - y_t) ** 2) + (1.0 / 2.0) * ((x_t - prev) ** 2)
            prev = x_t
        losses.append(total / float(len(w)))
        seqs.append(np.array(xs))
    return losses, seqs

def posthoc_mla_grid_analysis(ml_seqs: List[np.ndarray], windows: List[np.ndarray], m: float, theta_grid: List[float], scale_sq: float):
    results = []
    for theta in theta_grid:
        losses = []
        for w, h in zip(windows, ml_seqs):
            prev = 0.0
            total = 0.0
            denom = 1.0 + m + theta
            for t in range(len(w)):
                y_t = float(w[t])
                h_t = float(h[t])
                x_t = (m * y_t + prev + theta * h_t) / denom
                total += (m / 2.0) * ((x_t - y_t) ** 2) + (1.0 / 2.0) * ((x_t - prev) ** 2)
                prev = x_t
            losses.append(total / float(len(w)))
        avg_norm = float(np.mean(losses))
        worst_norm = float(np.max(losses))
        avg_mapped = avg_norm * scale_sq
        worst_mapped = worst_norm * scale_sq
        results.append({"theta": float(theta), "avg_loss": float(avg_mapped), "worst_loss": float(worst_mapped)})
    return results

# -------------------------
# Plot helpers (restored)
# -------------------------
def save_objective_hist_box(out_dir: Path, robd_losses: List[float], ml_losses: List[float], e2e_losses: List[float]):
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_out = out_dir / "objective_hist.png"
    box_out = out_dir / "objective_box.png"

    plt.figure(figsize=(8,4))
    plt.hist(robd_losses, bins=20, alpha=0.6, label="R-OBD")
    plt.hist(ml_losses, bins=20, alpha=0.6, label="ML-only")
    plt.hist(e2e_losses, bins=20, alpha=0.6, label="E2E")
    plt.legend()
    plt.title("Objective distribution on test windows")
    plt.xlabel("Loss")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(hist_out)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.boxplot([robd_losses, ml_losses, e2e_losses], tick_labels=["R-OBD", "ML-only", "E2E"])
    plt.title("Objective comparison (boxplot)")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(box_out)
    plt.close()

    return hist_out, box_out

def save_mla_tradeoff_plot(out_dir: Path, mla_df: pd.DataFrame, e2e_losses: List[float], robd_losses: List[float]):
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_out = out_dir / "mla_tradeoff.png"
    csv_out = out_dir / "mla_tradeoff.csv"
    mla_df.to_csv(csv_out, index=False)

    try:
        thetas = [float(x) for x in mla_df["theta"].tolist()]
        avg_losses = [float(x) for x in mla_df["avg_loss"].tolist()]
        worst_losses = [float(x) for x in mla_df["worst_loss"].tolist()]
    except Exception as e:
        print("Warning: failed to parse mla_tradeoff dataframe:", e)
        thetas, avg_losses, worst_losses = [], [], []

    plt.figure(figsize=(9,4))
    if len(thetas) > 0 and len(avg_losses) == len(thetas) and len(worst_losses) == len(thetas):
        x = np.arange(len(thetas))
        width = 0.35
        plt.bar(x - width/2, avg_losses, width, label="MLA-ROBD Avg Loss")
        plt.bar(x + width/2, worst_losses, width, label="MLA-ROBD Worst Loss")
        if len(e2e_losses) > 0:
            plt.axhline(y=float(np.mean(e2e_losses)), color='r', linestyle='--', label='E2E Avg Loss')
        if len(robd_losses) > 0:
            plt.axhline(y=float(np.mean(robd_losses)), color='g', linestyle='--', label='R-OBD Avg Loss')
        plt.xticks(x, [str(t) for t in thetas])
        plt.xlabel("Theta")
        plt.ylabel("Loss (mapped back)")
        plt.title("MLA-ROBD Trade-off (post-hoc)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_out)
        plt.close()
    else:
        plt.figure(figsize=(6,3))
        plt.text(0.1, 0.5, "No MLA tradeoff data available to plot", fontsize=12)
        if len(e2e_losses) > 0:
            plt.axhline(y=float(np.mean(e2e_losses)), color='r', linestyle='--', label='E2E Avg Loss')
        if len(robd_losses) > 0:
            plt.axhline(y=float(np.mean(robd_losses)), color='g', linestyle='--', label='R-OBD Avg Loss')
        plt.legend()
        plt.title("MLA-ROBD Trade-off (placeholder)")
        plt.tight_layout()
        plt.savefig(plot_out)
        plt.close()

    return csv_out, plot_out

# -------------------------
# Grid-run orchestration with DDP-aware distribution of combos and distributed retrain of top10
# -------------------------
def run_grid_search_and_final(args):
    # detect distributed environment (torchrun sets LOCAL_RANK, WORLD_SIZE, RANK)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    distributed = world_size > 1

    # init process group if distributed
    if distributed:
        # choose backend
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        try:
            # set NCCL async error handling to help debugging (no harm)
            os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
            torch.distributed.init_process_group(backend=backend)
        except Exception as e:
            print(f"[rank{rank}] Warning: init_process_group failed: {e}")
    # set device
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"[rank{rank}] Distributed={distributed} world_size={world_size} device={device}")

    set_seed(args.seed + rank)

    # load data
    df = pd.read_csv(args.csv)
    values = df.iloc[:, 1].values.astype(float)
    print(f"[rank{rank}] Loaded {len(values)} data points from {args.csv}")
    values_norm, vmin, vmax = minmax_normalize(values)
    scale_sq = (vmax - vmin) ** 2 if vmax != vmin else 1.0
    print(f"[rank{rank}] Normalization mapping: vmin={vmin}, vmax={vmax}, scale_sq={scale_sq:.6g}")

    windows = make_sliding_windows(values_norm, window_len=24, step=2)
    train_windows, test_windows = split_windows(windows, train_ratio=0.8)
    print(f"[rank{rank}] Generated {len(windows)} windows, train={len(train_windows)}, test={len(test_windows)}")

    train_sub, val_windows = train_val_split(train_windows, val_ratio=0.2)
    print(f"[rank{rank}] Using train_sub={len(train_sub)} for tuning, val={len(val_windows)}")

    # grid lists
    lambda1_list = args.lambda1_list if args.lambda1_list is not None else [0.8]
    lambda2_list = args.lambda2_list if args.lambda2_list is not None else [0.4]
    lambda3_list = args.lambda3_list if args.lambda3_list is not None else [0.7]
    mu_list = args.mu_list if args.mu_list is not None else [0.4]

    combos = [(l1, l2, l3, mu) for l1 in lambda1_list for l2 in lambda2_list for l3 in lambda3_list for mu in mu_list]
    total = len(combos)
    print(f"[rank{rank}] Grid search over {total} combinations (tune_epochs={args.tune_epochs}).")

    safe_makedirs(Path(args.out))

    # Distribute combos by (index-1) % world_size == rank
    summary_rows_local = []
    best_local_val = float('inf')
    best_local_state = None
    best_local_combo_dir = None

    for idx, (l1, l2, l3, mu) in enumerate(combos, start=1):
        # assign combos to ranks in round-robin
        if (idx - 1) % world_size != rank:
            continue

        # compose folder name including main training params
        combo_name = (f"l1={l1}_l2={l2}_l3={l3}_mu={mu}"
                      f"_hidden={args.hidden}_layers={args.layers}_dropout={args.dropout}"
                      f"_lr={args.lr}_seed={args.seed}")
        combo_dir = Path(args.out) / combo_name
        safe_makedirs(combo_dir)

        # skip if already exists (val_metrics.json)
        if (combo_dir / "val_metrics.json").exists():
            print(f"[rank{rank}] [{idx}/{total}] SKIP existing combo {combo_name}")
            # try to read to include into local summary
            try:
                vm = json.load(open(combo_dir / "val_metrics.json", "r", encoding="utf-8"))
                summary_rows_local.append({
                    "lambda1": float(l1), "lambda2": float(l2), "lambda3": float(l3), "mu": float(mu),
                    "val_post_avg": float(vm.get("val_post_avg", np.nan)), "val_pre_avg": float(vm.get("val_pre_avg", np.nan)),
                    "best_epoch": int(vm.get("best_epoch", 0)), "epochs_ran": int(vm.get("epochs_ran", 0)), "runtime": float(vm.get("runtime_seconds", 0.0)),
                    "combo_dir": str(combo_dir)
                })
                if vm.get("val_post_avg", np.nan) < best_local_val:
                    best_local_val = vm.get("val_post_avg", np.nan)
                    best_local_combo_dir = combo_dir
                # continue to next combo
            except Exception:
                pass
            continue

        print(f"[rank{rank}] [{idx}/{total}] Training combo {combo_name} ...")
        t0 = time.time()
        ddp_use = (distributed and torch.cuda.is_available() and world_size > 1)
        metrics, trained_base, train_log = train_for_combo(l1, l2, l3, mu,
                                                           train_sub, val_windows,
                                                           args, device,
                                                           ddp_use=ddp_use, world_size=world_size, rank=rank,
                                                           combo_dir=combo_dir)
        runtime = time.time() - t0

        # save logs and metrics
        try:
            with open(combo_dir / "train_log.txt", "w", encoding="utf-8") as f:
                for entry in train_log:
                    f.write(json.dumps(entry) + "\n")
            val_metrics = {"val_post_avg": metrics["val_post_avg"], "val_pre_avg": metrics["val_pre_avg"],
                           "best_epoch": metrics["best_epoch"], "epochs_ran": metrics["epochs_ran"],
                           "runtime_seconds": runtime}
            with open(combo_dir / "val_metrics.json", "w", encoding="utf-8") as f:
                json.dump(val_metrics, f, indent=2)
        except Exception as e:
            print(f"[rank{rank}] Warning: failed to save metrics for {combo_name}: {e}")

        # compute val_post_stats
        try:
            trainer_tmp = E2ETrainer(trained_base, m=args.m, lambda1=l1, lambda2=l2, lambda3=l3, mu=mu, device=device)
            trained_base.eval()
            val_post_list = []
            with torch.no_grad():
                for vw in val_windows:
                    y_v = torch.tensor(vw, dtype=torch.float32, device=device)
                    _, post_cost_v, _, _ = trainer_tmp.forward_window(y_v, x0=0.0)
                    val_post_list.append(float(post_cost_v.item()))
            val_post_arr = np.array(val_post_list) if len(val_post_list) > 0 else np.array([np.nan])
            val_stats = {
                "val_post_mean": float(np.mean(val_post_arr)),
                "val_post_median": float(np.median(val_post_arr)),
                "val_post_std": float(np.std(val_post_arr)),
                "val_post_max": float(np.max(val_post_arr))
            }
            with open(combo_dir / "val_post_stats.json", "w", encoding="utf-8") as f:
                json.dump(val_stats, f, indent=2)
        except Exception as e:
            print(f"[rank{rank}] Warning: val_post_stats failed for {combo_name}: {e}")

        # ensure model state saved
        try:
            if not (combo_dir / "model_state.pt").exists():
                torch.save(unwrap_model(trained_base).state_dict(), combo_dir / "model_state.pt")
        except Exception as e:
            print(f"[rank{rank}] Warning: saving model_state failed for {combo_name}: {e}")

        summary_rows_local.append({
            "lambda1": float(l1), "lambda2": float(l2), "lambda3": float(l3), "mu": float(mu),
            "val_post_avg": float(metrics["val_post_avg"]), "val_pre_avg": float(metrics["val_pre_avg"]),
            "best_epoch": int(metrics["best_epoch"]), "epochs_ran": int(metrics["epochs_ran"]), "runtime": runtime,
            "combo_dir": str(combo_dir)
        })

        if metrics["val_post_avg"] < best_local_val:
            best_local_val = metrics["val_post_avg"]
            try:
                best_local_state = {k: v.cpu().clone() for k, v in unwrap_model(trained_base).state_dict().items()}
            except Exception:
                best_local_state = None
            best_local_combo_dir = combo_dir

        print(f"[rank{rank}]   combo done. val_post_avg={metrics['val_post_avg']:.6f} (runtime {runtime:.1f}s)")

    # Done assigned combos for each rank. synchronize
    if distributed:
        try:
            torch.distributed.barrier()
        except Exception:
            pass

    # Only rank0 consolidates all combo results
    topk_list = None
    if rank == 0:
        # scan outputs directory for combo subdirs
        out_root = Path(args.out)
        final_rows = []
        for sub in sorted(out_root.iterdir()):
            if not sub.is_dir():
                continue
            vm_path = sub / "val_metrics.json"
            if vm_path.exists():
                try:
                    vm = json.load(open(vm_path, "r", encoding="utf-8"))
                    # attempt to parse l1,l2,l3,mu from folder name
                    name = sub.name
                    l1 = l2 = l3 = mu = np.nan
                    for part in name.split("_"):
                        if part.startswith("l1="):
                            try: l1 = float(part.split("=")[1])
                            except: pass
                        if part.startswith("l2="):
                            try: l2 = float(part.split("=")[1])
                            except: pass
                        if part.startswith("l3="):
                            try: l3 = float(part.split("=")[1])
                            except: pass
                        if part.startswith("mu="):
                            try: mu = float(part.split("=")[1])
                            except: pass
                    final_rows.append({
                        "lambda1": float(l1) if not np.isnan(l1) else np.nan,
                        "lambda2": float(l2) if not np.isnan(l2) else np.nan,
                        "lambda3": float(l3) if not np.isnan(l3) else np.nan,
                        "mu": float(mu) if not np.isnan(mu) else np.nan,
                        "val_post_avg": float(vm.get("val_post_avg", np.nan)),
                        "val_pre_avg": float(vm.get("val_pre_avg", np.nan)),
                        "best_epoch": int(vm.get("best_epoch", 0)),
                        "epochs_ran": int(vm.get("epochs_ran", 0)),
                        "runtime": float(vm.get("runtime_seconds", 0.0)),
                        "combo_dir": str(sub)
                    })
                except Exception as e:
                    print(f"[rank{rank}] Warning parsing {vm_path}: {e}")
        summary_df = pd.DataFrame(final_rows)
        summary_csv = Path(args.out) / "summary.csv"
        summary_df.to_csv(summary_csv, index=False)

        # pick top10
        if len(summary_df) > 0:
            summary_df_sorted = summary_df.sort_values(by="val_post_avg", ascending=True).reset_index(drop=True)
            topk = min(10, len(summary_df_sorted))
            top10 = summary_df_sorted.head(topk)
            topk_list = top10.to_dict(orient="records")
            # write intermediate file
            with open(Path(args.out) / "top10.json", "w", encoding="utf-8") as f:
                json.dump(topk_list, f, indent=2)
            print("=== GRID SEARCH COMPLETE ===")
            best_row = summary_df_sorted.iloc[0]
            print(f"Best combo (rank0 found): lambda1,lambda2,lambda3,mu = ({best_row['lambda1']}, {best_row['lambda2']}, {best_row['lambda3']}, {best_row['mu']}) with val_post_avg = {best_row['val_post_avg']:.6f}")
            print(f"Best combo dir: {best_row['combo_dir']}")
        else:
            print("No combos found; nothing to retrain.")
            topk_list = []
    # broadcast topk_list to other ranks
    if distributed:
        try:
            if rank == 0:
                obj = [topk_list]
            else:
                obj = [None]
            torch.distributed.broadcast_object_list(obj, src=0)
            topk_list = obj[0]
        except Exception as e:
            print(f"[rank{rank}] Warning: broadcast top10 failed: {e}")
            # fallback: try to read file
            try:
                topk_list = json.load(open(Path(args.out) / "top10.json", "r", encoding="utf-8"))
            except Exception:
                topk_list = []
    else:
        # not distributed -> rank0 made topk_list earlier
        if rank == 0:
            # already set
            pass
        else:
            topk_list = []

    # If topk_list empty, skip retrain
    if not topk_list:
        if rank == 0:
            print("No top10 to retrain; exiting.")
        if distributed:
            try:
                torch.distributed.barrier()
            except Exception:
                pass
            try:
                torch.distributed.destroy_process_group()
            except Exception:
                pass
        return

    # ---------- Distributed retrain of top10: assign tasks among ranks ----------
    # Each rank i trains tasks whose index % world_size == rank
    # Use single-process single-device training (no DDP) to avoid collectives during retrain
    assigned_indices = [i for i in range(len(topk_list)) if (i % world_size) == rank]
    if len(assigned_indices) == 0:
        print(f"[rank{rank}] No retrain tasks assigned.")
    else:
        print(f"[rank{rank}] Assigned retrain indices: {assigned_indices}")

    top_final_dirs = []
    for i in assigned_indices:
        rec = topk_list[i]
        # some recs may have 'mu' missing if parsed incorrectly; ensure casting
        try:
            l1 = float(rec.get("lambda1", rec.get("l1", np.nan)))
        except Exception:
            l1 = float(rec.get("lambda1", np.nan))
        try:
            l2 = float(rec.get("lambda2", rec.get("l2", np.nan)))
        except Exception:
            l2 = float(rec.get("lambda2", np.nan))
        try:
            l3 = float(rec.get("lambda3", rec.get("l3", np.nan)))
        except Exception:
            l3 = float(rec.get("lambda3", np.nan))
        try:
            mu = float(rec.get("mu", np.nan))
        except Exception:
            mu = float(rec.get("mu", np.nan))

        combo_dir = Path(rec.get("combo_dir", args.out))
        final_dir = Path(args.out) / f"retrain_rank{rank}_top{i}_l1={l1}_l2={l2}_l3={l3}_mu={mu}_hidden={args.hidden}_layers={args.layers}_dropout={args.dropout}_lr={args.lr}_seed={args.seed}"
        safe_makedirs(final_dir)

        print(f"[rank{rank}] Retraining top-{i+1} combo on full train: l1={l1}, l2={l2}, l3={l3}, mu={mu}")

        # build model and try to load previous state (if present)
        base_model = LSTMAction(hidden_size=args.hidden, num_layers=args.layers, dropout_rate=args.dropout).to(device)
        model_state_path = combo_dir / "model_state.pt"
        if model_state_path.exists():
            try:
                st = torch.load(model_state_path, map_location="cpu")
                base_model.load_state_dict(st)
            except Exception as e:
                print(f"[rank{rank}] Warning loading state for retrain from {model_state_path}: {e}")

        # retrain on full train_windows
        trainer_full = E2ETrainer(base_model, m=args.m, lambda1=l1, lambda2=l2, lambda3=l3, mu=mu, device=device)
        optimizer = optim.Adam(base_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        base_model.train()
        for ep in range(1, args.epochs + 1):
            tot = 0.0
            for w in train_windows:
                y_w = torch.tensor(w, dtype=torch.float32, device=device)
                optimizer.zero_grad()
                pre_mse, post_cost_avg, _, _ = trainer_full.forward_window(y_w, x0=0.0)
                loss = mu * pre_mse + (1.0 - mu) * post_cost_avg
                loss.backward()
                optimizer.step()
                tot += float(loss.item())
            if ep % max(1, args.epochs // 10) == 0 or ep == 1 or ep == args.epochs:
                print(f"[rank{rank} retrain top{i+1} ep {ep}/{args.epochs}] avg loss = {tot / float(len(train_windows)):.8f}")

        # save final state
        torch.save(base_model.state_dict(), final_dir / "final_model_state.pt")

        # evaluate on test set
        e2e_losses_norm, e2e_seqs = evaluate_post_calibrated(base_model, test_windows, args.m, l1, l2, l3, device)
        e2e_losses = [l * scale_sq for l in e2e_losses_norm]
        test_metrics = {"e2e_test_avg_mapped": float(np.mean(e2e_losses)), "e2e_test_mapped_list": [float(x) for x in e2e_losses]}
        with open(final_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)

        # ML-only baseline retrain
        ml_base = LSTMAction(hidden_size=args.hidden, num_layers=args.layers, dropout_rate=args.dropout).to(device)
        ml_optimizer = optim.Adam(ml_base.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        ml_base.train()
        for ep in range(1, args.epochs + 1):
            tot_ml = 0.0
            for w in train_windows:
                y_w = torch.tensor(w, dtype=torch.float32, device=device)
                ml_optimizer.zero_grad()
                T = y_w.shape[0]
                prev = torch.tensor(0.0, dtype=torch.float32, device=device)
                sq = torch.tensor(0.0, dtype=torch.float32, device=device)
                for t in range(1, T + 1):
                    y_hist = y_w[:t]
                    y_t = y_w[t - 1]
                    x_ml = ml_base.forward_once(y_hist, prev)
                    sq = sq + (x_ml - y_t) ** 2
                    prev = x_ml
                loss_ml = sq / float(T)
                loss_ml.backward()
                ml_optimizer.step()
                tot_ml += float(loss_ml.item())
            if ep % max(1, args.epochs // 10) == 0 or ep == 1 or ep == args.epochs:
                print(f"[rank{rank} ML-only retrain top{i+1} ep {ep}/{args.epochs}] avg pre-mse = {tot_ml / float(len(train_windows)):.8f}")

        ml_losses_norm, ml_seqs = evaluate_ml_only(ml_base, test_windows, args.m, device)
        ml_losses = [l * scale_sq for l in ml_losses_norm]

        # R-OBD baseline
        robd_losses_raw, robd_seqs = robd_baseline_np(test_windows, args.m)
        robd_losses = [l * scale_sq for l in robd_losses_raw]

        # MLA post-hoc
        mla_results = posthoc_mla_grid_analysis(ml_seqs, test_windows, args.m, args.theta_grid, scale_sq=scale_sq)
        mla_df = pd.DataFrame(mla_results)

        # plots saved into final_dir
        mla_csv_out, mla_plot_out = save_mla_tradeoff_plot(final_dir, mla_df, e2e_losses, robd_losses)
        hist_out, box_out = save_objective_hist_box(final_dir, robd_losses, ml_losses, e2e_losses)
        # try copy plots back to combo_dir (best effort)
        try:
            shutil.copy(hist_out, combo_dir / "objective_hist.png")
        except Exception:
            pass
        try:
            shutil.copy(box_out, combo_dir / "objective_box.png")
        except Exception:
            pass
        try:
            shutil.copy(mla_plot_out, combo_dir / "mla_tradeoff.png")
        except Exception:
            pass
        mla_df.to_csv(combo_dir / "mla_tradeoff.csv", index=False)

        top_final_dirs.append(str(final_dir))

    # After all ranks finish retrain tasks, we gather top_final_dirs (rank0 will collect)
    if distributed:
        try:
            # gather lists from all ranks to rank0 by using broadcast of a pickled object per rank in order
            # simpler: each rank writes its own partial file; rank0 reads them
            # each rank writes assigned list
            partial_file = Path(args.out) / f"top_final_dirs_rank{rank}.json"
            with open(partial_file, "w", encoding="utf-8") as f:
                json.dump(top_final_dirs, f, indent=2)
        except Exception:
            pass

        try:
            torch.distributed.barrier()
        except Exception:
            pass

        if rank == 0:
            # collect partial files from all ranks
            aggregated = []
            for r in range(world_size):
                pf = Path(args.out) / f"top_final_dirs_rank{r}.json"
                if pf.exists():
                    try:
                        lst = json.load(open(pf, "r", encoding="utf-8"))
                        aggregated.extend(lst)
                    except Exception:
                        pass
            final_summary = {"top10": topk_list, "top_final_dirs": aggregated}
            with open(Path(args.out) / "final_summary.json", "w", encoding="utf-8") as f:
                json.dump(final_summary, f, indent=2)
            print(f"[rank{rank}] Written final_summary.json with {len(aggregated)} retrain results.")
    else:
        # non-distributed: rank0 did all retrains sequentially, top_final_dirs already filled
        final_summary = {"top10": topk_list, "top_final_dirs": top_final_dirs}
        with open(Path(args.out) / "final_summary.json", "w", encoding="utf-8") as f:
            json.dump(final_summary, f, indent=2)
        if rank == 0:
            print(f"[rank{rank}] Written final_summary.json with {len(top_final_dirs)} retrain results.")

    # cleanup process group if created
    if distributed:
        try:
            torch.distributed.barrier()
        except Exception:
            pass
        try:
            torch.distributed.destroy_process_group()
        except Exception:
            pass

# -------------------------
# Argument parsing
# -------------------------
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="AI_workload.csv", help="CSV file; second column is used as the series")
    parser.add_argument("--out", type=str, default="outputs")
    parser.add_argument("--m", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=80, help="epochs for final retrain and ML-only full train")
    parser.add_argument("--tune_epochs", type=int, default=15, help="epochs for tuning each grid combo")
    parser.add_argument("--tune", type=int, default=1, help="1 to run grid tuning")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lambda1_list", nargs="+", type=float, default=None)
    parser.add_argument("--lambda2_list", nargs="+", type=float, default=None)
    parser.add_argument("--lambda3_list", nargs="+", type=float, default=None)
    parser.add_argument("--mu_list", nargs="+", type=float, default=None)
    parser.add_argument("--theta_grid", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0])
    args = parser.parse_args()

    run_grid_search_and_final(args)
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="AI_workload.csv", help="CSV file; second column is used as the series")
    parser.add_argument("--out", type=str, default="outputs")
    parser.add_argument("--m", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=80, help="epochs for final retrain and ML-only full train")
    parser.add_argument("--tune_epochs", type=int, default=15, help="epochs for tuning each grid combo")
    parser.add_argument("--tune", type=int, default=1, help="1 to run grid tuning")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lambda1_list", nargs="+", type=float, default=None)
    parser.add_argument("--lambda2_list", nargs="+", type=float, default=None)
    parser.add_argument("--lambda3_list", nargs="+", type=float, default=None)
    parser.add_argument("--mu_list", nargs="+", type=float, default=None)
    parser.add_argument("--theta_grid", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0])
    args = parser.parse_args()

    run_grid_search_and_final(args)