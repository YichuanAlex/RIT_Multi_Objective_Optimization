#!/usr/bin/env python3
# Distributed (DDP) version of the RIT training/grid search script.

# 多卡启动：
# torchrun --nproc_per_node=4 rit_GPU_v0.2.py --csv AI_workload.csv --out outputs --seed 42
# python -m torch.distributed.run --nproc_per_node=4 rit_v0.8_ddp.py --csv AI_workload.csv --out outputs --seed 42

# 单卡/CPU启动
# python rit_v0.8_ddp.py --csv AI_workload.csv --out outputs --seed 42

import argparse
import json
import os
import random
import time
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Distributed imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

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

# -------------------------
# Torch Dataset wrapper for windows
# -------------------------
class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx].astype(np.float32)

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
        # y_seq shape (t,), x_prev scalar tensor
        feat = y_seq.unsqueeze(0).unsqueeze(-1).contiguous()  # (1, t, 1)
        _, (h_T, _) = self.lstm(feat)  # h_T: (num_layers, 1, hidden)
        h = h_T[-1, 0, :]  # (hidden,)
        h = self.dropout(h)
        x_prev_ = x_prev.unsqueeze(0)  # (1,)
        hcat = torch.cat([h, x_prev_], dim=-1)  # (hidden+1)
        out = self.fc(hcat)  # (1,1) -> squeeze
        return out.squeeze()

# -------------------------
# E2E Trainer with closed-form MLA-ROBD calibrator
# -------------------------
class E2ETrainer:
    def __init__(self, base_model: nn.Module, m: float,
                 lambda1: float, lambda2: float, lambda3: float, mu: float,
                 device: torch.device):
        # base_model expected to be the underlying nn.Module (not necessarily DDP wrapper).
        self.base = base_model
        self.m = float(m)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.lambda3 = float(lambda3)
        self.mu = float(mu)
        self.device = device
        self.eps = 1e-9

    def _model_forward_once(self, model, y_hist: torch.Tensor, prev_cal: torch.Tensor):
        # helper to call forward_once whether model is DDP-wrapped or plain.
        if isinstance(model, DDP):
            return model.module.forward_once(y_hist, prev_cal)
        else:
            return model.forward_once(y_hist, prev_cal)

    def forward_window(self, model, y_seq: torch.Tensor, x0: float = 0.0):
        """
        model: the module (can be DDP or nn.Module)
        y_seq: 1D tensor length T on correct device
        returns (pre_mse, post_cost_avg, x_ml_seq, x_cal_seq) all tensors on device
        """
        device = self.device
        T = y_seq.shape[0]
        prev_cal = torch.tensor(x0, dtype=torch.float32, device=device)
        x_mls = []
        x_cals = []
        pre_sq = torch.tensor(0.0, dtype=torch.float32, device=device)
        post_cost = torch.tensor(0.0, dtype=torch.float32, device=device)

        denom = (self.m + self.lambda1 + self.lambda2 + self.lambda3) + self.eps

        for t in range(1, T + 1):
            y_hist = y_seq[:t]
            y_t = y_seq[t - 1]
            # call model's forward_once
            x_ml = self._model_forward_once(model, y_hist, prev_cal)
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
# small helpers for distributed reduction
# -------------------------
def dist_avg_tensor(x: float):
    """All-reduce scalar float across processes and return average (on each rank)."""
    t = torch.tensor(x, dtype=torch.float32, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t / float(dist.get_world_size())
    return float(t.item())

def dist_sum_tensor(x: float):
    t = torch.tensor(x, dtype=torch.float32, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())

# -------------------------
# Train for single hyperparam combo (distributed-aware)
# -------------------------
def train_for_combo(l1, l2, l3, mu,
                    train_dataset: WindowDataset, val_windows: np.ndarray,
                    args, device, rank, world_size, combo_dir: Path):
    """
    Distributed-aware training for one combo.
    Each process will get a DistributedSampler for train_dataset.
    Returns metrics (dict) and saves model_state.pt in combo_dir (on rank0).
    """
    # dataloaders
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, drop_last=False)

    # model & DDP
    base = LSTMAction(hidden_size=args.hidden, num_layers=args.layers, dropout_rate=args.dropout).to(device)
    # wrap DDP if distributed
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        base = DDP(base, device_ids=[device.index] if device.type == 'cuda' else None)
    trainer = E2ETrainer(base, m=args.m, lambda1=l1, lambda2=l2, lambda3=l3, mu=mu, device=device)

    optimizer = optim.Adam(base.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    base.train()
    train_log = []
    best_val_post = float('inf')
    best_epoch = -1
    patience = args.patience
    no_improve = 0
    best_state = None

    for ep in range(1, args.tune_epochs + 1):
        sampler.set_epoch(ep - 1)  # important for shuffling across epochs
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            # batch: tensor (batch_size, window_len)
            for w in batch:
                y_w = w.to(device)
                optimizer.zero_grad()
                pre_mse, post_cost_avg, _, _ = trainer.forward_window(base, y_w, x0=0.0)
                loss = mu * pre_mse + (1.0 - mu) * post_cost_avg
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                n_batches += 1
        avg_train_loss = total_loss / max(1, n_batches)

        # Validation: compute local sum and count then average across ranks
        base.eval()
        val_post_list_local = []
        val_pre_list_local = []
        with torch.no_grad():
            for vw in val_windows:
                y_v = torch.tensor(vw, dtype=torch.float32, device=device)
                pre_mse_v, post_cost_v, _, _ = trainer.forward_window(base, y_v, x0=0.0)
                val_post_list_local.append(float(post_cost_v.item()))
                val_pre_list_local.append(float(pre_mse_v.item()))
        # local averages
        local_post_avg = float(np.mean(val_post_list_local)) if len(val_post_list_local) > 0 else float('inf')
        local_pre_avg = float(np.mean(val_pre_list_local)) if len(val_pre_list_local) > 0 else float('inf')

        # all-reduce averages to get global average (so early stopping consistent)
        tensor_post = torch.tensor(local_post_avg, device=device, dtype=torch.float32)
        tensor_pre = torch.tensor(local_pre_avg, device=device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized() and world_size > 1:
            dist.all_reduce(tensor_post, op=dist.ReduceOp.SUM)
            dist.all_reduce(tensor_pre, op=dist.ReduceOp.SUM)
            tensor_post = tensor_post / float(world_size)
            tensor_pre = tensor_pre / float(world_size)
        val_post_avg = float(tensor_post.item())
        val_pre_avg = float(tensor_pre.item())

        base.train()
        train_log.append({"epoch": ep, "train_loss": avg_train_loss, "val_post_avg": val_post_avg, "val_pre_avg": val_pre_avg})

        # early stopping on validation post cost (use global val_post_avg)
        if val_post_avg + 1e-12 < best_val_post:
            best_val_post = val_post_avg
            best_epoch = ep
            no_improve = 0
            # only save best state from local model's state dict (move to cpu)
            # if DDP, get module
            state_to_save = base.module.state_dict() if isinstance(base, DDP) else base.state_dict()
            best_state = {k: v.cpu().clone() for k, v in state_to_save.items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # finalize: load best_state back (each rank) to have consistent final model weights
    if best_state is not None:
        # load into base (need to map keys to module if DDP)
        state_loaded = {k: v.to(device) for k, v in best_state.items()}
        if isinstance(base, DDP):
            base.module.load_state_dict(state_loaded)
        else:
            base.load_state_dict(state_loaded)

    # Save model_state.pt (CPU) and val metrics by rank 0 only
    if dist.get_rank() == 0:
        save_state = base.module.state_dict() if isinstance(base, DDP) else base.state_dict()
        torch.save({k: v.cpu() for k, v in save_state.items()}, combo_dir / "model_state.pt")
        val_metrics = {"val_post_avg": float(best_val_post), "val_pre_avg": float(val_pre_avg),
                       "best_epoch": int(best_epoch), "epochs_ran": int(ep)}
        with open(combo_dir / "val_metrics.json", "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, indent=2)
        # save train log
        with open(combo_dir / "train_log.txt", "w", encoding="utf-8") as f:
            for entry in train_log:
                f.write(json.dumps(entry) + "\n")
    # barrier to sync
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    metrics = {"val_post_avg": best_val_post, "val_pre_avg": float(val_pre_avg), "best_epoch": best_epoch, "epochs_ran": int(ep)}
    return metrics, base, train_log

# -------------------------
# Evaluation helpers (non-distributed; used on rank0)
# -------------------------
def evaluate_post_calibrated_plain(base_model: LSTMAction, windows: List[np.ndarray],
                                   m: float, lambda1: float, lambda2: float, lambda3: float, device):
    trainer = E2ETrainer(base_model, m=m, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, mu=0.0, device=device)
    base_model.eval()
    losses = []
    seqs = []
    with torch.no_grad():
        for w in windows:
            y_w = torch.tensor(w, dtype=torch.float32, device=device)
            _, post_cost_avg, _, x_cal_seq = trainer.forward_window(base_model, y_w, x0=0.0)
            losses.append(float(post_cost_avg.item()))
            seqs.append(x_cal_seq.cpu().numpy())
    return losses, seqs

def evaluate_ml_only_plain(base_model: LSTMAction, windows: List[np.ndarray], m: float, device):
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
                x_ml = base_model.forward_once(y_hist, prev)
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
# Plot helpers
# -------------------------
def save_objective_hist_box(out_dir: Path, robd_losses, ml_losses, e2e_losses):
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_out = out_dir / "objective_hist.png"
    box_out = out_dir / "objective_box.png"

    plt.figure()
    plt.hist([robd_losses, ml_losses, e2e_losses], label=["R-OBD", "ML-only", "E2E"], bins=20)
    plt.legend()
    plt.title("Objective distribution on test windows")
    plt.savefig(hist_out)
    plt.close()

    plt.figure()
    plt.boxplot([robd_losses, ml_losses, e2e_losses], tick_labels=["R-OBD", "ML-only", "E2E"])
    plt.title("Objective comparison")
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

    plt.figure(figsize=(8, 4))
    if len(thetas) > 0 and len(avg_losses) == len(thetas) and len(worst_losses) == len(thetas):
        x = np.arange(len(thetas))
        width = 0.35
        plt.bar(x - width/2, avg_losses, width, label="MLA-ROBD Avg Loss")
        plt.bar(x + width/2, worst_losses, width, label="MLA-ROBD Worst Loss")
        if len(e2e_losses) > 0:
            plt.axhline(y=np.mean(e2e_losses), color='r', linestyle='--', label='E2E Avg Loss')
        if len(robd_losses) > 0:
            plt.axhline(y=np.mean(robd_losses), color='g', linestyle='--', label='R-OBD Avg Loss')
        plt.xticks(x, [str(t) for t in thetas])
        plt.xlabel("Theta")
        plt.ylabel("Loss")
        plt.title("MLA-ROBD Trade-off (post-hoc)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_out)
        plt.close()
    else:
        plt.figure(figsize=(6,3))
        plt.text(0.1, 0.5, "No MLA tradeoff data available to plot", fontsize=12)
        if len(e2e_losses) > 0:
            plt.axhline(y=np.mean(e2e_losses), color='r', linestyle='--', label='E2E Avg Loss')
        if len(robd_losses) > 0:
            plt.axhline(y=np.mean(robd_losses), color='g', linestyle='--', label='R-OBD Avg Loss')
        plt.legend()
        plt.title("MLA-ROBD Trade-off (placeholder)")
        plt.savefig(plot_out)
        plt.close()

    return csv_out, plot_out

# -------------------------
# Grid-run orchestration (main)
# -------------------------
def run_grid_search_and_final(args):
    # distributed init if torchrun provided envs
    is_distributed = ("RANK" in os.environ and "WORLD_SIZE" in os.environ) or ("LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ)
    if is_distributed:
        # Init process group (NCCL if GPU available, else gloo)
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # set device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print("Distributed:", is_distributed, "World size:", world_size)
        print("Device for rank0:", device)

    set_seed(args.seed + rank)

    # load data (all ranks read csv)
    df = pd.read_csv(args.csv)
    values = df.iloc[:, 1].values.astype(float)
    if rank == 0:
        print(f"Loaded {len(values)} data points from {args.csv}")

    values_norm, vmin, vmax = minmax_normalize(values)
    scale_sq = (vmax - vmin) ** 2 if vmax != vmin else 1.0

    windows = make_sliding_windows(values_norm, window_len=24, step=2)
    train_windows, test_windows = split_windows(windows, train_ratio=0.8)
    train_sub, val_windows = train_val_split(train_windows, val_ratio=0.2)

    if rank == 0:
        print(f"Generated {len(windows)} windows, train={len(train_windows)}, test={len(test_windows)}")
        print(f"Using train_sub={len(train_sub)} for tuning, val={len(val_windows)}")

    # grid lists (use provided override or defaults)
    lambda1_list = args.lambda1_list if args.lambda1_list is not None else [0.5, 1.0, 2.0, 3.0]
    lambda2_list = args.lambda2_list if args.lambda2_list is not None else [0.5, 1.0, 2.0, 3.0]
    lambda3_list = args.lambda3_list if args.lambda3_list is not None else [0.5, 1.0, 2.0, 3.0]
    mu_list = args.mu_list if args.mu_list is not None else [0.5, 1.0, 2.0, 3.0]

    combos = [(l1, l2, l3, mu) for l1 in lambda1_list for l2 in lambda2_list for l3 in lambda3_list for mu in mu_list]
    total = len(combos)
    if rank == 0:
        print(f"Grid search over {total} combinations (tune_epochs={args.tune_epochs}).")

    os.makedirs(args.out, exist_ok=True)

    # share full combos list to all ranks (so same iteration order)
    # we just iterate in each rank but training uses DistributedSampler; only rank0 will write results & choose best
    best_combo = None
    best_val = float('inf')
    best_combo_dir = None

    train_dataset = WindowDataset(train_sub)

    combo_idx_global = 0
    for (l1, l2, l3, mu) in combos:
        combo_idx_global += 1
        # build name including main train params as requested
        combo_name = f"l1={l1}_l2={l2}_l3={l3}_mu={mu}_hidden={args.hidden}_layers={args.layers}_dropout={args.dropout}_lr={args.lr}_seed={args.seed}"
        combo_dir = Path(args.out) / combo_name
        # skip if exists and val_metrics.json present
        if combo_dir.exists() and (combo_dir / "val_metrics.json").exists():
            if rank == 0:
                print(f"[{combo_idx_global}/{total}] SKIP existing combo {combo_name}")
            # still need to barrier so all processes stay in sync
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            # optionally load val metrics on rank0 to update best record
            if rank == 0:
                try:
                    with open(combo_dir / "val_metrics.json", "r", encoding="utf-8") as f:
                        vm = json.load(f)
                        vpost = float(vm.get("val_post_avg", float('inf')))
                        if vpost < best_val:
                            best_val = vpost
                            best_combo = (l1, l2, l3, mu)
                            best_combo_dir = combo_dir
                except Exception:
                    pass
            continue

        if rank == 0:
            combo_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{combo_idx_global}/{total}] Training combo {combo_name} ...")
        # sync before training
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        t0 = time.time()
        metrics, trained_model, train_log = train_for_combo(l1, l2, l3, mu, train_dataset, val_windows, args, device, rank, world_size, combo_dir)
        t1 = time.time()
        runtime = t1 - t0

        if rank == 0:
            # save validation post stats (already saved in combo_dir by train_for_combo)
            # update summary best
            summary_row = {"lambda1": float(l1), "lambda2": float(l2), "lambda3": float(l3), "mu": float(mu),
                           "val_post_avg": float(metrics["val_post_avg"]), "val_pre_avg": float(metrics["val_pre_avg"]),
                           "best_epoch": int(metrics["best_epoch"]), "epochs_ran": int(metrics["epochs_ran"]), "runtime": runtime,
                           "combo_dir": str(combo_dir)}
            # append to summary file incremental
            summary_csv = Path(args.out) / "summary.csv"
            if summary_csv.exists():
                df = pd.read_csv(summary_csv)
                df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
            else:
                df = pd.DataFrame([summary_row])
                df.to_csv(summary_csv, index=False)


            if metrics["val_post_avg"] < best_val:
                best_val = metrics["val_post_avg"]
                best_combo = (l1, l2, l3, mu)
                best_combo_dir = combo_dir

            print(f"  combo done. val_post_avg={metrics['val_post_avg']:.6f} (runtime {runtime:.1f}s)")

        # sync before next combo
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    # After grid search: rank0 prints best; ensure all ranks wait
    if rank == 0:
        print("\n=== GRID SEARCH COMPLETE ===")
        print(f"Best combo: lambda1,lambda2,lambda3,mu = {best_combo} with val_post_avg = {best_val:.6f}")
        print(f"Best combo dir: {best_combo_dir}")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # If no best combo found, exit
    if best_combo is None:
        if rank == 0:
            print("No best combo found. Exiting.")
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        return

    # Retrain best combo on full train set (train_windows includes val previously held out)
    if rank == 0:
        print("Retraining best combo on full train set (train+val) ...")
    # create dataset for full train
    full_train_dataset = WindowDataset(train_windows)
    # We'll run distributed training again on full_train_dataset with same training routine but without early-stop saving (we keep existing behavior)
    final_combo_name = f"best_final_l1={best_combo[0]}_l2={best_combo[1]}_l3={best_combo[2]}_mu={best_combo[3]}_hidden={args.hidden}_layers={args.layers}_dropout={args.dropout}_lr={args.lr}_seed={args.seed}"
    final_dir = Path(args.out) / final_combo_name
    final_dir.mkdir(parents=True, exist_ok=True)

    # If a saved best model_state exists in best_combo_dir, copy it to final_dir as starting point
    saved_model_path = Path(best_combo_dir) / "model_state.pt" if best_combo_dir is not None else None
    if saved_model_path is not None and saved_model_path.exists():
        # copy to final_dir
        if rank == 0:
            shutil.copy(saved_model_path, final_dir / "init_model_state.pt")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Create model and DDP wrapper
    final_base = LSTMAction(hidden_size=args.hidden, num_layers=args.layers, dropout_rate=args.dropout).to(device)
    if saved_model_path is not None and saved_model_path.exists():
        # load saved state into final_base (map to device)
        map_loc = {"cuda:%d" % 0: f"cuda:{device.index}"} if device.type == "cuda" else "cpu"
        state_dict = torch.load(saved_model_path, map_location=map_loc)
        try:
            final_base.load_state_dict(state_dict)
        except Exception:
            # attempt partial load or key mapping if necessary
            final_base.load_state_dict(state_dict, strict=False)
    # wrap DDP
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        final_base = DDP(final_base, device_ids=[device.index] if device.type == 'cuda' else None)

    # Retrain (distributed)
    final_trainer = E2ETrainer(final_base, m=args.m, lambda1=best_combo[0], lambda2=best_combo[1], lambda3=best_combo[2], mu=best_combo[3], device=device)
    optimizer = optim.Adam(final_base.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # dataloader for full train
    final_sampler = DistributedSampler(WindowDataset(train_windows), num_replicas=world_size, rank=rank) if (dist.is_available() and dist.is_initialized() and world_size > 1) else None
    final_loader = DataLoader(WindowDataset(train_windows), batch_size=args.batch_size, sampler=final_sampler, drop_last=False)

    final_base.train()
    for ep in range(1, args.epochs + 1):
        if final_sampler is not None:
            final_sampler.set_epoch(ep - 1)
        tot = 0.0
        n_batches = 0
        for batch in final_loader:
            for w in batch:
                y_w = w.to(device)
                optimizer.zero_grad()
                pre_mse, post_cost_avg, _, _ = final_trainer.forward_window(final_base, y_w, x0=0.0)
                loss = best_combo[3] * pre_mse + (1.0 - best_combo[3]) * post_cost_avg
                loss.backward()
                optimizer.step()
                tot += float(loss.item())
                n_batches += 1
        if rank == 0 and (ep % max(1, args.epochs // 10) == 0 or ep == 1 or ep == args.epochs):
            print(f"[full retrain ep {ep}/{args.epochs}] avg loss = {tot / float(max(1, n_batches)):.8f}")

    # Save final model state (rank0)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if rank == 0:
        save_state = final_base.module.state_dict() if isinstance(final_base, DDP) else final_base.state_dict()
        torch.save({k: v.cpu() for k, v in save_state.items()}, final_dir / "final_model_state.pt")

    # Final evaluation and plotting done only on rank 0
    if rank == 0:
        device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # load final model into a plain LSTMAction for evaluation
        eval_model = LSTMAction(hidden_size=args.hidden, num_layers=args.layers, dropout_rate=args.dropout).to(device0)
        final_state = torch.load(final_dir / "final_model_state.pt", map_location=device0)
        eval_model.load_state_dict(final_state)
        # E2E (post-calibrated)
        e2e_losses_norm, e2e_seqs = evaluate_post_calibrated_plain(eval_model, test_windows, args.m, best_combo[0], best_combo[1], best_combo[2], device0)
        e2e_losses = [l * scale_sq for l in e2e_losses_norm]
        print(f"E2E (post-calibrated) Test avg loss (mapped back) = {np.mean(e2e_losses):.6f}")

        # ML-only baseline: train a standalone model (not DDP) on full train
        ml_base = LSTMAction(hidden_size=args.hidden, num_layers=args.layers, dropout_rate=args.dropout).to(device0)
        ml_optimizer = optim.Adam(ml_base.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        ml_base.train()
        for ep in range(1, args.epochs + 1):
            tot_ml = 0.0
            n_batches_ml = 0
            for w in train_windows:
                y_w = torch.tensor(w, dtype=torch.float32, device=device0)
                ml_optimizer.zero_grad()
                T = y_w.shape[0]
                prev = torch.tensor(0.0, dtype=torch.float32, device=device0)
                sq = torch.tensor(0.0, dtype=torch.float32, device=device0)
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
                n_batches_ml += 1
            if ep % max(1, args.epochs // 10) == 0 or ep == 1 or ep == args.epochs:
                print(f"[ML-only full train ep {ep}/{args.epochs}] avg pre-mse = {tot_ml / float(max(1, n_batches_ml)):.8f}")

        ml_losses_norm, ml_seqs = evaluate_ml_only_plain(ml_base, test_windows, args.m, device0)
        ml_losses = [l * scale_sq for l in ml_losses_norm]
        print(f"ML-only Test avg loss (mapped back) = {np.mean(ml_losses):.6f}")

        # R-OBD baseline
        robd_losses_raw, robd_seqs = robd_baseline_np(test_windows, args.m)
        robd_losses = [l * scale_sq for l in robd_losses_raw]
        print(f"R-OBD Test avg loss = {np.mean(robd_losses):.6f}")

        # MLA post-hoc tradeoff
        mla_results = posthoc_mla_grid_analysis(ml_seqs, test_windows, args.m, args.theta_grid, scale_sq=scale_sq)
        mla_df = pd.DataFrame(mla_results)
        mla_csv_out, mla_plot_out = save_mla_tradeoff_plot(Path(args.out), mla_df, e2e_losses, robd_losses)

        # objective plots
        hist_out, box_out = save_objective_hist_box(Path(args.out), robd_losses, ml_losses, e2e_losses)
        shutil.copy(hist_out, final_dir / "objective_hist.png")
        shutil.copy(box_out, final_dir / "objective_box.png")
        try:
            shutil.copy(mla_plot_out, final_dir / "mla_tradeoff.png")
        except Exception:
            pass
        mla_df.to_csv(final_dir / "mla_tradeoff.csv", index=False)

        final_summary = {
            "best_combo": {"lambda1": best_combo[0], "lambda2": best_combo[1], "lambda3": best_combo[2], "mu": best_combo[3]},
            "best_val_post_avg": float(best_val),
            "e2e_test_avg_mapped": float(np.mean(e2e_losses)),
            "ml_test_avg_mapped": float(np.mean(ml_losses)),
            "robd_test_avg_mapped": float(np.mean(robd_losses)),
            "final_dir": str(final_dir),
            "summary_csv": str(Path(args.out) / "summary.csv"),
            "mla_tradeoff_csv": str(Path(args.out) / "mla_tradeoff.csv"),
            "mla_tradeoff_png": str(Path(args.out) / "mla_tradeoff.png"),
            "objective_hist": str(hist_out),
            "objective_box": str(box_out)
        }
        with open(Path(args.out) / "final_summary.json", "w", encoding="utf-8") as f:
            json.dump(final_summary, f, indent=2)

        print("All done. Outputs written to:", args.out)

    # finalize distributed
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

# -------------------------
# Argument parsing
# -------------------------
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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lambda1_list", nargs="+", type=float, default=None)
    parser.add_argument("--lambda2_list", nargs="+", type=float, default=None)
    parser.add_argument("--lambda3_list", nargs="+", type=float, default=None)
    parser.add_argument("--mu_list", nargs="+", type=float, default=None)
    parser.add_argument("--theta_grid", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0])
    args = parser.parse_args()

    run_grid_search_and_final(args)
