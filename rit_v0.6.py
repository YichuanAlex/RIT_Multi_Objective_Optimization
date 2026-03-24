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
        feat = y_seq.unsqueeze(0).unsqueeze(-1).contiguous()  # (1, t, 1)
        _, (h_T, _) = self.lstm(feat)
        h = h_T[-1, 0, :]  # (hidden,)
        h = self.dropout(h)
        x_prev_ = x_prev.unsqueeze(0)  # (1,)
        hcat = torch.cat([h, x_prev_], dim=-1)
        out = self.fc(hcat)
        return out.squeeze()

# -------------------------
# E2E Trainer with closed-form MLA-ROBD calibrator
# -------------------------
class E2ETrainer:
    def __init__(self, base_model: LSTMAction, m: float,
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
        """
        Returns:
          pre_mse: avg_t (x_ml - v_t)^2
          post_cost_avg: avg_t [ (m/2)*(x_cal-y_t)^2 + (1/2)*(x_cal - prev)^2 ]
          x_ml_seq: tensor (T,)
          x_cal_seq: tensor (T,)
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
            x_ml = self.base.forward_once(y_hist, prev_cal)  # scalar tensor (requires grad)
            v_t = y_t  # hitting minimizer for squared hitting cost
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
                    args, device) -> Tuple[dict, LSTMAction, list]:
    base = LSTMAction(hidden_size=args.hidden, num_layers=args.layers, dropout_rate=args.dropout).to(device)
    trainer = E2ETrainer(base_model=base, m=args.m,
                         lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, mu=mu, device=device)
    optimizer = optim.Adam(base.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    base.train()
    train_log = []
    best_val_post = float('inf')
    best_epoch = -1
    patience = args.patience
    no_improve = 0
    best_state = None
    for ep in range(1, args.tune_epochs + 1):
        total_loss = 0.0
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
        base.eval()
        with torch.no_grad():
            val_post_list = []
            val_pre_list = []
            for vw in val_windows:
                y_v = torch.tensor(vw, dtype=torch.float32, device=device)
                pre_mse_v, post_cost_v, _, _ = trainer.forward_window(y_v, x0=0.0)
                val_post_list.append(float(post_cost_v.item()))
                val_pre_list.append(float(pre_mse_v.item()))
        val_post_avg = float(np.mean(val_post_list)) if len(val_post_list) > 0 else float('inf')
        val_pre_avg = float(np.mean(val_pre_list)) if len(val_pre_list) > 0 else float('inf')
        base.train()
        train_log.append({"epoch": ep, "train_loss": avg_train_loss, "val_post_avg": val_post_avg, "val_pre_avg": val_pre_avg})
        # early stopping on validation post cost
        if val_post_avg + 1e-12 < best_val_post:
            best_val_post = val_post_avg
            best_epoch = ep
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in base.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state is not None:
        base.load_state_dict(best_state)
    metrics = {
        "val_post_avg": best_val_post,
        "val_pre_avg": val_pre_avg,
        "best_epoch": best_epoch,
        "epochs_ran": ep
    }
    return metrics, base, train_log

# -------------------------
# Evaluation helpers
# -------------------------
def evaluate_post_calibrated(base_model: LSTMAction, windows: List[np.ndarray],
                             m: float, lambda1: float, lambda2: float, lambda3: float, device):
    trainer = E2ETrainer(base_model=base_model, m=m, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, mu=0.0, device=device)
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

def evaluate_ml_only(base_model: LSTMAction, windows: List[np.ndarray], m: float, device):
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
    """
    Compute post-hoc MLA-ROBD results and map them back to original units using scale_sq.
    Returns list of dicts with theta, avg_loss (mapped), worst_loss (mapped).
    """
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
        # map back to original units
        avg_mapped = avg_norm * scale_sq
        worst_mapped = worst_norm * scale_sq
        results.append({"theta": float(theta), "avg_loss": float(avg_mapped), "worst_loss": float(worst_mapped)})
    return results

# -------------------------
# Plot helpers (fixed)
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
    # use tick_labels to avoid deprecation warning
    plt.boxplot([robd_losses, ml_losses, e2e_losses], tick_labels=["R-OBD", "ML-only", "E2E"])
    plt.title("Objective comparison")
    plt.savefig(box_out)
    plt.close()

    return hist_out, box_out

def save_mla_tradeoff_plot(out_dir: Path, mla_df: pd.DataFrame, e2e_losses: List[float], robd_losses: List[float]):
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_out = out_dir / "mla_tradeoff.png"
    csv_out = out_dir / "mla_tradeoff.csv"
    # safe save csv
    mla_df.to_csv(csv_out, index=False)

    # convert to float lists and sanity-check
    try:
        thetas = [float(x) for x in mla_df["theta"].tolist()]
        avg_losses = [float(x) for x in mla_df["avg_loss"].tolist()]
        worst_losses = [float(x) for x in mla_df["worst_loss"].tolist()]
    except Exception as e:
        print("Warning: failed to parse mla_tradeoff dataframe:", e)
        thetas, avg_losses, worst_losses = [], [], []

    print(f"DEBUG: mla_tradeoff lengths: thetas={len(thetas)}, avg={len(avg_losses)}, worst={len(worst_losses)}")

    plt.figure(figsize=(8, 4))
    if len(thetas) > 0 and len(avg_losses) == len(thetas) and len(worst_losses) == len(thetas):
        x = np.arange(len(thetas))
        width = 0.35
        plt.bar(x - width/2, avg_losses, width, label="MLA-ROBD Avg Loss")
        plt.bar(x + width/2, worst_losses, width, label="MLA-ROBD Worst Loss")
        # draw reference lines (these are in mapped-back units)
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
        # fallback: write a placeholder figure showing no data
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
# Grid-run orchestration
# -------------------------
def run_grid_search_and_final(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = pd.read_csv(args.csv)
    values = df.iloc[:, 1].values.astype(float)
    print(f"Loaded {len(values)} data points from {args.csv}")
    values_norm, vmin, vmax = minmax_normalize(values)
    scale_sq = (vmax - vmin) ** 2 if vmax != vmin else 1.0
    print(f"Normalization mapping: vmin={vmin}, vmax={vmax}, scale_sq={scale_sq:.6g}")

    windows = make_sliding_windows(values_norm, window_len=24, step=2)
    train_windows, test_windows = split_windows(windows, train_ratio=0.8)
    print(f"Generated {len(windows)} windows, train={len(train_windows)}, test={len(test_windows)}")

    train_sub, val_windows = train_val_split(train_windows, val_ratio=0.2)

    # refined grid
    lambda1_list = args.lambda1_list if args.lambda1_list is not None else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    lambda2_list = args.lambda2_list if args.lambda2_list is not None else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    lambda3_list = args.lambda3_list if args.lambda3_list is not None else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    mu_list = args.mu_list if args.mu_list is not None else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    combos = [(l1, l2, l3, mu) for l1 in lambda1_list for l2 in lambda2_list for l3 in lambda3_list for mu in mu_list]
    total = len(combos)
    print(f"Grid search over {total} combinations (tune_epochs={args.tune_epochs}).")
    summary_rows = []
    os.makedirs(args.out, exist_ok=True)

    best_combo = None
    best_val = float('inf')
    best_state = None
    combo_idx = 0
    best_combo_dir = None

    for (l1, l2, l3, mu) in combos:
        combo_idx += 1
        combo_name = f"grid_l1={l1}_l2={l2}_l3={l3}_mu={mu}"
        combo_dir = Path(args.out) / combo_name
        combo_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{combo_idx}/{total}] Training combo {combo_name} ...")
        t0 = time.time()
        metrics, trained_base, train_log = train_for_combo(l1, l2, l3, mu, train_sub, val_windows, args, device)
        t1 = time.time()
        runtime = t1 - t0

        # save things
        with open(combo_dir / "train_log.txt", "w", encoding="utf-8") as f:
            for entry in train_log:
                f.write(json.dumps(entry) + "\n")
        val_metrics = {"val_post_avg": metrics["val_post_avg"], "val_pre_avg": metrics["val_pre_avg"],
                       "best_epoch": metrics["best_epoch"], "epochs_ran": metrics["epochs_ran"],
                       "runtime_seconds": runtime}
        with open(combo_dir / "val_metrics.json", "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, indent=2)

        # val post stats
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

        # save model state for this combo (cpu)
        torch.save(trained_base.state_dict(), combo_dir / "model_state.pt")

        summary_rows.append({
            "lambda1": float(l1), "lambda2": float(l2), "lambda3": float(l3), "mu": float(mu),
            "val_post_avg": float(metrics["val_post_avg"]), "val_pre_avg": float(metrics["val_pre_avg"]),
            "best_epoch": int(metrics["best_epoch"]), "epochs_ran": int(metrics["epochs_ran"]), "runtime": runtime,
            "combo_dir": str(combo_dir)
        })

        if metrics["val_post_avg"] < best_val:
            best_val = metrics["val_post_avg"]
            best_combo = (l1, l2, l3, mu)
            best_state = {k: v.cpu().clone() for k, v in trained_base.state_dict().items()}
            best_combo_dir = combo_dir

        print(f"  combo done. val_post_avg={metrics['val_post_avg']:.6f} (runtime {runtime:.1f}s)")

    # summary
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = Path(args.out) / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    best_txt = Path(args.out) / "best_config.txt"
    with open(best_txt, "w", encoding="utf-8") as f:
        f.write(f"best_combo: lambda1,lambda2,lambda3,mu = {best_combo}\n")
        f.write(f"best_val_post_avg = {best_val}\n")
        f.write(f"best_combo_dir = {best_combo_dir}\n")
    print("\n=== GRID SEARCH COMPLETE ===")
    print(f"Best combo: lambda1,lambda2,lambda3,mu = {best_combo} with val_post_avg = {best_val:.6f}")
    print(f"Best combo dir: {best_combo_dir}")

    # Retrain best on full train set (train_windows includes val previously held out)
    if best_combo is None:
        print("No best combo found; exiting.")
        return

    print("Retraining best combo on full train set (train+val) ...")
    final_base = LSTMAction(hidden_size=args.hidden, num_layers=args.layers, dropout_rate=args.dropout).to(device)
    if best_state is not None:
        final_base.load_state_dict(best_state)
    final_trainer = E2ETrainer(final_base, m=args.m, lambda1=best_combo[0], lambda2=best_combo[1], lambda3=best_combo[2], mu=best_combo[3], device=device)
    optimizer = optim.Adam(final_base.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    final_base.train()
    for ep in range(1, args.epochs + 1):
        tot = 0.0
        for w in train_windows:
            y_w = torch.tensor(w, dtype=torch.float32, device=device)
            optimizer.zero_grad()
            pre_mse, post_cost_avg, _, _ = final_trainer.forward_window(y_w, x0=0.0)
            loss = best_combo[3] * pre_mse + (1.0 - best_combo[3]) * post_cost_avg
            loss.backward()
            optimizer.step()
            tot += float(loss.item())
        if ep % max(1, args.epochs // 10) == 0 or ep == 1 or ep == args.epochs:
            print(f"[full retrain ep {ep}/{args.epochs}] avg loss = {tot / float(len(train_windows)):.8f}")

    final_dir = Path(args.out) / f"best_final_l1={best_combo[0]}_l2={best_combo[1]}_l3={best_combo[2]}_mu={best_combo[3]}"
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(final_base.state_dict(), final_dir / "final_model_state.pt")

    # Evaluate on test set (post-calibrated)
    e2e_losses_norm, e2e_seqs = evaluate_post_calibrated(final_base, test_windows, args.m, best_combo[0], best_combo[1], best_combo[2], device)
    e2e_losses = [l * scale_sq for l in e2e_losses_norm]
    test_metrics = {"e2e_test_avg_mapped": float(np.mean(e2e_losses)), "e2e_test_mapped_list": [float(x) for x in e2e_losses]}
    with open(final_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"E2E (post-calibrated) Test avg loss (mapped back) = {np.mean(e2e_losses):.6f}")

    # ML-only baseline (train on same full train set)
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
            print(f"[ML-only full train ep {ep}/{args.epochs}] avg pre-mse = {tot_ml / float(len(train_windows)):.8f}")

    ml_losses_norm, ml_seqs = evaluate_ml_only(ml_base, test_windows, args.m, device)
    ml_losses = [l * scale_sq for l in ml_losses_norm]
    print(f"ML-only Test avg loss (mapped back) = {np.mean(ml_losses):.6f}")

    # R-OBD baseline
    robd_losses_raw, robd_seqs = robd_baseline_np(test_windows, args.m)
    robd_losses = [l * scale_sq for l in robd_losses_raw]
    print(f"R-OBD Test avg loss = {np.mean(robd_losses):.6f}")

    # Post-hoc MLA analysis using ml_seqs -> produce mla_tradeoff.csv and mla_tradeoff.png in outputs root
    mla_results = posthoc_mla_grid_analysis(ml_seqs, test_windows, args.m, args.theta_grid, scale_sq=scale_sq)
    mla_df = pd.DataFrame(mla_results)
    mla_csv_out, mla_plot_out = save_mla_tradeoff_plot(Path(args.out), mla_df, e2e_losses, robd_losses)

    # Save objective hist/box to outputs and copy to final_dir
    hist_out, box_out = save_objective_hist_box(Path(args.out), robd_losses, ml_losses, e2e_losses)
    try:
        shutil.copy(hist_out, final_dir / "objective_hist.png")
    except Exception:
        pass
    try:
        shutil.copy(box_out, final_dir / "objective_box.png")
    except Exception:
        pass
    # copy mla plot as well
    try:
        shutil.copy(mla_plot_out, final_dir / "mla_tradeoff.png")
    except Exception:
        pass
    mla_df.to_csv(final_dir / "mla_tradeoff.csv", index=False)

    # final summary
    final_summary = {
        "best_combo": {"lambda1": best_combo[0], "lambda2": best_combo[1], "lambda3": best_combo[2], "mu": best_combo[3]},
        "best_val_post_avg": float(best_val),
        "e2e_test_avg_mapped": float(np.mean(e2e_losses)),
        "ml_test_avg_mapped": float(np.mean(ml_losses)),
        "robd_test_avg_mapped": float(np.mean(robd_losses)),
        "final_dir": str(final_dir),
        "summary_csv": str(summary_csv),
        "mla_tradeoff_csv": str(mla_csv_out),
        "mla_tradeoff_png": str(mla_plot_out),
        "objective_hist": str(hist_out),
        "objective_box": str(box_out)
    }
    with open(Path(args.out) / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    print("All done. Outputs written to:", args.out)
    return

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
    parser.add_argument("--lambda1_list", nargs="+", type=float, default=None)
    parser.add_argument("--lambda2_list", nargs="+", type=float, default=None)
    parser.add_argument("--lambda3_list", nargs="+", type=float, default=None)
    parser.add_argument("--mu_list", nargs="+", type=float, default=None)
    parser.add_argument("--theta_grid", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 5.0])
    args = parser.parse_args()

    run_grid_search_and_final(args)
