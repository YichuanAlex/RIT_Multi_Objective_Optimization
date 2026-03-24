#!/usr/bin/env python3
# e2e_mla_robd_tuned.py
"""
End-to-end EC-L2O style implementation with MLA-ROBD closed-form calibrator,
combined training loss (mu * prediction_error + (1-mu) * post-calibration cost),
and a small automated hyperparameter grid search to tune lambda1/lambda2/lambda3/mu.

python test.py --csv AI_workload.csv --epochs 60 --hidden 64 --layers 1 --lr 1e-3 --out outputs
"""

import argparse
import os
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# -----------------------
# Utilities and data prep
# -----------------------
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

def make_sliding_windows(values, window_len=24, step=2):
    xs = []
    i = 0
    while i + window_len <= len(values):
        xs.append(values[i:i + window_len])
        i += step
    return np.array(xs)

def split_windows(windows, train_ratio=0.8):
    n = len(windows)
    n_train = int(n * train_ratio)
    return windows[:n_train], windows[n_train:]

def train_val_split(windows, val_ratio=0.2):
    n = len(windows)
    n_val = max(1, int(n * val_ratio))
    return windows[:-n_val], windows[-n_val:]

# -----------------------
# Model: LSTMAction
# -----------------------
class LSTMAction(nn.Module):
    def __init__(self, hidden_size=16, num_layers=1, dropout_rate=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size + 1, 1)

    def forward_once(self, y_seq: torch.Tensor, x_prev: torch.Tensor):
        # y_seq: (t,), x_prev: scalar tensor
        feat = y_seq.unsqueeze(0).unsqueeze(-1).contiguous()  # (1,t,1)
        _, (h_T, _) = self.lstm(feat)  # h_T: (num_layers, 1, hidden)
        h = h_T[-1, 0, :]  # (hidden,)
        h = self.dropout(h)
        x_prev_ = x_prev.unsqueeze(0)  # (1,)
        h_cat = torch.cat([h, x_prev_], dim=-1)  # (hidden+1)
        x_out = self.fc(h_cat)  # (1,)
        return x_out.squeeze()  # scalar

# -----------------------
# Closed-form MLA-ROBD calibrator (per-step)
# For quadratic hitting f(x,y) = (m/2)*(x-y)^2 and switching c = (1/2)*(x - prev)^2
# The minimization over x of: f + lambda1*c(x,prev) + lambda2*c(x,v) + lambda3*c(x,x_ml)
# yields closed form:
# x = (m*y + lambda1*prev + lambda2*v + lambda3*x_ml) / (m + lambda1 + lambda2 + lambda3)
# (here c coefficients use 1/2 normalization consistent with switching cost used)
# -----------------------
# We'll use this closed-form across training: fully differentiable (linear ops)

# -----------------------
# End-to-end EC-L2O training routine (with hyperparams)
# -----------------------
class E2ETrainer:
    def __init__(self, base_model: LSTMAction, m: float = 5.0,
                 lambda1: float = 1.0, lambda2: float = 0.0, lambda3: float = 1.0,
                 mu: float = 0.2, device='cpu'):
        """
        base_model: LSTMAction (PyTorch Module)
        lambdas: calibration weights
        mu: weight for pre-calibration prediction error in total loss
        """
        self.device = device
        self.base = base_model.to(device)
        self.m = float(m)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.lambda3 = float(lambda3)
        self.mu = float(mu)
        # denom constant (scalar)
        self.eps = 1e-8

    def forward_window(self, y_seq: torch.Tensor, x0: float = 0.0, train_mode: bool = True):
        """
        Run through one window (T steps), return:
          - pre_mse: sum over t of (x_ml - v_t)^2 / T   (prediction error term)
          - post_cost: sum over t of (hitting + switching) / T   (post-calibration average cost)
          - sequences: x_ml_seq, x_cal_seq (torch tensors)
        """
        device = y_seq.device
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
            # ML prediction (depends on prev_cal to capture recurrence)
            x_ml = self.base.forward_once(y_hist, prev_cal)
            # hitting-cost minimizer v_t for (m/2)*(x - y)^2 is v_t = y_t
            v_t = y_t
            # closed-form calibrator:
            numerator = (self.m * y_t) + (self.lambda1 * prev_cal) + (self.lambda2 * v_t) + (self.lambda3 * x_ml)
            x_cal = numerator / denom
            # accumulate pre error (proxy: distance to v_t)
            pre_sq = pre_sq + (x_ml - v_t) ** 2
            # accumulate post cost: hitting + switching
            hit = (self.m / 2.0) * ((x_cal - y_t) ** 2)
            sw = (1.0 / 2.0) * ((x_cal - prev_cal) ** 2)
            post_cost = post_cost + (hit + sw)
            # save
            x_mls.append(x_ml)
            x_cals.append(x_cal)
            # recurrent update: prev_cal is calibrated action for next input
            prev_cal = x_cal

        pre_mse = pre_sq / float(T)
        post_cost_avg = post_cost / float(T)
        x_ml_seq = torch.stack(x_mls)  # (T,)
        x_cal_seq = torch.stack(x_cals)
        return pre_mse, post_cost_avg, x_ml_seq, x_cal_seq

# -----------------------
# Training & evaluation helpers
# -----------------------
def train_e2e_with_params(base_model: LSTMAction, train_windows, val_windows, m,
                          lambda1, lambda2, lambda3, mu, device, epochs, lr, weight_decay, verbose=True):
    """
    Train E2E model with specified hyperparameters for lambda1/2/3 and mu.
    Return validation avg post-calibration loss (mapped back) for selection, and trained model.
    """
    trainer = E2ETrainer(base_model, m=m, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, mu=mu, device=device)
    # We will train parameters of base_model only (trainer holds lambdas fixed)
    model = trainer.base
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # training loop
    for ep in range(epochs):
        total_loss = 0.0
        for w in train_windows:
            y_w = torch.tensor(w, dtype=torch.float32, device=device)
            optimizer.zero_grad()
            pre_mse, post_cost_avg, _, _ = trainer.forward_window(y_w, x0=0.0, train_mode=True)
            loss = mu * pre_mse + (1.0 - mu) * post_cost_avg
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        if verbose and ((ep + 1) % max(1, epochs // 10) == 0 or ep == 0 or ep == epochs - 1):
            print(f"  [ep {ep+1}/{epochs}] train loss avg = {total_loss / float(len(train_windows)):.8f}")
    # evaluate on validation windows: compute post_cost_avg
    model.eval()
    with torch.no_grad():
        val_losses = []
        val_pre = []
        for w in val_windows:
            y_w = torch.tensor(w, dtype=torch.float32, device=device)
            pre_mse, post_cost_avg, _, _ = trainer.forward_window(y_w, x0=0.0, train_mode=False)
            val_losses.append(float(post_cost_avg.item()))
            val_pre.append(float(pre_mse.item()))
    val_avg = float(np.mean(val_losses)) if len(val_losses) > 0 else float('inf')
    val_pre_avg = float(np.mean(val_pre)) if len(val_pre) > 0 else float('inf')
    return model, val_avg, val_pre_avg

def evaluate_post_calibrated(base_model: LSTMAction, windows, m, lambda1, lambda2, lambda3, device):
    trainer = E2ETrainer(base_model, m=m, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, mu=0.0, device=device)
    trainer.base = base_model  # use provided
    trainer.base.eval()
    losses = []
    sequences = []
    with torch.no_grad():
        for w in windows:
            y_w = torch.tensor(w, dtype=torch.float32, device=device)
            _, post_cost_avg, _, x_cal_seq = trainer.forward_window(y_w, x0=0.0, train_mode=False)
            losses.append(float(post_cost_avg.item()))
            sequences.append(x_cal_seq.cpu().numpy())
    return losses, sequences

def evaluate_ml_only(base_model: LSTMAction, windows, m, device):
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
                # treat final action as ML-only action (no calibration)
                hit = (m / 2.0) * ((x_ml - y_w[t - 1]) ** 2)
                sw = (1.0 / 2.0) * ((x_ml - prev) ** 2)
                total = total + (hit + sw)
                xs.append(float(x_ml.cpu().numpy()))
                prev = x_ml
            losses.append(float((total / float(T)).cpu().numpy()))
            seqs.append(np.array(xs))
    return losses, seqs

def robd_rollout_np(y_seq: np.ndarray, m: float, x0: float = 0.0):
    xs = []
    prev_x = float(x0)
    total_cost = 0.0
    for t in range(len(y_seq)):
        y_t = float(y_seq[t])
        x_t = (m * y_t + prev_x) / (1.0 + m)
        xs.append(x_t)
        cost = (m / 2.0) * ((x_t - y_t) ** 2) + (1.0 / 2.0) * ((x_t - prev_x) ** 2)
        total_cost += cost
        prev_x = x_t
    return total_cost / float(len(y_seq)), np.array(xs)

def mla_robd_posthoc(y_seq: np.ndarray, h_seq: np.ndarray, m: float, theta: float, x0: float = 0.0):
    # implement special case where lambda1=1, lambda2=0, lambda3=theta equivalent to earlier post-hoc
    y_seq = np.array(y_seq, dtype=float)
    h_seq = np.array(h_seq, dtype=float)
    xs = []
    prev_x = float(x0)
    total_cost = 0.0
    denom = 1.0 + m + theta
    for t in range(len(y_seq)):
        y_t = float(y_seq[t])
        h_t = float(h_seq[t])
        x_t = (m * y_t + prev_x + theta * h_t) / denom
        xs.append(x_t)
        cost = (m / 2.0) * ((x_t - y_t) ** 2) + (1.0 / 2.0) * ((x_t - prev_x) ** 2)
        total_cost += cost
        prev_x = x_t
    return total_cost / float(len(y_seq)), np.array(xs)

# -----------------------
# Hyperparameter grid search (small)
# -----------------------
def small_grid_search(train_windows, val_windows, device, args):
    """
    Try small grid over lambda1, lambda2, lambda3, mu to pick best based on validation post-calibration avg loss.
    Returns best params and trained base model on full train (train+val) with those params.
    """
    # candidate grids (small and pragmatic)
    lambda1_list = args.lambda1_list if args.lambda1_list is not None else [0.5, 1.0, 2.0]
    lambda2_list = args.lambda2_list if args.lambda2_list is not None else [0.0, 0.2, 0.5]
    lambda3_list = args.lambda3_list if args.lambda3_list is not None else [0.0, 0.5, 1.0, 2.0]
    mu_list = args.mu_list if args.mu_list is not None else [0.0, 0.2, 0.5]

    best = None
    best_val = float('inf')
    best_details = None

    # use small epochs for tuning to save time, then final full train
    tune_epochs = max(10, int(args.epochs // 4))

    total_trials = len(lambda1_list) * len(lambda2_list) * len(lambda3_list) * len(mu_list)
    trial = 0
    print(f"Grid search over {total_trials} combinations (epochs={tune_epochs} each).")

    for lam1 in lambda1_list:
        for lam2 in lambda2_list:
            for lam3 in lambda3_list:
                for mu in mu_list:
                    trial += 1
                    print(f"[Grid {trial}/{total_trials}] lam1={lam1}, lam2={lam2}, lam3={lam3}, mu={mu}")
                    # new base model instance
                    base = LSTMAction(hidden_size=args.hidden, num_layers=args.layers, dropout_rate=args.dropout).to(device)
                    model, val_avg, val_pre = train_e2e_with_params(base, train_windows, val_windows, args.m,
                                                                   lam1, lam2, lam3, mu, device,
                                                                   epochs=tune_epochs, lr=args.lr, weight_decay=args.weight_decay, verbose=False)
                    print(f"  -> val post-calib avg = {val_avg:.6f}, val pre-mse avg = {val_pre:.6f}")
                    if val_avg < best_val:
                        best_val = val_avg
                        best = (lam1, lam2, lam3, mu)
                        best_details = (model, val_avg, val_pre)
    print(f"Best grid params: lambda1,lambda2,lambda3,mu = {best} with val post-calib avg={best_val:.6f}")
    return best, best_details

# -----------------------
# Full main routine
# -----------------------
def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = pd.read_csv(args.csv)
    values = df.iloc[:, 1].values.astype(float)
    print(f"Loaded {len(values)} data points from {args.csv}")

    values_norm, vmin, vmax = minmax_normalize(values)
    scale_sq = (vmax - vmin) ** 2 if vmax != vmin else 1.0

    windows = make_sliding_windows(values_norm, window_len=24, step=2)
    train_windows, test_windows = split_windows(windows, train_ratio=0.8)
    print(f"Generated {len(windows)} windows, train={len(train_windows)}, test={len(test_windows)}")
    # create train/val split for grid search
    train_sub, val_windows = train_val_split(train_windows, val_ratio=0.2)

    np.random.shuffle(train_sub)

    # grid search tuning (if enabled)
    if args.tune:
        best_params, best_details = small_grid_search(train_sub, val_windows, device, args)
        lam1, lam2, lam3, mu = best_params
        # best_details contains a model trained on small epochs; we will now fully train on full train set (train_windows)
        print("Re-training best config on full train set for final model ...")
        base_model = LSTMAction(hidden_size=args.hidden, num_layers=args.layers, dropout_rate=args.dropout).to(device)
        # full train using full epochs
        trainer = E2ETrainer(base_model, m=args.m, lambda1=lam1, lambda2=lam2, lambda3=lam3, mu=mu, device=device)
        optimizer = optim.Adam(base_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        base_model.train()
        for ep in range(args.epochs):
            total_loss = 0.0
            for w in train_windows:
                y_w = torch.tensor(w, dtype=torch.float32, device=device)
                optimizer.zero_grad()
                pre_mse, post_cost_avg, _, _ = trainer.forward_window(y_w, x0=0.0, train_mode=True)
                loss = mu * pre_mse + (1.0 - mu) * post_cost_avg
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            if (ep + 1) % max(1, args.epochs // 10) == 0 or ep == 0:
                print(f"[full train ep {ep+1}/{args.epochs}] train loss avg = {total_loss / float(len(train_windows)):.8f}")
        final_base = base_model
        final_lam1, final_lam2, final_lam3, final_mu = lam1, lam2, lam3, mu
    else:
        # no tuning: use provided args
        final_base = LSTMAction(hidden_size=args.hidden, num_layers=args.layers, dropout_rate=args.dropout).to(device)
        # train with given params
        final_lam1, final_lam2, final_lam3, final_mu = args.lambda1, args.lambda2, args.lambda3, args.mu
        trainer = E2ETrainer(final_base, m=args.m, lambda1=final_lam1, lambda2=final_lam2, lambda3=final_lam3, mu=final_mu, device=device)
        optimizer = optim.Adam(final_base.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        final_base.train()
        for ep in range(args.epochs):
            total_loss = 0.0
            for w in train_windows:
                y_w = torch.tensor(w, dtype=torch.float32, device=device)
                optimizer.zero_grad()
                pre_mse, post_cost_avg, _, _ = trainer.forward_window(y_w, x0=0.0, train_mode=True)
                loss = final_mu * pre_mse + (1.0 - final_mu) * post_cost_avg
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            if (ep + 1) % max(1, args.epochs // 10) == 0 or ep == 0:
                print(f"[ep {ep+1}/{args.epochs}] train loss avg = {total_loss / float(len(train_windows)):.8f}")

    # Evaluate final E2E (post-calibrated actions)
    e2e_losses_norm, e2e_seqs = evaluate_post_calibrated(final_base, test_windows, args.m, final_lam1, final_lam2, final_lam3, device)
    e2e_losses = [l * scale_sq for l in e2e_losses_norm]
    print(f"E2E (post-calibrated) Test avg loss (mapped back) = {np.mean(e2e_losses):.6f}")

    # Evaluate ML-only (train base model as standalone for fair baseline)
    # We'll train a separate ML-only model on same train set but minimizing pre-calibration cost (MSE to v_t)
    ml_base = LSTMAction(hidden_size=args.hidden, num_layers=args.layers, dropout_rate=args.dropout).to(device)
    ml_optimizer = optim.Adam(ml_base.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ml_base.train()
    # train ml-only for same epochs
    for ep in range(args.epochs):
        total = 0.0
        for w in train_windows:
            y_w = torch.tensor(w, dtype=torch.float32, device=device)
            ml_optimizer.zero_grad()
            # compute pre-mse per-window: average over t of (x_ml - v_t)^2
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
            total += float(loss_ml.item())
        if (ep + 1) % max(1, args.epochs // 10) == 0 or ep == 0:
            print(f"[ML-only train ep {ep+1}/{args.epochs}] avg pre-mse = {total / float(len(train_windows)):.8f}")

    ml_losses_norm, ml_seqs = evaluate_ml_only(ml_base, test_windows, args.m, device)
    ml_losses = [l * scale_sq for l in ml_losses_norm]
    print(f"ML-only Test avg loss (mapped back) = {np.mean(ml_losses):.6f}")

    # R-OBD baseline
    robd_losses = []
    robd_seqs = []
    for w in test_windows:
        loss_norm, seq = robd_rollout_np(w, args.m, x0=0.0)
        robd_losses.append(loss_norm * scale_sq)
        robd_seqs.append(seq)
    print(f"R-OBD Test avg loss = {np.mean(robd_losses):.6f}")

    # Post-hoc MLA-ROBD grid (using ml_seqs as ML predictions)
    theta_grid = args.theta_grid
    mla_results = []
    for theta in theta_grid:
        losses = []
        for w, h in zip(test_windows, ml_seqs):
            loss_norm, _ = mla_robd_posthoc(w, h, args.m, theta, x0=0.0)
            losses.append(loss_norm * scale_sq)
        avg_loss = np.mean(losses)
        worst_loss = np.max(losses)
        mla_results.append({"theta": float(theta), "avg_loss": float(avg_loss), "worst_loss": float(worst_loss)})
        print(f"MLA-ROBD(post) theta={theta} avg={avg_loss:.6f} worst={worst_loss:.6f}")

    mla_df = pd.DataFrame(mla_results)
    os.makedirs(args.out, exist_ok=True)
    mla_df.to_csv(os.path.join(args.out, "mla_tradeoff.csv"), index=False)

    # Save plots
    plt.figure()
    plt.hist([robd_losses, ml_losses, e2e_losses], label=["R-OBD", "ML-only", "E2E"], bins=20)
    plt.legend()
    plt.title("Objective distribution on test windows")
    plt.savefig(os.path.join(args.out, "objective_hist.png"))

    plt.figure()
    plt.boxplot([robd_losses, ml_losses, e2e_losses], labels=["R-OBD", "ML-only", "E2E"])
    plt.title("Objective comparison")
    plt.savefig(os.path.join(args.out, "objective_box.png"))

    # MLA bar
    plt.figure(figsize=(8,4))
    thetas = mla_df["theta"].values
    avg_losses = mla_df["avg_loss"].values
    worst_losses = mla_df["worst_loss"].values
    x = np.arange(len(thetas))
    width = 0.35
    plt.bar(x - width/2, avg_losses, width, label="MLA-ROBD Avg Loss")
    plt.bar(x + width/2, worst_losses, width, label="MLA-ROBD Worst Loss")
    plt.axhline(y=np.mean(e2e_losses), color='r', linestyle='--', label='E2E Avg Loss')
    plt.axhline(y=np.mean(robd_losses), color='g', linestyle='--', label='R-OBD Avg Loss')
    plt.xticks(x, [str(t) for t in thetas])
    plt.xlabel("Theta")
    plt.ylabel("Loss")
    plt.title("MLA-ROBD Trade-off (post-hoc)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "mla_tradeoff.png"))

    print(f"Saved results and plots to {args.out}")
    print("Done.")

# -----------------------
# Argument parsing
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--m", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--out", type=str, default="outputs")
    parser.add_argument("--tune", type=int, default=1, help="1 to run small grid search tuning")
    parser.add_argument("--seed", type=int, default=42)
    # default lambdas (used when not tuning)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=0.0)
    parser.add_argument("--lambda3", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=0.2)
    # small grid options (optional override)
    parser.add_argument("--lambda1_list", nargs="+", type=float, default=None)
    parser.add_argument("--lambda2_list", nargs="+", type=float, default=None)
    parser.add_argument("--lambda3_list", nargs="+", type=float, default=None)
    parser.add_argument("--mu_list", nargs="+", type=float, default=None)
    parser.add_argument("--theta_grid", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0, 5.0])
    args = parser.parse_args()
    main(args)
