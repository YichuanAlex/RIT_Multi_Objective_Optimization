# python test.py --csv AI_workload.csv --epochs 60 --hidden 64 --layers 1 --lr 1e-3 --out outputs_new

import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ----------------------
# 数据处理工具
# ----------------------
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

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------------
# 基础模块：LSTMAction (unchanged structure, minor fixes)
# ----------------------
class LSTMAction(nn.Module):
    def __init__(self, hidden_size=16, num_layers=1, dropout_rate=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout_rate)
        # concat hidden vector with previous action scalar
        self.fc = nn.Linear(hidden_size + 1, 1)

    def forward_once(self, y_seq: torch.Tensor, x_prev: torch.Tensor):
        """
        y_seq: (t,) tensor, 1D
        x_prev: scalar tensor ()
        returns scalar tensor ()
        """
        # prepare sequence -> (1, t, 1)
        feat = y_seq.unsqueeze(0).unsqueeze(-1).contiguous()
        _, (h_T, _) = self.lstm(feat)  # h_T: (num_layers, batch=1, hidden)
        h_T = h_T[-1, 0, :]  # (hidden_size,)
        h_T = self.dropout(h_T)
        x_prev_ = x_prev.unsqueeze(0)  # (1,)
        h_concat = torch.cat([h_T, x_prev_], dim=-1)  # (hidden + 1)
        x_T = self.fc(h_concat)  # (1,)
        return x_T.squeeze()  # scalar

# ----------------------
# End-to-end model: LSTM + differentiable MLA-ROBD calibrator
# ----------------------
class E2EModel(nn.Module):
    def __init__(self, base_model: LSTMAction, m: float = 5.0, theta_init: float = 1.0, train_theta: bool = True):
        super().__init__()
        self.base = base_model
        self.m = float(m)
        # theta controls trust in ML prediction; make it positive via softplus if trainable
        # We keep a raw parameter r such that theta = softplus(r) to ensure positivity.
        r_init = np.log(np.exp(theta_init) - 1.0) if theta_init > 0 else 0.0
        self.r_theta = nn.Parameter(torch.tensor(float(r_init))) if train_theta else None
        if not train_theta:
            # store a fixed value
            self.theta_fixed = float(theta_init)
        # small epsilon to avoid zero denom
        self.eps = 1e-8

    def theta(self):
        if self.r_theta is not None:
            # softplus to ensure positive trust parameter
            return torch.nn.functional.softplus(self.r_theta)
        else:
            return torch.tensor(self.theta_fixed, dtype=torch.float32, device=next(self.parameters()).device)

    def forward_window(self, y_seq: torch.Tensor, x0: float = 0.0):
        """
        Run forward for a single window (length T).
        y_seq: (T,) tensor normalized in [0,1]
        Returns:
            total_loss (scalar tensor averaged over T), x_seq (T,) tensor
        The computation is fully differentiable and backprop flows through base model and theta (if trainable).
        """
        device = y_seq.device
        T = y_seq.shape[0]
        xs = []
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        prev_x = torch.tensor(x0, dtype=torch.float32, device=device)

        # fetch theta scalar (tensor)
        theta = self.theta().to(device) if self.r_theta is not None else torch.tensor(self.theta_fixed, dtype=torch.float32, device=device)

        denom = (1.0 + self.m + theta)  # tensor or scalar

        for t in range(1, T + 1):
            y_hist = y_seq[:t]
            # ML prediction uses all available y_hist and prev_x (note: prev_x is part of computational graph)
            x_ml = self.base.forward_once(y_hist, prev_x)  # scalar tensor (requires grad)
            # MLA-ROBD style calibrated action (closed-form), fully differentiable
            # x_t = (m*y_t + prev_x + theta * x_ml) / (1 + m + theta)
            y_t = y_seq[t - 1]
            x_t = (self.m * y_t + prev_x + theta * x_ml) / (denom + self.eps)
            # compute cost
            hit = (self.m / 2.0) * ((x_t - y_t) ** 2)
            sw = (1.0 / 2.0) * ((x_t - prev_x) ** 2)
            total_loss = total_loss + (hit + sw)
            xs.append(x_t)
            # IMPORTANT: do NOT detach prev_x - keep graph for backprop
            prev_x = x_t

        total_loss = total_loss / float(T)
        x_seq = torch.stack(xs)  # (T,)
        return total_loss, x_seq

# ----------------------
# Baselines: R-OBD and MLA-ROBD (non-train)
# ----------------------
def robd_rollout(y_seq: np.ndarray, m: float, x0: float = 0.0):
    xs = []
    prev_x = float(x0)
    total_cost = 0.0
    for t in range(len(y_seq)):
        y_t = float(y_seq[t])
        x_t = (m * y_t + prev_x) / (1 + m)
        xs.append(x_t)
        cost = (m / 2.0) * ((x_t - y_t) ** 2) + (1 / 2.0) * ((x_t - prev_x) ** 2)
        total_cost += cost
        prev_x = x_t
    return total_cost / float(len(y_seq)), np.array(xs)

def mla_robd_rollout(y_seq: np.ndarray, h_seq: np.ndarray, m: float, theta: float, x0: float = 0.0):
    # classical post-hoc MLA-ROBD (not train-time)
    y_seq = np.array(y_seq, dtype=float)
    h_seq = np.array(h_seq, dtype=float)
    assert len(y_seq) == len(h_seq)
    xs = []
    prev_x = float(x0)
    total_cost = 0.0
    denom = 1.0 + m + theta
    for t in range(len(y_seq)):
        y_t = float(y_seq[t])
        h_t = float(h_seq[t])
        x_t = (m * y_t + prev_x + theta * h_t) / denom
        xs.append(x_t)
        cost = (m / 2.0) * ((x_t - y_t) ** 2) + (1 / 2.0) * ((x_t - prev_x) ** 2)
        total_cost += cost
        prev_x = x_t
    return total_cost / float(len(y_seq)), np.array(xs)

# ----------------------
# 训练 / 评估工具
# ----------------------
def train_e2e(model: E2EModel, train_windows, optimizer, epochs=20, device="cpu", verbose=True, scheduler=None):
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        count = 0
        for w in train_windows:
            y_w = torch.tensor(w, dtype=torch.float32, device=device)
            optimizer.zero_grad()
            loss, _ = model.forward_window(y_w, x0=0.0)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step(loss.item())
            total_loss += float(loss.item())
            count += 1
        avg = total_loss / max(1, count)
        if verbose:
            print(f"[Epoch {ep+1}] avg window loss = {avg:.8f}")
    return

def evaluate_e2e(model: E2EModel, windows, device="cpu"):
    model.eval()
    losses = []
    x_seqs = []
    with torch.no_grad():
        for w in windows:
            y_w = torch.tensor(w, dtype=torch.float32, device=device)
            loss, x_seq = model.forward_window(y_w, x0=0.0)
            losses.append(float(loss.item()))
            x_seqs.append(x_seq.cpu().numpy())
    return losses, x_seqs

def evaluate_ml_only(base_model: LSTMAction, windows, device="cpu"):
    base_model.eval()
    losses = []
    h_seqs = []
    with torch.no_grad():
        for w in windows:
            y_w = torch.tensor(w, dtype=torch.float32, device=device)
            T = y_w.shape[0]
            prev_x = torch.tensor(0.0, dtype=torch.float32, device=device)
            xs = []
            total_loss = 0.0
            for t in range(1, T + 1):
                y_hist = y_w[:t]
                x_t = base_model.forward_once(y_hist, prev_x)
                hit = (5.0 / 2.0) * ((x_t - y_w[t - 1]) ** 2)  # note: use m externally (we assume m=5 by default here)
                sw = (1.0 / 2.0) * ((x_t - prev_x) ** 2)
                total_loss += (hit + sw)
                xs.append(x_t.cpu().numpy())
                prev_x = x_t
            losses.append(float(total_loss / float(T)))
            h_seqs.append(np.array(xs))
    return losses, h_seqs

# ----------------------
# 主流程
# ----------------------
def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # load csv: assume second column is the power values (same as your original)
    df = pd.read_csv(args.csv)
    values = df.iloc[:, 1].values.astype(float)
    print(f"Loaded {len(values)} data points from {args.csv}")

    # normalize full dataset 0-1
    values_norm, vmin, vmax = minmax_normalize(values)
    scale_sq = (vmax - vmin) ** 2 if vmax != vmin else 1.0

    # sliding windows
    windows = make_sliding_windows(values_norm, window_len=24, step=2)
    train_windows, test_windows = split_windows(windows, train_ratio=0.8)
    print(f"Generated {len(windows)} windows, train={len(train_windows)}, test={len(test_windows)}")

    # shuffle train windows
    np.random.shuffle(train_windows)

    # build base LSTM
    base_model = LSTMAction(hidden_size=args.hidden, num_layers=args.layers, dropout_rate=args.dropout).to(device)

    # build E2E model: here theta_init default from args
    e2e = E2EModel(base_model=base_model, m=args.m, theta_init=args.theta_init, train_theta=bool(args.train_theta)).to(device)

    # optimizer: include theta param if trainable
    opt_params = list(e2e.parameters())
    optimizer = optim.Adam(opt_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None

    # training
    if args.tune:
        # optional small grid search for hyperparams - but keep simple to avoid complexity
        print("Tuning is on but automated grid search disabled in this script; proceed with given hyperparams.")
    print("Start training end-to-end model...")
    train_e2e(e2e, train_windows, optimizer, epochs=args.epochs, device=device, verbose=True, scheduler=scheduler)

    # evaluate E2E
    e2e_losses_norm, e2e_seqs = evaluate_e2e(e2e, test_windows, device=device)
    e2e_losses = [l * scale_sq for l in e2e_losses_norm]
    print(f"E2E Model Test avg loss (mapped back) = {np.mean(e2e_losses):.6f}")

    # evaluate ML-only: use the base model part (but note base_model has been trained as part of e2e)
    ml_losses_norm, ml_seqs = evaluate_e2e(e2e, test_windows, device=device)  # NOTE: evaluating same forward gives calibrated x
    # To get pure ML predictions (without calibrator), we can run base model separately in eval mode
    ml_only_losses = []
    ml_only_preds = []
    with torch.no_grad():
        for w in test_windows:
            y_w = torch.tensor(w, dtype=torch.float32, device=device)
            T = y_w.shape[0]
            prev_x = torch.tensor(0.0, dtype=torch.float32, device=device)
            total_loss = 0.0
            preds = []
            for t in range(1, T + 1):
                y_hist = y_w[:t]
                x_t = base_model.forward_once(y_hist, prev_x)
                hit = (args.m / 2.0) * ((x_t - y_w[t - 1]) ** 2)
                sw = (1.0 / 2.0) * ((x_t - prev_x) ** 2)
                total_loss += (hit + sw)
                preds.append(float(x_t.cpu().numpy()))
                prev_x = x_t
            ml_only_losses.append(float((total_loss / float(T)).cpu().numpy()))
            ml_only_preds.append(np.array(preds))
    ml_only_losses = [l * scale_sq for l in ml_only_losses]
    print(f"ML-only Test avg loss (mapped back) = {np.mean(ml_only_losses):.6f}")

    # evaluate classical R-OBD baseline
    robd_losses = []
    robd_preds = []
    for w in test_windows:
        loss_norm, xs = robd_rollout(w, args.m, x0=0.0)
        robd_losses.append(loss_norm * scale_sq)
        robd_preds.append(xs)
    print(f"R-OBD Test avg loss = {np.mean(robd_losses):.6f}")

    # MLA-ROBD (post-hoc) using ML-only predictions and a theta grid (for comparison)
    theta_grid = args.theta_grid
    mla_results = []
    # use ml_only_preds as ML predictions (aligned per window)
    for theta in theta_grid:
        losses = []
        for w, h in zip(test_windows, ml_only_preds):
            loss_norm, _ = mla_robd_rollout(w, h, args.m, theta, x0=0.0)
            losses.append(loss_norm * scale_sq)
        avg_loss = np.mean(losses)
        worst_loss = np.max(losses)
        mla_results.append({"theta": theta, "avg_loss": float(avg_loss), "worst_loss": float(worst_loss)})
        print(f"MLA-ROBD(post) theta={theta} avg={avg_loss:.6f} worst={worst_loss:.6f}")

    # 保存 MLA 网格结果
    mla_df = pd.DataFrame(mla_results)
    os.makedirs(args.out, exist_ok=True)
    mla_df.to_csv(os.path.join(args.out, "mla_tradeoff.csv"), index=False)

    # 保存图像：对比 R-OBD、ML-only、E2E
    plt.figure()
    plt.hist([robd_losses, ml_only_losses, e2e_losses], label=["R-OBD", "ML-only", "E2E"], bins=20)
    plt.legend()
    plt.title("Objective distribution on test windows")
    plt.savefig(os.path.join(args.out, "objective_hist.png"))

    plt.figure()
    plt.boxplot([robd_losses, ml_only_losses, e2e_losses], labels=["R-OBD", "ML-only", "E2E"])
    plt.title("Objective comparison")
    plt.savefig(os.path.join(args.out, "objective_box.png"))

    # MLA tradeoff bar chart
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="CSV file (power). Second column used as values.")
    parser.add_argument("--m", type=float, default=5.0, help="m for hitting cost")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--out", type=str, default="outputs")
    parser.add_argument("--tune", type=int, default=0, help="placeholder (not used)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--theta_init", type=float, default=1.0, help="initial theta for E2E calibrator")
    parser.add_argument("--train_theta", type=int, default=1, help="1 to train theta, 0 to keep fixed")
    parser.add_argument("--theta_grid", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0, 5.0], help="grid for post-hoc MLA-ROBD eval")
    args = parser.parse_args()
    main(args)
