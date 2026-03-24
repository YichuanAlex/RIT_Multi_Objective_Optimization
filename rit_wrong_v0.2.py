import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import random


# ========= 数据处理 =========
def minmax_normalize(values: np.ndarray):
    """
    将整个数据集的功耗序列做 0-1 归一化，并返回归一化后的序列及 (min, max)。
    若所有值相同，则返回全零序列并保留原始 min/max。
    """
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
    n_val = int(n * val_ratio)
    return windows[:-n_val], windows[-n_val:]


def set_seed(seed: int = 42):
    """设置随机种子，保证搜索结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ========= 模型 =========
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
        self.fc = nn.Linear(hidden_size + 1, 1)

    def forward_once(self, y_seq: torch.Tensor, x_prev: torch.Tensor):
        feat = y_seq.unsqueeze(0).unsqueeze(-1)  # (1, t, 1)
        _, (h_T, _) = self.lstm(feat)
        h_T = h_T[-1, 0, :]  # (hidden_size,)
        h_T = self.dropout(h_T)
        h_concat = torch.cat([h_T, x_prev.unsqueeze(0)], dim=-1)
        x_T = self.fc(h_concat)
        return x_T.squeeze()


# ========= rollout =========
def rollout_objective(model: nn.Module, y_seq: torch.Tensor, m: float, x0: float):
    T = len(y_seq)
    xs = []
    total_loss = 0.0
    prev_x = torch.tensor(x0, dtype=torch.float32, device=y_seq.device)

    for t in range(1, T + 1):
        y_hist = y_seq[:t]
        x_t = model.forward_once(y_hist, prev_x)
        xs.append(x_t)
        cost = (m / 2.0) * ((x_t - y_seq[t - 1]) ** 2) + (1 / 2.0) * ((x_t - prev_x) ** 2)
        total_loss += cost
        prev_x = x_t.detach()

    return total_loss / T, torch.stack(xs)


def robd_rollout(y_seq: np.ndarray, m: float, x0: float = 0.0):
    xs = []
    prev_x = x0
    total_cost = 0.0
    for t in range(len(y_seq)):
        y_t = y_seq[t]
        x_t = (m * y_t + prev_x) / (1 + m)
        xs.append(x_t)
        cost = (m / 2.0) * ((x_t - y_t) ** 2) + (1 / 2.0) * ((x_t - prev_x) ** 2)
        total_cost += cost
        prev_x = x_t
    return total_cost / len(y_seq), np.array(xs)


# ========= MLA-ROBD =========
def mla_robd_rollout(y_seq: np.ndarray, h_seq: np.ndarray, m: float, theta: float, x0: float = 0.0):
    y_seq = np.array(y_seq, dtype=float)
    h_seq = np.array(h_seq, dtype=float)
    assert len(y_seq) == len(h_seq)

    xs = []
    prev_x = x0
    total_cost = 0.0
    denom = 1 + m + theta
    for t in range(len(y_seq)):
        y_t = y_seq[t]
        h_t = h_seq[t]
        x_t = (m * y_t + prev_x + theta * h_t) / denom
        xs.append(x_t)
        cost = (m / 2.0) * ((x_t - y_t) ** 2) + (1 / 2.0) * ((x_t - prev_x) ** 2)
        total_cost += cost
        prev_x = x_t
    return total_cost / len(y_seq), np.array(xs)


# ========= EC-L2O 校准器（可微分） =========
def calibrator_step_torch(y_t: torch.Tensor, prev_x: torch.Tensor, h_t: torch.Tensor, m: float, theta: float):
    """单步可微分校准器，将ML预测 h_t 与专家项结合，输出校准后的动作 x_t。"""
    denom = 1.0 + m + theta
    return (m * y_t + prev_x + theta * h_t) / denom


def rollout_objective_calibrated(model: nn.Module, y_seq: torch.Tensor, m: float, x0: float, theta: float, gamma: float = 0.0):
    """
    EC-L2O 端到端目标：在校准器输出的动作上计算运行+切换成本，并可选对 (x_t - h_t) 加偏离惩罚。
    返回：平均损失、校准后的动作序列 xs、ML原始预测序列 hs。
    """
    T = len(y_seq)
    xs = []
    hs = []
    total_loss = 0.0
    prev_x = torch.tensor(x0, dtype=torch.float32, device=y_seq.device)

    for t in range(1, T + 1):
        y_t = y_seq[t - 1]
        y_hist = y_seq[:t]
        h_t = model.forward_once(y_hist, prev_x)
        x_t = calibrator_step_torch(y_t, prev_x, h_t, m=m, theta=theta)
        xs.append(x_t)
        hs.append(h_t)
        run_cost = (m / 2.0) * ((x_t - y_t) ** 2)
        switch_cost = (1 / 2.0) * ((x_t - prev_x) ** 2)
        bias_cost = (gamma / 2.0) * ((x_t - h_t) ** 2) if gamma > 0.0 else 0.0
        total_loss += (run_cost + switch_cost + bias_cost)
        prev_x = x_t.detach()

    return total_loss / T, torch.stack(xs), torch.stack(hs)


def train_on_windows_calibrated(model, windows, m, x0, optimizer, theta: float, gamma: float = 0.0, scheduler=None, epochs=10, device="cpu", verbose=True):
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for w in windows:
            y_w = torch.tensor(w, dtype=torch.float32, device=device)
            loss, _, _ = rollout_objective_calibrated(model, y_w, m, x0, theta=theta, gamma=gamma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(windows)
        if scheduler is not None:
            scheduler.step(avg_loss)
        if verbose:
            print(f"[Calib Epoch {ep+1}] avg loss per window={avg_loss:.8f}")


def evaluate_on_windows_calibrated(model, windows, m, x0, theta: float, gamma: float = 0.0, device="cpu"):
    model.eval()
    losses = []
    x_seqs = []
    h_seqs = []
    with torch.no_grad():
        for w in windows:
            y_w = torch.tensor(w, dtype=torch.float32, device=device)
            loss, x_seq, h_seq = rollout_objective_calibrated(model, y_w, m, x0, theta=theta, gamma=gamma)
            losses.append(loss.item())
            x_seqs.append(x_seq.cpu().numpy())
            h_seqs.append(h_seq.cpu().numpy())
    return losses, x_seqs, h_seqs


# ========= 训练/测试 =========
def train_on_windows(model, windows, m, x0, optimizer, scheduler=None, epochs=10, device="cpu", verbose=True):
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for w in windows:
            y_w = torch.tensor(w, dtype=torch.float32, device=device)
            loss, _ = rollout_objective(model, y_w, m, x0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(windows)
        if scheduler is not None:
            scheduler.step(avg_loss)
        if verbose:
            print(f"[Epoch {ep+1}] avg loss per window={avg_loss:.8f}")


def evaluate_on_windows(model, windows, m, x0, device="cpu"):
    model.eval()
    losses = []
    h_seqs = []
    with torch.no_grad():
        for w in windows:
            y_w = torch.tensor(w, dtype=torch.float32, device=device)
            loss, x_seq = rollout_objective(model, y_w, m, x0)
            losses.append(loss.item())
            h_seqs.append(x_seq.cpu().numpy())
    return losses, h_seqs


# ========= 网格搜索 =========
def grid_search(train_windows, val_windows, m, device, search_space, epochs=50, scale_sq=None, use_scheduler=False):
    best_params = None
    best_val_loss = float("inf")

    for lr in search_space["lr"]:
        for hidden in search_space["hidden"]:
            for layers in search_space["layers"]:
                for dropout in search_space.get("dropout", [0.2]):
                    for weight_decay in search_space.get("weight_decay", [0.0]):
                        model = LSTMAction(hidden_size=hidden, num_layers=layers, dropout_rate=dropout).to(device)
                        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5) if use_scheduler else None
                        train_on_windows(model, train_windows, m, 0.0, optimizer, scheduler, epochs=epochs, device=device, verbose=False)
                        val_losses, _ = evaluate_on_windows(model, val_windows, m, 0.0, device=device)
                        val_avg = float(np.mean(val_losses))
                        val_worst = float(np.max(val_losses))
                        if scale_sq is not None:
                            print(
                                f"Tuning: lr={lr}, hidden={hidden}, layers={layers}, dropout={dropout}, wd={weight_decay} → "
                                f"val_avg={val_avg:.6f} | {scale_sq*val_avg:.6f}, val_worst={val_worst:.6f} | {scale_sq*val_worst:.6f}"
                            )
                        else:
                            print(
                                f"Tuning: lr={lr}, hidden={hidden}, layers={layers}, dropout={dropout}, wd={weight_decay} → "
                                f"val_avg={val_avg:.6f}, val_worst={val_worst:.6f}"
                            )
                        with open("hyperparameter_log.txt", "a") as log_file:
                            if scale_sq is not None:
                                log_file.write(
                                    f"lr={lr}, hidden={hidden}, layers={layers}, dropout={dropout}, wd={weight_decay}, "
                                    f"val_avg={val_avg:.6f} | {scale_sq*val_avg:.6f}, val_worst={val_worst:.6f} | {scale_sq*val_worst:.6f}\n"
                                )
                            else:
                                log_file.write(
                                    f"lr={lr}, hidden={hidden}, layers={layers}, dropout={dropout}, wd={weight_decay}, "
                                    f"val_avg={val_avg:.6f}, val_worst={val_worst:.6f}\n"
                                )
                        if val_avg < best_val_loss:
                            best_val_loss = val_avg
                            best_params = {"lr": lr, "hidden": hidden, "layers": layers, "dropout": dropout, "weight_decay": weight_decay}
    return best_params, best_val_loss


# ========= 主函数 =========
def main(args):
    set_seed(args.seed)
    # 读取数据
    df = pd.read_csv(args.csv)
    values = df.iloc[:, 1].values
    print(f"Loaded {len(values)} data points from {args.csv}")

    # 全数据 0-1 标准化（仅用于训练/推理），便于模型稳定训练
    values_norm, vmin, vmax = minmax_normalize(values)
    scale_sq = (vmax - vmin) ** 2 if vmax != vmin else 1.0  # 将损失从归一化空间映射回原始单位

    # 滑动窗口
    windows = make_sliding_windows(values_norm, window_len=24, step=2)
    train_windows, test_windows = split_windows(windows, train_ratio=0.8)
    print(f"Generated {len(windows)} windows, train={len(train_windows)}, test={len(test_windows)}")
    
    np.random.shuffle(train_windows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.tune:
        train_sub, val_windows = train_val_split(train_windows, val_ratio=0.2)
        search_space = {
            "lr": [0.0001],
            "hidden": [128],
            "layers": [2],
            "dropout": [0.2],
            "weight_decay": [0.0, 1e-4]
        }
        best_params, best_val = grid_search(
            train_sub,
            val_windows,
            args.m,
            device,
            search_space,
            epochs=args.epochs,
            scale_sq=scale_sq,
            use_scheduler=False
        )
        print(f"Best params: {best_params}, val_loss={best_val:.4f}")
        model = LSTMAction(hidden_size=best_params["hidden"], num_layers=best_params["layers"], dropout_rate=best_params["dropout"]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
        scheduler = None
        if args.use_calibrator:
            train_on_windows_calibrated(model, train_windows, args.m, 0.0, optimizer, theta=args.theta, gamma=args.gamma, scheduler=scheduler, epochs=args.epochs, device=device)
        else:
            train_on_windows(model, train_windows, args.m, 0.0, optimizer, scheduler, epochs=args.epochs, device=device)
    else:
        model = LSTMAction(hidden_size=args.hidden, num_layers=args.layers, dropout_rate=args.dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = None
        if args.use_calibrator:
            train_on_windows_calibrated(model, train_windows, args.m, 0.0, optimizer, theta=args.theta, gamma=args.gamma, scheduler=scheduler, epochs=args.epochs, device=device)
        else:
            train_on_windows(model, train_windows, args.m, 0.0, optimizer, scheduler, epochs=args.epochs, device=device)

    # 测试模型（根据是否使用校准器）
    if args.use_calibrator:
        model_losses_norm, x_seqs_calib, h_seqs = evaluate_on_windows_calibrated(model, test_windows, args.m, 0.0, theta=args.theta, gamma=args.gamma, device=device)
        model_losses = [l * scale_sq for l in model_losses_norm]
        print(f"EC-L2O (calibrated) Test avg loss={np.mean(model_losses):.4f}")
    else:
        lstm_losses_norm, h_seqs = evaluate_on_windows(model, test_windows, args.m, 0.0, device=device)
        model_losses = [l * scale_sq for l in lstm_losses_norm]
        print(f"LSTM Test avg loss={np.mean(model_losses):.4f}")

    # 测试 R-OBD baseline
    robd_losses = []
    for w in test_windows:
        loss_norm, _ = robd_rollout(w, args.m, x0=0.0)
        robd_losses.append(loss_norm * scale_sq)
    print(f"R-OBD Test avg loss={np.mean(robd_losses):.4f}")

    # MLA-ROBD 测试
    theta_grid = [0.0, 0.5, 1.0, 2.0, 5.0]
    mla_results = []
    for theta in theta_grid:
        losses = []
        for w, h in zip(test_windows, h_seqs):
            loss_norm, _ = mla_robd_rollout(w, h, args.m, theta, x0=0.0)
            losses.append(loss_norm * scale_sq)
        avg_loss = np.mean(losses)
        worst_loss = np.max(losses)
        mla_results.append({"theta": theta, "avg_loss": avg_loss, "worst_loss": worst_loss})
        print(f"MLA-ROBD (theta={theta}) avg={avg_loss:.4f}, worst={worst_loss:.4f}")

    mla_df = pd.DataFrame(mla_results)
    os.makedirs(args.out, exist_ok=True)
    mla_df.to_csv(os.path.join(args.out, "mla_tradeoff.csv"), index=False)

    # 保存结果图
    plt.figure()
    labels = ["R-OBD", "EC-L2O" if args.use_calibrator else "LSTM"]
    plt.hist([robd_losses, model_losses], label=labels, bins=20)
    plt.legend()
    plt.title("Objective distribution on test windows")
    plt.savefig(os.path.join(args.out, "objective_hist.png"))

    plt.figure()
    plt.boxplot([robd_losses, model_losses], tick_labels=labels)
    plt.title("Objective comparison")
    plt.savefig(os.path.join(args.out, "objective_box.png"))

    # 使用柱形图展示不同 theta 下的平均与最差损失，并保留 LSTM/R-OBD 的水平参考线
    plt.figure()
    thetas = mla_df["theta"].values
    avg_losses = mla_df["avg_loss"].values
    worst_losses = mla_df["worst_loss"].values
    x = np.arange(len(thetas))
    width = 0.35

    plt.bar(x - width/2, avg_losses, width, label="MLA-ROBD Avg Loss")
    plt.bar(x + width/2, worst_losses, width, label="MLA-ROBD Worst Loss")

    plt.axhline(y=np.mean(model_losses), color='r', linestyle='--', label=('EC-L2O Avg Loss' if args.use_calibrator else 'LSTM Avg Loss'))
    plt.axhline(y=np.mean(robd_losses), color='g', linestyle='--', label='R-OBD Avg Loss')

    plt.xticks(x, [str(t) for t in thetas])
    plt.xlabel("Theta")
    plt.ylabel("Loss")
    plt.title("MLA-ROBD Trade-off (Grouped Bars)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "mla_tradeoff.png"))

    print(f"Saved plots and MLA results to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--m", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for LSTM")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay for optimizer")
    parser.add_argument("--out", type=str, default="outputs")
    parser.add_argument("--tune", type=int, default=0, help="1 for hyperparameter tuning, 0 for normal run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_calibrator", type=int, default=1, help="1 to train with EC-L2O calibrator, 0 for pure ML")
    parser.add_argument("--theta", type=float, default=1.0, help="Calibrator weight theta")
    parser.add_argument("--gamma", type=float, default=0.0, help="Penalty weight on (x - h)")
    args = parser.parse_args()
    main(args)


