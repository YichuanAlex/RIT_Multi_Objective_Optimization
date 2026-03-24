### 项目概览

- 目标：在具有切换代价的在线优化问题中，学习一个基于 LSTM 的策略 `LSTMAction`，使在每个窗口上的平均目标函数值最小，并与解析基线 R-OBD 以及 MLA-ROBD 进行对比与可视化。
- 代码功能：
  - 读取 `AI_workload.csv` 的功耗序列并进行全局 0–1 归一化（仅用于训练/推理）。
  - 按 24 步长度、步进 2 的滑动窗口划分数据（可通过函数参数修改）。
  - 训练 LSTM 策略并在测试窗口上评估平均目标损失；与 R-OBD 进行对比。
  - 使用 LSTM 的动作序列作为建议 `h_t`，运行 MLA-ROBD 并生成不同 `theta` 下的 trade-off 曲线。
  - 将损失反归一化到原始功耗单位，保存直方图、箱线图、trade-off 折线和 CSV 结果。

### 环境配置

- 依赖（见 `requirements.txt`）：`numpy`、`pandas`、`matplotlib`、`seaborn`、`scikit-learn`、`torch==2.4.0`
- 推荐环境：Python 3.10+，Windows 或 Linux，GPU 可选（自动检测）。

### 安装步骤（Windows PowerShell 示例）：

1. 创建虚拟环境并激活（可选）：
   - `python -m venv .venv`
   - `./.venv/Scripts/Activate.ps1`
2. 安装依赖：
   - `pip install -r requirements.txt`

### 数据准备

- `AI_workload.csv` 两列：`Time (UTC)` 和 `Power (Watt)`。脚本默认取第二列为功耗序列并按整列进行全局归一化。

### 如何运行

- 常规训练与评估：
  - `python rit_ckx.py --csv AI_workload.csv --epochs 100 --hidden 64 --layers 2 --lr 0.01 --dropout 0.2 --weight_decay 0.001 --out outputs --seed 42`
  - 输出包含：
    - `outputs/objective_hist.png`（测试窗口目标分布）
    - `outputs/objective_box.png`（测试窗口目标箱线图）
    - `outputs/mla_tradeoff.png`（MLA-ROBD trade-off）
    - `outputs/mla_tradeoff.csv`（不同 `theta` 的平均/最差损失）
    - `hyperparameter_log.txt`（调参日志，若启用）

- 超参数调参（网格搜索）：
  - `python rit_ckx.py --csv AI_workload.csv --m 5 --epochs 80 --tune 1 --out outputs --seed 42`
  - 调参模式将：
    - 使用 `train_val_split` 在训练窗口中再切分出验证窗口。
    - 搜索 `lr、hidden、layers、dropout、weight_decay` 并打印/记录验证集的平均与最坏损失（均为归一化损失与反归一化损失）。
    - 选出验证均值最小的配置，用于后续完整训练。

命令行参数

- `--csv` 数据文件路径，必填。
- `--m` 切换代价权重，默认 `5.0`。
- `--epochs` 训练轮数。
- `--hidden` LSTM 隐层维度（常规模式）。
- `--layers` LSTM 层数（常规模式）。
- `--lr` 学习率（常规模式）。
- `--dropout` LSTM dropout 比例。
- `--weight_decay` Adam 的权重衰减。
- `--out` 输出目录。
- `--tune` 是否进行超参数调参（1/0）。
- `--seed` 随机种子，保证可复现。

### 代码结构与细节

- 数据处理
  - `minmax_normalize(values)`：返回归一化序列和 `(min, max)`；若全为同值，返回全零序列。
  - `make_sliding_windows(values, window_len=24, step=2)`：构造长度为 `window_len`、步进 `step` 的窗口数组。
  - `split_windows(windows, train_ratio=0.8)`：划分训练与测试窗口。
  - `train_val_split(windows, val_ratio=0.2)`：在训练窗口中再切分出验证窗口。
  - `set_seed(seed)`：统一设置 Python、NumPy、PyTorch 的随机种子。

- 模型
  - `class LSTMAction(nn.Module)`：
    - LSTM 以历史功耗序列 `y_{1:t}` 为输入，输出最后时刻的隐状态 `h_T`。
    - 通过 `torch.cat([h_T, x_{t-1}]) → Linear(hidden+1 → 1)` 生成控制动作 `x_t`。
    - `forward_once(y_seq, x_prev)`：单步前向，支持滚动调用。

- 目标函数与基线
  - `rollout_objective(model, y_seq, m, x0)`：
    - 在一个窗口内滚动生成 `x_t` 并累积代价：
      - 拟合项 `(m/2)*(x_t − y_t)^2`
      - 平滑项 `(1/2)*(x_t − x_{t−1})^2`
    - 返回“每步平均代价”和动作序列。
  - `robd_rollout(y_seq, m, x0)`：解析基线 R-OBD，更新 `x_t = (m*y_t + x_{t-1})/(1+m)`。

- MLA-ROBD
  - `mla_robd_rollout(y_seq, h_seq, m, theta, x0)`：将建议 `h_t` 融入解析更新：
    - 当前实现：`x_t = (m*y_t + x_{t-1} + theta*h_t)/(1 + m + theta)`。
    - 注意：文献中存在采用 `2*theta` 系数的变体；如需更强“服从建议”的力度，可将分子分母中的 `theta` 改为 `2*theta`。

- 训练/评估
  - `train_on_windows(model, windows, m, x0, optimizer, scheduler, epochs, device)`：遍历窗口，按窗口平均损失反向传播；可选 `ReduceLROnPlateau` 调度器。
  - `evaluate_on_windows(model, windows, m, x0)`：收集每个窗口的损失及 LSTM 动作序列，供可视化与 MLA 使用。
  - `grid_search(train_windows, val_windows, ...)`：扩展搜索 `lr/hidden/layers/dropout/weight_decay`；记录验证平均与最坏损失；可选是否使用调度器。
  - 主流程 `main(args)`：
    1) 读数并归一化；2) 构造窗口并划分；3) 训练（常规或调参）；4) 测试与基线对比；5) MLA-ROBD；6) 保存图与 CSV。
  - 反归一化：打印与保存的损失均乘以 `scale_sq = (max − min)^2`，以便回到原始功耗单位。

### 输出与可视化

- `objective_hist.png`：R-OBD 与 LSTM 的测试窗口损失分布直方图。
- `objective_box.png`：测试窗口损失箱线图对比。
- `mla_tradeoff.png`：不同 `theta` 下的 MLA-ROBD 平均/最坏损失；同时绘制 R-OBD 与 LSTM 的平均损失水平线。
- `mla_tradeoff.csv`：包含 `theta, avg_loss, worst_loss`。

### 最终结果

LSTM Test avg loss=59573.4651
R-OBD Test avg loss=59801.2555
MLA-ROBD (theta=0.0) avg=59801.2555, worst=107277.7317
MLA-ROBD (theta=0.5) avg=59761.0825, worst=107201.8011
MLA-ROBD (theta=1.0) avg=59730.4610, worst=107141.7006
MLA-ROBD (theta=2.0) avg=59687.6480, worst=107053.0972
MLA-ROBD (theta=5.0) avg=59627.2333, worst=106911.6917
在测试集上，LSTM的平均损失低于R-OBD，说明LSTM模型在该任务上有更好的拟合能力。
同时，MLA-ROBD模型在更大的`theta`值下，也能取得更好的性能。
由于LSTM和R-OBD的平均损失区别不大，在可视化结果中反应的差异较小。但LSTM仍然有较大的优化空间。

| θ   | Avg Loss | Worst Loss |
| --- | -------- | ---------- |
| 0.0 | 526871   | 789335     |
| 0.5 | 418243   | 713862     |
| 1.0 | 374431   | 670121     |
| 2.0 | 343955   | 624870     |
| 5.0 | 337903   | 634476     |
- 洞察： 当LSTM的平均损失低于R-OBD时，说明LSTM模型在该任务上有更好的拟合能力。但过度依赖LSTM可能意味着过度依赖 ML，如果极端时刻预测错误，导致Worst Loss增加，因此在实际应用中需要权衡。如上实验结果所示，当`theta=5.0`时，MLA-ROBD模型的平均损失与R-OBD模型的平均损失基本相同，而Worst Loss增加了100000多。


### 常见问题与建议

- 不同参数下验证损失差异小：该目标函数接近凸且简单、全局归一化压缩动态范围、平均指标与调度器会“稳态化”训练轨迹，导致差异被弱化。可尝试：扩大搜索范围、关闭调参阶段调度器、记录最坏值/分位数损失、增大窗口长度或引入非线性层。
- “最佳参数”训练后验证损失变高：调参时的验证子集与完整训练数据分布可能存在偏移；训练时启用 `scheduler` 与正则项也会改变轨迹。建议在训练结束后再次在验证集上评估（调用 `evaluate_on_windows`），并比较“平均/最坏”两类指标。
- MLA-ROBD trade-off 几乎平坦：考虑调整 `theta` 网格或使用 `2*theta` 版本，并引入“建议一致性”指标 `Advice-MSE = mean((x_t − h_t)^2)` 做二维权衡（Pareto 散点图）。

#### 导师视角的预期提问与答案

- 问：LSTM 是什么？为何适合本任务？
  答：LSTM 是带有门控（输入门、遗忘门、输出门）的循环神经网络，能在序列中保留长期依赖。这里用功耗序列驱动 LSTM，取最后时刻隐状态与上一时刻动作拼接，经线性层生成当前动作，既能拟合目标，又能平滑动作变化，符合含切换代价的在线优化场景。

- 问：为何对功耗数据做 0–1 归一化？是否会影响结果解释？
  答：归一化使不同尺度的序列在同一数值范围内，便于稳定训练与比较。训练与评估在归一化空间进行，最终损失通过 `(max−min)^2` 反归一化回原始功耗单位，保证可解释性不受影响；若序列常数则退化为全零避免数值异常。

- 问：R-OBD 与 MLA-ROBD 的关键区别是什么？`theta` 如何理解？
  答：R-OBD 仅依据当前目标与上一动作解析更新；MLA-ROBD 在此基础上引入建议序列 `h_t`，通过 `theta` 权衡“服从建议”与“优化目标”的力度。当前实现为 `(m*y_t + x_{t-1} + theta*h_t)/(1+m+theta)`，文献存在使用 `2*theta` 的变体，可用于更强的建议约束。

- 问：为什么超参数调参后验证损失差异很小？如何改进？
  答：原因包括：目标近似凸且简单、全局归一化压缩动态范围、按窗口平均损失稀释尖峰、调度器使训练轨迹稳态化、模型容量对任务已足够。改进建议：扩大搜索范围，记录最坏/分位数损失，引入 `Advice-MSE=mean((x_t−h_t)^2)` 作为第二维指标。

- 问：MLA-ROBD 的 trade-off 曲线为何较平？如何提升辨识度？
  答：可能因为 `theta` 网格范围与间距不足、采用的系数版本对建议影响偏弱、LSTM 建议与 R-OBD 行为相近。可扩大或非线性采样 `theta`，尝试 `2*theta` 版本，同时绘制目标损失与 `Advice-MSE` 的二维 Pareto 散点或等高线，提高信息密度与可辨识性。

- 问：如何量化“服从建议”的程度并用于可视化？
  答：使用 `Advice-MSE = mean((x_t − h_t)^2)` 衡量动作与建议的偏离。评估时同时记录“目标损失均值/最坏值”和 `Advice-MSE`，在二维平面上展示 Pareto 前沿，或绘制随 `theta` 变化的双轴曲线，更全面反映权衡关系。

- 问：AI 在完成作业中的具体辅助体现在哪里？
  答：

- 问：若继续深入研究，你会如何扩展与验证？
  答：可从“方法扩展”和“严谨验证”两条主线深入：
  
  ① 方法扩展（Advice-aware 多目标与模型增强）：
  - 明确双目标：同时最小化“目标损失（拟合+切换代价）”与“Advice-MSE（对建议的偏离）”，通过加权或分层优化实现权衡。
  - 模型增强：在 `LSTMAction` 上尝试双向 LSTM、注意力机制或增加非线性头（如 MLP 两层），提高对长依赖与非线性关系的表达能力。
  - 损失与正则：引入分位数/Huber 等鲁棒损失，降低异常尖峰的影响；对动作差分 `x_t−x_{t-1}` 加正则，进一步抑制频繁切换。
  - 特征与窗口：加入时间/负载相关外生特征（如时段标签、历史统计量），调试窗口长度与步进，或用多尺度窗口提升泛化。
  - 基线对比：扩展解析与学习基线（如 TCN、Transformer），并用相同指标与资源约束进行公平对比。

  ② 严谨验证（可复现、统计与稳健性）：
  - 消融与显著性：逐项移除模块（如注意力、Advice-MSE、正则）做消融；对关键指标使用 t 检验或 Wilcoxon 检验，报告统计显著性与效应量。
  - 稳健性分析：对输入序列施加噪声/缺失/漂移扰动，观察指标变化；在多数据集或不同负载分布上重复实验，评估泛化。
  - 误差分解与可视化：分别统计拟合项与切换项贡献，绘制 Pareto 散点、ECDF 与分位数曲线，结合最坏值与分位数指标更全面呈现权衡。
  - 工程与报告：输出完整日志与配置，保存模型权重与结果快照，编制实验表格与图示，确保第三方可复现与审阅。

