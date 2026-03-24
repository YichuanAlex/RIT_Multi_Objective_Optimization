# RIT Multi-Objective Optimization | RIT 多目标优化

<div align="center">

**Research Project | 研究项目**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Research](https://img.shields.io/badge/Research-Machine%20Learning%20%26%20Optimization-green.svg)]()

**Author | 作者**: YichuanAlex (Zixi Jiang)  
**Email | 邮箱**: jiangzixi1527435659@gmail.com  
**Last Updated | 最后更新**: 2026-03-24

</div>

---

## Table of Contents | 目录

- [Overview | 项目概述](#overview--项目概述)
- [Research Background | 研究背景](#research-background--研究背景)
- [Research Objectives | 研究目标](#research-objectives--研究目标)
- [Methodology | 研究方法](#methodology--研究方法)
- [Key Findings | 主要发现](#key-findings--主要发现)
- [Project Structure | 项目结构](#project-structure--项目结构)
- [Installation and Usage | 安装与使用](#installation-and-usage--安装与使用)
- [Citation | 引用建议](#citation--引用建议)
- [License | 许可证](#license--许可证)
- [Contact | 联系方式](#contact--联系方式)

---

## Overview | 项目概述

**English:**  
This project implements a multi-objective optimization framework for machine learning model training, focusing on balancing multiple performance metrics through hyperparameter tuning and regularization techniques. The research explores the trade-offs between different optimization objectives and identifies optimal configurations using grid search and post-hoc analysis.

**中文:**  
本项目实现了一个用于机器学习模型训练的多目标优化框架，重点通过超参数调优和正则化技术平衡多个性能指标。研究探索了不同优化目标之间的权衡，并使用网格搜索和后验分析识别最优配置。

---

## Research Background | 研究背景

**English:**  
In machine learning model deployment, practitioners often face the challenge of optimizing multiple conflicting objectives simultaneously, such as minimizing average loss while controlling worst-case performance. Traditional single-objective optimization fails to capture these complex trade-offs, necessitating multi-objective approaches that can identify Pareto-optimal solutions.

**中文:**  
在机器学习模型部署中，从业者经常面临同时优化多个冲突目标的挑战，例如最小化平均损失的同时控制最坏情况性能。传统的单目标优化无法捕捉这些复杂的权衡，需要能够识别帕累托最优解的多目标优化方法。

---

## Research Objectives | 研究目标

**English:**
1. Develop a multi-objective optimization framework for ML model training
2. Investigate the impact of different regularization strategies (L1, L2, L3, μ) on model performance
3. Identify optimal hyperparameter configurations through systematic grid search
4. Analyze trade-offs between average performance and worst-case robustness
5. Provide practical recommendations for multi-objective model optimization

**中文:**
1. 开发用于机器学习模型训练的多目标优化框架
2. 研究不同正则化策略（L1、L2、L3、μ）对模型性能的影响
3. 通过系统网格搜索识别最优超参数配置
4. 分析平均性能与最坏情况鲁棒性之间的权衡
5. 为多目标模型优化提供实践建议

---

## Methodology | 研究方法

### 1. Multi-Objective Optimization Framework | 多目标优化框架

**English:**
- **Objective Functions**: Multiple loss functions optimized simultaneously
- **Regularization**: L1, L2, L3 norms with tunable weights (λ₁, λ₂, λ₃)
- **Trade-off Parameter**: μ controls balance between objectives
- **Optimization**: Grid search over hyperparameter space

**中文:**
- **目标函数**: 同时优化多个损失函数
- **正则化**: 具有可调权重（λ₁、λ₂、λ₃）的 L1、L2、L3 范数
- **权衡参数**: μ控制目标之间的平衡
- **优化**: 超参数空间上的网格搜索

### 2. Hyperparameter Tuning | 超参数调优

**English:**
- **Learning Rate**: [0.0001, 0.001, 0.01, 0.02]
- **Hidden Units**: [16, 32, 64, 128]
- **Layers**: [2, 3]
- **Dropout**: [0.0, 0.2, 0.5]
- **Weight Decay**: [0.0, 0.0001, 0.001]
- **Lambda Weights**: λ₁∈[0.1, 0.2, 0.5], λ₂∈[0.0], λ₃∈[1.0, 2.0, 3.0]
- **Mu Parameter**: μ∈[0.3, 0.5, 0.7]

**中文:**
- **学习率**: [0.0001, 0.001, 0.01, 0.02]
- **隐藏单元**: [16, 32, 64, 128]
- **层数**: [2, 3]
- **Dropout**: [0.0, 0.2, 0.5]
- **权重衰减**: [0.0, 0.0001, 0.001]
- **Lambda 权重**: λ₁∈[0.1, 0.2, 0.5], λ₂∈[0.0], λ₃∈[1.0, 2.0, 3.0]
- **Mu 参数**: μ∈[0.3, 0.5, 0.7]

---

## Key Findings | 主要发现

**English:**
1. **Optimal Configuration**: λ₁=0.5, λ₂=0.0, λ₃=1.0, μ=0.5 achieved best overall performance
2. **Regularization Impact**: L1 regularization showed significant benefits; L2 had minimal effect
3. **Trade-off Analysis**: Clear Pareto frontier identified between average and worst-case performance
4. **Robustness**: Best configuration demonstrated consistent performance across different test sets
5. **Practical Insights**: Multi-objective approach outperformed single-objective baselines

**中文:**
1. **最优配置**: λ₁=0.5, λ₂=0.0, λ₃=1.0, μ=0.5实现了最佳整体性能
2. **正则化影响**: L1 正则化显示出显著益处；L2 影响最小
3. **权衡分析**: 在平均性能和最坏情况性能之间识别出清晰的帕累托前沿
4. **鲁棒性**: 最优配置在不同测试集上表现出一致的性能
5. **实践启示**: 多目标方法优于单目标基线

---

## Project Structure | 项目结构

```
RIT_Multi_Objective_Optimization/
│
├── README.md                              # Project documentation | 项目文档
├── requirements.txt                       # Python dependencies | Python 依赖
├── hyperparameter_log.txt                 # Hyperparameter tuning log | 超参数调优日志
│
├── rit_v*.py                              # Main optimization scripts | 主优化脚本
├── rit_GPU_v*.py                          # GPU-accelerated versions | GPU 加速版本
│
├── outputs_rit_v0.1/                      # Version 0.1 outputs | 版本 0.1 输出
│   ├── mla_tradeoff.csv                   # Trade-off analysis results | 权衡分析结果
│   ├── mla_tradeoff.png                   # Trade-off visualization | 权衡可视化
│   ├── objective_box.png                  # Objective distribution box plot | 目标分布箱线图
│   └── objective_hist.png                 # Objective histogram | 目标直方图
│
├── outputs_rit_v0.2/                      # Version 0.2 outputs | 版本 0.2 输出
│   └── ... (same structure)
│
└── outputs_rit_v0.3/                      # Version 0.3 outputs | 版本 0.3 输出
    ├── grid_l1=*_l2=*_l3=*_mu=*/          # Grid search results | 网格搜索结果
    │   ├── train_log.txt                  # Training logs | 训练日志
    │   ├── val_metrics.json               # Validation metrics | 验证指标
    │   └── val_post_stats.json            # Post-validation statistics | 后验证统计
    ├── best_config.txt                    # Best configuration | 最优配置
    ├── final_summary.json                 # Final summary | 最终总结
    └── summary.csv                        # Summary statistics | 汇总统计
```

---

## Installation and Usage | 安装与使用

### Prerequisites | 前置条件

**English:**
- Python 3.8 or higher
- pip package manager
- CUDA-compatible GPU (optional, for GPU acceleration)

**中文:**
- Python 3.8 或更高版本
- pip 包管理器
- CUDA 兼容的 GPU（可选，用于 GPU 加速）

### Installation Steps | 安装步骤

**English:**
```bash
# 1. Clone the repository
git clone https://github.com/YichuanAlex/RIT_Multi_Objective_Optimization.git

# 2. Navigate to project directory
cd RIT_Multi_Objective_Optimization

# 3. Install required packages
pip install -r requirements.txt
```

**中文:**
```bash
# 1. 克隆仓库
git clone https://github.com/YichuanAlex/RIT_Multi_Objective_Optimization.git

# 2. 进入项目目录
cd RIT_Multi_Objective_Optimization

# 3. 安装所需包
pip install -r requirements.txt
```

### Running the Analysis | 运行分析

**English:**
```bash
# Run main optimization (CPU version)
python rit_v0.6.py

# Run GPU-accelerated version
python rit_GPU_v0.3.py

# View hyperparameter tuning results
cat hyperparameter_log.txt
```

**中文:**
```bash
# 运行主优化（CPU 版本）
python rit_v0.6.py

# 运行 GPU 加速版本
python rit_GPU_v0.3.py

# 查看超参数调优结果
cat hyperparameter_log.txt
```

---

## Citation | 引用建议

**English:**
```bibtex
@misc{jiang2026rit,
  title={RIT Multi-Objective Optimization Framework for Machine Learning},
  author={Jiang, Zixi},
  year={2026},
  note={GitHub repository: https://github.com/YichuanAlex/RIT_Multi_Objective_Optimization}
}
```

**中文:**
```bibtex
@misc{jiang2026rit,
  title={RIT 机器学习多目标优化框架},
  author={姜，子溪},
  year={2026},
  note={GitHub 仓库：https://github.com/YichuanAlex/RIT_Multi_Objective_Optimization}
}
```

---

## License | 许可证

**English:**
This project is licensed under the MIT License - see the LICENSE file for details.

**中文:**
本项目采用 MIT 许可证 - 详见 LICENSE 文件。

---

## Contact | 联系方式

**English:**
- **Author**: Zixi Jiang (YichuanAlex)
- **Email**: jiangzixi1527435659@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/zixijiang/
- **GitHub**: https://github.com/YichuanAlex

**中文:**
- **作者**: 姜子溪 (YichuanAlex)
- **邮箱**: jiangzixi1527435659@gmail.com
- **领英**: https://www.linkedin.com/in/zixijiang/
- **GitHub**: https://github.com/YichuanAlex

---

## Keywords | 关键词

**English:**  
Multi-Objective Optimization, Machine Learning, Hyperparameter Tuning, Regularization, Pareto Optimization, Grid Search, Model Training, Deep Learning

**中文:**  
多目标优化、机器学习、超参数调优、正则化、帕累托优化、网格搜索、模型训练、深度学习

---

<div align="center">

**Thank you for your interest in this research!**  
**感谢您对本研究的关注!**

⭐ **If you find this project helpful, please give it a star!**  
**如果您觉得本项目有帮助，请给个星星!**

</div>
