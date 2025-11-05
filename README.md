# 深度学习实践与学习

个人深度学习学习仓库，基于 PyTorch 的系统化学习记录。

## 学习理念

- 理论先行 - 数学原理和算法本质
- 从零实现 - 手写核心算法
- 框架应用 - PyTorch 实践
- 渐进迭代 - 持续更新最新技术

## 环境配置

```bash
# 创建 conda 环境
conda create -n d2l python=3.10
conda activate d2l

# 安装依赖
pip install -r requirements.txt

# 启动 Jupyter Lab
jupyter lab
```

## 学习路径

### 核心学习（pytorch_2025/）

**month_11/ - 系统化基础**
- chapter_1: 预备知识
  - 数据预处理、线性代数、微积分
  - 概率论、自动微分
- chapter_2: 线性神经网络
  - 线性回归（从零实现 → 简洁实现）
  - Softmax 回归
  - 图像分类数据集
  - MLP（从零实现 → 简洁实现）
  - 欠拟合与过拟合

**month_10/ - 大模型实践**
- GPT-2 简单实现

**month_7/ - 优化技术**
- Flash Attention 实现
- 基础知识复习（prerequisites）

**month_2/ - 前沿模型**
- DeepSeek R1 实践

### 参考资料
- `d2l-zh/` - 《动手学深度学习》完整教材
- `pytorch_2024/` - 早期学习记录
- `tensorflow/` - TensorFlow 对比实现

## 项目结构

```
deeplearning/
├── pytorch_2025/               # 主要学习目录
│   ├── month_11/               # 系统化基础（2025.11）
│   │   ├── chapter_1/          # 预备知识（数据、线代、微积分、概率、自动微分）
│   │   └── chapter_2/          # 线性神经网络（回归、分类、MLP、过拟合）
│   ├── month_10/               # GPT-2 实现（2025.10）
│   ├── month_7/                # Flash Attention（2025.07）
│   └── month_2/                # DeepSeek R1（2025.02）
├── d2l-zh/                     # 《动手学深度学习》教材参考
├── pytorch_2024/               # 2024 年学习记录
└── tensorflow/                 # TensorFlow 对比学习
```

## 技术栈

- PyTorch 1.12.0 - 主要框架
- Jupyter Lab 4.3.4 - 开发环境
- D2L 1.0.3 - 《动手学深度学习》工具库

## 开发规范

**Notebook 结构**：
1. 导入依赖
2. 数据加载与预处理
3. 模型定义
4. 训练循环
5. 评估与可视化

**代码风格**：
- 中文注释、英文命名
- 模块化设计
- 从简单到复杂的渐进式实现

