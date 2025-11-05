# 深度学习实践与学习

个人深度学习学习仓库，记录从基础到进阶的完整学习过程。

## 学习理念

- 理论先行 - 理解数学原理和算法本质
- 从零实现 - 手写核心算法，掌握实现细节
- 框架应用 - PyTorch/TensorFlow 实践
- 对比学习 - 双框架对比实现

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

### 阶段 1：基础数学
**目录**: `d2l-zh/` 和 `tensorflow/week1/`
- 线性代数、微积分、概率论
- 数据操作与预处理

### 阶段 2：基础模型
**目录**: `pytorch_2024/` 和 `tensorflow/week2/`
- 线性回归（从零实现 → 框架实现）
- 多层感知机（MLP）
- Softmax 回归

### 阶段 3：核心网络
**目录**: `pytorch_2024/week3/`
- CNN、RNN、LSTM、GRU
- 批标准化、Dropout

### 阶段 4：前沿技术
**目录**: `pytorch_2024/week4/` 和 `pytorch_2025/`
- Transformer、注意力机制
- 文本生成、Diffusion Models
- Flash Attention

## 项目结构

```
deeplearning/
├── d2l-zh/                     # 《动手学深度学习》完整实现
├── pytorch_2024/               # PyTorch 系统学习
│   ├── week3/                  # 核心模型（Linear, MLP, CNN, RNN）
│   └── week4/                  # 高级主题（Transformer, 文本生成）
├── pytorch_2025/               # 最新技术实践
│   └── month_7/                # Flash Attention
├── tensorflow/                 # TensorFlow 学习路径
│   ├── week1/                  # 基础数学
│   └── week2/                  # 线性回归
└── practise/                   # 工具和实验
```

## 技术栈

- **PyTorch 2.1.0** - 主要框架
- **TensorFlow** - 对比学习
- **D2L 1.0.3** - 《动手学深度学习》工具库
- **Jupyter Lab** - 开发环境

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

