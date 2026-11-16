# 深度学习实践与学习

个人深度学习学习仓库。主线是 PyTorch 笔记，按时间和章节从基础写到卷积、检测、RNN、Transformer 和 BERT；旁边保留《动手学深度学习》教材快照，以及一份早期的 TensorFlow 对照练习。

笔记追求的顺序是：先看清公式，再从零写一遍，然后用框架里的现成层做对照。算法本身没有改写成另一套实现。

## 环境

- Python **3.12**（3.11 也可以）。PyTorch 2.15 不再支持 3.10。
- 没有 GPU 时直接用 CPU。训练代码按 `torch.cuda.is_available()` 选择设备。
- Linux 上 `pip install torch` 会顺便装 CUDA 13 的用户态库，机器上没有显卡也能在 CPU 上跑。

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install --no-deps -r requirements-d2l.txt

# 只有要跑 tensorflow/ 时才需要
pip install -r requirements-tensorflow.txt

jupyter lab
```

`d2l==1.0.3` 必须加 `--no-deps`。它自己的依赖声明会把 NumPy 钉回 1.23.5，和当前的 PyTorch 冲突。安装后 `from d2l import torch as d2l` 可以正常使用。

已核对、能一起安装的版本写在 `requirements.txt`：PyTorch 2.15.0、torchvision 0.29.2、NumPy 2.5.5、pandas 2.3.3、Matplotlib 3.11.2。pandas 停在 2.3，是为了躲开 3.x 对表格接口的调整。

快速自检：

```bash
python tests/test_notebooks.py
python examples/smoke_cpu.py
```

`examples/smoke_cpu.py` 用很少的步数跑线性回归、多层感知机、小型卷积网络和字符级 RNN。装了 TensorFlow 时还会跑一层线性回归。

## 目录

```
deeplearning/
├── pytorch_2025/          # 当前主线
├── pytorch_2024/          # 更早的周笔记
├── tensorflow/            # TensorFlow 对照，以及一份后来发现其实是 PyTorch 的草稿
├── d2l-zh/                # 《动手学深度学习》PyTorch 教材快照
├── practise/              # 和主线无关的小脚本
├── examples/smoke_cpu.py  # CPU 冒烟测试
└── tests/test_notebooks.py
```

同一主题经常有两份笔记：原文件调用 `d2l` 的工具函数，文件名带 `_no_d2l` 的那份把这些工具换成了本地实现。两份都保留。

`pytorch_2025/month_11/img` 是指向 `d2l-zh/pytorch/img` 的符号链接，避免把教材插图再存一份。克隆之后如果图片打不开，确认 Git 检出了符号链接（Windows 上需要开启 `core.symlinks`）。

### pytorch_2025

按月份排，不是严格的教材顺序。

| 路径 | 内容 |
| --- | --- |
| `month_2/` | 调用方舟 API 的对话示例。密钥从环境变量 `ARK_API_KEY` 读取 |
| `month_7/prerequisites/` | 预备知识复习 |
| `month_7/practise_1_flashattention.ipynb` | 传统注意力和分块注意力的对照 |
| `month_10/` | 一个很小的 GPT-2 结构，用来看张量形状 |
| `month_11/` | 成体系的章节笔记，说明见 [month_11/README.md](pytorch_2025/month_11/README.md) |

章节号从 8 直接到 10，中间没有第 9 章，按原来的编号保留。

### pytorch_2024

| 路径 | 内容 |
| --- | --- |
| `practise_1_getstarted.ipynb` | Fashion-MNIST 入门 |
| `week2/` | 从零实现线性回归。这份草稿原先误放在 `tensorflow/week2` |
| `week3/` | MLP、CNN、RNN、LSTM、Transformer |
| `week4/` | 字符级文本、图像扩散的示意和训练笔记。`poems.txt` 是几首公有领域的短诗 |

`week3/practise_4_cnn.ipynb` 原来是空文件，现已补成一个不下载数据、可在 CPU 上跑的小卷积网络。

### tensorflow

`week1` 是数据、线性代数、自动微分和概率。`week2/practise_1_linear_regression.ipynb` 用 TensorFlow 做向量加法计时和线性回归。概率笔记需要 `tensorflow-probability`。

### d2l-zh

教材快照，章节目录和上游一致，用来对照自己的笔记。请从 PyPI 安装 `d2l`，不要用这里的 `setup.py`（它还钉着更旧的依赖）。说明见 [d2l-zh/README.md](d2l-zh/README.md)。

### practise

调试、下载 JSON、调用图像翻译接口的零散练习。图像翻译的密钥从环境变量 `DASHSCOPE_API_KEY` 读取。

## 学习路线

1. `pytorch_2025/month_11/chapter_1_base`：数据、线代、微积分、自动微分、概率。
2. `chapter_2_mlp`：线性回归、softmax、多层感知机、权重衰减和 dropout。
3. `chapter_4_torch_base` 和 `chapter_5_cnn`：张量、自定义层、LeNet 到 ResNet。
4. `chapter_6_cnn_ddp` 到 `chapter_8_target_detection`：多卡、增广、微调，然后是检测、分割和风格迁移。
5. `chapter_10_rnn` 到 `chapter_13_bert`：语言模型、注意力、Transformer、词向量和 BERT。
6. 想对照框架时再看 `tensorflow/week1` 和教材目录 `d2l-zh/pytorch`。

`chapter_3_kaggle` 和 `chapter_7_classify_leaves` 依赖外部数据，见下面的说明。

## 数据

- 加州房价：完整的 `train.csv` / `test.csv` 没有放进仓库。目录里有各 200 行的 `train_sample.csv` 和 `test_sample.csv`，笔记会在找不到完整文件时自动改读样本。列包括 `Sold Price`、`Bedrooms` 等。把完整文件放回 `pytorch_2025/month_11/chapter_3_kaggle/california-house-prices/` 即可走原来的路径。
- 树叶分类、CIFAR-10、狗品种：笔记按 Kaggle 目录读取，数据本身不在仓库里。
- 教材图片和 `chapter_8` 用到的 `cat1.jpg`、`rainier.jpg` 等，都在 `d2l-zh/pytorch/img`。
- RNN 歌词笔记使用 `chapter_10_rnn/sample_corpus/` 里的自编短句。模型仍是字符级 GRU。

## 怎么跑

CPU 上适合先打开这些笔记：

- `pytorch_2024/week3/practise_3_mlp.ipynb`
- `pytorch_2024/week3/practise_4_cnn.ipynb`
- `pytorch_2024/week2/practise_1_linear_regression_scratch.ipynb`
- `pytorch_2025/month_7/practise_1_flashattention.ipynb`
- `pytorch_2025/month_11/chapter_1_base/` 下的线代和自动微分

检测、BERT、多卡和 Kaggle 微调需要下载预训练权重或大数据，有 GPU 时再跑。风格迁移和 FCN 里的预训练模型写成了 `weights="IMAGENET1K_V1"`，对应以前 `pretrained=True` 加载的那一版 ImageNet 权重。

## 参考

- [动手学深度学习](https://zh.d2l.ai/)（本仓库 `d2l-zh/` 是其中的 PyTorch 章节快照）
- [PyTorch 文档](https://pytorch.org/docs/stable/)
- [torchvision 模型权重](https://pytorch.org/vision/stable/models.html)
