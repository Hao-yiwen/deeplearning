"""在 CPU 上跑通本仓库里的几类核心算法。

覆盖：从零线性回归、多层感知机、小型卷积网络、字符级 RNN，
以及 TensorFlow 的一层线性回归（安装了 TensorFlow 时）。
步数很少，只检查损失可计算并且有下降，不追求最终精度。
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def _assert_decreased(name: str, first: float, last: float) -> None:
    if not math.isfinite(first) or not math.isfinite(last):
        raise SystemExit(f"{name} 损失不是有限数: {first} -> {last}")
    if not last < first:
        raise SystemExit(f"{name} 损失没有下降: {first:.4f} -> {last:.4f}")
    print(f"{name}: {first:.4f} -> {last:.4f}")


def train_linear_regression() -> None:
    """平方损失 + 小批量 SGD，和笔记里的从零实现一致。"""
    torch.manual_seed(0)
    true_w = torch.tensor([2.0, -3.4])
    true_b = 4.2
    features = torch.randn(1000, 2)
    labels = features @ true_w + true_b
    labels += torch.randn(1000) * 0.01

    w = torch.zeros(2, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    lr = 0.03
    batch_size = 10
    losses = []
    for _ in range(8):
        idx = torch.randperm(len(features))
        for start in range(0, len(features), batch_size):
            batch = idx[start : start + batch_size]
            pred = features[batch] @ w + b
            loss = ((pred - labels[batch]) ** 2 / 2).sum()
            loss.backward()
            with torch.no_grad():
                w -= lr * w.grad / batch.numel()
                b -= lr * b.grad / batch.numel()
                w.grad.zero_()
                b.grad.zero_()
        with torch.no_grad():
            losses.append(float(((features @ w + b - labels) ** 2).mean()))
    _assert_decreased("线性回归", losses[0], losses[-1])


def train_mlp() -> None:
    """一层隐藏层的分类网络。"""
    torch.manual_seed(0)
    features = torch.randn(512, 20)
    labels = (features[:, :3].sum(dim=1) > 0).long()
    net = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 2))
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for _ in range(12):
        optimizer.zero_grad()
        loss = loss_fn(net(features), labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    _assert_decreased("多层感知机", losses[0], losses[-1])


def train_cnn() -> None:
    """和 week3 的卷积神经网络笔记相同的小网络，只训练很少的步。"""
    torch.manual_seed(0)

    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(6, 16, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(16 * 4 * 4, 32),
                nn.ReLU(),
                nn.Linear(32, 10),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    def batch(n=64):
        y = torch.randint(0, 10, (n,))
        x = torch.randn(n, 1, 28, 28) * 0.1
        for i, label in enumerate(y):
            start = int(label) + 4
            x[i, 0, start : start + 6, 10:16] += 1.5
        return x, y

    net = SmallCNN()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for _ in range(8):
        x, y = batch()
        optimizer.zero_grad()
        loss = loss_fn(net(x), y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    _assert_decreased("卷积神经网络", losses[0], losses[-1])


def train_char_rnn() -> None:
    """用仓库里的短诗做一步字符级 RNN，确认序列维度接得上。"""
    torch.manual_seed(0)
    text = (ROOT / "pytorch_2024/week4/poems.txt").read_text(encoding="utf-8")
    chars = [ch for ch in text if ch.strip()]
    vocab = sorted(set(chars))
    index = {ch: i for i, ch in enumerate(vocab)}
    data = torch.tensor([index[ch] for ch in chars])
    seq_len = 16
    x = data[: seq_len * 8].view(8, seq_len)
    y = data[1 : seq_len * 8 + 1].view(8, seq_len)
    net = nn.Sequential()
    embedding = nn.Embedding(len(vocab), 16)
    rnn = nn.GRU(16, 32, batch_first=True)
    head = nn.Linear(32, len(vocab))
    optimizer = torch.optim.Adam(
        list(embedding.parameters()) + list(rnn.parameters()) + list(head.parameters()),
        lr=0.05,
    )
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for _ in range(15):
        optimizer.zero_grad()
        hidden, _ = rnn(embedding(x))
        loss = loss_fn(head(hidden.reshape(-1, 32)), y.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    _assert_decreased("字符级 RNN", losses[0], losses[-1])


def train_tensorflow_linear() -> None:
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow 未安装，跳过")
        return
    tf.random.set_seed(0)
    x = tf.random.normal((256, 2))
    y = x @ tf.constant([[2.0], [-3.4]]) + 4.2
    model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss="mse")
    history = model.fit(x, y, epochs=8, batch_size=32, verbose=0)
    losses = history.history["loss"]
    _assert_decreased("TensorFlow 线性回归", losses[0], losses[-1])


def check_imports() -> None:
    import numpy
    import pandas
    from d2l import torch as d2l

    frame = pandas.read_csv(
        ROOT
        / "pytorch_2025/month_11/chapter_3_kaggle/california-house-prices/train_sample.csv"
    )
    if "Sold Price" not in frame.columns or len(frame) < 50:
        raise SystemExit("加州房价样本表不可用")
    device = d2l.try_gpu()
    print(
        f"导入正常: torch {torch.__version__}, numpy {numpy.__version__}, "
        f"d2l 设备 {device}, 房价样本 {frame.shape}"
    )


def main() -> None:
    print("device", "cuda" if torch.cuda.is_available() else "cpu")
    check_imports()
    train_linear_regression()
    train_mlp()
    train_cnn()
    train_char_rnn()
    train_tensorflow_linear()
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
