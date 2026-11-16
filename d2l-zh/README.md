# d2l-zh

《动手学深度学习》中文版的 PyTorch 章节快照，放在这里给自己的笔记做对照。章节划分和教材一致，例如 `chapter_linear-networks`、`chapter_convolutional-modern`、`chapter_attention-mechanisms`。

代码和插图来自教材项目。学习时请同时参考 [zh.d2l.ai](https://zh.d2l.ai/)。本目录的 `setup.py` 仍是上游旧的依赖声明，不要用它安装环境；请在仓库根目录按 README 安装 `d2l==1.0.3`（必须 `--no-deps`）。

`pytorch/` 下面的 `img/` 是插图原件。`pytorch_2025/month_11/img` 用符号链接指到这里。

笔记运行后生成的 `mlp.params`、`submission.csv`、`my_mlp` 不再纳入版本库。预训练模型的加载已从 `pretrained=True` 改为 `weights="IMAGENET1K_V1"`，加载的仍是原先那一版 ImageNet 权重。
