# month_11 章节笔记

图片在 `img/`，这是指向 `d2l-zh/pytorch/img` 的符号链接。笔记里的 `../img/cat1.jpg` 因此仍然有效。

带 `_no_d2l` 的笔记不调用 `d2l` 包，把数据加载和画图换成了本地代码。

| 目录 | 内容 |
| --- | --- |
| `chapter_1_base` | 预备知识：数据、线性代数、微积分、自动微分、概率 |
| `chapter_2_mlp` | 线性回归、softmax、MLP、欠拟合与过拟合、权重衰减、dropout |
| `chapter_3_kaggle` | 加州房价。完整 CSV 不在仓库里，默认回退到 200 行样本。AutoGluon 笔记需另装 `autogluon` |
| `chapter_4_torch_base` | 参数、自定义层、读写文件、GPU |
| `chapter_5_cnn` | 卷积、LeNet、AlexNet、VGG、NiN、GoogLeNet、批量归一化、ResNet |
| `chapter_6_cnn_ddp` | 多 GPU、图像增广、微调、CIFAR-10 和狗品种 |
| `chapter_7_classify_leaves` | 树叶分类（ResNet）。图片数据需要另备 |
| `chapter_8_target_detection` | 锚框、SSD、语义分割、转置卷积、FCN、风格迁移 |
| `chapter_10_rnn` | 文本预处理、从零实现 RNN、简洁实现、GRU、LSTM、深层 RNN |
| `chapter_11_transformer` | 注意力提示、Nadaraya-Watson、评分函数、Bahdanau、多头注意力、Transformer |
| `chapter_12_word2vec` | 词嵌入数据与 skip-gram |
| `chapter_13_bert` | BERT 模型、掩码数据集、自然语言推断 |

没有 `chapter_9`，编号沿用原来的笔记。

`chapter_10_rnn/sample_corpus/` 是给字符级 GRU 用的自编短句。`chapter_4_torch_base` 里的 `mlp.params`、`x-file` 一类文件是笔记运行时生成的，不再提交。
