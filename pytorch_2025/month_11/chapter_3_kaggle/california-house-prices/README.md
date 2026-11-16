# 加州房价数据

笔记读取的完整文件是本目录下的 `train.csv` 和 `test.csv`。训练集大约 7.6 万行，两份 CSV 合计约 90MB，不再放进 Git。

仓库里保留了各 200 行的 `train_sample.csv`、`test_sample.csv`，以及 `sample_submission.csv`。笔记在找不到完整文件时会改读样本，用来检查读表和后面的预处理能不能跑通。

把完整的 `train.csv` 和 `test.csv` 放回这个目录后，笔记会自动使用完整数据。表头需要包含 `Sold Price`、`Bedrooms`、`Bathrooms` 等原有列。
