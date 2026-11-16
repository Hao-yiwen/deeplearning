# CLAUDE.md

Guidance for working in this learning repository.

## Environment

Python 3.12 (3.11 is also fine). PyTorch 2.15 does not support 3.10.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -r requirements-d2l.txt   # required; do not let d2l downgrade numpy
# optional TensorFlow track
pip install -r requirements-tensorflow.txt
```

CPU is the default. Notebooks pick `cuda` only when `torch.cuda.is_available()` is true. The Linux PyTorch wheel installs CUDA 13 user-space libraries and still runs on CPU.

Check the tree without training:

```bash
python tests/test_notebooks.py
python examples/smoke_cpu.py
```

## Layout

1. `pytorch_2025/` — current notes. `month_11/` is the chapter sequence (there is no chapter 9). `month_11/img` is a symlink to `d2l-zh/pytorch/img`.
2. `pytorch_2024/` — shorter weekly notes. `week2/` is the from-scratch linear regression that used to sit in `tensorflow/week2`.
3. `tensorflow/` — TensorFlow week 1 and one linear-regression notebook.
4. `d2l-zh/pytorch/` — Dive into Deep Learning snapshot. Install `d2l` from PyPI, not from `d2l-zh/pytorch/setup.py`.
5. `practise/` — unrelated snippets.
6. `examples/smoke_cpu.py` — short CPU runs of linear regression, MLP, CNN, and a character RNN.

`*_no_d2l.ipynb` files reimplement helpers that the sibling notebook imports from `d2l`. Keep both.

Large CSVs, checkpoints (`*.safetensors`, `*.params`, `mlp.params`), and generated `submission.csv` files stay out of git. The California housing notebooks fall back to `train_sample.csv` / `test_sample.csv`.

## Conventions

- Chinese comments, English names.
- Theory, then a from-scratch implementation, then the framework version.
- Do not change the algorithm when cleaning notebooks. `weights="IMAGENET1K_V1"` replaces the old `pretrained=True` argument and loads the same checkpoint family.
- Prefer educational clarity over production tuning.
