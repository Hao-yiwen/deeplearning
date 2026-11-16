"""检查笔记还能被解析，并且没有留下已知的坏写法。"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".ipynb_checkpoints", "__MACOSX"}


def notebook_paths():
    for path in ROOT.rglob("*.ipynb"):
        if SKIP_PARTS.intersection(path.parts):
            continue
        yield path


def code_text(source) -> str:
    if isinstance(source, list):
        return "".join(source)
    return source or ""


def main() -> int:
    errors = []
    notebooks = list(notebook_paths())
    if len(notebooks) < 100:
        errors.append(f"笔记数量异常: {len(notebooks)}")

    for path in notebooks:
        if path.stat().st_size == 0:
            errors.append(f"{path.relative_to(ROOT)} 是空文件")
            continue
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{path.relative_to(ROOT)} JSON 无法解析: {exc}")
            continue
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = code_text(cell.get("source", []))
            if "pretrained=True" in source:
                errors.append(f"{path.relative_to(ROOT)} 仍使用 pretrained=True")
            if "许巍_5770" in source:
                errors.append(f"{path.relative_to(ROOT)} 仍引用已删除的歌词目录")
            lines = []
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith(("!", "%", "?")):
                    continue
                lines.append(line)
            try:
                ast.parse("\n".join(lines))
            except SyntaxError as exc:
                errors.append(
                    f"{path.relative_to(ROOT)} 第 {index} 个代码单元语法错误: {exc.msg}"
                )

    image = ROOT / "pytorch_2025/month_11/img/cat1.jpg"
    if not image.is_file():
        errors.append("pytorch_2025/month_11/img 没有连到教材图片")
    sample = (
        ROOT
        / "pytorch_2025/month_11/chapter_3_kaggle/california-house-prices/train_sample.csv"
    )
    if "Sold Price" not in sample.read_text(encoding="utf-8").splitlines()[0]:
        errors.append("房价样本表头不对")

    if errors:
        print("\n".join(errors))
        return 1
    print(f"NOTEBOOKS_OK {len(notebooks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
