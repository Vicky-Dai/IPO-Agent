#!/usr/bin/env python3
"""
离线评测入口：等价于在项目根目录执行 `ipo-rag eval`。
需已激活虚拟环境并配置 OPENAI_API_KEY；需先 ingest 对应 collection。

用法：
  python benchmarks/run_eval.py
  python benchmarks/run_eval.py -g benchmarks/gold_qa.jsonl -c default
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, "-m", "ipo_rag", "eval"] + sys.argv[1:]
    raise SystemExit(subprocess.call(cmd, cwd=str(root)))


if __name__ == "__main__":
    main()
