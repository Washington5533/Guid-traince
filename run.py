#!/usr/bin/env python
"""向后兼容入口，等价于 `guarftrain` CLI。

pip install guarftrain 之后，`guarftrain` 命令全局可用。
此文件保留是为了兼容 `python run.py watch -- ...` 的旧用法。
"""
from guardian.cli import main
import sys

sys.exit(main())
