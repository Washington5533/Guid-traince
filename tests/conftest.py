"""tests/ 根级公共装置：把仓库根目录加入 sys.path，供直接 import guardian.*。"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
