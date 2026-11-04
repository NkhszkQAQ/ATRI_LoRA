"""把当前最新版本复制成一个新版本号。

用法：
    python "colab/new_version.py"            # 从最新版本复制出下一版
    python "colab/new_version.py" --dry-run  # 只看会建在哪

改完 notebook 之后：
1. 先把新文件覆盖到最新版本目录里
2. 再跑这个脚本，生成一个干净的下一版快照
"""
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

CLOUD = Path(__file__).resolve().parent


def latest_version() -> Path | None:
    best = None
    for day in sorted(p for p in CLOUD.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name)):
        vers = [(int(m.group(1)), int(m.group(2)), d)
                for d in day.iterdir()
                if d.is_dir() and (m := re.fullmatch(r"v(\d+)\.(\d+)", d.name))]
        if vers:
            major, minor, path = max(vers)
            best = path
    return best


def main():
    dry = "--dry-run" in sys.argv
    latest = latest_version()
    if not latest:
        print("还没任何版本，先手动建一个 日期/v0.1/")
        return

    day_dir = latest.parent
    m = re.fullmatch(r"v(\d+)\.(\d+)", latest.name)
    target = day_dir / f"v{m.group(1)}.{int(m.group(2)) + 1}"

    print(f"最新版本: {latest}")
    print(f"新版本  : {target}")

    if dry:
        return

    target.mkdir(parents=True, exist_ok=True)
    for f in latest.iterdir():
        if f.is_file():
            shutil.copy2(f, target / f.name)
            print(f"  ✓ {f.name}")
    print(f"\n完成：{target}")


if __name__ == "__main__":
    main()
