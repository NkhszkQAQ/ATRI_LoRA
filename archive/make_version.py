"""把当前项目的交付文件存成一个新版本目录。

用法：
    python releases/make_version.py           # 自动用今天的日期 + 下一个版本号
    python releases/make_version.py --dry-run # 只看会建在哪里，不复制

规则：
    releases/<YYYY-MM-DD>/v<major>.<minor>/
"""
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REL = ROOT / "archive"

# 要归档的文件（源路径 -> 归档名）
# 云端文件自动找「最新日期里版本号最大」的那一版
def latest_cloud_dir() -> Path | None:
    cloud = ROOT / "colab"
    best = None
    if not cloud.exists():
        return None
    for day in sorted(p for p in cloud.iterdir()
                      if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name)):
        vers = [(int(m.group(1)), int(m.group(2)), d)
                for d in day.iterdir()
                if d.is_dir() and (m := re.fullmatch(r"v(\d+)\.(\d+)", d.name))]
        if vers:
            best = max(vers)[2]
    return best


CLOUD = latest_cloud_dir()
FILES = [
    (CLOUD / "ATRI_v22_测试_Colab.ipynb", "ATRI_v22_测试_Colab.ipynb"),
    (CLOUD / "atri_bundle_v22.zip", "atri_bundle_v22.zip"),
    (CLOUD / "atri_adapter.zip", "atri_v22_adapter_only.zip"),
    ("webapp/我做的界面/webapp/index.html", "index.html"),
    ("local/ATRI_v22_本地.ipynb", "ATRI_v22_本地.ipynb"),
    ("local/run_local.py", "run_local.py"),
    ("local/本地运行.md", "本地运行.md"),
]


def next_version(day_dir: Path) -> tuple[int, int]:
    versions = []
    if day_dir.exists():
        for d in day_dir.iterdir():
            m = re.fullmatch(r"v(\d+)\.(\d+)", d.name)
            if m:
                versions.append((int(m.group(1)), int(m.group(2))))
    if not versions:
        return 0, 1
    major, minor = max(versions)
    return major, minor + 1


def main():
    dry = "--dry-run" in sys.argv
    today = datetime.now().strftime("%Y-%m-%d")
    day_dir = REL / today
    major, minor = next_version(day_dir)
    target = day_dir / f"v{major}.{minor}"

    print(f"目标目录: {target}")
    if dry:
        for src, name in FILES:
            p = src if isinstance(src, Path) else ROOT / src
            print(f"  {'OK ' if p.exists() else '缺失'} {name}")
        return

    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src, name in FILES:
        p = src if isinstance(src, Path) else ROOT / src
        if p.exists():
            shutil.copy2(p, target / name)
            copied += 1
            print(f"  ✓ {name}")
        else:
            print(f"  - 跳过（不存在）: {src}")
    print(f"\n完成：{target}（{copied} 个文件）")


if __name__ == "__main__":
    main()
