"""Export a self-contained AstrBot plugin without settings, secrets, or data.

The output must be a new directory. This command only copies source files;
it never installs the plugin or changes an AstrBot instance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def package_plugin(output, *, root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    output = Path(output)
    if output.exists():
        raise ValueError("plugin output must be a new directory")
    inputs = {}
    plugin = root / "integrations/astrbot/atri_rag_memory"
    for name in ("main.py", "metadata.yaml", "settings.example.json"):
        inputs[name] = plugin / name
    for name in ("_atri_core_bootstrap.py", "atri_bridge.py", "atri_memory.py",
                 "atri_memory_lifecycle.py", "atri_qq_v12.py"):
        inputs[name] = root / "tools" / name
    for source in (root / "src/atri_core").rglob("*.py"):
        if "__pycache__" not in source.parts:
            inputs[str(source.relative_to(root))] = source
    # Validate the complete input list before creating the destination.
    for source in inputs.values():
        if not source.is_file() or source.is_symlink():
            raise ValueError("missing or symlinked plugin source: " + str(source))
    output.mkdir(parents=True)
    manifest = {}
    for name, source in sorted(inputs.items()):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    (output / "source-manifest.json").write_text(
        json.dumps({"format": 1, "files": manifest}, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package_plugin(args.output), indent=2))


if __name__ == "__main__":
    main()
