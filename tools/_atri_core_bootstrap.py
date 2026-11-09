"""Locate the shared source package in a checkout or a self-contained plugin bundle."""
from importlib import import_module
from pathlib import Path
import sys


def core_module(name):
    here = Path(__file__).resolve().parent
    for source in (here.parent / "src", here / "src"):
        if (source / "atri_core" / "__init__.py").is_file():
            if str(source) not in sys.path:
                sys.path.insert(0, str(source))
            break
    return import_module("atri_core." + name)
