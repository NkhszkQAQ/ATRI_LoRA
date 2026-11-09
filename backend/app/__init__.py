"""ATRI LoRA API package."""
from pathlib import Path
import sys

# Support an unpacked checkout as well as an installed atri-core distribution.
_source = Path(__file__).resolve().parents[2] / "src"
if (_source / "atri_core").is_dir() and str(_source) not in sys.path:
    sys.path.insert(0, str(_source))
