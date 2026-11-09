"""Explicit deployment configuration; paths are relative to the config file.

No directory, database, model, or network connection is created while reading.
Credentials stay in referenced private files, never in this configuration.
"""
import json
import math
from pathlib import Path
import re

from .contracts import STAGES

PATHS = {"base", "adapter", "system", "database", "traces", "users", "qq_secret",
         "qq_context_store", "base_manifest", "memory_policy", "script_source"}
BOOLEANS = {"planner_adapter", "planner_schema"}
FIELDS = PATHS | BOOLEANS | {"adapter_sha", "story_cards", "story_dense_model",
                              "story_dense_threshold", "stage", "port"}


def load_runtime_config(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) - FIELDS:
        raise ValueError("runtime config must contain only documented server fields")
    result = {}
    for key, value in data.items():
        if key in PATHS or key == "story_dense_model":
            if not isinstance(value, str) or not value.strip():
                raise ValueError(key + " must be a non-empty local path")
            result[key] = (path.parent / Path(value).expanduser()).resolve()
            if key == "story_dense_model":
                result[key] = str(result[key])
        elif key == "story_cards":
            if not isinstance(value, list) or not value or any(
                    not isinstance(v, str) or not v.strip() for v in value):
                raise ValueError("story_cards must be a non-empty path list")
            result[key] = [(path.parent / Path(v).expanduser()).resolve() for v in value]
        elif key in BOOLEANS:
            if type(value) is not bool:
                raise ValueError(key + " must be a boolean")
            result[key] = value
        elif key == "port":
            if type(value) is not int or not 1 <= value <= 65535:
                raise ValueError("port must be an integer from 1 to 65535")
            result[key] = value
        elif key == "adapter_sha":
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("adapter_sha must be a lowercase SHA-256 digest")
            result[key] = value
        elif key == "stage":
            if not isinstance(value, str) or value not in STAGES:
                raise ValueError("unknown configured story stage")
            result[key] = value
        elif key == "story_dense_threshold":
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("story_dense_threshold must be finite and in [0,1]")
            result[key] = value
        else:
            result[key] = value
    if result.get("users") and result.get("qq_secret"):
        raise ValueError("choose either per-user Bearer or signed QQ authentication")
    return result
