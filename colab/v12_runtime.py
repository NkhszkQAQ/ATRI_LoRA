"""Mount authenticated V1.2 API around an already loaded Colab 8B model.

The existing anonymous demo page is not given a shared memory identity. Callers
use /v12/v1/chat/completions with a per-user Bearer token and event ID instead.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))

from atri_generation import hf_generator
from atri_memory_lifecycle import LifecycleStore, Policy
from serve_memory_rag import build_app
from atri_model_identity import verify_loaded


def mount_runtime(app, *, tokenizer, model, base_path, base_manifest, adapter_path, expected_adapter_sha,
                  users, database, system, search, stage='daily', policy=None,
                  trace_dir=None, model_lock=None):
    """Explicit opt-in; pass the same model lock used by the notebook's old route."""
    if model_lock is None:
        raise ValueError('shared notebook generation lock is required')
    identity=verify_loaded(tokenizer,model,base=base_path,manifest=base_manifest,
                           adapter=adapter_path,adapter_sha=expected_adapter_sha)
    store=LifecycleStore(database,policy or Policy())
    runtime_app=build_app(store,users,lambda body:hf_generator(tokenizer,model,model_lock,body),
                          system=system,identity=identity,search=search,stage=stage,trace_dir=trace_dir)
    app.mount('/v12',runtime_app)
    return runtime_app
