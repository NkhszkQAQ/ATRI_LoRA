import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from _atri_core_bootstrap import core_module
from package_astrbot_plugin import package_plugin


@pytest.mark.parametrize("old,new", [
    ("atri_runtime", "runtime"), ("atri_memory_lifecycle", "memory"),
    ("atri_story", "rag"), ("atri_generation", "inference"),
    ("atri_model_identity", "identity"), ("atri_context", "context"),
    ("serve_memory_rag", "http"), ("atri_qq_v12", "qq"),
])
def test_old_imports_share_the_actual_implementation(old, new):
    assert importlib.import_module(old) is core_module(new)


def test_full_pipeline_matches_frozen_preintegration_messages_and_database():
    result = subprocess.run([sys.executable, str(ROOT / "tests/_runtime_replay.py"),
                             str(ROOT / "tools")], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == json.loads(
        (ROOT / "tests/fixtures/runtime_before_integration.json").read_text())


def test_plugin_bundle_is_self_contained_without_checkout_or_site_packages(tmp_path):
    output = tmp_path / "plugin"
    manifest = package_plugin(output)
    assert "settings.json" not in manifest and "runtime.json" not in manifest
    assert all(name.endswith((".py", ".yaml", "example.json")) for name in manifest)
    # -S removes installed packages, including this checkout's editable install.
    code = """
import importlib, pathlib, sys, types
folder = pathlib.Path(sys.argv[1])
package = types.ModuleType('isolated_plugin')
package.__path__ = [str(folder)]
sys.modules['isolated_plugin'] = package
legacy = importlib.import_module('isolated_plugin.atri_memory')
memory = importlib.import_module('isolated_plugin.atri_memory_lifecycle')
bridge = importlib.import_module('isolated_plugin.atri_bridge')
import atri_core
assert pathlib.Path(atri_core.__file__).is_relative_to(folder)
store = memory.LifecycleStore(folder / 'fictional-test.sqlite3')
store.observe('fictional', 'one', '我喜欢画画')
assert store.records('fictional')[0]['value'] == '画画'
assert not any(x in sys.modules for x in ('torch', 'transformers', 'fastapi'))
"""
    subprocess.run([sys.executable, "-S", "-c", code, str(output)], check=True, capture_output=True, text=True)
    with pytest.raises(ValueError, match="new directory"):
        package_plugin(output)


def test_package_import_has_no_gpu_or_transport_side_effects():
    code = """
import sys
sys.path.insert(0, sys.argv[1])
import atri_core.runtime, atri_core.rag, atri_core.inference, atri_core.identity
assert not any(x in sys.modules for x in ('torch', 'transformers', 'peft', 'fastapi'))
assert not any(x in sys.modules for x in ('serve_memory_rag', 'serve_v1_1_8b'))
"""
    subprocess.run([sys.executable, "-S", "-c", code, str(ROOT / "src")],
                   check=True, capture_output=True, text=True)


def test_private_config_paths_and_credentials_are_unambiguous(tmp_path):
    config = core_module("config")
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps({"base": "models/base", "users": "private/users.json",
                                "planner_adapter": False, "port": 8022}))
    value = config.load_runtime_config(path)
    assert value["base"] == tmp_path / "models/base"
    assert value["users"] == tmp_path / "private/users.json"
    assert not (tmp_path / "private").exists()
    for bad in ({"token": "not-a-supported-field"}, {"planner_adapter": "false"},
                {"users": "u.json", "qq_secret": "s"}, {"port": True},
                {"stage": "future"}, {"adapter_sha": "wrong"},
                {"story_dense_threshold": float("nan")}):
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            config.load_runtime_config(path)


def test_compatibility_commands_can_show_help_without_models():
    for name in ("serve_memory_rag.py", "atri_rag.py", "package_astrbot_plugin.py"):
        result = subprocess.run([sys.executable, str(ROOT / "tools" / name), "--help"],
                                check=True, capture_output=True, text=True)
        assert "usage:" in result.stdout


def test_container_copy_sources_exist_in_the_new_build_context():
    for relative in ("misc/docker/api/Dockerfile", "misc/docker/web/Dockerfile"):
        for line in (ROOT / relative).read_text().splitlines():
            parts = line.split()
            if parts and parts[0] == "COPY" and not parts[1].startswith("--from="):
                for source in parts[1:-1]:
                    assert list(ROOT.glob(source)), (relative, source)


def test_qq_event_keeps_both_temporal_context_and_current_event_signature(tmp_path):
    bridge = core_module("bridge")
    qq = core_module("qq")
    from types import SimpleNamespace
    secret = b"fictional-only-context-and-event-signature"
    event = SimpleNamespace(get_platform_id=lambda: "fixture", get_sender_id=lambda: "123",
        is_private_chat=lambda: True, message_obj=SimpleNamespace(message_id="current"))
    clock = {"conversation_id": "same-conversation", "sent_at": 1000, "previous_at": 1}
    prompt = "当前话题"
    signed = bridge.attach_context("", bridge.identity(event), [], True, secret, now=1000, turn=clock)
    signed = qq.attach_event(signed, event, prompt, secret, now=1000)
    seen = []
    class Window:
        def select(self, messages, context, **kwargs):
            seen.append(context["turn"])
            return messages, {}
    owner, event_id, history, notice = qq.resolve_event({"messages": [
        {"role": "system", "content": signed}, {"role": "user", "content": prompt}]},
        secret, now=1000, context_window=Window())
    assert seen == [clock]
    assert owner == "qq:fixture:123" and event_id == "qq-event:current"
    assert history == [{"role": "user", "content": prompt}] and notice == ""
