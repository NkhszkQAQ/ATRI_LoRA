"""Execute notebook cells in one CPU runtime; no Colab/GPU model inference.

Default: small equal-size weight fixtures, with the catalog hashes replaced in memory.
ATRI_RELEASE_ASSETS=/path/to/zips: use unmodified catalog hashes and real release
archives named atri_adapter_<VERSION>.zip; verify their pinned release digests too.
Run: python -m unittest discover -s tests -p test_colab_version_cache.py -v
"""
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile


NOTEBOOK = Path(__file__).resolve().parents[1] / 'colab/ATRI_LoRA_Colab.ipynb'
CELLS = json.loads(NOTEBOOK.read_text())['cells']
ACQUIRE = ''.join(CELLS[6]['source'])
WRITE_SERVER = ''.join(CELLS[8]['source'])
CATALOG = ast.literal_eval(next(node.value for node in ast.parse(ACQUIRE).body
                               if isinstance(node, ast.Assign)
                               and any(isinstance(t, ast.Name) and t.id == 'CATALOG' for t in node.targets)))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def archive(weight, base):
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w') as z:
        z.writestr('adapter/adapter_model.safetensors', weight)
        z.writestr('adapter/adapter_config.json', json.dumps({'base_model_name_or_path': base}))
    return out.getvalue()


class Response:
    status_code = 200

    def __init__(self, data=None, payload=None):
        self.data, self.payload = data, payload

    def json(self):
        return self.payload

    def raise_for_status(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def iter_content(self, size):
        for start in range(0, len(self.data), size):
            yield self.data[start:start + size]


class NotebookRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = {v: dict(spec) for v, spec in CATALOG.items()}
        cls.packages = {}
        cls.weights = {}
        root = os.environ.get('ATRI_RELEASE_ASSETS')
        for version, spec in cls.catalog.items():
            if root:
                package = (Path(root) / spec['asset']).read_bytes()
                if digest(package) != spec['zip_sha256']:
                    raise AssertionError(f'{version}: release zip digest mismatch')
                with zipfile.ZipFile(io.BytesIO(package)) as z:
                    weight = z.read('adapter/adapter_model.safetensors')
                if len(weight) != spec['size'] or digest(weight) != spec['sha256']:
                    raise AssertionError(f'{version}: released weight identity mismatch')
            else:
                # Same base + same byte count, different identities: the original bug.
                weight = version.encode().ljust(64, b'!')
                base = spec['base'] if version.startswith('v1.1') else '/root/autodl-tmp/models/Qwen3-4B-ms'
                package = archive(weight, base)
                spec.update(size=len(weight), sha256=digest(weight), zip_sha256=digest(package))
            cls.packages[version] = package
            cls.weights[version] = weight

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        previous = os.getcwd()
        os.chdir(self.temp.name)
        self.addCleanup(os.chdir, previous)
        self.downloads = []
        self.offline = False
        self.bad_download = None
        self.corrupt_download = False
        self.network_error = False
        self.uploads = []
        self.upload = {}
        self.gpu_gb = 16
        self.loads = []
        requests = ModuleType('requests')
        requests.get = self.get
        requests.RequestException = ConnectionError
        torch = ModuleType('torch')
        torch.cuda = SimpleNamespace(
            is_available=lambda: self.gpu_gb is not None,
            get_device_properties=lambda _: SimpleNamespace(total_memory=self.gpu_gb * 1024 ** 3),
            is_bf16_supported=lambda: False)
        torch.float16 = 'fp16'
        torch.bfloat16 = 'bf16'
        colab = ModuleType('google.colab')
        colab.userdata = SimpleNamespace(get=lambda _: None)
        colab.files = SimpleNamespace(upload=self.do_upload)
        google = ModuleType('google')
        google.colab = colab
        transformers = ModuleType('transformers')
        transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **kw: object())
        transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=lambda *a, **kw: object())
        transformers.BitsAndBytesConfig = lambda **kw: kw
        transformers.TextIteratorStreamer = object
        peft = ModuleType('peft')
        peft.PeftModel = SimpleNamespace(from_pretrained=self.load_adapter)
        fastapi = ModuleType('fastapi')
        fastapi.FastAPI = lambda: SimpleNamespace(get=lambda _: lambda f: f, post=lambda _: lambda f: f)
        responses = ModuleType('fastapi.responses')
        responses.HTMLResponse = responses.StreamingResponse = lambda x, **kw: x
        uvicorn = ModuleType('uvicorn')
        uvicorn.run = lambda *a, **kw: None
        modules = {'requests': requests, 'torch': torch, 'google': google, 'google.colab': colab,
                   'transformers': transformers, 'peft': peft, 'fastapi': fastapi,
                   'fastapi.responses': responses, 'uvicorn': uvicorn}
        self.enterContext(patch.dict(sys.modules, modules))
        self.enterContext(patch.dict(os.environ, {'GH_TOKEN': ''}))
        self.enterContext(patch('time.sleep', lambda _: None))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.ns = {}

    def get(self, url, **kw):
        if '/releases/tags/' in url:
            if self.network_error:
                raise ConnectionError('offline test')
            if self.offline:
                return SimpleNamespace(status_code=503)
            assets = [{'name': s['asset'], 'size': len(self.packages[v]),
                       'browser_download_url': 'https://release.test/' + v}
                      for v, s in self.catalog.items() if url.endswith('/' + s['tag'])]
            return Response(payload={'assets': assets})
        version = url.rsplit('/', 1)[-1]
        self.downloads.append(version)
        data = self.packages[self.bad_download or version]
        if self.corrupt_download:
            data = bytes([data[0] ^ 1]) + data[1:]
        return Response(data=data)

    def do_upload(self):
        self.uploads.append(True)
        return self.upload

    def load_adapter(self, model, path):
        data = (Path(path) / 'adapter_model.safetensors').read_bytes()
        self.loads.append((path, digest(data)))
        return SimpleNamespace(eval=lambda: None)

    def acquire(self, version):
        tree = ast.parse(ACQUIRE)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id == 'VERSION' for t in node.targets):
                    node.value = ast.Constant(version)
                elif any(isinstance(t, ast.Name) and t.id == 'CATALOG' for t in node.targets):
                    node.value = ast.parse(repr(self.catalog), mode='eval').body
        exec(compile(ast.fix_missing_locations(tree), str(NOTEBOOK) + ':adapter', 'exec'), self.ns)

    def check_server(self, version):
        exec(compile(WRITE_SERVER, str(NOTEBOOK) + ':server', 'exec'), self.ns)
        server = {}
        exec(compile(self.ns['SERVER'], 'generated-server.py', 'exec'), server)
        expected_path = str(Path('adapters', version).resolve())
        self.assertEqual(self.loads[-1], (expected_path, self.catalog[version]['sha256']))
        self.assertEqual(server['BASE'], self.catalog[version]['base'])
        self.assertEqual(server['health'](), {'ok': True, 'model': 'ATRI ' + version,
                                            'thinking': self.catalog[version]['thinking']})
        self.assertIn(self.catalog[version]['base'].split('/')[-1] + ' · ' + version,
                      Path('index.html').read_text())
        return Path(expected_path) / 'adapter_model.safetensors'

    def test_same_runtime_R_DT_R_uses_correct_cached_weights(self):
        self.acquire('v1.1-R')
        original = self.check_server('v1.1-R')
        mtime = original.stat().st_mtime_ns
        self.acquire('v1.1-DT')
        self.check_server('v1.1-DT')
        self.acquire('v1.1-R')
        self.check_server('v1.1-R')
        self.assertEqual(self.downloads, ['v1.1-R', 'v1.1-DT'])
        self.assertEqual(original.stat().st_mtime_ns, mtime)

    def test_same_runtime_4B_R_C_R_uses_correct_cached_weights(self):
        for version in ('v1.0-R', 'v1.0-C', 'v1.0-R'):
            self.acquire(version)
            self.check_server(version)
        self.assertEqual(self.downloads, ['v1.0-R', 'v1.0-C'])

    def test_wrong_same_base_upload_is_rejected_and_state_cleared(self):
        for wanted, wrong in (('v1.1-DT', 'v1.1-R'), ('v1.0-C', 'v1.0-R')):
            with self.subTest(wanted=wanted):
                self.offline = False
                self.acquire(wrong)
                self.offline = True
                # Deliberately lie in the filename, as well as the UI selection.
                self.upload = {self.catalog[wanted]['asset']: self.packages[wrong]}
                with self.assertRaisesRegex(ValueError, 'SHA-256'):
                    self.acquire(wanted)
                self.assertIsNone(self.ns['READY_ADAPTER'])
                self.assertFalse(Path('adapters', wanted).exists())
                self.assertTrue(Path('adapters', wrong).exists())
                with self.assertRaisesRegex(RuntimeError, '第 3 格'):
                    exec(WRITE_SERVER, self.ns)

    def test_correct_upload_all_versions_accepts_renamed_zip(self):
        self.offline = True
        for version in self.catalog:
            with self.subTest(version=version):
                self.upload = {'renamed.zip': self.packages[version]}
                self.acquire(version)
                self.check_server(version)
        self.assertEqual(self.downloads, [])
        self.assertEqual(len(self.uploads), 4)

    def test_wrong_version_download_falls_back_to_verified_upload(self):
        self.bad_download = 'v1.1-R'
        self.upload = {'correct.zip': self.packages['v1.1-DT']}
        self.acquire('v1.1-DT')
        self.check_server('v1.1-DT')
        self.assertEqual(self.downloads, ['v1.1-DT'] * 3)
        self.assertEqual(len(self.uploads), 1)

    def test_same_size_corrupt_download_fails_zip_hash(self):
        self.corrupt_download = True
        self.upload = {'correct.zip': self.packages['v1.1-R']}
        self.acquire('v1.1-R')
        self.check_server('v1.1-R')
        self.assertEqual(self.downloads, ['v1.1-R'] * 3)
        self.assertEqual(len(self.uploads), 1)

    def test_release_network_error_allows_manual_upload(self):
        self.network_error = True
        self.upload = {'correct.zip': self.packages['v1.0-C']}
        self.acquire('v1.0-C')
        self.check_server('v1.0-C')
        self.assertEqual(self.downloads, [])
        self.assertEqual(len(self.uploads), 1)

    def test_generator_requires_successful_adapter_cell(self):
        with self.assertRaisesRegex(RuntimeError, '第 3 格'):
            exec(WRITE_SERVER, self.ns)

    def test_swapped_cache_is_repaired_without_touching_other_version(self):
        self.acquire('v1.1-R')
        r = self.check_server('v1.1-R')
        self.acquire('v1.1-DT')
        dt = self.check_server('v1.1-DT')
        mtime = r.stat().st_mtime_ns
        shutil.copyfile(r, dt)
        self.acquire('v1.1-DT')
        self.check_server('v1.1-DT')
        self.assertEqual(self.downloads, ['v1.1-R', 'v1.1-DT', 'v1.1-DT'])
        self.assertEqual(r.stat().st_mtime_ns, mtime)

    def test_low_vram_fallback_updates_path_label_and_thinking(self):
        self.gpu_gb = 6
        self.acquire('v1.1-DT')
        self.check_server('v1.0-R')
        self.assertEqual(self.downloads, ['v1.0-R'])

    def test_6_5_GB_and_cpu_do_not_fallback(self):
        for memory in (6.5, None):
            self.gpu_gb = memory
            self.acquire('v1.1-DT')
            self.check_server('v1.1-DT')
        self.assertEqual(self.downloads, ['v1.1-DT'])

    def test_ui_uses_verified_selection_even_if_VERSION_is_edited(self):
        self.acquire('v1.1-R')
        self.ns['VERSION'] = 'v1.1-DT'
        self.check_server('v1.1-R')

    def test_legacy_shared_adapter_directory_is_not_reused(self):
        Path('adapter').mkdir()
        old = Path('adapter/adapter_model.safetensors')
        old.write_bytes(self.weights['v1.1-R'])
        self.acquire('v1.1-DT')
        self.check_server('v1.1-DT')
        self.assertEqual(digest(old.read_bytes()), self.catalog['v1.1-R']['sha256'])

    def test_upload_requires_one_file(self):
        self.offline = True
        for upload in ({}, {'one.zip': b'1', 'two.zip': b'2'}):
            self.upload = upload
            with self.assertRaisesRegex(ValueError, '只上传一个'):
                self.acquire('v1.1-R')
            self.assertIsNone(self.ns['READY_ADAPTER'])

    def test_correct_weight_with_wrong_base_or_truncated_weight_rejected(self):
        self.offline = True
        for data in (archive(self.weights['v1.1-R'], 'Qwen/Qwen3-4B'),
                     archive(self.weights['v1.1-R'][:-1], 'Qwen/Qwen3-8B')):
            self.upload = {'wrong.zip': data}
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                self.acquire('v1.1-R')
            self.assertIsNone(self.ns['READY_ADAPTER'])

    def test_cache_modified_after_generation_is_rejected_at_server_start(self):
        self.acquire('v1.1-R')
        path = self.check_server('v1.1-R')
        path.write_bytes(self.weights['v1.1-DT'])
        count = len(self.loads)
        with self.assertRaisesRegex(RuntimeError, 'SHA-256'):
            exec(self.ns['SERVER'], {})
        self.assertEqual(len(self.loads), count)


if __name__ == '__main__':
    unittest.main(verbosity=2)
