"""Fail closed when mounting a previously loaded model in another entrance."""
import json
from pathlib import Path

from .files import file_sha

# Configuration identity recorded for this project's verified Qwen3-8B base.
QWEN3_8B_CONFIG_SHA = 'f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30'


def verify_loaded(tokenizer,model,*,base,manifest,adapter,adapter_sha):
    import torch
    from safetensors.torch import load_file
    base,adapter=Path(base).resolve(),Path(adapter).resolve()
    expected=json.loads(Path(manifest).read_text())
    files={**expected['base_files'],**expected['tokenizer_files']}
    if files.get('config.json') != QWEN3_8B_CONFIG_SHA:
        raise ValueError('this profile requires the approved Qwen3-8B configuration')
    for name,digest in files.items():
        if Path(name).name != name or file_sha(base/name)!=digest:
            raise ValueError('base/tokenizer file identity mismatch: '+name)
    if file_sha(adapter/'adapter_model.safetensors')!=adapter_sha:
        raise ValueError('adapter file identity mismatch')
    config=json.loads((base/'config.json').read_text())
    if config.get('model_type')!='qwen3' or Path(model.config._name_or_path).resolve()!=base:
        raise ValueError('loaded model base differs from verified base')
    for key in ('model_type','hidden_size','num_hidden_layers','num_attention_heads','vocab_size'):
        if getattr(model.config,key,None)!=config.get(key):
            raise ValueError('loaded model config differs: '+key)
    if Path(tokenizer.name_or_path).resolve()!=base:
        raise ValueError('loaded tokenizer differs from verified base')
    tensors=load_file(adapter/'adapter_model.safetensors')
    seen=set()
    for name,parameter in model.named_parameters():
        if 'lora_' in name:
            key=name.replace('.default.','.')
            if key not in tensors or not torch.equal(parameter.detach().cpu(),tensors[key]):
                raise ValueError('loaded adapter tensor differs: '+name)
            seen.add(key)
    if not seen or seen!=set(tensors):
        raise ValueError('loaded adapter tensor set differs')
    return {'model':'Qwen3-8B+adapter-'+adapter_sha[:12],'base_and_tokenizer_sha256':files,
            'adapter_sha256':adapter_sha,'loaded_tensors_exact':len(seen),
            'enable_thinking':False,'planner_adapter_enabled':False,'answer_adapter_enabled':True}
