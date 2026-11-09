"""One generation adapter shared by isolated service, evaluation, and Colab."""
from contextlib import nullcontext
import math
import time


class GeneratedText(str):
    def __new__(cls, text, metadata):
        obj = super().__new__(cls, text)
        obj.metadata = metadata
        return obj


def hf_generator(tokenizer, model, lock, options, *, planner_adapter=False, max_input_tokens=8192,
                 planner_schema=False, base_only=False):
    import torch
    if type(max_input_tokens) is not int or max_input_tokens < 512:
        raise ValueError('server input token budget must be at least 512')
    if type(planner_schema) is not bool:
        raise ValueError('planner_schema must be a server boolean')
    if type(base_only) is not bool or (base_only and hasattr(model, 'peft_config')):
        raise ValueError('base-only planning requires a verified model without an adapter')
    seed, limit = options.get('seed', 17), options.get('max_tokens', 256)
    if type(seed) is not int:
        raise ValueError('integer seed required')
    if type(limit) is not int or not 1 <= limit <= 1024:
        raise ValueError('max_tokens must be 1–1024')
    temperature, top_p = options.get('temperature', .7), options.get('top_p', .9)
    if type(temperature) not in (int,float) or not math.isfinite(temperature) or not 0 < temperature <= 2:
        raise ValueError('temperature must be finite and in (0,2]')
    if type(top_p) not in (int,float) or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError('top_p must be finite and in (0,1]')

    def generate(messages, *, kind):
        if kind not in {'plan', 'lookup_plan', 'review', 'answer'}:
            raise ValueError('unknown generation purpose')
        if base_only and (kind == 'answer' or (kind == 'plan' and planner_adapter)):
            raise ValueError('base-only diagnostics cannot generate role answers or enable an adapter')
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, return_tensors='pt')
        if inputs['input_ids'].shape[-1] > max_input_tokens:
            raise ValueError('conversation or memory inventory exceeds the server input token budget')
        inputs = inputs.to(model.device)
        kwargs = dict(max_new_tokens=512 if kind in {'plan', 'lookup_plan', 'review'} else limit,
                      do_sample=True, temperature=temperature, top_p=top_p,
                      repetition_penalty=1.1,
                      pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, use_cache=True)
        started = time.monotonic()
        constraint = None
        generation_metadata = dict(kwargs)
        if kind == 'plan' and planner_schema:
            from .schema import planner_constraint
            kwargs['prefix_allowed_tokens_fn'], constraint = planner_constraint(tokenizer)
        disabled = kind in {'lookup_plan', 'review'} or (kind == 'plan' and not planner_adapter)
        with lock, torch.inference_mode(), (model.disable_adapter() if disabled and not base_only else nullcontext()):
            actual_modes = {m.disable_adapters for m in model.modules()
                            if isinstance(getattr(m, 'disable_adapters', None), bool)}
            if actual_modes != (set() if base_only else {disabled}):
                raise ValueError('actual adapter state differs from requested generation mode')
            torch.manual_seed(seed)
            result = model.generate(**inputs, **kwargs)
        prompt_ids = inputs['input_ids'][0].detach().cpu().tolist()
        output_ids = result[0,len(prompt_ids):].detach().cpu().tolist()
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos,list) else [eos])
        finish = 'stop' if output_ids and output_ids[-1] in eos else ('length' if len(output_ids) >= kwargs['max_new_tokens'] else 'stop')
        return GeneratedText(tokenizer.decode(output_ids, skip_special_tokens=True), {
            'prompt_token_ids': prompt_ids, 'output_token_ids': output_ids,
            'seed': seed, 'generation': generation_metadata, 'enable_thinking': False,
            'format_constraint': constraint,
            'adapter_enabled': not disabled, 'finish_reason': finish,
            'seconds': time.monotonic()-started})
    return generate


def load_model(base, adapter):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    t0 = time.time()
    print(f"[init] base={base}", flush=True)
    print(f"[init] adapter={adapter}", flush=True)
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    print(f"[init] ready in {time.time() - t0:.1f}s", flush=True)
    return tok, model
