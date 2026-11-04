"""04: 构造 DPO 偏好对。chosen=合成的好回复，rejected=用模型改写成"出戏"版本。
默认老师模型 Qwen3-4B（4070/5060 ~8GB）。
用法: python scripts/04_build_dpo.py --inp data/synthetic/atri_synth_sft.jsonl --n 1500
"""
import argparse, json, random, concurrent.futures as cf
from pathlib import Path
from openai import OpenAI

MODES = {
 'assistant': '改写成典型 AI 助手/客服腔：礼貌、啰嗦、带"作为AI助手"或"很高兴为您服务"，完全没有角色语气。',
 'lecture':   '改写成说教式长篇建议：分点列出 3 条以上建议，语气像心理咨询师或老师。',
 'cold':      '改写成冷淡敷衍的回复：没有情绪、没有语气词、不回应用户情绪，像机器人读报。',
 'ooc_taboo': '改写成对用户的话毫无抗议和情绪的顺从回复；如果用户在贬低她，她也毫无反应地接受。',
}
PROMPT = '''下面是聊天里 ATRI 的一条回复。请按要求改写成"坏回复"，用于训练模型避免这类回答。
要求：{req}
只输出改写后的文字，不要解释，不要引号。
用户说：{user}
原回复：{reply}'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp', default='data/synthetic/atri_synth_sft.jsonl')
    ap.add_argument('--out', default='data/dpo/atri_dpo.jsonl')
    ap.add_argument('--n', type=int, default=1500)
    ap.add_argument('--base-url', default='http://127.0.0.1:11435/v1')
    ap.add_argument('--model', default='qwen3:4b')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    client = OpenAI(base_url=a.base_url, api_key='EMPTY')
    rows = [json.loads(l) for l in open(a.inp, encoding='utf8')]
    rows = [r for r in rows if r.get('source') == 'synthetic']
    rng.shuffle(rows); rows = rows[:a.n]

    def one(r):
        conv = r['conversations']
        ctx, chosen = conv[:-1], conv[-1]['value']
        mode = rng.choice(list(MODES))
        try:
            resp = client.chat.completions.create(
                model=a.model, temperature=0.8, max_tokens=1024,
                messages=[{'role': 'user', 'content': PROMPT.format(req=MODES[mode], user=ctx[-1]['value'], reply=chosen)}],
                extra_body={'think': False})
            bad = (resp.choices[0].message.content or '').strip()
        except Exception as e:
            print('err', e); return None
        if not bad or bad == chosen: return None
        return {'system': r['system'], 'conversations': ctx,
                'chosen': {'from': 'gpt', 'value': chosen}, 'rejected': {'from': 'gpt', 'value': bad}, 'mode': mode}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ok = 0
    with open(a.out, 'w', encoding='utf8') as f, cf.ThreadPoolExecutor(a.workers) as ex:
        for res in ex.map(one, rows):
            if res: f.write(json.dumps(res, ensure_ascii=False) + '\n'); ok += 1
    print(f'dpo pairs {ok} -> {a.out}')

if __name__ == '__main__': main()
