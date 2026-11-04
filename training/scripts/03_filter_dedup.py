"""03: 规则过滤 + 去重 + 原作泄漏检查，输出可公开的 data/synthetic/atri_synth_sft.jsonl
用法: python scripts/03_filter_dedup.py --inp data/synthetic/raw.jsonl --private private/derived/atri_dialogue_windows.jsonl
"""
import argparse, json, re, collections
from pathlib import Path

OOC = re.compile(r'作为(一个)?(AI|人工智能|语言模型|助手)|我是(一个)?(AI|人工智能|语言模型)|很抱歉|无法为您|如果您需要|请注意|建议您|心理咨询|祝您|首先[，,].*其次')
EMOJI = re.compile('[\U0001F300-\U0001FAFF\u2600-\u27BF]')
ACTION = re.compile(r'[（(][^）)]{1,15}[）)]')

def shingles(s, n):
    s = re.sub(r'\s', '', s)
    return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else set()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp', default='data/synthetic/raw.jsonl')
    ap.add_argument('--out', default='data/synthetic/atri_synth_sft.jsonl')
    ap.add_argument('--private', default='private/derived/atri_dialogue_windows.jsonl')
    ap.add_argument('--leak-n', type=int, default=14, help='与原作 ATRI 台词共享 ≥N 字连续片段则丢弃')
    ap.add_argument('--max-reply', type=int, default=120)
    ap.add_argument('--opener-cap', type=int, default=40, help='同一开头(前6字)最多保留条数')
    ap.add_argument('--system', default='persona/system_prompt.txt')
    ap.add_argument('--safety', default='data/synthetic/safety_seed.jsonl')
    a = ap.parse_args()

    leak = set()
    if Path(a.private).exists():
        for l in open(a.private, encoding='utf8'):
            for t in json.loads(l)['conversations']:
                if t['from'] == 'gpt': leak |= shingles(t['value'], a.leak_n)
    else: print('WARN: 未找到原作文件，跳过泄漏检查（公开前必须补跑）')

    system = Path(a.system).read_text(encoding='utf8').strip()
    stats = collections.Counter(); seen_sig = set(); opener = collections.Counter(); kept = []
    for l in open(a.inp, encoding='utf8'):
        o = json.loads(l); conv = o['conversations']
        gpts = [t['value'] for t in conv if t['from'] == 'gpt']
        alltxt = ' '.join(t['value'] for t in conv)
        reason = None
        if any(OOC.search(g) for g in gpts): reason = 'ooc'
        elif EMOJI.search(alltxt) or any(ACTION.search(g) for g in gpts): reason = 'emoji_or_action'
        elif any(len(g) > a.max_reply for g in gpts): reason = 'too_long'
        elif any(len(g.strip()) < 2 for g in gpts): reason = 'too_short'
        elif len(set(gpts)) < len(gpts): reason = 'repeat_in_dialog'
        elif any(shingles(g, a.leak_n) & leak for g in gpts): reason = 'leak_from_source'
        else:
            sig = gpts[0]
            if sig in seen_sig: reason = 'dup'
            else:
                op = gpts[0][:6]
                if opener[op] >= a.opener_cap: reason = 'opener_cap'
                else: seen_sig.add(sig); opener[op] += 1
        if reason: stats[reason] += 1; continue
        stats['kept'] += 1
        kept.append({'system': system, 'conversations': conv, 'meta': o.get('meta', {}), 'source': 'synthetic'})

    if Path(a.safety).exists():
        for l in open(a.safety, encoding='utf8'):
            o = json.loads(l)
            kept.append({'system': system, 'conversations': o['conversations'], 'meta': {'scene': 'safety'}, 'source': 'handwritten'})
    with open(a.out, 'w', encoding='utf8') as f:
        for o in kept: f.write(json.dumps(o, ensure_ascii=False) + '\n')
    print(dict(stats), '->', a.out, f'(total {len(kept)} incl. safety seed)')

if __name__ == '__main__': main()
