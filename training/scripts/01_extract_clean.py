"""01: 从原作 docx 提取并清洗对白，输出到 private/derived/（不进 git）。
用法: python scripts/01_extract_clean.py --docx private/raw/atri_raw_data.docx --drop-late
依赖: python-docx
"""
import argparse, json, re, random, collections
from pathlib import Path
from docx import Document

PAT = re.compile(r'^([^：:\s]{1,12})[：:](.*)$')
HEAVY = re.compile(r'死|记忆|沉没|消失|寿命|永别|再见|葬|腿|崩溃|坏掉|最后')
SIG = re.compile(r'高性能|哼哼|嘿嘿|呜|♪|夏生先生|螃蟹|机器人|心|废')

def is_bad(t): return len(re.sub(r'[…。、，！？!?~～♪\s.]', '', t)) < 2
def clean(t):
    t = re.sub(r'[（(][^）)]{1,12}[）)]', '', t).strip().strip('「」').strip()
    return re.sub(r'\s{2,}', ' ', t)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--docx', required=True)
    ap.add_argument('--out', default='private/derived')
    ap.add_argument('--user-name', default='夏生')
    ap.add_argument('--max-window', type=int, default=6)
    ap.add_argument('--drop-late', action='store_true', help='剔除后期剧情与沉重内容')
    a = ap.parse_args()
    tpl = lambda t: t.replace(a.user_name, '{{user}}')

    raw = [p.text.strip() for p in Document(a.docx).paragraphs if p.text.strip()]
    N = len(raw)
    items = []
    for i, l in enumerate(raw):
        m = PAT.match(l)
        items.append(dict(i=i, sp=m.group(1) if m else None, t=(m.group(2) if m else l).strip()))

    scenes, cur, nar = [], [], 0          # 连续 3 行旁白视为场景切换
    for it in items:
        if it['sp'] is None:
            nar += 1
            if nar == 3 and cur: scenes.append(cur); cur = []
        else:
            nar = 0; cur.append(it)
    if cur: scenes.append(cur)

    out = []
    for sc in scenes:
        turns = []
        for it in sc:
            t = clean(it['t'])
            if is_bad(t): continue
            if it['sp'] == 'ATRI':
                if turns and turns[-1]['r'] == 'gpt': turns[-1]['t'] += '\n' + t; turns[-1]['i'] = it['i']
                else: turns.append(dict(r='gpt', t=t, i=it['i'], sps=[]))
            else:
                seg = tpl(t) if it['sp'] == a.user_name else f"[{it['sp']}] {tpl(t)}"
                if turns and turns[-1]['r'] == 'human': turns[-1]['t'] += ' ' + seg; turns[-1]['sps'].append(it['sp'])
                else: turns.append(dict(r='human', t=seg, i=it['i'], sps=[it['sp']]))
        for k, tn in enumerate(turns):
            if tn['r'] != 'gpt' or k == 0: continue
            s = max(0, k - (a.max_window - 1))
            while s < k and turns[s]['r'] != 'human': s += 1
            w = turns[s:k + 1]
            if len(w) < 2 or a.user_name not in w[-2]['sps']: continue
            pos = tn['i'] / N
            out.append({'conversations': [{'from': x['r'], 'value': tpl(x['t'])} for x in w],
                        'stage': 'early' if pos < .33 else 'mid' if pos < .66 else 'late',
                        'heavy': bool(HEAVY.search(tn['t'])), 'n_turns': len(w)})
    if a.drop_late:
        out = [o for o in out if o['stage'] != 'late' and not o['heavy']]

    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    with open(od / 'atri_dialogue_windows.jsonl', 'w', encoding='utf8') as f:
        for o in out: f.write(json.dumps(o, ensure_ascii=False) + '\n')

    at = [tpl(clean(i['t'])) for i in items if i['sp'] == 'ATRI']
    at = [t for t in dict.fromkeys(at) if 8 <= len(t) <= 45 and SIG.search(t) and '「' not in t and not is_bad(t)]
    random.Random(7).shuffle(at)
    json.dump(at[:300], open(od / 'atri_style_samples.json', 'w', encoding='utf8'), ensure_ascii=False, indent=1)
    print(f'windows={len(out)} style_samples={min(len(at),300)} stages={dict(collections.Counter(o["stage"] for o in out))}')

if __name__ == '__main__': main()
