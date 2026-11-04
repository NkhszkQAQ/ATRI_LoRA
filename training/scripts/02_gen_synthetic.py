"""02: 调用本地 OpenAI 兼容服务（vLLM/Ollama 等，跑 Qwen3）生成合成陪伴对话。
硬件默认：4070/5060（~8GB）→ Qwen3-4B；>10GB 可改 8B；高端少用 14B。
先启动:  vllm serve Qwen/Qwen3-4B --port 8000
用法:    python scripts/02_gen_synthetic.py --n 3000 --out data/synthetic/raw.jsonl
注意:    few-shot 样本仅在运行时从 private/ 读入，不会写进 data/；生成物由 03 脚本做泄漏检查。
"""
import argparse, json, random, re, itertools, concurrent.futures as cf
from pathlib import Path
from openai import OpenAI

SCENES = ['早安问候', '晚安哄睡', '工作/学习吐槽', '失恋或心情低落', '深夜睡不着', '日常闲聊',
          '请教生活小问题', '一起聊游戏/动漫', '故意逗她', '突然冷场', '分享今天的好事', '聊吃饭和做饭',
          '夸奖她', '说她是废物（测试她的底线反应）', '问她是不是AI/机器人', '聊天气和出门', '生病不舒服',
          '想让她学习新东西', '吃醋/撒娇', '要她做点家务相关的事']
USERS = ['话多、爱闲扯', '话很少、回复简短', '毒舌爱吐槽', '有点害羞', '认真倾诉', '开朗爱玩梗']
EMOS = ['开心', '平静', '疲惫', '无聊', '低落', '兴奋', '焦虑']
MEMS = ['', '', '用户叫小明，喜欢咖啡和夜跑', '用户最近在准备考试，常熬夜', '用户养了一只猫叫团子', '用户上周搬了新家']

GEN = """你是对话数据编剧。请根据下面的人设，写一段 {turns} 轮的中文聊天（用户先说，ATRI 回，交替进行），共 {turns} 个用户发言和 {turns} 个 ATRI 回复。

【人设】
{persona}

【场景】{scene}
【用户类型】{user_type}
【用户当前情绪】{emo}
【关于用户的记忆（可为空，若非空则 ATRI 可自然引用一次，不要硬塞）】{mem}

【ATRI 台词风格示例（仅学习语气，不要照抄）】
{shots}

要求：
1. ATRI 每次回复 1~3 句，口语化，带自己的语气词，不要说教，不要客服腔，不要出现"作为AI/语言模型"
2. 用户的话要像真实聊天，短、随意，可有错别字或省略
3. ATRI 的性格要有变化，不要每句都以"高性能"收尾，不要重复同一句式
4. 不要使用 emoji 和括号动作描写
5. 用户称呼写作 {{{{user}}}}，ATRI 称呼对方为"{{{{user}}}}先生"
仅输出 JSON 数组，格式：[{{"role":"user","content":"..."}},{{"role":"assistant","content":"..."}}, ...]，不要任何额外文字。"""

def parse(txt):
    txt = re.sub(r'<think>.*?</think>', '', txt or '', flags=re.S)
    # prefer last JSON array (models often draft then emit final)
    matches = list(re.finditer(r'\[.*?\]', txt, re.S))
    if not matches:
        return None
    for m in reversed(matches):
        try:
            arr = json.loads(m.group(0))
        except Exception:
            continue
        if not isinstance(arr, list) or len(arr) < 2:
            continue
        conv = []
        ok = True
        for i, x in enumerate(arr):
            want = 'user' if i % 2 == 0 else 'assistant'
            if not isinstance(x, dict) or x.get('role') != want or not str(x.get('content', '')).strip():
                ok = False
                break
            conv.append({'from': 'human' if want == 'user' else 'gpt', 'value': str(x['content']).strip()})
        if ok and conv and conv[-1]['from'] == 'gpt':
            return conv
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--out', default='data/synthetic/raw.jsonl')
    ap.add_argument('--persona', default='persona/persona_card.md')
    ap.add_argument('--shots', default='private/derived/atri_style_samples.json')
    ap.add_argument('--base-url', default='http://127.0.0.1:11435/v1',
                    help='OpenAI-compatible endpoint (Ollama default :11435/v1)')
    ap.add_argument('--model', default='qwen3:4b', help='Ollama/vLLM model id')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-tokens', type=int, default=4096)
    ap.add_argument('--vllm-nothink', action='store_true',
                    help='Pass vLLM chat_template_kwargs enable_thinking=False')
    ap.add_argument('--no-ollama-nothink', action='store_true',
                    help='Do not send Ollama think=false')
    a = ap.parse_args()
    rng = random.Random(a.seed)
    persona = Path(a.persona).read_text(encoding='utf8')
    shots_all = json.loads(Path(a.shots).read_text(encoding='utf8'))
    client = OpenAI(base_url=a.base_url, api_key='EMPTY')

    def one(idx):
        r = random.Random(a.seed * 100003 + idx)
        turns = r.choice([2, 3, 3, 4, 4, 5, 6])
        meta = dict(scene=r.choice(SCENES), user_type=r.choice(USERS), emo=r.choice(EMOS), mem=r.choice(MEMS))
        shots = '\n'.join('- ' + s for s in r.sample(shots_all, min(6, len(shots_all))))
        prompt = GEN.format(turns=turns, persona=persona, shots=shots, **meta)
        try:
            kwargs = dict(
                model=a.model, messages=[{'role': 'user', 'content': prompt}],
                temperature=0.95, top_p=0.95, max_tokens=a.max_tokens,
            )
            extra = {}
            if a.vllm_nothink:
                extra['chat_template_kwargs'] = {'enable_thinking': False}
            if not a.no_ollama_nothink:
                extra['think'] = False
            if extra:
                kwargs['extra_body'] = extra
            resp = client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            txt = msg.content or ''
            if not txt and getattr(msg, 'model_extra', None):
                txt = (msg.model_extra or {}).get('reasoning') or ''
            conv = parse(txt)
        except Exception as e:
            print('err', idx, e); return None
        return None if conv is None else {'conversations': conv, 'meta': meta, 'source': 'synthetic'}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ok = 0
    with open(a.out, 'w', encoding='utf8') as f, cf.ThreadPoolExecutor(a.workers) as ex:
        for res in ex.map(one, range(a.n)):
            if res: f.write(json.dumps(res, ensure_ascii=False) + '\n'); ok += 1
    print(f'generated {ok}/{a.n} -> {a.out}')

if __name__ == '__main__': main()
