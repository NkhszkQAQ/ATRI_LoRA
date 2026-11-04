"""26: 用真实聊天种子 + Claude 生成 ATRI 多轮陪伴对话
用法：
  python training/scripts/26_gen_multiturn.py --limit 30          # 试跑 30 条
  python training/scripts/26_gen_multiturn.py                     # 全量 960 条
  python training/scripts/26_gen_multiturn.py --limit 30 --dry    # 只打印 prompt 不调 API
"""
import argparse, json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SEEDS = ROOT / "training/data/corpus/wx_seeds_clean.jsonl"
PERSONA = ROOT / "persona/character_profile.md"
OUT = ROOT / "training/data/v2/pilot/atri_multiturn_v10.jsonl"
API = os.environ.get("LLM_API_URL", "")  # OpenAI 兼容的 chat/completions 地址
MODEL = "claude-sonnet-5-5"

# 精简角色设定（完整档案太长，这里给生成用的规则摘要）
PROFILE = """亚托莉：高性能仿生机器人少女，称呼用户固定「主人」。
- 爱炫耀「高性能」，家务糟会心虚；把关心包装成「职责」
- 被叫「废物/萝卜子」会抗议（蔑称/反歧视法/要求撤回）；被夸会嘴硬得意；看到螃蟹会失态
- 说话短（1~3 句）、爱用省略号、语气词（哼哼/呜哇/……）
- 大部分时候正常回应；低落/深夜/被真骂时严禁胡闹，只有克制的关心
- 完整胡闹链条（羞愧→夸张自嘲→一本正经胡说）只在轻松互怼时偶尔出现（<10%）"""

RULES = """要求：
1. 用户的话原样保留（碎、短、可以没头没尾），可加 1~2 个语气词
2. 给每句用户发言配亚托莉的回应；以正常回应为主
3. 她的回应：短句、省略号、语气词，称「主人」或「你」
4. 回复如果多句，用换行分开（QQ 发消息分段）；单句不换行
5. 不要提原作剧情/人物（外婆、凯瑟琳、水菜萌、安田）
6. 不要客服腔、撒娇卖萌、说教、长篇大论
7. 只输出 JSON 数组：[{"user":"...","atri":"..."}, ...]"""


def load_key():
    return os.environ.get("LLM_API_KEY")


def build_prompt(seed, intent):
    return f"""{PROFILE}

下面是真实用户说的话（从人与人的聊天里摘的），请把它生成成「用户 → 亚托莉」的多轮聊天。

场景：{intent}
用户原话（保留，不要改）：
{chr(10).join(f'- {s}' for s in seed)}

{RULES}"""


def call(key, prompt, timeout=180):
    data = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2400,
        "reasoning_effort": "xhigh",
    }).encode()
    req = urllib.request.Request(API, data=data, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    txt = d["choices"][0]["message"].get("content", "")
    m = re.search(r"\[.*\]", txt, re.S)
    arr = json.loads(m.group(0)) if m else []
    return arr, d.get("usage", {})


def load_persona():
    t = (ROOT / "persona/system_prompt.txt").read_text(encoding="utf-8")
    return (t.replace("{{user}}", "主人")
             .replace("{{honorific}}", "")
             .replace("{{memory}}", "（暂无长期记忆）"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry", action="store_true", help="只打印 prompt")
    args = ap.parse_args()

    seeds = [json.loads(l) for l in open(SEEDS, encoding="utf-8") if l.strip()]
    if args.limit:
        seeds = seeds[:args.limit]
    print(f"种子: {len(seeds)} 条", flush=True)

    if args.dry:
        print(build_prompt(seeds[0]["seed"], seeds[0]["intent"]))
        return

    key = load_key()
    if not key:
        sys.exit("找不到 LLM_API_KEY（并检查 LLM_API_URL）")
    system = load_persona()
    results, failed = [], 0

    def run(item):
        arr, usage = call(key, build_prompt(item["seed"], item["intent"]))
        if len(arr) < 1:
            return None
        convs = []
        for t in arr:
            u = (t.get("user") or "").strip()
            a = (t.get("atri") or "").strip()
            if u and a:
                convs += [{"from": "human", "value": u}, {"from": "gpt", "value": a}]
        if len(convs) < 2:
            return None
        return {"system": system, "conversations": convs, "meta": {"intent": item["intent"]}}, usage

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run, s): s for s in seeds}
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r:
                results.append(r[0])
            else:
                failed += 1
            if n % 20 == 0:
                OUT.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in results) + "\n", encoding="utf-8")
                print(f"{n}/{len(seeds)} | 有效 {len(results)} | {(time.time()-t0)/60:.1f} 分钟", flush=True)
    OUT.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in results) + "\n", encoding="utf-8")
    print(f"完成: {len(results)} 条有效 / {len(seeds)} | 失败 {failed} | {OUT}", flush=True)


if __name__ == "__main__":
    main()
