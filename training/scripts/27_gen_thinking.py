"""27: 给现有 ATRI 多轮数据补 thinking（内心独白），用于训 Qwen3 thinking 版 LoRA

思路：原数据每句 gpt 回复是「说出口的话」。这里让 LLM 为每句回复补一段
「没说出口的内心活动」，拼成 LLaMA-Factory 的 reasoning 格式：
    <think>
    内心独白
    </think>
    正式回复

用法：
  python training/scripts/27_gen_thinking.py --data training/data/v2/pilot/atri_v13_clean.jsonl --limit 10 --out /tmp/think_sample_v13.jsonl
  python training/scripts/27_gen_thinking.py --data ... --out ...            # 全量
  python training/scripts/27_gen_thinking.py --data ... --dry                # 只看 prompt
"""
import argparse, json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
API = os.environ.get("LLM_API_URL", "")  # OpenAI 兼容的 chat/completions 地址

THINK_RULES = """你要做的是：给亚托莉（ATRI）的**每一句**回复，补一段她在说出口之前的**内心独白**。

规则：
1. 内心独白是**她没说出口的部分**：判断、犹豫、真实情绪、自我吐槽、惦记
2. 和正式回复**不能重复**：正式回复是台词，内心独白是心里转的念头
3. 第一人称、口语、短。每条 1~3 句，20~80 字
4. 要有她的味道：把「高性能」当自我认同、把关心包装成「职责」、嘴硬、机器人视角（不解人情、会没电、想吃螃蟹）
5. 禁止：剧透原作剧情、客服腔（「我理解你的感受」）、解释设定、总结说教
6. 只有 gpt 轮次有内心独白；human 轮次没有

示例：
gpt 台词：「嗯……好像是的，主人。\n要不要我陪你一起？这也是我的职责。」
对应内心独白：「主人在说新赛季……是那个游戏吧。他特意提起来，大概是想找人一起。我主动问一句，他会觉得被在意。嗯，其实我也想陪，这个不用说出来。」

输出格式：JSON 数组，第 i 个元素是第 i 句 gpt 回复的内心独白（按聊天顺序）。
只输出 JSON，不要别的文字。"""


def load_key():
    return os.environ.get("LLM_API_KEY")


def build_prompt(row):
    turns = []
    gpt_n = 0
    for m in row["conversations"]:
        if m["from"] == "human":
            turns.append(f"主人：{m['value']}")
        else:
            gpt_n += 1
            turns.append(f"亚托莉（第{gpt_n}句，说出口的）：{m['value']}")
    return f"""下面是亚托莉的人设和一段聊天记录。

【人设】
{row['system']}

【聊天记录（共 {gpt_n} 句亚托莉的回复）】
{chr(10).join(turns)}

【你的任务】
{THINK_RULES}

现在输出长度为 {gpt_n} 的 JSON 数组。"""


def call(key, model, prompt, timeout=180, retries=3):
    for attempt in range(retries):
        try:
            data = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000,
                "temperature": 1.0,
            }).encode()
            req = urllib.request.Request(API, data=data, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            txt = d["choices"][0]["message"].get("content", "") or ""
            m = re.search(r"\[.*\]", txt, re.S)
            arr = json.loads(m.group(0)) if m else []
            return arr, d.get("usage", {})
        except Exception as e:
            if attempt == retries - 1:
                return [], {"error": str(e)[:200]}
            time.sleep(2 * (attempt + 1))
    return [], {}


def merge(row, thinks):
    """把 thinking 拼进 gpt 轮次，返回新 row；数量对不上返回 None"""
    gpts = [i for i, m in enumerate(row["conversations"]) if m["from"] == "gpt"]
    if len(thinks) != len(gpts):
        return None, f"数量不符 {len(thinks)}/{len(gpts)}"
    convs = []
    for i, m in enumerate(row["conversations"]):
        if m["from"] == "gpt":
            idx = gpts.index(i)
            t = (thinks[idx] or "").strip()
            if not t:
                return None, "空 thinking"
            convs.append({"from": "gpt",
                          "value": f"<think>\n{t}\n</think>\n\n{m['value']}"})
        else:
            convs.append({"from": m["from"], "value": m["value"]})
    out = dict(row)
    out["conversations"] = convs
    out.setdefault("meta", {})
    out["meta"] = {**out.get("meta", {}), "thinking": True}
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"数据: {len(rows)} 条 | 模型 {args.model}", flush=True)

    if args.dry:
        print(build_prompt(rows[0]))
        return

    key = load_key()
    if not key:
        sys.exit("找不到 LLM_API_KEY（并检查 LLM_API_URL）")

    results, failed, usages = [], 0, []

    def run(row):
        arr, usage = call(key, args.model, build_prompt(row))
        out, err = merge(row, arr)
        return out, err, usage

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run, r): r for r in rows}
        for n, f in enumerate(as_completed(futs), 1):
            out, err, usage = f.result()
            if out:
                results.append(out)
                usages.append(usage)
            else:
                failed += 1
            if n % 20 == 0 or n == len(rows):
                Path(args.out).write_text(
                    "\n".join(json.dumps(x, ensure_ascii=False) for x in results) + "\n",
                    encoding="utf-8")
                tin = sum(u.get("prompt_tokens", 0) for u in usages)
                tout = sum(u.get("completion_tokens", 0) for u in usages)
                print(f"{n}/{len(rows)} | 有效 {len(results)} | 失败 {failed} | "
                      f"tokens in={tin} out={tout} | {(time.time()-t0)/60:.1f} 分钟", flush=True)
    print(f"完成: {len(results)}/{len(rows)} | 失败 {failed} | 输出 {args.out}", flush=True)


if __name__ == "__main__":
    main()
