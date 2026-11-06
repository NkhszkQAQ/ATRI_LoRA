"""Small, source-backed answer plans; no database writes or model downloads."""
from dataclasses import dataclass, field
import re

VERSION = "1.3.0"
UNKNOWN = "主人，这个设定目前没有查到可靠依据，我不能确认。"
NAMES = r"亚托莉|ATRI|夏生|水菜萌|凯瑟琳|凛凛花|凜凜花|洋子|龙司|安田"
SPEC_TERMS = r"充(?:满|一次)?电|电压|接口|续航|供电|电源|能源|电池容量|生产日期|出厂日期|制造商|厂家|生日|身高|体重|多高|多重"


@dataclass
class AnswerTask:
    question: str
    kind: str
    reference: str = ""
    contract: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)


def split_questions(text):
    """Split explicit separate requests, not names joined by 和."""
    parts = re.split(r"[？?]+\s*|(?:另外|还有|顺便|同时|而且|并且)[，,：:\s]*", text)
    return [p.strip(" \n，,。") for p in parts if p.strip(" \n，,。")] or [text]


def memory_question(text):
    return bool(re.search(r"什么|啥|哪|谁|怎么|多少|是否|是不是|会不会|能不能|吗|呢|记得|记住了|保存了", text) and re.search(r"我|用户|主人", text) and re.search(
        r"记得|记住|保存|偏好|喜欢|最爱|名字|姓名|昵称|称呼|叫|代号|口令|颜色|喝|爱好", text))


def memory_reference(question, records, context=()):
    groups = [("喝", "饮品", "饮料"), ("颜色",), ("名字", "姓名", "昵称", "称呼", "叫", "name"),
              ("代号", "口令", "校验码"), ("爱好", "兴趣")]
    selected = []
    for row in records:
        key = row["key"]
        if key in question or row["value"] in question or any(any(w in question for w in group) and any(w in key for w in group) for group in groups):
            selected.append(row)
    if not selected and re.search(r"记得我|关于我|保存了什么|记住了什么", question):
        selected = records
    if not selected and re.search(r"喜欢什么|偏好是什么", question):
        selected = [row for row in records if re.search(r"喜好|偏好|喜欢", row["key"])]
    asking_saved = bool(re.search(r"保存|长期记忆", question))
    lookups = selected
    if not selected and not asking_saved:
        for words, key in [(r"喝|饮品|饮料", "偏好饮品"), (r"叫|名字|称呼|昵称", "称呼"),
                           (r"颜色", "喜欢的颜色"), (r"代号|校验码", "校验代号")]:
            if re.search(words, question):
                lookups = [{"key": key, "value": ""}]
                break
    # A current explicit correction takes precedence without silently updating
    # SQLite. Only unambiguous attribute statements are recognized here.
    for message in reversed(context):
        if message.get("role") != "user" or message.get("content", "").lstrip().startswith("/memory"):
            continue
        text = message.get("content", "")
        for row in lookups:
            pattern = re.escape(row["key"]) + r"(?:现在)?(?:是|改成|改为)[：:]?([^，,。?？\n]{1,80})"
            match = re.search(pattern, text)
            if not match and re.search(r"饮品|饮料", row["key"]):
                match = re.search(r"我(?:现在|最近)?(?:更)?喜欢喝([^，,。?？\n]{1,40})", text)
            if not match and re.search(r"称呼|名字|姓名", row["key"]):
                match = re.search(r"我(?:叫|的名字是|名字叫)([A-Za-z\u4e00-\u9fff·]{1,20})(?:[，,。!！\n]|$)", text)
                if match and match[1].startswith(("你", "他", "她", "大家")):
                    match = None
            if match and not re.search(r"什么|哪|吗|呢", match[1]):
                if asking_saved and selected:
                    return "主人，你保存的" + row["key"] + "是“" + row["value"] + "”；你本轮说明的是“" + match[1].strip() + "”，尚未改写已保存记忆。"
                return "主人，你本轮说明的" + row["key"] + "是“" + match[1].strip() + "”（尚未改写已保存记忆）。"
    if not selected:
        return "主人，目前没有找到你问的这项已保存信息。"
    content = "；".join(f"{row['key']}是“{row['value']}”" for row in selected)
    return "主人，你保存的" + content + "。"


def role_fact_question(text):
    return bool(re.search(r"谁|什么|啥|何|哪|几|怎么|为什么|吗|是否|是不是|会不会|能不能|要不要|多少|多高|多重|多久|能否|有没有", text) and (re.search(NAMES, text, re.I) or (
        re.search(r"你|她|战斗型号", text) and re.search(
            r"机器人|人类|身份|味觉|营养|电池|型号|能吃|吃饭|家务|料理|做饭|下厨|拿手菜|烧菜|炒菜|做菜|厨艺|记忆|学校|" + SPEC_TERMS, text))))


def make_plan(messages, rag=None, records=None, *, external_memory=False, memory_notice=None):
    if not messages or messages[-1].get("role") != "user":
        return []
    question = messages[-1]["content"]
    tasks = []
    parts = split_questions(question)
    if len(parts) > 6:
        return [AnswerTask(question, "unknown", "主人，这次的问题有点多，请分几次问，每次最多六个，我再逐项回答。")]
    for index, part in enumerate(parts):
        context = [*messages[:-1], *[{"role": "user", "content": p} for p in parts[:index]]]
        if records is None and memory_notice and memory_question(part):
            tasks.append(AnswerTask(part, "memory", memory_notice))
            continue
        if records is not None and memory_question(part):
            tasks.append(AnswerTask(part, "memory", memory_reference(part, records, messages)))
            continue
        if records is None and not external_memory and memory_question(part) and re.search(r"保存|长期记忆|校验代号", part):
            tasks.append(AnswerTask(part, "memory", "主人，长期记忆功能尚未开启，我不能读取已保存的信息。"))
            continue
        hits = rag.search(part, context) if rag is not None and hasattr(rag, "search") else []
        facts = [hit for hit in hits if hit["metadata"].get("kind") == "canonical_fact"]
        requested_specs = re.findall(SPEC_TERMS, part)
        if requested_specs and role_fact_question(part) and not any(
                spec in hit["metadata"].get("answer", "") for hit in facts for spec in requested_specs):
            tasks.append(AnswerTask(part, "unknown", UNKNOWN))
            continue
        if facts:
            # One explicit question gets its strongest source-checked summary;
            # unrelated later scenes are not mixed into an early-story contract.
            fact = facts[0]
            meta = fact["metadata"]
            reference = meta.get("answer", "")
            if reference:
                import json
                tasks.append(AnswerTask(part, "fact", reference,
                                        json.loads(meta.get("answer_contract", "{}")), [meta]))
                continue
        if rag is not None and role_fact_question(part) and not hits:
            tasks.append(AnswerTask(part, "unknown", UNKNOWN))
        elif hits and role_fact_question(part):
            reference = "\n".join(hit.get("text", "") for hit in hits)[:4000]
            tasks.append(AnswerTask(part, "reference", reference, sources=[hit["metadata"] for hit in hits]))
        else:
            tasks.append(AnswerTask(part, "chat"))
    return tasks if any(task.kind != "chat" for task in tasks) else []


def check_answer(answer, task):
    if not answer.strip():
        return ["empty_answer"]
    problems = []
    for pattern in task.contract.get("required", []):
        if not re.search(pattern, answer, re.I):
            problems.append("missing:" + pattern)
    for pattern in task.contract.get("forbidden", []):
        if re.search(pattern, answer, re.I):
            problems.append("contradiction:" + pattern)
    # These cards explicitly distinguish eating from nutrition. A taste answer
    # must not reintroduce the same wrong eating claim under another wording.
    if "进食" in task.reference and "营养" in task.reference and re.search(
            r"(?:不能|无法|不会)(?:真正|实际|直接)?(?:吃进去|吞下|吞咽|吃东西|吃饭|进食|消化)", answer):
        problems.append("unsupported_eating_limit")
    if "味觉" in task.reference and re.search(
            r"(?:味道|味觉|酸甜).*(?:得|需要|要|依赖|靠).*(?:主人|你).*(?:告诉|描述)|"
            r"(?:得|需要|依赖|靠).*(?:主人|你).*(?:告诉|描述).*(?:味道|味觉|酸甜)", answer):
        problems.append("taste_assigned_to_user")
    if "夏生" in task.reference:
        if "夏生" in task.question and "夏生" not in answer:
            problems.append("missing_story_subject")
        if re.search(r"主人(?:是|在|把|曾|从|发现|救|失去|找到)|(?:我和主人|主人和我).*(?:初遇|相遇|认识)", answer):
            problems.append("story_assigned_to_user")
    for match in re.finditer(r"([\u4e00-\u9fff]{1,4})(?:先生|小姐)", answer):
        if match[1] not in task.reference:
            problems.append("unsupported_address")
    return problems


def normalize_names(answer):
    return re.sub(r"阿托莉|阿特莉|亚托里|亞托莉|\bAttry\b", "亚托莉", answer)


def normalize_output(answer, context=""):
    """Keep grounded/story names; replace unprovided honorific addresses.

    Saved values are rendered separately and are never passed through this
    function. It only cleans model-generated prose, not the user's records.
    """
    prefixes = ("我是", "我就是", "不过", "今天", "还是", "认为", "请帮", "请给", "帮", "给", "对", "和", "与", "是")
    def name_parts(token):
        for prefix in prefixes:
            if token.startswith(prefix) and len(token) > len(prefix):
                return prefix, token[len(prefix):]
        return "", token
    allowed = set(re.findall(NAMES, context, re.I))
    for match in re.finditer(r"([\u4e00-\u9fff]{1,4})(?:先生|小姐)", context):
        allowed.add(name_parts(match[1])[1])
    def replace_address(match):
        prefix, name = name_parts(match[1])
        if name == "主人" or name in allowed:
            return match[0]
        return prefix + "主人"
    return re.sub(r"([\u4e00-\u9fff]{1,4})(?:先生|小姐)", replace_address, normalize_names(answer))


def render_plan(tasks, system, generate, history=(), prepare_chat=None):
    """generate(messages) returns a model draft; verify before exposing it.

    Unknown facts and saved user values are rendered without asking the model to
    invent them. Fact drafts get one repair attempt; failed contracts fall back
    to their checked summary, never to an unsupported model answer.
    """
    answers, trace = [], []
    for task in tasks:
        if task.kind in {"memory", "unknown"}:
            answers.append(task.reference)
            trace.append({"kind": task.kind, "method": "direct_reference"})
            continue
        if task.kind == "fact" and not task.contract.get("required"):
            answers.append("主人，按原作资料，" + task.reference)
            trace.append({"kind": "fact", "method": "source_reference"})
            continue
        prompt = system
        if task.kind in {"fact", "reference"}:
            prompt += ("\n本次只回答当前小问题。下面是已经核对出处的事实，不要混用其他剧情阶段，"
                       "不要添加没有依据的姓名、能力或经历。完整表达这些事实，保留否定与不确定性。\n"
                       + task.reference)
        method = "model"
        prior = [dict(m) for m in history if m.get("role") != "system"] if task.kind == "chat" else []
        model_messages = [{"role": "system", "content": prompt}, *prior,
                          {"role": "user", "content": task.question}]
        if task.kind == "chat" and prepare_chat is not None:
            model_messages = prepare_chat(model_messages)
        answer = normalize_output(generate(model_messages), task.reference + task.question)
        if task.kind == "fact" and check_answer(answer, task):
            answer = normalize_output(generate([
                {"role": "system", "content": prompt + "\n请重新回答：只用已核实的事实，不自夸、不补充其他结论。"},
                {"role": "user", "content": task.question}]), task.reference + task.question)
            method = "repair"
            if check_answer(answer, task):
                answer = "主人，按原作资料，" + task.reference
                method = "source_fallback"
        answers.append(answer or "主人，这个问题我暂时答不上来。")
        trace.append({"kind": task.kind, "method": method})
    return "\n".join(answers), trace
