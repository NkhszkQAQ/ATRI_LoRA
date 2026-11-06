"""Small, source-backed answer plans; no database writes or model downloads."""
from dataclasses import dataclass, field
import re

VERSION = "1.3.0"
UNKNOWN = "这个……我还不能确认，不能乱说啦。"
VOICE = ("\n你就是亚托莉，在和眼前的人聊天，不是介绍亚托莉的旁白。"
         "说自己的事用‘我’，夏生等其他人的经历仍用他们的名字，不把夏生叫成主人。"
         "资料只用来核对事实，不朗读资料，不说‘按原作’‘资料显示’‘设定中’。"
         "直接回答，短句，可以有一点得意、犹豫或不好意思，但不硬贴口头禅；"
         "不用每次都说‘主人’‘高性能’或汇报记忆系统。‘高性能’是对自己的自夸，"
         "不要拿它泛泛夸饮料、作业或用户。围绕眼前这句话，不替用户编时间、地点或剧情。"
         "保留事实，不为耍宝编造能力或经历。")
NAMES = r"亚托莉|ATRI|夏生|水菜萌|凯瑟琳|凛凛花|凜凜花|洋子|龙司|安田"
SPEC_TERMS = r"充(?:满|一次)?电|电压|接口|续航|供电|电源|能源|电池容量|生产日期|出厂日期|制造商|厂家|生日|身高|体重|多高|多重"


@dataclass
class AnswerTask:
    question: str
    kind: str
    reference: str = ""
    contract: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)
    spoken_answer: str = ""


def memory_phrase(key, value):
    """Only present literal values; never ask a model to rewrite user records."""
    if re.search(r"饮品|饮料", key):
        return f"你喜欢喝“{value}”"
    if re.search(r"颜色", key):
        return f"你喜欢的颜色是“{value}”"
    if re.fullmatch(r"名字|姓名|昵称|称呼|name", key, re.I):
        return f"你的{key}是“{value}”"
    return f"你的{key}是“{value}”"


def split_questions(text):
    """Split explicit separate requests, not names joined by 和."""
    parts = re.split(r"[？?]+\s*|(?:另外|还有|顺便|同时|而且|并且)[，,：:\s]*", text)
    return [p.strip(" \n，,。") for p in parts if p.strip(" \n，,。")] or [text]


def memory_question(text):
    return bool(re.search(r"什么|啥|哪|谁|怎么|多少|是否|是不是|会不会|能不能|吗|呢|记得|记住了|保存了", text) and re.search(r"我|用户|主人", text) and re.search(
        r"记得|记住|保存|偏好|喜欢|最爱|名字|姓名|昵称|称呼|叫|代号|口令|颜色|喝|爱好", text))


def select_memory(question, records):
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
    return selected


def memory_reference(question, records, context=()):
    selected = select_memory(question, records)
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
                    return "之前记下的是：" + memory_phrase(row["key"], row["value"]) + "。不过你刚才说的是“" + match[1].strip() + "”，之前那条还没改。"
                return (memory_phrase(row["key"], match[1].strip()) + "，你刚才说的，我听到啦。"
                        + ("这次先按这个来，之前记下的那条还没改。" if selected else ""))
    if not selected:
        return "这件事我还没记下来。你再告诉我一次，好吗？"
    content = "；".join(memory_phrase(row['key'], row['value']) for row in selected)
    return content + "。我记着呢。"


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
        if records is not None and re.search(r"(?:用|根据|按).{0,12}我(?:的|喜欢|最爱|偏好|爱喝)", part):
            selected = select_memory(part, records)
            if selected:
                tasks.append(AnswerTask(part, "memory_chat", memory_reference(part, selected),
                                        {"literal_values": [row["value"] for row in selected]}))
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
                                        json.loads(meta.get("answer_contract", "{}")), [meta], meta.get("spoken_answer", "")))
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
    if re.search(r"我.{0,12}(?:认真研究过|研究过|亲自验证过|查阅过资料|调查过)", answer) and not re.search(
            r"研究过|验证过|查阅过|调查过", task.reference):
        problems.append("unsupported_personal_experience")
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


def check_voice(answer, task):
    problems = []
    if re.search(r"原作|资料显示|根据资料|按.*资料|设定中|事实卡|当前资料|已核实|此段|此处|序章", answer):
        problems.append("narrator")
    if re.search(r"(?:亚托莉|ATRI)(?:（亚托莉）)?(?:是|有|能|会|想|说|承认|搭载|具有|自称|在|对|不能|不擅长|并|表达)", answer, re.I):
        problems.append("self_third_person")
    fact_id = next((m.get("fact_id") for m in task.sources if m.get("fact_id")), None)
    if fact_id in {"identity", "high_performance", "eating", "taste", "housework", "school", "memory", "leg_role"} and re.search(r"她|(?:亚托莉|ATRI)的", answer, re.I):
        problems.append("self_third_person")
    return problems


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
    to a separately authored in-character answer, never to source narration.
    """
    answers, trace = [], []
    for task in tasks:
        if task.kind in {"memory", "unknown"}:
            answers.append(task.reference)
            trace.append({"kind": task.kind, "method": "direct_reference"})
            continue
        if task.kind == "memory_chat":
            # Allow natural chat, but do not send a draft that drops or changes
            # the specific saved values the user explicitly asked us to use.
            answer, method = task.reference, "memory_fallback"
            for attempt in range(2):
                prompt = system + VOICE + "\n用户要求用自己的信息回应。必须自然用到相应保存值，信息属于用户，不是你的喜好。"
                if attempt:
                    prompt += "上一稿漏用了信息或把归属说错，请按这次的要求重新回答。"
                msgs = [{"role": "system", "content": prompt},
                        *[dict(m) for m in history if m.get("role") != "system"],
                        {"role": "user", "content": task.question}]
                if prepare_chat is not None:
                    msgs = prepare_chat(msgs)
                else:
                    # Standalone callers also get literal data with instruction
                    # isolation, without needing access to a memory database.
                    from atri_memory import augment_records
                    msgs = augment_records(msgs, [{"key": "用户要求使用的信息", "value": v}
                                                  for v in task.contract["literal_values"]])
                draft = normalize_output(generate(msgs), task.question)
                wrong_owner = re.search(r"我(?:最|也|很|特别)?喜欢(?:喝|吃)?", draft)
                if (not wrong_owner and not check_voice(draft, task)
                        and all(v in draft for v in task.contract["literal_values"])):
                    answer, method = draft, "repair" if attempt else "model"
                    break
            answers.append(answer)
            trace.append({"kind": task.kind, "method": method})
            continue
        if task.kind == "fact" and not task.contract.get("required"):
            fallback = task.spoken_answer
            usable = fallback and not (check_answer(fallback, task) + check_voice(fallback, task))
            answers.append(fallback if usable else UNKNOWN)
            trace.append({"kind": "fact", "method": "voice_fallback" if usable else "unknown_fallback"})
            continue
        prompt = system + VOICE
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
        if task.kind in {"fact", "reference"} and (check_answer(answer, task) + check_voice(answer, task)):
            answer = normalize_output(generate([
                {"role": "system", "content": prompt + "\n请重新回答：保留事实，作为亚托莉直接说话，自己的事说‘我’，不要用旁白或资料介绍腔。"},
                {"role": "user", "content": task.question}]), task.reference + task.question)
            method = "repair"
            if check_answer(answer, task) + check_voice(answer, task):
                fallback = task.spoken_answer
                answer = fallback if fallback and not (check_answer(fallback, task) + check_voice(fallback, task)) else UNKNOWN
                method = "voice_fallback" if answer == fallback else "unknown_fallback"
        answers.append(answer or "主人，这个问题我暂时答不上来。")
        trace.append({"kind": task.kind, "method": method})
    return "\n".join(answers), trace
