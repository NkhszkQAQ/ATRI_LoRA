#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""微信聊天记录 -> 干净语料种子（v2：合并碎片 + 技术话题保留）

关键点（2026-11-03 主人补充）：
  1. 主人的说话习惯是「一句话拆成好多条消息发」——所以不能把每条微信消息当成一句话。
     一段连续发言要把这些碎片按原顺序拼回成一整段，才是他真正说的话。
  2. 技术话题保留：他和朋友日常就聊项目/代码/模型，这是真实语料，不要删。
     只去掉真正的隐私细节（家庭、住址、学校、证件、联系方式）。

输入（二选一，自动探测，也可 --source 指定）：
  A. WeChatExporter 的 chat.json
  B. WeChatExporter 的聊天记录.epub

输出：
  training/data/corpus/wx_seeds_clean.jsonl            主交付
     每行 {"intent": 场景, "seed": ["合并后的一整段话"]}
  training/data/corpus/wx_seeds_clean.review.jsonl     被剔除的隐私条目（带原因，供主人确认）

只用本地文件，不上传；不生成对话、不训练。
"""
import argparse
import html as htmlmod
import json
import os
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime

ME_WXID = '<用户wxid>'          # 用户侧
OTHER_WXID = '<对方wxid>'       # 对方

DEFAULT_ROOTS = [os.path.expanduser('~/Downloads/微信聊天记录导出')]
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, '..', 'data', 'corpus', 'wx_seeds_clean.jsonl')


# ============ 1. 数据源 ============

def load_chat_json(path):
    data = json.load(open(path, encoding='utf-8'))
    items = data.get('items') if isinstance(data, dict) else data
    out = []
    for it in items:
        c = it.get('content')
        if isinstance(c, dict):
            text = c.get('Text') or ''
            if not text:
                text = f'[{next(iter(c), "")}]'
        else:
            text = c if isinstance(c, str) else ''
        ts = it.get('create_time') or it.get('CreateTime') or 0
        try:
            ts = int(float(ts))
        except (TypeError, ValueError):
            ts = 0
        sender = it.get('sender') or it.get('wxid') or it.get('Sender') or ''
        out.append({'wxid': sender, 'ts': ts, 'text': text})
    out.sort(key=lambda m: m['ts'])
    return out


def load_epub(path):
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith('chapter1.xhtml'))
        src = z.read(name).decode('utf-8')
    blocks = re.split(r'<h2>(\d{4}-\d{2})</h2>', src)
    pat = re.compile(
        r'<p><span class="sender">(.*?)</span><span class="time">\[(\d\d)-(\d\d) (\d\d):(\d\d)\] </span>(.*?)</p>',
        re.S)
    out = []
    for i in range(1, len(blocks), 2):
        ym = blocks[i]
        for name_raw, _mm, dd, hh, mi, content in pat.findall(blocks[i + 1]):
            name_raw = htmlmod.unescape(name_raw)
            content = htmlmod.unescape(content)
            m = re.match(r'^(.*)（([^）]*)）$', name_raw)
            wxid = m.group(2) if m else ''
            dt = datetime.strptime(f'{ym}-{dd} {hh}:{mi}', '%Y-%m-%d %H:%M')
            out.append({'wxid': wxid, 'ts': int(dt.timestamp()),
                        'iso': dt.strftime('%Y-%m-%d %H:%M'), 'text': content})
    return out


def find_source(hint=None):
    if hint and os.path.exists(hint):
        return hint
    for root in DEFAULT_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            if 'chat.json' in files:
                return os.path.join(dirpath, 'chat.json')
            for f in files:
                if f.endswith('.epub') and '聊天记录' in f:
                    return os.path.join(dirpath, f)
    return None


# ============ 2. 单条消息可用性 ============

PLACE = re.compile(
    r'^\[(emoji|image|voice|video|file|link|location|转账|聊天记录|小程序|引用|拍一拍|system|type=|'
    r'表情|图片|语音|视频|位置|文件|链接|动画表情|音乐|红包|名片|卡券|直播|通话|实时位置|Text|Image|Voice|Video)')
BRACKET_ONLY = re.compile(r'^\[[^\]]{0,80}\]$')
EMOJI_ONLY = re.compile(r'^[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200d\s]+$')
PUNCT_ONLY = re.compile(
    r'^[\s\?？!！\.。、,，~～…\-—_/\\|:：;；\'"“”‘’()（）\[\]【】<>《》@#$%^&*+=`0-9]{1,}$')


# 噪音：长串 ID / 哈希、连续重复片段（hhhihihi）、大片乱码
JUNK_ID = re.compile(r'[A-Za-z0-9]{16,}')
REPEAT_CHUNK = re.compile(r'(.{1,3})\1{3,}')


def junk(t):
    return bool(JUNK_ID.search(t)) or bool(REPEAT_CHUNK.search(t))


def is_placeholder(t):
    return bool(PLACE.match(t)) or bool(BRACKET_ONLY.match(t))


def usable(t):
    """单条消息是否是可用文本；返回清洗后字符串或 None"""
    if not isinstance(t, str):
        return None
    s = t.strip()
    if not s:
        return None
    if is_placeholder(s) or EMOJI_ONLY.match(s) or PUNCT_ONLY.match(s):
        return None
    if '\n' in s:
        s = ' '.join(s.split())
    if len(s) > 200:                # 单条异常长，先挡掉
        return None
    return s


# ============ 3. 隐私词表 ============
# 只挡真正的隐私细节；技术/项目/代码/模型一律放行

PRIV_PRIVATE = re.compile(
    r'(我妈|我爸|我父母|我家里人|我家|住址|地址|小区|门牌|学校|班级|班里|学号|身份证|手机号|'
    r'手机号码|密码|银行卡|支付宝|微信支付|家庭住址|宿舍|寝室|班主任|辅导员|学院|专业|院系|'
    r'公司名|工位|微信号|加我|加好友|推他|搜不到|联系方式|家庭|换房子|换房|我们家)')

# 能反推出「主人在哪、要考哪、什么身份」的，剔除并标注，交主人确认
PRIV_IDENTITY = re.compile(
    r'(深圳|香港|上海|北京|广州|杭州|成都|南京|武汉|新加坡|法国|英国|日本|美国|'
    r'护照|签证|留学|机构|ucb|UCB|safe college|offer|Offer|录取|申请|出国)')


# 玩梗/固定口头禅：不算真实说话方式，剔除
MEME = ['受不了来', '受不了來']


def is_meme(t):
    return any(w in t for w in MEME)


# ============ 4. 场景分类 ============

SCENE = [
    ('晚安', ['晚安', '睡了', '困了', '困死', '我要睡', '去睡', '拜拜', '再见', '下线', '明天见',
              '先睡', '睡觉了', '睡吧', '该睡']),
    ('问候', ['早上好', '早安', '在吗', '在不在', 'hello', 'hi', '嗨', '你好', '哈喽', '起床',
              '在干嘛', '干嘛呢', '中午好', '下午好', '晚上好']),
    ('游戏', ['游戏', '打游戏', '上号', '开黑', '段位', 'kd', 'kda', 'go', 'csgo', 'apex', '三角洲',
              '巴雷特', '坦克', '枪', '对局', '排位', '战绩', '吃鸡', '开服', '赛季', '匹配', '队友',
              '打野', '辅助', '射手', '上分', '超模', '技能', '这局', '赢了', '输了', '带飞', 'carry',
              '野区', '装备', 'boss', '副本', '掉线', '挂机', '演员', '演我', '五排', '双排', '单排',
              '跳伞', '毒圈', '外挂', '死斗', '主线', 'cg', 'CG', '邦多利', '抽卡', '卡池', '氪金',
              '体力', 'buff', '打团', '开团', '上单', '中单', '下路', '对面', '开局', '残血', '回血',
              '5e', '上s', 'fcp', '剪映', '游戏素材', '手游', '电竞', '上份']),
    ('学习工作', ['上课', '作业', '报告', '论文', '考试', '复习', '老师', '课题', '答辩', '写完', '进度',
                  'ddl', 'deadline', '布置', '开会', '老板', '甲方', '需求', '方案', '稿子', '画稿',
                  '设计稿', '原型', '标书', '会议', '汇报', '提交', '交作业', '导师', '组会', '文献',
                  '实验', '放假', '开学', '课程', '专业课', '学分', '绩点', '任务', '项目', '排期',
                  '加班', '出差', '面试', '简历', '上班', '下班', '打工', '实习', '期中', '期末',
                  '自习', '图书馆', '笔记', 'precal', '考试周', '晚自习']),
    ('心情', ['累', '烦', '难受', '不开心', 'emo', '郁闷', '压力', '焦虑', '崩溃', '没头绪',
              '浑浑噩噩', '无聊', '孤独', '想哭', '烦死', '好累', '心累', '难过', '开心', '高兴',
              '爽', '自卑', '生气', '委屈', '害怕', '怕', '恐怖', '慌', '紧张', '释然', '感动',
              '后悔', '遗憾', '自闭', '破防', '心塞', '压抑', '烦躁', '疲惫', '沮丧', '失落',
              '兴奋', '激动', '期待', '无语', '服了', '舍不得', '尴尬', '道歉', '唉', '哎',
              '自信', '不敢', '心疼', '不自信', '烦人']),
    ('搞笑', ['哈哈哈', 'hhh', '笑死', '草', '离谱', '绷不住', '逆天', '抽象', '难绷', '乐死',
              '整活', '卧槽', '我操', '牛逼', '绝了', '太草', '乐子', '笑话', '好玩', '有趣',
              '搞笑', '绷', '乐了', 'dllm', '笑']),
    ('日常', ['吃', '饭', '饿', '渴', '外卖', '奶茶', '咖啡', '起床', '洗澡', '出门', '回家', '天气',
              '冷', '热', '雨', '太阳', '生日', '零食', '小龙虾', '西瓜', '水果', '面条', '火锅',
              '烧烤', '早饭', '午饭', '晚饭', '夜宵', '喝水', '午睡', '熬夜', '通宵', '刷牙',
              '洗脸', '洗头', '理发', '快递', '超市', '逛街', '电影', '电视剧', '番', '动漫', '听歌',
              '音乐', '睡', '剪头发', '头发', '机场', '航班', '飞机', '转机', '机翼', '登机',
              '虹桥', '浦东', '打车', '坐车', '地铁', '公交', '高铁', '火车', '妈妈', '奶奶',
              '室友', '旅游', '玩', '出去', '房子', '搬家', '电话', '打电话', '视频', '语音',
              '照片', '相机', '拍', '买']),
]


def classify(text):
    low = text.lower()
    scores = [(sum((2 if len(kw) >= 2 else 1) for kw in kws if kw.lower() in low), name)
              for name, kws in SCENE]
    scores.sort(reverse=True)
    return scores[0][1] if scores[0][0] > 0 else '其他'


# ============ 5. 主流程 ============

CJK = re.compile(r'[\u4e00-\u9fff]')


def join_parts(parts):
    """把碎片拼回一整段：中文之间不留空格，中英/数字边界留空格"""
    out = parts[0]
    for a, b in zip(parts, parts[1:]):
        need = not (CJK.search(a[-1]) and CJK.match(b[0]))
        out += (' ' if need else '') + b
    return out


def build(msgs, seg_gap=600, min_msgs=2, max_msgs=5,
          lo_len=12, hi_len=50, target=700, seed=20261103):
    # 按 >10 分钟切对话段
    segs, cur = [], [msgs[0]]
    for a, b in zip(msgs, msgs[1:]):
        if b['ts'] - a['ts'] > seg_gap:
            segs.append(cur)
            cur = [b]
        else:
            cur.append(b)
    segs.append(cur)

    # 每段里取「主人连续发的消息」——中间插了对方的话就断开
    runs = []
    for s in segs:
        buf = []
        for x in s:
            if x['wxid'] == ME_WXID:
                buf.append(x)
            else:
                if buf:
                    runs.append(buf)
                    buf = []
        if buf:
            runs.append(buf)

    stat = Counter()
    cands, review = [], []

    for r in runs:
        # 占位消息跳过，不打断整段（图片/表情夹在中间时继续拼）
        parts = [usable(x['text']) for x in r]
        parts = [p for p in parts if p]
        n = len(parts)
        if not (min_msgs <= n <= max_msgs):
            stat['n_msg'] += 1
            continue

        text = join_parts(parts)        # 碎片拼回一整段，不加标点、不改字
        L = len(text)
        if L < lo_len:
            stat['too_short'] += 1
            continue
        if L > hi_len:
            stat['too_long'] += 1
            continue
        if len(set(parts)) < max(2, n * 0.6):     # 刷屏
            stat['repeat'] += 1
            continue
        if junk(text):                            # 长串 ID / 连续重复噪音
            stat['junk'] += 1
            continue
        if is_meme(text):                         # 玩梗/固定口头禅
            stat['meme'] += 1
            continue

        # 隐私：直接剔除并记录
        if PRIV_PRIVATE.search(text):
            stat['privacy'] += 1
            review.append({'reason': 'privacy', 'iso': r[0].get('iso', ''), 'text': text})
            continue
        if PRIV_IDENTITY.search(text):
            stat['identity'] += 1
            review.append({'reason': 'identity_or_location', 'iso': r[0].get('iso', ''),
                           'text': text})
            continue

        cands.append({'ts': r[0]['ts'], 'iso': r[0].get('iso', ''),
                      'parts': parts, 'text': text})

    # 精确去重
    def norm(x):
        return re.sub(r'[\s\W_]+', '', x).lower()

    seen = set()
    uniq = []
    for c in sorted(cands, key=lambda x: x['ts']):
        k = norm(c['text'])
        if k in seen:
            stat['dup'] += 1
            continue
        seen.add(k)
        uniq.append(c)

    # 近似去重（同一句话换个顺序/加点语气词）
    out = []
    seen_sets = []
    for c in uniq:
        s = set(norm(p) for p in c['parts'])
        if any(len(s & s2) and len(s & s2) / len(s | s2) >= 0.85 for s2 in seen_sets):
            stat['dup_near'] += 1
            continue
        seen_sets.append(s)
        out.append(c)

    # 数量控制：超出上限时按月份分层抽样，保证 4~11 月各段都有
    if len(out) > target:
        rng = random.Random(seed)
        by_month = defaultdict(list)
        for c in out:
            by_month[c['iso'][:7]].append(c)
        total = len(out)
        picked = []
        for mo in sorted(by_month):
            items = by_month[mo]
            k = max(1, round(len(items) * target / total))
            picked += rng.sample(items, min(k, len(items)))
        picked.sort(key=lambda x: x['ts'])
        if len(picked) > target:
            picked = rng.sample(picked, target)
            picked.sort(key=lambda x: x['ts'])
        stat['sampled_out'] = total - len(picked)
        out = picked

    for c in out:
        c['intent'] = classify(c['text'])

    return out, review, stat, len(segs), len(runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', help='chat.json 或 .epub；不给则自动找')
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--review', default=None)
    ap.add_argument('--lo', type=int, default=12, help='合并后最小字数')
    ap.add_argument('--hi', type=int, default=50, help='合并后最大字数')
    ap.add_argument('--target', type=int, default=700, help='超过则分层抽样到此数')
    args = ap.parse_args()

    src = find_source(args.source)
    if not src:
        sys.exit('找不到数据源：用 --source 指定 chat.json 或 .epub')
    print(f'数据源: {src}')
    msgs = load_chat_json(src) if src.endswith('.json') else load_epub(src)
    print(f'消息数: {len(msgs)}')

    out, review, stat, n_seg, n_run = build(msgs, lo_len=args.lo, hi_len=args.hi,
                                            target=args.target)
    print(f'对话段(>10min): {n_seg} | 主人连续发言段: {n_run}')
    print(f'过滤统计: {dict(stat)}')
    print(f'场景分布: {Counter(c["intent"] for c in out).most_common()}')
    print(f'产出: {len(out)} 条')
    if not (500 <= len(out) <= 800):
        print(f'⚠️ 条数 {len(out)} 不在 500~800 区间，可调 --lo/--hi/--target')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for c in out:
            f.write(json.dumps({'intent': c['intent'], 'seed': [c['text']]},
                               ensure_ascii=False) + '\n')
    print(f'已写出: {args.out}')

    rp = args.review or (args.out + '.review.jsonl')
    with open(rp, 'w', encoding='utf-8') as f:
        for r in review:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'已写出待确认隐私条目: {rp}（{len(review)} 条）')


if __name__ == '__main__':
    main()
