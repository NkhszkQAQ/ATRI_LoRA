import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from atri_memory_lifecycle import LifecycleStore
from atri_runtime import Runtime, explicit_quotation_frame


def dialogue(announcement,current,role='user',middle=()):
 return [{'role':'system','content':'你是亚托莉。'}, {'role':role,'content':announcement},
         {'role':'assistant','content':'好，我听着。'},*middle,{'role':'user','content':current}]


def candidate(text):
 return {'subject':'user','predicate':'statement','value':text,'evidence':text,
         'polarity':'positive','degree':'ordinary','importance':.8,'stability':.9}


def generate(plan):
 def run(messages,*,kind):
  return json.dumps(plan,ensure_ascii=False) if kind=='plan' else '保留首次生成。'
 return run


@pytest.mark.parametrize('opening',[
 '接下来我会读一段戏剧人物的台词。','接下来的两句是戏剧台词。','下一句是我读的戏剧台词。',
 '以下是文章的原文。','下面引用一段对白。','随后我来转述访谈的转述内容。'])
def test_explicit_context_blocks_wrong_self_report(tmp_path,opening):
 store=LifecycleStore(tmp_path/'memory.db');text='我是老师。'
 plan={'utterance_kind':'self_report','candidates':[candidate(text)]}
 raw,trace=Runtime(store,generate(plan)).reply('fictional:a','1',dialogue(opening,text))
 assert store.records('fictional:a')==[]
 assert trace['memory_frame_guard']['evidence']==opening
 assert json.loads(trace['calls'][0]['raw_first'])==plan
 assert raw==trace['calls'][-1]['raw_first']=='保留首次生成。'


@pytest.mark.parametrize('text',['我喜欢画画','我的名字是小雨','我对花生过敏'])
def test_explicit_parser_cannot_bypass_active_frame(tmp_path,text):
 store=LifecycleStore(tmp_path/'memory.db')
 Runtime(store,generate({'utterance_kind':'self_report'})).reply('fictional:a','1',dialogue('以下是原文。',text))
 assert store.records('fictional:a')==[]


def test_wrong_model_delete_cannot_override_quotation(tmp_path):
 store=LifecycleStore(tmp_path/'memory.db');ident=store.observe('fictional:a','setup','我喜欢画画')['saved'][0]
 plan={'utterance_kind':'self_report','forget_ids':['m1']}
 _,trace=Runtime(store,generate(plan)).reply('fictional:a','1',dialogue('下面是对白。','请忘记我喜欢画画这件事'))
 assert store.records('fictional:a')[0]['id']==ident
 assert not any(x.get('status')=='deleted' for x in trace['operations'])


def test_quoted_correction_does_not_hide_or_change_real_fact(tmp_path):
 store=LifecycleStore(tmp_path/'memory.db')
 from atri_runtime import explicit_attributes
 text='我的生日是五月五日。';store.observe('fictional:a','setup',text,explicit_attributes(text,[]))
 old=store.records('fictional:a')[0]
 now='更正一下，我的生日是六月六日。';new=candidate('我的生日是六月六日');new.update(evidence=now,supersedes='m1')
 plan={'utterance_kind':'self_report','candidates':[new],'memory_scores':[{'id':'m1','relevance':1}]}
 _,trace=Runtime(store,generate(plan)).reply('fictional:a','1',dialogue('以下是原文。',now))
 assert store.records('fictional:a')[0]['id']==old['id']
 assert not trace.get('unresolved_corrections')
 assert trace['selected_memories'][0]['id']==old['id']


@pytest.mark.parametrize('ending',['引用结束。','台词到此结束。','结束转述。','我已经读完了。'])
def test_explicit_end_restores_normal_auto_save(tmp_path,ending):
 store=LifecycleStore(tmp_path/'memory.db')
 middle=[{'role':'user','content':'我喜欢画画'}, {'role':'assistant','content':'嗯。'}, {'role':'user','content':ending}]
 _,trace=Runtime(store,generate({'utterance_kind':'self_report'})).reply('fictional:a','1',dialogue('以下是引文。','我喜欢唱歌',middle=middle))
 assert [x['value'] for x in store.records('fictional:a')]==['唱歌']
 assert 'memory_frame_guard' not in trace


@pytest.mark.parametrize('opening',[
 '我喜欢读戏剧台词。','接下来不是戏剧台词。','接下来我不会读台词。','以下是原文吗？',
 '如果接下来是原文。','她说“接下来是台词”。','接下来是职业介绍。'])
def test_ordinary_or_uncertain_text_does_not_create_frame(opening):
 assert explicit_quotation_frame(dialogue(opening,'我是老师。')) is None


@pytest.mark.parametrize('role',['assistant','system'])
def test_non_user_text_cannot_open_frame(role):
 assert explicit_quotation_frame(dialogue('以下是原文。','我是老师。',role=role)) is None


def test_assistant_cannot_close_user_frame():
 middle=[{'role':'assistant','content':'引用结束。'}]
 assert explicit_quotation_frame(dialogue('以下是原文。','我是老师。',middle=middle)) is not None


def test_negated_or_quoted_end_cannot_close_frame():
 for ending in ['不要结束引用。','“引用结束。”','如果引用结束。']:
  assert explicit_quotation_frame(dialogue('以下是原文。','我是老师。',middle=[{'role':'user','content':ending}]))


def test_no_inferred_end_after_a_declared_sentence_count():
 middle=[{'role':'user','content':'我是老师。'},{'role':'assistant','content':'嗯。'}]
 assert explicit_quotation_frame(dialogue('下一句是戏剧台词。','我喜欢画画',middle=middle))


def test_new_conversation_does_not_inherit_frame(tmp_path):
 store=LifecycleStore(tmp_path/'memory.db');runtime=Runtime(store,generate({'utterance_kind':'self_report'}))
 runtime.reply('fictional:a','1',dialogue('以下是引文。','我喜欢画画'))
 runtime.reply('fictional:a','2',[{'role':'user','content':'我喜欢唱歌'}])
 assert [x['value'] for x in store.records('fictional:a')]==['唱歌']
 assert store.records('fictional:b')==[]
