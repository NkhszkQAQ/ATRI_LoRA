"""Control-flow smoke test; synthetic generators never count as model evidence."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import eval_v12_entrances as evaluator


def test_batch_covers_all_transports_and_leaves_semantic_review_pending(tmp_path,monkeypatch):
    import atri_generation
    import atri_model_identity
    import atri_rag
    import atri_story
    import serve_v1_1_8b as loader
    rows=[{'id':name,'stage':'daily','setup':[],
           'turns':[{'user':'我喜欢画画','expect':'fixture only'}],
           'fail_save':name=='dev.d.failure'} for name in evaluator.CASES]
    cases=tmp_path/'development.jsonl'
    cases.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    digest=evaluator.file_sha(cases)
    seal=tmp_path/'seal.json';seal.write_text(json.dumps({'files':{cases.name:{'sha256':digest}}}))
    review=tmp_path/'review.json';review.write_text(json.dumps({'ready_for_entrance_validation':True,
        'dataset_sha256':digest,'adapter_sha256':'fixture-not-real-adapter'}))
    system=tmp_path/'system.txt';system.write_text('你是亚托莉。')
    identity={'model':'test-double','adapter_sha256':'fixture-not-real-adapter'}
    monkeypatch.setattr(evaluator,'verify_loaded',lambda *a,**kw:dict(identity))
    monkeypatch.setattr(atri_model_identity,'verify_loaded',lambda *a,**kw:dict(identity))
    monkeypatch.setattr(loader,'load',lambda:loader.state.update(tok=SimpleNamespace(chat_template='fixture'),model=None))
    monkeypatch.setattr(evaluator,'version',lambda name:'test-double')
    monkeypatch.setattr(atri_generation,'hf_generator',lambda *a,**kw:
        lambda messages,kind:'{"utterance_kind":"self_report"}' if kind=='plan' else '未经改写的替身输出')
    monkeypatch.setattr(atri_rag,'LocalReranker',lambda *a:None)
    monkeypatch.setattr(atri_story,'StageCorpus',lambda *a,**kw:
        SimpleNamespace(source_hashes={},search=lambda *a:[]))
    out=tmp_path/'out'
    argv=['eval_v12_entrances','--adapter-sha','fixture-not-real-adapter']
    for name,path in {'base':tmp_path/'base','adapter':tmp_path/'adapter','base-manifest':tmp_path/'unused.json',
            'system':system,'seal':seal,'cases':cases,'script-source':tmp_path/'scripts','out':out,
            'development-review':review}.items():argv.extend(['--'+name,str(path)])
    monkeypatch.setattr(sys,'argv',argv)
    evaluator.main()
    completion=json.loads((out/'completion.json').read_text())
    assert completion['http_turns']==8*7 and completion['semantic_review']=='pending'
    assert completion['v1_2_accepted'] is False
    for path in out.glob('*/results.jsonl'):
        results=list(map(json.loads,path.read_text().splitlines()))
        assert len(results)==8 and all(r['final']=='未经改写的替身输出' for r in results)
        failure=next(r for r in results if r['case']=='dev.d.failure')
        assert failure['program_state']==[]
        assert any(op['status']=='failed' for op in failure['metadata']['operations'])
    assert len(list(out.glob('*/traces/*.json')))==56
