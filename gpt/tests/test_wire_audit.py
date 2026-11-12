import json

import pytest

from gpt.wire_audit import PERMISSIONS, audit_trace
from wsarena.llm.base import Message, TurnRequest, render_transcript


def fixture(tmp_path):
    req = TurnRequest(run_id='test', turn_idx=1, model='gpt-6.1-sol', effort='medium', system='Follow the request exactly.',
                      messages=[Message(role='user', content='Reply ready.')], workdir=str(tmp_path))
    root = tmp_path/'trace'; bundle = root/'one'; bundle.mkdir(parents=True)
    (bundle/'manifest.json').write_text(json.dumps({'raw_event_log':'trace.jsonl'}))
    lines = [{'thread_id':'t','codex_turn_id':'r','payload':{'type':'inference_started','request_payload':{'path':'request.json'}}},
             {'payload':{'type':'recording_completed'}}]
    (bundle/'trace.jsonl').write_text('\n'.join(map(json.dumps, lines)))
    payload = {'tools':[], 'model':req.model, 'reasoning':{'effort':req.effort}, 'input':[
        {'type':'message','role':role,'content':[{'type':'input_text','text':text}]}
        for role,text in [('developer',req.system),('developer',PERMISSIONS),('user',render_transcript(req.messages))]]}
    return root, bundle/'request.json', req, payload


def test_exact_payload_passes(tmp_path):
    root, path, req, p = fixture(tmp_path); path.write_text(json.dumps(p))
    assert audit_trace(root, req)['tools_count'] == 0


def test_bundled_skills_do_not_allow_user_skill_injection(tmp_path):
    from gpt.codex_cli import unexpected_auth_home_content
    (tmp_path/'skills/.system').mkdir(parents=True)
    assert not unexpected_auth_home_content(tmp_path)
    (tmp_path/'skills/local').mkdir()
    assert unexpected_auth_home_content(tmp_path)


@pytest.mark.parametrize('change', ['tool','instruction','model','effort','input'])
def test_different_payload_rejected(tmp_path, change):
    root, path, req, p = fixture(tmp_path)
    if change == 'tool': p['tools'] = [{'name':'unexpected'}]
    elif change == 'instruction': p['instructions'] = 'unexpected'
    elif change == 'model': p['model'] = 'gpt-6-astra'
    elif change == 'effort': p['reasoning']['effort'] = 'high'
    else: p['input'][-1]['content'][0]['text'] += ' changed'
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError): audit_trace(root, req)
