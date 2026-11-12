"""Audit the pinned CLI's recorded generation payload, not just tool events."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from wsarena.llm.base import render_transcript

PERMISSIONS = """<permissions instructions>
Filesystem sandboxing defines which files can be read or written. `sandbox_mode` is `read-only`: The sandbox only permits reading files. Network access is restricted.
Approval policy is currently never. Do not provide the `sandbox_permissions` for any reason, commands will be rejected.
</permissions instructions>"""


def audit_trace(trace: Path, req):
    manifests = list(trace.glob('*/manifest.json'))
    if len(manifests) != 1:
        raise ValueError('expected one completed trace bundle')
    bundle = manifests[0].parent.resolve()
    def inside(name):
        p = (bundle/name).resolve()
        if not p.is_relative_to(bundle):
            raise ValueError('trace path escapes bundle')
        return p
    manifest = json.loads(manifests[0].read_text())
    events = [json.loads(line) for line in inside(manifest['raw_event_log']).read_text().splitlines()]
    requests = [e for e in events if e['payload']['type'] == 'inference_started']
    if not events or events[-1]['payload']['type'] != 'recording_completed' or len(requests) != 1:
        raise ValueError('missing final trace event or unexpected generation count')
    request = requests[0]
    raw = inside(request['payload']['request_payload']['path']).read_bytes()
    payload = json.loads(raw)
    if payload.get('tools') != []:
        raise ValueError('generation payload did not expose an empty tool table')
    if payload.get('model') != req.model or payload.get('reasoning', {}).get('effort') != req.effort:
        raise ValueError('model or effort in payload differs')
    if payload.get('instructions') or payload.get('previous_response_id'):
        raise ValueError('extra instructions or previous response in generation payload')
    visible = []
    for item in payload['input']:
        content = item.get('content')
        if item.get('type') != 'message' or not isinstance(content, list) or len(content) != 1 or content[0].get('type') != 'input_text':
            raise ValueError('unexpected input item shape')
        visible.append((item['role'], content[0]['text']))
    expected = [('developer', req.system), ('developer', PERMISSIONS), ('user', render_transcript(req.messages))]
    if visible != expected:
        raise ValueError('model-visible inputs differ from the expected role/text sequence')
    return {'request_sha256': hashlib.sha256(raw).hexdigest(), 'thread_id': request['thread_id'],
            'turn_id': request['codex_turn_id'], 'model': req.model, 'effort': req.effort,
            'tools_count': 0, 'input_exact': True, 'generation_requests': 1,
            'permission_note_sha256': hashlib.sha256(PERMISSIONS.encode()).hexdigest()}
