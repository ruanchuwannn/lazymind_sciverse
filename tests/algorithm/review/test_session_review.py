from __future__ import annotations

import os
import sys
from types import SimpleNamespace


_ALGO = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'algorithm')
_LAZYLLM_ROOT = os.path.join(_ALGO, 'lazyllm')
if _ALGO not in sys.path:
    sys.path.insert(0, _ALGO)
if _LAZYLLM_ROOT not in sys.path:
    sys.path.insert(0, _LAZYLLM_ROOT)


def test_memory_review_prompt_excludes_preferences_and_workflows():
    from review.session_review import build_session_review_prompt

    prompt = build_session_review_prompt(
        target='memory',
        current_content='',
    )

    assert 'ONLY for agent working memory' in prompt
    assert "memory(target='memory'" in prompt
    assert 'operations' in prompt
    assert 'Prefer replace_text whenever current content is non-empty' in prompt
    assert 'Do not output suggestions' in prompt
    assert 'Environment context' not in prompt
    assert 'Do NOT save multi-step reusable workflows' in prompt
    assert 'reusable workflows' in prompt


def test_user_preference_review_prompt_excludes_session_history():
    from review.session_review import build_session_review_prompt

    prompt = build_session_review_prompt(
        target='user_preference',
        current_content='',
    )

    assert 'ONLY for user_preference' in prompt
    assert "memory(target='user_preference'" in prompt
    assert "Do not call memory with target='memory'" in prompt


def test_review_session_runs_agent_with_memory_tool(monkeypatch):
    from review import session_review
    from review.session_review import ChatMessage, SessionReviewRequest

    calls = {}

    class FakeModel:
        def __init__(self, *args, **kwargs):
            calls['model_args'] = (args, kwargs)

    class FakeReactAgent:
        def __init__(self, **kwargs):
            calls['agent_kwargs'] = kwargs

        def __call__(self, prompt, llm_chat_history=None):
            calls['prompt'] = prompt
            calls['history'] = llm_chat_history
            fake_lazyllm.locals['_lazyllm_agent'] = {
                'completed': [
                    {
                        'function': {'name': 'memory'},
                        'tool_call_result': {
                            'success': True,
                            'result': {'persisted': 'memory_review'},
                        },
                    }
                ],
            }
            return '已保存。'

    fake_lazyllm = SimpleNamespace(
        AutoModel=FakeModel,
        globals={},
        locals={'_lazyllm_agent': {'completed': [{'stale': True}]}},
        tools=SimpleNamespace(agent=SimpleNamespace(ReactAgent=FakeReactAgent)),
    )
    fake_fs_module = SimpleNamespace(FS=object)
    fake_config = SimpleNamespace(config={'core_api_url': 'http://core', 'review_max_retries': 2})
    monkeypatch.setitem(sys.modules, 'lazyllm', fake_lazyllm)
    monkeypatch.setitem(sys.modules, 'lazyllm.tools.fs.client', fake_fs_module)
    monkeypatch.setitem(sys.modules, 'chat.tools.memory', SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        'chat.utils.load_config',
        SimpleNamespace(get_config_path=lambda: '/tmp/config.yaml'),
    )
    monkeypatch.setitem(sys.modules, 'config', fake_config)

    result = session_review.review_session(
        SessionReviewRequest(
            target='user_preference',
            session_id='sid-1',
            history=[ChatMessage(role='user', content='以后请用中文简洁回答')],
        )
    )

    assert result.submitted is True
    assert calls['agent_kwargs']['tools'] == ['memory']
    assert calls['history'] == [{'role': 'user', 'content': '以后请用中文简洁回答'}]
    assert fake_lazyllm.globals['agentic_config']['session_id'] == 'sid-1'
    assert fake_lazyllm.globals['agentic_config']['user_preference'] == ''


def test_review_session_reports_no_tool_submission(monkeypatch):
    from review import session_review
    from review.session_review import ChatMessage, SessionReviewRequest

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

    class FakeReactAgent:
        def __init__(self, **kwargs):
            pass

        def __call__(self, prompt, llm_chat_history=None):
            return 'Nothing to save.'

    fake_lazyllm = SimpleNamespace(
        AutoModel=FakeModel,
        globals={},
        locals={'_lazyllm_agent': {}},
        tools=SimpleNamespace(agent=SimpleNamespace(ReactAgent=FakeReactAgent)),
    )
    monkeypatch.setitem(sys.modules, 'lazyllm', fake_lazyllm)
    monkeypatch.setitem(sys.modules, 'lazyllm.tools.fs.client', SimpleNamespace(FS=object))
    monkeypatch.setitem(sys.modules, 'chat.tools.memory', SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        'chat.utils.load_config',
        SimpleNamespace(get_config_path=lambda: '/tmp/config.yaml'),
    )
    monkeypatch.setitem(
        sys.modules,
        'config',
        SimpleNamespace(config={'core_api_url': 'http://core', 'review_max_retries': 2}),
    )

    result = session_review.review_session(
        SessionReviewRequest(
            target='memory',
            session_id='sid-1',
            history=[ChatMessage(role='user', content='你好')],
        )
    )

    assert result.submitted is False
    assert result.agent_result == 'Nothing to save.'


def test_memory_tool_submission_can_be_read_from_tool_history():
    from review.session_review import _memory_tool_submitted

    assert _memory_tool_submitted(
        {
            'history': [
                {
                    'role': 'tool',
                    'name': 'memory',
                    'content': (
                        "{'success': True, "
                        "'result': {'persisted': 'memory_review'}}"
                    ),
                }
            ]
        }
    )


def test_failed_memory_tool_result_is_not_submitted():
    from review.session_review import _memory_tool_submitted

    assert not _memory_tool_submitted(
        {
            'completed': [
                {
                    'function': {'name': 'memory'},
                    'tool_call_result': {
                        'success': False,
                        'reason': 'session snapshot not found',
                    },
                }
            ]
        }
    )
