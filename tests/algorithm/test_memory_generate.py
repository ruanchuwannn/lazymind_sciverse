import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _load_memory_generate_module():
    package_dir = (
        Path(__file__).resolve().parents[2]
        / 'algorithm/chat/pipelines/memory_generate'
    )
    spec = importlib.util.spec_from_file_location(
        'chat.pipelines.memory_generate',
        package_dir / '__init__.py',
        submodule_search_locations=[str(package_dir)],
    )
    assert spec is not None
    assert spec.loader is not None

    fake_lazyllm = ModuleType('lazyllm')
    fake_lazyllm.AutoModel = lambda *args, **kwargs: object()

    fake_pipelines_pkg = ModuleType('chat.pipelines')
    fake_pipelines_pkg.__path__ = []

    fake_skill_manager = ModuleType('chat.tools.skill_manager')
    fake_skill_manager._validate_skill_content = lambda *_args, **_kwargs: None

    fake_load_config = ModuleType('chat.utils.load_config')
    fake_load_config.get_config_path = lambda: ''

    original_modules = {
        'lazyllm': sys.modules.get('lazyllm'),
        'chat.pipelines': sys.modules.get('chat.pipelines'),
        'chat.tools.skill_manager': sys.modules.get('chat.tools.skill_manager'),
        'chat.utils.load_config': sys.modules.get('chat.utils.load_config'),
    }
    module_names = [
        name for name in list(sys.modules)
        if name == 'chat.pipelines.memory_generate'
        or name.startswith('chat.pipelines.memory_generate.')
    ]
    original_memory_generate_modules = {
        name: sys.modules.get(name)
        for name in module_names
    }

    try:
        sys.modules['lazyllm'] = fake_lazyllm
        sys.modules['chat.pipelines'] = fake_pipelines_pkg
        sys.modules['chat.tools.skill_manager'] = fake_skill_manager
        sys.modules['chat.utils.load_config'] = fake_load_config
        for name in module_names:
            sys.modules.pop(name, None)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        for name in module_names:
            sys.modules.pop(name, None)
        for name, original in original_memory_generate_modules.items():
            if original is not None:
                sys.modules[name] = original


memory_generate = _load_memory_generate_module()
BadRequestError = memory_generate.BadRequestError
MemoryGeneratePipeline = memory_generate.MemoryGeneratePipeline
UnprocessableContentError = memory_generate.UnprocessableContentError
_apply_edit_operations = memory_generate._apply_edit_operations
_build_generate_prompt = memory_generate._build_generate_prompt
_format_inputs_block = memory_generate._format_inputs_block
generate_memory_content = memory_generate.generate_memory_content


def _load_memory_generate_routes_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / 'algorithm/chat/app/api/memory_generate_routes.py'
    )
    spec = importlib.util.spec_from_file_location('test_memory_generate_routes', module_path)
    assert spec is not None
    assert spec.loader is not None

    fake_pipelines_pkg = ModuleType('chat.pipelines')
    fake_pipelines_pkg.__path__ = []

    original_modules = {
        'chat.pipelines': sys.modules.get('chat.pipelines'),
        'chat.pipelines.memory_generate': sys.modules.get('chat.pipelines.memory_generate'),
    }

    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules['chat.pipelines'] = fake_pipelines_pkg
        sys.modules['chat.pipelines.memory_generate'] = memory_generate
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.GeneratePayload.model_rebuild()
        return module
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_format_inputs_block_includes_only_suggestions_when_user_instruct_missing():
    block = _format_inputs_block(
        content='old content',
        suggestions=[{'title': 't', 'content': 'c'}],
        user_instruct=None,
    )

    assert '2) suggestions' in block
    assert '3) user_instruct' not in block


def test_format_inputs_block_includes_only_user_instruct_when_suggestions_missing():
    block = _format_inputs_block(
        content='old content',
        suggestions=[],
        user_instruct='rewrite this',
    )

    assert '2) user_instruct' in block
    assert '2) suggestions' not in block


def test_generate_memory_content_requires_suggestions_or_user_instruct():
    try:
        generate_memory_content(
            memory_type='memory',
            content='old content',
            suggestions=[],
            user_instruct='  ',
        )
    except BadRequestError as exc:
        assert "At least one of 'suggestions' or 'user_instruct' must be provided." == str(exc)
    else:
        raise AssertionError('Expected BadRequestError')


def test_generate_prompts_include_stale_content_governance():
    suggestions = [{
        'title': 'Update stale memory',
        'content': 'Replace old KB failure diagnosis with the current service-level cause.',
        'reason': 'Previous memory is outdated',
        'outdated': True,
    }]

    for memory_type in ('skill', 'memory', 'user_preference'):
        prompt = _build_generate_prompt(
            memory_type=memory_type,
            content='old content that may now be stale',
            suggestions=suggestions,
            user_instruct=None,
        )

        assert 'bounded, continuously maintained store' in prompt
        assert 'not an append-only log' in prompt
        assert 'Outdated=TRUE is only one stale signal' in prompt
        assert 'Even when the limit is not exceeded' in prompt
        assert 'proactively compress, consolidate, or delete stale information' in prompt
        assert 'Current content length after removing whitespace' in prompt
        assert 'Remaining budget before merging suggestions' in prompt


def test_skill_generate_prompt_uses_unified_edit_operations():
    prompt = _build_generate_prompt(
        memory_type='skill',
        content=(
            '---\n'
            'name: test-skill\n'
            'description: Old description\n'
            '---\n\n'
            '## Steps\n'
            '- Old step'
        ),
        suggestions=[{'title': 'Update', 'content': 'Add validation guidance.'}],
        user_instruct=None,
    )

    assert 'Preferred JSON structure is {"operations": [...]}' in prompt
    assert 'Supported operations are only replace_text and replace_all' in prompt
    assert 'replace_text is the primary edit path for skill drafts' in prompt
    assert 'Apply only the exact target explicitly requested' in prompt
    assert 'Do not infer related cleanup in other sections' in prompt
    assert 'Do not rewrite Usage, Examples, or neighboring sections' in prompt
    assert 'Prefer multiple small replace_text operations' in prompt
    assert 'Hard limit for replace_text old' in prompt
    assert 'never exceed 200 characters' in prompt
    assert 'Never use a whole section' in prompt
    assert 'Do not make a replace_text old value span multiple markdown sections' in prompt
    assert 'MUST NOT appear in new' in prompt
    assert 'never leave numbering gaps' in prompt
    assert 'do not output any operation for that missing target' in prompt
    assert 'delete, clear, or remove all skill content' in prompt
    assert 'mentally apply operations in order' in prompt
    assert 'old must be found exactly in the content state' in prompt
    assert 'Do not output no-op replace_text operations' in prompt
    assert 'You may output multiple replace_text operations' in prompt
    assert 'exact old text is absent, outdated, ambiguous' in prompt
    assert 'output full {"content": "..."} instead of operations' in prompt


def test_skill_generate_prompt_prefers_replace_text_for_single_line_deletion():
    prompt = _build_generate_prompt(
        memory_type='skill',
        content=(
            '---\n'
            'name: test-skill\n'
            'description: Old description\n'
            '---\n\n'
            '## Steps\n'
            '- Keep this\n'
            '- Delete this exact line'
        ),
        suggestions=[{
            'title': 'Remove one line',
            'content': 'Delete the line "- Delete this exact line".',
        }],
        user_instruct=None,
    )

    assert 'use replace_text only if you can copy the exact line' in prompt
    assert 'do not fabricate old text' in prompt
    assert 'do NOT update frontmatter description' in prompt
    assert 'Supported operations are only replace_text and replace_all' in prompt


def test_skill_edit_operations_can_delete_one_line_with_replace_text():
    current = (
        '---\n'
        'name: test-skill\n'
        'description: Keep description\n'
        '---\n\n'
        '## Steps\n'
        '- Keep this\n'
        '- Delete this exact line\n'
        '- Keep that'
    )

    edited = _apply_edit_operations(
        current,
        {
            'operations': [
                {
                    'op': 'replace_text',
                    'old': '- Delete this exact line\n',
                    'new': '',
                },
            ],
        },
        entity_name='skill',
    )

    assert edited == (
        '---\n'
        'name: test-skill\n'
        'description: Keep description\n'
        '---\n\n'
        '## Steps\n'
        '- Keep this\n'
        '- Keep that'
    )


def test_edit_operations_reject_unsupported_operations():
    with pytest.raises(UnprocessableContentError, match='Unsupported skill operation'):
        _apply_edit_operations(
            'old',
            {
                'operations': [
                    {
                        'op': 'append_text',
                        'content': '- New content',
                    },
                ],
            },
            entity_name='skill',
        )


def test_skill_edit_operations_preserve_legacy_content_payload():
    edited = _apply_edit_operations(
        'old',
        {
            'content': (
                '---\n'
                'name: test-skill\n'
                'description: New description\n'
                '---\n\n'
                '## Steps\n'
                '- New step'
            ),
        },
        entity_name='skill',
    )

    assert edited == (
        '---\n'
        'name: test-skill\n'
        'description: New description\n'
        '---\n\n'
        '## Steps\n'
        '- New step'
    )


def test_skill_generate_retries_with_common_content_when_operations_old_missing():
    class FakeLLM:
        def __init__(self):
            self.prompts = []

        def __call__(self, prompt):
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                return (
                    '{"operations":[{"op":"replace_text",'
                    '"old":"- Missing exact line","new":"- Replacement"}]}'
                )
            return (
                '{"content":"---\\n'
                'name: test-skill\\n'
                'description: Keep description\\n'
                '---\\n\\n'
                '## Steps\\n'
                '- Replacement"}'
            )

    pipeline = MemoryGeneratePipeline.__new__(MemoryGeneratePipeline)
    fake_llm = FakeLLM()
    pipeline.llm = fake_llm

    generated = pipeline.generate(
        memory_type='skill',
        content=(
            '---\n'
            'name: test-skill\n'
            'description: Keep description\n'
            '---\n\n'
            '## Steps\n'
            '- Original'
        ),
        suggestions=[{'title': 'Update', 'content': 'Replace the step.'}],
        user_instruct=None,
    )

    assert generated == (
        '---\n'
        'name: test-skill\n'
        'description: Keep description\n'
        '---\n\n'
        '## Steps\n'
        '- Replacement'
    )
    assert len(fake_llm.prompts) == 2
    assert 'Previous output was invalid' in fake_llm.prompts[1]
    assert 'replace_text could not find' in fake_llm.prompts[1]
    assert 'output full {"content": "..."} instead of operations' in fake_llm.prompts[1]


def test_skill_edit_operations_skip_missing_delete_targets():
    edited = _apply_edit_operations(
        '- Keep this',
        {
            'operations': [
                {
                    'op': 'replace_text',
                    'old': '- Missing line',
                    'new': '',
                },
                {
                    'op': 'replace_text',
                    'old': '- Keep this',
                    'new': '- Keep that',
                },
            ],
        },
        entity_name='skill',
    )

    assert edited == '- Keep that'


def test_skill_edit_operations_delete_numbered_line_and_normalize_list():
    edited = _apply_edit_operations(
        '1. First\n2. Second\n3. Remove this\n4. Fourth\n5. Fifth',
        {
            'operations': [
                {
                    'op': 'replace_text',
                    'old': '3. Remove this',
                    'new': '',
                },
                {
                    'op': 'replace_text',
                    'old': '5. Fifth',
                    'new': '4. Fifth',
                },
            ],
        },
        entity_name='skill',
    )

    assert edited == '1. First\n2. Second\n3. Fourth\n4. Fifth'


def test_memory_edit_operations_skip_missing_delete_targets():
    edited = _apply_edit_operations(
        '- Keep this',
        {
            'operations': [
                {
                    'op': 'replace_text',
                    'old': '- Missing line',
                    'new': '',
                },
            ],
        },
        entity_name='memory',
    )

    assert edited == '- Keep this'


def test_user_preference_edit_operations_skip_missing_delete_targets():
    edited = _apply_edit_operations(
        '- Prefers concise replies',
        {
            'operations': [
                {
                    'op': 'replace_text',
                    'old': '- Missing preference',
                    'new': '',
                },
            ],
        },
        entity_name='user_preference',
    )

    assert edited == '- Prefers concise replies'


def test_edit_operations_reject_missing_non_delete_targets():
    with pytest.raises(UnprocessableContentError, match='replace_text could not find'):
        _apply_edit_operations(
            '- Keep this',
            {
                'operations': [
                    {
                        'op': 'replace_text',
                        'old': '- Missing line',
                        'new': '- Replacement line',
                    },
                ],
            },
            entity_name='memory',
        )


def test_memory_edit_operations_can_replace_text():
    current = (
        '- 2026-05-14\n'
        '  用户在做:\n'
        '  - old task\n'
        '  状态/冲突:\n'
        '  - likes tea'
    )

    edited = _apply_edit_operations(
        current,
        {
            'operations': [
                {
                    'op': 'replace_text',
                    'old': 'likes tea',
                    'new': 'likes coffee',
                },
            ],
        },
        entity_name='memory',
    )

    assert edited == (
        '- 2026-05-14\n'
        '  用户在做:\n'
        '  - old task\n'
        '  状态/冲突:\n'
        '  - likes coffee'
    )


def test_edit_operations_can_apply_multiple_replace_text_operations():
    edited = _apply_edit_operations(
        '- Prefers verbose replies\n- Uses old workflow',
        {
            'operations': [
                {
                    'op': 'replace_text',
                    'old': 'verbose',
                    'new': 'concise',
                },
                {
                    'op': 'replace_text',
                    'old': 'old workflow',
                    'new': 'new workflow',
                },
            ],
        },
        entity_name='user_preference',
    )

    assert edited == '- Prefers concise replies\n- Uses new workflow'


def test_memory_edit_operations_can_clear_all_memory_via_replace_all():
    edited = _apply_edit_operations(
        '- 2026-05-14\n  用户在做:\n  - old task',
        {
            'operations': [
                {
                    'op': 'replace_all',
                    'content': '',
                },
            ],
        },
        entity_name='memory',
    )

    assert edited == ''


def test_memory_edit_operations_preserve_legacy_content_payload():
    edited = _apply_edit_operations(
        'old memory',
        {
            'content': '- 2026-05-26\n  我们讨论了:\n  - restored complete draft flow',
        },
        entity_name='memory',
    )

    assert edited == '- 2026-05-26\n  我们讨论了:\n  - restored complete draft flow'


def test_memory_edit_operations_reject_replace_all_with_extra_operations():
    with pytest.raises(UnprocessableContentError, match='replace_all must be the only operation'):
        _apply_edit_operations(
            '- 2026-05-25\n  我们讨论了:\n  - old',
            {
                'operations': [
                    {
                        'op': 'replace_all',
                        'content': '- 2026-05-26\n  我们讨论了:\n  - new full draft',
                    },
                    {
                        'op': 'replace_text',
                        'old': 'old',
                        'new': 'ignored because replace_all is complete',
                    },
                ],
            },
            entity_name='memory',
        )


def test_user_preference_edit_operations_preserve_legacy_content_payload():
    edited = _apply_edit_operations(
        'old preference',
        {
            'content': '- Prefers concise technical explanations',
        },
        entity_name='user_preference',
    )

    assert edited == '- Prefers concise technical explanations'


def test_user_preference_edit_operations_can_clear_all_content_via_replace_all():
    edited = _apply_edit_operations(
        'Prefers concise replies',
        {
            'operations': [
                {
                    'op': 'replace_all',
                    'content': '',
                },
            ],
        },
        entity_name='user_preference',
    )

    assert edited == ''


def test_memory_generate_route_accepts_suggestions_without_user_instruct(monkeypatch):
    memory_generate_routes = _load_memory_generate_routes_module()
    app = FastAPI()
    app.include_router(memory_generate_routes.router)
    client = TestClient(app)

    def fake_generate_memory_content(**kwargs):
        assert kwargs['suggestions'] == [{
            'title': 'Update',
            'content': 'Apply change',
            'reason': None,
            'outdated': None,
        }]
        assert kwargs['user_instruct'] is None
        return 'new content'

    monkeypatch.setattr(
        memory_generate_routes,
        'generate_memory_content',
        fake_generate_memory_content,
    )

    response = client.post(
        '/api/chat/memory/generate',
        json={
            'content': 'old content',
            'suggestions': [{'title': 'Update', 'content': 'Apply change'}],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        'code': 0,
        'msg': 'ok',
        'data': {'content': 'new content'},
    }


def test_memory_generate_route_rejects_missing_suggestions_and_user_instruct():
    memory_generate_routes = _load_memory_generate_routes_module()
    app = FastAPI()
    app.include_router(memory_generate_routes.router)
    client = TestClient(app)

    response = client.post(
        '/api/chat/memory/generate',
        json={'content': 'old content'},
    )

    assert response.status_code == 422
