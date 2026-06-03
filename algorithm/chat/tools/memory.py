from functools import wraps
from typing import Any, Dict, List, Literal, Optional

import lazyllm
from lazyllm import fc_register
from typing_extensions import TypedDict

from chat.pipelines.memory_generate.common import (
    UnprocessableContentError,
    apply_edit_operations,
    reject_unchanged_content,
    validate_generated_content,
)
from review.memory_review_db import insert_memory_review_record

DEFAULT_CORE_API_TIMEOUT = 30


def _tool_failure(tool_name: str, exc: Exception) -> Dict[str, Any]:
    return {
        'success': False,
        'reason': f'{tool_name} failed: {exc}',
        'error': str(exc),
        'error_type': type(exc).__name__,
    }


def _handle_tool_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            return _tool_failure(func.__name__, exc)

    return wrapper


def _agentic_config() -> Dict[str, Any]:
    config = lazyllm.globals.get('agentic_config') or {}
    return config if isinstance(config, dict) else {}


def _core_api_base_url(agentic_config: Optional[Dict[str, Any]] = None) -> str:
    config = agentic_config if isinstance(agentic_config, dict) else _agentic_config()
    return str(config.get('core_api_url'))


def _core_api_endpoint(path: str, agentic_config: Optional[Dict[str, Any]] = None) -> str:
    base_url = _core_api_base_url(agentic_config)
    normalized_path = '/' + path.lstrip('/')
    return f'{base_url}{normalized_path}'


def _session_id(agentic_config: Optional[Dict[str, Any]] = None) -> str:
    config = agentic_config if isinstance(agentic_config, dict) else _agentic_config()
    return str(config.get('session_id') or lazyllm.globals._sid or '').strip()


def _post_core_api(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    import requests

    config = _agentic_config()
    url = _core_api_endpoint(path, config)
    timeout = config.get('core_api_timeout', DEFAULT_CORE_API_TIMEOUT)
    with requests.sessions.Session() as session:
        session.trust_env = False
        response = session.post(url, json=payload, timeout=timeout)

    try:
        body = response.json()
    except ValueError:
        body = {'text': response.text}

    if not response.ok:
        msg = (
            body.get('msg') or body.get('message')
            if isinstance(body, dict)
            else response.text
        )
        raise RuntimeError(f'POST {url} failed with HTTP {response.status_code}: {msg}')

    if isinstance(body, dict) and body.get('code') not in (None, 0):
        msg = body.get('msg') or body.get('message') or body
        raise RuntimeError(f'POST {url} failed: {msg}')

    return {
        'persisted': 'core_api',
        'url': url,
        'response': body,
    }


class EditOperation(TypedDict, total=False):
    """JSON edit operation applied to the current memory or user_preference text.

    Fields:
        op (str, required): either ``replace_text`` or ``replace_all``.
        old (str, required for replace_text): exact substring to replace.
        new (str, required for replace_text): replacement text.
        content (str, required for replace_all): full replacement content.
    """

    op: str
    old: str
    new: str
    content: str


MemoryTarget = Literal['memory', 'user_preference']


def _current_content_for_target(agentic_config: Dict[str, Any], target: str) -> str:
    value = agentic_config.get(target)
    if isinstance(value, str):
        return value
    fallback = agentic_config.get('current_content')
    if isinstance(fallback, str):
        return fallback
    return ''


@fc_register('tool', execute_in_sandbox=False)
@_handle_tool_errors
def memory(
    target: MemoryTarget,
    operations: List[EditOperation],
) -> Dict[str, Any]:
    """Apply edit operations to memory or user_preference and submit a review row.

    Call this tool only after comparing the conversation with the current full
    target text. The tool applies the supplied JSON edit operations to that
    original text, validates the edited full text, and writes one pending row to
    the algorithm-side ``memory_review`` table. It returns status metadata only;
    it does not return the edited content.

    Args:
        target: Which buffer the edit operations belong to. ``'memory'`` is the
            agent's own working memory about the user's ongoing context and
            prior discussions; ``'user_preference'`` is the user profile /
            preference text.
        operations: Ordered JSON edit operations. Supported operations:

            - ``{"op": "replace_text", "old": "...", "new": "..."}``:
              replace the first exact ``old`` substring with ``new``. Prefer
              this whenever the current content is non-empty, including when
              adding a new entry to an existing section.
            - ``{"op": "replace_all", "content": "..."}``: replace the
              full original target text with ``content``. Use this only when
              the current content is empty, no exact substring can safely
              anchor the edit, or the update needs global deduplication,
              conflict resolution, or broader reorganization.
    """
    def _ok(result: Dict[str, Any]) -> Dict[str, Any]:
        return {'success': True, 'result': result}

    def _fail(reason: str) -> Dict[str, Any]:
        return {'success': False, 'reason': reason}

    tool_target = str(target).strip()
    if tool_target not in {'memory', 'user_preference'}:
        return _fail(
            f"Unknown target {target!r}; expected one of 'memory', 'user_preference'."
        )
    if not operations:
        return _fail("'operations' must be a non-empty list.")

    agentic_config = _agentic_config()
    session_id = _session_id(agentic_config)
    if not session_id:
        return _fail("'session_id' is required in agentic_config.")

    current_content = _current_content_for_target(agentic_config, tool_target)
    operation_payload = [dict(op) for op in operations]
    try:
        edited_content = apply_edit_operations(
            current_content,
            {'operations': operation_payload},
            entity_name=tool_target,
        )
        edited_content = reject_unchanged_content(
            tool_target,
            current_content,
            edited_content,
        )
        edited_content = validate_generated_content(tool_target, edited_content)
    except UnprocessableContentError as exc:
        return _fail(str(exc))

    result: Dict[str, Any] = {
        'target': tool_target,
        'status': 'success',
        'operation_count': len(operation_payload),
    }
    record = insert_memory_review_record(
        target=tool_target,
        session_id=session_id,
        source_content=current_content,
        content=edited_content,
        operations=operation_payload,
    )
    result.update({
        'persisted': 'memory_review',
        'record_id': record.get('id'),
        'review_status': record.get('review_status', 'pending'),
    })

    return _ok(result)
