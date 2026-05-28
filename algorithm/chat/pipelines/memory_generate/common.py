from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Optional

from chat.tools.skill_manager import _validate_skill_content

try:
    from json_repair import repair_json as _repair_json  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _repair_json = None

MemoryType = Literal['skill', 'memory', 'user_preference']

MAX_GENERATE_ATTEMPTS = 3
MAX_MANAGED_CONTENT_CHARS = 1400
# Core rejects empty strings; U+200B renders as a blank added line in draft diff.
EMPTY_DRAFT_PLACEHOLDER = '\u200b'

_JSON_BLOCK_RE = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL)
_CODE_BLOCK_RE = re.compile(r'```(?:[a-zA-Z0-9_+-]+)?\s*(.*?)\s*```', re.DOTALL)
_THINK_BLOCK_RE = re.compile(r'<think>.*?</think\s*>', re.DOTALL | re.IGNORECASE)


class BadRequestError(ValueError):
    """Raised when request body fields are missing or malformed."""


class UnprocessableContentError(ValueError):
    """Raised when generated content is repeatedly invalid."""


def normalize_suggestions(raw_suggestions: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if raw_suggestions is None:
        return []
    if not isinstance(raw_suggestions, list):
        raise BadRequestError("'suggestions' must be an array when provided.")

    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_suggestions):
        if not isinstance(item, dict):
            raise BadRequestError(f"'suggestions[{idx}]' must be an object.")

        title = item.get('title')
        content = item.get('content')
        reason = item.get('reason')
        outdated = item.get('outdated')

        if not isinstance(title, str) or not title.strip():
            raise BadRequestError(
                f"'suggestions[{idx}].title' must be a non-empty string."
            )
        if not isinstance(content, str) or not content.strip():
            raise BadRequestError(
                f"'suggestions[{idx}].content' must be a non-empty string."
            )
        if reason is not None and not isinstance(reason, str):
            raise BadRequestError(f"'suggestions[{idx}].reason' must be a string.")
        if outdated is not None and not isinstance(outdated, bool):
            raise BadRequestError(f"'suggestions[{idx}].outdated' must be a boolean.")

        normalized_item: Dict[str, Any] = {
            'title': title.strip(),
            'content': content.strip(),
        }
        if isinstance(reason, str) and reason.strip():
            normalized_item['reason'] = reason.strip()
        if outdated is not None:
            normalized_item['outdated'] = outdated
        normalized.append(normalized_item)
    return normalized


def extract_json_object(raw: Any) -> Dict[str, Any]:
    text = str(raw).strip()
    text = _THINK_BLOCK_RE.sub('', text).strip()

    match = _JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1).strip()

    candidates: List[str] = [text]
    left = text.find('{')
    right = text.rfind('}')
    if left >= 0 and right > left:
        trimmed = text[left: right + 1]
        if trimmed != text:
            candidates.append(trimmed)

    parsed: Any = None
    last_error: Optional[json.JSONDecodeError] = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            last_error = exc
    else:
        try:
            if _repair_json is None:
                raise ImportError('json_repair is not installed')
            for candidate in candidates:
                repaired = _repair_json(candidate, return_objects=True)
                if isinstance(repaired, dict):
                    parsed = repaired
                    break
        except Exception:
            pass

    if parsed is None:
        if last_error is not None:
            raise UnprocessableContentError(
                f'Model output is not valid JSON: {last_error}'
            ) from last_error
        raise UnprocessableContentError('Model output is not valid JSON.')

    if not isinstance(parsed, dict):
        raise UnprocessableContentError('Model output must be a JSON object.')
    return parsed


def extract_skill_content(raw: Any) -> str:
    text = str(raw).strip()
    text = _THINK_BLOCK_RE.sub('', text).strip()

    match = _CODE_BLOCK_RE.search(text)
    if match:
        text = match.group(1).strip()

    frontmatter_start = text.find('---')
    if frontmatter_start > 0:
        text = text[frontmatter_start:].strip()

    if text.endswith('```'):
        text = text[:-3].rstrip()
    return text


def validate_generated_content(memory_type: MemoryType, content: Any) -> str:
    if not isinstance(content, str):
        raise UnprocessableContentError("Generated field 'content' must be a string.")

    if not content.strip() or content == EMPTY_DRAFT_PLACEHOLDER:
        return EMPTY_DRAFT_PLACEHOLDER

    if memory_type == 'skill':
        validation_error = _validate_skill_content(content)
        if validation_error:
            raise UnprocessableContentError(
                f'Generated SKILL.md is invalid: {validation_error}'
            )
    elif memory_type in ('memory', 'user_preference'):
        compact_content = ''.join(content.split())
        content_length = len(compact_content)
        if content_length > MAX_MANAGED_CONTENT_CHARS:
            raise UnprocessableContentError(
                f'Generated content exceeds {MAX_MANAGED_CONTENT_CHARS} characters '
                f'after removing whitespace; current length is {content_length}. '
                f'Reduce the content length to {MAX_MANAGED_CONTENT_CHARS} characters '
                'or less after removing whitespace, keeping only the most important '
                'concise entries.'
            )
    return content


def reject_unchanged_content(memory_type: MemoryType, original: str, generated: str) -> str:
    if generated.strip() == original.strip():
        raise UnprocessableContentError(
            f'Generated {memory_type} content is unchanged from current content. '
            'A draft must contain at least one real content change.'
        )
    return generated


def build_noop_draft_content(content: str) -> str:
    base = content.strip()
    if not base:
        return EMPTY_DRAFT_PLACEHOLDER
    return f'{base}\n{EMPTY_DRAFT_PLACEHOLDER}'


def normalize_user_instruct(raw_user_instruct: Any) -> Optional[str]:
    if raw_user_instruct is None:
        return None
    if not isinstance(raw_user_instruct, str):
        raise BadRequestError("'user_instruct' must be a string when provided.")

    normalized = raw_user_instruct.strip()
    return normalized or None


def apply_replace_text(current: str, old: str, new: str, *, entity_name: str) -> str:
    if old not in current:
        raise UnprocessableContentError(
            f"replace_text could not find the requested 'old' substring in current {entity_name} content. "
            'Please correct the old text or use replace_all if necessary.'
        )
    return current.replace(old, new, 1)


def apply_replace_text_operation(current: str, old: str, new: str, *, entity_name: str) -> str:
    replacement = '' if not new.strip() else new
    if not replacement:
        lines = current.splitlines()
        for idx, line in enumerate(lines):
            if line == old:
                return '\n'.join(lines[:idx] + lines[idx + 1:])
    return apply_replace_text(current, old, replacement, entity_name=entity_name)


def normalize_numbered_lists(content: str) -> str:
    lines = content.splitlines()
    normalized: List[str] = []
    expected: Optional[int] = None
    last_indent: Optional[str] = None
    item_re = re.compile(r'^(\s*)(\d+)\.\s+(.*)$')

    for line in lines:
        match = item_re.match(line)
        if not match:
            normalized.append(line)
            if line.strip():
                expected = None
                last_indent = None
            continue

        indent, number, body = match.groups()
        if expected is None or indent != last_indent:
            expected = int(number)
            last_indent = indent
        normalized.append(f'{indent}{expected}. {body}')
        expected += 1

    return '\n'.join(normalized)


def parse_edit_operations(payload: Dict[str, Any], *, entity_name: str) -> List[Dict[str, Any]]:
    if 'content' in payload and 'operations' not in payload:
        content = payload.get('content')
        if not isinstance(content, str):
            raise UnprocessableContentError("Generated field 'content' must be a string.")
        return [{'op': 'replace_all', 'content': content.strip()}]

    operations = payload.get('operations')
    if not isinstance(operations, list) or not operations:
        raise UnprocessableContentError(
            f"Model output for {entity_name} must contain a non-empty 'operations' array."
        )

    normalized_ops: List[Dict[str, Any]] = []
    for idx, raw_op in enumerate(operations):
        if not isinstance(raw_op, dict):
            raise UnprocessableContentError(f"'operations[{idx}]' must be an object.")
        op_name = str(raw_op.get('op') or '').strip()
        if op_name == 'replace_all':
            content = raw_op.get('content')
            if not isinstance(content, str):
                raise UnprocessableContentError("replace_all requires a string field 'content'.")
            if len(operations) != 1:
                raise UnprocessableContentError('replace_all must be the only operation when used.')
            return [{'op': 'replace_all', 'content': content.strip()}]
        if op_name == 'replace_text':
            old = raw_op.get('old')
            new = raw_op.get('new')
            if not isinstance(old, str):
                raise UnprocessableContentError("replace_text requires a string field 'old'.")
            if not isinstance(new, str):
                raise UnprocessableContentError("replace_text requires a string field 'new'.")
            if old == '' and new != '':
                raise UnprocessableContentError(
                    "replace_text with an empty 'old' is only allowed when 'new' is also empty."
                )
            normalized_ops.append({
                'op': 'replace_text',
                'old': old,
                'new': new,
            })
            continue
        raise UnprocessableContentError(
            f"Unsupported {entity_name} operation {op_name!r}; expected 'replace_text' or 'replace_all'."
        )
    return normalized_ops


def apply_edit_operations(current_content: str, payload: Dict[str, Any], *, entity_name: str) -> str:
    operations = parse_edit_operations(payload, entity_name=entity_name)
    if operations[0]['op'] == 'replace_all':
        return operations[0]['content']

    current = current_content
    applied_delete = False
    skipped_noop = False
    for op in operations:
        if op['old'] == op['new']:
            if op['old'] == '':
                skipped_noop = True
            continue
        try:
            current = apply_replace_text_operation(
                current,
                op['old'],
                op['new'],
                entity_name=entity_name,
            )
            if not op['new'].strip():
                applied_delete = True
        except UnprocessableContentError:
            if not op['new'].strip():
                skipped_noop = True
                continue
            raise
    if applied_delete:
        current = normalize_numbered_lists(current)
    result = current.strip()
    if skipped_noop and result == current_content.strip():
        return build_noop_draft_content(result)
    return result
