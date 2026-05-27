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

_JSON_BLOCK_RE = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL)
_CODE_BLOCK_RE = re.compile(r'```(?:[a-zA-Z0-9_+-]+)?\s*(.*?)\s*```', re.DOTALL)
_THINK_BLOCK_RE = re.compile(r'<think>.*?</think\s*>', re.DOTALL | re.IGNORECASE)
_SINGLE_STRING_FIELD_RE = re.compile(
    r'^\{\s*"(?P<key>[^"\\]+)"\s*:\s*"(?P<value>(?:[^"\\]|\\.)*)"\s*,?\s*\}\s*$',
    re.DOTALL,
)

COMMON_OUTPUT_SPEC = (
    'Output requirements:\n'
    '1. Output only a JSON object; no markdown code blocks, no extra text.\n'
    '2. JSON structure must be {"content": "<new complete text>"}.\n'
    '3. content must be the final complete text after merging all valid input modification requests; do not provide only a patch.\n'  # noqa: E501
)

COMMON_LANGUAGE_RULES = (
    '[Language]\n'
    '- Determine the output language from the language used in current content, suggestions, and user_instruct.\n'
    '- If the majority of the input is in Chinese (简体中文), write the generated content in Chinese.\n'
    '- If the majority of the input is in English, write the generated content in English.\n'
    '- Be consistent: do not mix languages within the generated content.\n'
)


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
        for candidate in candidates:
            parsed = extract_single_string_field_object(candidate)
            if isinstance(parsed, dict):
                break

    if parsed is None:
        if last_error is not None:
            raise UnprocessableContentError(
                f'Model output is not valid JSON: {last_error}'
            ) from last_error
        raise UnprocessableContentError('Model output is not valid JSON.')

    if not isinstance(parsed, dict):
        raise UnprocessableContentError('Model output must be a JSON object.')
    return parsed


def extract_single_string_field_object(text: str) -> Optional[Dict[str, str]]:
    match = _SINGLE_STRING_FIELD_RE.match(text.strip())
    if not match:
        return None

    key = match.group('key').strip()
    raw_value = match.group('value').strip()
    if raw_value.endswith(','):
        raw_value = raw_value[:-1].rstrip()
    if len(raw_value) < 2 or not raw_value.startswith('"') or not raw_value.endswith('"'):
        return None

    inner = raw_value[1:-1]
    try:
        value = json.loads(f'"{inner}"')
    except json.JSONDecodeError:
        value = (
            inner.replace('\\"', '"')
            .replace('\\\\', '\\')
            .replace('\\r', '\r')
            .replace('\\n', '\n')
            .replace('\\t', '\t')
        )
    return {key: value}


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


def format_preservation_rules(entity: str) -> str:
    return (
        '[Content preservation rules (CRITICAL)]\n'
        f'- You MUST preserve ALL existing {entity} that are NOT explicitly targeted by suggestions or user_instruct.\n'  # noqa: E501
        f'- When a suggestion only affects one {entity}, keep all others IDENTICAL to the original (same wording, same order).\n'  # noqa: E501
        '- Do NOT rephrase, reformat, or reorganize anything that is not being changed.\n'
        '- If nothing in the current content needs to change for a particular part, copy it VERBATIM into your output.\n'  # noqa: E501
        '- Only remove content that is explicitly marked as outdated by a suggestion, or explicitly contradicted by user_instruct.\n'  # noqa: E501
    )


def format_prompt_tail(
    content: str,
    suggestions: List[Dict[str, Any]],
    user_instruct: Optional[str],
    output_spec: str = COMMON_OUTPUT_SPEC,
    previous_error: Optional[str] = None,
) -> str:
    return (
        f'{format_retry_note(previous_error)}'
        f'{format_inputs_block(content, suggestions, user_instruct)}'
        f'{output_spec}'
    )


def format_inputs_block(
    content: str,
    suggestions: List[Dict[str, Any]],
    user_instruct: Optional[str],
) -> str:
    sections = [
        'Input information:\n'
        '1) Current content (full old text):\n'
        f'{content}\n\n'
    ]

    next_index = 2
    if suggestions:
        sections.append(
            f'{next_index}) suggestions (JSON array; each item may contain an outdated field):\n'
            '- outdated=TRUE means the suggestion is expired and for reference only; ignore if irrelevant to the current modification.\n'  # noqa: E501
            '- outdated=FALSE or missing means the suggestion is still valid and content should be updated accordingly.\n'  # noqa: E501
            f'{json.dumps(suggestions, ensure_ascii=False)}\n\n'
        )
        next_index += 1

    if user_instruct:
        sections.append(
            f'{next_index}) user_instruct (direct user instruction):\n{user_instruct}\n\n'
        )

    return ''.join(sections)


def normalize_user_instruct(raw_user_instruct: Any) -> Optional[str]:
    if raw_user_instruct is None:
        return None
    if not isinstance(raw_user_instruct, str):
        raise BadRequestError("'user_instruct' must be a string when provided.")

    normalized = raw_user_instruct.strip()
    return normalized or None


def format_retry_note(previous_error: Optional[str]) -> str:
    if not previous_error:
        return ''
    if 'replace_text could not find' in previous_error or "field 'old'" in previous_error:
        return (
            f'\nPrevious output was invalid, error: {previous_error}\n'
            'Correction requirement: do not retry with any replace_text operation unless each old value is copied '
            'verbatim from current content and can be found by exact plain string search. If you cannot guarantee '
            'that, output full {"content": "..."} instead of operations.\n'
        )
    return f'\nPrevious output was invalid, error: {previous_error}\nPlease correct and regenerate.\n'


def compact_len(text: Any) -> int:
    return len(''.join(str(text).split()))


def managed_content_governance_note(
    content: str,
    suggestions: List[Dict[str, Any]],
    limit: int,
) -> str:
    suggestions_length = sum(
        compact_len(item.get('title', ''))
        + compact_len(item.get('content', ''))
        + compact_len(item.get('reason', ''))
        for item in suggestions
    )
    current_length = compact_len(content)
    remaining = limit - current_length
    return (
        f'- Current content length after removing whitespace: {current_length} characters.\n'
        f'- Suggestions total length after removing whitespace: {suggestions_length} characters.\n'
        f'- Remaining budget before merging suggestions: {remaining} characters.\n'
        '- Treat existing content as a bounded, continuously maintained store, not an append-only log.\n'  # noqa: E501
        '- Outdated=TRUE is only one stale signal; also remove or rewrite existing content that is proven outdated, wrong, conflicting, redundant, overly specific, or low-value based on the new suggestions, user_instruct, or current context.\n'  # noqa: E501
        '- Even when the limit is not exceeded, proactively compress, consolidate, or delete stale information instead of preserving it by default.\n'  # noqa: E501
        '- Add new information only after resolving stale or conflicting old information; keep the final content concise and useful.\n'  # noqa: E501
    )


def normalize_string_list(raw: Any, *, field_name: str) -> List[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise UnprocessableContentError(f"Operation field '{field_name}' must be an array of strings.")

    normalized: List[str] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            raise UnprocessableContentError(
                f"Operation field '{field_name}[{idx}]' must be a non-empty string."
            )
        value = item.strip()
        if value not in normalized:
            normalized.append(value)
    return normalized


def apply_replace_text(current: str, old: str, new: str, *, entity_name: str) -> str:
    if old not in current:
        raise UnprocessableContentError(
            f"replace_text could not find the requested 'old' substring in current {entity_name} content. "
            'Please correct the old text or use replace_all if necessary.'
        )
    return current.replace(old, new, 1)
