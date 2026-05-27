from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from .common import (
    COMMON_LANGUAGE_RULES,
    MAX_MANAGED_CONTENT_CHARS,
    UnprocessableContentError,
    apply_replace_text,
    format_preservation_rules,
    format_prompt_tail,
    managed_content_governance_note,
    normalize_string_list,
)

_DATE_BULLET_RE = re.compile(r'^-\s+(.+?)(?::\s*(.*))?$')
_SECTION_HEADER_TO_KEY = OrderedDict((
    ('用户在做', 'doing'),
    ('我们讨论了', 'discussed'),
    ('状态/冲突', 'status'),
))
_SECTION_KEY_TO_HEADER = {v: k for k, v in _SECTION_HEADER_TO_KEY.items()}
_MEMORY_SECTION_KEYS = tuple(_SECTION_KEY_TO_HEADER.keys())

MEMORY_EDIT_OUTPUT_SPEC = (
    'Output requirements:\n'
    '1. Output only a JSON object; no markdown code blocks, no extra text.\n'
    '2. JSON structure must be {"operations": [...]}.\n'
    '3. operations is a list of edit commands that will be applied inside the generate endpoint and then rendered back to full memory text.\n'  # noqa: E501
    '4. Prefer {"op":"upsert_day","date":"YYYY-MM-DD","doing":[...],"discussed":[...],"status":[...],"replace":[...]} for structured day-level updates.\n'  # noqa: E501
    '5. You may use {"op":"replace_text","old":"<exact old text>","new":"<new text>"} for one local text fix; replace_text always replaces the first matching occurrence only.\n'  # noqa: E501
    '6. For every replace_text operation, old MUST be a non-empty substring copied verbatim from the current content input; before outputting, verify old can be found exactly by plain string search in current content.\n'  # noqa: E501
    '7. Never paraphrase, summarize, translate, re-indent, normalize whitespace, or invent old; old must preserve the exact characters, punctuation, spaces, and line breaks from current content.\n'  # noqa: E501
    '8. replace is optional and may contain any of ["doing","discussed","status"]; for listed sections, replace the old section summary for that day instead of merging.\n'  # noqa: E501
    '9. If the exact old text is absent, outdated, ambiguous, or not copied verbatim from current content, output full {"content": "..."} instead of operations.\n'  # noqa: E501
    '10. Use {"op":"replace_all","content":"<new full memory text>"} only as a last resort when the current memory is too malformed or legacy to edit safely with local text replacement or day-level operations.\n'  # noqa: E501
    '11. If you use replace_all, it MUST be the only operation in the operations array; do not output replace_all together with any other operation.\n'  # noqa: E501
)


def build_memory_prompt(
    content: str,
    suggestions: List[Dict[str, Any]],
    user_instruct: Optional[str],
    previous_error: Optional[str] = None,
) -> str:
    return (
        'You are an agent memory editor. Generate the complete new memory content based on the input; no explanations or summaries.\n'  # noqa: E501
        'memory type: memory\n'
        "memory stores the agent's own working memory about the user across sessions, such as: when a discussion happened, "  # noqa: E501
        'what the user and agent discussed, what the user was working on, ongoing context the agent may need to recall later, and other concise session-history facts.\n'  # noqa: E501
        'The input suggestions are candidate memory events, not final text patches. Your job is to merge those events into the existing memory and regenerate the full compact memory.\n'  # noqa: E501
        '\n'
        '[Content boundaries]\n'
        '- Only record concise working-memory entries with future recall value; do not write raw chat logs, full transcript summaries, pure emotional expressions, or unrelated small talk.\n'  # noqa: E501
        '- Do not record user profile information (identity, role, long-term preferences, communication style, etc.) here; those belong to user_preference.\n'  # noqa: E501
        '- Each entry should be self-contained and easy to scan: prefer a time anchor when known, then state what was discussed, what the user was doing, or what active context the agent should remember.\n'  # noqa: E501
        '\n'
        '[Writing and merging rules]\n'
        '- You do NOT output final memory text directly unless you must use replace_all. Normally you output edit operations that will be applied to the existing memory inside the generate endpoint.\n'  # noqa: E501
        '- If the current memory has local wording problems, slight format drift, or a user-edited phrase that should be corrected without rewriting the whole day structure, you may use `replace_text` for a local fix.\n'  # noqa: E501
        '- Preferred final format after editing: group by day. Use one top-level bullet per day, ideally `- YYYY-MM-DD`, then summarize that day under concise sub-lines such as `用户在做:`, `我们讨论了:`, and `状态/冲突:` when needed.\n'  # noqa: E501
        '- If the exact date is unknown, use the best available time anchor such as month, week, or relative session marker, but still merge nearby events together when they clearly belong to the same day or session window.\n'  # noqa: E501
        '- Treat each suggestion as one atomic memory event to absorb into the day summary, not as a ready-made final line that must be copied verbatim.\n'  # noqa: E501
        '- For day-level edits, prefer one `upsert_day` operation per affected day. Put concise final section summaries for that day into `doing`, `discussed`, and `status`.\n'  # noqa: E501
        '- Use `replace` to overwrite a section when new information should supersede the old summary for that day; omit `replace` when simple merge is enough.\n'  # noqa: E501
        '- When many events happen in one day, merge them into one daily entry and keep only the main threads, decisions, and follow-up context. Do not create a long bullet list of every small action.\n'  # noqa: E501
        '- When merging, deduplicate and consolidate: combine same or similar working-memory items into a more accurate statement; do not stack duplicates.\n'  # noqa: E501
        '- Conflict handling: if a new suggestion clearly supersedes an older memory on the same topic, keep only the new conclusion and record it under `状态/冲突:` as `已更新:` or `已废弃旧方案:` when useful.\n'  # noqa: E501
        '- If conflicting information is still unresolved, keep only the current best summary and mark it as `待定:` or `当前倾向:` under `状态/冲突:`.\n'  # noqa: E501
        '- Keep language concise and objective; compress aggressively so memory remains a compact aide-memoire rather than a diary.\n'  # noqa: E501
        '- `replace_text` always replaces the first matching occurrence only. If first-match replacement is unsafe or too ambiguous, use `replace_all` instead of trying to target a later occurrence.\n'  # noqa: E501
        '- Use `replace_all` only when the current content cannot be edited safely with local text replacement or day-level operations.\n'  # noqa: E501
        '\n'
        f'{COMMON_LANGUAGE_RULES}'
        '\n'
        f'{format_preservation_rules("entries")}'
        '\n'
        '[Length control]\n'
        f'- The final content must be within {MAX_MANAGED_CONTENT_CHARS} characters after removing all whitespace; if needed, reduce low-value details and keep only the most important concise entries.\n'  # noqa: E501
        f'{managed_content_governance_note(content, suggestions, MAX_MANAGED_CONTENT_CHARS)}'
        '\n'
        f'{format_prompt_tail(content, suggestions, user_instruct, MEMORY_EDIT_OUTPUT_SPEC, previous_error)}'
    )


def parse_memory_operations(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if 'content' in payload and 'operations' not in payload:
        content = payload.get('content')
        if not isinstance(content, str):
            raise UnprocessableContentError("Generated field 'content' must be a string.")
        return [{'op': 'replace_all', 'content': content.strip()}]

    operations = payload.get('operations')
    if not isinstance(operations, list) or not operations:
        raise UnprocessableContentError("Model output for memory must contain a non-empty 'operations' array.")

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
            if not isinstance(old, str) or not old:
                raise UnprocessableContentError("replace_text requires a non-empty string field 'old'.")
            if not isinstance(new, str):
                raise UnprocessableContentError("replace_text requires a string field 'new'.")
            normalized_ops.append({
                'op': 'replace_text',
                'old': old,
                'new': new,
            })
            continue

        if op_name != 'upsert_day':
            raise UnprocessableContentError(
                f"Unsupported memory operation {op_name!r}; expected 'replace_text', 'upsert_day' or 'replace_all'."
            )

        date = raw_op.get('date')
        if not isinstance(date, str) or not date.strip():
            raise UnprocessableContentError("upsert_day requires a non-empty string field 'date'.")

        replace = normalize_string_list(raw_op.get('replace'), field_name='replace')
        invalid_keys = [key for key in replace if key not in _MEMORY_SECTION_KEYS]
        if invalid_keys:
            raise UnprocessableContentError(
                f"replace contains unsupported sections: {', '.join(invalid_keys)}."
            )

        normalized_ops.append({
            'op': 'upsert_day',
            'date': date.strip(),
            'doing': normalize_string_list(raw_op.get('doing'), field_name='doing'),
            'discussed': normalize_string_list(raw_op.get('discussed'), field_name='discussed'),
            'status': normalize_string_list(raw_op.get('status'), field_name='status'),
            'replace': replace,
        })

    return normalized_ops


def append_unique(existing: List[str], values: List[str]) -> List[str]:
    merged = list(existing)
    for value in values:
        if value not in merged:
            merged.append(value)
    return merged


def new_day_record() -> Dict[str, List[str]]:
    return {key: [] for key in _MEMORY_SECTION_KEYS}


def parse_existing_memory(content: str) -> 'OrderedDict[str, Dict[str, List[str]]]':
    days: 'OrderedDict[str, Dict[str, List[str]]]' = OrderedDict()
    current_date: Optional[str] = None
    current_section: Optional[str] = None

    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        date_match = _DATE_BULLET_RE.match(stripped)
        if line.startswith('- ') and date_match:
            current_date = date_match.group(1).strip()
            current_section = None
            days.setdefault(current_date, new_day_record())
            inline_text = (date_match.group(2) or '').strip()
            if inline_text:
                days[current_date]['discussed'] = append_unique(
                    days[current_date]['discussed'],
                    [inline_text],
                )
            continue

        if current_date is None:
            continue

        header = stripped.rstrip(':')
        if header in _SECTION_HEADER_TO_KEY:
            current_section = _SECTION_HEADER_TO_KEY[header]
            continue

        bullet_value = stripped
        if stripped.startswith('- '):
            bullet_value = stripped[2:].strip()
        if current_section and bullet_value:
            days[current_date][current_section] = append_unique(
                days[current_date][current_section],
                [bullet_value],
            )

    return days


def render_memory(days: 'OrderedDict[str, Dict[str, List[str]]]') -> str:
    lines: List[str] = []
    for date, sections in days.items():
        has_content = any(sections.get(key) for key in _MEMORY_SECTION_KEYS)
        if not has_content:
            continue
        lines.append(f'- {date}')
        for key in _MEMORY_SECTION_KEYS:
            items = sections.get(key) or []
            if not items:
                continue
            lines.append(f'  {_SECTION_KEY_TO_HEADER[key]}:')
            for item in items:
                lines.append(f'  - {item}')
    return '\n'.join(lines).strip()


def apply_memory_edit_operations(current_content: str, payload: Dict[str, Any]) -> str:
    operations = parse_memory_operations(payload)
    if operations[0]['op'] == 'replace_all':
        return operations[0]['content']

    current = current_content
    days: Optional['OrderedDict[str, Dict[str, List[str]]]'] = None
    for op in operations:
        if op['op'] == 'replace_text':
            if days is not None:
                # Flush pending day-level edits before applying free-form text replacement.
                current = render_memory(days)
            current = apply_replace_text(current, op['old'], op['new'], entity_name='memory')
            days = None
            continue

        if days is None:
            days = parse_existing_memory(current)
        date = op['date']
        day = days.setdefault(date, new_day_record())
        replace = set(op.get('replace') or [])
        for key in _MEMORY_SECTION_KEYS:
            new_values = op.get(key) or []
            if not new_values and key not in replace:
                continue
            if key in replace:
                day[key] = list(new_values)
            else:
                day[key] = append_unique(day.get(key, []), new_values)

    if days is None:
        return current.strip()

    return render_memory(days)
