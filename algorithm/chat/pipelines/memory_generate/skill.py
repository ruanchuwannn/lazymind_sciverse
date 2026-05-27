from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .common import (
    COMMON_LANGUAGE_RULES,
    UnprocessableContentError,
    apply_replace_text,
    format_preservation_rules,
    format_prompt_tail,
    managed_content_governance_note,
)

SKILL_EDIT_OUTPUT_SPEC = (
    'Output requirements:\n'
    '1. Output only a JSON object; no markdown code blocks, no extra text.\n'
    '2. Preferred JSON structure is {"content": "<new complete SKILL.md>"} so draft preview can always be generated.\n'  # noqa: E501
    '3. You may instead output {"operations": [...]} only when a small local edit can be targeted safely with exact old text copied verbatim from current content.\n'  # noqa: E501
    '4. operations is a list of edit commands that will be applied inside the generate endpoint and then validated as full SKILL.md.\n'  # noqa: E501
    '5. For operations, prefer {"op":"replace_text","old":"<exact old text>","new":"<new text>"} for small local edits; replace_text always replaces the first matching occurrence only.\n'  # noqa: E501
    '6. For deleting one exact line or phrase, use replace_text with old set to that exact text and new set to ""; do not update frontmatter or nearby sections for a deletion-only request.\n'  # noqa: E501
    '7. If the exact old text is absent, outdated, ambiguous, or not copied verbatim from current content, output full {"content": "..."} instead of operations.\n'  # noqa: E501
    '8. Use {"op":"update_frontmatter","fields":{"description":"<new description>"}} when only frontmatter fields need targeted updates; keep name unchanged unless user_instruct explicitly requests a rename.\n'  # noqa: E501
    '9. Use {"op":"replace_section","heading":"<exact markdown heading>","content":"<new section body>","mode":"replace"} for a focused body-section rewrite. Use mode="append" to append to that section.\n'  # noqa: E501
    '10. Use {"op":"replace_all","content":"<new full SKILL.md>"} only as a last resort when the current SKILL.md cannot be edited safely with local operations.\n'  # noqa: E501
    '11. If you use replace_all, it MUST be the only operation in the operations array; do not output replace_all together with any other operation.\n'  # noqa: E501
)


def build_skill_prompt(
    content: str,
    suggestions: List[Dict[str, Any]],
    user_instruct: Optional[str],
    previous_error: Optional[str] = None,
) -> str:
    return (
        'You are a SKILL.md editor. Generate the complete new SKILL.md content based on the input; no explanations or summaries.\n'  # noqa: E501
        'memory type: skill\n'
        'SKILL.md is an abstract SOP (Standard Operating Procedure) that guides the agent to complete tasks '
        'using a unified methodology when the description scope is satisfied.\n'
        '\n'
        '[Format requirements]\n'
        '1. Must start with YAML frontmatter containing at least name and description fields, '
        'followed by a blank line, then the markdown body.\n'
        '2. Keep the existing name value; do not rename unless user_instruct explicitly requests it.\n'
        '3. description should describe the applicable scope and trigger conditions in one sentence; '
        'this is the sole basis for routing/recalling this skill.\n'
        '\n'
        '[Scope and description linkage (important)]\n'
        '- When suggestions or user_instruct involve expanding/narrowing/adjusting the skill scope, trigger scenarios, or coverage, '  # noqa: E501
        'update the frontmatter description accordingly to accurately reflect the new scope.\n'
        '- When changes only affect methodology details in the body without changing the scope, keep description unchanged.\n'  # noqa: E501
        '- When the requested change is only deleting or editing one body line, do NOT update frontmatter description, title, tags, version, author, created, or updated.\n'  # noqa: E501
        '\n'
        '[Body content rules]\n'
        '- Return a complete final SKILL.md in `content` when that is the most reliable way to produce a valid draft preview.\n'  # noqa: E501
        '- You may use local operations for small edits when the exact target text or heading is present in current content. Keep untouched user-authored wording and structure exactly as-is whenever possible.\n'  # noqa: E501
        '- For a request like "delete/remove this line", use a single `replace_text` operation only if you can copy the exact line from current content as `old`; otherwise return full `content` with the requested deletion applied.\n'  # noqa: E501
        '- The body must be an abstract SOP: steps, decision criteria, checklists, general rules, output format requirements, etc.\n'  # noqa: E501
        '- Do not include specific cases, project names, specific data, conversation snippets, or one-time examples in the SKILL.md body; '  # noqa: E501
        'if examples are needed, use only highly abstract placeholder illustrations.\n'
        '- If suggestions or user_instruct contain specific cases, abstract the reusable experience into general rules '
        'before writing to the body; do not copy cases verbatim.\n'
        '- `replace_text` always replaces the first matching occurrence only. If first-match replacement is unsafe or too ambiguous, use replace_section or replace_all instead.\n'  # noqa: E501
        '- For `replace_section`, heading must exactly match an existing markdown heading line, such as `## Steps`; content should be the final section body without repeating the heading.\n'  # noqa: E501
        '- Use `replace_all` only when the current SKILL.md cannot be edited safely with local text replacement, frontmatter update, or section operations.\n'  # noqa: E501
        '- Recommended body structure: Applicable conditions / Steps / Judgment & validation / Common pitfalls / Output spec (trim as needed).\n'  # noqa: E501
        '\n'
        f'{COMMON_LANGUAGE_RULES}'
        '\n'
        f'{format_preservation_rules("body content")}'
        '\n'
        '[Length control]\n'
        '- Total length of SKILL.md (including frontmatter) must be within 2000 characters; keep it concise.\n'
        f'{managed_content_governance_note(content, suggestions, 2000)}'
        '\n'
        f'{format_prompt_tail(content, suggestions, user_instruct, SKILL_EDIT_OUTPUT_SPEC, previous_error)}'
    )


def parse_skill_operations(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if 'content' in payload and 'operations' not in payload:
        content = payload.get('content')
        if not isinstance(content, str):
            raise UnprocessableContentError("Generated field 'content' must be a string.")
        return [{'op': 'replace_all', 'content': content.strip()}]

    operations = payload.get('operations')
    if not isinstance(operations, list) or not operations:
        raise UnprocessableContentError("Model output for skill must contain a non-empty 'operations' array.")

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
        if op_name == 'update_frontmatter':
            fields = raw_op.get('fields')
            if not isinstance(fields, dict) or not fields:
                raise UnprocessableContentError("update_frontmatter requires a non-empty object field 'fields'.")
            normalized_fields: Dict[str, Any] = {}
            for key, value in fields.items():
                field_name = str(key).strip()
                if not field_name:
                    raise UnprocessableContentError('update_frontmatter field names must be non-empty strings.')
                if value is None:
                    raise UnprocessableContentError(
                        f"update_frontmatter field '{field_name}' must not be null."
                    )
                normalized_fields[field_name] = value
            normalized_ops.append({
                'op': 'update_frontmatter',
                'fields': normalized_fields,
            })
            continue
        if op_name == 'replace_section':
            heading = raw_op.get('heading')
            content = raw_op.get('content')
            mode = str(raw_op.get('mode') or 'replace').strip() or 'replace'
            if not isinstance(heading, str) or not heading.strip():
                raise UnprocessableContentError("replace_section requires a non-empty string field 'heading'.")
            if not isinstance(content, str):
                raise UnprocessableContentError("replace_section requires a string field 'content'.")
            if mode not in {'replace', 'append'}:
                raise UnprocessableContentError("replace_section mode must be 'replace' or 'append'.")
            normalized_ops.append({
                'op': 'replace_section',
                'heading': heading.strip(),
                'content': content.strip(),
                'mode': mode,
            })
            continue
        raise UnprocessableContentError(
            f"Unsupported skill operation {op_name!r}; expected 'replace_text', "
            "'update_frontmatter', 'replace_section' or 'replace_all'."
        )
    return normalized_ops


def split_skill_frontmatter(content: str) -> tuple[str, str, str]:
    match = re.match(r'^(---\s*\n)(.*?)(\n---\s*\n?)(.*)$', content or '', re.DOTALL)
    if not match:
        raise UnprocessableContentError('Current SKILL.md must contain YAML frontmatter for local edits.')
    opening, yaml_text, closing, body = match.groups()
    return opening, yaml_text, closing, body


def format_yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).strip()
    if not text:
        return '""'
    if re.match(r'^[A-Za-z0-9_./@+-]+(?: [A-Za-z0-9_./@+-]+)*$', text):
        return text
    return json.dumps(text, ensure_ascii=False)


def apply_frontmatter_update(current: str, fields: Dict[str, Any]) -> str:
    opening, yaml_text, closing, body = split_skill_frontmatter(current)
    lines = yaml_text.splitlines()
    consumed = set()

    for idx, line in enumerate(lines):
        match = re.match(r'^(\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*:\s*)(.*)$', line)
        if not match:
            continue
        key = match.group(2)
        if key not in fields:
            continue
        lines[idx] = f'{match.group(1)}{key}{match.group(3)}{format_yaml_scalar(fields[key])}'
        consumed.add(key)

    for key, value in fields.items():
        if key not in consumed:
            lines.append(f'{key}: {format_yaml_scalar(value)}')

    return f'{opening}{chr(10).join(lines)}{closing}{body}'


def heading_level(heading: str) -> int:
    match = re.match(r'^(#{1,6})\s+\S', heading.strip())
    if not match:
        raise UnprocessableContentError(
            "replace_section heading must be an exact markdown heading line such as '## Steps'."
        )
    return len(match.group(1))


def apply_section_edit(current: str, heading: str, content: str, mode: str) -> str:
    target_heading = heading.strip()
    target_level = heading_level(target_heading)
    lines = current.splitlines()
    start_idx: Optional[int] = None

    for idx, line in enumerate(lines):
        if line.strip() == target_heading:
            start_idx = idx
            break

    if start_idx is None:
        raise UnprocessableContentError(
            f"replace_section could not find heading {target_heading!r} in current skill content."
        )

    end_idx = len(lines)
    heading_re = re.compile(r'^(#{1,6})\s+\S')
    for idx in range(start_idx + 1, len(lines)):
        match = heading_re.match(lines[idx].strip())
        if match and len(match.group(1)) <= target_level:
            end_idx = idx
            break

    replacement_lines = content.splitlines() if content else []
    if mode == 'append':
        section_lines = lines[start_idx:end_idx]
        while section_lines and not section_lines[-1].strip():
            section_lines.pop()
        if replacement_lines:
            if len(section_lines) > 1:
                section_lines.append('')
            section_lines.extend(replacement_lines)
        new_lines = lines[:start_idx] + section_lines + lines[end_idx:]
    else:
        new_lines = lines[:start_idx + 1]
        if replacement_lines:
            new_lines.extend(replacement_lines)
            if end_idx < len(lines):
                new_lines.append('')
        new_lines.extend(lines[end_idx:])

    return '\n'.join(new_lines).strip()


def apply_skill_edit_operations(current_content: str, payload: Dict[str, Any]) -> str:
    operations = parse_skill_operations(payload)
    if operations[0]['op'] == 'replace_all':
        return operations[0]['content']

    current = current_content
    for op in operations:
        if op['op'] == 'replace_text':
            current = apply_replace_text(current, op['old'], op['new'], entity_name='skill')
            continue
        if op['op'] == 'update_frontmatter':
            current = apply_frontmatter_update(current, op['fields'])
            continue
        if op['op'] == 'replace_section':
            current = apply_section_edit(current, op['heading'], op['content'], op['mode'])
            continue
    return current.strip()
