from __future__ import annotations

from typing import Any, Dict, List, Optional

from .common import (
    COMMON_LANGUAGE_RULES,
    MAX_MANAGED_CONTENT_CHARS,
    UnprocessableContentError,
    apply_replace_text,
    format_preservation_rules,
    format_prompt_tail,
    managed_content_governance_note,
)

USER_PREFERENCE_EDIT_OUTPUT_SPEC = (
    'Output requirements:\n'
    '1. Output only a JSON object; no markdown code blocks, no extra text.\n'
    '2. JSON structure must be {"operations": [...]}.\n'
    '3. operations is a list of edit commands that will be applied inside the generate endpoint and then rendered back to full user_preference text.\n'  # noqa: E501
    '4. When the current text is free-form or not clearly section-structured, prefer {"op":"replace_text","old":"<exact old text>","new":"<new text>"} for local edits.\n'  # noqa: E501
    '5. For every replace_text operation, old MUST be a non-empty substring copied verbatim from the current content input; before outputting, verify old can be found exactly by plain string search in current content.\n'  # noqa: E501
    '6. Never paraphrase, summarize, translate, re-indent, normalize whitespace, or invent old; old must preserve the exact characters, punctuation, spaces, and line breaks from current content.\n'  # noqa: E501
    '7. replace_text always replaces the first matching occurrence only. If that is not safe enough, use replace_all instead of trying to target a later match.\n'  # noqa: E501
    '8. You may output multiple replace_text operations and they will be applied in order.\n'
    '9. If the exact old text is absent, outdated, ambiguous, or not copied verbatim from current content, output full {"content": "..."} instead of operations.\n'  # noqa: E501
    '10. Use {"op":"replace_all","content":"<new full user_preference text>"} only as a last resort when the current text is too malformed or legacy to edit safely with local text replacement operations.\n'  # noqa: E501
    '11. If you use replace_all, it MUST be the only operation in the operations array; do not output replace_all together with any other operation.\n'  # noqa: E501
)


def build_user_preference_prompt(
    content: str,
    suggestions: List[Dict[str, Any]],
    user_instruct: Optional[str],
    previous_error: Optional[str] = None,
) -> str:
    return (
        'You are a user_preference editor. Generate the complete new user_preference content based on the input; no explanations or summaries.\n'  # noqa: E501
        'memory type: user_preference\n'
        'user_preference stores long-term stable user profile information, such as: user identity / role / domain, '
        'long-term preferences (communication tone, output format, language, level of detail), taboos, common workflow preferences, default context assumptions, etc.\n'  # noqa: E501
        '\n'
        '[Content boundaries]\n'
        '- Only record long-term stable profile information that can be reused in every future interaction.\n'
        '- Do not record specific experiences, specific project knowledge, or one-time events here; those belong to memory.\n'  # noqa: E501
        '- Do not write as chat logs or journals; organize as itemized profile entries that the agent can quickly read.\n'  # noqa: E501
        '\n'
        '[Writing and merging rules]\n'
        '- You do NOT output final user_preference text directly unless you must use replace_all. Normally you output edit operations that will be applied to the existing user_preference inside the generate endpoint.\n'  # noqa: E501
        "- If the current text is free-form, paragraph-based, or otherwise not clearly section-structured, prefer `replace_text` with an exact old substring and the desired new substring so the user's own writing structure is preserved.\n"  # noqa: E501
        '- `replace_text` always replaces the first matching occurrence only. If first-match replacement is unsafe or not enough, use `replace_all` instead.\n'  # noqa: E501
        '- Prefer small, local `replace_text` edits over rewriting the whole text. Keep untouched user-authored wording and structure exactly as-is whenever possible.\n'  # noqa: E501
        '- When preferences conflict, the new preference should replace the old text directly, and user_instruct takes precedence.\n'  # noqa: E501
        '- Keep language concise and neutral; no anthropomorphic comments; only state factual user profile entries.\n'
        '- Use `replace_all` only when the current content cannot be edited safely with local text replacement operations.\n'  # noqa: E501
        '\n'
        f'{COMMON_LANGUAGE_RULES}'
        '\n'
        f'{format_preservation_rules("profile entries")}'
        '\n'
        '[Length control]\n'
        f'- The final content must be within {MAX_MANAGED_CONTENT_CHARS} characters after removing all whitespace; if needed, reduce low-value details and keep only the most important concise entries.\n'  # noqa: E501
        f'{managed_content_governance_note(content, suggestions, MAX_MANAGED_CONTENT_CHARS)}'
        '\n'
        f'{format_prompt_tail(content, suggestions, user_instruct, USER_PREFERENCE_EDIT_OUTPUT_SPEC, previous_error)}'
    )


def parse_user_preference_operations(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if 'content' in payload and 'operations' not in payload:
        content = payload.get('content')
        if not isinstance(content, str):
            raise UnprocessableContentError("Generated field 'content' must be a string.")
        return [{'op': 'replace_all', 'content': content.strip()}]

    operations = payload.get('operations')
    if not isinstance(operations, list) or not operations:
        raise UnprocessableContentError("Model output for user_preference must contain a non-empty 'operations' array.")

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
        raise UnprocessableContentError(
            f"Unsupported user_preference operation {op_name!r}; expected 'replace_text' or 'replace_all'."
        )
    return normalized_ops


def apply_user_preference_edit_operations(current_content: str, payload: Dict[str, Any]) -> str:
    operations = parse_user_preference_operations(payload)
    if operations[0]['op'] == 'replace_all':
        return operations[0]['content']

    current = current_content
    for op in operations:
        if op['op'] == 'replace_text':
            current = apply_replace_text(current, op['old'], op['new'], entity_name='user_preference')
    return current.strip()
