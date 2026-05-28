from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

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

EDIT_OUTPUT_SPEC = (
    'Output requirements:\n'
    '1. Output only a JSON object; no markdown code blocks, no extra text.\n'
    '2. Preferred JSON structure is {"operations": [...]}.\n'
    '3. Supported operations are only replace_text and replace_all.\n'
    '4. Do not output any operation except replace_text or replace_all.\n'
    '5. Prefer {"op":"replace_text","old":"<exact old text>","new":"<new text>"} for exact local edits when old is a non-empty substring copied verbatim from current content.\n'  # noqa: E501
    '6. You may output multiple replace_text operations; they will be applied in order.\n'
    '7. Before final output, mentally apply operations in order to the current content using exact plain string search.\n'  # noqa: E501
    '8. For every replace_text, old must be found exactly in the content state at the moment that operation runs.\n'  # noqa: E501
    '9. Keep every replace_text old value as short as safely possible: use one exact line for line deletion/replacement, or one exact phrase/sentence for wording edits.\n'  # noqa: E501
    '10. Never use a whole section, a heading plus body, multiple bullets/list items, or unrelated paragraphs as one replace_text old. Split the change into several smaller replace_text operations instead.\n'  # noqa: E501
    '11. Never paraphrase, summarize, translate, re-indent, normalize whitespace, or invent old; old must preserve the exact characters, punctuation, spaces, and line breaks from current content.\n'  # noqa: E501
    '12. replace_text always replaces the first matching occurrence only.\n'
    '13. For delete/remove suggestions, the target text may appear in old but MUST NOT appear in new. Do not add, restore, or reword text that a suggestion asks to delete.\n'  # noqa: E501
    '14. If a deletion target or quoted sentence from a suggestion is absent from current content, treat that deletion as already satisfied and do not output any operation for that missing target.\n'  # noqa: E501
    '15. Do not output no-op replace_text operations where old and new are identical.\n'
    '16. If the exact old text is absent, outdated, ambiguous, not copied verbatim, or not enough to apply all requested changes safely, output full {"content": "..."} instead of operations.\n'  # noqa: E501
    '17. You may also use {"op":"replace_all","content":"<new full text>"} for full replacement.\n'
    '18. If you use replace_all, it MUST be the only operation in the operations array; do not output replace_all together with any other operation.\n'  # noqa: E501
)


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


def format_retry_note(previous_error: Optional[str]) -> str:
    if not previous_error:
        return ''
    if 'replace_text could not find' in previous_error or "field 'old'" in previous_error:
        return (
            f'\nPrevious output was invalid, error: {previous_error}\n'
            'Correction requirement: do not retry with any replace_text operation unless each old value is copied '
            'verbatim from current content and can be found by exact plain string search. If a delete/remove target '
            'is already absent, omit that operation. If you cannot guarantee a non-delete edit, output full '
            '{"content": "..."} instead of operations.\n'
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
