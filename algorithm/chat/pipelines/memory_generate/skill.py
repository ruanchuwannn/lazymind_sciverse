from __future__ import annotations

from typing import Any, Dict, List, Optional

from .prompt_common import (
    COMMON_LANGUAGE_RULES,
    EDIT_OUTPUT_SPEC,
    format_preservation_rules,
    format_prompt_tail,
    managed_content_governance_note,
)


def build_skill_prompt(
    content: str,
    suggestions: List[Dict[str, Any]],
    user_instruct: Optional[str],
    previous_error: Optional[str] = None,
) -> str:
    return (
        'You are a SKILL.md editor. Generate a JSON draft update based on the input; no explanations or summaries.\n'  # noqa: E501
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
        '- replace_text is the primary edit path for skill drafts. Prefer multiple small replace_text operations over full replacement whenever the exact targets exist in current content.\n'  # noqa: E501
        '- Apply only the exact target explicitly requested by each suggestion or user_instruct. Do not infer related cleanup in other sections.\n'  # noqa: E501
        '- When a suggestion quotes a line, phrase, or word to remove/edit, modify only that quoted target and any necessary numbered-list renumbering.\n'  # noqa: E501
        '- Do not rewrite Usage, Examples, or neighboring sections unless a suggestion explicitly targets text in those sections.\n'  # noqa: E501
        '- Use replace_text only for exact local edits whose old text is copied verbatim from current content.\n'
        '- Keep every replace_text old value as short as safely possible: use one exact full line for line deletion/replacement, or one exact phrase/sentence for wording edits.\n'  # noqa: E501
        '- Hard limit for replace_text old: prefer 1 line; never include more than 1 newline; never exceed 200 characters unless the exact single line itself is longer.\n'  # noqa: E501
        '- Never use a whole section, a heading plus body, or multiple bullets/list items as one replace_text old. Edit each affected line or phrase separately.\n'  # noqa: E501
        '- Do not make a replace_text old value span multiple markdown sections, headings, or unrelated paragraphs. Split the change into several smaller replace_text operations instead.\n'  # noqa: E501
        '- For numbered-list deletion or insertion, use one replace_text to delete/insert the target line and separate replace_text operations to renumber each affected line; after deleting item N, renumber N+1 to N, N+2 to N+1, and so on. Never leave numbering gaps.\n'  # noqa: E501
        '- For delete/remove suggestions, the quoted target text may appear in old but MUST NOT appear in new. Do not add, restore, or reword text that a suggestion asks to delete.\n'  # noqa: E501
        '- If a deletion target or quoted sentence from a suggestion is absent from current content, treat that deletion as already satisfied and output {"op":"replace_text","old":"","new":""} for that missing target.\n'  # noqa: E501
        '- Only when the user explicitly asks to delete, clear, or remove all skill content, output an empty draft via full {"content": ""} or a single replace_all operation with empty content.\n'  # noqa: E501
        '- Example for deleting a numbered item: output one replace_text with old equal to the exact numbered line plus its newline and new equal to "", then separate replace_text operations for each later line number that must change.\n'  # noqa: E501
        '- For adding, removing, or rewriting sections, updating frontmatter, or any edit that cannot be safely targeted by exact short text replacement, output full {"content": "..."} or a single replace_all operation.\n'  # noqa: E501
        '- For a request like "delete/remove this line", use replace_text only if you can copy the exact line from current content as old; if the target is already absent, use old="" and new="" instead of fabricating old text.\n'  # noqa: E501
        '- The body must be an abstract SOP: steps, decision criteria, checklists, general rules, output format requirements, etc.\n'  # noqa: E501
        '- Do not include specific cases, project names, specific data, conversation snippets, or one-time examples in the SKILL.md body; '  # noqa: E501
        'if examples are needed, use only highly abstract placeholder illustrations.\n'
        '- If suggestions or user_instruct contain specific cases, abstract the reusable experience into general rules '
        'before writing to the body; do not copy cases verbatim.\n'
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
        f'{format_prompt_tail(content, suggestions, user_instruct, EDIT_OUTPUT_SPEC, previous_error)}'
    )
