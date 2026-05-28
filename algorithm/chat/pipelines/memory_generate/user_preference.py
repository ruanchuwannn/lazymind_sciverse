from __future__ import annotations

from typing import Any, Dict, List, Optional

from .common import MAX_MANAGED_CONTENT_CHARS
from .prompt_common import (
    COMMON_LANGUAGE_RULES,
    EDIT_OUTPUT_SPEC,
    format_preservation_rules,
    format_prompt_tail,
    managed_content_governance_note,
)


def build_user_preference_prompt(
    content: str,
    suggestions: List[Dict[str, Any]],
    user_instruct: Optional[str],
    previous_error: Optional[str] = None,
) -> str:
    return (
        'You are a user_preference editor. Generate a JSON draft update based on the input; no explanations or summaries.\n'  # noqa: E501
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
        '- Only use replace_text for exact local edits whose old text is copied verbatim from current content.\n'
        '- Use replace_all, or top-level {"content": "..."}, when adding new preferences, resolving conflicts, deleting stale profile entries, or changing structure.\n'  # noqa: E501
        '- Prefer small replace_text edits only when untouched user-authored wording and structure can be preserved exactly.\n'  # noqa: E501
        '- When preferences conflict, the new preference should replace the old text directly, and user_instruct takes precedence.\n'  # noqa: E501
        '- Keep language concise and neutral; no anthropomorphic comments; only state factual user profile entries.\n'
        '\n'
        f'{COMMON_LANGUAGE_RULES}'
        '\n'
        f'{format_preservation_rules("profile entries")}'
        '\n'
        '[Length control]\n'
        f'- The final content must be within {MAX_MANAGED_CONTENT_CHARS} characters after removing all whitespace; if needed, reduce low-value details and keep only the most important concise entries.\n'  # noqa: E501
        f'{managed_content_governance_note(content, suggestions, MAX_MANAGED_CONTENT_CHARS)}'
        '\n'
        f'{format_prompt_tail(content, suggestions, user_instruct, EDIT_OUTPUT_SPEC, previous_error)}'
    )
