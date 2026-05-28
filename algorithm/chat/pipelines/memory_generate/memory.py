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


def build_memory_prompt(
    content: str,
    suggestions: List[Dict[str, Any]],
    user_instruct: Optional[str],
    previous_error: Optional[str] = None,
) -> str:
    return (
        'You are an agent memory editor. Generate a JSON draft update based on the input; no explanations or summaries.\n'  # noqa: E501
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
        '- Only use replace_text for exact local edits whose old text is copied verbatim from current content.\n'
        '- Use replace_all, or top-level {"content": "..."}, when adding new entries, consolidating days, deleting stale entries, resolving conflicts, or changing structure.\n'  # noqa: E501
        '- Preferred final format after editing: group by day. Use one top-level bullet per day, ideally `- YYYY-MM-DD`, then summarize that day under concise sub-lines such as `用户在做:`, `我们讨论了:`, and `状态/冲突:` when needed.\n'  # noqa: E501
        '- If the exact date is unknown, use the best available time anchor such as month, week, or relative session marker, but still merge nearby events together when they clearly belong to the same day or session window.\n'  # noqa: E501
        '- Treat each suggestion as one atomic memory event to absorb into the final compact memory, not as a ready-made final line that must be copied verbatim.\n'  # noqa: E501
        '- When many events happen in one day, merge them into one daily entry and keep only the main threads, decisions, and follow-up context. Do not create a long bullet list of every small action.\n'  # noqa: E501
        '- When merging, deduplicate and consolidate: combine same or similar working-memory items into a more accurate statement; do not stack duplicates.\n'  # noqa: E501
        '- Conflict handling: if a new suggestion clearly supersedes an older memory on the same topic, keep only the new conclusion and record it under `状态/冲突:` as `已更新:` or `已废弃旧方案:` when useful.\n'  # noqa: E501
        '- If conflicting information is still unresolved, keep only the current best summary and mark it as `待定:` or `当前倾向:` under `状态/冲突:`.\n'  # noqa: E501
        '- Keep language concise and objective; compress aggressively so memory remains a compact aide-memoire rather than a diary.\n'  # noqa: E501
        '\n'
        f'{COMMON_LANGUAGE_RULES}'
        '\n'
        f'{format_preservation_rules("entries")}'
        '\n'
        '[Length control]\n'
        f'- The final content must be within {MAX_MANAGED_CONTENT_CHARS} characters after removing all whitespace; if needed, reduce low-value details and keep only the most important concise entries.\n'  # noqa: E501
        f'{managed_content_governance_note(content, suggestions, MAX_MANAGED_CONTENT_CHARS)}'
        '\n'
        f'{format_prompt_tail(content, suggestions, user_instruct, EDIT_OUTPUT_SPEC, previous_error)}'
    )
