from __future__ import annotations

import ast
from typing import Any, Dict, Iterable, List, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from chat.prompts.agentic import _MEMORY_REVIEW_PROMPT

ReviewTarget = Literal['memory', 'user_preference']


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra='allow')

    role: str = Field(
        ...,
        description='Message role, such as user, assistant, tool, or system',
    )
    content: str = Field(default='', description='Message content')


class SessionReviewRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    session_id: str = Field(..., description='Backend session ID being reviewed')
    target: ReviewTarget = Field(..., description='Review target')
    history: List[ChatMessage] = Field(
        default_factory=list,
        description='Chat history passed by backend for review',
    )
    current_content: str = Field(
        default='',
        description='Current full memory or user_preference text to edit',
    )

    @model_validator(mode='after')
    def validate_history(self) -> 'SessionReviewRequest':
        if not self.session_id.strip():
            raise ValueError("'session_id' must be non-empty.")
        if not any(
            message.role == 'user' and message.content.strip()
            for message in self.history
        ):
            raise ValueError("'history' must contain at least one user message.")
        return self


class SessionReviewResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    target: ReviewTarget
    session_id: str
    submitted: bool = False
    agent_result: str = ''


def _iter_tool_traces(agent_state: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    completed = agent_state.get('completed') or []
    if isinstance(completed, list):
        for item in completed:
            if isinstance(item, dict):
                yield item

    workspace = agent_state.get('workspace') or {}
    if isinstance(workspace, dict):
        trace = workspace.get('tool_call_trace') or []
        if isinstance(trace, list):
            for item in trace:
                if isinstance(item, dict):
                    yield item

    history = agent_state.get('history') or []
    if isinstance(history, list):
        for item in history:
            if not isinstance(item, dict) or item.get('role') != 'tool':
                continue
            yield {
                'function': {'name': item.get('name')},
                'tool_call_result': _parse_tool_result(item.get('content')),
            }


def _parse_tool_result(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _is_successful_memory_tool_result(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get('success') is not True:
        return False

    payload = result.get('result')
    if isinstance(payload, dict):
        return payload.get('persisted') in {'core_api', 'memory_review'}
    return result.get('persisted') in {'core_api', 'memory_review'}


def _memory_tool_submitted(agent_state: Dict[str, Any]) -> bool:
    for trace in _iter_tool_traces(agent_state):
        function = trace.get('function') or {}
        if not isinstance(function, dict) or function.get('name') != 'memory':
            continue
        if _is_successful_memory_tool_result(trace.get('tool_call_result')):
            return True
    return False


def _reset_agent_tool_trace(lazyllm: Any) -> None:
    lazyllm.locals['_lazyllm_agent'] = {}


def build_session_review_prompt(
    *,
    target: ReviewTarget,
    current_content: str,
) -> str:
    if target == 'memory':
        target_instruction = (
            "This backend-triggered review is ONLY for agent working memory. "
            "If saving is warranted, call memory(target='memory', operations=[...]). "
            "Do not call memory with target='user_preference'."
        )
    else:
        target_instruction = (
            "This backend-triggered review is ONLY for user_preference. "
            "If saving is warranted, call memory(target='user_preference', operations=[...]). "
            "Do not call memory with target='memory'."
        )

    existing_label = (
        'Current agent working memory'
        if target == 'memory'
        else 'Current user_preference'
    )
    return (
        f'{_MEMORY_REVIEW_PROMPT}\n\n'
        '# Backend-triggered target constraint\n'
        f'{target_instruction}\n'
        'For this endpoint, do not call skill_manage, get_skill, vocab_manage, '
        'or any tool except memory. Do not output suggestions.\n\n'
        'Do NOT save multi-step reusable workflows, troubleshooting procedures, '
        'lessons learned, tool usage patterns, implementation recipes, SOPs, '
        'or general task conventions as memory or user_preference. Those belong '
        'in skills, but this endpoint must only submit memory edit operations.\n\n'
        '# Required memory edit operation output\n'
        'When a durable update is warranted, output exactly one memory tool call '
        'with an operations array. Supported operations are:\n'
        '- replace_text: {"op": "replace_text", "old": "...", "new": "..."}; '
        "'old' MUST be an exact substring copied from the current content.\n"
        '- replace_all: {"op": "replace_all", "content": "..."}; use this '
        'only when current content is empty, or when the update truly requires '
        'rewriting the full target text.\n'
        'Prefer replace_text whenever current content is non-empty. For adding '
        'a new entry to existing content, replace the smallest exact existing '
        'section or block with the same block plus the new entry. Do not use '
        'replace_all merely because you are adding one item. Use replace_all '
        'only if no exact substring can safely anchor the edit, or the content '
        'needs global deduplication/conflict resolution/reorganization.\n'
        'The operations are applied to the current content below, and the edited '
        'full text is written to the memory_review table for human review. '
        'If no durable update is warranted, do not call memory; reply with '
        '`Nothing to save` and a brief reason.\n\n'
        '--- CURRENT CONTENT ---\n'
        f'## {existing_label}\n{current_content or ""}\n'
        '--- END CURRENT CONTENT ---\n\n'
        'The conversation to review is provided as llm_chat_history by the caller. '
        'Use that history as the source of truth.'
    )


def review_session(request: SessionReviewRequest) -> SessionReviewResult:
    import lazyllm
    from lazyllm import AutoModel
    from lazyllm.tools.fs.client import FS

    from chat.tools import memory as _memory_tool  # noqa: F401
    from chat.utils.load_config import get_config_path
    from config import config as _cfg

    prompt = build_session_review_prompt(
        target=request.target,
        current_content=request.current_content,
    )

    config = {
        'session_id': request.session_id,
        'core_api_url': _cfg['core_api_url'],
        'current_content': request.current_content,
    }
    if request.target == 'memory':
        config['memory'] = request.current_content
    else:
        config['user_preference'] = request.current_content
    lazyllm.globals['agentic_config'] = config

    llm = AutoModel(model='llm', config=get_config_path())
    review_agent = lazyllm.tools.agent.ReactAgent(
        llm=llm,
        tools=['memory'],
        max_retries=_cfg['review_max_retries'],
        return_trace=False,
        prompt=' ',
        keep_full_turns=3,
        fs=FS,
        enable_builtin_tools=False,
        force_summarize=True,
    )
    _reset_agent_tool_trace(lazyllm)
    raw = review_agent(
        prompt,
        llm_chat_history=[message.model_dump() for message in request.history],
    )
    agent_result = raw if isinstance(raw, str) else str(raw)
    agent_state = lazyllm.locals.get('_lazyllm_agent', {})
    return SessionReviewResult(
        target=request.target,
        session_id=request.session_id,
        submitted=_memory_tool_submitted(agent_state if isinstance(agent_state, dict) else {}),
        agent_result=agent_result,
    )
