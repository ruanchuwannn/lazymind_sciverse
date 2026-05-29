from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from review.session_review import (
    ChatMessage,
    ReviewTarget,
    SessionReviewRequest,
    review_session,
)

router = APIRouter()


class SessionReviewPayload(BaseModel):
    model_config = ConfigDict(extra='forbid')

    session_id: str = Field(..., description='Backend session ID')
    history: List[ChatMessage] = Field(
        default_factory=list,
        description='Chat history passed by backend for review',
    )
    current_content: str = Field(
        default='',
        description='Current full target content, used to avoid duplicate suggestions',
    )
    max_suggestions: int = Field(default=5, ge=1, le=5)
    environment_context: Optional[Dict[str, Any]] = None
    llm_config: Dict[str, Any] = Field(
        ...,
        description='Required per-request model configuration loaded by core for the current user',
    )

    @model_validator(mode='after')
    def validate_payload(self) -> 'SessionReviewPayload':
        if not self.session_id.strip():
            raise ValueError("'session_id' must be non-empty.")
        if not any(
            message.role == 'user' and message.content.strip()
            for message in self.history
        ):
            raise ValueError("'history' must contain at least one user message.")
        if not self.llm_config:
            raise ValueError("'llm_config' must be a non-empty object.")
        return self


def _ok(data: Dict[str, Any]) -> Dict[str, Any]:
    return {'code': 0, 'msg': 'ok', 'data': data}


def _fail(status_code: int, msg: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={'code': status_code, 'msg': msg, 'data': None},
    )


def _init_review_session(
    session_id: str,
    target: ReviewTarget,
    model_config: Dict[str, Any],
) -> None:
    import lazyllm
    from chat.utils.load_config import inject_model_config

    sid = f'{target}_review_{session_id.strip() or uuid4().hex}'
    lazyllm.globals._init_sid(sid=sid)
    lazyllm.locals._init_sid(sid=sid)
    inject_model_config(model_config)


def _run_review(target: ReviewTarget, payload: SessionReviewPayload):
    try:
        _init_review_session(payload.session_id, target, payload.llm_config)
        request = SessionReviewRequest(
            target=target,
            session_id=payload.session_id,
            history=payload.history,
            current_content=payload.current_content,
            max_suggestions=payload.max_suggestions,
            environment_context=payload.environment_context,
        )
        result = review_session(request)
        return _ok(result.model_dump())
    except ValueError as exc:
        return _fail(422, str(exc))
    except Exception as exc:
        return _fail(500, f'session review failed: {exc}')


@router.post(
    '/api/chat/memory/review',
    summary='Review backend-provided history for memory suggestions',
)
async def review_memory(payload: SessionReviewPayload):
    return _run_review('memory', payload)


@router.post(
    '/api/chat/user_preference/review',
    summary='Review backend-provided history for user_preference suggestions',
)
async def review_user_preference(payload: SessionReviewPayload):
    return _run_review('user_preference', payload)
