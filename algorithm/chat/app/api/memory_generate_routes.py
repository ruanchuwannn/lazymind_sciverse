from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import JSONResponse
import lazyllm
from pydantic import BaseModel, ConfigDict, Field
from pydantic import model_validator

from chat.pipelines.memory_generate import (
    BadRequestError,
    MemoryType,
    UnprocessableContentError,
    generate_memory_content,
)
from chat.utils.load_config import inject_model_config

router = APIRouter()


class SuggestionPayload(BaseModel):
    model_config = ConfigDict(extra='forbid')

    title: str = Field(..., description='Suggestion title')
    content: str = Field(..., description='Natural language modification suggestion')
    reason: Optional[str] = Field(default=None, description='Reason for the suggestion')
    outdated: Optional[bool] = Field(default=None, description='Whether the suggestion is outdated')


class GeneratePayload(BaseModel):
    model_config = ConfigDict(extra='forbid')

    content: str = Field(..., description='Current full text of the target content')
    suggestions: Optional[List[SuggestionPayload]] = Field(
        default=None,
        description='List of suggestions to merge',
    )
    user_instruct: Optional[str] = Field(default=None, description='Natural language instruction directly from the user')  # noqa: E501
    llm_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description='Per-request model configuration loaded by core for the current user',
    )

    @model_validator(mode='after')
    def validate_generation_inputs(self) -> 'GeneratePayload':
        has_suggestions = bool(self.suggestions)
        has_user_instruct = bool(self.user_instruct and self.user_instruct.strip())
        if not has_suggestions and not has_user_instruct:
            raise ValueError("At least one of 'suggestions' or 'user_instruct' must be provided.")
        return self


def _ok(content: str) -> Dict[str, Any]:
    return {'code': 0, 'msg': 'ok', 'data': {'content': content}}


def _fail(status_code: int, msg: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={'code': status_code, 'msg': msg, 'data': None},
    )


def _init_generate_session(memory_type: MemoryType, model_config: Optional[Dict[str, Any]]) -> None:
    session_id = f'{memory_type}_generate_{uuid4().hex}'
    lazyllm.globals._init_sid(sid=session_id)
    lazyllm.locals._init_sid(sid=session_id)
    inject_model_config(model_config)


def _handle_generate(memory_type: MemoryType, payload: GeneratePayload):
    try:
        _init_generate_session(memory_type, payload.llm_config)
        generated = generate_memory_content(
            memory_type=memory_type,
            content=payload.content,
            suggestions=[s.model_dump() for s in payload.suggestions] if payload.suggestions else None,
            user_instruct=payload.user_instruct,
        )
        return _ok(generated)
    except BadRequestError as exc:
        return _fail(400, str(exc))
    except UnprocessableContentError as exc:
        return _fail(422, str(exc))
    except Exception as exc:
        return _fail(500, f'generate failed: {exc}')


@router.post('/api/chat/skill/generate', summary='Generate new skill content')
async def generate_skill(payload: GeneratePayload):
    return _handle_generate('skill', payload)


@router.post('/api/chat/memory/generate', summary='Generate new memory content')
async def generate_memory(payload: GeneratePayload):
    return _handle_generate('memory', payload)


@router.post('/api/chat/user_preference/generate', summary='Generate new user_preference content')
async def generate_user_preference(payload: GeneratePayload):
    return _handle_generate('user_preference', payload)
