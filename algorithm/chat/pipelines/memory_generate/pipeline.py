from __future__ import annotations

from typing import Any, Dict, List, Optional

from lazyllm import AutoModel

from chat.tools.skill_manager import _validate_skill_content
from chat.utils.load_config import get_config_path

from .common import (
    MAX_GENERATE_ATTEMPTS,
    BadRequestError,
    MemoryType,
    UnprocessableContentError,
    extract_json_object,
    extract_skill_content,
    normalize_suggestions,
    normalize_user_instruct,
    validate_generated_content,
)
from .memory import apply_memory_edit_operations, build_memory_prompt
from .skill import apply_skill_edit_operations, build_skill_prompt
from .user_preference import (
    apply_user_preference_edit_operations,
    build_user_preference_prompt,
)

PROMPT_BUILDERS = {
    'skill': build_skill_prompt,
    'memory': build_memory_prompt,
    'user_preference': build_user_preference_prompt,
}


def build_generate_prompt(
    memory_type: MemoryType,
    content: str,
    suggestions: List[Dict[str, Any]],
    user_instruct: Optional[str],
    previous_error: Optional[str] = None,
) -> str:
    try:
        builder = PROMPT_BUILDERS[memory_type]
    except KeyError as exc:
        raise BadRequestError(f'Unsupported memory type: {memory_type!r}') from exc
    return builder(
        content=content,
        suggestions=suggestions,
        user_instruct=user_instruct,
        previous_error=previous_error,
    )


class MemoryGeneratePipeline:
    def __init__(self) -> None:
        self.llm = AutoModel(model='llm', config=get_config_path())

    def generate(
        self,
        memory_type: MemoryType,
        content: Any,
        suggestions: Optional[List[Dict[str, Any]]],
        user_instruct: Any,
    ) -> str:
        if not isinstance(content, str):
            raise BadRequestError("'content' is required and must be a string.")

        normalized_suggestions = normalize_suggestions(suggestions)
        normalized_user_instruct = normalize_user_instruct(user_instruct)
        if not normalized_suggestions and normalized_user_instruct is None:
            raise BadRequestError(
                "At least one of 'suggestions' or 'user_instruct' must be provided."
            )

        error: Optional[str] = None
        for _ in range(MAX_GENERATE_ATTEMPTS):
            prompt = build_generate_prompt(
                memory_type=memory_type,
                content=content,
                suggestions=normalized_suggestions,
                user_instruct=normalized_user_instruct,
                previous_error=error,
            )
            raw = self.llm(prompt)
            try:
                parsed = extract_json_object(raw)
                if memory_type == 'skill':
                    edited_content = apply_skill_edit_operations(content, parsed)
                    return validate_generated_content(memory_type, edited_content)
                if memory_type == 'memory':
                    edited_content = apply_memory_edit_operations(content, parsed)
                    return validate_generated_content(memory_type, edited_content)
                if memory_type == 'user_preference':
                    edited_content = apply_user_preference_edit_operations(content, parsed)
                    return validate_generated_content(memory_type, edited_content)
            except UnprocessableContentError as exc:
                if memory_type == 'skill':
                    skill_content = extract_skill_content(raw)
                    validation_error = _validate_skill_content(skill_content)
                    if validation_error is None:
                        return skill_content
                    error = f'{exc}; raw skill fallback invalid: {validation_error}'
                    continue
                error = str(exc)

        raise UnprocessableContentError(
            f'Failed to generate valid content after {MAX_GENERATE_ATTEMPTS} attempts: {error}'
        )


memory_generate_pipeline = MemoryGeneratePipeline()


def generate_memory_content(
    memory_type: MemoryType,
    content: Any,
    suggestions: Optional[List[Dict[str, Any]]],
    user_instruct: Any,
) -> str:
    return memory_generate_pipeline.generate(
        memory_type=memory_type,
        content=content,
        suggestions=suggestions,
        user_instruct=user_instruct,
    )
