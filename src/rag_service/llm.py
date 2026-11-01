from __future__ import annotations

import inspect
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable


def init_llm(model: str, temperature: float | None = None) -> BaseChatModel:
    kwargs = {} if temperature is None else {"temperature": temperature}
    return init_chat_model(model, **kwargs)


def structured(llm: BaseChatModel, schema: type) -> Runnable[Any, Any]:
    """Native constrained decoding (`json_schema`) when the model profile supports it."""
    if "method" not in inspect.signature(llm.with_structured_output).parameters:
        return llm.with_structured_output(schema)
    profile = getattr(llm, "profile", None) or {}
    method = "json_schema" if profile.get("structured_output") else "function_calling"
    return llm.with_structured_output(schema, method=method)
