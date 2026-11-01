"""Corrective RAG as a LangGraph StateGraph.

    START → contextualize → retrieve → grade ──relevant──► generate → END
                               ▲          │ nothing relevant & budget left
                               └─ rewrite ┘

* contextualize — turns "and for Enterprise?" into a standalone query using chat history;
* grade         — one LLM call filters irrelevant fragments (less noise → fewer hallucinations);
* rewrite       — if nothing relevant was found, try a reformulated query once;
* generate      — answer with [n] citations; citations are validated and mapped back to
                  sources in code, so a model cannot cite a fragment it never saw.

Pattern background: Corrective RAG (Yan et al., 2024, https://arxiv.org/abs/2401.15884)
and LangGraph's agentic RAG tutorial
(https://docs.langchain.com/oss/python/langgraph/agentic-rag).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import RetryPolicy

from rag_service import prompts
from rag_service.llm import structured
from rag_service.retrieval import Chunk
from rag_service.schemas import (
    Answer,
    Citation,
    GeneratedAnswer,
    RelevanceGrades,
    SearchQuery,
)

_CITE = re.compile(r"\[(\d+)\]")
HISTORY_WINDOW = 6


@dataclass(frozen=True)
class RagDeps:
    retrieve: Callable[[str], list[Chunk]]
    contextualize: Callable[[str, list[AnyMessage]], str]
    grade: Callable[[str, list[Chunk]], list[int]]
    rewrite: Callable[[str, str], str]
    generate: Callable[[str, list[Chunk]], GeneratedAnswer]


class RagState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # conversation memory
    question: str
    search_query: str
    candidates: list[Chunk]
    context: list[Chunk]
    rewrites: int
    answer: Answer


def format_fragments(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.id}] ({c.location()})\n{c.text}" for c in chunks)


def llm_deps(retrieve: Callable[[str], list[Chunk]], chat: BaseChatModel, fast: BaseChatModel):
    """Production implementations of the LLM steps."""

    def contextualize(question: str, history: list[AnyMessage]) -> str:
        transcript = "\n".join(f"{m.type}: {m.text}" for m in history)
        out = structured(fast, SearchQuery).invoke(
            [
                SystemMessage(prompts.CONTEXTUALIZE),
                HumanMessage(f"Conversation:\n{transcript}\n\nLatest question: {question}"),
            ]
        )
        return out.query

    def grade(question: str, chunks: list[Chunk]) -> list[int]:
        out = structured(fast, RelevanceGrades).invoke(
            [
                SystemMessage(prompts.GRADE),
                HumanMessage(f"Question: {question}\n\nFragments:\n{format_fragments(chunks)}"),
            ]
        )
        return out.relevant_ids

    def rewrite(question: str, failed_query: str) -> str:
        out = structured(fast, SearchQuery).invoke(
            [
                SystemMessage(prompts.REWRITE),
                HumanMessage(f"Question: {question}\nQuery that failed: {failed_query}"),
            ]
        )
        return out.query

    def generate(question: str, chunks: list[Chunk]) -> GeneratedAnswer:
        return structured(chat, GeneratedAnswer).invoke(
            [
                SystemMessage(prompts.ANSWER),
                HumanMessage(f"Fragments:\n{format_fragments(chunks)}\n\nQuestion: {question}"),
            ]
        )

    return RagDeps(retrieve, contextualize, grade, rewrite, generate)


def finalize_answer(generated: GeneratedAnswer, context: list[Chunk], query: str, rewrites: int):
    """Keep only citations that point to fragments the model actually received."""
    by_id = {c.id: c for c in context}
    inline = {int(n) for n in _CITE.findall(generated.answer)}
    cited = sorted((set(generated.cited_ids) | inline) & by_id.keys())
    text = _CITE.sub(lambda m: m.group(0) if int(m.group(1)) in by_id else "", generated.answer)
    return Answer(
        answer=text.strip(),
        citations=[
            Citation(
                n=i,
                location=by_id[i].location(),
                source=by_id[i].source,
                snippet=by_id[i].text[:300],
            )
            for i in cited
        ],
        insufficient_context=generated.insufficient_context or not cited,
        search_query=query,
        rewrites=rewrites,
    )


STATE_TYPES = [
    ("rag_service.retrieval", "Chunk"),
    ("rag_service.schemas", "Answer"),
    ("rag_service.schemas", "Citation"),
]


def make_checkpointer() -> InMemorySaver:
    """Conversation memory per thread_id. Production: PostgresSaver / RedisSaver."""
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=STATE_TYPES))


def build_rag_graph(
    deps: RagDeps,
    *,
    grade_documents: bool = True,
    max_rewrites: int = 1,
    checkpointer: Any = None,
):
    def contextualize(state: RagState) -> dict:
        history = state.get("messages", [])[-HISTORY_WINDOW:]
        q = state["question"]
        query = deps.contextualize(q, history) if history else q
        return {"search_query": query, "rewrites": 0, "candidates": [], "context": []}

    def retrieve(state: RagState) -> dict:
        return {"candidates": deps.retrieve(state["search_query"])}

    def grade(state: RagState) -> dict:
        candidates = state["candidates"]
        if not candidates or not grade_documents:
            relevant = candidates
        else:
            keep = set(deps.grade(state["question"], candidates))
            relevant = [c for c in candidates if c.id in keep]
        # renumber 1..n so the generator sees contiguous citation ids
        return {"context": [c.model_copy(update={"id": i}) for i, c in enumerate(relevant, 1)]}

    def route(state: RagState) -> Literal["generate", "rewrite"]:
        if state["context"] or state.get("rewrites", 0) >= max_rewrites:
            return "generate"
        return "rewrite"

    def rewrite(state: RagState) -> dict:
        return {
            "search_query": deps.rewrite(state["question"], state["search_query"]),
            "rewrites": state.get("rewrites", 0) + 1,
        }

    def generate(state: RagState) -> dict:
        q, ctx = state["question"], state["context"]
        if not ctx:  # no LLM call: refusing is free and can't hallucinate
            answer = Answer(
                answer=prompts.NO_CONTEXT_ANSWER,
                insufficient_context=True,
                search_query=state["search_query"],
                rewrites=state.get("rewrites", 0),
            )
        else:
            answer = finalize_answer(
                deps.generate(q, ctx), ctx, state["search_query"], state.get("rewrites", 0)
            )
        return {"answer": answer, "messages": [HumanMessage(q), AIMessage(answer.answer)]}

    retry = RetryPolicy(max_attempts=3, initial_interval=1.0)
    g = StateGraph(RagState)
    g.add_node("contextualize", contextualize, retry_policy=retry)
    g.add_node("retrieve", retrieve, retry_policy=retry)
    g.add_node("grade", grade, retry_policy=retry)
    g.add_node("rewrite", rewrite, retry_policy=retry)
    g.add_node("generate", generate, retry_policy=retry)
    g.add_edge(START, "contextualize")
    g.add_edge("contextualize", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route, ["generate", "rewrite"])
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    return g.compile(checkpointer=checkpointer, name="corrective_rag")
