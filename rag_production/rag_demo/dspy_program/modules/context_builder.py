from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ...core.config import RAGConfig
    from ...core.schemas import FullDocContext, RetrievedContext

_CHARS_PER_TOKEN = 4
_PROMPT_OVERHEAD_CHARS = 5_000
_MIN_CHARS_PER_CHUNK = 1_500
_MAX_CHARS_PER_CHUNK = 10_000


@dataclass(frozen=True)
class BuiltContext:
    context_block: str
    citation_style: str
    source_scope: str


class ContextBuilder:
    def __init__(self, config: "RAGConfig") -> None:
        self.config = config

    def _chars_per_chunk(self) -> int:
        total_budget = self.config.ollama_num_ctx * _CHARS_PER_TOKEN
        available_budget = max(
            total_budget - _PROMPT_OVERHEAD_CHARS,
            _MIN_CHARS_PER_CHUNK,
        )
        per_chunk = available_budget // max(self.config.rerank_top_k, 1)
        return max(_MIN_CHARS_PER_CHUNK, min(_MAX_CHARS_PER_CHUNK, per_chunk))

    def build(
        self,
        *,
        contexts: List["RetrievedContext"],
        full_docs: Optional[List["FullDocContext"]],
        used_web: bool,
    ) -> BuiltContext:
        context_block, citation_style = self._build_context_block(contexts, full_docs)
        source_scope = (
            "internal knowledge base and web search"
            if used_web
            else "internal knowledge base"
        )
        return BuiltContext(
            context_block=context_block,
            citation_style=citation_style,
            source_scope=source_scope,
        )

    def _build_context_block(
        self,
        contexts: List["RetrievedContext"],
        full_docs: Optional[List["FullDocContext"]],
    ) -> tuple[str, str]:
        chars_per_chunk = self._chars_per_chunk()
        contexts = contexts[: self.config.prompt_context_limit]

        if full_docs:
            total_chars = sum(len(doc.full_text) for doc in full_docs)
            budget = chars_per_chunk * max(len(full_docs), 1)
            if total_chars <= budget:
                parts = []
                for index, doc in enumerate(full_docs, start=1):
                    parts.append(
                        f"[DOCUMENT {index}]\n"
                        f"Source: {doc.source}\n"
                        f"Doc ID: {doc.doc_id}\n\n"
                        f"{doc.full_text}"
                    )
                return "\n\n---\n\n".join(parts), "[DOCUMENT n]"

        parts = []
        for index, ctx in enumerate(contexts, start=1):
            child_text = ctx.best_child_text()
            parent_text = (ctx.text or "").strip()
            if child_text and child_text.strip() != parent_text:
                content = (
                    f"[Matched passage]\n{child_text}\n\n"
                    f"[Surrounding context]\n{parent_text}"
                )
            else:
                content = parent_text

            content = content[:chars_per_chunk]
            score_line = ""
            if ctx.rerank_score is not None:
                score_line = f"Relevance score: {ctx.rerank_score:.3f}\n"

            parts.append(
                f"[SOURCE {index}]\n"
                f"File: {ctx.source}\n"
                f"Doc ID: {ctx.doc_id}\n"
                f"{score_line}"
                f"Type: {ctx.source_type}\n\n"
                f"{content}"
            )

        return "\n\n---\n\n".join(parts), "[SOURCE n]"
