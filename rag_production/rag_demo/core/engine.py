from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .confidence import evaluate_internal_results
from .config import RAGConfig
from .database import ProductionDatabase
from .model import GPUModelManager
from .prompter import Synthesizer
from .schemas import ConfidenceDecision, FullDocContext, RetrievedContext
from .tools import WebSearchTool

log = logging.getLogger(__name__)


class RAGEngine:
    """
    Production RAG Engine.

    Flow:
    1. Encode query with BAAI/bge-m3
    2. Hybrid search on child_chunks
    3. Retrieve parent_chunks
    4. Rerank with BAAI/bge-reranker-v2-m3
    5. Use web fallback only if internal evidence is weak/missing
    6. Fetch FULL parent document texts for all retrieved internal doc_ids
    7. Generate final answer using local open-source LLM via Ollama,
       passing full document texts so the LLM has maximum context
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.config.validate()

        self.db = ProductionDatabase(self.config)
        self.models = GPUModelManager(self.config)
        self.web = WebSearchTool(self.config)
        self.synthesizer = Synthesizer(self.config)

    def close(self) -> None:
        self.db.close()

    def retrieve_contexts(
        self,
        query: str,
        *,
        use_web: Optional[bool] = None,
        language: Optional[str] = None,
        doc_id: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> tuple[List[RetrievedContext], ConfidenceDecision, bool, Dict[str, Any]]:
        """
        Retrieve and rerank evidence without generating the final LLM answer.
        Useful for debugging retrieval quality.
        """

        query = (query or "").strip()

        if not query:
            decision = ConfidenceDecision(
                should_use_web=False,
                reason="Empty query.",
                confidence="none",
            )
            return [], decision, False, {"reason": "empty_query"}

        # 1. Query embedding
        query_vector = self.models.encode_query(query)

        # 2. Hybrid search on child_chunks, then parent expansion
        internal_contexts = self.db.hybrid_search(
            query_vector=query_vector,
            query_text=query,
            language=language,
            doc_id=doc_id,
            source_type=source_type,
        )

        # 3. Rerank internal contexts
        internal_reranked = self.models.rerank(
            query=query,
            contexts=internal_contexts,
            top_k=self.config.rerank_top_k,
        )

        # 4. Confidence decision
        decision = evaluate_internal_results(
            contexts=internal_reranked,
            config=self.config,
        )

        should_try_web = self.config.web_fallback_enabled and decision.should_use_web

        if use_web is True:
            should_try_web = True

        if use_web is False:
            should_try_web = False

        used_web = False
        final_contexts = internal_reranked

        # 5. Optional web fallback
        if should_try_web:
            log.info("Trying web fallback. Reason: %s", decision.reason)

            web_contexts = self.web.search(query)

            if web_contexts:
                used_web = True
                combined_contexts = internal_reranked + web_contexts

                final_contexts = self.models.rerank(
                    query=query,
                    contexts=combined_contexts,
                    top_k=self.config.rerank_top_k,
                )

        debug_info = {
            "internal_retrieved_count": len(internal_contexts),
            "internal_reranked_count": len(internal_reranked),
            "final_context_count": len(final_contexts),
            "confidence_reason": decision.reason,
            "confidence": decision.confidence,
            "web_fallback_enabled": self.config.web_fallback_enabled,
            "used_web": used_web,
            "embedding_model": self.config.embedding_model,
            "embedding_version": self.config.embedding_version,
            "reranker_model": self.config.reranker_model,
            "llm_provider": self.config.llm_provider,
            "ollama_model": self.config.ollama_model,
        }

        return final_contexts, decision, used_web, debug_info

    def answer_question(
        self,
        query: str,
        *,
        use_web: Optional[bool] = None,
        language: Optional[str] = None,
        doc_id: Optional[str] = None,
        source_type: Optional[str] = None,
        answer_style: Optional[str] = None,
        debug: bool = False,
    ) -> Dict[str, Any]:
        query = (query or "").strip()

        if not query:
            return {
                "answer": "Please provide a question.",
                "sources": [],
                "used_web": False,
                "confidence": "none",
                "debug": {"reason": "empty_query"} if debug else None,
            }

        log.info("RAG query started.")

        final_contexts, decision, used_web, debug_info = self.retrieve_contexts(
            query=query,
            use_web=use_web,
            language=language,
            doc_id=doc_id,
            source_type=source_type,
        )

        # Separate internal contexts from web contexts.
        # Full-doc fetching only applies to internal (DB) documents —
        # web results already carry their full snippet text.
        internal_contexts = [c for c in final_contexts if c.source_type == "internal"]
        web_contexts = [c for c in final_contexts if c.source_type != "internal"]

        # Fetch the complete text of every retrieved internal document.
        # This replaces the per-chunk snippet with the full parent doc content,
        # giving the LLM far more context to work with.
        full_docs: List[FullDocContext] = []
        if internal_contexts:
            log.info(
                "Fetching full parent docs for %d internal context(s).",
                len(internal_contexts),
            )
            full_docs = self.db.fetch_full_parent_docs(internal_contexts)
            debug_info["full_docs_fetched"] = len(full_docs)
            debug_info["full_doc_ids"] = [d.doc_id for d in full_docs]

        # If web results are present but no full docs, fall back to snippet mode
        # for web contexts so they are still included in the answer.
        # When both exist, full_docs take priority and web snippets are appended
        # to the context_text inside the Synthesizer automatically via the
        # fallback path in generate_response (contexts param still carries them).
        answer = self.synthesizer.generate_response(
            query=query,
            contexts=final_contexts,   # kept for web results & fallback
            used_web=used_web,
            full_docs=full_docs or None,
            answer_style=answer_style,
        )

        result: Dict[str, Any] = {
            "answer": answer,
            "sources": [context.to_public_dict() for context in final_contexts],
            "used_web": used_web,
            "confidence": decision.confidence,
        }

        if debug:
            result["debug"] = debug_info

        log.info("RAG query finished.")

        return result