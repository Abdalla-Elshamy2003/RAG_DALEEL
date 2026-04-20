from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .confidence import evaluate_internal_results
from .config import RAGConfig
from .database import ProductionDatabase
from .model import GPUModelManager
from .prompter import Synthesizer
from .query_processing import process_query
from .schemas import ConfidenceDecision, FullDocContext, RetrievedContext
from .tools import WebSearchTool

log = logging.getLogger(__name__)

# Only attempt to fetch full docs when the total document count is small
# AND the documents are likely to be short enough to fit the context window.
# For large docs (20-100 pages), the prompter will automatically fall back
# to reranked chunks — but we avoid the DB round-trip entirely here.
_FULL_DOC_MAX_UNIQUE_DOCS = 2
_FULL_DOC_MAX_CHARS_ESTIMATE = 30_000  # rough guard before we even try


class RAGEngine:
    """
    Production RAG Engine.

    Flow:
    1. Encode query with BAAI/bge-m3
    2. Hybrid search on child_chunks (vector + keyword + RRF)
    3. Retrieve parent_chunks
    4. Rerank with BAAI/bge-reranker-v2-m3
    5. Confidence decision based on reranker logit scores
    6. Web fallback only if internal confidence is low/none
    7. Optionally fetch full parent docs (only for small doc sets)
    8. Generate answer via Ollama (Qwen3), using reranked chunks as primary context
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

    def _should_use_general_fallback(
        self,
        *,
        decision: ConfidenceDecision,
        contexts: List[RetrievedContext],
        used_web: bool,
    ) -> bool:
        if not self.config.allow_general_knowledge_fallback or used_web:
            return False

        if not contexts:
            return True

        top_score = contexts[0].rerank_score if contexts and contexts[0].rerank_score is not None else 0.0
        return decision.confidence in {"none", "low"} and top_score < 0.1

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
        Retrieve and rerank evidence. Does not call the LLM.
        Useful for debugging retrieval quality and running offline evals.
        """
        query = (query or "").strip()

        if not query:
            decision = ConfidenceDecision(
                should_use_web=True,
                reason="Empty query.",
                confidence="none",
            )
            return [], decision, False, {"reason": "empty_query"}

        processed_query = process_query(query)

        # 1. Query embedding
        t0 = time.perf_counter()
        query_vector = self.models.encode_query(processed_query.retrieval_query)
        embedding_ms = round((time.perf_counter() - t0) * 1000, 1)

        # 2. Hybrid search: vector + keyword → RRF fusion → parent expansion
        t0 = time.perf_counter()
        internal_contexts = self.db.hybrid_search(
            query_vector=query_vector,
            query_text=processed_query.retrieval_query,
            keyword_query=processed_query.keyword_query,
            language=language,
            doc_id=doc_id,
            source_type=source_type,
        )
        retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)

        # 3. Rerank internal contexts with cross-encoder
        t0 = time.perf_counter()
        internal_reranked = self.models.rerank(
            query=processed_query.retrieval_query,
            contexts=internal_contexts,
            top_k=self.config.rerank_top_k,
        )
        rerank_ms = round((time.perf_counter() - t0) * 1000, 1)

        # 4. Confidence decision based on reranker logit scores
        decision = evaluate_internal_results(
            contexts=internal_reranked,
            config=self.config,
        )

        # Resolve web usage: explicit override > confidence decision
        if use_web is True:
            should_try_web = True
        elif use_web is False:
            should_try_web = False
        else:
            should_try_web = self.config.web_fallback_enabled and decision.should_use_web

        used_web = False
        final_contexts = internal_reranked

        # 5. Web fallback
        if should_try_web:
            log.info("Web fallback triggered. Reason: %s", decision.reason)
            t0 = time.perf_counter()
            web_contexts = self.web.search(processed_query.retrieval_query)
            web_search_ms = round((time.perf_counter() - t0) * 1000, 1)

            if web_contexts:
                used_web = True
                combined = internal_reranked + web_contexts
                t0 = time.perf_counter()
                final_contexts = self.models.rerank(
                    query=processed_query.retrieval_query,
                    contexts=combined,
                    top_k=self.config.rerank_top_k,
                )
                web_rerank_ms = round((time.perf_counter() - t0) * 1000, 1)
            else:
                web_rerank_ms = 0.0
        else:
            web_search_ms = 0.0
            web_rerank_ms = 0.0

        # Drop any context with no text (shouldn't happen, but guard it)
        final_contexts = [ctx for ctx in final_contexts if ctx.text and ctx.text.strip()]

        debug_info: Dict[str, Any] = {
            "internal_retrieved_count": len(internal_contexts),
            "internal_reranked_count": len(internal_reranked),
            "final_context_count": len(final_contexts),
            "confidence": decision.confidence,
            "confidence_reason": decision.reason,
            "web_fallback_enabled": self.config.web_fallback_enabled,
            "used_web": used_web,
            "top_rerank_score": (
                internal_reranked[0].rerank_score
                if internal_reranked and internal_reranked[0].rerank_score is not None
                else None
            ),
            "embedding_model": self.config.embedding_model,
            "embedding_version": self.config.embedding_version,
            "reranker_model": self.config.reranker_model,
            "llm_provider": self.config.llm_provider,
            "ollama_model": self.config.ollama_model,
            "model_device": self.models.device,
            "cpu_skip_reranker": self.config.cpu_skip_reranker,
            "prompt_context_limit": self.config.prompt_context_limit,
            "query_processing": {
                "original": processed_query.original,
                "normalized": processed_query.normalized,
                "retrieval_query": processed_query.retrieval_query,
                "keyword_query": processed_query.keyword_query,
                "language_hint": processed_query.language_hint,
            },
            "timing_ms": {
                "embedding": embedding_ms,
                "retrieval": retrieval_ms,
                "rerank": rerank_ms,
                "web_search": web_search_ms,
                "web_rerank": web_rerank_ms,
            },
        }

        return final_contexts, decision, used_web, debug_info

    def _should_fetch_full_docs(
        self,
        internal_contexts: List[RetrievedContext],
    ) -> bool:
        """
        Decide whether to fetch full parent documents.

        Only worthwhile when the number of unique docs is small AND
        the documents are likely short enough to fit the context window.
        For 20-100 page documents this will almost always return False,
        meaning the synthesizer uses reranked chunks directly (better quality).
        """
        unique_docs = {ctx.doc_id for ctx in internal_contexts if ctx.doc_id}

        if len(unique_docs) > _FULL_DOC_MAX_UNIQUE_DOCS:
            log.debug(
                "Skipping full doc fetch: %d unique docs exceeds limit of %d.",
                len(unique_docs),
                _FULL_DOC_MAX_UNIQUE_DOCS,
            )
            return False

        # Estimate total size from what we already have (parent text is a sample)
        estimated_chars = sum(len(ctx.text or "") for ctx in internal_contexts)
        scale_factor = 20  # rough multiplier: parent chunk is ~1/20 of full doc

        if estimated_chars * scale_factor > _FULL_DOC_MAX_CHARS_ESTIMATE:
            log.debug(
                "Skipping full doc fetch: estimated doc size ~%d chars exceeds budget.",
                estimated_chars * scale_factor,
            )
            return False

        return True

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
        stream: bool = False,
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

        log.info("RAG query: %s", query[:120])
        request_start = time.perf_counter()

        final_contexts, decision, used_web, debug_info = self.retrieve_contexts(
            query=query,
            use_web=use_web,
            language=language,
            doc_id=doc_id,
            source_type=source_type,
        )

        if not final_contexts:
            if self.config.allow_general_knowledge_fallback:
                answer = self.synthesizer.generate_general_response(
                    query=query,
                    answer_style=answer_style,
                    stream=stream,
                )
                debug_info["fallback_mode"] = "general_knowledge"
                return {
                    "answer": answer,
                    "sources": [],
                    "used_web": used_web,
                    "confidence": decision.confidence,
                    "debug": debug_info if debug else None,
                }
            return {
                "answer": "I could not find relevant information to answer this question.",
                "sources": [],
                "used_web": used_web,
                "confidence": decision.confidence,
                "debug": debug_info if debug else None,
            }

        # Only fetch full docs for small/short documents.
        # Large docs (20-100 pages) use reranked chunks directly — better quality,
        # faster, and avoids context window overflow.
        internal_contexts = [c for c in final_contexts if c.source_type == "internal"]
        full_docs: Optional[List[FullDocContext]] = None

        if internal_contexts and self._should_fetch_full_docs(internal_contexts):
            t0 = time.perf_counter()
            full_docs = self.db.fetch_full_parent_docs(internal_contexts)
            debug_info["timing_ms"]["fetch_full_docs"] = round((time.perf_counter() - t0) * 1000, 1)
            debug_info["full_docs_fetched"] = len(full_docs)
            debug_info["full_doc_ids"] = [d.doc_id for d in full_docs]
            log.info("Full docs fetched: %d", len(full_docs))
        else:
            debug_info["full_docs_fetched"] = 0
            debug_info["context_mode"] = "reranked_chunks"
            debug_info["timing_ms"]["fetch_full_docs"] = 0.0

        if self._should_use_general_fallback(
            decision=decision,
            contexts=final_contexts,
            used_web=used_web,
        ):
            t0 = time.perf_counter()
            answer = self.synthesizer.generate_general_response(
                query=query,
                answer_style=answer_style,
                stream=stream,
            )
            debug_info["timing_ms"]["generation_setup"] = round((time.perf_counter() - t0) * 1000, 1)
            debug_info["timing_ms"]["total_before_stream"] = round((time.perf_counter() - request_start) * 1000, 1)
            debug_info["fallback_mode"] = "general_knowledge"
            return {
                "answer": answer,
                "sources": [ctx.to_public_dict() for ctx in final_contexts],
                "used_web": used_web,
                "confidence": decision.confidence,
                "debug": debug_info if debug else None,
            }

        t0 = time.perf_counter()
        answer = self.synthesizer.generate_response(
            query=query,
            contexts=final_contexts,
            used_web=used_web,
            full_docs=full_docs,
            answer_style=answer_style,
            stream=stream,
        )
        debug_info["timing_ms"]["generation_setup"] = round((time.perf_counter() - t0) * 1000, 1)
        debug_info["timing_ms"]["total_before_stream"] = round((time.perf_counter() - request_start) * 1000, 1)

        return {
            "answer": answer,
            "sources": [ctx.to_public_dict() for ctx in final_contexts],
            "used_web": used_web,
            "confidence": decision.confidence,
            "debug": debug_info if debug else None,
        }
