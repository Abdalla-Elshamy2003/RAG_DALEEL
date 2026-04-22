from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from .confidence import evaluate_internal_results, filter_irrelevant_results
from .config import RAGConfig
from .database import ProductionDatabase
from .model import GPUModelManager
from .prompter import Synthesizer
from .schemas import ConfidenceDecision, FullDocContext, RetrievedContext
from .tools import WebSearchTool

log = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.config.validate()

        self.db = ProductionDatabase(self.config)
        self.models = GPUModelManager(self.config)
        
        # PRODUCTION CHECK: Ensure we aren't accidentally on CPU
        if self.models.device != "cuda":
            log.error("CRITICAL: GPUModelManager initialized on CPU. Reranking will be extremely slow.")
            # In a strict production env, you might want to raise an error here
        
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
    ) -> Tuple[List[RetrievedContext], ConfidenceDecision, bool, Dict[str, Any]]:
        
        total_start = time.perf_counter()
        timings = {}

        query = (query or "").strip()
        if not query:
            return [], ConfidenceDecision(should_use_web=True, reason="Empty query", confidence="none"), False, {"error": "empty"}

        # 1. Query embedding (GPU)
        embed_start = time.perf_counter()
        query_vector = self.models.encode_query(query)
        timings["embedding_time"] = time.perf_counter() - embed_start

        # 2. Hybrid search (Postgres) - Now returns (results, db_timings)
        internal_contexts, db_timings = self.db.hybrid_search(
            query_vector=query_vector,
            query_text=query,
            language=language,
            doc_id=doc_id,
            source_type=source_type,
        )
        timings.update(db_timings)

        # 3. Rerank internal contexts (GPU)
        rerank_start = time.perf_counter()
        # We send up to FUSED_CHILD_LIMIT to the reranker
        internal_reranked = self.models.rerank(
            query=query,
            contexts=internal_contexts,
            top_k=self.config.rerank_top_k,
        )
        timings["rerank_time"] = time.perf_counter() - rerank_start

        # 4. Accuracy Enhancement: Filter Noise (The "Spain/Sports" fix)
        # This removes results that are mathematically far from the top result
        internal_reranked = filter_irrelevant_results(
            internal_reranked, 
            threshold_ratio=0.6 # Only keep results > 60% of top score
        )

        # 5. Confidence decision
        decision = evaluate_internal_results(
            contexts=internal_reranked,
            config=self.config,
        )

        # Web logic
        should_try_web = use_web if use_web is not None else (self.config.web_fallback_enabled and decision.should_use_web)
        used_web = False
        final_contexts = internal_reranked

        if should_try_web:
            log.info("Web fallback triggered: %s", decision.reason)
            web_contexts = self.web.search(query)
            if web_contexts:
                used_web = True
                # Rerank the combination of web + internal
                combined = internal_reranked + web_contexts
                final_contexts = self.models.rerank(
                    query=query,
                    contexts=combined,
                    top_k=self.config.rerank_top_k,
                )

        timings["total_retrieval_time"] = time.perf_counter() - total_start
        
        debug_info = {
            "timings": timings,
            "confidence": decision.confidence,
            "used_web": used_web,
            "count": len(final_contexts)
        }

        return final_contexts, decision, used_web, debug_info

    def answer_question(
        self,
        query: str,
        **kwargs
    ) -> Dict[str, Any]:
        
        # Get contexts with full timing data
        final_contexts, decision, used_web, debug_info = self.retrieve_contexts(query, **kwargs)

        if not final_contexts:
            return {
                "answer": "I don't have sufficient information in the available sources to answer this question.",
                "sources": [],
                "used_web": used_web,
                "debug": debug_info
            }

        # LLM Generation
        gen_start = time.perf_counter()
        
        # Decide if we need full docs (only for very specific/small docs)
        full_docs = None
        if not used_web and len(final_contexts) <= 2:
            full_docs, _ = self.db.fetch_full_parent_docs(final_contexts)

        answer = self.synthesizer.generate_response(
            query=query,
            contexts=final_contexts,
            used_web=used_web,
            full_docs=full_docs,
        )
        
        debug_info["timings"]["llm_generation_time"] = time.perf_counter() - gen_start
        debug_info["timings"]["end_to_end_time"] = sum(debug_info["timings"].values())

        return {
            "answer": answer,
            "sources": [ctx.to_public_dict() for ctx in final_contexts],
            "used_web": used_web,
            "confidence": decision.confidence,
            "debug": debug_info,
        }
    