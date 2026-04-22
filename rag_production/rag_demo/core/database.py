
from __future__ import annotations

import json
import logging
import time  
from typing import Any, Dict, List, Optional, Tuple
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import RAGConfig
from .schemas import ChildEvidence, FullDocContext, RetrievedContext

log = logging.getLogger(__name__)

class ProductionDatabase:
    def __init__(self, config: RAGConfig):
        config.validate()
        self.config = config

        self.pool = ConnectionPool(
            conninfo=config.db_conn,
            min_size=1,
            max_size=10,
            open=True,
            configure=self._configure_connection,
        )

    def _configure_connection(self, conn) -> None:
        try:
            ef_search = int(self.config.hnsw_ef_search)
            with conn.cursor() as cur:
                cur.execute(f"SET hnsw.ef_search = {ef_search}")
            conn.commit()
        except Exception:
            conn.rollback()
            log.warning("Could not set hnsw.ef_search. Using PostgreSQL default.")

    def close(self) -> None:
        self.pool.close()

    @staticmethod
    def _vector_literal(vector: List[float]) -> str:
        if not vector:
            raise ValueError("Query vector is empty.")
        return "[" + ",".join(str(float(x)) for x in vector) + "]"

    def hybrid_search(
        self,
        *,
        query_vector: List[float],
        query_text: str,
        keyword_query: Optional[str] = None,
        language: Optional[str] = None,
        doc_id: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> Tuple[List[RetrievedContext], Dict[str, float]]:
        """
        Optimized Hybrid Search with Internal Timing.
        Returns (Results, Timings).
        """
        timings = {}
        start_total = time.perf_counter()
        
        vector_literal = self._vector_literal(query_vector)
        
        # Build filters dynamically for both branches of the hybrid search
        filters = [
            "c.embedding_model = %(embedding_model)s",
            "c.embedding_version = %(embedding_version)s"
        ]
        params = {
            "query_vector": vector_literal,
            "query_text": query_text,
            "keyword_query": keyword_query or query_text,
            "embedding_model": self.config.embedding_model,
            "embedding_version": self.config.embedding_version,
            "vector_candidates": self.config.vector_candidates,
            "keyword_candidates": self.config.keyword_candidates,
            "fused_child_limit": self.config.fused_child_limit,
            "parent_limit": self.config.parent_limit,
            "rrf_k": self.config.rrf_k,
        }

        if language:
            filters.append("c.metadata->>'doc_language' = %(language)s")
            params["language"] = language
        if doc_id:
            filters.append("c.doc_id = %(doc_id)s")
            params["doc_id"] = doc_id

        filter_sql = " AND ".join(filters)

        # PRODUCTION OPTIMIZED SQL: 
        # Using CTEs to force Postgres to use the HNSW and GIN indexes independently.
        sql = f"""
        WITH vector_ranked AS (
            SELECT
                c.child_id, c.parent_id, c.doc_id, c.text, c.metadata,
                ROW_NUMBER() OVER (ORDER BY c.embedding <=> %(query_vector)s::vector) as rank
            FROM child_chunks c
            WHERE {filter_sql} AND c.embedding IS NOT NULL
            ORDER BY c.embedding <=> %(query_vector)s::vector
            LIMIT %(vector_candidates)s
        ),
        keyword_ranked AS (
            SELECT
                c.child_id, c.parent_id, c.doc_id, c.text, c.metadata,
                ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.content_tsv, websearch_to_tsquery('simple', %(keyword_query)s)) DESC) as rank
            FROM child_chunks c
            WHERE {filter_sql} AND c.content_tsv @@ websearch_to_tsquery('simple', %(keyword_query)s)
            ORDER BY rank
            LIMIT %(keyword_candidates)s
        ),
        fused AS (
            SELECT 
                child_id, parent_id, doc_id, text, metadata,
                SUM(
                    CASE
                        WHEN source = 'vector' THEN (1.0 / (rank + %(rrf_k)s)) * 0.7
                        WHEN source = 'keyword' THEN (1.0 / (rank + %(rrf_k)s)) * 0.3
                    END
                ) as fusion_score,
                ARRAY_AGG(source) as match_types
            FROM (
                SELECT *, 'vector' as source FROM vector_ranked
                UNION ALL
                SELECT *, 'keyword' as source FROM keyword_ranked
            ) combined
            GROUP BY child_id, parent_id, doc_id, text, metadata
            ORDER BY fusion_score DESC
            LIMIT %(fused_child_limit)s
        )
        SELECT
            p.parent_id, p.doc_id, p.text AS parent_text, p.metadata AS parent_metadata,
            COALESCE(p.metadata->>'doc_file_name', p.doc_id) AS source,
            SUM(f.fusion_score) AS parent_fusion_score,
            jsonb_agg(jsonb_build_object(
                'child_id', f.child_id, 'child_text', f.text,
                'fusion_score', f.fusion_score, 'match_types', f.match_types
            )) AS matched_children
        FROM fused f
        JOIN parent_chunks p ON p.parent_id = f.parent_id
        GROUP BY p.parent_id, p.doc_id, p.text, p.metadata
        ORDER BY parent_fusion_score DESC
        LIMIT %(parent_limit)s;
        """

        query_start = time.perf_counter()
        with self.pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        timings["db_query_time"] = time.perf_counter() - query_start

        processing_start = time.perf_counter()
        results = [self._row_to_context(dict(row)) for row in rows]
        timings["data_processing_time"] = time.perf_counter() - processing_start
        
        timings["total_db_layer_time"] = time.perf_counter() - start_total
        return results, timings

    def fetch_full_parent_docs(
        self,
        contexts: List[RetrievedContext],
    ) -> Tuple[List[FullDocContext], float]:
        """ Fetches full docs with timing. """
        start = time.perf_counter()
        
        seen = set()
        ordered_doc_ids = []
        source_lookup = {}
        metadata_lookup = {}
        
        for ctx in contexts:
            if ctx.doc_id and ctx.doc_id not in seen:
                seen.add(ctx.doc_id)
                ordered_doc_ids.append(ctx.doc_id)
                source_lookup[ctx.doc_id] = ctx.source
                metadata_lookup[ctx.doc_id] = ctx.metadata

        if not ordered_doc_ids:
            return [], 0.0

        sql = "SELECT doc_id, text FROM parent_chunks WHERE doc_id = ANY(%(doc_ids)s) ORDER BY doc_id, parent_id;"
        
        full_docs = []
        with self.pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, {"doc_ids": ordered_doc_ids})
                rows = cur.fetchall()
                
        # Group by doc_id
        doc_map = {}
        for row in rows:
            d_id = str(row["doc_id"])
            doc_map.setdefault(d_id, []).append(row["text"] or "")

        for d_id in ordered_doc_ids:
            if d_id in doc_map:
                full_docs.append(FullDocContext(
                    doc_id=d_id,
                    source=source_lookup[d_id],
                    full_text="\n\n".join(doc_map[d_id]),
                    metadata=metadata_lookup[d_id]
                ))

        return full_docs, time.perf_counter() - start


    def _row_to_context(self, row: Dict[str, Any]) -> RetrievedContext:
        parent_metadata = self._ensure_json(row.get("parent_metadata")) or {}
        raw_children = self._ensure_json(row.get("matched_children")) or []

        children: List[ChildEvidence] = []

        for item in raw_children:
            if not isinstance(item, dict):
                continue

            children.append(
                ChildEvidence(
                    child_id=str(item.get("child_id", "")),
                    child_text=item.get("child_text") or "",
                    fusion_score=float(item.get("fusion_score") or 0.0),
                    match_types=list(item.get("match_types") or []),
                    metadata=self._ensure_json(item.get("metadata")) or {},
                )
            )

        return RetrievedContext(
            source_type="internal",
            doc_id=str(row.get("doc_id") or ""),
            parent_id=str(row.get("parent_id") or ""),
            text=row.get("parent_text") or "",
            source=row.get("source") or "unknown",
            metadata=parent_metadata,
            matched_children=children,
            fusion_score=float(row.get("parent_fusion_score") or 0.0),
        )
