from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
import psycopg
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
            check=ConnectionPool.check_connection,
        )

    def _configure_connection(self, conn) -> None:
        """
        Configure each new PostgreSQL connection.

        hnsw.ef_search improves recall for pgvector HNSW search.
        Higher = better recall but slower search.
        """
        try:
            ef_search = int(self.config.hnsw_ef_search)

            with conn.cursor() as cur:
                cur.execute(f"SET hnsw.ef_search = {ef_search}")

            conn.commit()

        except Exception:
            conn.rollback()
            log.warning("Could not set hnsw.ef_search. Continuing with PostgreSQL default.")

    def close(self) -> None:
        self.pool.close()

    @staticmethod
    def _is_retryable_connection_error(exc: BaseException) -> bool:
        if not isinstance(exc, psycopg.Error):
            return False

        message = str(exc).lower()
        retry_markers = (
            "server closed the connection unexpectedly",
            "consuming input failed",
            "connection is closed",
            "terminating connection",
            "connection not open",
        )
        return any(marker in message for marker in retry_markers)

    def _run_fetchall(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        row_factory=None,
        operation_name: str,
    ) -> list[Any]:
        last_exc: BaseException | None = None

        for attempt in range(2):
            try:
                with self.pool.connection() as conn:
                    with conn.cursor(row_factory=row_factory) as cur:
                        cur.execute(sql, params)
                        return cur.fetchall()
            except Exception as exc:
                last_exc = exc
                if attempt == 0 and self._is_retryable_connection_error(exc):
                    log.warning(
                        "Retrying database operation '%s' after connection failure: %s",
                        operation_name,
                        exc,
                    )
                    continue
                raise

        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _vector_literal(vector: List[float]) -> str:
        """
        pgvector accepts string format like:
        [0.1,0.2,0.3]

        This avoids psycopg adapter problems when passing Python lists.
        """
        if not vector:
            raise ValueError("Query vector is empty.")

        return "[" + ",".join(str(float(x)) for x in vector) + "]"

    @staticmethod
    def _ensure_json(value: Any) -> Any:
        if value is None:
            return None

        if isinstance(value, (dict, list)):
            return value

        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value

        return value

    def health_check(self) -> Dict[str, Any]:
        """
        Simple DB check before running the full RAG pipeline.
        This does not need Ollama or embedding models.
        """
        sql_content_tsv = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'child_chunks'
          AND column_name = 'content_tsv';
        """

        sql_embedding_versions = """
        SELECT embedding_model, embedding_version, COUNT(*)
        FROM child_chunks
        GROUP BY embedding_model, embedding_version
        ORDER BY embedding_model, embedding_version;
        """

        rows = self._run_fetchall(
            """
            SELECT
                (SELECT COUNT(*) FROM parent_chunks) AS parent_count,
                (SELECT COUNT(*) FROM child_chunks) AS child_count,
                EXISTS(
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_name = 'child_chunks'
                      AND column_name = 'content_tsv'
                ) AS content_tsv_exists;
            """,
            operation_name="health_check_summary",
        )
        parent_count = rows[0][0]
        child_count = rows[0][1]
        content_tsv_exists = bool(rows[0][2])
        embedding_versions = self._run_fetchall(
            sql_embedding_versions,
            operation_name="health_check_embedding_versions",
        )

        return {
            "db_connected": True,
            "parent_chunks": parent_count,
            "child_chunks": child_count,
            "content_tsv_exists": content_tsv_exists,
            "embedding_versions": embedding_versions,
        }

    def hybrid_search(
        self,
        *,
        query_vector: List[float],
        query_text: str,
        keyword_query: Optional[str] = None,
        language: Optional[str] = None,
        doc_id: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> List[RetrievedContext]:
        """
        Production hybrid retrieval.

        Flow:
        1. Vector search on child_chunks
        2. Keyword search on child_chunks
        3. RRF fusion
        4. Retrieve parent_chunks for selected children
        """

        vector_literal = self._vector_literal(query_vector)

        filters = [
            "c.embedding IS NOT NULL",
            "c.embedding_model = %(embedding_model)s",
            "c.embedding_version = %(embedding_version)s",
        ]

        params: Dict[str, Any] = {
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

        if source_type:
            filters.append("c.metadata->>'doc_source_type' = %(source_type)s")
            params["source_type"] = source_type

        filter_sql = " AND ".join(filters)

        sql = f"""
        WITH vector_candidates AS (
            SELECT
                c.child_id,
                c.parent_id,
                c.doc_id,
                c.text AS child_text,
                c.metadata AS child_metadata,
                c.embedding <=> %(query_vector)s::vector AS vector_distance
            FROM child_chunks c
            WHERE {filter_sql}
            ORDER BY c.embedding <=> %(query_vector)s::vector
            LIMIT %(vector_candidates)s
        ),
        vector_ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    ORDER BY vector_distance ASC
                ) AS rank_position
            FROM vector_candidates
        ),
        keyword_candidates AS (
            SELECT
                c.child_id,
                c.parent_id,
                c.doc_id,
                c.text AS child_text,
                c.metadata AS child_metadata,
                ts_rank_cd(
                    c.content_tsv,
                    websearch_to_tsquery('simple', %(keyword_query)s)
                ) AS keyword_score
            FROM child_chunks c
            WHERE {filter_sql}
              AND c.content_tsv @@ websearch_to_tsquery('simple', %(keyword_query)s)
            ORDER BY keyword_score DESC
            LIMIT %(keyword_candidates)s
        ),
        keyword_ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    ORDER BY keyword_score DESC
                ) AS rank_position
            FROM keyword_candidates
        ),
        fused AS (
            SELECT
                child_id,
                parent_id,
                doc_id,
                child_text,
                child_metadata,
                1.0 / (%(rrf_k)s + rank_position) AS rrf_score,
                'vector' AS match_type
            FROM vector_ranked

            UNION ALL

            SELECT
                child_id,
                parent_id,
                doc_id,
                child_text,
                child_metadata,
                1.0 / (%(rrf_k)s + rank_position) AS rrf_score,
                'keyword' AS match_type
            FROM keyword_ranked
        ),
        fused_children AS (
            SELECT
                child_id,
                parent_id,
                doc_id,
                MAX(child_text) AS child_text,
                MAX(child_metadata::text)::jsonb AS child_metadata,
                SUM(rrf_score) AS fusion_score,
                ARRAY_AGG(DISTINCT match_type) AS match_types
            FROM fused
            GROUP BY child_id, parent_id, doc_id
            ORDER BY fusion_score DESC
            LIMIT %(fused_child_limit)s
        ),
        parent_contexts AS (
            SELECT
                p.parent_id,
                p.doc_id,
                p.text AS parent_text,
                p.metadata AS parent_metadata,
                COALESCE(p.metadata->>'doc_file_name', p.doc_id) AS source,
                SUM(fc.fusion_score) AS parent_fusion_score,
                jsonb_agg(
                    jsonb_build_object(
                        'child_id', fc.child_id,
                        'child_text', fc.child_text,
                        'fusion_score', fc.fusion_score,
                        'match_types', fc.match_types,
                        'metadata', fc.child_metadata
                    )
                    ORDER BY fc.fusion_score DESC
                ) AS matched_children
            FROM fused_children fc
            JOIN parent_chunks p
              ON p.parent_id = fc.parent_id
            GROUP BY
                p.parent_id,
                p.doc_id,
                p.text,
                p.metadata
        )
        SELECT
            parent_id,
            doc_id,
            parent_text,
            parent_metadata,
            source,
            parent_fusion_score,
            matched_children
        FROM parent_contexts
        ORDER BY parent_fusion_score DESC
        LIMIT %(parent_limit)s;
        """

        rows = self._run_fetchall(
            sql,
            params,
            row_factory=dict_row,
            operation_name="hybrid_search",
        )

        # Convert to RetrievedContext but only return the parent content to LLM
        return [
            self._row_to_context(dict(row)) for row in rows if "parent_text" in row
        ]

    def fetch_full_parent_docs(
        self,
        contexts: List[RetrievedContext],
    ) -> List[FullDocContext]:
        """
        For each unique doc_id in the retrieved contexts, fetch ALL parent_chunks
        belonging to that document and concatenate them in chunk order.

        This gives the LLM the full document text rather than just the matched
        parent chunk snippets, which results in much better answers.

        Returns one FullDocContext per unique doc_id, ordered by the doc_id's
        first appearance in the ranked contexts list (most relevant doc first).
        """
        # Collect unique doc_ids, preserving relevance order from reranker
        seen: set[str] = set()
        ordered_doc_ids: List[str] = []
        for ctx in contexts:
            if ctx.doc_id and ctx.doc_id not in seen:
                seen.add(ctx.doc_id)
                ordered_doc_ids.append(ctx.doc_id)

        if not ordered_doc_ids:
            return []

        # Build a source label lookup from the already-retrieved contexts
        # (uses the first context seen for each doc_id as the display source)
        source_lookup: Dict[str, str] = {}
        metadata_lookup: Dict[str, Dict[str, Any]] = {}
        for ctx in contexts:
            if ctx.doc_id not in source_lookup:
                source_lookup[ctx.doc_id] = ctx.source
                metadata_lookup[ctx.doc_id] = ctx.metadata

        sql = """
        SELECT
            doc_id,
            parent_id,
            text,
            metadata
        FROM parent_chunks
        WHERE doc_id = ANY(%(doc_ids)s)
        ORDER BY doc_id, parent_id;
        """

        rows_by_doc: Dict[str, List[str]] = {doc_id: [] for doc_id in ordered_doc_ids}

        rows = self._run_fetchall(
            sql,
            {"doc_ids": ordered_doc_ids},
            row_factory=dict_row,
            operation_name="fetch_full_parent_docs",
        )
        for row in rows:
            doc_id = str(row["doc_id"])
            text = row.get("text") or ""
            if doc_id in rows_by_doc and text.strip():
                rows_by_doc[doc_id].append(text)

        full_docs: List[FullDocContext] = []
        for doc_id in ordered_doc_ids:
            chunks = rows_by_doc.get(doc_id, [])

            if not chunks:
                log.warning("No parent chunks found for doc_id=%s — skipping.", doc_id)
                continue

            full_text = "\n\n".join(chunks)

            full_docs.append(
                FullDocContext(
                    doc_id=doc_id,
                    source=source_lookup.get(doc_id, doc_id),
                    full_text=full_text,
                    metadata=metadata_lookup.get(doc_id, {}),
                )
            )

        log.info(
            "fetch_full_parent_docs: %d unique doc(s), %d returned with content.",
            len(ordered_doc_ids),
            len(full_docs),
        )

        return full_docs

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
