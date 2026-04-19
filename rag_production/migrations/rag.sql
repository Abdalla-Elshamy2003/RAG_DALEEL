-- Migration: 004_rag_search_indexes
-- Description: Production indexes for hybrid RAG search
-- Date: 2026-04

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- Full-text search column for hybrid keyword retrieval
-- ---------------------------------------------------------------------------
ALTER TABLE child_chunks
ADD COLUMN IF NOT EXISTS content_tsv tsvector
GENERATED ALWAYS AS (
    to_tsvector('simple', coalesce(text, ''))
) STORED;

CREATE INDEX IF NOT EXISTS idx_child_chunks_content_tsv
ON child_chunks USING GIN(content_tsv);

-- ---------------------------------------------------------------------------
-- Fast embedding readiness filters
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_child_chunks_embedding_ready
ON child_chunks(embedding_model, embedding_version)
WHERE embedding IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_child_chunks_doc_model_version
ON child_chunks(doc_id, embedding_model, embedding_version)
WHERE embedding IS NOT NULL;

-- ---------------------------------------------------------------------------
-- Useful metadata filters
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_child_chunks_language_expr
ON child_chunks ((metadata->>'doc_language'));

CREATE INDEX IF NOT EXISTS idx_child_chunks_source_type_expr
ON child_chunks ((metadata->>'doc_source_type'));

CREATE INDEX IF NOT EXISTS idx_child_chunks_file_name_expr
ON child_chunks ((metadata->>'doc_file_name'));

-- ---------------------------------------------------------------------------
-- Make sure HNSW vector index exists
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_child_chunks_embedding_hnsw
ON child_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_parent_chunks_parent_id
ON parent_chunks(parent_id);