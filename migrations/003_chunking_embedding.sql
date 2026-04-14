-- Migration: 003_chunking_embedding
-- Description: Add chunking tables and embedding vector schema
-- Date: 2026-04

-- Vector extension (required for pgvector)
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- Chunking tables
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS parent_chunks (
    id BIGSERIAL PRIMARY KEY,
    parent_id TEXT NOT NULL UNIQUE,
    doc_id TEXT NOT NULL REFERENCES post_processing_data(doc_id) ON DELETE CASCADE,
    parent_index INT NOT NULL,
    text TEXT NOT NULL,
    token_count INT NOT NULL,
    char_count INT NOT NULL,
    start_char INT,
    end_char INT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (doc_id, parent_index)
);

CREATE TABLE IF NOT EXISTS child_chunks (
    id BIGSERIAL PRIMARY KEY,
    child_id TEXT NOT NULL UNIQUE,
    parent_id TEXT NOT NULL REFERENCES parent_chunks(parent_id) ON DELETE CASCADE,
    doc_id TEXT NOT NULL REFERENCES post_processing_data(doc_id) ON DELETE CASCADE,
    child_index INT NOT NULL,
    text TEXT NOT NULL,
    token_count INT NOT NULL,
    char_count INT NOT NULL,
    start_char INT,
    end_char INT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (doc_id, child_index)
);

CREATE INDEX IF NOT EXISTS idx_parent_chunks_doc_id ON parent_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_child_chunks_doc_id ON child_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_child_chunks_parent_id ON child_chunks(parent_id);
CREATE INDEX IF NOT EXISTS idx_parent_chunks_metadata_gin ON parent_chunks USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_child_chunks_metadata_gin ON child_chunks USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Embedding columns
-- ---------------------------------------------------------------------------
ALTER TABLE parent_chunks
ADD COLUMN IF NOT EXISTS embedding vector(1024);

ALTER TABLE parent_chunks
ADD COLUMN IF NOT EXISTS embedding_model TEXT;

ALTER TABLE parent_chunks
ADD COLUMN IF NOT EXISTS embedding_version INT;

ALTER TABLE parent_chunks
ADD COLUMN IF NOT EXISTS embedding_updated_at TIMESTAMPTZ;

ALTER TABLE child_chunks
ADD COLUMN IF NOT EXISTS embedding vector(1024);

ALTER TABLE child_chunks
ADD COLUMN IF NOT EXISTS embedding_model TEXT;

ALTER TABLE child_chunks
ADD COLUMN IF NOT EXISTS embedding_version INT;

ALTER TABLE child_chunks
ADD COLUMN IF NOT EXISTS embedding_updated_at TIMESTAMPTZ;

-- Optional query-helper indexes (model/version filters)
CREATE INDEX IF NOT EXISTS idx_parent_chunks_embed_model_version
    ON parent_chunks(embedding_model, embedding_version);
CREATE INDEX IF NOT EXISTS idx_child_chunks_embed_model_version
    ON child_chunks(embedding_model, embedding_version);

-- Vector ANN indexes
CREATE INDEX IF NOT EXISTS idx_parent_chunks_embedding_hnsw
    ON parent_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_child_chunks_embedding_hnsw
    ON child_chunks USING hnsw (embedding vector_cosine_ops);
