-- ============================================================
-- Summarization Pipeline Schema
-- Requires: pgvector extension
-- ============================================================

-- COMMENTED OUT FOR TESTING (skip pgvector for now)
-- CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- ADDITIONS to existing tables (if columns are missing)
-- Uncomment and run if you get constraint errors
-- ============================================================

-- Add documents column to post_processing_data if missing
-- This should contain cleaned documents for summarization
-- ALTER TABLE post_processing_data ADD COLUMN documents TEXT;

-- Add language column to parent_chunks if missing
-- ALTER TABLE parent_chunks ADD COLUMN language VARCHAR(20);

-- Verify your parent_chunks has these columns:
-- - id (primary key)
-- - doc_id (foreign key to post_processing_data)
-- - parent_index (integer, for ordering within a document) ← NOT doc_id!
-- - text (TEXT, the actual chunk content)
-- - language (VARCHAR, or extract from metadata)
-- - metadata (JSONB, can store additional info)

-- ============================================================
-- Main summaries table
-- Stores all 3 levels of summaries with embeddings
-- ============================================================
CREATE TABLE IF NOT EXISTS summaries (
    id              BIGSERIAL PRIMARY KEY,

    -- Level: 1=parent summary, 2=document summary, 3=cross-doc cluster summary
    level           SMALLINT NOT NULL CHECK (level IN (1, 2, 3)),

    -- References to source data
    -- Level 1: points to a parent chunk id
    -- Level 2: points to a document id
    -- Level 3: NULL (cluster has no single source)
    source_id       BIGINT,

    -- For level 3: which doc_ids are in this cluster (array of doc ids)
    cluster_doc_ids BIGINT[],

    -- The generated summary text (Arabic or mixed)
    summary_text    TEXT NOT NULL,

    -- Embedding vector — 1024-dim for multilingual-e5-large
    -- Change to 768 if using paraphrase-multilingual-mpnet-base-v2
    embedding       VECTOR(1024),

    -- Rich metadata stored as JSONB
    -- Level 1 example: {"doc_id": 5, "doc_title": "...", "parent_num": 3, "lang": "ar", "chunk_count": 4}
    -- Level 2 example: {"doc_id": 5, "doc_title": "...", "lang": "ar", "parent_count": 12}
    -- Level 3 example: {"cluster_id": "uuid", "topic_tag": "...", "doc_count": 4, "lang": "ar"}
    metadata        JSONB NOT NULL DEFAULT '{}',

    -- Status tracking for pipeline reruns
    status          VARCHAR(20) NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'processing', 'done', 'failed')),
    error_message   TEXT,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Indexes
-- ============================================================

-- Fast lookup: all level-1 summaries for a given doc
CREATE INDEX IF NOT EXISTS idx_summaries_level_source
    ON summaries (level, source_id);

-- Fast lookup by doc_id stored in metadata (level 1 and 2)
CREATE INDEX IF NOT EXISTS idx_summaries_metadata_doc_id
    ON summaries USING GIN ((metadata -> 'doc_id'));

-- Fast lookup by topic_tag for level 3
CREATE INDEX IF NOT EXISTS idx_summaries_metadata_topic
    ON summaries USING GIN ((metadata -> 'topic_tag'));

-- Status index for pipeline resumption (find all pending/failed)
CREATE INDEX IF NOT EXISTS idx_summaries_status
    ON summaries (status) WHERE status IN ('pending', 'failed');

-- Vector similarity search (HNSW — best for < 1M rows, fast recall)
-- m=16, ef_construction=64 are safe defaults for 1K docs
-- COMMENTED OUT FOR TESTING (skip pgvector for now)
-- CREATE INDEX IF NOT EXISTS idx_summaries_embedding_hnsw
--     ON summaries USING hnsw (embedding vector_cosine_ops)
--     WITH (m = 16, ef_construction = 64);

-- ============================================================
-- Trigger: auto-update updated_at
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_summaries_updated_at ON summaries;
CREATE TRIGGER trg_summaries_updated_at
    BEFORE UPDATE ON summaries
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================
-- pipeline_runs: audit log for each pipeline execution
-- Useful for tracking backfill and incremental runs
-- ============================================================
CREATE TABLE IF NOT EXISTS summarization_pipeline_runs (
    id              BIGSERIAL PRIMARY KEY,
    run_type        VARCHAR(20) NOT NULL CHECK (run_type IN ('backfill', 'incremental', 'recluster')),
    status          VARCHAR(20) NOT NULL DEFAULT 'running'
                    CHECK (status IN ('running', 'done', 'failed')),
    docs_processed  INTEGER DEFAULT 0,
    summaries_created INTEGER DEFAULT 0,
    error_message   TEXT,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at     TIMESTAMPTZ
);

-- ============================================================
-- Helpful view: summaries with doc_title flattened out
-- ============================================================
CREATE OR REPLACE VIEW v_summaries_flat AS
SELECT
    id,
    level,
    source_id,
    cluster_doc_ids,
    summary_text,
    metadata->>'doc_title'  AS doc_title,
    metadata->>'lang'       AS lang,
    (metadata->>'doc_id')::BIGINT AS doc_id,
    (metadata->>'parent_num')::INTEGER AS parent_num,
    status,
    created_at
FROM summaries;