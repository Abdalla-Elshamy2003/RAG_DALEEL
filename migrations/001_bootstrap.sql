-- Migration: 001_bootstrap
-- Description: Initial table schemas — file_path is the UNIQUE constraint
-- Date: 2026-04

CREATE TABLE IF NOT EXISTS preprocessing_data (
    id                BIGSERIAL PRIMARY KEY,
    doc_id            TEXT NOT NULL UNIQUE,
    file_name         TEXT NOT NULL,
    file_path         TEXT NOT NULL,
    file_ext          TEXT,
    file_hash         TEXT,
    source_type       TEXT,
    extraction_status TEXT,
    language          TEXT,
    page_count        INT,
    payload           JSONB NOT NULL,
    created_at        TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_preprocessing_data_payload_gin
    ON preprocessing_data USING GIN (payload);

CREATE INDEX IF NOT EXISTS idx_preprocessing_data_file_hash
    ON preprocessing_data(file_hash);

CREATE INDEX IF NOT EXISTS idx_preprocessing_data_file_path
    ON preprocessing_data(file_path);

CREATE TABLE IF NOT EXISTS post_processing_data (
    id                BIGSERIAL PRIMARY KEY,
    doc_id            TEXT NOT NULL UNIQUE,
    file_name         TEXT NOT NULL,
    file_path         TEXT NOT NULL,
    file_ext          TEXT,
    file_hash         TEXT,
    source_type       TEXT,
    extraction_status TEXT,
    language          TEXT,
    page_count        INT,
    payload           JSONB NOT NULL,
    created_at        TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_post_processing_data_payload_gin
    ON post_processing_data USING GIN (payload);

CREATE INDEX IF NOT EXISTS idx_post_processing_data_file_hash
    ON post_processing_data(file_hash);

CREATE INDEX IF NOT EXISTS idx_post_processing_data_file_path
    ON post_processing_data(file_path);
