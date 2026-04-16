-- Migration: 002_unique_path
-- Description: Change unique constraint from file_hash to file_path
-- Date: 2026-04

-- Note: Before running this in a production environment, you MUST resolve any duplicate `file_path` records.
-- If duplicate `file_path` entries exist, the constraint creation will fail.

-- For preprocessing_data
DROP INDEX IF EXISTS idx_preprocessing_data_file_hash_unique;
ALTER TABLE preprocessing_data DROP CONSTRAINT IF EXISTS preprocessing_data_file_hash_key;

CREATE INDEX IF NOT EXISTS idx_preprocessing_data_file_hash ON preprocessing_data(file_hash);
CREATE UNIQUE INDEX IF NOT EXISTS idx_preprocessing_data_file_path_unique ON preprocessing_data(file_path);

-- For post_processing_data
DROP INDEX IF EXISTS idx_post_processing_data_file_hash_unique;
ALTER TABLE post_processing_data DROP CONSTRAINT IF EXISTS post_processing_data_file_hash_key;

CREATE INDEX IF NOT EXISTS idx_post_processing_data_file_hash ON post_processing_data(file_hash);
CREATE UNIQUE INDEX IF NOT EXISTS idx_post_processing_data_file_path_unique ON post_processing_data(file_path);

-- Clean up unused columns
ALTER TABLE preprocessing_data DROP COLUMN IF EXISTS markdown_text;
ALTER TABLE post_processing_data DROP COLUMN IF EXISTS markdown_text;
