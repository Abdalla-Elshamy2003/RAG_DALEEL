-- Fix the constraint to allow cleanup-chinese
ALTER TABLE summarization_pipeline_runs 
DROP CONSTRAINT IF EXISTS summarization_pipeline_runs_run_type_check;

ALTER TABLE summarization_pipeline_runs 
ADD CONSTRAINT summarization_pipeline_runs_run_type_check 
CHECK (run_type IN ('backfill', 'incremental', 'recluster', 'cleanup-chinese'));
