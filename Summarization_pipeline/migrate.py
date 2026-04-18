#!/usr/bin/env python3
"""
Migration script to create the summarization pipeline tables with pgvector support
"""

import psycopg2
from config import config

def migrate():
    # Connect to database
    conn = psycopg2.connect(config.db_conn)
    cur = conn.cursor()

    try:
        # Step 1: Create pgvector extension
        print("Creating pgvector extension...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        print("✓ pgvector extension created/exists")
    except Exception as e:
        print(f"Warning: Could not create pgvector extension: {e}")
        print("Continuing without pgvector support...")
        conn.commit()

    # Step 2: Create the summaries table with pgvector support
    create_summaries = '''
    CREATE TABLE IF NOT EXISTS summaries (
        id              BIGSERIAL PRIMARY KEY,
        level           SMALLINT NOT NULL CHECK (level IN (1, 2, 3)),
        source_id       BIGINT,
        cluster_doc_ids BIGINT[],
        summary_text    TEXT NOT NULL,
        embedding       vector(1024),
        metadata        JSONB NOT NULL DEFAULT '{}',
        status          VARCHAR(20) NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'processing', 'done', 'failed')),
        error_message   TEXT,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    '''

    cur.execute(create_summaries)
    conn.commit()
    print("✓ Table summaries created successfully!")

    # Step 3: Create unique constraint for ON CONFLICT logic
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_summaries_level_source_unique 
        ON summaries (level, source_id) WHERE level < 3;
    """)
    conn.commit()
    print("✓ Unique constraint created successfully!")

    # Step 3b: Create additional indexes on summaries table
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_summaries_metadata_doc_id ON summaries USING GIN ((metadata -> 'doc_id'));",
        "CREATE INDEX IF NOT EXISTS idx_summaries_metadata_topic ON summaries USING GIN ((metadata -> 'topic_tag'));",
        "CREATE INDEX IF NOT EXISTS idx_summaries_status ON summaries (status) WHERE status IN ('pending', 'failed');",
    ]
    
    for idx_query in indexes:
        cur.execute(idx_query)
    conn.commit()
    print("✓ Indexes created successfully!")

    # Step 4: Create trigger for auto-update updated_at
    trigger_func = '''
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
    '''
    
    cur.execute(trigger_func)
    conn.commit()
    print("✓ Trigger created successfully!")

    # Step 5: Create the pipeline_runs table
    create_pipeline_runs = '''
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
    '''

    cur.execute(create_pipeline_runs)
    conn.commit()
    print("✓ Table summarization_pipeline_runs created successfully!")

    # Verify the tables exist
    cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)", ("summaries",))
    exists = cur.fetchone()[0]
    print(f"✓ Table summaries exists in database: {exists}")

    cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)", ("summarization_pipeline_runs",))
    exists = cur.fetchone()[0]
    print(f"✓ Table summarization_pipeline_runs exists in database: {exists}")

    cur.close()
    conn.close()
    print("\n✓ All migrations completed successfully!")

if __name__ == "__main__":
    migrate()
