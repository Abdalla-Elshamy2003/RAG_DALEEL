import db
db.init_pool()
with db.get_cursor() as cur:
    cur.execute("""
        SELECT conname, pg_get_constraintdef(oid) 
        FROM pg_constraint 
        WHERE conrelid = 'summarization_pipeline_runs'::regclass
    """)
    for row in cur.fetchall():
        print(f"{row['conname']}: {row['pg_get_constraintdef']}")
