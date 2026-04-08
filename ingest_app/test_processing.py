from text_utils import clean_text, detect_lang, token_count_simple
import psycopg2
import sys
import os

try:
    from ingest_app.text_utils import clean_text, detect_lang
except ImportError:
    from text_utils import clean_text, detect_lang

def test_db_processing():
    db_config = {
        "host": "docs_ingestion_db",
        "database": "docs_ingestion",
        "user": "esraa",
        "password": "2128102003",
        "port": 5433
    }

    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        
        cur.execute("SELECT id, chunk_text FROM sections LIMIT 10;")
        rows = cur.fetchall()

        if not rows:
            print("No data found.")
            return

        for doc_id, raw_text in rows:
            cleaned = clean_text(raw_text)
            lang = detect_lang(cleaned)

            print(f"ID: {doc_id} | LANG: {lang}")
            print(f"CLEAN: {cleaned[:100]}...")
            print("-" * 20)

        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_db_processing()
    