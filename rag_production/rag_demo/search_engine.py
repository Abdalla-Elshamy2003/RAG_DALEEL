import psycopg
from embedding.model import BGEEmbeddingModel
from embedding.config import EmbeddingConfig

class HybridSearcher:
    def __init__(self, db_conn_str):
        self.db_conn_str = db_conn_str
        # Initialize the same model used for embedding to encode the query
        cfg = EmbeddingConfig(use_fp16=True)
        self.embedder = BGEEmbeddingModel(cfg)

    def search(self, query_text, limit=10):
        query_vector = self.embedder.encode([query_text])[0]
        
        with psycopg.connect(self.db_conn_str) as conn:
            with conn.cursor() as cur:
                # Hybrid RRF (Reciprocal Rank Fusion) Query
                # Combines Vector Search and Keyword Search
                sql = """
                WITH vector_search AS (
                    SELECT child_id, parent_id, doc_id, text,
                           ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) as rank
                    FROM child_chunks
                    LIMIT 40
                ),
                keyword_search AS (
                    SELECT child_id, parent_id, doc_id, text,
                           ROW_NUMBER() OVER (ORDER BY ts_rank_cd(to_tsvector('simple', text), websearch_to_tsquery('simple', %s)) DESC) as rank
                    FROM child_chunks
                    WHERE to_tsvector('simple', text) @@ websearch_to_tsquery('simple', %s)
                    LIMIT 40
                )
                SELECT 
                    COALESCE(v.parent_id, k.parent_id) as parent_id,
                    p.text as parent_text,
                    p.metadata->>'doc_file_name' as file_name,
                    -- RRF Score formula: 1/(k + rank)
                    COALESCE(1.0 / (60 + v.rank), 0.0) + COALESCE(1.0 / (60 + k.rank), 0.0) as score
                FROM vector_search v
                FULL OUTER JOIN keyword_search k ON v.child_id = k.child_id
                JOIN parent_chunks p ON p.parent_id = COALESCE(v.parent_id, k.parent_id)
                ORDER BY score DESC
                LIMIT %s;
                """
                cur.execute(sql, (query_vector, query_text, query_text, limit))
                results = []
                for row in cur.fetchall():
                    results.append({
                        "parent_id": row[0],
                        "text": row[1],
                        "source": row[2],
                        "score": row[3]
                    })
                return results