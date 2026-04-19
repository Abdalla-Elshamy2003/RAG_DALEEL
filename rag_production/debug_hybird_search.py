from __future__ import annotations

import os

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

from rag_demo.core import RAGConfig, GPUModelManager, ProductionDatabase


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in vector) + "]"


def main() -> None:
    load_dotenv(".env", override=True)

    query = "What are the main points in the document?"

    config = RAGConfig()
    config.validate()

    print("CONFIG")
    print("------")
    print("embedding_model:", config.embedding_model)
    print("embedding_version:", config.embedding_version)
    print("vector_candidates:", config.vector_candidates)
    print("keyword_candidates:", config.keyword_candidates)
    print("parent_limit:", config.parent_limit)
    print()

    models = GPUModelManager(config)
    query_vector = models.encode_query(query)
    qvec = vector_literal(query_vector)

    print("QUERY VECTOR")
    print("------------")
    print("length:", len(query_vector))
    print("first 5:", query_vector[:5])
    print()

    with psycopg.connect(config.db_conn) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            print("BASIC COUNTS")
            print("------------")

            cur.execute("SELECT COUNT(*) AS count FROM parent_chunks;")
            print("parent_chunks:", cur.fetchone()["count"])

            cur.execute("SELECT COUNT(*) AS count FROM child_chunks;")
            print("child_chunks:", cur.fetchone()["count"])

            cur.execute("SELECT COUNT(*) AS count FROM child_chunks WHERE embedding IS NOT NULL;")
            print("child_chunks with embedding:", cur.fetchone()["count"])

            cur.execute(
                """
                SELECT embedding_model, embedding_version, COUNT(*) AS count
                FROM child_chunks
                GROUP BY embedding_model, embedding_version
                ORDER BY count DESC;
                """
            )
            print("\nEMBEDDING GROUPS")
            print("----------------")
            for row in cur.fetchall():
                print(dict(row))

            cur.execute(
                """
                SELECT COUNT(*) AS count
                FROM child_chunks
                WHERE embedding IS NOT NULL
                  AND embedding_model = %s
                  AND embedding_version = %s;
                """,
                (config.embedding_model, config.embedding_version),
            )
            print("\nmatching current config:", cur.fetchone()["count"])

            print("\nPARENT JOIN CHECK")
            print("-----------------")
            cur.execute(
                """
                SELECT COUNT(*) AS count
                FROM child_chunks c
                JOIN parent_chunks p ON p.parent_id = c.parent_id
                WHERE c.embedding IS NOT NULL
                  AND c.embedding_model = %s
                  AND c.embedding_version = %s;
                """,
                (config.embedding_model, config.embedding_version),
            )
            print("children with valid parent join:", cur.fetchone()["count"])

            print("\nVECTOR ONLY TEST")
            print("----------------")
            cur.execute(
                """
                SELECT
                    c.child_id,
                    c.parent_id,
                    c.doc_id,
                    c.embedding <=> %s::vector AS distance,
                    LEFT(c.text, 300) AS preview
                FROM child_chunks c
                WHERE c.embedding IS NOT NULL
                  AND c.embedding_model = %s
                  AND c.embedding_version = %s
                ORDER BY c.embedding <=> %s::vector
                LIMIT 5;
                """,
                (
                    qvec,
                    config.embedding_model,
                    config.embedding_version,
                    qvec,
                ),
            )

            rows = cur.fetchall()
            print("vector rows:", len(rows))
            for row in rows:
                print(dict(row))
                print("---")

            print("\nVECTOR + PARENT JOIN TEST")
            print("-------------------------")
            cur.execute(
                """
                SELECT
                    c.child_id,
                    c.parent_id,
                    p.parent_id AS joined_parent_id,
                    c.doc_id,
                    p.metadata->>'doc_file_name' AS source,
                    c.embedding <=> %s::vector AS distance,
                    LEFT(c.text, 200) AS child_preview,
                    LEFT(p.text, 200) AS parent_preview
                FROM child_chunks c
                JOIN parent_chunks p ON p.parent_id = c.parent_id
                WHERE c.embedding IS NOT NULL
                  AND c.embedding_model = %s
                  AND c.embedding_version = %s
                ORDER BY c.embedding <=> %s::vector
                LIMIT 5;
                """,
                (
                    qvec,
                    config.embedding_model,
                    config.embedding_version,
                    qvec,
                ),
            )

            rows = cur.fetchall()
            print("vector joined rows:", len(rows))
            for row in rows:
                print(dict(row))
                print("---")

    print("\nRUNNING ACTUAL ProductionDatabase.hybrid_search")
    print("-----------------------------------------------")
    db = ProductionDatabase(config)
    try:
        contexts = db.hybrid_search(
            query_vector=query_vector,
            query_text=query,
        )
        print("contexts:", len(contexts))
        for context in contexts:
            print("source:", context.source)
            print("doc_id:", context.doc_id)
            print("parent_id:", context.parent_id)
            print("fusion_score:", context.fusion_score)
            print("best child:", context.best_child_text()[:300])
            print("---")
    finally:
        db.close()


if __name__ == "__main__":
    main()