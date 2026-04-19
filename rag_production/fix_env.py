from pathlib import Path

env_content = '''DB_CONN="host=135.181.117.17 port=5432 dbname=docs_ingestion user=abdu password=Neurix@123!@#"

EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_VERSION=2
RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3

VECTOR_CANDIDATES=80
KEYWORD_CANDIDATES=80
FUSED_CHILD_LIMIT=40
PARENT_LIMIT=10
RERANK_TOP_K=5
RRF_K=60
HNSW_EF_SEARCH=100

WEB_FALLBACK_ENABLED=true
TAVILY_MAX_RESULTS=5
TAVILY_API_KEY="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"
WEB_FALLBACK_ENABLED=true

LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b 

ANSWER_LANGUAGE=same_as_question
ANSWER_STYLE=professional, clear, grounded, and concise
'''

Path(".env").write_text(env_content, encoding="utf-8")

print("Fixed .env successfully")
print("Path:", Path(".env").resolve())
print("Length:", Path(".env").stat().st_size)