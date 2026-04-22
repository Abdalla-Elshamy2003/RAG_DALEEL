from pathlib import Path

env_content = '''DB_CONN="host=135.181.117.17 port=5432 dbname=docs_ingestion user=abdu password=Neurix@123!@#"

EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_VERSION=2
RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3

VECTOR_CANDIDATES=100
KEYWORD_CANDIDATES=100
FUSED_CHILD_LIMIT=12
PARENT_LIMIT=6
RERANK_TOP_K=3
RERANKER_MAX_CHARS=2500
CPU_SKIP_RERANKER=true
RRF_K=60
HNSW_EF_SEARCH=100

WEB_FALLBACK_ENABLED=false
TAVILY_MAX_RESULTS=3
TAVILY_SEARCH_DEPTH=basic
WEB_TIMEOUT_SECONDS=5
TAVILY_API_KEY="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"

LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b 
OLLAMA_CONNECT_TIMEOUT_SECONDS=10
OLLAMA_REQUEST_TIMEOUT_SECONDS=600
OLLAMA_STREAM_TIMEOUT_SECONDS=600
OLLAMA_NUM_CTX=4096
OLLAMA_NUM_PREDICT=128
OLLAMA_KEEP_ALIVE=30m
PROMPT_CONTEXT_LIMIT=2
ALLOW_GENERAL_KNOWLEDGE_FALLBACK=true

ANSWER_LANGUAGE=same_as_question
ANSWER_STYLE=detailed, professional, well-explained, and explicit about evidence
'''

Path(".env").write_text(env_content, encoding="utf-8")

print("Fixed .env successfully")
print("Path:", Path(".env").resolve())
print("Length:", Path(".env").stat().st_size)
