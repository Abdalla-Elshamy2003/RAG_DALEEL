"""
pipeline.py — Summarization pipeline orchestrator.
Optimized for Local Execution (Ollama + CPU Embeddings).
Integrated with keyword extraction and structured summaries.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import db
from summarizer import Summarizer
from config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ── Lazy singletons (Memory Safety for RTX 2060) ─────────────────────────────

_summarizer: Optional[Summarizer] = None
_embedder = None

def get_summarizer() -> Summarizer:
    global _summarizer
    if _summarizer is None:
        _summarizer = Summarizer()
    return _summarizer

def get_embedder():
    global _embedder
    if _embedder is None:
        from embedder import Embedder
        _embedder = Embedder()
    return _embedder


# ── Level 1: Parent Chunk Summarization ──────────────────────────────────────

def run_level1_for_doc(doc_pk: int) -> list[int]:
    doc = db.fetch_doc_metadata(doc_pk)
    if not doc: return []

    doc_text_id = doc["doc_id"]
    doc_title = doc.get("file_name") or f"doc_{doc_pk}"
    parents = db.fetch_parents_for_doc(doc_text_id)
    
    if not parents: return []

    summarizer = get_summarizer()
    embedder = get_embedder()
    summary_ids = []

    for idx, parent in enumerate(parents):
        parent_row_id = parent["id"]

        if config.skip_done and db.already_summarized(level=1, source_id=parent_row_id):
            continue

        text = (parent.get("text") or "").strip()
        if not text: continue

        language = parent.get("language") or doc.get("language") or "ar"

        logger.info(f"  [L1] doc_pk={doc_pk} parent {idx + 1}/{len(parents)}")

        # 1. Summarize
        summary_text = summarizer.summarize_parent(
            chunk_text=text,
            doc_title=doc_title,
        )

        # 2. Embed (BGE-M3 on CPU)
        embedding = embedder.embed(summary_text)

        metadata = {
            "doc_pk": doc_pk,
            "doc_id": doc_text_id,
            "doc_title": doc_title,
            "lang": language,
        }

        # 3. Save
        summary_id = db.upsert_summary(
            level=1,
            source_id=parent_row_id,
            summary_text=summary_text,
            embedding=embedding,
            metadata=metadata,
        )
        summary_ids.append(summary_id)

    return summary_ids


# ── Level 2: Document Summarization ──────────────────────────────────────────
def run_level2_for_doc(doc_pk: int) -> Optional[int]:
    l1_summaries = db.fetch_level1_summaries_for_doc(doc_pk)
    if not l1_summaries: return None

    texts = [s["summary_text"] for s in l1_summaries if s.get("summary_text")]
    doc_metadata = db.fetch_doc_metadata(doc_pk)
    doc_title = doc_metadata.get("file_name") or f"doc_{doc_pk}"

    summarizer = get_summarizer()
    embedder = get_embedder()
    
    # 1. Generate the Structured Summary
    res = summarizer.summarize_document(texts, doc_title)
    summary_text = res["text"]
    keywords = res["keywords"]

    # 2. Automated Quality Audit (Sampling)
    quality_audit = {}
    if doc_pk % 10 == 0: # Every 10th document
        logger.info(f"  [Audit] Running LLM-Judge for doc_pk={doc_pk}...")
        # Give the judge the first 3 chunks as context to check against the summary
        judge_context = "\n\n".join(texts[:3])
        eval_res = summarizer.judge_mod(original_text=judge_context, summary=summary_text)
        
        quality_audit = {
            "faithfulness": eval_res.faithfulness,
            "relevance": eval_res.relevance,
            "coherence": eval_res.coherence,
            "critique": eval_res.critique
        }

    # 3. Prepare Metadata
    metadata = {
        "doc_pk": doc_pk,
        "doc_id": doc_metadata.get("doc_id"),
        "doc_title": doc_title,
        "keywords": keywords,
        "quality_audit": quality_audit, # Saved into JSONB column!
        "parent_count": len(l1_summaries),
        "lang": "ar"
    }

    # 4. Embed and Save
    
    embedding = embedder.embed(summary_text)
    return db.upsert_summary(level=2, source_id=doc_pk, summary_text=summary_text, metadata=metadata, embedding=embedding)

# def run_level2_for_doc(doc_pk: int) -> Optional[int]:
#     l1_summaries = db.fetch_level1_summaries_for_doc(doc_pk)
#     if not l1_summaries: return None

#     texts = [s["summary_text"] for s in l1_summaries if s.get("summary_text")]
#     source_parent_ids = [s["source_id"] for s in l1_summaries] 
    
#     doc = db.fetch_doc_metadata(doc_pk)
#     doc_title = doc.get("file_name") or f"doc_{doc_pk}"

#     summarizer = get_summarizer()
#     embedder = get_embedder()
    
#     keywords = []
    
#     if len(l1_summaries) == 1:
#         # Simple case: wrap the L1 summary with context
#         summary_text = f"This document focuses on: {texts[0]}"
#         logger.info(f"  [L2] doc_pk={doc_pk} has 1 parent. Reference generated.")
#     else:
#         # Advanced case: Structural synthesis with keyword extraction
#         logger.info(f"  [L2] Synthesizing summary for doc_pk={doc_pk}")
#         res = summarizer.summarize_document(texts, doc_title)
#         summary_text = res["text"]
#         keywords = res["keywords"]

#     # Generate metadata with provenance and new keywords
#     metadata = {
#         "doc_pk": doc_pk,
#         "doc_id": doc.get("doc_id"),
#         "doc_title": doc_title,
#         "parent_count": len(l1_summaries),
#         "source_parents": source_parent_ids,
#         "keywords": keywords,
#         "lang": "ar"
#     }

#     # Embed the high-level summary
#     embedding = embedder.embed(summary_text)

#     return db.upsert_summary(
#         level=2,
#         source_id=doc_pk,
#         summary_text=summary_text,
#         metadata=metadata,
#         embedding=embedding
#     )


# ── Level 3: Cross-Doc Clustering ────────────────────────────────────────────

def run_level3_clustering() -> int:
    logger.info("=== [L3] Starting Semantic Clustering ===")
    
    # Wipe old clusters to ensure a clean thematic map (Idempotency)
    db.delete_summaries_at_level(3)

    l2_data = db.fetch_l2_summaries_with_embeddings()
    if len(l2_data) < 2:
        logger.info("[L3] Not enough documents to cluster.")
        return 0

    embedder = get_embedder()
    summarizer = get_summarizer()

    # Perform Agglomerative Clustering
    clusters, singletons = embedder.cluster_by_similarity(l2_data)
    total_created = 0

    # 1. Handle Thematic Clusters
    for cluster_doc_ids in clusters:
        cluster_docs = [d for d in l2_data if d["doc_id"] in cluster_doc_ids]
        texts = [d["summary_text"] for d in cluster_docs]

        # Use AI to generate a meaningful topic name
        topic_tag = summarizer.generate_topic_tag(texts)
        logger.info(f"  [L3] Creating cluster summary for: {topic_tag}")

        res = summarizer.summarize_cluster(texts, topic_tag)
        summary_text = res["text"]
        keywords = res["keywords"]
        
        embedding = embedder.embed(summary_text)

        db.upsert_summary(
            level=3,
            source_id=None,
            summary_text=summary_text,
            embedding=embedding,
            cluster_doc_ids=cluster_doc_ids,
            metadata={
                "topic_tag": topic_tag, 
                "keywords": keywords, 
                "type": "thematic"
            },
        )
        total_created += 1

    # 2. Handle Unique Documents (Singletons)
    if singletons:
        # Group singletons in batches of 10 for efficiency
        for i in range(0, len(singletons), 10):
            batch_ids = singletons[i:i + 10]
            batch_docs = [d for d in l2_data if d["doc_id"] in batch_ids]
            texts = [d["summary_text"] for d in batch_docs]

            logger.info(f"  [L3] Batching {len(batch_ids)} unique documents")
            summary_text = summarizer.summarize_miscellaneous(texts)
            embedding = embedder.embed(summary_text)

            db.upsert_summary(
                level=3,
                source_id=None,
                summary_text=summary_text,
                embedding=embedding,
                cluster_doc_ids=batch_ids,
                metadata={
                    "topic_tag": "Miscellaneous Topics", 
                    "type": "miscellaneous"
                },
            )
            total_created += 1

    return total_created


# ── Execution Logic ──────────────────────────────────────────────────────────

def run_backfill():
    logger.info("=== BACKFILL START ===")
    run_id = db.start_pipeline_run("backfill")
    doc_pks = db.fetch_all_doc_ids()

    total_summaries = 0
    docs_done = 0

    for doc_pk in doc_pks:
        try:
            logger.info(f"── Processing doc_pk {doc_pk} ──")
            l1 = run_level1_for_doc(doc_pk)
            l2 = run_level2_for_doc(doc_pk)
            total_summaries += len(l1) + (1 if l2 else 0)
            docs_done += 1
        except Exception as exc:
            logger.error(f"Error on doc_pk {doc_pk}: {exc}")

    clusters = run_level3_clustering()
    db.finish_pipeline_run(run_id, docs=docs_done, summaries=total_summaries + clusters)

def run_incremental(doc_id: int):
    run_id = db.start_pipeline_run("incremental")
    try:
        l1 = run_level1_for_doc(doc_id)
        l2 = run_level2_for_doc(doc_id)
        l3 = run_level3_clustering()
        db.finish_pipeline_run(run_id, docs=1, summaries=len(l1)+1+l3)
    except Exception as exc:
        db.finish_pipeline_run(run_id, docs=0, summaries=0, error=str(exc))

def run_cleanup_chinese():
    """تنظيف جميع النصوص الصينية من قاعدة البيانات وتحويلها إلى عربي/إنجليزي"""
    from summarizer import clean_output
    
    run_id = db.start_pipeline_run("cleanup_chinese")
    try:
        logger.info("=== CLEANUP CHINESE TEXT START ===")
        summaries = db.fetch_summaries_with_chinese_text()
        
        if not summaries:
            logger.info("✓ No Chinese text found in summaries!")
            db.finish_pipeline_run(run_id, docs=0, summaries=0)
            return
        
        cleaned_count = 0
        logger.info(f"Found {len(summaries)} summaries with Chinese text")
        
        for idx, summary in enumerate(summaries):
            summary_id = summary["id"]
            level = summary["level"]
            source_id = summary["source_id"]
            original_text = summary["summary_text"]
            
            # تنظيف النص من النصوص الصينية
            cleaned_text = clean_output(original_text)
            
            if cleaned_text != original_text:
                logger.info(f"  [{idx+1}/{len(summaries)}] Cleaning L{level} summary (ID={summary_id})")
                logger.info(f"    Before: {original_text[:100]}...")
                logger.info(f"    After:  {cleaned_text[:100]}...")
                
                db.update_summary_text(summary_id, cleaned_text)
                cleaned_count += 1
            else:
                logger.info(f"  [{idx+1}/{len(summaries)}] Already clean L{level} summary (ID={summary_id})")
        
        logger.info(f"✓ Cleaned {cleaned_count} summaries successfully!")
        db.finish_pipeline_run(run_id, docs=cleaned_count, summaries=cleaned_count)
        
    except Exception as exc:
        logger.error(f"Error during cleanup: {exc}")
        db.finish_pipeline_run(run_id, docs=0, summaries=0, error=str(exc))

# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arabic RAG Summarization Pipeline")
    parser.add_argument("--mode", choices=["backfill", "incremental", "recluster", "cleanup-chinese"], required=True)
    parser.add_argument("--doc-id", type=int)
    args = parser.parse_args()

    db.init_pool()
    t0 = time.time()

    if args.mode == "backfill":
        run_backfill()
    elif args.mode == "incremental" and args.doc_id:
        run_incremental(args.doc_id)
    elif args.mode == "recluster":
        run_level3_clustering()
    elif args.mode == "cleanup-chinese":
        run_cleanup_chinese()

    logger.info(f"Total time: {time.time() - t0:.1f}s")
