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
import json
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

# ── Checkpointing ────────────────────────────────────────────────────────────

CHECKPOINT_FILE = "summarization_checkpoint.json"

def save_checkpoint(last_doc_pk: int, start_doc: int = None, end_doc: int = None):
    """Save progress checkpoint"""
    checkpoint = {
        "last_doc_pk": last_doc_pk,
        "start_doc": start_doc,
        "end_doc": end_doc,
        "timestamp": time.time()
    }
    try:
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2)
        logger.info(f"💾 Checkpoint saved: last_doc_pk={last_doc_pk}")
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")

def load_checkpoint():
    """Load progress checkpoint"""
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    
    try:
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None

def show_checkpoint_status():
    """Show current checkpoint status and summarization progress"""
    checkpoint = load_checkpoint()
    if checkpoint:
        logger.info("📊 Checkpoint Status:")
        logger.info(f"   Last processed doc: {checkpoint['last_doc_pk']}")
        logger.info(f"   Range: {checkpoint.get('start_doc', 'N/A')} - {checkpoint.get('end_doc', 'N/A')}")
        logger.info(f"   Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(checkpoint['timestamp']))}")
    else:
        logger.info("📊 No checkpoint found - will start fresh")
    
    # Show DB summary stats
    try:
        show_summary_stats()
    except Exception as e:
        logger.warning(f"Could not fetch summary stats: {e}")

def show_summary_stats():
    """Show detailed summary statistics from database"""
    doc_pks = db.fetch_all_doc_ids()
    total_docs = len(doc_pks)
    
    fully_summarized = 0
    l2_only = 0
    partial_l1 = 0
    not_started = 0
    
    for doc_pk in doc_pks:
        doc = db.fetch_doc_metadata(doc_pk)
        if not doc:
            continue
        doc_id = doc.get("doc_id", "")
        
        # Check L2
        has_l2 = db.already_summarized(level=2, source_id=doc_pk)
        
        # Check L1 count
        with db.get_cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) as cnt FROM {config.table_parent_chunks} WHERE doc_id = %s",
                (doc_id,),
            )
            parent_count = cur.fetchone()["cnt"]
            
            cur.execute(
                f"""
                SELECT COUNT(*) as cnt 
                FROM {config.table_summaries} 
                WHERE level = 1 
                  AND (metadata->>'doc_pk')::BIGINT = %s
                  AND status = 'done'
                """,
                (doc_pk,),
            )
            l1_count = cur.fetchone()["cnt"]
        
        if has_l2 and l1_count >= parent_count and parent_count > 0:
            fully_summarized += 1
        elif has_l2:
            l2_only += 1
        elif l1_count > 0:
            partial_l1 += 1
        else:
            not_started += 1
    
    logger.info("📈 Summarization Progress:")
    logger.info(f"   Total documents: {total_docs}")
    logger.info(f"   ✅ Fully summarized (L1+L2): {fully_summarized}")
    logger.info(f"   ⚠️  L2 only (missing L1): {l2_only}")
    logger.info(f"   📝 Partial L1 only: {partial_l1}")
    logger.info(f"   ⏳ Not started: {not_started}")
    logger.info(f"   📊 Completion: {fully_summarized}/{total_docs} ({100*fully_summarized//total_docs if total_docs > 0 else 0}%)")

def clear_checkpoint():
    """Clear checkpoint file"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
            logger.info("🗑️ Checkpoint cleared")
        except Exception as e:
            logger.warning(f"Failed to clear checkpoint: {e}")


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
        if not text:
            logger.warning(f"  [L1] Skipping empty parent chunk (ID={parent_row_id})")
            continue

        language = parent.get("language") or doc.get("language") or "ar"

        logger.info(f"  [L1] doc_pk={doc_pk} parent {idx + 1}/{len(parents)}")

        # 1. Summarize
        summary_text = summarizer.summarize_parent(
            chunk_text=text,
            doc_title=doc_title,
        )
        
        # Validate summary is not empty
        if not summary_text or not summary_text.strip():
            logger.warning(f"  [L1] Skipping - empty summary for parent chunk (ID={parent_row_id})")
            continue

        # 2. Embed (BGE-M3 on CPU)
        try:
            embedding = embedder.embed(summary_text)
        except Exception as e:
            logger.error(f"  [L1] Embedding failed for parent {parent_row_id}: {e}")
            continue

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
    # Check if already summarized at Level 2
    if config.skip_done and db.already_summarized(level=2, source_id=doc_pk):
        logger.info(f"  [L2] doc_pk={doc_pk} already summarized, skipping")
        return None

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
    
    # Validate summary is not empty
    if not summary_text or not summary_text.strip():
        logger.warning(f"  [L2] Document summary is empty for doc_pk={doc_pk}, using fallback")
        summary_text = "\n".join(texts[:3]) if texts else "لا توجد ملخصات متاحة"

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
    try:
        embedding = embedder.embed(summary_text)
    except Exception as e:
        logger.error(f"  [L2] Embedding failed for doc_pk={doc_pk}: {e}")
        embedding = None
    
    if embedding:
        return db.upsert_summary(level=2, source_id=doc_pk, summary_text=summary_text, metadata=metadata, embedding=embedding)
    else:
        logger.warning(f"  [L2] Skipping summary save - no embedding for doc_pk={doc_pk}")
        return None

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

def run_backfill(start_doc: int = None, end_doc: int = None, resume: bool = False):
    logger.info("=== BACKFILL START ===")
    
    # Load checkpoint if resuming
    checkpoint = load_checkpoint() if resume else None
    if resume and checkpoint:
        logger.info(f"🔄 Resuming from checkpoint: last_doc_pk={checkpoint['last_doc_pk']}")
        start_doc = checkpoint.get('start_doc') or start_doc
        end_doc = checkpoint.get('end_doc') or end_doc
        resume_from = checkpoint['last_doc_pk'] + 1
    else:
        resume_from = None
        clear_checkpoint()  # Clear old checkpoint if not resuming
    
    run_id = db.start_pipeline_run("backfill")
    doc_pks = db.fetch_all_doc_ids()

    # Filter documents by range if specified
    if start_doc is not None and end_doc is not None:
        doc_pks = [pk for pk in doc_pks if start_doc <= pk <= end_doc]
        logger.info(f"📋 Processing documents from {start_doc} to {end_doc} (filtered from {len(db.fetch_all_doc_ids())} total)")
    elif start_doc is not None:
        doc_pks = [pk for pk in doc_pks if pk >= start_doc]
        logger.info(f"📋 Processing documents from {start_doc} onwards (filtered from {len(db.fetch_all_doc_ids())} total)")
    elif end_doc is not None:
        doc_pks = [pk for pk in doc_pks if pk <= end_doc]
        logger.info(f"📋 Processing documents up to {end_doc} (filtered from {len(db.fetch_all_doc_ids())} total)")
    
    # If resuming, start from the next document
    if resume_from:
        doc_pks = [pk for pk in doc_pks if pk >= resume_from]
        logger.info(f"🔄 Resuming from document {resume_from}")

    # Pre-filter: Check which documents are already fully summarized
    docs_to_process = []
    skipped_count = 0
    for doc_pk in doc_pks:
        doc = db.fetch_doc_metadata(doc_pk)
        if not doc:
            continue
        doc_id = doc.get("doc_id")
        if config.skip_done and db.is_doc_fully_summarized(doc_pk, doc_id):
            skipped_count += 1
            continue
        docs_to_process.append(doc_pk)
    
    logger.info(f"📊 Total documents: {len(doc_pks)}, Already summarized: {skipped_count}, To process: {len(docs_to_process)}")

    total_summaries = 0
    docs_done = 0
    last_successful_doc = None

    for doc_pk in docs_to_process:
        try:
            logger.info(f"── Processing doc_pk {doc_pk} ──")
            l1 = run_level1_for_doc(doc_pk)
            l2 = run_level2_for_doc(doc_pk)
            total_summaries += len(l1) + (1 if l2 else 0)
            docs_done += 1
            last_successful_doc = doc_pk
            
            # Save checkpoint every 5 documents
            if docs_done % 5 == 0:
                save_checkpoint(last_successful_doc, start_doc, end_doc)
                
        except Exception as exc:
            logger.error(f"Error on doc_pk {doc_pk}: {exc}")
            # Save checkpoint on error too
            if last_successful_doc:
                save_checkpoint(last_successful_doc, start_doc, end_doc)
            continue

    clusters = run_level3_clustering()
    db.finish_pipeline_run(run_id, docs=docs_done, summaries=total_summaries + clusters)
    
    # Clear checkpoint on successful completion
    clear_checkpoint()
    logger.info("✅ Backfill completed successfully!")

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

def run_clear_partial_summaries():
    """Clear all partial (incomplete) summaries so they can be reprocessed from scratch."""
    logger.info("=== CLEAR PARTIAL SUMMARIES START ===")
    
    run_id = db.start_pipeline_run("clear_partial")
    doc_pks = db.fetch_all_doc_ids()
    
    cleared_count = 0
    skipped_count = 0
    
    for doc_pk in doc_pks:
        doc = db.fetch_doc_metadata(doc_pk)
        if not doc:
            continue
        doc_id = doc.get("doc_id", "")
        
        # Check if fully summarized
        if db.is_doc_fully_summarized(doc_pk, doc_id):
            skipped_count += 1
            continue
        
        # Not fully summarized - clear all its summaries
        db.delete_partial_summaries_for_doc(doc_pk, doc_id)
        cleared_count += 1
        logger.info(f"Cleared partial summaries for doc_pk={doc_pk} ({doc.get('file_name', 'N/A')})")
    
    logger.info(f"✅ Cleared {cleared_count} documents, skipped {skipped_count} fully summarized documents")
    db.finish_pipeline_run(run_id, docs=cleared_count, summaries=cleared_count)

# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arabic RAG Summarization Pipeline")
    parser.add_argument("--mode", choices=["backfill", "incremental", "recluster", "cleanup-chinese", "status", "stats", "clear-partial"], required=True)
    parser.add_argument("--doc-id", type=int, help="Document ID for incremental mode")
    parser.add_argument("--start-doc", type=int, help="Start document ID for backfill range")
    parser.add_argument("--end-doc", type=int, help="End document ID for backfill range")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    
    print(f"DEBUG: mode={args.mode}, resume={args.resume}")  # Debug print

    db.init_pool()
    t0 = time.time()

    if args.mode == "backfill":
        run_backfill(start_doc=args.start_doc, end_doc=args.end_doc, resume=args.resume)
    elif args.mode == "incremental" and args.doc_id:
        run_incremental(args.doc_id)
    elif args.mode == "recluster":
        run_level3_clustering()
    elif args.mode == "cleanup-chinese":
        run_cleanup_chinese()
    elif args.mode == "status":
        show_checkpoint_status()
    elif args.mode == "stats":
        show_summary_stats()
    elif args.mode == "clear-partial":
        run_clear_partial_summaries()

    logger.info(f"Total time: {time.time() - t0:.1f}s")
