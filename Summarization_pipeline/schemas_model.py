from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import psycopg2
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    inspect,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import event
from pgvector.sqlalchemy import Vector

from config import config


# ============================================================
# Base
# ============================================================

class Base(DeclarativeBase):
    pass


# ============================================================
# Existing table: post_processing_data
# Real schema from your database
# ============================================================

class PostProcessingData(Base):
    __tablename__ = "post_processing_data"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    doc_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)

    file_name: Mapped[Optional[str]] = mapped_column(Text)
    file_path: Mapped[Optional[str]] = mapped_column(Text)
    file_ext: Mapped[Optional[str]] = mapped_column(Text)
    file_hash: Mapped[Optional[str]] = mapped_column(Text)
    source_type: Mapped[Optional[str]] = mapped_column(Text)
    extraction_status: Mapped[Optional[str]] = mapped_column(Text)
    language: Mapped[Optional[str]] = mapped_column(Text)
    page_count: Mapped[Optional[int]] = mapped_column(Integer)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


# ============================================================
# Existing table: parent_chunks
# Real schema from your database
# ============================================================

class ParentChunk(Base):
    __tablename__ = "parent_chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    parent_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)

    # IMPORTANT:
    # parent_chunks.doc_id references post_processing_data.doc_id (TEXT), not post_processing_data.id
    doc_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("post_processing_data.doc_id", ondelete="CASCADE"),
        nullable=False,
    )

    parent_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)

    start_char: Mapped[Optional[int]] = mapped_column(Integer)
    end_char: Mapped[Optional[int]] = mapped_column(Integer)

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("doc_id", "parent_index", name="uq_parent_chunks_doc_id_parent_index"),
    )


# ============================================================
# New table: summaries
# Compatible with your real tables
# ============================================================

class Summary(Base):
    __tablename__ = "summaries"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    # 1 = parent summary
    # 2 = document summary
    # 3 = cross-document cluster summary
    level: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    # Level 1 -> parent_chunks.id
    # Level 2 -> post_processing_data.id
    # Level 3 -> NULL
    source_id: Mapped[Optional[int]] = mapped_column(BigInteger)

    # IMPORTANT:
    # store NUMERIC document PKs here (post_processing_data.id),
    # because your real doc_id key is TEXT and this column is BIGINT[]
    cluster_doc_ids: Mapped[Optional[list[int]]] = mapped_column(ARRAY(BigInteger))

    summary_text: Mapped[str] = mapped_column(Text, nullable=False)

    # BAAI/bge-m3 = 1024 dims
    embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(1024))

    # Metadata conventions:
    # Level 1:
    # {
    #   "doc_pk": 248,
    #   "doc_id": "doc_abc123",
    #   "file_name": "foo.pdf",
    #   "parent_num": 3,
    #   "parent_chunk_id": 991,
    #   "parent_id": "parent_991",
    #   "lang": "ar"
    # }
    #
    # Level 2:
    # {
    #   "doc_pk": 248,
    #   "doc_id": "doc_abc123",
    #   "file_name": "foo.pdf",
    #   "parent_count": 12,
    #   "lang": "ar"
    # }
    #
    # Level 3:
    # {
    #   "cluster_id": "topic-001",
    #   "topic_tag": "sports",
    #   "doc_count": 4,
    #   "doc_ids": ["doc_abc123", "doc_xyz999"],
    #   "lang": "ar"
    # }
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint("level IN (1, 2, 3)", name="ck_summaries_level"),
        CheckConstraint(
            "status IN ('pending', 'processing', 'done', 'failed')",
            name="ck_summaries_status",
        ),
        UniqueConstraint("level", "source_id", name="uq_summaries_level_source"),
    )

    def flat_dict(self) -> dict[str, Any]:
        """
        Python replacement for the old SQL view.
        """
        meta = self.metadata_json or {}
        return {
            "id": self.id,
            "level": self.level,
            "source_id": self.source_id,
            "cluster_doc_ids": self.cluster_doc_ids,
            "summary_text": self.summary_text,
            "file_name": meta.get("file_name"),
            "lang": meta.get("lang"),
            "doc_pk": meta.get("doc_pk"),
            "doc_id": meta.get("doc_id"),
            "parent_num": meta.get("parent_num"),
            "status": self.status,
            "created_at": self.created_at,
        }


# ============================================================
# New table: summarization_pipeline_runs
# ============================================================

class SummarizationPipelineRun(Base):
    __tablename__ = "summarization_pipeline_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    run_type: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="running")

    docs_processed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    summaries_created: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    error_message: Mapped[Optional[str]] = mapped_column(Text)

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(
            "run_type IN ('backfill', 'incremental', 'recluster')",
            name="ck_pipeline_runs_type",
        ),
        CheckConstraint(
            "status IN ('running', 'done', 'failed')",
            name="ck_pipeline_runs_status",
        ),
    )


# ============================================================
# Indexes
# ============================================================

Index("idx_summaries_level_source", Summary.level, Summary.source_id)

Index(
    "idx_summaries_status",
    Summary.status,
    postgresql_where=Summary.status.in_(["pending", "failed"]),
)

# General JSONB GIN index: cleaner and more flexible in Python
Index(
    "idx_summaries_metadata_gin",
    Summary.metadata_json,
    postgresql_using="gin",
)

# HNSW vector index for pgvector
Index(
    "idx_summaries_embedding_hnsw",
    Summary.embedding,
    postgresql_using="hnsw",
    postgresql_ops={"embedding": "vector_cosine_ops"},
    postgresql_with={"m": 16, "ef_construction": 64},
)


# ============================================================
# Auto-update updated_at in Python
# Replaces SQL trigger
# ============================================================

@event.listens_for(Summary, "before_update", propagate=True)
def update_summary_updated_at(mapper, connection, target):
    target.updated_at = datetime.now(timezone.utc)


@event.listens_for(ParentChunk, "before_update", propagate=True)
def update_parent_chunk_updated_at(mapper, connection, target):
    target.updated_at = datetime.now(timezone.utc)


# ============================================================
# Engine / bootstrap
# ============================================================

def build_engine():
    """
    Uses your existing libpq DSN from config.db_conn.
    No need to convert it to a SQLAlchemy URL manually.
    """
    return create_engine(
        "postgresql+psycopg2://",
        creator=lambda: psycopg2.connect(config.db_conn),
        future=True,
        pool_pre_ping=True,
    )


def validate_existing_schema(engine) -> None:
    """
    Validates the existing tables that your pipeline depends on.
    This does NOT alter them; it only checks compatibility.
    """
    inspector = inspect(engine)

    required = {
        "post_processing_data": {
            "id",
            "doc_id",
            "file_name",
            "payload",
        },
        "parent_chunks": {
            "id",
            "parent_id",
            "doc_id",
            "parent_index",
            "text",
            "metadata",
        },
    }

    for table_name, expected_cols in required.items():
        if not inspector.has_table(table_name):
            raise RuntimeError(f"Missing required table: {table_name}")

        real_cols = {col["name"] for col in inspector.get_columns(table_name)}
        missing = expected_cols - real_cols
        if missing:
            raise RuntimeError(
                f"Table '{table_name}' is missing columns: {sorted(missing)}"
            )


def init_schema() -> None:
    """
    Creates only the new summarization tables and indexes.
    Existing source tables are validated, not recreated.
    """
    engine = build_engine()
    validate_existing_schema(engine)

    Base.metadata.create_all(
        engine,
        tables=[
            Summary.__table__,
            SummarizationPipelineRun.__table__,
        ],
    )

    print("Schema ready.")