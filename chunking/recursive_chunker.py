from __future__ import annotations

from typing import Any

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
except Exception:
    RecursiveCharacterTextSplitter = None  # type: ignore

from .config import ChunkConfig, SEMANTIC_MODEL, STRATEGY_NAME, STRATEGY_VERSION
from .helpers import metadata_int, normalize_text, smart_concat, stable_id, utc_now_iso
from .models import ChildRow, ParentRow, TempChild
from .tokenizer import HFTokenizerWrapper


class RecursiveParentChildChunker:
    def __init__(self, tokenizer_model: str, config: ChunkConfig) -> None:
        if RecursiveCharacterTextSplitter is None:
            raise RuntimeError(
                "langchain-text-splitters is required. Install: pip install langchain-text-splitters"
            )

        self.config = config
        self.tokenizer = HFTokenizerWrapper(tokenizer_model)
        self.tokenizer_model = tokenizer_model

        self.parent_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer.tokenizer,
            chunk_size=config.parent_chunk_size,
            chunk_overlap=config.parent_chunk_overlap,
            separators=list(config.separators),
            keep_separator=True,
            add_start_index=True,
            strip_whitespace=True,
        )

        self.child_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer.tokenizer,
            chunk_size=config.child_chunk_size,
            chunk_overlap=config.child_chunk_overlap,
            separators=list(config.separators),
            keep_separator=True,
            add_start_index=True,
            strip_whitespace=True,
        )

    def build_rows_for_document(
        self,
        doc: dict[str, Any],
        build_run_id: str,
    ) -> tuple[list[ParentRow], list[ChildRow]]:
        doc_id = str(doc["doc_id"])
        text = normalize_text(doc.get("raw_cleaned_content") or "")
        if not text:
            return [], []

        source_meta = {
            "doc_file_name": doc.get("file_name"),
            "doc_file_path": doc.get("file_path"),
            "doc_file_hash": doc.get("file_hash"),
            "doc_source_type": doc.get("source_type"),
            "doc_language": doc.get("language"),
            "doc_page_count": doc.get("page_count"),
            "doc_extraction_status": doc.get("extraction_status"),
        }

        parent_docs = self.parent_splitter.create_documents([text])

        parents: list[ParentRow] = []
        children: list[ChildRow] = []
        global_child_index = 0

        for parent_index, parent_doc in enumerate(parent_docs):
            parent_text = normalize_text(parent_doc.page_content)
            if not parent_text:
                continue

            parent_start = metadata_int(parent_doc.metadata.get("start_index"))
            parent_end = (parent_start + len(parent_text)) if parent_start is not None else None
            parent_token_count = self.tokenizer.count_tokens(parent_text)
            parent_id = stable_id("parent", doc_id, parent_index, parent_start, parent_end)

            parent_metadata = {
                "build_run_id": build_run_id,
                "build_timestamp_utc": utc_now_iso(),
                "strategy_name": STRATEGY_NAME,
                "strategy_version": STRATEGY_VERSION,
                "semantic_model": SEMANTIC_MODEL,
                "tokenizer_model": self.tokenizer_model,
                "parent_chunk_size": self.config.parent_chunk_size,
                "parent_chunk_overlap": self.config.parent_chunk_overlap,
                "child_chunk_size": self.config.child_chunk_size,
                "child_chunk_overlap": self.config.child_chunk_overlap,
                "min_child_chunk_tokens": self.config.min_child_chunk_tokens,
                "separators": list(self.config.separators),
                "parent_index": parent_index,
                "source_kind": "raw_cleaned_content",
                **source_meta,
            }

            parents.append(
                ParentRow(
                    parent_id=parent_id,
                    doc_id=doc_id,
                    parent_index=parent_index,
                    text=parent_text,
                    token_count=parent_token_count,
                    char_count=len(parent_text),
                    start_char=parent_start,
                    end_char=parent_end,
                    metadata=parent_metadata,
                )
            )

            raw_children: list[TempChild] = []
            child_docs = self.child_splitter.create_documents([parent_text])

            for child_doc in child_docs:
                child_text = normalize_text(child_doc.page_content)
                if not child_text:
                    continue

                child_local_start = metadata_int(child_doc.metadata.get("start_index"))
                child_global_start = (
                    parent_start + child_local_start
                    if parent_start is not None and child_local_start is not None
                    else None
                )
                child_global_end = (
                    child_global_start + len(child_text)
                    if child_global_start is not None
                    else None
                )
                child_token_count = self.tokenizer.count_tokens(child_text)

                raw_children.append(
                    TempChild(
                        text=child_text,
                        token_count=child_token_count,
                        char_count=len(child_text),
                        start_char=child_global_start,
                        end_char=child_global_end,
                    )
                )

            merged_children = self._merge_small_children(raw_children)

            for temp_child in merged_children:
                child_id = stable_id(
                    "child",
                    doc_id,
                    global_child_index,
                    temp_child.start_char,
                    temp_child.end_char,
                )

                child_metadata = {
                    "build_run_id": build_run_id,
                    "build_timestamp_utc": utc_now_iso(),
                    "strategy_name": STRATEGY_NAME,
                    "strategy_version": STRATEGY_VERSION,
                    "semantic_model": SEMANTIC_MODEL,
                    "tokenizer_model": self.tokenizer_model,
                    "parent_id": parent_id,
                    "parent_index": parent_index,
                    "parent_chunk_size": self.config.parent_chunk_size,
                    "parent_chunk_overlap": self.config.parent_chunk_overlap,
                    "child_chunk_size": self.config.child_chunk_size,
                    "child_chunk_overlap": self.config.child_chunk_overlap,
                    "min_child_chunk_tokens": self.config.min_child_chunk_tokens,
                    "separators": list(self.config.separators),
                    "source_kind": "raw_cleaned_content",
                    **source_meta,
                }

                children.append(
                    ChildRow(
                        child_id=child_id,
                        parent_id=parent_id,
                        doc_id=doc_id,
                        child_index=global_child_index,
                        text=temp_child.text,
                        token_count=temp_child.token_count,
                        char_count=temp_child.char_count,
                        start_char=temp_child.start_char,
                        end_char=temp_child.end_char,
                        metadata=child_metadata,
                    )
                )
                global_child_index += 1

        return parents, children

    def _merge_small_children(self, items: list[TempChild]) -> list[TempChild]:
        if not items:
            return []

        merged: list[TempChild] = []
        pending: TempChild | None = None

        for item in items:
            current = item

            if pending is not None:
                current = self._combine_children(pending, current)
                pending = None

            if current.token_count < self.config.min_child_chunk_tokens:
                if merged:
                    merged[-1] = self._combine_children(merged[-1], current)
                else:
                    pending = current
            else:
                merged.append(current)

        if pending is not None:
            if merged:
                merged[-1] = self._combine_children(merged[-1], pending)
            else:
                merged.append(pending)

        return merged

    def _combine_children(self, left: TempChild, right: TempChild) -> TempChild:
        combined_text = smart_concat(left.text, right.text)
        combined_tokens = self.tokenizer.count_tokens(combined_text)

        return TempChild(
            text=combined_text,
            token_count=combined_tokens,
            char_count=len(combined_text),
            start_char=left.start_char if left.start_char is not None else right.start_char,
            end_char=right.end_char if right.end_char is not None else left.end_char,
        )