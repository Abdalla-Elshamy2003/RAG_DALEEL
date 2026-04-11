from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import fitz
from docx import Document

try:
    from ingest_app.file_utils import chunk_list
    from ingest_app.text_utils import (
        clean_text, detect_lang, token_count_simple,
        build_structured_json, build_markdown_text, extract_links
    )
except ModuleNotFoundError:
    from file_utils import chunk_list
    from text_utils import (
        clean_text, detect_lang, token_count_simple,
        build_structured_json, build_markdown_text, extract_links
    )

def build_asset_prefix(file_hash: str) -> str:
    return f"file_{file_hash[:12]}"


def extract_pdf_assets(page: fitz.Page, asset_prefix: str) -> list[dict[str, Any]]:
    data = page.get_text("dict")
    blocks = data.get("blocks", [])
    assets = []
    asset_no = 1

    for block in blocks:
        if block.get("type") == 1 and block.get("image"):
            bbox = block.get("bbox") or []
            assets.append({
                "asset_id": f"{asset_prefix}_p{page.number + 1}_a{asset_no}",
                "asset_type": "image",
                "image_type": "embedded_image",
                "bbox": {
                    "x1": bbox[0] if len(bbox) > 0 else None,
                    "y1": bbox[1] if len(bbox) > 1 else None,
                    "x2": bbox[2] if len(bbox) > 2 else None,
                    "y2": bbox[3] if len(bbox) > 3 else None
                },
                "storage_uri": None,
                "ocr_text": None,
                "caption": None,
                "vision_summary": None,
                "tags": ["embedded_image"],
                "ocr_confidence": None,
                "vision_confidence": None,
                "is_relevant": True
            })
            asset_no += 1
    return assets


def build_pdf_payload(file_path: Path, file_hash: str) -> dict[str, Any]:
    with fitz.open(file_path) as doc:
        asset_prefix = build_asset_prefix(file_hash)
        pages = []
        all_text = []

        for page in doc:
            text_raw = page.get_text("text", sort=True) or ""
            text_raw = clean_text(text_raw)  # Clean and check for removal
            if not text_raw:  # Skip pages with empty text after cleaning
                continue
            all_text.append(text_raw)
            assets = extract_pdf_assets(page, asset_prefix)

            pages.append({
                "text_raw": text_raw,
                "language": detect_lang(text_raw),
                "char_count": len(text_raw),
                "token_count": token_count_simple(text_raw),
                "page_metadata": {
                    "has_tables": False,
                    "has_images": len(assets) > 0,
                    "has_footnotes": False
                },
                "assets": assets,
                "ocr_confidence": None
            })

        combined_text = "\n\n".join(t for t in all_text if t)
        raw_cleaned_content = combined_text
        structured_json = build_structured_json(file_hash, file_path.name, "pdf", raw_cleaned_content)
        markdown_text = structured_json["markdown_text"]

        return {
            "doc_id": file_hash,
            "file_name": file_path.name,
            "file_path": str(file_path).replace("\\", "/"),
            "file_hash": file_hash,
            "source_type": "pdf",
            "extraction_status": "success",
            "language": structured_json.get("language", "unknown"),
            "page_count": len(pages),
            "text_raw": raw_cleaned_content,
            "links": extract_links(raw_cleaned_content),
            "raw_cleaned_content": raw_cleaned_content,
            "structured_json": structured_json,
            "markdown_text": markdown_text,
            "legacy_payload": {
                "file_name": file_path.name,
                "file_path": str(file_path).replace("\\", "/"),
                "file_size_bytes": file_path.stat().st_size,
                "file_hash": file_hash,
                "language": detect_lang(combined_text),
                "source_type": "pdf",
                "pages": pages
            }
        }


def extract_docx_media_count(file_path: Path) -> int:
    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            return sum(1 for name in zf.namelist() if name.startswith("word/media/"))
    except Exception:
        return 0


def build_docx_payload(file_path: Path, file_hash: str, logical_page_paragraphs: int = 20) -> dict[str, Any]:
    doc = Document(file_path)
    asset_prefix = build_asset_prefix(file_hash)

    paragraphs = []
    for p in doc.paragraphs:
        txt = p.text or ""
        if txt.strip():
            paragraphs.append(txt)

    table_texts = []
    has_tables = len(doc.tables) > 0
    for table in doc.tables:
        rows = []
        for row in table.rows:
            cells = [(cell.text or "").strip() for cell in row.cells]
            rows.append(" | ".join(cell for cell in cells if cell))
        table_text = "\n".join(r for r in rows if r)
        table_text = clean_text(table_text)  # Clean table text
        if table_text.strip():
            table_texts.append(table_text)

    media_count = extract_docx_media_count(file_path)

    logical_pages_raw = chunk_list(paragraphs, logical_page_paragraphs) if paragraphs else [[]]
    pages = []
    all_text = []

    for idx, para_group in enumerate(logical_pages_raw, start=1):
        text_raw = "\n".join(para_group).strip()
        text_raw = clean_text(text_raw)  # Clean and check for removal
        if not text_raw:  # Skip pages with empty text after cleaning
            continue
        all_text.append(text_raw)

        assets = []
        if idx == 1 and media_count > 0:
            for a in range(1, media_count + 1):
                assets.append({
                    "asset_id": f"{asset_prefix}_p{idx}_a{a}",
                    "asset_type": "image",
                    "image_type": "embedded_image",
                    "bbox": None,
                    "storage_uri": None,
                    "ocr_text": None,
                    "caption": None,
                    "vision_summary": None,
                    "tags": ["embedded_image"],
                    "ocr_confidence": None,
                    "vision_confidence": None,
                    "is_relevant": True
                })

        pages.append({
            "text_raw": text_raw,
            "language": detect_lang(text_raw),
            "char_count": len(text_raw),
            "token_count": token_count_simple(text_raw),
            "page_metadata": {
                "has_tables": has_tables if idx == 1 else False,
                "has_images": len(assets) > 0,
                "has_footnotes": False
            },
            "assets": assets
        })

    for table_text in table_texts:
        table_text = clean_text(table_text)  # Ensure table text is cleaned
        if not table_text:  # Skip if empty after cleaning
            continue
        all_text.append(table_text)
        pages.append({
            "text_raw": table_text,
            "language": detect_lang(table_text),
            "char_count": len(table_text),
            "token_count": token_count_simple(table_text),
            "page_metadata": {
                "has_tables": True,
                "has_images": False,
                "has_footnotes": False
            },
            "assets": []
        })

    combined_text = "\n\n".join(t for t in all_text if t)
    raw_cleaned_content = combined_text
    structured_json = build_structured_json(file_hash, file_path.name, "docx", raw_cleaned_content)
    markdown_text = structured_json["markdown_text"]

    return {
        "doc_id": file_hash,
        "file_name": file_path.name,
        "file_path": str(file_path).replace("\\", "/"),
        "file_hash": file_hash,
        "source_type": "docx",
        "extraction_status": "success",
        "language": structured_json.get("language", "unknown"),
        "page_count": len(pages),
        "text_raw": raw_cleaned_content,
        "links": extract_links(raw_cleaned_content),
        "raw_cleaned_content": raw_cleaned_content,
        "structured_json": structured_json,
        "markdown_text": markdown_text,
        # Keep old payload for compatibility
        "legacy_payload": {
            "file_name": file_path.name,
            "file_path": str(file_path).replace("\\", "/"),
            "file_size_bytes": file_path.stat().st_size,
            "file_hash": file_hash,
            "language": detect_lang(combined_text),
            "source_type": "docx",
            "pages": pages
        }
    }


def build_txt_payload(file_path: Path, file_hash: str) -> dict[str, Any]:
    raw = file_path.read_text(encoding="utf-8", errors="ignore")
    raw_cleaned_content = clean_text(raw)  # Clean the text
    if not raw_cleaned_content:
        return {
            "doc_id": file_hash,
            "file_name": file_path.name,
            "file_path": str(file_path).replace("\\", "/"),
            "file_hash": file_hash,
            "source_type": "txt",
            "extraction_status": "empty",
            "language": "unknown",
            "page_count": 0,
            "text_raw": "",
            "links": [],
            "raw_cleaned_content": "",
            "structured_json": {},
            "markdown_text": "",
            "legacy_payload": {
                "file_name": file_path.name,
                "file_path": str(file_path).replace("\\", "/"),
                "file_size_bytes": file_path.stat().st_size,
                "file_hash": file_hash,
                "language": "unknown",
                "source_type": "txt",
                "pages": []
            }
        }

    structured_json = build_structured_json(file_hash, file_path.name, "txt", raw_cleaned_content)
    markdown_text = structured_json["markdown_text"]

    return {
        "doc_id": file_hash,
        "file_name": file_path.name,
        "file_path": str(file_path).replace("\\", "/"),
        "file_hash": file_hash,
        "source_type": "txt",
        "extraction_status": "success",
        "language": structured_json.get("language", "unknown"),
        "page_count": 1,
        "text_raw": raw_cleaned_content,
        "links": extract_links(raw_cleaned_content),
        "raw_cleaned_content": raw_cleaned_content,
        "structured_json": structured_json,
        "markdown_text": markdown_text,
        "legacy_payload": {
            "file_name": file_path.name,
            "file_path": str(file_path).replace("\\", "/"),
            "file_size_bytes": file_path.stat().st_size,
            "file_hash": file_hash,
            "language": detect_lang(raw_cleaned_content),
            "source_type": "txt",
            "pages": [
                {
                    "text_raw": raw_cleaned_content,
                    "language": detect_lang(raw_cleaned_content),
                    "char_count": len(raw_cleaned_content),
                    "token_count": token_count_simple(raw_cleaned_content),
                    "page_metadata": {
                        "has_tables": False,
                        "has_images": False,
                        "has_footnotes": False
                    },
                    "assets": []
                }
            ]
        }
    }