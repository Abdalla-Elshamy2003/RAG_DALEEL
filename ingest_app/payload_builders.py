from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import fitz
from docx import Document

from ingest_app.file_utils import chunk_list
from ingest_app.text_utils import clean_text, detect_lang, token_count_simple


def extract_pdf_assets(page: fitz.Page, doc_id: str) -> list[dict[str, Any]]:
    data = page.get_text("dict")
    blocks = data.get("blocks", [])
    assets = []
    asset_no = 1

    for block in blocks:
        if block.get("type") == 1 and block.get("image"):
            bbox = block.get("bbox") or []
            assets.append({
                "asset_id": f"{doc_id}_p{page.number + 1}_a{asset_no}",
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
        meta = dict(doc.metadata or {})
        doc_id = f"doc_{file_hash[:12]}"
        pages = []
        all_text = []

        for idx, page in enumerate(doc, start=1):
            native_raw = page.get_text("text", sort=True) or ""
            native_clean = clean_text(native_raw)

            extraction_method = "text"
            final_raw = native_raw
            final_clean = native_clean
            ocr_conf = None

            if not final_clean:
                extraction_method = "empty"

            all_text.append(final_clean)
            assets = extract_pdf_assets(page, doc_id)

            pages.append({
                "page_no": idx,
                "extraction_method": extraction_method,
                "text_raw": final_raw,
                "text_clean": final_clean,
                "language": detect_lang(final_clean),
                "char_count": len(final_clean),
                "token_count": token_count_simple(final_clean),
                "page_metadata": {
                    "has_tables": False,
                    "has_images": len(assets) > 0,
                    "has_footnotes": False
                },
                "assets": assets,
                "ocr_confidence": ocr_conf
            })

        combined_text = "\n\n".join(t for t in all_text if t)

        return {
            "doc_id": doc_id,
            "file_name": file_path.name,
            "file_path": str(file_path).replace("\\", "/"),
            "file_size_bytes": file_path.stat().st_size,
            "file_hash": file_hash,
            "language": detect_lang(combined_text),
            "page_count": len(doc),
            "extraction_status": "completed",
            "source_type": "pdf",
            "metadata": {
                "title": meta.get("title"),
                "author": meta.get("author"),
                "created_at_source": meta.get("creationDate"),
                "keywords": []
            },
            "pages": pages
        }


def extract_docx_media_count(file_path: Path) -> int:
    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            return sum(1 for name in zf.namelist() if name.startswith("word/media/"))
    except Exception:
        return 0


def build_docx_payload(file_path: Path, file_hash: str, logical_page_paragraphs: int = 20) -> dict[str, Any]:
    doc = Document(file_path)
    props = doc.core_properties
    doc_id = f"doc_{file_hash[:12]}"

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
            cells = [clean_text(cell.text or "") for cell in row.cells]
            rows.append(" | ".join(cell for cell in cells if cell))
        table_text = "\n".join(r for r in rows if r)
        if table_text.strip():
            table_texts.append(table_text)

    media_count = extract_docx_media_count(file_path)

    logical_pages_raw = chunk_list(paragraphs, logical_page_paragraphs) if paragraphs else [[]]
    pages = []
    all_text = []

    for idx, para_group in enumerate(logical_pages_raw, start=1):
        text_raw = "\n".join(para_group).strip()
        text_clean = clean_text(text_raw)
        all_text.append(text_clean)

        assets = []
        if idx == 1 and media_count > 0:
            for a in range(1, media_count + 1):
                assets.append({
                    "asset_id": f"{doc_id}_p{idx}_a{a}",
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
            "page_no": idx,
            "extraction_method": "logical_docx_page",
            "text_raw": text_raw,
            "text_clean": text_clean,
            "language": detect_lang(text_clean),
            "char_count": len(text_clean),
            "token_count": token_count_simple(text_clean),
            "page_metadata": {
                "has_tables": has_tables if idx == 1 else False,
                "has_images": len(assets) > 0,
                "has_footnotes": False
            },
            "assets": assets
        })

    start_page_no = len(pages) + 1
    for offset, table_text in enumerate(table_texts, start=0):
        text_clean = clean_text(table_text)
        all_text.append(text_clean)
        pages.append({
            "page_no": start_page_no + offset,
            "extraction_method": "table_from_docx",
            "text_raw": table_text,
            "text_clean": text_clean,
            "language": detect_lang(text_clean),
            "char_count": len(text_clean),
            "token_count": token_count_simple(text_clean),
            "page_metadata": {
                "has_tables": True,
                "has_images": False,
                "has_footnotes": False
            },
            "assets": []
        })

    combined_text = "\n\n".join(t for t in all_text if t)

    return {
        "doc_id": doc_id,
        "file_name": file_path.name,
        "file_path": str(file_path).replace("\\", "/"),
        "file_size_bytes": file_path.stat().st_size,
        "file_hash": file_hash,
        "language": detect_lang(combined_text),
        "page_count": len(pages),
        "extraction_status": "completed",
        "source_type": "docx",
        "metadata": {
            "title": getattr(props, "title", None),
            "author": getattr(props, "author", None),
            "created_at_source": str(getattr(props, "created", None)) if getattr(props, "created", None) else None,
            "keywords": []
        },
        "pages": pages
    }


def build_txt_payload(file_path: Path, file_hash: str) -> dict[str, Any]:
    raw = file_path.read_text(encoding="utf-8", errors="ignore")
    clean = clean_text(raw)
    doc_id = f"doc_{file_hash[:12]}"

    return {
        "doc_id": doc_id,
        "file_name": file_path.name,
        "file_path": str(file_path).replace("\\", "/"),
        "file_size_bytes": file_path.stat().st_size,
        "file_hash": file_hash,
        "language": detect_lang(clean),
        "page_count": 1,
        "extraction_status": "completed",
        "source_type": "txt",
        "metadata": {
            "title": file_path.stem,
            "author": None,
            "created_at_source": None,
            "keywords": []
        },
        "pages": [
            {
                "page_no": 1,
                "extraction_method": "text",
                "text_raw": raw,
                "text_clean": clean,
                "language": detect_lang(clean),
                "char_count": len(clean),
                "token_count": token_count_simple(clean),
                "page_metadata": {
                    "has_tables": False,
                    "has_images": False,
                    "has_footnotes": False
                },
                "assets": []
            }
        ]
    }
