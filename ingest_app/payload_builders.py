from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import fitz
from docx import Document

try:
    from ingest_app.file_utils import chunk_list
    from ingest_app.text_utils import detect_lang, token_count_simple
except ModuleNotFoundError:
    from file_utils import chunk_list
    from text_utils import detect_lang, token_count_simple

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

        return {
            "file_name": file_path.name,
            "file_path": str(file_path).replace("\\", "/"),
            "file_size_bytes": file_path.stat().st_size,
            "file_hash": file_hash,
            "language": detect_lang(combined_text),
            "source_type": "pdf",
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
        if table_text.strip():
            table_texts.append(table_text)

    media_count = extract_docx_media_count(file_path)

    logical_pages_raw = chunk_list(paragraphs, logical_page_paragraphs) if paragraphs else [[]]
    pages = []
    all_text = []

    for idx, para_group in enumerate(logical_pages_raw, start=1):
        text_raw = "\n".join(para_group).strip()
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

    return {
        "file_name": file_path.name,
        "file_path": str(file_path).replace("\\", "/"),
        "file_size_bytes": file_path.stat().st_size,
        "file_hash": file_hash,
        "language": detect_lang(combined_text),
        "source_type": "docx",
        "pages": pages
    }


def build_txt_payload(file_path: Path, file_hash: str) -> dict[str, Any]:
    raw = file_path.read_text(encoding="utf-8", errors="ignore")

    return {
        "file_name": file_path.name,
        "file_path": str(file_path).replace("\\", "/"),
        "file_size_bytes": file_path.stat().st_size,
        "file_hash": file_hash,
        "language": detect_lang(raw),
        "source_type": "txt",
        "pages": [
            {
                "text_raw": raw,
                "language": detect_lang(raw),
                "char_count": len(raw),
                "token_count": token_count_simple(raw),
                "page_metadata": {
                    "has_tables": False,
                    "has_images": False,
                    "has_footnotes": False
                },
                "assets": []
            }
        ]
    }
