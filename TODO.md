# TODO: JSON Structure Updates for RAG Pipeline

## Status: [0/4] ⏳ In Progress

### 1. [ ] Edit `ingest_app/payload_builders.py`
   - Rename `"link":` → `"links":` in:
     * `build_pdf_payload()`
     * `build_docx_payload()`  
     * `build_txt_payload()`

### 2. [ ] Edit `ingest_app/text_utils.py`
   - In `build_structured_json()`: `"link": links,` → `"links": links,`

### 3. [ ] Test ingestion
   - Upload sample PDF from `Data 2/`
   - Verify in Streamlit "المخرجات الجديدة":
     * `text_raw`: All continuous text
     * `markdown_text`: Markdown version
     * `links`: Array of extracted URLs

### 4. [ ] Run project real-time
   - `docker-compose up`
   - Verify changes appear in DB/Streamlit immediately via watcher_service
