"""
Docs Ingestion Pipeline — Streamlit UI
Professional, card-based design. No sidebar.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import psycopg
import streamlit as st

try:
    from ingest_app.config import AppConfig
    from ingest_app.db import (
        MAIN_TABLE, POST_TABLE,
        create_tables, get_existing_hashes, get_recent_records, get_stats,
        insert_payload, insert_post_processing_payload,
        sync_post_processing_from_main,
    )
    from ingest_app.file_utils import compute_sha256
    from ingest_app.payload_builders import build_docx_payload, build_pdf_payload, build_txt_payload
except ModuleNotFoundError:
    from config import AppConfig
    from db import (
        MAIN_TABLE, POST_TABLE,
        create_tables, get_existing_hashes, get_recent_records, get_stats,
        insert_payload, insert_post_processing_payload,
        sync_post_processing_from_main,
    )
    from file_utils import compute_sha256
    from payload_builders import build_docx_payload, build_pdf_payload, build_txt_payload


# ── DB helpers ────────────────────────────────────────────────────────────────
def get_all_filenames(conn, table="preprocessing_data"):
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT doc_id, file_name, source_type, language, page_count
            FROM {table} ORDER BY created_at DESC
        """)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_document_pages(conn, doc_id, table="preprocessing_data"):
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT file_name, source_type, language, page_count, payload
            FROM {table} WHERE doc_id = %s
        """, (doc_id,))
        row = cur.fetchone()
    if not row:
        return {}
    payload = row[4] or {}
    # Check if new structure with legacy_payload
    if "legacy_payload" in payload:
        pages = payload["legacy_payload"].get("pages", [])
    else:
        pages = payload.get("pages", [])
    return {
        "file_name":   row[0],
        "source_type": row[1],
        "language":    row[2],
        "page_count":  row[3],
        "pages":       pages,
        "payload":     payload,
    }


def delete_document(conn, doc_id):
    """Delete from both tables by doc_id."""
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {MAIN_TABLE} WHERE doc_id = %s", (doc_id,))
        cur.execute(f"DELETE FROM {POST_TABLE} WHERE doc_id = %s", (doc_id,))
    conn.commit()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Docs Ingestion Pipeline",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
[data-testid="collapsedControl"],
section[data-testid="stSidebar"] { display: none !important; }

.card {
    background: #151521; border-radius: 14px; padding: 22px 18px 16px;
    text-align: center; border: 1px solid rgba(255,255,255,0.07);
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
    transition: transform .18s, box-shadow .18s; height: 100%;
}
.card:hover { transform: translateY(-4px); box-shadow: 0 8px 32px rgba(0,0,0,0.5); }
.card .c-icon  { font-size: 1.6rem; margin-bottom: 6px; }
.card .c-label { font-size: 0.72rem; font-weight: 700; letter-spacing:.07em;
                 text-transform: uppercase; opacity: .55; margin-bottom: 4px; }
.card .c-value { font-size: 2.4rem; font-weight: 800; line-height: 1; }
.card .c-sub   { font-size: 0.7rem; opacity: .4; margin-top: 4px; }
.card-blue   { border-top: 3px solid #3B82F6; } .card-blue   .c-value { color: #3B82F6; }
.card-emerald{ border-top: 3px solid #10B981; } .card-emerald .c-value{ color: #10B981; }
.card-violet { border-top: 3px solid #8B5CF6; } .card-violet .c-value { color: #8B5CF6; }
.card-amber  { border-top: 3px solid #F59E0B; } .card-amber  .c-value { color: #F59E0B; }
.card-rose   { border-top: 3px solid #F43F5E; } .card-rose   .c-value { color: #F43F5E; }

.s-header {
    font-size: 1rem; font-weight: 700; color: #E2E8F0;
    border-left: 4px solid #3B82F6; padding-left: 10px; margin: 28px 0 14px;
}
.drop-zone {
    text-align: center; padding: 56px 20px;
    border: 2px dashed rgba(255,255,255,0.09);
    border-radius: 16px; color: #475569;
}
.fbadge {
    display: inline-flex; align-items: center; gap: 8px;
    background: #1E1E2E; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px; padding: 7px 13px; margin: 3px;
    font-size: 0.83rem; color: #CBD5E1;
}
.r-ok   { background:#022C2022; border-left:4px solid #10B981; border-radius:8px; padding:11px 16px; margin:5px 0; }
.r-skip { background:#2D1B0022; border-left:4px solid #F59E0B; border-radius:8px; padding:11px 16px; margin:5px 0; }
.r-fail { background:#2D000022; border-left:4px solid #F43F5E; border-radius:8px; padding:11px 16px; margin:5px 0; }
.r-ok .rt { color:#10B981; font-weight:700; }
.r-skip .rt { color:#F59E0B; font-weight:700; }
.r-fail .rt { color:#F43F5E; font-weight:700; }
.rd { font-size:0.77rem; color:#64748B; margin-top:3px; }
.hr { border:none; border-top:1px solid rgba(255,255,255,0.06); margin:26px 0; }
.info-box {
    background:#0F172A; border:1px solid rgba(255,255,255,0.07);
    border-radius:10px; padding:16px 20px; margin-bottom:18px;
    font-size:0.88rem; color:#94A3B8; line-height:1.6;
}
.text-raw {
    background:#0A0A14; border-radius:8px; padding:14px 16px;
    font-size:0.82rem; color:#94A3B8; white-space:pre-wrap; line-height:1.7;
    border:1px solid rgba(255,255,255,0.05); max-height:320px; overflow-y:auto;
}
.text-clean {
    background:#021A0A; border-radius:8px; padding:14px 16px;
    font-size:0.85rem; color:#A7F3D0; white-space:pre-wrap; line-height:1.8;
    border:1px solid rgba(16,185,129,0.18); max-height:320px; overflow-y:auto;
    direction: auto;
}
.doc-info-bar {
    background:#151521; border:1px solid rgba(255,255,255,0.07);
    border-radius:10px; padding:14px 18px; margin-bottom:18px;
    display:flex; flex-wrap:wrap; gap:20px; align-items:center;
}
.dib-item { font-size:0.8rem; color:#64748B; }
.dib-item strong { color:#CBD5E1; }
.pg-meta { font-size:0.73rem; color:#475569; margin-top:6px; }
.pg-meta span { margin-right:14px; }
.delete-card {
    background:#1A0A0A; border:1px solid rgba(244,63,94,0.2);
    border-radius:12px; padding:20px; margin-bottom:10px;
    display:flex; align-items:center; justify-content:space-between;
    flex-wrap:wrap; gap:12px;
}
.delete-card .dc-name { font-weight:600; color:#F1F5F9; font-size:0.92rem; }
.delete-card .dc-meta { font-size:0.75rem; color:#64748B; margin-top:3px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DB INIT
# ══════════════════════════════════════════════════════════════════════════════
cfg = AppConfig()
DB_OK = False
DB_ERR = ""
try:
    with psycopg.connect(cfg.db_conn) as _c:
        create_tables(_c)
    DB_OK = True
except Exception as _e:
    DB_ERR = str(_e)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="padding:24px 0 6px">
  <h1 style="margin:0;font-size:1.9rem;font-weight:800;color:#F1F5F9">
    📄 Docs Ingestion Pipeline
  </h1>
  <p style="color:#475569;margin:6px 0 0;font-size:0.93rem">
    رفع ومعالجة الملفات (PDF / DOCX / TXT) وحفظها في PostgreSQL
  </p>
</div>
""", unsafe_allow_html=True)

if not DB_OK:
    st.error(f"⚠️ تعذّر الاتصال بقاعدة البيانات:\n```\n{DB_ERR}\n```")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# STATS CARDS
# ══════════════════════════════════════════════════════════════════════════════
def render_stats(key_suffix=""):
    try:
        with psycopg.connect(cfg.db_conn) as conn:
            s = get_stats(conn)
    except Exception as e:
        st.warning(f"تعذّر جلب الإحصائيات: {e}")
        return
    pre  = s.get(f"{MAIN_TABLE}_total", 0)
    post = s.get(f"{POST_TABLE}_total", 0)
    by_t = s.get("by_type", {})
    specs = [
        ("card-blue",    "🗃️", "preprocessing_data",  pre,                  "الجدول الرئيسي"),
        ("card-emerald", "✅", "post_processing_data", post,                 "بعد المعالجة"),
        ("card-violet",  "📕", "PDF",                  by_t.get("pdf",  0),  "ملفات"),
        ("card-amber",   "📘", "DOCX",                 by_t.get("docx", 0),  "ملفات"),
        ("card-rose",    "📄", "TXT",                  by_t.get("txt",  0),  "ملفات"),
    ]
    cols = st.columns(5)
    for col, (cls, icon, label, val, sub) in zip(cols, specs):
        with col:
            st.markdown(f"""
            <div class="card {cls}">
              <div class="c-icon">{icon}</div>
              <div class="c-label">{label}</div>
              <div class="c-value">{val}</div>
              <div class="c-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

render_stats("top")
st.markdown('<hr class="hr">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_upload, tab_sync, tab_main, tab_post, tab_viewer, tab_reprocess, tab_delete = st.tabs([
    "📤  رفع ملفات",
    "🔄  مزامنة الجداول",
    "📋  preprocessing_data",
    "✅  post_processing_data",
    "🔍  عرض المستند",
    "🔄  إعادة معالجة جميع الملفات",
    "🗑️  حذف مستند",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — UPLOAD
# ──────────────────────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown('<div class="s-header">رفع وتحليل الملفات</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "upload", type=["pdf", "docx", "txt"],
        accept_multiple_files=True, label_visibility="collapsed",
    )

    if uploaded_files:
        icons = {".pdf": "📕", ".docx": "📘", ".txt": "📄"}
        badges = "".join(
            f'<span class="fbadge">{icons.get(Path(f.name).suffix.lower(),"📁")} '
            f'{f.name} <span style="opacity:.45">({round(f.size/1024,1)} KB)</span></span>'
            for f in uploaded_files
        )
        st.markdown(badges, unsafe_allow_html=True)
        st.markdown("")

        if st.button("🚀 ابدأ المعالجة", type="primary"):
            results = []
            prog   = st.progress(0)
            status = st.empty()
            total  = len(uploaded_files)

            try:
                with psycopg.connect(cfg.db_conn) as conn:
                    create_tables(conn)
                    sync_post_processing_from_main(conn)

                    for i, uf in enumerate(uploaded_files):
                        status.markdown(f"⏳ **معالجة** `{uf.name}` ({i+1}/{total})")
                        prog.progress(i / total)

                        suffix = Path(uf.name).suffix
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uf.getvalue())
                            tmp_path = Path(tmp.name)

                        try:
                            file_hash = compute_sha256(tmp_path)

                            if file_hash in get_existing_hashes(conn, [file_hash]):
                                results.append(("skip", uf.name, "موجود بالفعل في الجدول الرئيسي"))
                                continue

                            ext = tmp_path.suffix.lower()
                            if   ext == ".pdf":  payload = build_pdf_payload(tmp_path, file_hash)
                            elif ext == ".docx": payload = build_docx_payload(tmp_path, file_hash)
                            elif ext == ".txt":  payload = build_txt_payload(tmp_path, file_hash)
                            else:
                                raise ValueError(f"نوع غير مدعوم: {ext}")

                            # ── ضمان وجود الحقول المطلوبة ──
                            if not payload.get("doc_id"):
                                payload["doc_id"] = f"doc_{file_hash[:12]}"
                            if not payload.get("page_count"):
                                payload["page_count"] = len(payload.get("pages", []))
                            if not payload.get("extraction_status"):
                                payload["extraction_status"] = "completed"
                            payload["file_name"] = uf.name
                            payload["file_path"] = f"uploaded/{uf.name}"

                            insert_payload(conn, payload)
                            insert_post_processing_payload(conn, payload)
                            conn.commit()

                            pages = payload.get("page_count", 0)
                            lang  = payload.get("language", "unknown")
                            results.append(("ok", uf.name, f"{pages} صفحة | لغة: {lang} | hash: {file_hash[:10]}…"))

                        except Exception as e:
                            conn.rollback()
                            results.append(("fail", uf.name, str(e)))
                        finally:
                            try: os.unlink(tmp_path)
                            except: pass

                    prog.progress(1.0)
                    status.empty()

            except Exception as e:
                st.error(f"خطأ عام: {e}")

            st.markdown('<div class="s-header">نتائج المعالجة</div>', unsafe_allow_html=True)
            for kind, name, detail in results:
                cls, icon = {"ok": ("r-ok","✅"), "skip": ("r-skip","⚠️"), "fail": ("r-fail","❌")}[kind]
                st.markdown(f'<div class="{cls}"><span class="rt">{icon} {name}</span><div class="rd">{detail}</div></div>',
                            unsafe_allow_html=True)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("✅ تم الحفظ",   sum(1 for r in results if r[0]=="ok"))
            mc2.metric("⚠️ تم التخطي", sum(1 for r in results if r[0]=="skip"))
            mc3.metric("❌ فشل",        sum(1 for r in results if r[0]=="fail"))
            st.markdown('<hr class="hr">', unsafe_allow_html=True)
            render_stats("after_upload")
    else:
        st.markdown("""
        <div class="drop-zone">
          <div style="font-size:3rem">📂</div>
          <div style="font-size:1.05rem;margin-top:10px;color:#64748B">اسحب ملفات PDF أو DOCX أو TXT هنا</div>
          <div style="font-size:0.78rem;margin-top:6px;opacity:.5">يدعم رفع أكثر من ملف في نفس الوقت</div>
        </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — SYNC
# ──────────────────────────────────────────────────────────────────────────────
with tab_sync:
    st.markdown('<div class="s-header">مزامنة الجداول</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
      هذه الأداة تنسخ أو تحدث كل الملفات الموجودة في <code>preprocessing_data</code>
      إلى <code>post_processing_data</code> تلقائياً.<br>
      الملفات الموجودة بالفعل في الجدول الثاني ستُحدث.
    </div>""", unsafe_allow_html=True)

    if st.button("🔄  مزامنة الآن", type="primary"):
        with st.spinner("جاري المزامنة..."):
            try:
                with psycopg.connect(cfg.db_conn) as conn:
                    create_tables(conn)
                    copied = sync_post_processing_from_main(conn)
                st.success(f"✅ تمت المزامنة — تم نسخ/تحديث **{copied}** سجل في `post_processing_data`")
                render_stats("after_sync")
            except Exception as e:
                st.error(f"❌ فشلت المزامنة: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — preprocessing_data
# ──────────────────────────────────────────────────────────────────────────────
with tab_main:
    st.markdown('<div class="s-header">سجلات preprocessing_data</div>', unsafe_allow_html=True)
    limit_m = st.slider("عدد السجلات", 5, 200, 40, key="lim_main")
    try:
        with psycopg.connect(cfg.db_conn) as conn:
            rows = get_recent_records(conn, table=MAIN_TABLE, limit=limit_m)
        if rows:
            df = pd.DataFrame(rows)
            df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(df[["id","doc_id","file_name","source_type","language","page_count","created_at"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "id": st.column_config.NumberColumn("ID", width="small"),
                    "doc_id": st.column_config.TextColumn("Doc ID", width="medium"),
                    "file_name": st.column_config.TextColumn("اسم الملف", width="large"),
                    "source_type": st.column_config.TextColumn("النوع", width="small"),
                    "language": st.column_config.TextColumn("اللغة", width="small"),
                    "page_count": st.column_config.NumberColumn("صفحات", width="small"),
                    "created_at": st.column_config.TextColumn("التاريخ", width="medium"),
                })
        else:
            st.info("لا توجد سجلات بعد.")
    except Exception as e:
        st.error(f"خطأ: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — post_processing_data
# ──────────────────────────────────────────────────────────────────────────────
with tab_post:
    st.markdown('<div class="s-header">سجلات post_processing_data</div>', unsafe_allow_html=True)
    limit_p = st.slider("عدد السجلات", 5, 200, 40, key="lim_post")
    try:
        with psycopg.connect(cfg.db_conn) as conn:
            rows = get_recent_records(conn, table=POST_TABLE, limit=limit_p)
        if rows:
            df = pd.DataFrame(rows)
            df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(df[["id","doc_id","file_name","source_type","language","page_count","created_at"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "id": st.column_config.NumberColumn("ID", width="small"),
                    "doc_id": st.column_config.TextColumn("Doc ID", width="medium"),
                    "file_name": st.column_config.TextColumn("اسم الملف", width="large"),
                    "source_type": st.column_config.TextColumn("النوع", width="small"),
                    "language": st.column_config.TextColumn("اللغة", width="small"),
                    "page_count": st.column_config.NumberColumn("صفحات", width="small"),
                    "created_at": st.column_config.TextColumn("التاريخ", width="medium"),
                })
        else:
            st.info("لا توجد سجلات بعد — اضغط على 'مزامنة' أولاً.")
    except Exception as e:
        st.error(f"خطأ: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — DOCUMENT VIEWER
# ──────────────────────────────────────────────────────────────────────────────
with tab_viewer:
    st.markdown('<div class="s-header">عرض النص قبل وبعد المعالجة</div>', unsafe_allow_html=True)

    viewer_table = st.radio("اختر الجدول", [MAIN_TABLE, POST_TABLE],
                            horizontal=True, key="viewer_table")

    try:
        with psycopg.connect(cfg.db_conn) as conn:
            all_docs = get_all_filenames(conn, viewer_table)
    except Exception as e:
        st.error(f"خطأ: {e}")
        all_docs = []

    if not all_docs:
        st.info("لا توجد ملفات في هذا الجدول بعد.")
    else:
        type_icons = {"pdf": "📕", "docx": "📘", "txt": "📄"}

        # ── Dropdown اختيار الملف ──
        col_sel, col_info = st.columns([3, 2])
        with col_sel:
            options = {
                f"{type_icons.get(d['source_type'],'📁')}  {d['file_name']}  "
                f"[{d['source_type']} | {d['language']} | {d['page_count']} صفحة]": d["doc_id"]
                for d in all_docs
            }
            chosen_label = st.selectbox("📂 اختر الملف", list(options.keys()), key="viewer_select")
            chosen_doc_id = options[chosen_label]

        with psycopg.connect(cfg.db_conn) as conn:
            doc = get_document_pages(conn, chosen_doc_id, viewer_table)

        if doc:
            pages = doc.get("pages", [])
            t_icon = type_icons.get(doc.get("source_type",""), "📁")

            # Info bar
            st.markdown(f"""
            <div class="doc-info-bar">
              <div class="dib-item">{t_icon} <strong>{doc['file_name']}</strong></div>
              <div class="dib-item">النوع: <strong>{doc['source_type']}</strong></div>
              <div class="dib-item">اللغة: <strong>{doc['language']}</strong></div>
              <div class="dib-item">الصفحات: <strong>{doc['page_count']}</strong></div>
              <div class="dib-item">Doc ID: <strong style="font-size:0.7rem;opacity:.6">{chosen_doc_id}</strong></div>
            </div>""", unsafe_allow_html=True)

            # View mode
            view_mode = st.radio(
                "وضع العرض",
                ["✨ النص النظيف فقط", "↕️ نص خام + نص نظيف", "📄 صفحة واحدة", "🔧 المخرجات الجديدة"],
                horizontal=True, key="view_mode",
            )

            if view_mode == "🔧 المخرجات الجديدة":
                # Show new outputs
                full_payload = doc.get("payload", {})
                structured_json = full_payload.get("structured_json", {})
                raw_cleaned = structured_json.get("text_raw", "")
                markdown_text = structured_json.get("markdown_text", "")

                st.markdown("### 1) Text Raw")
                st.text_area("النص الخام", raw_cleaned, height=200, key="raw_cleaned")

                st.markdown("### 2) Structured JSON")
                st.json(structured_json)
                
                # Show links separately if present
                links = structured_json.get("link", [])
                if links:
                    st.markdown("### روابط مستخرجة")
                    for link in links:
                        st.markdown(f"- {link}")

                st.markdown("### 3) Markdown Text")
                st.markdown(markdown_text)

            elif view_mode == "📄 صفحة واحدة":
                page_nos = [p.get("page_no", i+1) for i, p in enumerate(pages)]
                sel_page = st.selectbox("رقم الصفحة", page_nos, key="sel_page")
                pages_to_show = [p for p in pages if p.get("page_no", 0) == sel_page]
            else:
                max_pages = st.slider("عدد الصفحات للعرض", 1, max(len(pages),1),
                                      min(5, len(pages)), key="max_pg")
                pages_to_show = pages[:max_pages]

            st.markdown("")

            for page in pages_to_show:
                pg_no  = page.get("page_no", "?")
                chars  = page.get("char_count", 0)
                tokens = page.get("token_count", 0)
                lang   = page.get("language", "")
                method = page.get("extraction_method", "")
                has_img = page.get("page_metadata", {}).get("has_images", False)
                has_tbl = page.get("page_metadata", {}).get("has_tables", False)
                raw_text      = page.get("text_raw", "") or ""
                clean_text_val = page.get("text_clean", raw_text) or raw_text

                meta_parts = []
                if lang:    meta_parts.append(f"🌐 {lang}")
                if chars:   meta_parts.append(f"📝 {chars} حرف")
                if tokens:  meta_parts.append(f"🔤 {tokens} كلمة")
                if method:  meta_parts.append(f"⚙️ {method}")
                if has_img: meta_parts.append("🖼️ صور")
                if has_tbl: meta_parts.append("📊 جداول")
                meta_html = "".join(f"<span>{m}</span>" for m in meta_parts)

                with st.expander(f"📄 صفحة {pg_no}  —  {chars} حرف  |  {tokens} كلمة", expanded=True):
                    st.markdown(f'<div class="pg-meta">{meta_html}</div>', unsafe_allow_html=True)
                    st.markdown("")

                    if view_mode == "↕️ نص خام + نص نظيف":
                        col_r, col_c = st.columns(2)
                        with col_r:
                            st.markdown("**📄 النص الخام**")
                            # escape HTML special chars for safe display
                            safe_raw = raw_text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                            st.markdown(f'<div class="text-raw">{safe_raw or "—"}</div>', unsafe_allow_html=True)
                        with col_c:
                            st.markdown("**✨ النص النظيف**")
                            safe_clean = clean_text_val.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                            st.markdown(f'<div class="text-clean">{safe_clean or "—"}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown("**✨ النص النظيف**")
                        safe_clean = clean_text_val.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                        st.markdown(f'<div class="text-clean">{safe_clean or "—"}</div>', unsafe_allow_html=True)

                    assets = page.get("assets", [])
                    if assets:
                        st.markdown(f'<div class="pg-meta">🖼️ {len(assets)} صورة مضمّنة</div>',
                                    unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# TAB 6 — REPROCESS ALL
# ──────────────────────────────────────────────────────────────────────────────
with tab_reprocess:
    st.markdown('<div class="s-header">إعادة معالجة جميع الملفات</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
      حدد مجلد يحتوي على الملفات الأصلية (PDF، DOCX، TXT).<br>
      سيتم البحث عن الملفات الموجودة في <code>preprocessing_data</code> وإعادة معالجتها إذا وُجدت في المجلد المحدد.<br>
      هذا مفيد لتحديث السجلات عند تعديل منطق الـ preprocessing.
    </div>""", unsafe_allow_html=True)

    reprocess_folder = st.text_input("مسار المجلد لإعادة المعالجة (مثال: C:\\Users\\YourName\\Documents)", key="reprocess_folder")
    
    if reprocess_folder and Path(reprocess_folder).exists() and Path(reprocess_folder).is_dir():
        st.success(f"✅ المجلد موجود: {reprocess_folder}")
        
        if st.button("🔄 ابدأ إعادة المعالجة من المجلد", type="primary"):
            results = []
            prog = st.progress(0)
            status = st.empty()
            
            try:
                with psycopg.connect(cfg.db_conn) as conn:
                    # Get all records from preprocessing_data
                    with conn.cursor() as cur:
                        cur.execute(f"SELECT doc_id, file_name, file_path, file_hash, source_type FROM {MAIN_TABLE}")
                        records = cur.fetchall()
                    
                    total = len(records)
                    
                    for i, (doc_id, file_name, file_path, file_hash, source_type) in enumerate(records):
                        status.markdown(f"⏳ إعادة معالجة `{file_name}` ({i+1}/{total})")
                        prog.progress(i / total)
                        
                        try:
                            # Try to find the file in the specified folder
                            file_in_folder = Path(reprocess_folder) / file_name
                            if not file_in_folder.exists():
                                results.append(("skip", file_name, "الملف غير موجود في المجلد المحدد"))
                                continue
                            
                            # Rebuild payload using the file from the folder
                            ext = file_in_folder.suffix.lower()
                            if ext == ".pdf":
                                payload = build_pdf_payload(file_in_folder, file_hash)
                            elif ext == ".docx":
                                payload = build_docx_payload(file_in_folder, file_hash)
                            elif ext == ".txt":
                                payload = build_txt_payload(file_in_folder, file_hash)
                            else:
                                results.append(("skip", file_name, f"نوع غير مدعوم: {ext}"))
                                continue
                            
                            # Update payload
                            insert_payload(conn, payload)
                            insert_post_processing_payload(conn, payload)
                            conn.commit()
                            
                            pages = payload.get("page_count", 0)
                            lang = payload.get("language", "unknown")
                            results.append(("update", file_name, f"تم إعادة المعالجة | {pages} صفحة | لغة: {lang}"))
                            
                        except Exception as e:
                            conn.rollback()
                            results.append(("fail", file_name, str(e)))
                
                    prog.progress(1.0)
                    status.empty()
                    
            except Exception as e:
                st.error(f"خطأ عام: {e}")
        
        st.markdown('<div class="s-header">نتائج إعادة المعالجة</div>', unsafe_allow_html=True)
        for kind, name, detail in results:
            cls, icon = {"update": ("r-ok","✅"), "skip": ("r-skip","⚠️"), "fail": ("r-fail","❌")}[kind]
            st.markdown(f'<div class="{cls}"><span class="rt">{icon} {name}</span><div class="rd">{detail}</div></div>',
                        unsafe_allow_html=True)
        
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("✅ تم التحديث", sum(1 for r in results if r[0]=="update"))
        mc2.metric("⚠️ تم التخطي", sum(1 for r in results if r[0]=="skip"))
        mc3.metric("❌ فشل", sum(1 for r in results if r[0]=="fail"))
        
        # Sync to post_processing_data (open fresh connection — previous one is closed)
        try:
            with psycopg.connect(cfg.db_conn) as sync_conn:
                sync_count = sync_post_processing_from_main(sync_conn)
            st.info(f"🔄 تم مزامنة {sync_count} سجل إلى post_processing_data")
        except Exception as e:
            st.warning(f"تحذير في المزامنة: {e}")
        
        st.markdown('<hr class="hr">', unsafe_allow_html=True)
        render_stats("after_reprocess")
    else:
        if reprocess_folder:
            st.error("❌ المجلد غير موجود أو غير صحيح.")
        st.info("أدخل مسار مجلد صحيح يحتوي على الملفات لإعادة المعالجة.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 7 — DELETE
# ──────────────────────────────────────────────────────────────────────────────
with tab_delete:
    st.markdown('<div class="s-header">حذف مستند من قاعدة البيانات</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box" style="border-color:rgba(244,63,94,0.3);">
      ⚠️ الحذف نهائي — سيتم حذف المستند من <strong>كلا الجدولين</strong>
      (<code>preprocessing_data</code> و <code>post_processing_data</code>) في نفس الوقت.
    </div>""", unsafe_allow_html=True)

    try:
        with psycopg.connect(cfg.db_conn) as conn:
            del_docs = get_all_filenames(conn, MAIN_TABLE)
    except Exception as e:
        st.error(f"خطأ: {e}")
        del_docs = []

    if not del_docs:
        st.info("لا توجد ملفات في قاعدة البيانات.")
    else:
        type_icons = {"pdf": "📕", "docx": "📘", "txt": "📄"}

        del_options = {
            f"{type_icons.get(d['source_type'],'📁')}  {d['file_name']}  "
            f"[{d['source_type']} | {d['language']} | {d['page_count']} صفحة]": d
            for d in del_docs
        }

        chosen_del_label = st.selectbox(
            "🗂️ اختر الملف للحذف",
            list(del_options.keys()),
            key="del_select",
        )
        chosen_del = del_options[chosen_del_label]

        # Preview card
        t_icon = type_icons.get(chosen_del["source_type"], "📁")
        st.markdown(f"""
        <div class="delete-card">
          <div>
            <div class="dc-name">{t_icon} {chosen_del['file_name']}</div>
            <div class="dc-meta">
              النوع: {chosen_del['source_type']} &nbsp;|&nbsp;
              اللغة: {chosen_del['language']} &nbsp;|&nbsp;
              الصفحات: {chosen_del['page_count']} &nbsp;|&nbsp;
              Doc ID: {chosen_del['doc_id']}
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Confirm checkbox + delete button
        confirm = st.checkbox("✅ أنا متأكد إني عاوز أحذف الملف ده نهائياً", key="del_confirm")

        if st.button("🗑️ احذف الآن", type="primary", disabled=not confirm):
            try:
                with psycopg.connect(cfg.db_conn) as conn:
                    delete_document(conn, chosen_del["doc_id"])
                st.success(f"✅ تم حذف **{chosen_del['file_name']}** من كلا الجدولين بنجاح.")
                st.rerun()
            except Exception as e:
                st.error(f"❌ فشل الحذف: {e}")