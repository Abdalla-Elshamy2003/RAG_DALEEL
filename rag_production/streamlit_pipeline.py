from __future__ import annotations
import traceback
from typing import Any, Dict, List  # ← ADD

import streamlit as st
from rag_demo.core import RAGConfig, RAGEngine


st.set_page_config(
    page_title="RAG Daleel Assistant",
    page_icon="📚",
    layout="wide",
)

@st.cache_resource
def load_engine() -> RAGEngine:
    config = RAGConfig()
    config.validate()
    return RAGEngine(config)

def render_source_card(index: int, source: Dict[str, Any]) -> None:
    with st.expander(f"Source {index}: {source.get('source', 'Unknown')}", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Source Type:**", source.get("source_type"))
            st.write("**Doc ID:**", source.get("doc_id"))
            st.write("**Parent ID:**", source.get("parent_id"))
        with col2:
            st.write("**Fusion Score:**", source.get("fusion_score"))
            st.write("**Rerank Score:**", source.get("rerank_score"))

        matched_children = source.get("matched_children") or []
        if matched_children:
            st.markdown("### Best Matched Child Preview")
            st.write(matched_children[0].get("preview", ""))
            st.markdown("### Match Types")
            st.write(matched_children[0].get("match_types", []))

        metadata = source.get("metadata") or {}
        if metadata:
            st.markdown("### Metadata")
            st.json(metadata)

def render_sources(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        st.warning("No sources returned.")
        return
    for i, source in enumerate(sources, start=1):
        render_source_card(i, source)

def main() -> None:
    st.title("📚 RAG Daleel Assistant")
    st.caption("Hybrid Search + Web Fallback + Reranking → Final Answer")

    with st.sidebar:
        st.header("Settings")

        retrieval_only = st.checkbox(
            "Retrieval only",
            value=False,  # ← default off so answers generate
            help="Skip LLM answer generation (useful if Ollama is not running).",
        )
        force_web = st.checkbox(
            "Force web search",
            value=False,
            help="Force web fallback if Tavily API key exists.",
        )
        show_debug = st.checkbox(
            "Show debug info",
            value=False,
        )

    query = st.text_area(
        "Ask a question",
        value="What is this document about?",
        height=120,
        placeholder="Type your question here...",
    )

    run_button = st.button("Ask", type="primary")

    if not run_button:
        st.info("Enter a question and click **Ask**.")
        return

    query = query.strip()
    if not query:
        st.warning("Please enter a question.")
        return

    try:
        engine = load_engine()

        if retrieval_only:
            with st.spinner("Retrieving contexts..."):
                contexts, decision, used_web, debug_info = engine.retrieve_contexts(
                    query=query,
                    use_web=True if force_web else None,
                )

            st.success(f"Retrieved {len(contexts)} context(s). Confidence: {decision.confidence}")
            st.info("Answer generation is disabled (retrieval-only mode).")

            if show_debug:
                with st.expander("Debug info"):
                    st.json(debug_info)

            render_sources([ctx.to_public_dict() for ctx in contexts])
            return

        # Full RAG pipeline with streaming
        with st.spinner("Retrieving and preparing context..."):
            result = engine.answer_question(
                query=query,
                use_web=True if force_web else None,
                debug=show_debug,
                stream=True,  # ← streaming enabled
            )

        st.markdown("## Answer")

        # st.write_stream natively handles any Iterator[str]
        # It accumulates tokens and renders them progressively
        answer_text = st.write_stream(result["answer"])

        st.markdown("## Sources")
        render_sources(result.get("sources", []))

        if show_debug and result.get("debug"):
            with st.expander("Debug info"):
                st.json(result["debug"])

    except Exception as exc:
        st.error("An error occurred while running the RAG system.")
        st.code(str(exc))
        with st.expander("Full traceback"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()