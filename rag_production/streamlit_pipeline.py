from __future__ import annotations

import traceback
from typing import Any, Dict, List

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
    st.caption("Hybrid Search on child_chunks → Parent Retrieval → Reranker → Local LLM via Ollama")

    with st.sidebar:
        st.header("Settings")

        retrieval_only = st.checkbox(
            "Retrieval only",
            value=True,
            help="Use this while Ollama/Qwen is still downloading. It will not generate the final LLM answer.",
        )

        force_web = st.checkbox(
            "Force web search",
            value=False,
            help="Force web fallback/search if Tavily API key exists.",
        )

        show_debug = st.checkbox(
            "Show debug info",
            value=True,
        )

        st.divider()

        st.markdown("### Test Questions")
        sample_question = st.selectbox(
            "Choose sample",
            [
                "",
                "What is this document about?",
                "Summarize the main topic of the documents.",
                "What are the most important points mentioned?",
            ],
        )

        st.divider()

        if st.button("Reload engine"):
            try:
                engine = load_engine()
                engine.close()
            except Exception:
                pass

            st.cache_resource.clear()
            st.success("Engine cache cleared. It will reload on next query.")

    query = st.text_area(
        "Ask a question",
        value=sample_question,
        height=120,
        placeholder="Type your question here...",
    )

    run_button = st.button("Run RAG", type="primary")

    if not run_button:
        st.info("Enter a question and click **Run RAG**.")
        return

    query = query.strip()

    if not query:
        st.warning("Please enter a question.")
        return

    try:
        with st.spinner("Loading RAG engine and models..."):
            engine = load_engine()

        if retrieval_only:
            with st.spinner("Running retrieval + reranking..."):
                contexts, decision, used_web, debug_info = engine.retrieve_contexts(
                    query=query,
                    use_web=True if force_web else None,
                )

            st.success("Retrieval completed.")

            st.subheader("Retrieval Result")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Confidence", decision.confidence)

            with col2:
                st.metric("Used Web", str(used_web))

            with col3:
                st.metric("Contexts", len(contexts))

            st.markdown("### Decision Reason")
            st.write(decision.reason)

            sources = [context.to_public_dict() for context in contexts]

            st.markdown("## Sources")
            render_sources(sources)

            if show_debug:
                st.markdown("## Debug Info")
                st.json(debug_info)

            return

        with st.spinner("Running full RAG pipeline..."):
            result = engine.answer_question(
                query=query,
                use_web=True if force_web else None,
                debug=show_debug,
            )

        st.success("Answer generated.")

        st.markdown("## Answer")
        st.write(result.get("answer", ""))

        st.markdown("## Sources")
        render_sources(result.get("sources", []))

        st.markdown("## Metadata")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Used Web", str(result.get("used_web")))

        with col2:
            st.metric("Confidence", result.get("confidence"))

        if show_debug:
            st.markdown("## Debug Info")
            st.json(result.get("debug", {}))

    except Exception as exc:
        st.error("An error occurred while running the RAG system.")
        st.code(str(exc))
        with st.expander("Full traceback"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()