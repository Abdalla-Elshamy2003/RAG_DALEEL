import traceback
import time
import streamlit as st
from rag_demo.core import RAGConfig, RAGEngine

# Set up the page configuration
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

        # User chooses whether to enable retrieval-only mode
        retrieval_only = st.checkbox(
            "Retrieval only",
            value=False,  # Set to False to enable model answer generation
            help="Use this while Ollama/Qwen is still downloading. It will not generate the final LLM answer.",
        )

        force_web = st.checkbox(
            "Force web search",
            value=False,
            help="Force web fallback/search if Tavily API key exists.",
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
        with st.spinner("Processing..."):
            engine = load_engine()

        # If retrieval-only is enabled, perform only retrieval and reranking
        if retrieval_only:
            # Debug logging: Check if retrieval-only is set
            st.write("Retrieval only mode is ENABLED.")  # <-- Logging for debugging

            # Retrieval + Reranking, without full answer generation
            contexts, decision, used_web, _ = engine.retrieve_contexts(
                query=query,
                use_web=True if force_web else None,
            )

            st.success("Retrieved and reranked contexts.")

            # Display retrieved contexts (parent and child content)
            render_retrieved_contexts(contexts)

            # Display just the final answer
            st.markdown("## Answer")
            st.write("Answer generation is not enabled in this mode.")
            return

        # If retrieval-only is disabled, proceed with the full pipeline
        st.write("Answer generation mode ENABLED.")  # <-- Logging for debugging

        # Full RAG pipeline (retrieval + reranking + final answer generation)
        result = engine.answer_question(
            query=query,
            use_web=True if force_web else None,
            debug=show_debug,
        )

        st.success("Answer generated.")

        # Display final answer only
        st.markdown("## Answer")
        st.write(result.get("answer", ""))

    except Exception as exc:
        st.error("An error occurred while running the RAG system.")
        st.code(str(exc))
        with st.expander("Full traceback"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()