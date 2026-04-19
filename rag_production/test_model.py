from rag_demo.core import ChildEvidence, GPUModelManager, RAGConfig, RetrievedContext


def main() -> None:
    config = RAGConfig()
    config.validate()

    models = GPUModelManager(config)

    query = "What is this document about?"

    print("Encoding query...")
    vector = models.encode_query(query)

    print("Embedding length:", len(vector))
    print("First 5 values:", vector[:5])

    fake_contexts = [
        RetrievedContext(
            source_type="internal",
            doc_id="doc_1",
            parent_id="parent_1",
            source="test_source_1",
            text="This document explains artificial intelligence, retrieval systems, and document search.",
            matched_children=[
                ChildEvidence(
                    child_id="child_1",
                    child_text="Artificial intelligence and retrieval systems are discussed in this document.",
                    match_types=["test"],
                )
            ],
        ),
        RetrievedContext(
            source_type="internal",
            doc_id="doc_2",
            parent_id="parent_2",
            source="test_source_2",
            text="This text is about cooking recipes and food preparation.",
            matched_children=[
                ChildEvidence(
                    child_id="child_2",
                    child_text="Cooking recipes and food preparation are discussed here.",
                    match_types=["test"],
                )
            ],
        ),
    ]

    print("Reranking fake contexts...")
    ranked = models.rerank(
        query=query,
        contexts=fake_contexts,
        top_k=2,
    )

    for item in ranked:
        print(item.source, item.rerank_score)


if __name__ == "__main__":
    main()