from __future__ import annotations

import argparse
import logging

from rag_demo.core import RAGConfig, RAGEngine


def print_sources(sources: list[dict]) -> None:
    print("\nSOURCES")
    print("-------")

    if not sources:
        print("No sources returned.")
        return

    for i, source in enumerate(sources, start=1):
        print(f"\n[{i}]")
        print("source_type:", source.get("source_type"))
        print("source:", source.get("source"))
        print("doc_id:", source.get("doc_id"))
        print("parent_id:", source.get("parent_id"))
        print("fusion_score:", source.get("fusion_score"))
        print("rerank_score:", source.get("rerank_score"))

        children = source.get("matched_children") or []
        if children:
            print("best_child_preview:", children[0].get("preview"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Test production RAG pipeline")

    parser.add_argument(
        "--query",
        default="What is this document about?",
        help="Question to ask the RAG system.",
    )

    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only run retrieval + reranking. Do not call Ollama.",
    )

    parser.add_argument(
        "--use-web",
        action="store_true",
        help="Force web fallback/search.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    config = RAGConfig()
    config.validate()

    engine = RAGEngine(config)

    try:
        if args.retrieval_only:
            contexts, decision, used_web, debug_info = engine.retrieve_contexts(
                query=args.query,
                use_web=args.use_web,
            )

            print("\nRETRIEVAL ONLY RESULT")
            print("---------------------")
            print("confidence:", decision.confidence)
            print("reason:", decision.reason)
            print("used_web:", used_web)
            print("contexts:", len(contexts))

            sources = [context.to_public_dict() for context in contexts]
            print_sources(sources)

            if args.debug:
                print("\nDEBUG")
                print("-----")
                for key, value in debug_info.items():
                    print(f"{key}: {value}")

            return

        result = engine.answer_question(
            query=args.query,
            use_web=args.use_web,
            debug=args.debug,
        )

        print("\nANSWER")
        print("------")
        print(result["answer"])

        print_sources(result["sources"])

        print("\nMETA")
        print("----")
        print("used_web:", result["used_web"])
        print("confidence:", result["confidence"])

        if args.debug:
            print("\nDEBUG")
            print("-----")
            for key, value in result.get("debug", {}).items():
                print(f"{key}: {value}")

    finally:
        engine.close()


if __name__ == "__main__":
    main()