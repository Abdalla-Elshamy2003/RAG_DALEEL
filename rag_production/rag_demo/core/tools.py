from __future__ import annotations

import logging
from typing import List

import requests

from .config import RAGConfig
from .schemas import ChildEvidence, RetrievedContext

log = logging.getLogger(__name__)


class WebSearchTool:
    """
    Optional fallback web search.

    If TAVILY_API_KEY is missing, it safely returns [].
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.api_key = config.tavily_api_key
        self.base_url = "https://api.tavily.com/search"
        self.session = requests.Session()

    def search(self, query: str) -> List[RetrievedContext]:
        query = (query or "").strip()

        if not query:
            return []

        if not self.api_key:
            log.warning("TAVILY_API_KEY is missing. Web fallback skipped.")
            return []

        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": self.config.tavily_max_results,
            "include_answer": False,
            "include_raw_content": False,
        }

        try:
            response = self.session.post(
                self.base_url,
                json=payload,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()

        except Exception:
            log.exception("Web search failed.")
            return []

        contexts: List[RetrievedContext] = []

        for index, item in enumerate(data.get("results", []), start=1):
            url = item.get("url") or f"web_result_{index}"
            title = item.get("title") or url
            content = item.get("content") or ""

            if not content.strip():
                continue

            contexts.append(
                RetrievedContext(
                    source_type="web",
                    doc_id=url,
                    parent_id=url,
                    text=content,
                    source=f"{title} - {url}",
                    metadata={
                        "url": url,
                        "title": title,
                        "score": item.get("score"),
                    },
                    matched_children=[
                        ChildEvidence(
                            child_id=url,
                            child_text=content,
                            fusion_score=float(item.get("score") or 0.0),
                            match_types=["web"],
                            metadata={"url": url, "title": title},
                        )
                    ],
                    fusion_score=float(item.get("score") or 0.0),
                )
            )

        return contexts