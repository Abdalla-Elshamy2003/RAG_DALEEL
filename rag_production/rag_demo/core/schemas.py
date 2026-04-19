from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChildEvidence:
    child_id: str
    child_text: str
    fusion_score: float = 0.0
    match_types: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedContext:
    source_type: str
    doc_id: str
    parent_id: str
    text: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    matched_children: List[ChildEvidence] = field(default_factory=list)
    fusion_score: float = 0.0
    rerank_score: Optional[float] = None

    def best_child_text(self) -> str:
        if not self.matched_children:
            return ""
        return self.matched_children[0].child_text or ""

    def to_rerank_text(self, max_chars: int = 6000) -> str:
        child_text = self.best_child_text()
        parent_text = self.text or ""

        combined = (
            "Matched evidence:\n"
            f"{child_text}\n\n"
            "Parent context:\n"
            f"{parent_text}"
        )

        return combined[:max_chars]

    def to_prompt_source(self, index: int, max_chars: int = 7000) -> str:
        child_text = self.best_child_text()
        parent_text = self.text or ""

        if child_text:
            content = (
                f"Matched Evidence:\n{child_text}\n\n"
                f"Parent Context:\n{parent_text}"
            )
        else:
            content = parent_text

        content = content[:max_chars]

        score_text = ""
        if self.rerank_score is not None:
            score_text = f"\nRerank Score: {self.rerank_score:.4f}"

        return (
            f"SOURCE [{index}]\n"
            f"Type: {self.source_type}\n"
            f"Source: {self.source}\n"
            f"Doc ID: {self.doc_id}\n"
            f"Parent ID: {self.parent_id}"
            f"{score_text}\n"
            f"Content:\n{content}"
        )

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "doc_id": self.doc_id,
            "parent_id": self.parent_id,
            "source": self.source,
            "fusion_score": self.fusion_score,
            "rerank_score": self.rerank_score,
            "metadata": self.metadata,
            "matched_children": [
                {
                    "child_id": child.child_id,
                    "fusion_score": child.fusion_score,
                    "match_types": child.match_types,
                    "metadata": child.metadata,
                    "preview": child.child_text[:500],
                }
                for child in self.matched_children
            ],
        }


@dataclass
class ConfidenceDecision:
    should_use_web: bool
    reason: str
    confidence: str