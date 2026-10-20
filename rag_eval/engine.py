"""Deterministic, access-aware RAG baseline using synthetic documentation.

This is a retrieval/evaluation workbench, not a hosted LLM or semantic embedding
service. Responses are verbatim spans from retrieved documents, so the data
path and citations can be tested offline.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Iterable

STOP = frozenset('''a an and are as at be by can do does for from how i in into is it its of on or our the their these this to was were what when where which who why will with you your'''.split())
TOKEN_RE = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def tokens(text: str) -> list[str]:
    """Lexical tokenizer; intentionally small and dependency-free."""
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP and len(t) > 1]


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    visibility: str
    paragraphs: tuple[str, ...]


@dataclass(frozen=True)
class Passage:
    id: str
    document_id: str
    title: str
    visibility: str
    text: str


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float


class RAGWorkbench:
    def __init__(self, documents: Iterable[Document]):
        docs = list(documents)
        if len({d.id for d in docs}) != len(docs):
            raise ValueError("Duplicate document ID")
        if any(d.visibility not in {"public", "support"} for d in docs):
            raise ValueError("Unsupported document visibility")
        if any(not d.paragraphs or not d.title for d in docs):
            raise ValueError("Document must have title and paragraphs")
        self.passages: list[Passage] = [
            Passage(f"{d.id}#p{i+1}", d.id, d.title, d.visibility, p)
            for d in docs for i, p in enumerate(d.paragraphs) if p.strip()
        ]

    def search(self, query: str, *, role: str = "public", top_k: int = 3) -> list[Hit]:
        if role not in {"public", "support"}:
            raise ValueError("role must be public or support")
        if not 1 <= top_k <= 10:
            raise ValueError("top_k must be 1 through 10")
        qterms = set(tokens(query))
        if not qterms:
            return []
        accessible = [p for p in self.passages if p.visibility == "public" or role == "support"]
        count = len(accessible)
        if not count:
            return []
        bags = [Counter(tokens(p.title) + tokens(p.text)) for p in accessible]
        document_freq = Counter(t for bag in bags for t in bag)
        avglen = sum(sum(bag.values()) for bag in bags) / count
        matches: list[Hit] = []
        for p, bag in zip(accessible, bags):
            score = 0.0
            length = sum(bag.values())
            for t in qterms:
                tf = bag[t]
                if not tf:
                    continue
                idf = math.log(1 + (count - document_freq[t] + 0.5) / (document_freq[t] + 0.5))
                score += idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * length / max(avglen, 1)))
            coverage = len(qterms & set(bag)) / len(qterms)
            if score > 0 and coverage >= 0.34:
                matches.append(Hit(p, round(score, 5)))
        return sorted(matches, key=lambda hit: (-hit.score, hit.passage.id))[:top_k]

    def answer(self, query: str, *, role: str = "public", top_k: int = 3) -> dict:
        hits = self.search(query, role=role, top_k=top_k)
        if not hits:
            return {"answer": "Insufficient evidence in accessible documents.", "abstained": True,
                    "citations": [], "retrieved": []}
        best = hits[0].passage
        terms_query = set(tokens(query))
        sentences = [x.strip() for x in SENTENCE_RE.split(best.text) if x.strip()]
        # Extract an exact source sentence; never turn retrieved text into instructions.
        sentence = max(sentences, key=lambda s: (len(set(tokens(s)) & terms_query), -len(s)))
        return {"answer": sentence, "abstained": False, "citations": [best.id],
                "retrieved": [{"id": hit.passage.id, "document_id": hit.passage.document_id,
                               "score": hit.score} for hit in hits]}

    def verify(self, response: dict, *, role: str = "public") -> bool:
        """Verify citations and verbatim answer support; NOT factual truth in general."""
        if role not in {"public", "support"}:
            raise ValueError("Invalid role")
        if response["abstained"]:
            return response["citations"] == []
        permitted = {p.id: p for p in self.passages
                     if p.visibility == "public" or role == "support"}
        cites = response.get("citations", [])
        return bool(cites) and all(cid in permitted for cid in cites) and any(
            response.get("answer", "") in permitted[cid].text for cid in cites
        )


def load_corpus(path: Path) -> list[Document]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return [Document(id=x["id"], title=x["title"], visibility=x["visibility"],
                     paragraphs=tuple(x["paragraphs"])) for x in records]
