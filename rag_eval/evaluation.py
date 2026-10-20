"""Transparent golden-set evaluation of a lexical RAG baseline."""
from __future__ import annotations
from .engine import RAGWorkbench


def evaluate(engine: RAGWorkbench, questions: list[dict], *, top_k: int = 3) -> dict:
    reports = []
    for item in questions:
        role = item.get("role", "public")
        response = engine.answer(item["question"], role=role, top_k=top_k)
        found = [p["document_id"] for p in response["retrieved"]]
        expected = item.get("expected_document")
        expected_abstain = expected is None
        hit = expected in found if expected else response["abstained"]
        reciprocal = 1 / (found.index(expected) + 1) if expected and expected in found else 0.0
        verified = engine.verify(response, role=role)
        correct = response["abstained"] == expected_abstain
        reports.append({"id": item["id"], "role": role, "expected_document": expected,
                        "retrieved_documents": found, "recall_at_k": bool(hit),
                        "reciprocal_rank": reciprocal, "correct_abstention": correct,
                        "citation_verbatim_valid": verified})
    size = max(len(reports), 1)
    non_abstain = [r for r in reports if r["expected_document"] is not None]
    def mean(items):
        return round(sum(items) / len(items), 3) if items else 0.0
    return {"dataset_size": len(reports), "recall_at_k": mean([r["recall_at_k"] for r in non_abstain]),
            "mrr": mean([r["reciprocal_rank"] for r in non_abstain]),
            "abstention_accuracy": mean([r["correct_abstention"] for r in reports]),
            "verbatim_citation_validity": mean([r["citation_verbatim_valid"] for r in reports]),
            "cases": reports,
            "disclaimer": "Scores are for small synthetic fixtures, not a production benchmark."}
