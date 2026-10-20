"""RAG lab: offline retrieval, exact-sentence citations and evaluations."""
import argparse
import json
from pathlib import Path
from rag_eval import RAGWorkbench
from rag_eval.engine import load_corpus
from rag_eval.evaluation import evaluate

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="Ask about the synthetic support documents")
    parser.add_argument("--role", choices=["public", "support"], default="public")
    parser.add_argument("--evaluate", action="store_true", help="Run synthetic golden-set evaluation")
    args = parser.parse_args()
    engine = RAGWorkbench(load_corpus(HERE / "data" / "documents.json"))
    if args.evaluate:
        cases = json.loads((HERE / "data" / "golden.json").read_text(encoding="utf-8"))
        result = evaluate(engine, cases)
    else:
        if not args.query:
            parser.error("Supply --query or --evaluate")
        result = engine.answer(args.query, role=args.role)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
