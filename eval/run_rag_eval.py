"""RAG retrieval evaluation.

Each question is paraphrased so it does NOT contain the answer keyword; a retrieved
chunk counts as relevant if it contains one of `relevant_if_contains`.
Metrics: Hit@1, Hit@3, Hit@5 and MRR@5, for pure vector search vs our hybrid re-rank.

    python eval/run_rag_eval.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from finsight import rag  # noqa: E402

CONFIGS = {"vector only (alpha=1.0)": 1.0, "hybrid (alpha=0.75)": 0.75, "hybrid (alpha=0.5)": 0.5}


def evaluate(alpha, items):
    rows = []
    for it in items:
        hits = rag.search(it["ticker"], it["question"], k=5, alpha=alpha, log=False)
        rel = [any(kw.lower() in h["text"].lower() for kw in it["relevant_if_contains"]) for h in hits]
        rank = rel.index(True) + 1 if True in rel else None
        rows.append({**it, "rank": rank, "top_citation": hits[0]["citation"] if hits else None})
    n = len(rows)
    metrics = {f"hit@{k}": round(sum(1 for r in rows if r["rank"] and r["rank"] <= k) / n, 3) for k in (1, 3, 5)}
    metrics["mrr@5"] = round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 3)
    return metrics, rows


if __name__ == "__main__":
    items = json.loads((ROOT / "eval" / "rag_eval_set.json").read_text())
    for t in {i["ticker"] for i in items}:
        rag.ingest_ticker(t)
    results = {}
    for name, alpha in CONFIGS.items():
        metrics, rows = evaluate(alpha, items)
        results[name] = {"metrics": metrics, "rows": rows}
        print(f"{name:26s} {metrics}")
    best = max(results, key=lambda k: results[k]["metrics"]["mrr@5"])
    print(f"\nBest configuration: {best}")
    print("\nPer-question ranks (best config):")
    for r in results[best]["rows"]:
        print(f"  {str(r['rank']):>4}  {r['ticker']}  {r['question']}")
    out = ROOT / "eval" / "rag_eval_results.json"
    out.write_text(json.dumps({"n_questions": len(items), "results": results}, indent=1))
    print(f"\nSaved {out}")
