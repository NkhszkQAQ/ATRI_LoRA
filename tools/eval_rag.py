"""Evaluate a small, curated retrieval regression set, without calling the LLM."""
import argparse
from dataclasses import asdict, replace
import json
import time
from pathlib import Path

from atri_rag import Config, Retriever


def evaluate(rag, cases):
    rows = []
    for case in cases:
        started = time.perf_counter()
        hits = rag.search(case["query"], case.get("history", ()))
        ids = [hit["metadata"].get("fact_id") for hit in hits]
        passed = case["expected_fact"] in ids if case["expected_action"] == "retrieve" else not hits
        rows.append({**case, "passed": passed, "seconds": round(time.perf_counter() - started, 4), "hits": hits})
    return {"passed": sum(row["passed"] for row in rows), "total": len(rows), "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path("tests/fixtures/rag_eval.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())["cases"]
    rag = Retriever(args.storage, Config.from_env())
    if args.calibrate:
        # Only the calibration split selects the threshold. Holdout is not read here.
        results = []
        score_field = "rerank_min_score" if rag.config.rerank_enabled else "min_score"
        thresholds = (0.2, 0.35, 0.5, 0.65, 0.8) if rag.config.rerank_enabled else (0.5, 0.55, 0.6, 0.65, 0.7)
        for threshold in thresholds:
            rag.config = replace(rag.config, **{score_field: threshold})
            result = evaluate(rag, [c for c in cases if c["split"] == "calibration"])
            results.append({"threshold": threshold, **result})
        report = {"calibrated_field": score_field, "calibration": results}
        print(json.dumps([{k: r[k] for k in ("threshold", "passed", "total")} for r in results]))
    else:
        report = {"settings": asdict(rag.config), "threshold": rag.config.min_score,
                  "splits": {split: evaluate(rag, [c for c in cases if c["split"] == split])
                             for split in dict.fromkeys(c["split"] for c in cases)}}
        print(json.dumps({split: {k: result[k] for k in ("passed", "total")}
                          for split, result in report["splits"].items()}))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
