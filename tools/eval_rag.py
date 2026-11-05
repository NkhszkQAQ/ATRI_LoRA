"""Evaluate a small, curated retrieval regression set, without calling the LLM."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from atri_rag import Config, Retriever


def evaluate(rag, cases):
    rows = []
    for case in cases:
        hits = rag.search(case["query"])
        ids = [hit["metadata"].get("fact_id") for hit in hits]
        passed = case["expected_fact"] in ids if case["expected_action"] == "retrieve" else not hits
        rows.append({**case, "passed": passed, "hits": hits})
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
        for threshold in (0.5, 0.55, 0.6, 0.65, 0.7):
            rag.config = replace(rag.config, min_score=threshold)
            result = evaluate(rag, [c for c in cases if c["split"] == "calibration"])
            results.append({"threshold": threshold, **result})
        report = {"calibration": results}
        print(json.dumps([{k: r[k] for k in ("threshold", "passed", "total")} for r in results]))
    else:
        report = {"threshold": rag.config.min_score,
                  "splits": {split: evaluate(rag, [c for c in cases if c["split"] == split])
                             for split in ("calibration", "heldout")}}
        print(json.dumps({split: {k: result[k] for k in ("passed", "total")}
                          for split, result in report["splits"].items()}))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
