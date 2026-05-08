"""Export verifier reranking cases as JSONL for human inspection.

Reads a verifier-derived experiment + its base (from `meta.json`) and
writes the selected cases to
`outputs/mathvision/<split>/<experiment>/verifier_cases_<case_type>.jsonl`.

Usage:
    uv run python scripts/export_verifier_cases.py <split> <experiment>
        [--case-type ...] [--limit N]

Example:
    uv run python scripts/export_verifier_cases.py testmini k8_2048.majority \
        --case-type wrong-to-right --limit 20
"""

from __future__ import annotations

import argparse
import json

from mllm_eval_pipeline.io import read_jsonl, write_jsonl
from mllm_eval_pipeline.metrics import is_correct
from mllm_eval_pipeline.parser import extract_answer
from mllm_eval_pipeline.paths import meta_path, predictions_path, verifier_cases_path

CASE_TYPES = [
    "changed",
    "wrong-to-right",
    "right-to-wrong",
    "right-to-right",
    "wrong-to-wrong",
    "all",
]


def check_mathvision_response(record: dict, response: str) -> bool:
    checked_record = dict(record)
    checked_record["model_answer"] = extract_answer(response)
    return is_correct(checked_record)


def read_base_experiment(split: str, experiment: str) -> str:
    with meta_path("mathvision", split, experiment).open() as file:
        meta = json.load(file)
    base = meta.get("base_experiment")
    if not base:
        raise ValueError(
            f"Experiment {experiment!r} has no base_experiment in meta.json; "
            "is it a verifier-derived experiment?"
        )
    return base


def get_case_type(first_is_correct: bool, verified_is_correct: bool) -> str:
    if not first_is_correct and verified_is_correct:
        return "wrong-to-right"
    if first_is_correct and not verified_is_correct:
        return "right-to-wrong"
    if first_is_correct and verified_is_correct:
        return "right-to-right"
    return "wrong-to-wrong"


def format_candidate_case(
    idx: int, candidate: dict, record: dict, selected_candidate_index: int
) -> dict:
    response = candidate["response"]
    answer = extract_answer(response)
    correct = check_mathvision_response(record, response)
    verification = candidate.get("verification", {})
    return {
        "candidate_index": idx,
        "selected": idx == selected_candidate_index,
        "score": verification.get("score"),
        "parsed_answer": answer,
        "correct": correct,
        "verification": verification,
        "response": response,
    }


def export(split: str, experiment: str, case_type: str, limit: int | None) -> None:
    base = read_base_experiment(split, experiment)
    source_records = list(read_jsonl(predictions_path("mathvision", split, base)))
    verified_records = list(
        read_jsonl(predictions_path("mathvision", split, experiment))
    )

    output = verifier_cases_path("mathvision", split, experiment, case_type)
    exported_cases = []
    exported = 0

    for source, verified in zip(source_records, verified_records, strict=True):
        candidates = source.get("candidates") or [{"response": source["response"]}]

        first_is_correct = check_mathvision_response(source, candidates[0]["response"])
        verified_is_correct = check_mathvision_response(source, verified["response"])
        current_case_type = get_case_type(first_is_correct, verified_is_correct)
        selected_candidate_index = verified.get("selected_candidate_index", 0)
        changed = selected_candidate_index != 0

        if case_type == "changed" and not changed:
            continue
        if case_type not in {"all", "changed", current_case_type}:
            continue

        exported_cases.append(
            {
                "id": source["id"],
                "case_type": current_case_type,
                "split": split,
                "base_experiment": base,
                "experiment": experiment,
                "selected_candidate_index": selected_candidate_index,
                "ground_truth": source["answer"],
                "first_correct": first_is_correct,
                "verified_correct": verified_is_correct,
                "question": source["question"],
                "candidates": [],
            }
        )
        for idx, candidate in enumerate(verified["candidates"]):
            exported_cases[-1]["candidates"].append(
                format_candidate_case(idx, candidate, source, selected_candidate_index)
            )

        exported += 1
        if limit is not None and exported >= limit:
            break

    write_jsonl(output, exported_cases)
    print(f"exported: {exported}, output: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export verifier reranking cases as JSONL for human inspection."
    )
    parser.add_argument("split", choices=["testmini", "test"])
    parser.add_argument("experiment")
    parser.add_argument("--case-type", choices=CASE_TYPES, default="changed")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    export(args.split, args.experiment, args.case_type, args.limit)


if __name__ == "__main__":
    main()
