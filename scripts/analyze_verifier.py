"""Analyze a verifier-derived experiment.

Reports transition statistics between the base experiment's first
candidate and the verified experiment's selected candidate. Reads the
`base_experiment` reference from the derived experiment's `meta.json`.

Usage:
    uv run python scripts/analyze_verifier.py <split> <experiment>

Example:
    uv run python scripts/analyze_verifier.py testmini k8_2048.majority

Oracle-related statistics have been removed here because `oracle` is now
a first-class verifier (see `--verifier oracle` in `mllm-eval verify`).
To get oracle accuracy, run the oracle verifier and read its result.json.
"""

from __future__ import annotations

import argparse
import json

from mllm_eval_pipeline.io import read_jsonl
from mllm_eval_pipeline.metrics import format_accuracy, is_correct
from mllm_eval_pipeline.parser import extract_answer
from mllm_eval_pipeline.paths import meta_path, predictions_path


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


def analyze(split: str, experiment: str) -> None:
    base = read_base_experiment(split, experiment)
    source_records = list(read_jsonl(predictions_path("mathvision", split, base)))
    verified_records = list(
        read_jsonl(predictions_path("mathvision", split, experiment))
    )

    first_correct = 0
    verified_correct = 0
    changed = 0
    wrong_to_right = 0
    right_to_wrong = 0
    right_to_right = 0
    wrong_to_wrong = 0

    for source, verified in zip(source_records, verified_records, strict=True):
        candidates = source.get("candidates") or [{"response": source["response"]}]
        first_is_correct = check_mathvision_response(source, candidates[0]["response"])
        verified_is_correct = check_mathvision_response(source, verified["response"])

        first_correct += int(first_is_correct)
        verified_correct += int(verified_is_correct)
        changed += int(verified.get("selected_candidate_index", 0) != 0)

        wrong_to_right += int(not first_is_correct and verified_is_correct)
        right_to_wrong += int(first_is_correct and not verified_is_correct)
        right_to_right += int(first_is_correct and verified_is_correct)
        wrong_to_wrong += int(not first_is_correct and not verified_is_correct)

    total = len(source_records)
    print(f"base:     {base}")
    print(f"first:    {format_accuracy(first_correct, total)}")
    print(f"verified: {format_accuracy(verified_correct, total)}")
    print(f"changed:  {changed}/{total}")
    print(f"wrong -> right: {wrong_to_right}")
    print(f"right -> wrong: {right_to_wrong}")
    print(f"right -> right: {right_to_right}")
    print(f"wrong -> wrong: {wrong_to_wrong}")
    print(f"net gain: {verified_correct - first_correct:+d}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a verifier-derived experiment."
    )
    parser.add_argument("split", choices=["testmini", "test"])
    parser.add_argument(
        "experiment",
        help="Verifier-derived experiment (e.g. k8_2048.majority)",
    )
    args = parser.parse_args()
    analyze(args.split, args.experiment)


if __name__ == "__main__":
    main()
