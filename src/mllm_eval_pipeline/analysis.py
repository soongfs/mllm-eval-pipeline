from mllm_eval_pipeline.io import read_jsonl
from mllm_eval_pipeline.metrics import format_accuracy, is_correct
from mllm_eval_pipeline.parser import extract_answer
from mllm_eval_pipeline.paths import mathvision_predictions_jsonl


def check_mathvision_response(record: dict, response: str) -> bool:
    checked_record = dict(record)
    checked_record["model_answer"] = extract_answer(response)
    return is_correct(checked_record)


def analyze_mathvision_verifier(split: str, suffix: str) -> None:
    source_records = list(read_jsonl(mathvision_predictions_jsonl(split, suffix)))
    verified_records = list(
        read_jsonl(mathvision_predictions_jsonl(split, f"{suffix}_verified"))
    )

    first_correct = 0
    verified_correct = 0
    oracle_correct = 0
    changed = 0
    wrong_to_right = 0
    right_to_wrong = 0
    right_to_right = 0
    wrong_to_wrong = 0

    for source, verified in zip(source_records, verified_records, strict=True):
        candidates = source.get("candidates")
        if not candidates:
            candidates = [{"response": source["response"]}]

        first_is_correct = check_mathvision_response(
            source,
            candidates[0]["response"],
        )
        verified_is_correct = check_mathvision_response(
            source,
            verified["response"],
        )
        oracle_is_correct = any(
            check_mathvision_response(source, candidate["response"])
            for candidate in candidates
        )

        first_correct += int(first_is_correct)
        verified_correct += int(verified_is_correct)
        oracle_correct += int(oracle_is_correct)

        selected_candidate_index = verified.get("selected_candidate_index", 0)
        changed += int(selected_candidate_index != 0)

        wrong_to_right += int(not first_is_correct and verified_is_correct)
        right_to_wrong += int(first_is_correct and not verified_is_correct)
        right_to_right += int(first_is_correct and verified_is_correct)
        wrong_to_wrong += int(not first_is_correct and not verified_is_correct)

    total = len(source_records)
    print(f"first:    {format_accuracy(first_correct, total)}")
    print(f"verified: {format_accuracy(verified_correct, total)}")
    print(f"oracle:   {format_accuracy(oracle_correct, total)}")
    print(f"changed:  {changed}/{total}")
    print(f"wrong -> right: {wrong_to_right}")
    print(f"right -> wrong: {right_to_wrong}")
    print(f"right -> right: {right_to_right}")
    print(f"wrong -> wrong: {wrong_to_wrong}")
    print(f"net gain: {verified_correct - first_correct:+d}")
