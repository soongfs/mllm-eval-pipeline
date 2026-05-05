from typing import Any

from mllm_eval_pipeline.io import read_jsonl, write_jsonl
from mllm_eval_pipeline.parser import extract_answer
from mllm_eval_pipeline.paths import mathvision_predictions_jsonl

MATH_REASONING_WORDS = [
    "therefore",
    "thus",
    "so",
    "because",
    "calculate",
    "equation",
    "substitute",
    "simplify",
    "solve",
]


def score_mathvision_candidate(record: dict[str, Any], response: str) -> dict[str, Any]:
    score = 0
    details = {}
    answer = extract_answer(response).strip()
    boxed_count = response.count("oxed{")

    details["answer"] = answer
    details["has_answer"] = bool(answer)
    if details["has_answer"]:
        score += 3

    details["single_boxed_answer"] = boxed_count == 1
    if details["single_boxed_answer"]:
        score += 2

    details["multiple_boxed_answers"] = boxed_count > 1
    if details["multiple_boxed_answers"]:
        score -= 2

    if record.get("options"):
        details["valid_choice_answer"] = answer.upper() in {"A", "B", "C", "D", "E"}
        if details["valid_choice_answer"]:
            score += 1
    else:
        details["valid_choice_answer"] = None

    response_lower = response.lower()
    details["reasoning_signal"] = any(
        word in response_lower for word in MATH_REASONING_WORDS
    )
    if details["reasoning_signal"]:
        score += 1

    word_count = len(response.split())
    details["word_count"] = word_count
    details["length_sane"] = 20 <= word_count <= 260
    if details["length_sane"]:
        score += 1

    return {
        "score": score,
        "model_answer": answer,
        "details": details,
    }


def verify_mathvision_predictions(
    split: str,
    suffix: str | None = None,
) -> None:
    output_suffix = f"{suffix}_verified" if suffix else "verified"

    records = []
    selected = 0
    total = 0

    for record in read_jsonl(mathvision_predictions_jsonl(split, suffix)):
        candidates = record.get("candidates")
        if not candidates:
            candidates = [{"response": record["response"]}]

        scored_candidates = []
        for idx, candidate in enumerate(candidates):
            response = candidate["response"]
            verification = score_mathvision_candidate(record, response)
            scored_candidate = {
                **candidate,
                "verification": verification,
                "candidate_index": idx,
            }
            scored_candidates.append(scored_candidate)

        best_candidate = max(
            scored_candidates,
            key=lambda candidate: candidate["verification"]["score"],
        )

        record["candidates"] = scored_candidates
        record["selected_candidate_index"] = best_candidate["candidate_index"]
        record["response"] = best_candidate["response"]
        record["verification"] = best_candidate["verification"]
        records.append(record)

        selected += int(best_candidate["candidate_index"] != 0)
        total += 1

    write_jsonl(mathvision_predictions_jsonl(split, output_suffix), records)
    print(f"verified: {total}/{total}, changed: {selected}/{total}")
