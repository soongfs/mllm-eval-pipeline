from typing import Any, Callable

from mllm_eval_pipeline.io import read_jsonl, write_json, write_jsonl
from mllm_eval_pipeline.metadata import now_iso, read_git_sha
from mllm_eval_pipeline.metrics import normalize_for_comparison
from mllm_eval_pipeline.parser import extract_answer
from mllm_eval_pipeline.paths import meta_path, predictions_path

VERIFIER_RULE = "rule"
VERIFIER_MAJORITY = "majority"
VERIFIER_MAJORITY_RULE = "majority+rule"
VERIFIERS = [VERIFIER_RULE, VERIFIER_MAJORITY, VERIFIER_MAJORITY_RULE]


def score_rule(record: dict[str, Any], response: str) -> dict[str, Any]:
    """Score a candidate by self-features only.

    Kept signals filter out obvious junk: missing answer, zero or multiple
    boxed answers, invalid multi-choice letters. Removed signals (reasoning
    keyword match, word-count range) were zero- or negative-value on 3B CoT
    output and skewed with max_tokens.
    """
    score = 0
    details: dict[str, Any] = {}
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

    return {
        "score": score,
        "model_answer": answer,
        "details": details,
    }


def scored_candidates(
    record: dict[str, Any], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result = []
    for idx, candidate in enumerate(candidates):
        response = candidate["response"]
        verification = score_rule(record, response)
        result.append(
            {
                **candidate,
                "candidate_index": idx,
                "verification": verification,
            }
        )
    return result


def select_rule(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the highest rule-scored candidate; ties break by first occurrence."""
    return max(
        candidates,
        key=lambda candidate: (
            candidate["verification"]["score"],
            -candidate["candidate_index"],
        ),
    )


def bucket_key(candidate: dict[str, Any]) -> str | None:
    answer = candidate["verification"]["model_answer"]
    if not answer:
        return None
    return normalize_for_comparison(answer)


def select_majority(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Group candidates by normalized answer, pick any from the largest bucket.

    Candidates without an extractable answer do not participate. If no
    candidate has an answer, fall back to the first candidate.
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for candidate in candidates:
        key = bucket_key(candidate)
        if key is None:
            continue
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(candidate)

    if not buckets:
        return candidates[0]

    # Largest bucket; earliest-seen key wins ties so we stay deterministic.
    best_key = max(order, key=lambda key: (len(buckets[key]), -order.index(key)))
    return buckets[best_key][0]


def select_majority_rule(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Majority vote; tie-break the largest bucket by rule score."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for candidate in candidates:
        key = bucket_key(candidate)
        if key is None:
            continue
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(candidate)

    if not buckets:
        return select_rule(candidates)

    max_bucket_size = max(len(bucket) for bucket in buckets.values())
    top_keys = [key for key in order if len(buckets[key]) == max_bucket_size]

    # With a single majority bucket, pick best-scoring candidate inside it.
    # On a tie between buckets, compare the best-scoring candidate of each.
    bucket_champions = [select_rule(buckets[key]) for key in top_keys]
    return select_rule(bucket_champions)


SELECTORS: dict[str, Callable[[list[dict[str, Any]]], dict[str, Any]]] = {
    VERIFIER_RULE: select_rule,
    VERIFIER_MAJORITY: select_majority,
    VERIFIER_MAJORITY_RULE: select_majority_rule,
}


def verify_mathvision_predictions(
    split: str,
    base: str,
    experiment: str,
    verifier: str,
) -> None:
    if verifier not in SELECTORS:
        raise ValueError(f"Unknown verifier {verifier!r}; expected one of {VERIFIERS}")
    select = SELECTORS[verifier]

    records = []
    changed = 0
    total = 0

    for record in read_jsonl(predictions_path("mathvision", split, base)):
        candidates = record.get("candidates") or [{"response": record["response"]}]
        candidates = scored_candidates(record, candidates)

        best = select(candidates)

        record["candidates"] = candidates
        record["selected_candidate_index"] = best["candidate_index"]
        record["response"] = best["response"]
        record["verification"] = best["verification"]
        records.append(record)

        changed += int(best["candidate_index"] != 0)
        total += 1

    write_jsonl(predictions_path("mathvision", split, experiment), records)
    write_json(
        meta_path("mathvision", split, experiment),
        {
            "dataset": "mathvision",
            "split": split,
            "experiment": experiment,
            "stage": "verify",
            "verifier": verifier,
            "base_experiment": base,
            "num_samples": total,
            "num_changed": changed,
            "git_sha": read_git_sha(),
            "timestamp": now_iso(),
        },
        indent=2,
    )
    print(f"verified: {total}/{total}, changed: {changed}/{total}")
