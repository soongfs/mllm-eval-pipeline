import re

from latex2sympy2 import latex2sympy

from mllm_eval_pipeline.io import read_jsonl, write_json, write_jsonl
from mllm_eval_pipeline.paths import (
    mathvision_parsed_jsonl,
    mathvision_result_json,
    vstar_parsed_jsonl,
    vstar_result_json,
)

UNIT_PATTERN = re.compile(
    r"(?:\\(?:text|mathrm)\{[^{}]*(?:cm|km|m)[^{}]*\}|"
    r"(?:cm|km|m)(?:\^\{?[23]\}?)?)$",
)
LATEX_TEXT_PATTERN = re.compile(r"\\(?:text|mathrm)\{([^{}]*)\}")
BROKEN_TEXT_PATTERN = re.compile(r"\\text([a-z])\b")
WHITESPACE_PATTERN = re.compile(r"\s+")


def unwrap_latex_text(answer: str) -> str:
    """Remove simple LaTeX text wrappers.

    Examples:
        `\\text{ cm}` becomes ` cm`.
        `\\mathrm{~cm}^{2}` becomes `~cm^{2}`.
        `\\textc` becomes `c`.

    Args:
        answer: Answer string that may contain LaTeX text commands.

    Returns:
        Answer with text command wrappers removed.
    """
    answer = LATEX_TEXT_PATTERN.sub(r"\1", answer)
    return BROKEN_TEXT_PATTERN.sub(r"\1", answer)


def remove_trailing_units(answer: str) -> str:
    """Remove common trailing units from a comparison string.

    Examples:
        `12m` becomes `12`.
        `125cm^{2}` becomes `125`.

    Args:
        answer: Normalized answer candidate.

    Returns:
        Answer with simple trailing length/area/volume units removed.
    """
    previous = None
    while previous != answer:
        previous = answer
        answer = UNIT_PATTERN.sub("", answer).strip()
    return answer


def normalize_for_comparison(answer: str) -> str:
    """Normalize answer text for symmetric metric comparison.

    This is intentionally used only in metrics, not in parser output.
    The parser keeps the extracted answer close to the model response; metric
    comparison can be more permissive because both model answer and ground truth
    go through the same normalization.

    Args:
        answer: Model answer or ground-truth answer text.

    Returns:
        A compact string suitable for exact or symbolic comparison.
    """
    answer = answer.lower().strip()

    # Remove display wrappers: `$12$` -> `12`, `\textdollar12` -> `12`.
    answer = answer.replace("\\textdollar", "")
    answer = answer.replace("$", "")

    # Unwrap text commands before deleting units: `12 \text{ cm}` -> `12 cm`.
    answer = unwrap_latex_text(answer)

    # Drop spacing-only LaTeX commands: `1\,000` -> `1000`.
    answer = answer.replace("~", "")
    answer = answer.replace("\\,", "")
    answer = answer.replace("\\!", "")

    # Drop visual sizing and degree markers: `\left(1\right)` -> `(1)`,
    # `15^{\circ}` -> `15`.
    answer = answer.replace("\\left", "")
    answer = answer.replace("\\right", "")
    answer = answer.replace("^{\\circ}", "")
    answer = answer.replace("^\\circ", "")

    answer = WHITESPACE_PATTERN.sub(" ", answer).strip()

    # Delete trailing units symmetrically: `125cm^{2}` and `125` both compare
    # as `125`.
    answer = remove_trailing_units(answer)
    return answer.replace(" ", "")


def eval_tuple(answer: str) -> str:
    """Evaluate tuple/list elements when possible.

    This keeps the MathVision-style tuple behavior while making failures
    non-fatal. Non-tuple answers are returned unchanged.

    Example:
        `(2*3,5+2)` becomes `(6,7)`.

    Args:
        answer: Candidate tuple/list string.

    Returns:
        Evaluated tuple/list string, or the original input when it cannot be
        safely evaluated.
    """
    elements = answer[1:-1].split(",")

    try:
        if answer[0] == "(" and answer[-1] == ")" and len(elements) > 1:
            evaluated = ",".join(evaluate_tuple_element(item) for item in elements)
            return f"({evaluated})"

        if answer[0] == "[" and answer[-1] == "]" and len(elements) > 1:
            evaluated = ",".join(evaluate_tuple_element(item) for item in elements)
            return f"[{evaluated}]"
    except Exception:
        return answer

    return answer


def evaluate_tuple_element(element: str) -> str:
    """Evaluate one tuple/list element unless it should stay symbolic."""
    if "infty" in element or element in ["a", "-a"]:
        return element
    return str(round(eval(str(latex2sympy(element))), 2))


def is_blank(answer: str) -> bool:
    """Return whether an answer is empty after removing spaces."""
    return answer.replace(" ", "") == ""


def is_symbolically_equal(answer: str, gt_answer: str) -> bool:
    """Compare two normalized math strings through latex2sympy.

    Example:
        `1/2` and `\\frac{1}{2}` compare as equal.
    """
    try:
        return round(eval(str(latex2sympy(gt_answer))), 2) == round(
            eval(str(latex2sympy(answer))), 2
        )
    except Exception:
        return False


def is_equal(answer: str, gt_answer: str) -> bool:
    """Judge whether a model answer is equivalent to the ground truth.

    Args:
        answer: Model answer string.
        gt_answer: Ground-truth answer string or option value.

    Returns:
        True when the answers match exactly, after metric normalization, after
        tuple normalization, or by symbolic comparison.
    """
    answer = answer.lower()
    gt_answer = gt_answer.lower()

    if is_blank(answer) or is_blank(gt_answer):
        return False

    if gt_answer.strip() == answer.strip():
        return True

    answer = normalize_for_comparison(answer)
    gt_answer = normalize_for_comparison(gt_answer)

    if is_blank(answer) or is_blank(gt_answer):
        return False

    if gt_answer == answer:
        return True

    answer = eval_tuple(answer)
    gt_answer = eval_tuple(gt_answer)

    if gt_answer == answer:
        return True

    return is_symbolically_equal(answer, gt_answer)


def get_gt_answer_value(record: dict) -> str:
    gt_answer = record["answer"]
    options = record["options"]

    if options:
        option_idx = ord(gt_answer.upper()) - ord("A")
        if 0 <= option_idx < len(options):
            return options[option_idx]

    return ""


def is_correct(record: dict) -> bool:
    gt_answer = record["answer"]
    gt_answer_value = get_gt_answer_value(record)
    model_answer = record["model_answer"]

    return is_equal(gt_answer, model_answer) or is_equal(gt_answer_value, model_answer)


def evaluate_model_answer(split: str, output_suffix: str | None = None) -> None:
    records = []
    parsed_jsonl = mathvision_parsed_jsonl(split, output_suffix)

    for record in read_jsonl(parsed_jsonl):
        record["is_correct"] = is_correct(record)
        records.append(record)

    write_jsonl(parsed_jsonl, records)


def format_accuracy(correct: int, total: int) -> str:
    if total == 0:
        return "0/0=0%"
    return f"{correct}/{total}={round(correct / total * 100, 2)}%"


def add_metric(results: dict[str, list[int]], key: str, correct: bool) -> None:
    if key not in results:
        results[key] = [0, 0]

    results[key][0] += int(correct)
    results[key][1] += 1


def compute_mathvision_accuracy(
    split: str,
    output_suffix: str | None = None,
) -> dict[str, str]:
    results: dict[str, list[int]] = {}

    for record in read_jsonl(mathvision_parsed_jsonl(split, output_suffix)):
        correct = bool(record["is_correct"])
        level = record["level"]
        subject = record["subject"]

        for key in [
            "all",
            f"level{level}",
            subject,
            f"{subject}_level{level}",
        ]:
            add_metric(results, key, correct)

    return {
        key: format_accuracy(correct, total)
        for key, (correct, total) in sorted(results.items())
    }


def evaluate_mathvision(split: str, output_suffix: str | None = None) -> None:
    evaluate_model_answer(split, output_suffix)
    results = compute_mathvision_accuracy(split, output_suffix)
    write_json(mathvision_result_json(split, output_suffix), results)
    print(f"mathvision/{split}:\t{results['all']}")


def evaluate_vstar(output_suffix: str | None = None) -> None:
    records = []
    parsed_jsonl = vstar_parsed_jsonl(output_suffix)

    for record in read_jsonl(parsed_jsonl):
        model_answer = record["model_answer"].strip().upper()
        gt_answer = record["answer"].strip().upper()
        record["is_correct"] = model_answer == gt_answer
        records.append(record)

    write_jsonl(parsed_jsonl, records)

    results: dict[str, list[int]] = {}
    for record in records:
        correct = bool(record["is_correct"])
        for key in ["all", record["category"]]:
            if key not in results:
                results[key] = [0, 0]
            results[key][0] += int(correct)
            results[key][1] += 1

    formatted = {
        key: format_accuracy(correct, total)
        for key, (correct, total) in sorted(results.items())
    }

    write_json(vstar_result_json(output_suffix), formatted)
    print(f"vstar:\t{formatted['all']}")
    for key, value in formatted.items():
        if key != "all":
            print(f"  {key}:\t{value}")
