from latex2sympy2 import latex2sympy

from mllm_eval_pipeline.io import read_jsonl, write_json, write_jsonl
from mllm_eval_pipeline.paths import (
    mathvision_parsed_jsonl,
    mathvision_result_json,
    vstar_parsed_jsonl,
    vstar_result_json,
)


# Refer to https://github.com/mathllm/MATH-V/blob/main/evaluation/utils.py
def eval_tuple(s):
    """
    Evaluates the mathematical expressions within tuples
    or lists represented as strings.

    Args:
        s (str): The string representation of a tuple or list.
                 E.g., "(a,b,c,...)" or "[a,b,c,...]"

    Returns:
        str: A string representation of the tuple or list with evaluated expressions.
             Returns the original string if it doesn't match
             the expected format or if an error occurs.

    Example:
        eval_tuple("(2*3, 5+2)") -> "(6,7)"

    Note:
        This function relies on the latex2sympy function
        which is assumed to be defined elsewhere in the code.
    """
    # Split the string by commas to get individual elements
    sl = s[1:-1].split(",")

    try:
        # Check if string is a tuple representation and has more than one element
        if s[0] == "(" and s[-1] == ")" and len(sl) > 1:
            # Evaluate each element using latex2sympy
            # and round the result to 2 decimal places
            # Skip evaluation if element is 'infty', 'a', or '-a'
            s = ",".join(
                [
                    str(round(eval(str(latex2sympy(sub))), 2))
                    if "infty" not in sub and sub not in ["a", "-a"]
                    else sub
                    for sub in sl
                ]
            )
            return f"({s})"

        # Check if string is a list representation and has more than one element
        elif s[0] == "[" and s[-1] == "]" and len(sl) > 1:
            # Same evaluation process as for tuples
            s = ",".join(
                [
                    str(round(eval(str(latex2sympy(sub))), 2))
                    if "infty" not in sub and sub not in ["a", "-a"]
                    else sub
                    for sub in sl
                ]
            )
            return f"[{s}]"

    except Exception:  # Catch any exceptions and return the original string
        return s

    # Return original string if it doesn't match tuple or list format
    return s


def is_equal(asw: str, gt_asw: str) -> bool:
    """
    Judge if `asw` is equivalent to `gt_asw`.

    This function checks if the given answers are equivalent, considering
    various scenarios such as tuples, lists separated by commas, and
    mathematical equivalence in LaTeX format.

    Args:
        asw (str): The answer string to be checked.
        gt_asw (str): The ground truth answer string to be matched against.

    Returns:
        bool: True if the answers are equivalent, otherwise False.

    """

    # return gt_asw == asw

    # Check for empty strings after removing spaces
    # and return False if any of them is empty.
    asw = asw.lower()
    gt_asw = gt_asw.lower()

    if asw.replace(" ", "") == "" or gt_asw.replace(" ", "") == "":
        return False

    if gt_asw.strip() == asw.strip():
        return True

    # Convert the string to a tuple format.
    asw = eval_tuple(asw)
    gt_asw = eval_tuple(gt_asw)

    # Check for simple tuple containment.
    # Return True if one tuple is contained in the other.
    if gt_asw == asw:
        return True

    try:
        # Convert LaTeX format to a sympy expression and evaluate both expressions.
        # If the evaluated results are close enough (up to 2 decimal places),
        # return True.
        if round(eval(str(latex2sympy(gt_asw))), 2) == round(
            eval(str(latex2sympy(asw))), 2
        ):
            return True

        else:
            return False
    except Exception:
        # If any error occurs during comparison, return False.
        return False


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


def evaluate_model_answer(split: str) -> None:
    records = []
    parsed_jsonl = mathvision_parsed_jsonl(split)

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


def compute_mathvision_accuracy(split: str) -> dict[str, str]:
    results: dict[str, list[int]] = {}

    for record in read_jsonl(mathvision_parsed_jsonl(split)):
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


def evaluate_mathvision(split: str) -> None:
    evaluate_model_answer(split)
    results = compute_mathvision_accuracy(split)
    write_json(mathvision_result_json(split), results)
    print(f"mathvision/{split}:\t{results['all']}")


def evaluate_vstar() -> None:
    records = []
    parsed_jsonl = vstar_parsed_jsonl()

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

    write_json(vstar_result_json(), formatted)
    print(f"vstar:\t{formatted['all']}")
    for key, value in formatted.items():
        if key != "all":
            print(f"  {key}:\t{value}")
