from mllm_eval_pipeline.io import read_jsonl, write_jsonl
from mllm_eval_pipeline.paths import (
    mathvision_parsed_jsonl,
    mathvision_predictions_jsonl,
)
from mllm_eval_pipeline.utils import find_math_answer, is_number


# Refer to https://github.com/mathllm/MATH-V/blob/main/evaluation/evaluate.py
def extract_answer(response: str) -> str:
    model_answer = response.strip()
    has_oxed = "oxed{" in model_answer
    matched_answer_phrase = False

    for c in "ABCDE":
        if (
            model_answer.endswith(f" {c}.")
            or model_answer.endswith(f" ({c}).")
            or model_answer.startswith(f"{c}\n")
            or model_answer.startswith(f"({c})\n")
            or model_answer.startswith(f"({c}) {c}\n")
        ):
            model_answer = c

    if is_number(model_answer.split("is ")[-1].rstrip(".")):
        model_answer = model_answer.split("is ")[-1].rstrip(".")

    if "oxed{" not in model_answer:
        for flag in [
            "the final answer is",
            "the answer is",
            "the correct answer is",
            "the answer should be",
        ]:
            raw_model_answer = model_answer
            model_answer = model_answer.split(flag)[-1].strip()
            if flag in raw_model_answer:
                matched_answer_phrase = True
                model_answer = model_answer.split("\n")[0].split(". ")[0]
            flag = flag.replace("the", "The")
            raw_model_answer = model_answer
            model_answer = model_answer.split(flag)[-1].strip()
            if flag in raw_model_answer:
                matched_answer_phrase = True
                model_answer = model_answer.split("\n")[0].split(". ")[0]

    elif model_answer.count("oxed{") > 1:
        model_answer = "\\boxed{" + model_answer.split("oxed{")[-1]

    if not has_oxed and not matched_answer_phrase:
        return ""

    model_answer = (
        find_math_answer(model_answer)
        .replace("(a)", "a")
        .replace("(b)", "b")
        .replace("(c)", "c")
        .replace("(d)", "d")
        .replace("(e)", "e")
        .replace("{a}", "a")
        .replace("{b}", "b")
        .replace("{c}", "c")
        .replace("{d}", "d")
        .replace("{e}", "e")
        .rstrip(".")
        .lstrip(":")
        .strip()
    )
    return model_answer


def parse_predictions(split: str) -> None:
    total = 0
    parsed = 0
    records = []

    for record in read_jsonl(mathvision_predictions_jsonl(split)):
        model_answer = extract_answer(record["response"])
        record["model_answer"] = model_answer
        records.append(record)

        total += 1
        parsed += int(bool(model_answer))

    write_jsonl(mathvision_parsed_jsonl(split), records)
    print(f"parsed: {parsed}/{total}")
