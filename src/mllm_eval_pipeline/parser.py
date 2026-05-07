import re
from pathlib import Path

from mllm_eval_pipeline.io import read_jsonl, write_jsonl
from mllm_eval_pipeline.paths import (
    mathvision_parsed_jsonl,
    mathvision_predictions_jsonl,
    vstar_parsed_jsonl,
    vstar_predictions_jsonl,
)

ANSWER_PHRASES = [
    "the final answer is",
    "the answer is",
    "the correct answer is",
    "the answer should be",
]

CHOICE_WRAPPER_REPLACEMENTS = [
    ("(a)", "a"),  # Multiple-choice wrapper: `(a)` -> `a`.
    ("(b)", "b"),
    ("(c)", "c"),
    ("(d)", "d"),
    ("(e)", "e"),
    ("{a}", "a"),  # Braced choice wrapper: `{a}` -> `a`.
    ("{b}", "b"),
    ("{c}", "c"),
    ("{d}", "d"),
    ("{e}", "e"),
]

LATEX_DISPLAY_REPLACEMENTS = [
    ("\\left", ""),  # Display delimiter: `\left(3\right)` -> `(3)`.
    ("\\right", ""),
    ("\\,", ""),  # Thin space: `1\,2` -> `12`.
    ("\\!", ""),  # Negative thin space: `1\!2` -> `12`.
    ("\\$", ""),  # Escaped dollar wrapper: `\$3\$` -> `3`.
    ("$", ""),  # Math-mode wrapper: `$3$` -> `3`.
]


def is_number(text: str) -> bool:
    """Check whether text can be parsed as a number.

    Args:
        text: Candidate numeric string.

    Returns:
        True if the string can be parsed by `float`, otherwise False.
    """
    try:
        float(text)
    except ValueError:
        return False
    return True


def apply_replacements(text: str, replacements: list[tuple[str, str]]) -> str:
    """Apply ordered string replacements.

    Args:
        text: Input text to normalize.
        replacements: Ordered `(old, new)` replacement pairs.

    Returns:
        Text after all replacements are applied.
    """
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def fix_sqrt(text: str) -> str:
    """Add braces to compact square-root notation.

    Example:
        `\\sqrt2` becomes `\\sqrt{2}`.
    """
    if "\\sqrt" not in text:
        return text

    parts = text.split("\\sqrt")
    fixed = parts[0]
    for part in parts[1:]:
        if part and part[0] != "{":
            fixed += "\\sqrt{" + part[0] + "}" + part[1:]
        else:
            fixed += "\\sqrt" + part
    return fixed


def fix_fracs(text: str) -> str:
    """Add braces to compact LaTeX fraction notation.

    Example:
        `\\frac12` becomes `\\frac{1}{2}`.
    """
    parts = text.split("\\frac")
    if len(parts) == 1:
        return text

    fixed = parts[0]
    for part in parts[1:]:
        fixed += "\\frac"
        if part.startswith("{"):
            fixed += part
            continue

        if len(part) < 2:
            return text

        numerator = part[0]
        denominator = part[1]
        rest = part[2:]
        if denominator == "{":
            fixed += "{" + numerator + "}" + denominator + rest
        else:
            fixed += "{" + numerator + "}{" + denominator + "}" + rest
    return fixed


def fix_slash_fraction(text: str) -> str:
    """Convert simple bare fractions to LaTeX fractions.

    Example:
        `1/2` becomes `\\frac{1}{2}`.
    """
    parts = text.split("/")
    if len(parts) != 2:
        return text

    numerator, denominator = parts
    try:
        numerator_int = int(numerator)
        denominator_int = int(denominator)
    except ValueError:
        return text

    if text != f"{numerator_int}/{denominator_int}":
        return text
    return f"\\frac{{{numerator_int}}}{{{denominator_int}}}"


def extract_last_boxed_content(text: str) -> str | None:
    """Extract the content of the last boxed answer.

    Args:
        text: Raw model response text.

    Returns:
        The final boxed payload, or None if no boxed answer is found.

    Example:
        `first \\boxed{A}, final \\boxed{B}` returns `B`.
    """
    matches = list(re.finditer(r"(?:\\)?boxed\{", text, flags=re.IGNORECASE))
    if not matches:
        return None

    start = matches[-1].end()
    depth = 1
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index]

    return text[start:]


def normalize_answer(text: str) -> str:
    """Normalize an extracted answer into the parser's output contract.

    Args:
        text: Candidate answer text extracted from the model response.

    Returns:
        A compact answer string, or an empty string if the input is empty.
    """
    answer = text.strip().lower()
    answer = answer.lstrip(":").strip()
    answer = answer.rstrip(".").strip()

    # Remove display-only wrappers: `$3$` -> `3`, `\(3\)` -> `3`.
    if answer.startswith("\\(") and answer.endswith("\\)"):
        answer = answer[2:-2].strip()
    if answer.startswith("$") and answer.endswith("$"):
        answer = answer[1:-1].strip()

    # Keep the value side of simple answer equations: `x = 3` -> `3`.
    if answer.count("=") == 1:
        answer = answer.split("=")[-1].strip()
    if answer.count("\\approx") == 1:
        answer = answer.split("\\approx")[-1].strip()

    answer = apply_replacements(answer, LATEX_DISPLAY_REPLACEMENTS)

    if not answer:
        return answer

    # Leading decimal shorthand: `.5` -> `0.5`.
    if answer.startswith("."):
        answer = "0" + answer

    # Compact LaTeX forms: `\sqrt2` -> `\sqrt{2}`.
    if "sqrt" in answer:
        answer = fix_sqrt(answer)

    # Compact LaTeX fractions: `\frac12` -> `\frac{1}{2}`.
    if "\\frac" in answer:
        answer = fix_fracs(answer)

    # Bare integer fractions: `1/2` -> `\frac{1}{2}`.
    answer = fix_slash_fraction(answer)

    # Choice wrappers: `(a)` -> `a`, `{a}` -> `a`.
    answer = apply_replacements(answer, CHOICE_WRAPPER_REPLACEMENTS)
    return answer.strip()


def extract_choice_shortcut(text: str) -> str | None:
    """Extract simple option-letter answer shortcuts.

    Args:
        text: Raw model response text.

    Returns:
        The matched choice letter, or None if no shortcut pattern is found.
    """
    for choice in "ABCDE":
        if (
            text.endswith(f" {choice}.")
            or text.endswith(f" ({choice}).")
            or text.startswith(f"{choice}\n")
            or text.startswith(f"({choice})\n")
            or text.startswith(f"({choice}) {choice}\n")
        ):
            return choice
    return None


def extract_numeric_suffix(text: str) -> str | None:
    """Extract a numeric answer from a response ending with `is <number>`.

    Args:
        text: Raw model response text.

    Returns:
        Numeric answer text, or None if no numeric suffix is found.
    """
    answer = text.split("is ")[-1].rstrip(".")
    if is_number(answer):
        return answer
    return None


def extract_answer_phrase(text: str) -> str | None:
    """Extract answer text after common final-answer phrases.

    Args:
        text: Raw model response text.

    Returns:
        Text following the first matched answer phrase, or None if no phrase is
        found.
    """
    for phrase in ANSWER_PHRASES:
        answer = split_answer_phrase(text, phrase)
        if answer is not None:
            return answer

        answer = split_answer_phrase(text, phrase.replace("the", "The"))
        if answer is not None:
            return answer
    return None


def split_answer_phrase(text: str, phrase: str) -> str | None:
    """Split response text on a final-answer phrase.

    Args:
        text: Raw model response text.
        phrase: Answer phrase to search for.

    Returns:
        The answer segment following the phrase, or None if the phrase is not
        present.
    """
    if phrase not in text:
        return None

    answer = text.split(phrase)[-1].strip()
    return answer.split("\n")[0].split(". ")[0]


def extract_raw_answer(response: str) -> str:
    """Extract a raw final-answer span from a model response.

    Args:
        response: Raw model response text.

    Returns:
        The raw answer span, or an empty string when no reliable answer pattern
        is found.
    """
    response = response.strip()

    # Boxed final answer span: `therefore \boxed{\frac{1}{2}}` -> `\frac{1}{2}`.
    boxed_answer = extract_last_boxed_content(response)
    if boxed_answer is not None:
        return boxed_answer

    # Final-answer phrase: `the answer is 61.` -> `61`.
    phrase_answer = extract_answer_phrase(response)
    if phrase_answer is not None:
        return phrase_answer

    # Direct multiple-choice output: `(A).` -> `A`.
    choice = extract_choice_shortcut(response)
    if choice is not None:
        return choice

    # Simple numeric tail: `the answer is 3.` -> `3`.
    numeric_answer = extract_numeric_suffix(response)
    if numeric_answer is not None:
        return numeric_answer

    return ""


def extract_answer(response: str) -> str:
    """Extract a normalized final answer from a model response.

    Extraction and normalization are intentionally conservative. The parser
    only accepts explicit final-answer patterns and leaves mathematical
    equivalence checks to the metric layer.

    Args:
        response: Raw model response text.

    Returns:
        Normalized final answer, or an empty string if no reliable answer is
        found.
    """
    raw_answer = extract_raw_answer(response)
    if not raw_answer:
        return ""
    return normalize_answer(raw_answer)


def prediction_path(dataset: str, split: str, output_suffix: str | None) -> Path:
    """Build the prediction file path for a dataset.

    Args:
        dataset: Dataset name.
        split: MathVision split name. Ignored for V* because it only has one
            processed split.
        output_suffix: Optional experiment suffix for the prediction file.

    Returns:
        Path to the prediction JSONL file.
    """
    if dataset == "mathvision":
        return mathvision_predictions_jsonl(split, output_suffix)
    if dataset == "vstar":
        return vstar_predictions_jsonl(output_suffix)
    raise ValueError(f"Unsupported dataset: {dataset}")


def parsed_path(dataset: str, split: str, output_suffix: str | None) -> Path:
    """Build the parsed-output file path for a dataset.

    Args:
        dataset: Dataset name.
        split: MathVision split name. Ignored for V* because it only has one
            processed split.
        output_suffix: Optional experiment suffix for the parsed file.

    Returns:
        Path to the parsed JSONL file.
    """
    if dataset == "mathvision":
        return mathvision_parsed_jsonl(split, output_suffix)
    if dataset == "vstar":
        return vstar_parsed_jsonl(output_suffix)
    raise ValueError(f"Unsupported dataset: {dataset}")


def parse_predictions(
    dataset: str,
    split: str,
    output_suffix: str | None = None,
) -> None:
    """Parse prediction responses into model answers.

    Args:
        dataset: Dataset name.
        split: MathVision split name. Ignored for V* because it only has one
            processed split.
        output_suffix: Optional experiment suffix for input/output files.
    """
    total = 0
    parsed = 0
    records = []

    for record in read_jsonl(prediction_path(dataset, split, output_suffix)):
        model_answer = extract_answer(record["response"])
        record["model_answer"] = model_answer
        records.append(record)

        total += 1
        parsed += int(bool(model_answer))

    write_jsonl(parsed_path(dataset, split, output_suffix), records)
    print(f"parsed: {parsed}/{total}")
