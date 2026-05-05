from pathlib import Path

# Data and outputs
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PREDICTIONS_DIR = Path("outputs/predictions")
PARSED_DIR = Path("outputs/parsed")
RESULT_DIR = Path("outputs/result")


# MathVision
MATHVISION_DATASET_NAME = "MathLLMs/MathVision"
MATHVISION_DEFAULT_SPLIT = "testmini"
MATHVISION_SPLITS = ["testmini", "test"]


def add_output_suffix(path: Path, suffix: str | None) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def mathvision_processed_jsonl(split: str) -> Path:
    return PROCESSED_DIR / "mathvision" / split / "samples.jsonl"


def mathvision_image_dir(split: str) -> Path:
    return PROCESSED_DIR / "mathvision" / split / "images"


def mathvision_predictions_jsonl(split: str, suffix: str | None = None) -> Path:
    path = PREDICTIONS_DIR / "mathvision" / split / "predictions.jsonl"
    return add_output_suffix(path, suffix)


def mathvision_parsed_jsonl(split: str, suffix: str | None = None) -> Path:
    path = PARSED_DIR / "mathvision" / split / "parsed.jsonl"
    return add_output_suffix(path, suffix)


def mathvision_result_json(split: str, suffix: str | None = None) -> Path:
    path = RESULT_DIR / "mathvision" / split / "result.json"
    return add_output_suffix(path, suffix)


# Qwen2.5-VL
QWEN25_VL_3B_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


# V*
VSTAR_DATASET_NAME = "craigwu/vstar_bench"
VSTAR_SPLIT = "test"
VSTAR_REPO_DIR = RAW_DATA_DIR / "vstar_repo"


def vstar_processed_dir() -> Path:
    return PROCESSED_DIR / "vstar" / VSTAR_SPLIT


def vstar_processed_jsonl() -> Path:
    return vstar_processed_dir() / "samples.jsonl"


def vstar_image_dir() -> Path:
    return vstar_processed_dir() / "images"


def vstar_predictions_jsonl(suffix: str | None = None) -> Path:
    path = PREDICTIONS_DIR / "vstar" / VSTAR_SPLIT / "predictions.jsonl"
    return add_output_suffix(path, suffix)


def vstar_parsed_jsonl(suffix: str | None = None) -> Path:
    path = PARSED_DIR / "vstar" / VSTAR_SPLIT / "parsed.jsonl"
    return add_output_suffix(path, suffix)


def vstar_result_json(suffix: str | None = None) -> Path:
    path = RESULT_DIR / "vstar" / VSTAR_SPLIT / "result.json"
    return add_output_suffix(path, suffix)
