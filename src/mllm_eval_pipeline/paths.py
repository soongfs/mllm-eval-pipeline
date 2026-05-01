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


def mathvision_processed_jsonl(split: str) -> Path:
    return PROCESSED_DIR / "mathvision" / split / "samples.jsonl"


def mathvision_image_dir(split: str) -> Path:
    return PROCESSED_DIR / "mathvision" / split / "images"


def mathvision_predictions_jsonl(split: str) -> Path:
    return PREDICTIONS_DIR / "mathvision" / split / "predictions.jsonl"


def mathvision_parsed_jsonl(split: str) -> Path:
    return PARSED_DIR / "mathvision" / split / "parsed.jsonl"


def mathvision_result_json(split: str) -> Path:
    return RESULT_DIR / "mathvision" / split / "result.json"


# Qwen2.5-VL
QWEN25_VL_3B_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
