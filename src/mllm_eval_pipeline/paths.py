from pathlib import Path

# Data and outputs
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PREDICTIONS_DIR = Path("outputs/predictions")
PARSED_DIR = Path("outputs/parsed")
RESULT_DIR = Path("outputs/result")


# MathVision
MATHVISION_DATASET_NAME = "MathLLMs/MathVision"
MATHVISION_SPLIT = "testmini"

MATHVISION_PROCESSED_JSONL = PROCESSED_DIR / "mathvision/mathvision_testmini.jsonl"
MATHVISION_IMAGE_DIR = PROCESSED_DIR / "mathvision/images"
MATHVISION_PREDICTIONS_JSONL = PREDICTIONS_DIR / "mathvision/predictions.jsonl"
MATHVISION_PARSED_JSONL = PARSED_DIR / "mathvision/parsed.jsonl"
MATHVISION_RESULT_JSON = RESULT_DIR / "result.json"


# Qwen2.5-VL
QWEN25_VL_3B_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
