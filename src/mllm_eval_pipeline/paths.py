from pathlib import Path

# Data
RAW_DATA_DIR = Path("data/raw")

# MathVision
MATHVISION_DATASET_NAME = "MathLLMs/MathVision"
MATHVISION_SPLIT = "testmini"
MATHVISION_PROCESSED_DIR = Path("data/processed/mathvision")
MATHVISION_PROCESSED_JSONL = MATHVISION_PROCESSED_DIR / "mathvision_testmini.jsonl"
MATHVISION_IMAGE_DIR = MATHVISION_PROCESSED_DIR / "images"
MATHVISION_PREDICTIONS_DIR = Path("outputs/mathvision")
MATHVISION_PREDICTIONS_JSONL = MATHVISION_PREDICTIONS_DIR / "predictions.jsonl"


# Qwen2.5-VL
QWEN25_VL_3B_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
