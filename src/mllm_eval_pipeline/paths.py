from pathlib import Path

# Data and outputs
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUTPUTS_DIR = Path("outputs")


# MathVision
MATHVISION_DATASET_NAME = "MathLLMs/MathVision"
MATHVISION_DEFAULT_SPLIT = "testmini"
MATHVISION_SPLITS = ["testmini", "test"]


def mathvision_processed_jsonl(split: str) -> Path:
    return PROCESSED_DIR / "mathvision" / split / "samples.jsonl"


def mathvision_image_dir(split: str) -> Path:
    return PROCESSED_DIR / "mathvision" / split / "images"


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


# Experiment outputs (one directory per experiment, holding all artifacts)
def experiment_dir(dataset: str, split: str, experiment: str) -> Path:
    return OUTPUTS_DIR / dataset / split / experiment


def predictions_path(dataset: str, split: str, experiment: str) -> Path:
    return experiment_dir(dataset, split, experiment) / "predictions.jsonl"


def parsed_path(dataset: str, split: str, experiment: str) -> Path:
    return experiment_dir(dataset, split, experiment) / "parsed.jsonl"


def result_path(dataset: str, split: str, experiment: str) -> Path:
    return experiment_dir(dataset, split, experiment) / "result.json"


def meta_path(dataset: str, split: str, experiment: str) -> Path:
    return experiment_dir(dataset, split, experiment) / "meta.json"


def verifier_cases_path(
    dataset: str, split: str, experiment: str, case_type: str
) -> Path:
    return (
        experiment_dir(dataset, split, experiment)
        / f"verifier_cases_{case_type}.jsonl"
    )
