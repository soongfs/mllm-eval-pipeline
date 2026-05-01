# TODO Download and preprocess dataset:
# MathVision-testmini
# V*


import json
from pathlib import Path

from datasets import DownloadConfig, load_dataset
from huggingface_hub import snapshot_download

from mllm_eval_pipeline.paths import (
    MATHVISION_DATASET_NAME,
    RAW_DATA_DIR,
    VSTAR_DATASET_NAME,
    VSTAR_REPO_DIR,
    VSTAR_SPLIT,
    mathvision_image_dir,
    mathvision_processed_jsonl,
    vstar_image_dir,
    vstar_processed_jsonl,
)


def download_mathvision(split: str) -> None:
    dataset = load_dataset(
        MATHVISION_DATASET_NAME,
        split=split,
        cache_dir=str(RAW_DATA_DIR),
    )
    print(dataset)


def preprocess_mathvision(split: str) -> None:
    dataset = load_dataset(
        MATHVISION_DATASET_NAME,
        split=split,
        cache_dir=str(RAW_DATA_DIR),
    )

    processed_jsonl = mathvision_processed_jsonl(split)
    image_dir = mathvision_image_dir(split)

    processed_jsonl.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    with processed_jsonl.open("w") as file:
        for sample in dataset:
            image_path = image_dir / f"{sample['id']}.jpg"

            image = sample.pop("decoded_image")
            image.save(image_path)

            file.write(json.dumps(sample) + "\n")


def download_vstar() -> None:
    snapshot_download(
        repo_id=VSTAR_DATASET_NAME,
        repo_type="dataset",
        local_dir=VSTAR_REPO_DIR,
    )
    dataset = load_dataset(
        VSTAR_DATASET_NAME,
        split=VSTAR_SPLIT,
        cache_dir=str(RAW_DATA_DIR),
    )
    print(dataset)


def preprocess_vstar() -> None:
    dataset = load_dataset(
        VSTAR_DATASET_NAME,
        split=VSTAR_SPLIT,
        cache_dir=str(RAW_DATA_DIR),
        download_config=DownloadConfig(local_files_only=True),
    )

    processed_jsonl = vstar_processed_jsonl()
    image_dir = vstar_image_dir()
    processed_jsonl.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    with processed_jsonl.open("w") as file:
        for sample in dataset:
            image = Path(sample["image"])
            source_image = VSTAR_REPO_DIR / image
            target_image = image_dir / image
            target_image.parent.mkdir(parents=True, exist_ok=True)
            if not target_image.exists():
                target_image.symlink_to(source_image.resolve())

            record = {
                "id": sample["question_id"],
                "question": sample["text"],
                "image": str(Path("images") / image),
                "answer": sample["label"],
                "category": sample["category"],
            }
            file.write(json.dumps(record) + "\n")
