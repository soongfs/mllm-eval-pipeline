# TODO Download and preprocess dataset:
# MathVision-testmini
# V*


import json

from datasets import load_dataset

from mllm_eval_pipeline.paths import (
    MATHVISION_DATASET_NAME,
    MATHVISION_IMAGE_DIR,
    MATHVISION_PROCESSED_JSONL,
    MATHVISION_SPLIT,
    RAW_DATA_DIR,
)


def download_mathvision() -> None:
    dataset = load_dataset(
        MATHVISION_DATASET_NAME,
        split=MATHVISION_SPLIT,
        cache_dir=str(RAW_DATA_DIR),
    )
    print(dataset)


def preprocess_mathvision() -> None:
    dataset = load_dataset(
        MATHVISION_DATASET_NAME,
        split=MATHVISION_SPLIT,
        cache_dir=str(RAW_DATA_DIR),
    )

    MATHVISION_PROCESSED_JSONL.parent.mkdir(parents=True, exist_ok=True)
    MATHVISION_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    with MATHVISION_PROCESSED_JSONL.open("w") as file:
        for sample in dataset:
            image_path = MATHVISION_IMAGE_DIR / f"{sample['id']}.jpg"

            image = sample.pop("decoded_image")
            image.save(image_path)

            file.write(json.dumps(sample) + "\n")
