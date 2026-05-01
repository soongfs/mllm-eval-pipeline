# TODO Download and preprocess dataset:
# MathVision-testmini
# V*


import json

from datasets import load_dataset

from mllm_eval_pipeline.paths import (
    MATHVISION_DATASET_NAME,
    RAW_DATA_DIR,
    mathvision_image_dir,
    mathvision_processed_jsonl,
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
