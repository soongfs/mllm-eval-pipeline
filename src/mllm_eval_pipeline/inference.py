# TODO Implement model inference
# Qwen/Qwen2.5-VL-3B-Instruct
# LLaVA


import os

# vLLM is non-deterministic by default for throughput. For pipeline
# reproducibility we force deterministic scheduling per the official
# docs: https://docs.vllm.ai/en/v0.19.1/usage/reproducibility/
# (`VLLM_ENABLE_V1_MULTIPROCESSING=0` in offline mode). Verified
# bit-identical output across two independent greedy_2048 runs on
# Qwen2.5-VL-3B-Instruct (diff = empty). Setting via `setdefault` so
# users can opt out by exporting the variable in their shell.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from pathlib import Path  # noqa: E402  -- env var must be set before vllm import
from typing import Any  # noqa: E402

from PIL import Image  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TextPrompt  # noqa: E402

from mllm_eval_pipeline.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from mllm_eval_pipeline.metadata import (  # noqa: E402
    now_iso,
    package_version,
    read_git_sha,
)
from mllm_eval_pipeline.paths import (  # noqa: E402
    QWEN25_VL_3B_MODEL,
    VSTAR_SPLIT,
    mathvision_image_dir,
    mathvision_processed_jsonl,
    meta_path,
    predictions_path,
    vstar_processed_dir,
    vstar_processed_jsonl,
)


def build_inference_metadata(
    dataset: str,
    split: str,
    experiment: str,
    model: str,
    sampling_params: SamplingParams,
    num_samples: int,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "split": split,
        "experiment": experiment,
        "stage": "inference",
        "model": model,
        "num_samples": num_samples,
        "sampling": {
            "max_tokens": sampling_params.max_tokens,
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "n": sampling_params.n,
            "seed": sampling_params.seed,
        },
        "vllm_version": package_version("vllm"),
        "transformers_version": package_version("transformers"),
        "git_sha": read_git_sha(),
        "timestamp": now_iso(),
    }


# Refer to https://github.com/mathllm/MATH-V/blob/main/models/Qwen-VL.py
def build_mathvision_prompt(sample: dict) -> str:
    question = sample["question"]
    options = ""
    if len(sample["options"]) > 0:
        assert len(sample["options"]) == 5, sample
        if "".join(sample["options"]) != "ABCDE":
            options = (
                f"(A) {sample['options'][0]}\n"
                f"(B) {sample['options'][1]}\n"
                f"(C) {sample['options'][2]}\n"
                f"(D) {sample['options'][3]}\n"
                f"(E) {sample['options'][4]}\n"
            )
    return (
        'Please solve the problem step by step and put your answer in one "\\boxed{}". '
        "If it is a multiple choice question, "
        'only one letter is allowed in the "\\boxed{}".\n'
        f"{question}\n{options}"
    )


def build_mathvision_messages(sample: dict, image_path: Path) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": build_mathvision_prompt(sample)},
            ],
        }
    ]


def build_mathvision_requests(
    split: str,
) -> tuple[list[TextPrompt], list[dict[str, Any]]]:
    processed_jsonl = mathvision_processed_jsonl(split)
    image_dir = mathvision_image_dir(split)

    processor = AutoProcessor.from_pretrained(QWEN25_VL_3B_MODEL)
    requests: list[TextPrompt] = []
    records: list[dict[str, Any]] = []

    for sample in read_jsonl(processed_jsonl):
        image_path = image_dir / f"{sample['id']}.jpg"
        with Image.open(image_path) as image_file:
            image = image_file.copy()
        messages = build_mathvision_messages(sample, image_path)
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        requests.append(
            {
                "prompt": text,
                "multi_modal_data": {"image": image},
            }
        )
        record = sample
        record["prompt"] = text
        records.append(record)
    return requests, records


def run_mathvision_inference(
    split: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    num_candidates: int,
    seed: int,
    experiment: str,
) -> None:
    llm = LLM(model=QWEN25_VL_3B_MODEL)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=num_candidates,
        seed=seed,
    )

    requests, records = build_mathvision_requests(split)
    outputs = llm.generate(requests, sampling_params)

    for record, output in zip(records, outputs, strict=True):
        candidates = [{"response": candidate.text} for candidate in output.outputs]
        record["response"] = candidates[0]["response"]
        if num_candidates > 1:
            record["candidates"] = candidates
    write_jsonl(predictions_path("mathvision", split, experiment), records)
    write_json(
        meta_path("mathvision", split, experiment),
        build_inference_metadata(
            "mathvision",
            split,
            experiment,
            QWEN25_VL_3B_MODEL,
            sampling_params,
            len(records),
        ),
        indent=2,
    )


def build_vstar_prompt(sample: dict) -> str:
    question = sample["question"].replace(
        "\nAnswer with the option's letter from the given choices directly.",
        "",
    )
    return (
        "Please solve the problem step by step and put your final answer in "
        'one "\\boxed{}". '
        'Only one letter from A, B, C, and D is allowed in the "\\boxed{}".\n'
        f"{question}"
    )


def build_vstar_messages(sample: dict, image_path: Path) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": build_vstar_prompt(sample)},
            ],
        }
    ]


def build_vstar_requests() -> tuple[list[TextPrompt], list[dict[str, Any]]]:
    processed_jsonl = vstar_processed_jsonl()
    processed_dir = vstar_processed_dir()

    processor = AutoProcessor.from_pretrained(QWEN25_VL_3B_MODEL)
    requests: list[TextPrompt] = []
    records: list[dict[str, Any]] = []

    for sample in read_jsonl(processed_jsonl):
        image_path = processed_dir / sample["image"]
        with Image.open(image_path) as image_file:
            image = image_file.copy()
        messages = build_vstar_messages(sample, image_path)
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        requests.append(
            {
                "prompt": text,
                "multi_modal_data": {"image": image},
            }
        )
        record = sample
        record["prompt"] = text
        records.append(record)
    return requests, records


def run_vstar_inference(
    max_tokens: int,
    temperature: float,
    top_p: float,
    num_candidates: int,
    seed: int,
    experiment: str,
) -> None:
    llm = LLM(model=QWEN25_VL_3B_MODEL)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=num_candidates,
        seed=seed,
    )

    requests, records = build_vstar_requests()
    outputs = llm.generate(requests, sampling_params)

    for record, output in zip(records, outputs, strict=True):
        candidates = [{"response": candidate.text} for candidate in output.outputs]
        record["response"] = candidates[0]["response"]
        if num_candidates > 1:
            record["candidates"] = candidates
    write_jsonl(predictions_path("vstar", VSTAR_SPLIT, experiment), records)
    write_json(
        meta_path("vstar", VSTAR_SPLIT, experiment),
        build_inference_metadata(
            "vstar",
            VSTAR_SPLIT,
            experiment,
            QWEN25_VL_3B_MODEL,
            sampling_params,
            len(records),
        ),
        indent=2,
    )
