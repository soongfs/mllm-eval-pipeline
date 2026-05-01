# TODO Implement model inference
# Qwen/Qwen2.5-VL-3B-Instruct
# LLaVA


from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.inputs import TextPrompt

from mllm_eval_pipeline.io import read_jsonl, write_jsonl
from mllm_eval_pipeline.paths import (
    QWEN25_VL_3B_MODEL,
    mathvision_image_dir,
    mathvision_predictions_jsonl,
    mathvision_processed_jsonl,
)


# Refer to https://github.com/mathllm/MATH-V/blob/main/models/Qwen-VL.py
def build_mathvision_prompt(sample: dict) -> str:
    question = sample["question"]
    options = ""
    if len(sample["options"]) > 0:
        assert len(sample["options"]) == 5, sample
        if "".join(sample["options"]) != "ABCDE":
            options = (
                f"(A) {sample['options'][0]}\n",
                f"(B) {sample['options'][1]}\n",
                f"(C) {sample['options'][2]}\n",
                f"(D) {sample['options'][3]}\n",
                f"(E) {sample['options'][4]}\n",
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
        records.append(sample)
    return requests, records


def run_mathvision_inference(split: str) -> None:
    llm = LLM(model=QWEN25_VL_3B_MODEL)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
    )

    requests, records = build_mathvision_requests(split)
    outputs = llm.generate(requests, sampling_params)

    for record, output in zip(records, outputs, strict=True):
        record["response"] = output.outputs[0].text
    write_jsonl(mathvision_predictions_jsonl(split), records)
