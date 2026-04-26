import argparse

from mllm_eval_pipeline.dataset import (
    download_mathvision,
    preprocess_mathvision,
)
from mllm_eval_pipeline.inference import run_mathvision_inference
from mllm_eval_pipeline.parser import parse_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="MLLM evaluation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("download", help="Download MathVision-testmini")
    subparsers.add_parser("preprocess", help="Preprocess MathVision-testmini")
    subparsers.add_parser("infer", help="Run Qwen2.5-VL inference on MathVision")
    subparsers.add_parser("parse", help="Parse answers from model responses")

    args = parser.parse_args()
    if args.command == "download":
        download_mathvision()
    elif args.command == "preprocess":
        preprocess_mathvision()
    elif args.command == "infer":
        run_mathvision_inference()
    elif args.command == "parse":
        parse_predictions()


if __name__ == "__main__":
    main()
