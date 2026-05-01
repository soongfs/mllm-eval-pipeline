import argparse

from mllm_eval_pipeline.paths import MATHVISION_DEFAULT_SPLIT, MATHVISION_SPLITS


def add_split_argument(command_parser: argparse.ArgumentParser) -> None:
    command_parser.add_argument(
        "--split",
        choices=MATHVISION_SPLITS,
        default=MATHVISION_DEFAULT_SPLIT,
        help="MathVision split to use",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MLLM evaluation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download MathVision")
    add_split_argument(download_parser)

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess MathVision",
    )
    add_split_argument(preprocess_parser)

    infer_parser = subparsers.add_parser(
        "infer",
        help="Run Qwen2.5-VL inference on MathVision",
    )
    add_split_argument(infer_parser)
    infer_parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of generated tokens per sample",
    )

    parse_parser = subparsers.add_parser(
        "parse",
        help="Parse answers from model responses",
    )
    add_split_argument(parse_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate parsed MathVision predictions",
    )
    add_split_argument(evaluate_parser)

    args = parser.parse_args()
    if args.command == "download":
        from mllm_eval_pipeline.dataset import download_mathvision

        download_mathvision(args.split)
    elif args.command == "preprocess":
        from mllm_eval_pipeline.dataset import preprocess_mathvision

        preprocess_mathvision(args.split)
    elif args.command == "infer":
        from mllm_eval_pipeline.inference import run_mathvision_inference

        run_mathvision_inference(args.split, args.max_tokens)
    elif args.command == "parse":
        from mllm_eval_pipeline.parser import parse_predictions

        parse_predictions(args.split)
    elif args.command == "evaluate":
        from mllm_eval_pipeline.metrics import evaluate_mathvision

        evaluate_mathvision(args.split)


if __name__ == "__main__":
    main()
