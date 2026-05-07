import argparse

from mllm_eval_pipeline.analysis import CASE_TYPES
from mllm_eval_pipeline.paths import MATHVISION_DEFAULT_SPLIT, MATHVISION_SPLITS

DATASETS = ["mathvision", "vstar"]


def add_split_argument(command_parser: argparse.ArgumentParser) -> None:
    command_parser.add_argument(
        "--split",
        choices=MATHVISION_SPLITS,
        default=MATHVISION_DEFAULT_SPLIT,
        help="MathVision split to use",
    )


def add_experiment_argument(command_parser: argparse.ArgumentParser) -> None:
    command_parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment name (output directory under outputs/<dataset>/<split>/)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MLLM evaluation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download a dataset")
    download_parser.add_argument("dataset", choices=DATASETS)
    add_split_argument(download_parser)

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess a dataset",
    )
    preprocess_parser.add_argument("dataset", choices=DATASETS)
    add_split_argument(preprocess_parser)

    infer_parser = subparsers.add_parser(
        "infer",
        help="Run inference on MathVision or V*",
    )
    infer_parser.add_argument("dataset", choices=DATASETS)
    add_split_argument(infer_parser)
    infer_parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of generated tokens per sample",
    )
    infer_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    infer_parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling probability",
    )
    infer_parser.add_argument(
        "--num-candidates",
        type=int,
        default=1,
        help="Number of generated candidates per sample",
    )
    infer_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    add_experiment_argument(infer_parser)

    parse_parser = subparsers.add_parser(
        "parse",
        help="Parse answers from model responses",
    )
    parse_parser.add_argument("dataset", choices=DATASETS)
    add_split_argument(parse_parser)
    add_experiment_argument(parse_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate parsed predictions",
    )
    evaluate_parser.add_argument("dataset", choices=DATASETS)
    add_split_argument(evaluate_parser)
    add_experiment_argument(evaluate_parser)

    verify_parser = subparsers.add_parser(
        "verify",
        help="Select candidates with a lightweight rule verifier",
    )
    verify_parser.add_argument("dataset", choices=["mathvision"])
    add_split_argument(verify_parser)
    verify_parser.add_argument(
        "--base",
        required=True,
        help="Base experiment to verify",
    )
    verify_parser.add_argument(
        "--experiment",
        default=None,
        help="Output experiment name (default: <base>.rule)",
    )

    analyze_verifier_parser = subparsers.add_parser(
        "analyze-verifier",
        help="Analyze lightweight verifier reranking results",
    )
    analyze_verifier_parser.add_argument("dataset", choices=["mathvision"])
    add_split_argument(analyze_verifier_parser)
    add_experiment_argument(analyze_verifier_parser)

    export_verifier_cases_parser = subparsers.add_parser(
        "export-verifier-cases",
        help="Export verifier reranking cases as JSONL",
    )
    export_verifier_cases_parser.add_argument("dataset", choices=["mathvision"])
    add_split_argument(export_verifier_cases_parser)
    add_experiment_argument(export_verifier_cases_parser)
    export_verifier_cases_parser.add_argument(
        "--case-type",
        choices=CASE_TYPES,
        default="changed",
        help="Verifier case type to export",
    )
    export_verifier_cases_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of cases to export",
    )

    args = parser.parse_args()
    if args.command == "download":
        from mllm_eval_pipeline.dataset import download_mathvision, download_vstar

        if args.dataset == "mathvision":
            download_mathvision(args.split)
        elif args.dataset == "vstar":
            download_vstar()
    elif args.command == "preprocess":
        from mllm_eval_pipeline.dataset import preprocess_mathvision, preprocess_vstar

        if args.dataset == "mathvision":
            preprocess_mathvision(args.split)
        elif args.dataset == "vstar":
            preprocess_vstar()
    elif args.command == "infer":
        from mllm_eval_pipeline.inference import (
            run_mathvision_inference,
            run_vstar_inference,
        )

        if args.dataset == "mathvision":
            run_mathvision_inference(
                args.split,
                args.max_tokens,
                args.temperature,
                args.top_p,
                args.num_candidates,
                args.seed,
                args.experiment,
            )
        elif args.dataset == "vstar":
            run_vstar_inference(
                args.max_tokens,
                args.temperature,
                args.top_p,
                args.num_candidates,
                args.seed,
                args.experiment,
            )
    elif args.command == "parse":
        from mllm_eval_pipeline.parser import parse_predictions

        parse_predictions(args.dataset, args.split, args.experiment)
    elif args.command == "evaluate":
        from mllm_eval_pipeline.metrics import evaluate_mathvision, evaluate_vstar

        if args.dataset == "mathvision":
            evaluate_mathvision(args.split, args.experiment)
        elif args.dataset == "vstar":
            evaluate_vstar(args.experiment)
    elif args.command == "verify":
        from mllm_eval_pipeline.verifier import (
            VERIFIER_RULE,
            verify_mathvision_predictions,
        )

        experiment = args.experiment or f"{args.base}.{VERIFIER_RULE}"
        if args.dataset == "mathvision":
            verify_mathvision_predictions(args.split, args.base, experiment)
    elif args.command == "analyze-verifier":
        from mllm_eval_pipeline.analysis import analyze_mathvision_verifier

        if args.dataset == "mathvision":
            analyze_mathvision_verifier(args.split, args.experiment)
    elif args.command == "export-verifier-cases":
        from mllm_eval_pipeline.analysis import export_mathvision_verifier_cases

        if args.dataset == "mathvision":
            export_mathvision_verifier_cases(
                args.split,
                args.experiment,
                args.case_type,
                args.limit,
            )


if __name__ == "__main__":
    main()
