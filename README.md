# MLLM Eval Pipeline

English | [简体中文](docs/README.zh-CN.md)

A multimodal LLM evaluation pipeline over the open-source benchmarks
**MathVision** and **V\***. Data preprocessing, CoT inference via
[vLLM](https://github.com/vllm-project/vllm), rule-based answer parsing,
metric computation, and three verifier strategies (plus oracle upper-bound analysis).

Model: `Qwen/Qwen2.5-VL-3B-Instruct`.

Full results: **[`docs/experiments.md`](docs/experiments.md)**.

## Headline numbers

MathVision (max_tokens=2048, deterministic single run):

| Strategy | testmini (304) | test (3040) |
| --- | ---: | ---: |
| Greedy (`temp=0`) | 76 / 304 = 25.00% | 685 / 3040 = 22.53% |
| k=8 first candidate | 58 / 304 = 19.08% | 654 / 3040 = 21.51% |
| k=8 `rule` | 61 / 304 = 20.07% | 697 / 3040 = 22.93% |
| k=8 `majority` | 78 / 304 = 25.66% | 843 / 3040 = 27.73% |
| k=8 `majority+rule` | 78 / 304 = 25.66% | 846 / 3040 = 27.83% |
| k=8 oracle | 203 / 304 = 66.78% | 1829 / 3040 = 60.16% |

V\*: greedy 122 / 191 = **63.87%** (direct_attributes 66.96%, relative_position 59.21%).

Test greedy 22.53% on MathVision full (3040 items) aligns with the
published Qwen2.5-VL-3B-Instruct MathVision number
([21.2%](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct));
testmini (304 items) is not separately reported by the Qwen team.

## Environment

Requires Python ≥ 3.10. Dependency management uses
[`uv`](https://github.com/astral-sh/uv).

```bash
# Install uv if missing
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and sync dependencies into .venv
git clone https://github.com/soongfs/mllm-eval-pipeline.git
cd mllm-eval-pipeline
uv sync
```

All commands below use `uv run` which activates the project `.venv`
automatically.

## One-command pipeline

The `run-all` subcommand chains download → preprocess → infer → parse →
evaluate → verify (rule / majority / majority+rule).

MathVision testmini — greedy baseline:

```bash
uv run mllm-eval run-all mathvision --split testmini \
  --experiment greedy_2048 \
  --num-candidates 1 --temperature 0.0 --top-p 1.0
# -> mathvision/testmini/greedy_2048:  76/304=25.00%
```

MathVision testmini — k=8 with all four verifiers (including `oracle` upper bound):

```bash
uv run mllm-eval run-all mathvision --split testmini --experiment k8_2048
# -> mathvision/testmini/k8_2048:                 58/304=19.08%
# -> mathvision/testmini/k8_2048.rule:            61/304=20.07%
# -> mathvision/testmini/k8_2048.majority:        78/304=25.66%
# -> mathvision/testmini/k8_2048.majority+rule:   78/304=25.66%
# -> mathvision/testmini/k8_2048.oracle:          203/304=66.78%  (upper bound)
```

MathVision full test split:

```bash
uv run mllm-eval run-all mathvision --split test --experiment k8_2048
```

V\*:

```bash
uv run mllm-eval run-all vstar --experiment vstar_2048 \
  --num-candidates 1 --temperature 0.0 --top-p 1.0
# -> vstar/vstar_2048:  122/191=63.87%
```

See `uv run mllm-eval run-all --help` for every flag (`--skip`,
`--verifiers`, `--seed`, etc.). Individual stages are also available as
dedicated subcommands — see `uv run mllm-eval --help`.

Each experiment's products live at
`outputs/<dataset>/<split>/<experiment>/` with `predictions.jsonl`,
`parsed.jsonl`, `result.json`, and `meta.json` (model ID, sampling
params, vLLM version, git SHA, timestamp). Reference execution logs
live at `docs/logs/<dataset>/<split>/`.

## Reproducibility

vLLM's default v1 multiprocessing scheduler is non-deterministic even
at `temperature=0`. Force deterministic offline scheduling via
`VLLM_ENABLE_V1_MULTIPROCESSING=0` (set automatically by
`src/mllm_eval_pipeline/inference.py`). Verified deterministic output
(`predictions.jsonl` byte-identical across two independent runs) — see
[docs/experiments.md § Inference determinism](docs/experiments.md#inference-determinism)
and the
[vLLM reproducibility guide](https://docs.vllm.ai/en/v0.19.1/usage/reproducibility/).

## Layout

```
src/mllm_eval_pipeline/
  dataset.py      # MathVision / V* download + preprocess
  inference.py    # vLLM multi-candidate inference (deterministic) + meta
  parser.py       # CoT answer extraction (boxed → phrase → letter → numeric tail)
  metrics.py      # normalize + tuple eval + latex2sympy symbolic compare
  verifier.py     # rule / majority / majority+rule / oracle selectors
  cli.py          # argparse entry, incl. `run-all`
  paths.py        # centralized path helpers
scripts/
  analyze_verifier.py         # verifier transition analysis (wrong↔right)
  export_verifier_cases.py    # export bad cases as JSONL for inspection
docs/
  experiments.md  # full results, findings, reproducibility
  logs/           # reference execution logs per dataset/split
outputs/<dataset>/<split>/<experiment>/
  predictions.jsonl parsed.jsonl result.json meta.json
data/processed/<dataset>/<split>/
  samples.jsonl  images/
```

One experiment = one directory. Verifier-derived experiments are named
`<base>.<verifier>` (e.g. `k8_2048.majority`) and their `meta.json`
records the base experiment.

## Acknowledgments

- MathVision benchmark from
  [mathllm/MATH-V](https://github.com/mathllm/MATH-V)
  (data hosted at [`MathLLMs/MathVision`](https://huggingface.co/datasets/MathLLMs/MathVision)).
  Answer-extraction and scoring logic in `parser.py` / `metrics.py`
  reference the official evaluation code.
- V\* benchmark from
  [penghao-wu/vstar](https://github.com/penghao-wu/vstar)
  (data hosted at [`craigwu/vstar_bench`](https://huggingface.co/datasets/craigwu/vstar_bench)).
- vLLM reproducibility method from the
  [vLLM reproducibility guide](https://docs.vllm.ai/en/v0.19.1/usage/reproducibility/).
