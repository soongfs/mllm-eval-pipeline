# Experiment Notes

Every number here comes from a **deterministic single run** with
`VLLM_ENABLE_V1_MULTIPROCESSING=0`. See *Inference determinism* below.

## Setup

| Item | Setting |
| --- | --- |
| Model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Inference engine | vLLM 0.19.1 with `VLLM_ENABLE_V1_MULTIPROCESSING=0` |
| Datasets | MathVision (testmini 304 / test 3040); V\* (191) |
| Prompt | MathVision official CoT template |
| Parser | Boxed → phrase → choice-letter → numeric-tail (4-layer fallback) |
| Metric | Exact → symmetric normalize → tuple → latex2sympy symbolic |
| Verifiers | `rule`, `majority`, `majority+rule`, `oracle` |
| Greedy decoding | `temperature=0.0, top_p=1.0, num_candidates=1` |
| Multi-candidate | `temperature=0.7, top_p=0.9, num_candidates=8, seed=42` |
| `max_tokens` | **2048** throughout |

## Inference determinism

vLLM's default v1 multiprocessing scheduler is non-deterministic even at
`temperature=0`. Setting `VLLM_ENABLE_V1_MULTIPROCESSING=0` forces
deterministic offline scheduling per the official
[vLLM reproducibility guide](https://docs.vllm.ai/en/v0.19.1/usage/reproducibility/).
Applied via `os.environ.setdefault` at `src/mllm_eval_pipeline/inference.py`.
Verified: two independent `greedy_2048` runs produce byte-identical
`predictions.jsonl` (`jq -c` diff = empty).

## MathVision results (max_tokens=2048)

| Strategy | testmini (304) | test (3040) |
| --- | ---: | ---: |
| Greedy (`temperature=0`) | **76 / 304 = 25.00%** | 685 / 3040 = 22.53% |
| k=8 first candidate | 58 / 304 = 19.08% | 654 / 3040 = 21.51% |
| k=8 `rule` | 61 / 304 = 20.07% | 697 / 3040 = 22.93% |
| k=8 `majority` | **78 / 304 = 25.66%** | 843 / 3040 = 27.73% |
| k=8 `majority+rule` | 78 / 304 = 25.66% | **846 / 3040 = 27.83%** |
| k=8 `oracle` (upper bound) | **203 / 304 = 66.78%** | **1829 / 3040 = 60.16%** |

Parser success rate: greedy testmini 287/304 (94.4%), k=8 first-pass 291/304 (95.7%).

test split greedy 22.53% aligns with the published Qwen2.5-VL-3B-Instruct
MathVision number ([21.2%](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)).
testmini is not separately reported by the Qwen team.

## V\* results (max_tokens=2048, greedy)

| Category | Correct | Accuracy |
| --- | ---: | ---: |
| direct\_attributes | 77 / 115 | 66.96% |
| relative\_position | 45 / 76 | 59.21% |
| **all** | **122 / 191** | **63.87%** |

## Output layout

```
outputs/<dataset>/<split>/<experiment>/
  predictions.jsonl  parsed.jsonl  result.json  meta.json
```

Verifier-derived experiments are named `<base>.<verifier>` (e.g.
`k8_2048.majority`).

## Commands

```bash
# MathVision testmini greedy baseline
uv run mllm-eval run-all mathvision --split testmini \
  --experiment greedy_2048 \
  --num-candidates 1 --temperature 0.0 --top-p 1.0

# MathVision testmini k=8 + three verifiers + oracle
uv run mllm-eval run-all mathvision --split testmini --experiment k8_2048

# Same for full test split
uv run mllm-eval run-all mathvision --split test --experiment k8_2048

# V* greedy
uv run mllm-eval run-all vstar --experiment vstar_2048 \
  --num-candidates 1 --temperature 0.0 --top-p 1.0
```

## Notes

- Oracle is an upper-bound analysis, not a deployable method.
- Multi-candidate decoding is not directly comparable to single-pass
  benchmark settings because the inference compute budget differs.
- Execution logs live at `docs/logs/<dataset>/<split>/`.
