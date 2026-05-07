# Experiment Notes

This document records MathVision results under different inference settings and
the lightweight verifier mechanism.

## MathVision

### Setup

| Item | Setting |
| --- | --- |
| Model | Qwen/Qwen2.5-VL-3B-Instruct |
| Dataset | MathVision |
| Split | testmini / test |
| Prompt | MathVision official CoT prompt |
| Parser | MathVision-style answer extraction |
| Baseline decoding | temperature=0.0, top_p=1.0, num_candidates=1 |
| Multi-candidate decoding | temperature=0.7, top_p=0.9, num_candidates=3, seed=42 |
| Rule verifier | Select the highest-scored candidate from k=3 responses |

### max_tokens=1024

#### testmini

| Variant | Accuracy | Parsed | Notes |
| --- | ---: | ---: | --- |
| Greedy baseline | 70/304=23.03% | 277/304 | `num_candidates=1` |
| k=3 first candidate | 53/304=17.43% | 288/304 | First sampled candidate |
| k=3 rule verifier | 63/304=20.72% | 302/304 | Rule-selected candidate |
| k=3 oracle | 121/304=39.8% | N/A | Upper bound, not a real inference method |

Verifier transition analysis:

| Metric | Value |
| --- | ---: |
| Changed | 93/304 |
| Wrong -> Right | 17 |
| Right -> Wrong | 7 |
| Net gain | +10 |

#### test

| Variant | Accuracy | Parsed | Notes |
| --- | ---: | ---: | --- |
| Greedy baseline | 592/3040=19.47% | 2700/3040 | `num_candidates=1` |
| k=3 first candidate | 565/3040=18.59% | 2809/3040 | First sampled candidate |
| k=3 rule verifier | 645/3040=21.22% | 3011/3040 | Rule-selected candidate |
| k=3 oracle | 1133/3040=37.27% | N/A | Upper bound, not a real inference method |

Verifier transition analysis:

| Metric | Value |
| --- | ---: |
| Changed | 982/3040 |
| Wrong -> Right | 152 |
| Right -> Wrong | 72 |
| Net gain | +80 |

### max_tokens=2048

#### testmini

| Variant | Accuracy | Parsed | Notes |
| --- | ---: | ---: | --- |
| Greedy baseline | 63/304=20.72% | 285/304 | `num_candidates=1` |
| k=3 first candidate | 54/304=17.76% | 296/304 | First sampled candidate |
| k=3 rule verifier | 61/304=20.07% | 304/304 | Rule-selected candidate |
| k=3 oracle | 128/304=42.11% | N/A | Upper bound, not a real inference method |

Verifier transition analysis:

| Metric | Value |
| --- | ---: |
| Changed | 85/304 |
| Wrong -> Right | 13 |
| Right -> Wrong | 6 |
| Net gain | +7 |

#### test

| Variant | Accuracy | Parsed | Notes |
| --- | ---: | ---: | --- |
| Greedy baseline | 632/3040=20.79% | 2800/3040 | `num_candidates=1` |
| k=3 first candidate | 596/3040=19.61% | 2919/3040 | First sampled candidate |
| k=3 rule verifier | 636/3040=20.92% | 3037/3040 | Rule-selected candidate |
| k=3 oracle | 1173/3040=38.59% | N/A | Upper bound, not a real inference method |

Verifier transition analysis:

| Metric | Value |
| --- | ---: |
| Changed | 948/3040 |
| Wrong -> Right | 134 |
| Right -> Wrong | 94 |
| Net gain | +40 |

### Output layout

Each experiment lives in its own directory:

```
outputs/<dataset>/<split>/<experiment>/
  predictions.jsonl
  parsed.jsonl
  result.json
  meta.json
  verifier_cases_<case_type>.jsonl   # only for verifier-derived experiments
```

Verifier-derived experiments are named `<base>.<verifier>` (e.g.
`k3_1024.rule`) and `meta.json` records the base experiment.

### Commands

Greedy baseline:

```bash
uv run mllm-eval infer mathvision --split testmini --max-tokens 1024 --experiment greedy_1024
uv run mllm-eval parse mathvision --split testmini --experiment greedy_1024
uv run mllm-eval evaluate mathvision --split testmini --experiment greedy_1024
```

Multi-candidate sampling:

```bash
uv run mllm-eval infer mathvision --split testmini \
  --max-tokens 1024 \
  --temperature 0.7 \
  --top-p 0.9 \
  --num-candidates 3 \
  --seed 42 \
  --experiment k3_1024

uv run mllm-eval parse mathvision --split testmini --experiment k3_1024
uv run mllm-eval evaluate mathvision --split testmini --experiment k3_1024
```

Rule verifier (writes to `outputs/mathvision/testmini/k3_1024.rule/`):

```bash
uv run mllm-eval verify mathvision --split testmini --base k3_1024
uv run mllm-eval parse mathvision --split testmini --experiment k3_1024.rule
uv run mllm-eval evaluate mathvision --split testmini --experiment k3_1024.rule
uv run mllm-eval analyze-verifier mathvision --split testmini --experiment k3_1024.rule
```

For `max_tokens=2048`, replace `--max-tokens 1024` with `--max-tokens 2048`
and use a different experiment name such as `k3_2048`.

### Notes

- Oracle accuracy is only an upper-bound analysis. It uses ground truth and is
  not a deployable inference method.
- Rule verifier uses multiple sampled candidates, so it is not directly
  comparable to a single-pass benchmark setting.
- Improvements may come from decoding settings, reduced truncation, answer
  formatting, candidate diversity, or reranking quality.
