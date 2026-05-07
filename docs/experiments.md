# Experiment Notes

This document records MathVision results under different inference settings and
verifier mechanisms.

## MathVision

### Setup

| Item | Setting |
| --- | --- |
| Model | Qwen/Qwen2.5-VL-3B-Instruct |
| Dataset | MathVision |
| Split | testmini |
| Prompt | MathVision official CoT prompt |
| Parser | MathVision-style answer extraction |
| Baseline decoding | `temperature=0.0, top_p=1.0, num_candidates=1` |
| Multi-candidate decoding | `temperature=0.7, top_p=0.9, num_candidates=k, seed=42` |
| Verifiers | `rule`, `majority`, `majority+rule` |

### Inference determinism

vLLM does not guarantee bit-for-bit reproducibility even at `temperature=0.0`
with a fixed seed. KV-cache prefix reuse, batch composition, and
attention/cublas kernel selection all depend on runtime shapes, and their
floating-point paths are not invariant under reordering. Repeating a single
config produces different accuracy numbers — sometimes by several percentage
points — so **all downstream results are reported as mean ± std across
multiple runs with the same `seed=42`**. Single-run numbers are kept as
"snapshots" only and should not be cited as final results.

### Variance (testmini, max_tokens=1024, 4 runs each)

Greedy baseline (`num_candidates=1`, `temperature=0.0`):

| Run | Correct | Accuracy |
| --- | ---: | ---: |
| 1 | 61 | 20.07% |
| 2 | 70 | 23.03% |
| 3 | 82 | 26.97% |
| 4 | 61 | 20.07% |
| **mean** | **68.5** | **22.53%** |
| **std** | ±9.95 | ±3.27% |

k=8 first candidate (`num_candidates=8`, `temperature=0.7`, `top_p=0.9`):

| Run | Correct | Accuracy |
| --- | ---: | ---: |
| 1 | 58 | 19.08% |
| 2 | 57 | 18.75% |
| 3 | 58 | 19.08% |
| 4 | 59 | 19.41% |
| **mean** | **58.0** | **19.08%** |
| **std** | ±0.82 | ±0.27% |

k=8 majority voting:

| Run | Correct | Accuracy |
| --- | ---: | ---: |
| 1 | 65 | 21.38% |
| 2 | 64 | 21.05% |
| 3 | 62 | 20.39% |
| 4 | 69 | 22.70% |
| **mean** | **65.0** | **21.38%** |
| **std** | ±2.94 | ±0.97% |

`majority` beats `first` by +7.0 questions on average. Combined std ≈ 3.05,
so the gap is ~2.3σ — statistically meaningful on n=4 runs.

### Single-run snapshots

Snapshots are single-run results kept for qualitative comparison; they should
not be used as final numbers.

k=3 (`num_candidates=3`):

| Variant | Accuracy |
| --- | ---: |
| `rule` | 55/304 = 18.09% |
| `majority` | 56/304 = 18.42% |
| `majority+rule` | 59/304 = 19.41% |

k=8 (`num_candidates=8`):

| Variant | Accuracy |
| --- | ---: |
| `rule` | 60/304 = 19.74% |
| `majority` | 72/304 = 23.68% |
| `majority+rule` | 72/304 = 23.68% |

k=8 oracle (upper bound, not a deployable inference method):

| Metric | Value |
| --- | ---: |
| `first` | 56/304 = 18.42% |
| `majority` (this run) | 72/304 = 23.68% |
| `oracle` | 202/304 = 66.45% |
| Candidates changed by verifier | 141/304 |
| Wrong → Right | 31 |
| Right → Wrong | 15 |
| Right → Right | 41 |
| Wrong → Wrong | 217 |
| Net gain over `first` | +16 |

### Findings

1. **Rule verifier is saturated.** Mean k=8 first is 19.08% while single-run
   k=8 `rule` is 19.74% — inside the sampling noise band. Rule signals look
   only at a single candidate's self-features (answer presence, `\boxed{}`
   formatting, option letter validity), and at k≥3 that signal is already
   fully extracted. Larger k does not help `rule`.

2. **Majority voting is the real win at k=8.** Mean 21.38% (±0.97%) is
   +2.30 points over mean first (19.08%) and is outside the combined noise
   band. At k=3 the gap between rule, majority, and majority+rule is
   ≤1 question per run and is indistinguishable from noise — k=3 is simply
   too small for a vote to dominate.

3. **`majority+rule` collapses to `majority` at k=8.** With 8 candidates,
   nearly every item has a single largest bucket, so the rule tie-breaker
   almost never fires. The combination is retained as a safety net but does
   not contribute additional signal at k=8.

4. **Oracle headroom is 45 points.** First 19.08% → oracle 66.45%. Majority
   captures only ~5% of that gap. The remaining ~43 points is the space a
   stronger verifier (learned reward model, process-level verifier) can
   exploit.

5. **Pipeline aligns with the official baseline.** Greedy mean 22.53% is
   consistent with the published Qwen2.5-VL-3B-Instruct MathVision testmini
   number (~21%). The ±3.27% greedy variance is larger than the sampled
   `first` variance (±0.27%) — counter-intuitive but expected: `temperature=0`
   puts the whole probability mass on the argmax token at each step, so any
   kernel non-determinism that shifts logits near a tie can flip a long
   reasoning chain entirely. With sampling, local noise is averaged over the
   distribution and the final accuracy is more stable.

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
`k8_1024.majority`) and `meta.json` records the base experiment.

### Commands

Greedy baseline:

```bash
uv run mllm-eval infer mathvision --split testmini --max-tokens 1024 --experiment greedy_1024
uv run mllm-eval parse mathvision --split testmini --experiment greedy_1024
uv run mllm-eval evaluate mathvision --split testmini --experiment greedy_1024
```

Multi-candidate sampling (k=8):

```bash
uv run mllm-eval infer mathvision --split testmini \
  --max-tokens 1024 \
  --temperature 0.7 \
  --top-p 0.9 \
  --num-candidates 8 \
  --seed 42 \
  --experiment k8_1024

uv run mllm-eval parse mathvision --split testmini --experiment k8_1024
uv run mllm-eval evaluate mathvision --split testmini --experiment k8_1024
```

Verifiers (writes to `outputs/mathvision/testmini/k8_1024.<verifier>/`):

```bash
for V in rule majority majority+rule; do
  uv run mllm-eval verify mathvision --split testmini --base k8_1024 --verifier $V
  uv run mllm-eval parse mathvision --split testmini --experiment k8_1024.$V
  uv run mllm-eval evaluate mathvision --split testmini --experiment k8_1024.$V
done
uv run mllm-eval analyze-verifier mathvision --split testmini --experiment k8_1024.majority
```

### Notes

- Oracle accuracy is an upper-bound analysis, not a deployable inference
  method.
- A prompt-construction bug — multi-choice options emitted as a tuple literal
  — was fixed before the runs above; earlier numbers in the project history
  are not comparable to the ones reported here.
- Multi-candidate decoding is not directly comparable to a single-pass
  benchmark setting because the compute budget differs.
