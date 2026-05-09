# MLLM Eval Pipeline

[English](../README.md) | 简体中文

本项目是一个多模态大模型评测流水线，基于开源数据集 **MathVision** 和 **V\***。数据预处理、
[vLLM](https://github.com/vllm-project/vllm) 的 CoT 推理、规则化答案解析、指标计算、三种 verifier 策略（另含 oracle 上限分析）。

模型：`Qwen/Qwen2.5-VL-3B-Instruct`。

完整结果见
**[`experiments.md`](experiments.md)**。

## 核心数据

MathVision（max_tokens=2048，确定性单次运行）：

| 策略 | testmini (304) | test (3040) |
| --- | ---: | ---: |
| Greedy (`temp=0`) | 76 / 304 = 25.00% | 685 / 3040 = 22.53% |
| k=8 first candidate | 58 / 304 = 19.08% | 654 / 3040 = 21.51% |
| k=8 `rule` | 61 / 304 = 20.07% | 697 / 3040 = 22.93% |
| k=8 `majority` | 78 / 304 = 25.66% | 843 / 3040 = 27.73% |
| k=8 `majority+rule` | 78 / 304 = 25.66% | 846 / 3040 = 27.83% |
| k=8 oracle | 203 / 304 = 66.78% | 1829 / 3040 = 60.16% |

V\*：greedy 122 / 191 = **63.87%**（direct_attributes 66.96%，relative_position 59.21%）。

MathVision full split（3040 题）greedy 22.53% 与 Qwen2.5-VL-3B-Instruct
官方 MathVision 报告
[21.2%](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) 一致；
testmini（304 题）官方未单独报告。

## 环境

需要 Python ≥ 3.10。项目使用 [`uv`](https://github.com/astral-sh/uv) 管理依赖。

```bash
# 如未安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆 + 同步依赖到 .venv
git clone https://github.com/soongfs/mllm-eval-pipeline.git
cd mllm-eval-pipeline
uv sync
```

后续所有命令都通过 `uv run` 执行，它会自动激活项目 `.venv`。

## 一键完整流水线

`run-all` 子命令一条龙：下载 → 预处理 → 推理 → 解析 → 评测 → verifier
（rule / majority / majority+rule）。

MathVision testmini —— Greedy baseline：

```bash
uv run mllm-eval run-all mathvision --split testmini \
  --experiment greedy_2048 \
  --num-candidates 1 --temperature 0.0 --top-p 1.0
# -> mathvision/testmini/greedy_2048:  76/304=25.00%
```

MathVision testmini —— k=8 多候选 + 四种 verifier（包括 `oracle` 上限）：

```bash
uv run mllm-eval run-all mathvision --split testmini --experiment k8_2048
# -> mathvision/testmini/k8_2048:                 58/304=19.08%
# -> mathvision/testmini/k8_2048.rule:            61/304=20.07%
# -> mathvision/testmini/k8_2048.majority:        78/304=25.66%
# -> mathvision/testmini/k8_2048.majority+rule:   78/304=25.66%
# -> mathvision/testmini/k8_2048.oracle:          203/304=66.78%  (上限)
```

MathVision 完整 test split：

```bash
uv run mllm-eval run-all mathvision --split test --experiment k8_2048
```

V\*：

```bash
uv run mllm-eval run-all vstar --experiment vstar_2048 \
  --num-candidates 1 --temperature 0.0 --top-p 1.0
# -> vstar/vstar_2048:  122/191=63.87%
```

完整参数看 `uv run mllm-eval run-all --help`（`--skip`、`--verifiers`、
`--seed` 等）。每个阶段也有单独子命令 —— 见 `uv run mllm-eval --help`。

每个实验的产物自成目录 `outputs/<dataset>/<split>/<experiment>/`，包含
`predictions.jsonl`、`parsed.jsonl`、`result.json`、`meta.json`（模型 ID、
采样参数、vLLM 版本、git SHA、时间戳）。参考执行日志位于
`docs/logs/<dataset>/<split>/`。

## 可复现性

vLLM 默认的 v1 多进程调度器即使在 `temperature=0` 也是非确定性的。通过 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 强制确定性调度（由
`src/mllm_eval_pipeline/inference.py` 自动设置）。验证：同一配置连跑两次
`greedy_2048`，predictions 完全一致——详见
[experiments.md § Inference determinism](experiments.md#inference-determinism)
和 [vLLM 官方可复现性文档](https://docs.vllm.ai/en/v0.19.1/usage/reproducibility/)。

## 目录结构

```
src/mllm_eval_pipeline/
  dataset.py      # MathVision / V* 下载 + 预处理
  inference.py    # vLLM 多候选推理（确定性）+ 元数据
  parser.py       # CoT 答案抽取（boxed → phrase → 选项字母 → 尾部数字）
  metrics.py      # 归一化 + tuple 求值 + latex2sympy 符号比较
  verifier.py     # rule / majority / majority+rule / oracle 选择器
  cli.py          # argparse 入口，含 `run-all`
  paths.py        # 所有磁盘路径的帮助函数
scripts/
  analyze_verifier.py         # verifier 变迁分析（wrong↔right）
  export_verifier_cases.py    # 导出 bad case JSONL 供人工检查
docs/
  experiments.md  # 完整结果、Findings、可复现性
  logs/           # 按 dataset/split 组织的参考执行日志
outputs/<dataset>/<split>/<experiment>/
  predictions.jsonl parsed.jsonl result.json meta.json
data/processed/<dataset>/<split>/
  samples.jsonl  images/
```

一个实验 = 一个目录。Verifier 派生实验命名为 `<base>.<verifier>`
（如 `k8_2048.majority`），其 `meta.json` 记录 base 实验名。

## 致谢

- `parser.py` / `metrics.py` 的答案抽取与打分逻辑参考
  [MathLLMs/MathVision](https://github.com/mathllm/MATH-V) 官方评测代码
- V\* benchmark 来自
  [penghao-wu/vstar](https://github.com/penghao-wu/vstar)
  （数据托管在 [`craigwu/vstar_bench`](https://huggingface.co/datasets/craigwu/vstar_bench)）
- vLLM 可复现性方案参考
  [vLLM 官方可复现性文档](https://docs.vllm.ai/en/v0.19.1/usage/reproducibility/)
