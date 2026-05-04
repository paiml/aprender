# §49 MODEL-2 strategy pivot — empirical evidence

**Date:** 2026-05-04
**Author:** Loop iteration after operator pivot directive

## 1. Live evidence: MODEL-2 from-scratch hits the same ceiling at 500 vs 80,000 steps

Fresh 500-step `apr pretrain --mode from-scratch --device cuda` run, 2026-05-04:

```
Run Result: OK CONVERGED  final val_loss=9.7255 after 5 epoch(s)
  Steps recorded: 500
  Epochs recorded: 5
```

Compare to memory `project_2026_04_27_4x_corpus_memorization_disproof.md` (80K-step LR-budget falsification on the same 4× corpus):

```
val_loss=9.7507 epoch 4
```

**Within 0.026 of each other across 160× difference in step count.** The ceiling is not step-budget-limited.

## 2. Implementation pre-conditions verified

| Pre-condition | Path | Status |
|---|------|--------|
| Qwen2.5-Coder-0.5B-Instruct cache | `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/ea3f2471cf1b1f0db85067f1ef93848e38e88c25/` | ✅ 950 MB, has `model.safetensors` + `config.json` + `tokenizer.json` |
| `apr convert` HF safetensors → APR fp16 | `apr convert <model.safetensors> --quantize fp16 -o <out.apr>` | ✅ verified live: 290 tensors, 942 MiB → 942 MiB, 5.1 sec |
| Output | `/mnt/nvme-raid0/models/qwen2.5-coder-0.5b-instruct-fp16.apr` | ✅ exists, 942 MiB |
| RTX 4090 + cuBLAS + custom PTX backward | sm_89, no Blackwell JIT bug | ✅ verified live (smoke run completed) |
| `apr pretrain --device cuda` + custom backward kernels | sm_89 cuda_compute_capability | ✅ verified live (24 transformer blocks uploaded, backward kernels pre-warmed) |
| Codeparrot+CSN-Python tokenized corpus | `/mnt/nvme-raid0/data/codeparrot-python-permissive-shards/` | ✅ 565M tokens, manifest.json present |

## 3. Implementation gap

**The remaining engineering work is one missing flag**: `apr pretrain` lacks `--init <model.apr>` to load weights instead of random init. From `apr pretrain --help`, no `init`/`resume`/`checkpoint`/`load`/`from-checkpoint` option exists.

§49.6 step 4 covers this: "Wire `--init <model.apr>` flag into `apr pretrain` → loads weights instead of random init (~50 LOC)".

That is the load-bearing implementation PR. It is not authored in this commit.

## 4. Industry precedent for the strategy

| Production small-LM | Param count | Initialization |
|---|---|---|
| StableCode-3B | 3B | StableLM (general LM) |
| Qwen2.5-Coder-0.5B | 0.5B | Qwen2.5 (general LM) |
| Qwen2.5-Coder-7B | 7B | Qwen2.5-7B |
| DeepSeek-Coder-1.3B | 1.3B | DeepSeek-LLM |
| StarCoder2-3B | 3B | from-scratch on **3.3T tokens** |
| SmolLM-360M | 360M | from-scratch on **1T tokens** |

Two patterns:
- **Pretrained-init** (most production code-LMs): cheaper, faster, hits target loss
- **From-scratch** (StarCoder, SmolLM): only works at 1T+ tokens

MODEL-2 attempted from-scratch on 565M tokens — **3× to 6× too few** to match the from-scratch precedents. Hence §49's pivot to pretrained-init.

## 5. What this commit ships

1. **Spec v2.93.0 → v2.94.0** with §49 amendment — strategic record
2. **This evidence file** — empirical justification

What this commit does NOT ship:
- `apr pretrain --init` flag (next PR)
- Live fine-tune run (gated on the flag)
- val_loss < 9.38 evidence (gated on the run)

MODEL-2 ship % stays at 57% until the live fine-tune produces empirical evidence.
