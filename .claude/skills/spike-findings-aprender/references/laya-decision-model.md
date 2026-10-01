# Laya Decision Model: Quality, Adaptation and Cost vs Kev and SetFit

Laya (github.com/NandhaKishorM/laya @ `4066d5d`, weights `convaiinnovations/laya` @ `55cf4c4e`, Apache-2.0) is an
encoder-only decision model: ModernBERT-large (421M, **0.84 GB F16**) plus a 2-layer transformer head and a shared
scorer over per-option `[MASK]` markers. It speaks the same `choice` / `score` / `noul` types and `/v1/systemone`
wire protocol as Kev and Jev. **With a full fine-tune it is the best few-shot base measured here.**

## Requirements

From idea `llm-decision-classifier`:
- **Few-shot steering is the product**: the business supplies 8–64 labelled examples per class. Zero-shot with
  `criteria` descriptions is the baseline to beat.
- **Training may stay in Python; inference must be Rust on aprender.** The handoff is part of the contract:
  probability parity to Python fp32.
- **Laya was evaluated as an alternative base to Kev** (English root and `typed-decisions` checkpoints; build order
  024 → 025 → 026; decided 2026-09-25).

## How to Build It

### 1. The back-office adaptation is a full fine-tune
Frozen-feature adapters do not work on Laya (see What to Avoid). Fine-tune the whole model from the **English
root** with Laya's own notebook optimiser (`sources/024-laya-vs-kev-few-shot/tools/ft_laya.py`):
- AdamW: **encoder lr 2.5e-5, head lr 1e-4**, cosine decay to 1e-6, gradient clip 1.0, batch 8.
- Loss = CE + (−`laya.common.proper_reward`(softmax(z / T_bucket), onehot), w_sph 0.75).
- Build rows with Laya's own `Agent._encode_state`, so training rows are byte-identical to inference rows.
- Epochs: **12 at ≤ 16 shots/class**; **4–12 at 64**. Stance favoured 12, emotion favoured 4. Declare the recipe
  before reading a score.

| shots/class | SetFit | Kev-0.8B best | Kev-4B | Laya frozen best | **Laya full FT** |
|---|---|---|---|---|---|
| stance F_avg @16 | 0.512 | 0.507 (LoRA) | **0.642** | 0.482 | **0.538 ± 0.017** (r2) |
| stance F_avg @64 | 0.561 | 0.554 (LoRA) | **0.645** | 0.491 | **0.608 ± 0.050** (r2; best run 0.679) |
| emotion macro-F1 @64 | **0.705** | 0.530 | 0.577 | 0.482 | **0.697 ± 0.020** (r1) |

Training takes 26–316 s on stance and 190–570 s on emotion (M4 MPS), against 29 min for SetFit at 64 stance shots.

### 2. Calibrate after fine-tuning
Every fine-tuned run is **over-confident (ECE 0.17–0.38)**, because training loss goes to ~0 as the model
memorises the shots. Refit the per-bucket temperature on a held-out slice of the shots before serving.

### 3. Pick the checkpoint
- English root: the fine-tuning base.
- `typed-decisions` (the root fine-tuned on typed-decision data): the stronger **zero-shot** option on stance
  (F_avg 0.421 vs 0.364).
- Multilingual (mmBERT-base 322M, 256k vocabulary) is out of scope and untested here.

### 4. Evaluation harness
`sources/024-laya-vs-kev-few-shot/tools/`:
- `laya_eval.py` captures frozen features by forward hooks on Laya's own `Agent.predict` (the recompute matches the
  API to its 4-decimal rounding).
- `fewshot_laya.py` runs 015's recipes. `fewshot015.py` is a **verbatim** copy of 015's metric, bias fit and shot
  selection.
- `ft_laya.py` is the full fine-tune, `diag_head_ft.py` the train-side adapter diagnosis, `cpu_latency.py` the CPU
  timing.

## What to Avoid
- **Head-only adaptation (`bias` / `head_ft` / `logreg`) as the product.** On Laya it moves results by ≤ 0.05.
  Its scorer is one shared function applied to each option's marker state, so on frozen features it cannot learn a
  labelling convention: at lr 1e-4, 1e-3 and 1e-2, train accuracy on 64 shots stays 0.52–0.55. (Kev's pointer head
  could re-weight its `<decide>` query; Laya's cannot.)
- **Expecting criteria descriptions to steer it zero-shot.** Names-only beat descriptions (stance acc 0.411 vs
  0.339 on en, 0.439 vs 0.400 on typed-decisions) — the reverse of Kev.
- **Reading one seed.** Stance at 64 ranged 0.545–0.679 across seeds; report means over ≥ 3.
- **Trusting Laya's shipped `choice:11+` temperature (0.10).** Laya's own loader clamps it to 0.5 and warns.
- **Comparing torch CPU on a Mac with a Graviton target.** Laya's 65 ms / 6 threads there is Apple AMX
  (~1 TFLOP/s).

## Constraints
- Zero-shot: Laya-en ≈ Kev-0.8B (stance 0.364 vs 0.267, emotion 0.477 vs 0.478). Kev-4B is far ahead zero-shot (0.607).
- **Per-tenant artifact = a full 0.84 GB checkpoint** (Kev: 3 floats, a 2 MB head or a 41 MB LoRA). This is a storage
  and cold-start cost per deployed model, and it fits the one-model-per-thin-server rule.
- Contamination: Laya's README lists DAIR Emotion as held out; TweetEval stance appears in no Laya table.
- **Not measured**: full fine-tune of `typed-decisions`; 8 or 32 shots with full FT; more than 3 seeds; a
  business-shaped task; Kev-4B with full FT.

## Origin
Synthesized from spike 024 (VALIDATED), on spike 015's rows and code.
Source files: `sources/024-laya-vs-kev-few-shot/` (README, tools/, results/). The `.npz` features and `vendor/laya`
stay in `.planning/spikes/024-laya-vs-kev-few-shot/`.
