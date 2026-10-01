---
spike: 024
idea: llm-decision-classifier
name: laya-vs-kev-few-shot
type: comparison
validates: "Given the spike-015 stance and emotion rows (paired with SetFit and Kev), when Laya (English root and typed-decisions checkpoints) is scored zero-shot and with the same head-only adaptations (bias, head_ft, logreg) at 8-64 shots, plus a full fine-tune, then we know whether Laya matches Kev-4B / Kev-0.8B and SetFit"
verdict: VALIDATED
related: [015, 017, 019]
tags: [laya, modernbert, kev, setfit, few-shot, classifier, benchmark]
---

# Spike 024: Laya vs Kev vs SetFit, few-shot

## What This Validates
Laya (github.com/NandhaKishorM/laya @ `4066d5d`, weights `convaiinnovations/laya` @ `55cf4c4e`, Apache-2.0) is an
encoder-only alternative to Kev: ModernBERT-large (421M, 843 MB fp16) + a 2-layer transformer head + a shared
scorer over per-option `[MASK]` markers, one bidirectional pass per question, the same `choice`/`score`/`noul`
types and `/v1/systemone` wire. Same rows, seeds, metrics and SetFit pairing as spike 015.

## Research
| | Kev-0.8B / 4B | Laya (English root) |
|---|---|---|
| backbone | Qwen3.5 hybrid decoder + LoRA r16 | ModernBERT-large encoder (28 layers, d 1024, RoPE, global every 3rd layer, 128-token sliding otherwise, GeGLU 2624) |
| weights | 3.0 GB f32 / 1.5 GB bf16 (0.8B); ~9 GB bf16 (4B) | **0.84 GB fp16** |
| readout | `k(h_</opt>) . q(h_<decide>)` pointer head | `[CLS] <type> question: ins [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]` -> +type embedding -> 2-layer `nn.TransformerEncoder` (pre-norm, 16 heads, FFN 4d) -> shared scorer LN->Linear->GELU->Linear on each marker -> softmax / T(bucket) |
| temperature | one fitted T | per bucket: choice:2 1.91, 3-5 **1.76**, 6-10 **1.00**, noul 1.98, score 1.25; `choice:11+` ships 0.10 (Laya's own loader clamps it to 0.5 and warns) |
| training | SFT + LoRA | "RLCD" (strictly proper scoring reward + CE); `typed-decisions` = the English root fine-tuned on typed-decision data |

Contamination: Laya's README lists DAIR Emotion as **held out** (README line 947, 0.573 reported); TweetEval stance
appears in no Laya benchmark table.

Capture: forward hooks on Laya's own `Agent.predict` path (no re-implemented sequence builder): `m_opts` = scorer
input (marker states after the head), `pooled` = post-head `[CLS]` (act-head input), `logits` = raw scorer output.
Recomputing probabilities from the hooks matches the API to 5.0e-5 (its 4-decimal rounding).

## Pre-declared recipes (written before any score of that recipe was read)
- `zs`, `bias`, `head_ft`, `logreg` — spike 015's recipes verbatim (`tools/fewshot015.py` is a verbatim copy of 015's
  metric, bias fit and shot selection); `head_ft` fine-tunes Laya's released scorer on frozen `m_opts`.
- **Full fine-tune (Laya's intended adaptation), declared 2026-09-25 after the head-only results but before any
  fine-tune score**: from the English root, stance only, shots {16, 64}, seeds {13, 17, 23}, encoder unfrozen,
  Laya notebook optimiser (AdamW, encoder lr 2.5e-5, head lr 1e-4, cosine -> 1e-6, clip 1.0, loss = CE + Laya
  `proper_reward`), batch 8 on one MPS device:
  - `ft-r1`: the notebook's 4 epochs
  - `ft-r2`: 12 epochs (the small-data variant, same reason as 015's LoRA r2: 4 epochs at few shots is few steps)
- **Extension, declared 2026-09-25 after the stance fine-tune but before any emotion fine-tune score**: the same two
  recipes, unchanged, on **emotion** at 64 shots, seeds {13, 17, 23} (emotion is where SetFit won by 13 points).

## Investigation Trail
1. **English root, stance zero-shot F_avg 0.364** (acc 0.339): above Kev-0.8B zs (0.267), far below Kev-4B (0.607).
2. **`head_ft` did not move** (acc 0.3393, std 0.0000 at 64 shots — identical to zs). Train-side diagnosis
   (`tools/diag_head_ft.py`, no test rows read): with lr 1e-4 / 1e-3 / 1e-2 the scorer's TRAIN accuracy on 64 shots
   stays 0.52–0.55. Laya's scorer is one shared function applied to each option's marker state, so on frozen
   features it cannot encode a class convention; Kev's pointer head could re-weight the `<decide>` query. Head-only
   adaptation is a Kev property, not a transferable recipe.
3. **Emotion zero-shot macro-F1 0.477** (acc 0.565 — Laya's README reports 0.573): equal to Kev-0.8B (0.478), and
   no cheap adaptation moves it (best 0.482 at 64 shots). `head_ft` improves calibration only (ECE 0.33 -> 0.07).

4. **`typed-decisions` checkpoint** is the stronger zero-shot Laya on stance (F_avg 0.421, acc 0.400 vs 0.339) and
   equal on emotion (0.466). Its cheap adaptations top out at 0.491 (logreg @64) — still below SetFit and Kev-0.8B.
5. **Names beat descriptions for Laya**: stance names-only acc 0.411 (en) / 0.439 (typed) vs 0.339 / 0.400 with
   the business descriptions — the reverse of Kev (0.268 names vs 0.325 descriptions). Zero-shot steering by
   `criteria` text is not something Laya rewards.
6. **Full fine-tune changes the verdict.** 12 runs on stance + 6 on emotion (pre-declared). Training loss goes to
   ~0 on every run (the model memorises the shots); test accuracy still climbs far past every frozen variant.
7. **CPU latency** (`tools/cpu_latency.py`, M4 Pro, torch fp32, 30 stance rows, 86 tokens p50): **65 ms p50 at 6
   threads** (80 ms at 14 — more threads do not help at this size). Kev-0.8B in torch fp32 CPU on the same tweets
   (spike-017 fixture): 4.2 s per question; our optimised Rust Kev-0.8B: 0.36 s. A bidirectional encoder is a
   handful of dense GEMMs; the hybrid decoder pays a sequential DeltaNet recurrence.

## Results

**Stance (TweetEval F_avg; SetFit = committed 10-seed matrix; every Laya cell replays the same selections)**

| shots | SetFit | Kev-0.8B best frozen / LoRA r2 | Kev-4B zs / bias | Laya-en best frozen | Laya typed best frozen | **Laya-en full FT r1** | **Laya-en full FT r2** |
|---|---|---|---|---|---|---|---|
| 0 | – | 0.267 | **0.607** | 0.364 | 0.421 | – | – |
| 8 | 0.475 | 0.423 / 0.512 | **0.642** | 0.434 | 0.477 | – | – |
| 16 | 0.512 | 0.467 / 0.507 | **0.642** | 0.442 | 0.482 | 0.476 ± 0.026 | **0.538 ± 0.017** |
| 32 | 0.535 | 0.500 / – | **0.646** | 0.444 | 0.481 | – | – |
| 64 | 0.561 | 0.533 / 0.554 | **0.645** | 0.450 | 0.491 | 0.594 ± 0.043 | **0.608 ± 0.050** (best run 0.679) |

**Emotion (macro-F1, 1000 test rows; held out for Laya per its README)**

| shots | SetFit | Kev-0.8B zs / head_ft | Kev-4B head_ft | Laya-en best frozen | **Laya-en full FT r1** | Laya-en full FT r2 |
|---|---|---|---|---|---|---|
| 0 | – | 0.478 | 0.504 (zs) | 0.477 | – | – |
| 64 | **0.705** | 0.530 | 0.577 | 0.482 | **0.697 ± 0.020** | 0.672 ± 0.011 |

**Cost (M4 Pro)**

| | SetFit | Kev-0.8B | Kev-4B | **Laya (en)** |
|---|---|---|---|---|
| weights | 91 MB | 3.0 GB f32 / 1.5 GB bf16 | ~9 GB bf16 | **0.84 GB fp16** (1.7 GB f32) |
| decision, CPU 6 threads | 32 ms (Rust) | 360 ms (Rust, spike 020) · 4.2 s torch | ~2 s est. | **65 ms (torch fp32, not yet ported)** |
| decision, laptop GPU | – | 33 ms (MLX) | 165 ms (MLX) | 21–29 ms (MPS) |
| few-shot training @64 | 29 min stance (Rust CPU) | LoRA 6.2 min (MPS) | not run | **105–315 s stance, 190–570 s emotion (MPS)** |
| per-tenant artifact | 91 MB | 3 floats (bias) / 2 MB head / 41 MB LoRA | 3 floats / 2.6 MB head | **a full 0.84 GB checkpoint** |
| calibration after adaptation | ok | head_ft over-confident (ECE 0.14–0.37) | same | **FT over-confident (ECE 0.17–0.38)** |

**Verdict: VALIDATED ✓ — Laya is the better base for few-shot steering, with a different adaptation shape.**
- **Fully fine-tuned from the English root on the shots, Laya beats SetFit on stance at 16 and 64 shots
  (0.538 / 0.608 vs 0.512 / 0.561) and ties it on emotion at 64 (0.697 vs 0.705)** — the task where every Kev
  variant lost to SetFit by 13+ points. Kev-0.8B loses to it everywhere; only Kev-4B (9 GB, ~2 s/decision on CPU)
  is still ahead on stance (0.645), and Kev-4B loses emotion to Laya by 0.12.
- **Zero-shot and cheap adaptation are Kev's strengths, not Laya's**: Laya zero-shot ≈ Kev-0.8B, criteria
  descriptions do not help it, and bias / head_ft / logreg on frozen features move it by ≤ 0.05 (its shared marker
  scorer structurally cannot learn a label convention on frozen states — train acc stays 0.52–0.55).
- **It is 5.5x faster on CPU than our optimised Rust Kev-0.8B before any porting** (65 ms vs 360 ms), and its
  weights are 3.6x smaller — the two quantities spike 021 showed dominate Lambda latency and cold start.

**Signal for the build (and for 025/026)**
- The back-office adaptation is **full fine-tuning** (Laya's notebook optimiser, 4–12 epochs, 2–10 min on a laptop
  GPU); the handoff artifact is therefore a **whole 0.84 GB checkpoint per tenant/task**, not a few floats. That is
  a storage and cold-start cost per deployed model, and it fits the one-model-per-thin-server rule.
- **Refit the temperature on held-out shots** before serving: every fine-tuned run is over-confident.
- Variance across seeds is real (stance @64: 0.545–0.679); report means over ≥ 3 seeds, never one run.
- **Not measured**: full FT of the `typed-decisions` checkpoint; 8 / 32 shots FT; > 3 seeds; a business-shaped task;
  Kev-4B full FT. SetFit's cells have 10 (stance) / 5 (emotion) seeds vs Laya FT's 3.
