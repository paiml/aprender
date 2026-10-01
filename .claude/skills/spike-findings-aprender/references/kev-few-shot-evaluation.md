# Kev Decision Model: Few-Shot Evaluation and Adaptation

> **Since spike 024: a fully fine-tuned Laya (0.84 GB) beats SetFit on stance and ties it on emotion.** Every Kev
> variant loses emotion to SetFit, and only Kev-4B stays ahead on stance. Read `laya-decision-model.md` before
> choosing a base. Kev keeps the strongest *zero-shot* and *cheap-adapter* results.

Where Kev (Qwen3.5 + LoRA + pointer head) beats SetFit, which adaptation recipe to ship, and how to
calibrate it. Kev sits **next to** SetFit, not in place of it.

## Requirements

From idea `llm-decision-classifier`:

- **Few-shot steering is the product.** The business supplies a handful of labelled examples
  (SetFit's 8–64 per class) to add its own knowledge and bias; massive datasets are out of scope.
  Zero-shot Kev with `criteria` descriptions is the baseline few-shot must beat, because it already
  carries some steering.
- **Training may stay in Python; inference must be Rust on aprender.** Fine-tuning is a back-office
  process whose only output is weights. Rust-side training is a bonus, not a requirement.
- **The Python-to-Rust weight handoff is part of the contract**: an adapted Kev checkpoint must
  export to an artifact the Rust path loads, with probability parity to Python fp32
  (see `qwen35-decision-inference.md`).

## How to Build It

### 1. Know what Kev is

`jaredpalmer/kev` @ `7405b72`, Apache-2.0. Weights `jaredpalmer/kev-0.8b`, `kev-4b`, `kev-9b`
(LoRA adapter + `head.pt`; the base is pulled from Qwen at a pinned revision).

- Qwen3.5 base + rank-16 LoRA over **all** linear layers, including the Gated-DeltaNet projections.
- `PointerHead`: `q, k: d → 256`; `logit_j = k(h_</opt>_j) · q(h_<decide>) / 16`; served
  `softmax(logit / T)` with a fitted temperature (**T = 2.41** for 0.8B, **2.14** for 4B).
- Row form on the hybrid: each question is its own causal row
  `[<state> text][<q> instr <opt> o </opt> … <decide>]`. Three question types: `noul` (yes/no),
  `choice`, `score`. It never generates text.

### 2. Pick the adaptation recipe by shots per class

All recipes run on the **unchanged released weights**, so Rust inference needs only the released
Kev weights plus a small artifact:

| Approach | Where it trains | Trains | Artifact to ship |
|---|---|---|---|
| zero-shot | nothing | nothing | none |
| **bias**: K per-class logit offsets on zero-shot | CPU, < 1 s | K floats | K floats |
| **head_ft**: fine-tune the released PointerHead | CPU, seconds | 0.5M (0.8B) / 1.3M (4B) params | `head.pt` (1–2 MB) |
| logreg: `LogisticRegression(C=1)` on `h_decide` | CPU, < 1 s | d·K | K×d matrix |
| LoRA: `kev.train --init_from` | MPS, 30–375 s | 11.3M | adapter + head |

Measured (M4 Pro). Stance: TweetEval F_avg, paired with the committed SetFit 10-seed matrix. Emotion:
macro-F1, Python SetFit reference with 5 seeds.

| shots/class | SetFit stance | **Kev-4B stance** (zs / bias / head_ft) | SetFit emotion | Kev-4B emotion head_ft |
|---|---|---|---|---|
| 0 | – | **0.607** zs | – | 0.504 zs |
| 8 | 0.475 | **0.642** bias | 0.343 | **0.488** |
| 16 | 0.512 | **0.642** bias | 0.437 | **0.521** |
| 32 | 0.535 | **0.646** bias | **0.579** | 0.556 |
| 64 | 0.561 | **0.645** bias | **0.705** | 0.577 |

Routing rule that falls out of this:
- **0–16 shots**: Kev wins on both tasks. Use Kev-4B; ship zero-shot or `bias`.
- **32+ shots on a lexical task** (emotion): SetFit wins, by 13 points at 64. Kev's frozen-backbone
  adapters plateau, while SetFit's contrastive encoder fine-tune keeps scaling.
- **Kev-0.8B is at best SetFit-level** (stance: head_ft 0.423 → logreg 0.533; LoRA r2 0.512 → 0.554).
  The quality gain needs 4B, which is 5× slower and 100× bigger than SetFit. Deployment is therefore
  the deciding risk (see `llm-classifier-lambda-deployment.md`).

### 3. Calibrate before serving

Zero-shot and `bias` are well calibrated (ECE 0.04–0.16). `head_ft` and `logreg` on few shots are
**over-confident (ECE 0.14–0.37)**: training sets T = 1 and nothing re-fits it. Fit a temperature on
a held-out slice of the shots and ship it with the head. The Rust head reads `temperature` from the
safetensors metadata (spike 017's export does this).

### 4. If you must LoRA, declare the small-data recipe

Kev's README recipe (r1) is **12 optimizer steps at 16 shots** and under-trains (F_avg 0.387, below
head_ft). The small-data recipe **r2: 4 epochs, grad-accum 1, lr 5e-5** fixes it (0.512 at 8 shots).
Declare both recipes before reading any test score.

### 5. Evaluation harness (reuse it)

`sources/015-kev-vs-setfit-few-shot/tools/`:
- `kev_eval.py` runs a checkpoint over a split and captures the head inputs `(h_decide, h_opts)` by
  wrapping `model.head`. That is the one call both torch and MLX make, so every head-only variant
  shares one forward pass. Save them as `.npz`.
- `fewshot.py` trains bias / head_ft / logreg on the frozen features and prints mean ± std tables.
- `tasks.py` holds task criteria (with descriptions) and loaders; `setfit_ref.py` is the Python SetFit
  stand-in; `run_kev_grid.sh` and `run_lora_grid.sh` drive the grids.

```bash
cd vendor/kev   # git clone https://github.com/jaredpalmer/kev && git checkout 7405b72 && uv sync --extra serve
../../tools/run_kev_grid.sh                        # zero-shot + frozen features, 0.8B + 4B (~45 min)
uv run --with pandas --with scikit-learn python ../../tools/fewshot.py --run kev-4b --task stance-abortion
```

## What to Avoid

- **Don't read low zero-shot accuracy as a label bug before looking at the confusion matrix.** 0.8B
  stance scored 0.325, below the 0.675 majority baseline, because 143/189 `against` were read as
  `none`. TweetEval labels *"love means to be willing to give until it hurts - Mother Teresa"* as
  `against`. Kev's reading is defensible; the gold labels encode an annotator convention. That makes
  this the textbook few-shot steering case, not a defect.
- **`bias` cannot teach a convention.** It is a prior shift, so it stays flat (0.375 for 0.8B stance)
  while the trained heads climb with shots. It works on 4B because 4B already reads the convention.
- **Don't drop the criteria descriptions.** Class-name-only criteria scored lower (0.268 vs 0.325).
- **Don't serve head_ft probabilities uncalibrated** (ECE up to 0.37).
- **Don't compare against SetFit on different rows.** Replay the committed
  `benchmarks/tweeteval-stance/selections/sK-seedS/` manifests. A task `apr setfit` cannot ingest
  uses Python `setfit` with `train-config.json`'s knobs, flagged as a stand-in.
- **Don't treat emotion numbers as clean.** `emotion` is a Kev transfer-v4 **dev** set: not trained on,
  but used for checkpoint selection. `tweet_eval stance_abortion` is in no Kev source.

## Constraints

- Inference per row on the M4 Pro: SetFit 32–34 ms (Rust CPU); Kev-0.8B 33 ms, Kev-4B 165 ms (both
  MLX bf16 on the **GPU**; the Rust CPU figures are in the inference reference).
- Weights to ship: SetFit 91 MB; Kev-0.8B ~1.7 GB bf16 base + 41 MB adapter; Kev-4B ~9 GB bf16.
- Full fine-tune at 64 shots: SetFit 29 min stance (Rust CPU); Kev-0.8B LoRA r2 6.2 min (MPS).
- **Not measured**: 4B LoRA, Kev-9B (disk), and a business-shaped task (support routing with policy
  criteria), where Kev's descriptions should matter more than on these two academic sets. Measure the
  business task before claiming a production win.

## Origin

Synthesized from spike 015 (PARTIAL).
Source files: `sources/015-kev-vs-setfit-few-shot/` (README, tools/, results/).
The frozen features (`runs/*.npz`, ~200 MB) and `vendor/kev` stay in
`.planning/spikes/015-kev-vs-setfit-few-shot/`.
