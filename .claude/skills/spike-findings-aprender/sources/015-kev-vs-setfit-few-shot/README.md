---
spike: 015
idea: llm-decision-classifier
name: kev-vs-setfit-few-shot
type: standard
validates: "Given the SetFit tasks at 8/16/32/64 shots per class, when SetFit, zero-shot Kev (0.8B/4B) with criteria descriptions, Kev with head-only adaptation, and Kev with a LoRA fine-tune from the released weights are compared on the same rows, then accuracy, calibration, latency and training cost show whether few-shot Kev beats SetFit enough to justify the port"
verdict: PARTIAL
related: []
tags: [kev, jev, setfit, few-shot, qwen3.5, classifier, benchmark]
---

# Spike 015: Kev vs SetFit, few-shot

## What This Validates
Given the tasks our SetFit already covers, at 8/16/32/64 labelled examples per class, when SetFit, zero-shot Kev
and three Kev adaptation methods are scored on identical rows, then we know where (if anywhere) Kev beats SetFit.

## Research
- **Kev** (github.com/jaredpalmer/kev @ `7405b72`, Apache-2.0): Qwen3.5 base + rank-16 LoRA over all linear layers
  (incl. Gated-DeltaNet projections) + `PointerHead` (q,k: d→256; logit = k(h_</opt>)·q(h_<decide>)/16, divided by a
  fitted temperature: 2.41 for 0.8B, 2.14 for 4B). On hybrid bases each question runs as its own row
  `[<state> text][<q> instr <opt> o </opt>… <decide>]` with the state prefix cached. Weights: `jaredpalmer/kev-0.8b`,
  `kev-4b`, `kev-9b` (adapter + `head.pt`; base pulled from Qwen at a pinned revision).
- **Contamination check**: Kev trains on banking77, SST-5, AG News, MNLI, BoolQ, Yelp + generated policy data.
  `tweet_eval stance_abortion` appears in no Kev source — clean. `emotion` is a Kev transfer-v4 **dev** source (not
  trained on, but used for checkpoint selection) — flagged.
- **SetFit baseline**: stance uses our committed EVAL-03 matrix (`benchmarks/tweeteval-stance/report.md`, `apr setfit`,
  MiniLM-L6, 10 seeds). Every Kev stance cell replays those exact `selections/sK-seedS` rows, so it is **paired**.
  `apr setfit` ingests only the attested stance dataset, so emotion uses Python `setfit` with the same encoder and
  the knobs of `crates/apr-cli/tests/fixtures/setfit/train-config.json` (5 seeds) — phases 1–5 proved the two agree.

| Approach | Where it runs | Trains | Output artifact |
|---|---|---|---|
| zs — released Kev | any | nothing | none |
| bias — K per-class logit offsets on zs | CPU, <1 s | K floats | K floats |
| head_ft — fine-tune released PointerHead | CPU, seconds | 0.5M (0.8B) / 1.3M (4B) | head.pt |
| logreg — LogisticRegression(C=1) on h_decide | CPU, <1 s | d·K | K×d matrix |
| LoRA — `kev.train --init_from` (r1 README recipe, r2 small-data recipe) | MPS, 30–375 s | 11.3M | adapter + head |

All recipes were fixed before any test score was read; nothing is selected by test error.

## How to Run
```bash
cd vendor/kev   # git clone https://github.com/jaredpalmer/kev && git checkout 7405b72 ; uv sync --extra serve
../../tools/run_kev_grid.sh                       # zero-shot + frozen features, 0.8B and 4B, both tasks (~45 min)
uv run --with pandas --with scikit-learn python ../../tools/fewshot.py --run kev-4b --task stance-abortion
uv run python ../../tools/make_shots_jsonl.py && ../../tools/run_lora_grid.sh   # LoRA, 0.8B (~40 min)
uv run --with setfit --with scikit-learn python ../../tools/setfit_ref.py       # emotion SetFit (~45 min)
```

## What to Expect
`results/*.json` per-cell rows; `fewshot.py` prints mean/std tables.

## Investigation Trail
1. **Zero-shot 0.8B stance = 0.325 accuracy, below the 0.675 majority baseline.** Suspected a label-mapping bug.
   Confusion matrix: 44/45 `none` correct; **143/189 `against` read as `none`**. TweetEval labels tweets like
   *"love means to be willing to give until it hurts - Mother Teresa"* as `against`. Kev's reading is defensible;
   the gold labels encode an annotator convention. Class-name-only criteria were worse (0.268), so the descriptions help.
2. That makes stance the textbook few-shot-steering case: **bias** (prior shift) cannot teach a convention and stays
   flat at 0.375 for 0.8B; the trained heads climb with shots.
3. **4B changes the stance picture**: zero-shot F_avg 0.607 already exceeds SetFit at 64 shots (0.561).
4. **LoRA with the README recipe under-trains at few shots** (12 optimizer steps at 16 shots → F_avg 0.387, below
   head_ft). The pre-declared small-data recipe r2 (4 epochs, accum 1, lr 5e-5) fixes it: 0.512 at 8 shots.
5. **Emotion inverts at larger K**: SetFit goes 0.343 → 0.705 from 8 to 64 shots; the best Kev (4B head_ft) reaches
   only 0.577. Frozen-backbone Kev plateaus; SetFit's contrastive encoder fine-tune keeps scaling on a lexical task.

## Results

**Stance (TweetEval F_avg = mean F1 of against/favor; SetFit = committed 10-seed matrix, paired rows)**

| shots/class | SetFit | Kev-0.8B best frozen | Kev-0.8B LoRA r2 (3 seeds) | Kev-4B zs | Kev-4B bias | Kev-4B head_ft |
|---|---|---|---|---|---|---|
| 0 | – | 0.267 (zs) | – | **0.607** | – | – |
| 8 | 0.475 | 0.423 head_ft | 0.512 | | **0.642** | 0.623 |
| 16 | 0.512 | 0.467 head_ft | 0.507 | | **0.642** | 0.609 |
| 32 | 0.535 | 0.500 head_ft | – | | **0.646** | 0.626 |
| 64 | 0.561 | 0.533 logreg | 0.554 | | **0.645** | 0.639 |

**Emotion (macro-F1, 1000 test rows; SetFit = Python reference, 5 seeds; Kev flagged: emotion was a Kev dev set)**

| shots/class | SetFit | Kev-0.8B zs / head_ft | Kev-4B zs / head_ft |
|---|---|---|---|
| 0 | – | 0.478 | 0.504 |
| 8 | 0.343 | 0.464 | **0.488** |
| 16 | 0.437 | 0.490 | **0.521** |
| 32 | **0.579** | 0.516 | 0.556 |
| 64 | **0.705** | 0.530 | 0.577 |

**Cost (M4 Pro, 48 GB)**

| | SetFit | Kev-0.8B | Kev-4B |
|---|---|---|---|
| inference per row | 32–34 ms warm (Rust, CPU) | 33 ms (MLX bf16, GPU) | 165 ms (MLX bf16, GPU) |
| head adaptation (bias / head_ft / logreg) | – | < 5 s CPU | < 10 s CPU |
| full fine-tune at 64 shots | 29 min stance (Rust CPU), 7.7 min emotion (Py MPS) | LoRA r2: 6.2 min MPS | not run |
| weights to ship | 91 MB | ~1.7 GB bf16 base + 41 MB adapter | ~9 GB bf16 |

**Calibration**: Kev zero-shot and bias are well calibrated (ECE 0.04–0.16); head_ft and logreg on few shots are
over-confident (ECE 0.14–0.37) because training sets T=1 and nothing re-fits the temperature. A temperature fit on a
held-out slice of the shots is needed before head_ft probabilities are served.

**Verdict: PARTIAL ⚠.** Kev is worth adding next to SetFit, not instead of it:
- **Kev wins the few end.** At 0–16 shots Kev beats SetFit on both tasks. The difference is largest on stance: 4B
  zero-shot beats SetFit trained on 64 shots, and 8 shots of prior correction (3 floats) adds 3.5 points.
- **SetFit wins at 32+ shots on emotion**, by 13 points at 64. Kev's frozen-backbone adapters plateau.
- **Size decides quality**: 0.8B is at best SetFit-level; the gains need 4B, which is 5× slower and 100× bigger than
  SetFit. That makes 019 (Lambda) the next deciding risk, not an afterthought.
- **The back-office shape works**: the strongest few-shot adapters (bias, head_ft) train in seconds on CPU and emit
  a few floats or a 1–2 MB head on top of unchanged released weights. Rust inference needs only the released Kev
  weights plus that small artifact; full LoRA fine-tuning stays in Python on the (validated) Mac/MPS path.

**Not measured**: 4B LoRA fine-tuning; Kev-9B (disk: 28 GB free); a business-shaped task (support routing with
policy criteria), where Kev's descriptions should matter more than on these two academic sets.
