# SHIP-TWO: Sovereign Stack First Model Releases

Version: 1.0
Status: proposed
Date: 2026-04-10

**Document ID:** SPEC-SHIP-TWO-001
**Version:** 1.0.0
**Status:** IN PROGRESS
**Author:** PAIML Engineering
**Date:** 2026-04-10
**Priority:** P0 -- First shipped artifacts from the sovereign AI stack
**Contracts (apr-leaderboard):** 28 YAML contracts (67/68 passing), `contracts/` in paiml/apr-leaderboard
**Contracts (albor):** 54/54 YAML contracts, `contracts/` in paiml/albor
**Contracts (aprender):** `contracts/model-families/qwen2.yaml`, `contracts/model-families/llama.yaml`
**Falsification:** 22 FALSIFY-SHIP conditions (12 Model 1, 10 Model 2)
**Dependencies:** PMAT-014 (apr-leaderboard), ALB-010 (albor), entrenar CUDA training pipeline
**Hardware:** RTX 4060 Laptop (yoga), Blackwell GB10 (gx10), RTX 4090 (albor training), Jetson GB10 (teacher inference)
**PMAT Epic:** PMAT-514 (subtasks: PMAT-515..525)
**Kaizen Contracts:** `contracts/aprender/kaizen/ship-m1-ties-merge-v1.yaml`, `contracts/aprender/kaizen/ship-m2-sft-eval-v1.yaml`

---

## 1. Abstract

This specification documents the plan to ship two models from the sovereign AI stack's
external POC/research repositories. These are the first model artifacts published by the
PAIML ecosystem -- proof that the pure-Rust stack (aprender, entrenar, realizar, trueno)
can produce competitive code models without any US cloud dependency.

**Model 1** (apr-leaderboard): a distilled, DPO-aligned Qwen2.5-Coder-7B targeting 87%+
HumanEval and 80%+ MBPP. Critical path: ~36 hours from PMAT-014 completion.

**Model 2** (albor): a sovereign 370M Python code completion model trained entirely from
scratch in Rust, targeting 30%+ HumanEval. Critical path: 3-4 weeks.

These are not production deployments. They are the minimum viable model releases that prove
the sovereign stack can train, distill, merge, evaluate, and publish code models.

---

## 2. Motivation

Three motivations, each falsifiable:

1. **Proof-of-pipeline.** The sovereign stack has 547 YAML contracts, 74 workspace crates,
   25K+ tests -- but zero shipped model artifacts. A framework with no models is an
   untested claim. Shipping two models closes the loop from alimentar (data) through
   entrenar (training) through aprender (eval) through realizar (inference) through
   HuggingFace Hub (distribution).

2. **Two-speed validation.** Model 1 (36 hours) validates fast iteration on an existing
   open-weight foundation. Model 2 (3-4 weeks) validates from-scratch sovereign training
   where the stack controls every byte. If both succeed, the stack is proven at both
   extremes.

3. **Momentum.** Model 1 ships first. Its published infrastructure (eval harness, contract
   templates, Makefile targets, HuggingFace model card) becomes reusable scaffolding for
   Model 2 and all subsequent models. Dependency ordering is deliberate.

---

## 3. Design Principles

| Principle | Enforcement | Reference |
|-----------|-------------|-----------|
| Contract-first | Every shipped claim has a FALSIFY-SHIP-NNN test | This spec, Section 8 |
| Falsification-focused | Gates are binary pass/fail. "Close enough" is a FAIL | Popperian protocol |
| Sovereign stack | No US cloud API used for training, eval, or publish | Local hardware only |
| Lean on existing infra | 56 Makefile targets (apr-leaderboard), 54 contracts (albor) -- no new tooling | Repo READMEs |
| Ship fast, inspect deeply | Release the model, then run extended falsification | Toyota Way: stop the line for defects, not perfectionism |

---

## 4. Model 1 -- apr-leaderboard (Distilled Qwen2.5-Coder-7B)

### 4.1 Current State

| Dimension | Value | Evidence |
|-----------|-------|----------|
| Architecture | Qwen2.5-Coder-7B (28 layers, h=3584, 28/4 GQA, vocab 152064) | `contracts/model-families/qwen2.yaml` |
| Method | DPO alignment on N-sampled teacher completions (32B->7B) | GH-580/581 fix, DPO contract proven |
| HumanEval (few-shot) | 87.20% (143/164) | apr-leaderboard eval results |
| MBPP | 76.2% (381/500) -- gate: 80% | +3.8pp gap remaining |
| N-sampling | 1157/1640 on gx10 | PMAT-014 in progress |
| Infrastructure | 56 Makefile targets, 24 scripts, 22 YAML configs | apr-leaderboard repo |
| Contracts | 28 YAML, 67/68 passing | 1 failing = MBPP gate |
| Result artifacts | 1103 JSON result files | git-tracked eval output |

### 4.2 Acceptance Criteria

| ID | Criterion | Threshold | Measurement | Status |
|----|-----------|-----------|-------------|--------|
| AC-SHIP1-001 | HumanEval pass@1 (few-shot) | >= 87.0% | `make eval-humaneval` | PASS (87.20%) |
| AC-SHIP1-002 | MBPP pass@1 | >= 80.0% | `make eval-mbpp` | PENDING (76.2%) |
| AC-SHIP1-003 | All 28 contracts passing | 28/28 | `make contracts` | PENDING (67/68) |
| AC-SHIP1-004 | PMAT-014 N-sampling complete | 1640/1640 | gx10 job status | PENDING (1157/1640) |
| AC-SHIP1-005 | DPO training converges | val_loss decreasing, no NaN | Training log | PENDING |
| AC-SHIP1-006 | TIES merge produces loadable model | `apr validate` passes | apr-cli | PENDING |
| AC-SHIP1-007 | Final eval >= pre-merge scores | HE delta >= 0pp, MBPP delta >= +3.8pp | Eval harness | PENDING |
| AC-SHIP1-008 | Model card published to HuggingFace | All gates with evidence | HF Hub API | PENDING |
| AC-SHIP1-009 | Model runs via `apr run` | Valid Python on standard prompts | apr-cli | PENDING |
| AC-SHIP1-010 | No US cloud dependency in pipeline | Zero cloud API calls in logs | Hardware audit | PASS |

### 4.3 Critical Path

```
PMAT-014 completes (~10h remaining)
    |
    v
DPO training (~40 min on gx10)
    |
    v
TIES merge (LoRA adapters into base)
    |
    v
Final eval: HumanEval + MBPP (~3h)
    |
    v
Model card + HF upload
    |
    v
`apr run` smoke test
```

Total wall-clock: ~36 hours from PMAT-014 completion.

### 4.4 Contract Registry

| Contract | Domain | Status |
|----------|--------|--------|
| `n-sampling-v1.yaml` | Teacher completion generation | PASS |
| `dpo-alignment-v1.yaml` | DPO training convergence | PASS |
| `lora-merge-v1.yaml` | LoRA adapter merge (GH-580/581) | PASS |
| `ties-merge-v1.yaml` | TIES weight resolution | PENDING |
| `eval-humaneval-v1.yaml` | HumanEval harness correctness | PASS |
| `eval-mbpp-v1.yaml` | MBPP harness correctness | PASS |
| ... (22 additional) | Various pipeline stages | 67/68 PASS |

Full 28-contract registry in `paiml/apr-leaderboard/contracts/`.

---

## 5. Model 2 -- albor (Sovereign 370M Python Code Completion)

### 5.1 Current State

| Dimension | Value | Evidence |
|-----------|-------|----------|
| Architecture | LLaMA-style 370M (24 layers, h=1024, 16/4 GQA, SwiGLU 4096, 32K vocab, 1024 ctx) | `contracts/model-families/llama.yaml` pattern |
| Method | From-scratch pre-training in pure Rust + teacher distillation | albor training pipeline |
| Throughput | 12.3K tok/s, 38.7% MFU on RTX 4090 | albor benchmarks |
| Best val_ppl | 38.53 (v28, stopped -- raw data diverged) | Training logs |
| v29 status | Config ready, 2.04B filtered tokens (AST-verified) | Data pipeline |
| Distillation pilot | 982/1K prompts, 400K tokens | albor distillation logs |
| Contracts | 54/54 passing | `make contracts` |
| Gaps discovered | 129+ gaps, 80+ closed | Gap tracker |

### 5.2 Acceptance Criteria

| ID | Criterion | Threshold | Measurement | Status |
|----|-----------|-----------|-------------|--------|
| AC-SHIP2-001 | v29 pre-training completes without divergence | val_ppl monotonically decreasing for final 20% | Training log | PENDING |
| AC-SHIP2-002 | v29 val_ppl | < 60.0 AND monotonically decreasing | Final checkpoint | PENDING |
| AC-SHIP2-003 | HumanEval pass@1 (base, pre-SFT) | >= 15.0% | `make eval-humaneval` | PENDING |
| AC-SHIP2-004 | ALB-010 teacher loading complete | Qwen3-Coder-30B MoE runs on Jetson GB10 | Inference smoke test | PENDING |
| AC-SHIP2-005 | Teacher completions scaled to 100K | 100K filtered, rejection sampled | Data pipeline count | PENDING |
| AC-SHIP2-006 | SFT on teacher completions converges | val_loss decreasing, no NaN | Training log | PENDING |
| AC-SHIP2-007 | HumanEval pass@1 (post-SFT) | >= 30.0% | `make eval-humaneval` | PENDING |
| AC-SHIP2-008 | All 54 contracts passing | 54/54 | `make contracts` | PASS |
| AC-SHIP2-009 | Model runs via `apr run` | Valid Python on standard prompts | apr-cli | PENDING |
| AC-SHIP2-010 | Entire pipeline on sovereign hardware | RTX 4090 + Jetson GB10 only | Hardware audit | PASS |
| AC-SHIP2-011 | Model card published to HuggingFace | Architecture, training details, all gates | HF Hub API | PENDING |
| AC-SHIP2-012 | Training is 100% Rust | Zero Python in training loop | Code audit | PASS |

### 5.3 Critical Path

```
v29 pre-train (2.4 days on RTX 4090)
    |                              |
    v                              v (parallel)
  Evaluate base model         ALB-010: Load Qwen3-Coder-30B MoE
  (HumanEval baseline)           on Jetson GB10 (3-5 days)
    |                              |
    v                              v
    +------------- join -----------+
                    |
                    v
          Scale teacher completions to 100K
          (rejection sampling, 5-7 days)
                    |
                    v
          SFT on filtered completions (1-2 days)
                    |
                    v
          Final eval: HumanEval + manual inspection (~4h)
                    |
                    v
          Model card + HF upload
                    |
                    v
          `apr run` smoke test
```

Total wall-clock: 3-4 weeks. ALB-010 runs in parallel with v29 pre-training.

### 5.4 Contract Registry

| Contract | Domain | Status |
|----------|--------|--------|
| `model-architecture-v1.yaml` | LLaMA-style arch invariants | PASS |
| `tokenizer-v1.yaml` | 32K BPE tokenizer roundtrip | PASS |
| `training-loop-v1.yaml` | Forward/backward/optimizer step | PASS |
| `cuda-kernel-v1.yaml` | CUDA numerical parity with CPU | PASS |
| `checkpoint-v1.yaml` | Checkpoint save/resume identity | PASS |
| `data-pipeline-v1.yaml` | Filtered token count, dedup, quality | PASS |
| ... (48 additional) | Various pipeline stages | 54/54 PASS |

Full 54-contract registry in `paiml/albor/contracts/`.

---

## 6. Compound Gates

| Gate ID | Gate Name | Model | Criteria | Pass Condition | Ship-Blocking? |
|---------|-----------|-------|----------|----------------|----------------|
| GATE-SHIP-001 | HumanEval 7B | M1 | AC-SHIP1-001 | >= 87.0% | YES |
| GATE-SHIP-002 | MBPP 7B | M1 | AC-SHIP1-002 | >= 80.0% | YES |
| GATE-SHIP-003 | Contract Coverage 7B | M1 | AC-SHIP1-003 | 28/28 | YES |
| GATE-SHIP-004 | Pipeline Integrity 7B | M1 | AC-SHIP1-004..006 | All three PASS | YES |
| GATE-SHIP-005 | Model Loadable 7B | M1 | AC-SHIP1-009 | `apr run` succeeds | YES |
| GATE-SHIP-006 | Sovereignty 7B | M1 | AC-SHIP1-010 | No cloud API calls | YES |
| GATE-SHIP-007 | Pre-train Stable | M2 | AC-SHIP2-001..002 | No divergence, val_ppl < 60 AND decreasing | YES |
| GATE-SHIP-008 | Teacher Pipeline | M2 | AC-SHIP2-004..005 | 100K completions generated | YES |
| GATE-SHIP-009 | HumanEval 370M | M2 | AC-SHIP2-007 | >= 30.0% | YES |
| GATE-SHIP-010 | Contract Coverage 370M | M2 | AC-SHIP2-008 | 54/54 | YES |
| GATE-SHIP-011 | Model Loadable 370M | M2 | AC-SHIP2-009 | `apr run` succeeds | YES |
| GATE-SHIP-012 | Full Sovereignty 370M | M2 | AC-SHIP2-010..012 | No cloud, 100% Rust | YES |

All gates are binary. "Close to passing" is FAIL.

---

## 7. Falsification Tests

If ANY condition becomes true, the corresponding ship hypothesis is falsified.

### 7.1 Model 1 (apr-leaderboard)

| ID | Hypothesis Falsified If... | Threshold | Mitigation |
|----|---------------------------|-----------|------------|
| FALSIFY-SHIP-001 | HumanEval regresses after DPO | HE < 85.0% (2pp below current) | Roll back to pre-DPO checkpoint |
| FALSIFY-SHIP-002 | MBPP fails to reach gate after DPO | MBPP < 80.0% after full pipeline | Add MBPP-targeted teacher completions |
| FALSIFY-SHIP-003 | DPO training produces NaN loss | Any NaN after step 10 | Reduce LR by 10x; check data normalization |
| FALSIFY-SHIP-004 | TIES merge corrupts weights | `apr validate` fails OR >10% tensor stat deviation | Debug TIES resolution; fall back to linear merge |
| FALSIFY-SHIP-005 | PMAT-014 N-sampling stalls | < 50 completions/hr for > 2h | Check GPU util; restart with smaller batch |
| FALSIFY-SHIP-006 | LoRA merge bug recurs (GH-580/581) | Merged model outputs garbage | Re-apply fix; add regression contract |
| FALSIFY-SHIP-007 | Model fails `apr run` smoke test | Invalid Python on 3+ of 10 prompts | Debug tokenizer/generation pipeline |
| FALSIFY-SHIP-008 | Eval harness non-deterministic | > 1pp variance across 3 identical runs | Fix seed; check temperature/sampling |
| FALSIFY-SHIP-009 | 68th contract remains failing at ship | 27/28, not 28/28 | Fix or document exception with evidence |
| FALSIFY-SHIP-010 | Model card claims != measured gates | Any AC value differs from eval JSON | Regenerate card from JSON (no manual edit) |
| FALSIFY-SHIP-011 | Timeline exceeds 72h from PMAT-014 | Wall-clock > 72h (2x estimate) | Skip non-blocking steps; ship with gaps |
| FALSIFY-SHIP-012 | Cloud API detected in pipeline logs | Any OpenAI/Anthropic/AWS API call | Remove dependency; sovereignty non-negotiable |

### 7.2 Model 2 (albor)

| ID | Hypothesis Falsified If... | Threshold | Mitigation |
|----|---------------------------|-----------|------------|
| FALSIFY-SHIP-013 | v29 pre-training diverges like v28 | val_ppl increases > 500 consecutive steps after 5K | Stop; audit data pipeline; check LR schedule |
| FALSIFY-SHIP-014 | val_ppl does not reach < 60 | val_ppl >= 60.0 at end of v29 | Extend to 2 epochs; val set uses raw v3 data (distribution mismatch with filtered v4 training — see falsification round 2 notes) |
| FALSIFY-SHIP-015 | ALB-010 cannot load Qwen3-Coder-30B | OOM on Jetson GB10 or load fails | Fall back to Qwen2.5-Coder-7B (from M1) |
| FALSIFY-SHIP-016 | Rejection sampling filters > 80% | < 20K usable from 100K generated | Lower threshold; generate 200K raw |
| FALSIFY-SHIP-017 | SFT produces worse HumanEval than base | Post-SFT HE < pre-SFT HE | Check SFT data quality; reduce epochs |
| FALSIFY-SHIP-018 | HumanEval fails to reach 30% after SFT | HE < 30.0% | Accept 20%+ as preview; plan DPO follow-up |
| FALSIFY-SHIP-019 | CUDA != CPU reference | Divergence > 1e-4 on 100-step run | Debug CUDA kernel; file contract violation |
| FALSIFY-SHIP-020 | Throughput drops below 8K tok/s | Sustained < 8K (35% below baseline) | Profile GPU; check memory fragmentation |
| FALSIFY-SHIP-021 | Degenerate output | > 50% of 100 prompts produce repetition | Check data dedup; adjust sampling temperature |
| FALSIFY-SHIP-022 | Timeline exceeds 5 weeks | Wall-clock > 5 weeks (1.25x estimate) | Ship base without SFT; document plan |

---

## 8. Execution Plan

```
                    +-------------------------------------+
                    |           PHASE 0: SHARED            |
                    |  Set up eval harness compatibility   |
                    |  between apr-leaderboard + aprender  |
                    +----------------+--------------------+
                                     |
              +----------------------+---------------------+
              |                                            |
              v                                            v
+---------------------------+              +---------------------------+
|   PHASE 1: MODEL 1        |              |   PHASE 2: MODEL 2        |
|   (36h critical path)     |              |   (3-4 week critical      |
|                           |              |    path, starts day 1)    |
| 1a. PMAT-014 finishes     |              |                           |
|     (~10h)                |              | 2a. v29 pre-train         |
|         |                 |              |     (2.4 days)            |
|         v                 |              |         |                 |
| 1b. DPO train (~40min)   |              |         v                 |
|         |                 |              | 2b. Base eval             |
|         v                 |              |         |                 |
| 1c. TIES merge            |              | 2c. ALB-010 teacher       |
|         |                 |              |     load (parallel,       |
|         v                 |              |     3-5 days)             |
| 1d. Final eval (~3h)     |              |         |                 |
|         |                 |              |         v                 |
|         v                 |              | 2d. Scale completions     |
| 1e. Model card + upload  |              |     to 100K (5-7 days)    |
|         |                 |              |         |                 |
|         v                 |              |         v                 |
| 1f. `apr run` smoke      |              | 2e. SFT (1-2 days)       |
|                           |              |         |                 |
| --- SHIP MODEL 1 ---     |              |         v                 |
|                           |              | 2f. Final eval (~4h)     |
|         |                 |              |         |                 |
|         v                 |              |         v                 |
| 1g. Publish reusable     |              | 2g. Model card + upload  |
|     scaffolding           |-----reuse-->|         |                 |
|     (eval harness,        |  scaffolding|         v                 |
|      model card template, |              | 2h. `apr run` smoke      |
|      contract patterns)   |              |                           |
+---------------------------+              | --- SHIP MODEL 2 ---     |
                                           +---------------------------+
                                                        |
                                                        v
                                           +---------------------+
                                           |   PHASE 3: HANSEI   |
                                           |  Post-ship retro    |
                                           |  (both models)      |
                                           +---------------------+
```

**Dependency rules:**
- Phase 1 and Phase 2 run in parallel
- Phase 1 step 1g feeds into Phase 2 (reusable scaffolding)
- Phase 2 step 2c (ALB-010) starts day 1, parallel with 2a
- Phase 3 starts after both models ship

---

## 9. Risk Matrix

| Risk | Probability | Impact | Model | Mitigation |
|------|-------------|--------|-------|------------|
| MBPP +3.8pp gap not closed by DPO | Medium | HIGH | M1 | Add MBPP-focused teacher completions |
| PMAT-014 gx10 hardware failure | Low | HIGH | M1 | Resume from last checkpoint; ECC memory |
| v29 diverges like v28 | Medium | HIGH | M2 | Filtered data (vs raw in v28) should prevent |
| ALB-010 OOM on Jetson GB10 | Medium | HIGH | M2 | 4-bit quant or smaller teacher from M1 |
| Rejection sampling yield too low | Medium | MEDIUM | M2 | Lower threshold; generate 200K raw |
| `apr run` integration breaks | Low | MEDIUM | Both | Use realizar directly as fallback |
| HumanEval harness version mismatch | Low | HIGH | Both | Pin harness version; validate baselines |
| Model card accuracy | Low | MEDIUM | Both | Generate programmatically from eval JSON |
| Eval harness non-determinism | Low | MEDIUM | Both | Fix seed, temp=0, greedy decode |
| Hardware unavailability | Low | HIGH | Both | All data on local NVMe; no cloud dependency |

---

## 10. Failure Protocol / Hansei

### 10.1 Model 1

If GATE-SHIP-001 or GATE-SHIP-002 fails after full critical path:

1. **Five Whys.** Root cause: data (teacher completions), method (DPO hyperparams),
   or merge (TIES corruption)?
2. **Ship with documented gap.** If HumanEval >= 85% but MBPP < 80%, ship with model
   card documenting MBPP as "below target" with iteration plan.
3. **Escalation gate.** If HumanEval < 85%, do NOT ship. Return to N-sampling with
   expanded prompts.
4. **Timeline.** Maximum 1 additional iteration (48h). If still failing, document
   findings and pivot to Model 2 focus.

### 10.2 Model 2

If GATE-SHIP-009 fails (HumanEval < 30% after SFT):

1. **Five Whys.** Pre-train quality (val_ppl too high)? SFT data (bad teacher
   completions)? Model capacity (370M too small)?
2. **Reduced gate ship.** If HumanEval >= 20%, ship as "preview" with documented gap
   and planned DPO follow-up.
3. **Do not ship below 15%.** A 370M model below 15% HumanEval provides no signal
   beyond random. Document lessons and plan architecture revision.
4. **Hansei document.** Within 48 hours of failure, write Five Whys document in
   `docs/qa/`.

### 10.3 Full Failure

If both models fail to ship within timelines:

1. Publish "State of the Sovereign Stack Training Pipeline" with all measurements
2. Identify top 3 infrastructure gaps that prevented ship
3. Create PMAT work items for each gap
4. Set 30-day sprint to close gaps and retry

---

## 11. Integration with Aprender Monorepo

Both shipped models must be loadable via `apr run` (using realizar inference path):

```bash
# Model 1: Distilled 7B
apr run paiml/qwen2.5-coder-7b-dpo \
    --prompt "def fizzbuzz(n: int) -> list[str]:" --max-tokens 256

# Model 2: Sovereign 370M
apr run paiml/albor-370m \
    --prompt "def fibonacci(n: int) -> int:" --max-tokens 128
```

Requirements:
- Model 1: GGUF export (Qwen2 family, `contracts/model-families/qwen2.yaml`)
- Model 2: GGUF or SafeTensors export (LLaMA-style, `contracts/model-families/llama.yaml`)
- Both: HuggingFace Hub upload with model card
- Both: realizar inference pathway (per CLAUDE.md "Realizar-First Architecture")

---

## 12. PMAT Work Items

Epic **PMAT-514** decomposes into 11 subtasks across both models:

### Model 1 (36h critical path)

| PMAT | Title | Blocked By | Est. | Status |
|------|-------|------------|------|--------|
| PMAT-515 | Complete N-sampling (1640/1640) | -- | 10h | inprogress |
| PMAT-516 | DPO alignment training | PMAT-515 | 1h | todo |
| PMAT-517 | TIES merge + final eval (HE>=87%, MBPP>=80%) | PMAT-516 | 4h | todo |
| PMAT-518 | Model card + HF upload + `apr run` smoke | PMAT-517 | 2h | todo |
| PMAT-519 | Publish reusable scaffolding for Model 2 | PMAT-518 | 2h | todo |

### Model 2 (3-4 week critical path)

| PMAT | Title | Blocked By | Est. | Status |
|------|-------|------------|------|--------|
| PMAT-520 | v29 pre-training (2.04B filtered tokens) | -- | 3d | todo |
| PMAT-521 | ALB-010 teacher loading (Qwen3-Coder-30B MoE) | -- | 5d | todo |
| PMAT-522 | Scale teacher completions to 100K | PMAT-520, PMAT-521 | 7d | todo |
| PMAT-523 | SFT + final eval (HE>=30%) | PMAT-522 | 2d | todo |
| PMAT-524 | Model card + HF upload + `apr run` smoke | PMAT-523 | 2h | todo |

### Shared

| PMAT | Title | Blocked By | Est. | Status |
|------|-------|------------|------|--------|
| PMAT-525 | Post-ship Hansei retrospective | PMAT-518, PMAT-524 | 4h | todo |

### Kaizen Contracts

| Contract | Gate | File |
|----------|------|------|
| C-SHIPM1-001 | F-SHIPM1-001 | `contracts/aprender/kaizen/ship-m1-ties-merge-v1.yaml` |
| C-SHIPM2-001 | F-SHIPM2-001 | `contracts/aprender/kaizen/ship-m2-sft-eval-v1.yaml` |

---

## 13. References

| Reference | Location |
|-----------|----------|
| Qwen2 model family contract | `contracts/model-families/qwen2.yaml` |
| LLaMA model family contract | `contracts/model-families/llama.yaml` |
| HuggingFace pipeline spec | `docs/specifications/aprender-train/hugging-face-distill-learn-pipeline-spec.md` |
| Model eval framework spec | `docs/specifications/aprender-train/model-eval-framework-spec.md` |
| Fine-tune pipeline spec | `docs/specifications/aprender-train/fine-tune-rust-test-gen.md` |
| Entrenar training spec | `docs/specifications/aprender-train/entrenar-spec.md` |
| Monorepo consolidation spec | `docs/specifications/aprender-monorepo-consolidation.md` |
| apr-leaderboard repo | `paiml/apr-leaderboard` (external, 428 commits) |
| albor repo | `paiml/albor` (external, 446 commits) |

---

*End of specification SPEC-SHIP-TWO-001.*
