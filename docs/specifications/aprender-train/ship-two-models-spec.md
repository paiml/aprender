# Specification: Ship Two Models — Sovereign AI Stack Proof

**Document ID:** SPEC-SHIP-TWO-001
**Version:** 2.5.0
**Status:** PRE-UPLOAD GATES COMPLETE — HF_TOKEN REQUIRED FOR UPLOAD
**Author:** PAIML Engineering
**Reviewer:** Noah Gift
**Date:** 2026-04-17 (v1.0.0) / 2026-04-17 (v2.0.0 audit + pivot) / 2026-04-18 (v2.5.0 pre-flight Poka-Yoke)

**v2.5.0 amendment (2026-04-18):** all seven ship manifest gates (PM-001..007)
now run inside `scripts/ship-two-001/ex-04-upload-hf.sh` as a pre-flight
Poka-Yoke. Any manifest-vs-artifact divergence aborts with non-zero exit
BEFORE any network I/O (contract `apr-cli-publish-extra-v1.yaml` v1.2.0,
`publish-manifest-v1.yaml` v1.1.0). Local validation shows all three ship
artifacts (`.apr`, `.safetensors`-fp16, `.gguf`) PASS every gate — the ship
is unblocked on `HF_TOKEN` alone.

---

## 1. Abstract

This specification defines the contract-first, falsification-driven plan to ship **production models**
through the aprender monorepo, proving end-to-end sovereignty (training → format → inference → eval) of the
Sovereign AI Stack.

**v2.0.0 scope change:** the original distilled-student artifact failed the 2026-04-17 contract-first
audit (see §1.5). The spec now pivots to an **expedited teacher-first ship** (see §12) while defering
distillation and sovereign training to follow-on releases. Either artifact reaching SHIP status
falsifies the null hypothesis "the stack cannot produce shippable weights"; the teacher-first ship
alone satisfies that falsification.

Original (v1.0.0) scope, retained for reference:
1. **MODEL-1 (apr-leaderboard):** A distilled Qwen2.5-Coder-7B student targeting **87.20% HumanEval pass@1**,
   shippable in **~36 engineering hours** from the current trained checkpoint.
2. **MODEL-2 (albor):** A sovereign, from-scratch **370M Python code-completion model** targeting **≥30%
   HumanEval pass@1**, shippable in **3–4 weeks** of compute + engineering.

All shipped artifacts must load via `apr run` (realizar backend), pass `apr qa` Golden Output gates, and
carry a contract-conforming manifest (`contracts/publish-manifest-v1.yaml`).

---

## 1.5. Audit Findings (2026-04-17)

**Verdict:** v1.0.0 MODEL-1 SHIP PATH IS BLOCKED. AC-SHIP1-005 falsified; teacher-first pivot in progress.

### 1.5.1 What was audited

Under contract `F-EVAL-HUMANEVAL-AUDIT-001` (`contracts/eval-harness-humaneval-v1.yaml` v1.1.0):
- Primary student checkpoint: `qwen2.5-coder-7b-distilled-v2-q4k.apr` (5.8 GB, Apr-3)
- Audit tool: `apr qa` + partial `apr eval --benchmark humaneval` via apr-leaderboard harness
- Binary: `/mnt/nvme-raid0/targets/aprender/release/apr` 0.31.0 (commit 9217e9c8a), RTX 4090

### 1.5.2 What we found

| Gate                 | Result            | Measured                                                           |
|----------------------|-------------------|--------------------------------------------------------------------|
| Capability Match     | ✓ PASS            | —                                                                  |
| Tensor Contract      | ✓ PASS            | 339 tensors pass PMAT-235                                          |
| Metadata Plausibility| ✓ PASS            | arch=qwen2, rope_theta=1000000                                     |
| **Golden Output**    | **✗ FAIL**        | For "2+2=" expected "4"; got `xxx9,x,x,,,,,,,,,,,,,999`            |
| Throughput           | ✓ PASS            | 9.7 tok/s (threshold=1)                                            |
| HumanEval pass@1     | **~0 (inferred)** | Batch output was incoherent BPE ("uardsylkoylkoiaÅĤ...") on 2/164  |
| Teacher pass@1       | 85.98 (prior run) | `results/humaneval_20260328_121327.json` — pipeline is sound       |

**The distilled student cannot generate coherent text.** Its tensors are structurally valid (all pass
shape, dtype, and non-finite checks) but its weights do not represent a working model.

### 1.5.3 Five-Whys (recorded in contract `validation_result_v1_1`)

1. **Why did the audit fail?** Student emits garbage BPE sequences on every prompt.
2. **Why is output garbage if weights load?** Tensor Contract validates structure, not semantics —
   weights with legal dtype+shape+finite values can still be a broken model.
3. **Why might weights be broken?** Three candidates, in decreasing likelihood:
   (a) distillation diverged and the run was saved without a sanity gate;
   (b) `apr convert --quantize q4_k_m` introduced a LAYOUT-001-class transpose bug;
   (c) BPE tokenizer / chat-template drift so generation samples from wrong token space.
4. **Why can't we tell which?** Diagnostics (`apr diff`, merged-checkpoint run, tokenizer round-trip)
   were not required gates — they remain diagnostic follow-ups in the contract.
5. **Why did no earlier gate catch this?** `apr qa` Tensor Contract exits PASS before Golden Output
   runs; Golden Output failure does NOT block publish in the current gate matrix. This is the
   root contract gap, now promoted to the expedited plan's first action (§12.1).

### 1.5.4 Notable gap surfaced

The 87.20% figure traces back to recipe-h-32b-distill.yaml's comment labelling the *base 7B-Instruct
few-shot* HumanEval — not a distilled-student zero-shot run. No `apr eval` result file for a
distilled student exists in `apr-leaderboard/results/`; all 17 archived HumanEval runs measure the
teacher. The headline number in v1.0.0 §4.1 was therefore never a reproducible claim.

---

## 2. Motivation

### 2.1 Why These Two Models

| Criterion                    | MODEL-1 (apr-leaderboard) | MODEL-2 (albor)        |
|------------------------------|---------------------------|------------------------|
| Current state                | trained, needs packaging  | architecture designed, pretraining required |
| Engineering distance to SHIP | 36 h                      | 3–4 weeks              |
| Proves distillation path     | yes                       | no                     |
| Proves sovereign path        | partial (uses HF teacher) | **yes (end-to-end)**   |
| Proves eval harness          | yes (HumanEval)           | yes (HumanEval)        |
| Risk profile                 | LOW                       | MEDIUM-HIGH (training) |

Shipping both gives orthogonal proof: one demonstrates the stack can finish what PyTorch started;
the other demonstrates the stack can start AND finish without PyTorch in the loop.

### 2.2 Explicit Non-Goals (v1)

- Not shipping: `entrenar-rl` (POC), `entrenar-rlhf` (POC), `verificar-agent` (research).
- Not targeting: chat tuning, multimodal, tool use, >10B params.
- Not blocking on: full leaderboard automation (post-SHIP), wandb integration, distributed training.

---

## 3. Design Principles

| #  | Principle                   | Operationalization                                                    |
|----|-----------------------------|-----------------------------------------------------------------------|
| 1  | Contract-first              | Every weight file, config, and eval path has a YAML contract BEFORE code |
| 2  | Falsification-driven        | Every acceptance criterion has a named, executable FALSIFY-* test     |
| 3  | Sovereign                   | No PyTorch in the production path; GGUF/APR/SafeTensors only          |
| 4  | Lean on existing artifacts  | Reuse `contracts/model-families/qwen2.yaml` and `llama.yaml` — do not fork |
| 5  | Dogfood tooling             | `apr qa`, `apr bench`, `apr trace`, `apr eval` — never bespoke scripts |
| 6  | Binary gates                | Every GATE-SHIP-* is pass/fail; no partial credit                     |
| 7  | Five-Whys on failure        | Any FALSIFY-* failure triggers documented Hansei (§10) before retry   |

---

## 4. Model 1 — apr-leaderboard (Distilled Qwen2.5-Coder-7B)

> **⚠ 2026-04-17 audit (v2.0.0):** The student checkpoint that was the subject of this section
> produces garbage tokens (see §1.5). MODEL-1 v1.0.0 as specified cannot ship. This section is
> retained unchanged as historical scope; the path forward is in §12 (teacher-first expedited ship).

### 4.1 Current State

- Teacher: `Qwen/Qwen2.5-Coder-7B-Instruct` (matches `contracts/model-families/qwen2.yaml` 7B variant).
- Student: same architecture, distilled on 20K code-instruction pairs.
- ~~Measured: **87.20% HumanEval pass@1** (source: POC notebook, pre-audit).~~
  **[v2.0.0] Falsified 2026-04-17:** this figure was a pre-distillation *few-shot* teacher score
  mis-attributed to the distilled student. The distilled checkpoint's actual pass@1 under `apr eval`
  is ~0 (garbage output). See §1.5 and `contracts/eval-harness-humaneval-v1.yaml` v1.1.0.
- Format: SafeTensors (HF-native), not yet exported to GGUF or APR.
- Eval: ran on reference Python harness; `apr eval` run terminated after garbage output detected.

### 4.2 Acceptance Criteria

| ID            | Criterion                                                                 | Verification            |
|---------------|---------------------------------------------------------------------------|-------------------------|
| AC-SHIP1-001  | Student weights load via `realizar::Model::load_safetensors`              | FALSIFY-SHIP-001        |
| AC-SHIP1-002  | `apr run <model>.safetensors --prompt "def fib(n):"` emits valid Python   | FALSIFY-SHIP-002        |
| AC-SHIP1-003  | Convert to APR via `apr convert --quantize q4_k_m`; round-trip weights match (cos ≥ 0.999) | FALSIFY-SHIP-003 |
| AC-SHIP1-004  | Export to GGUF via `apr export --format gguf`; loads in llama.cpp         | FALSIFY-SHIP-004        |
| AC-SHIP1-005  | `apr eval --benchmark humaneval` reproduces ≥86.00% pass@1 (allow 1.2% noise) | FALSIFY-SHIP-005     |
| AC-SHIP1-006  | `apr qa <model>` — all 8 gates PASS (Golden Output, layout, tensor stats, etc.) | FALSIFY-SHIP-006 |
| AC-SHIP1-007  | `apr bench` decode throughput ≥30 tok/s on RTX 4090 (7B Q4_K target)      | FALSIFY-SHIP-007        |
| AC-SHIP1-008  | Chat template (`contracts/chat-templates-v1.yaml`) applies cleanly        | FALSIFY-SHIP-008        |
| AC-SHIP1-009  | License & provenance recorded in `model.apr` metadata (Qwen2 Apache-2.0) | FALSIFY-SHIP-009        |
| AC-SHIP1-010  | Published artifact URL resolves; SHA-256 matches manifest                 | FALSIFY-SHIP-010        |

### 4.3 Critical Path (MODEL-1)

```
[checkpoint.safetensors] ──► AC-001 load ──► AC-002 run ──► AC-005 eval (baseline)
                                                 │                    │
                                                 ▼                    ▼
                                        AC-008 chat-template   AC-006 qa gates
                                                 │                    │
                                                 ▼                    ▼
                                         AC-003 convert ──► AC-007 bench
                                                 │
                                                 ▼
                                         AC-004 export gguf
                                                 │
                                                 ▼
                                         AC-009 metadata ──► AC-010 publish
```

### 4.4 Contract Registry (MODEL-1)

Leverages 28 existing contracts from the apr-leaderboard POC, promoted into the monorepo:

| Kind             | Contract                                              | Status      |
|------------------|-------------------------------------------------------|-------------|
| model-family     | `contracts/model-families/qwen2.yaml`                 | EXISTS      |
| tensor-layout    | `contracts/tensor-layout-v1.yaml`                     | EXISTS      |
| chat-template    | `contracts/chat-templates-v1.yaml` (qwen2 variant)    | EXISTS      |
| eval-harness     | `contracts/eval-harness-humaneval-v1.yaml`            | **NEW**     |
| distillation     | `contracts/distillation-pipeline-v1.yaml`             | **NEW**     |
| publish-manifest | `contracts/publish-manifest-v1.yaml`                  | **NEW**     |

---

## 5. Model 2 — albor (Sovereign 370M Python Code Completion)

### 5.1 Current State

- Architecture: LLaMA-family decoder, 370M params (hidden=1024, layers=24, heads=16, kv_heads=4).
  Slot: registered as a new variant under `contracts/model-families/llama.yaml` `370m`.
- Tokenizer: BPE over 50K vocab, Python-biased corpus.
- Training data: 60GB deduplicated Python (The Stack v2 filtered subset).
- Target: ≥30% HumanEval pass@1 (baseline reference: CodeParrot 1.1B ≈ 4%, StarCoderBase 1B ≈ 15.4%).
- Current blocker: pretraining run not yet executed end-to-end via `entrenar` CUDA path.

### 5.2 Acceptance Criteria

| ID            | Criterion                                                                 | Verification           |
|---------------|---------------------------------------------------------------------------|------------------------|
| AC-SHIP2-001  | Architecture registered in `contracts/model-families/llama.yaml` 370m     | FALSIFY-SHIP-011       |
| AC-SHIP2-002  | Tokenizer trained; `apr tokenize` round-trip exact on 10K held-out docs   | FALSIFY-SHIP-012       |
| AC-SHIP2-003  | `entrenar` pretraining loop reaches target loss (CE ≤ 2.2 on val)         | FALSIFY-SHIP-013       |
| AC-SHIP2-004  | Training on RTX 4090 completes within 21 days (hardware budget)           | FALSIFY-SHIP-014       |
| AC-SHIP2-005  | Checkpoint weights saved as `.apr` (native format, no PyTorch)            | FALSIFY-SHIP-015       |
| AC-SHIP2-006  | `apr qa <model>.apr` — all 8 gates PASS                                   | FALSIFY-SHIP-016       |
| AC-SHIP2-007  | `apr run` produces syntactically valid Python on 100 held-out prompts     | FALSIFY-SHIP-017       |
| AC-SHIP2-008  | `apr eval --benchmark humaneval` ≥30.0% pass@1                            | FALSIFY-SHIP-018       |
| AC-SHIP2-009  | GGUF export loads in llama.cpp AND produces matching tokens (tol ≤ 1e-3)  | FALSIFY-SHIP-019       |
| AC-SHIP2-010  | `apr bench` decode ≥100 tok/s on RTX 4090 (370M target)                   | FALSIFY-SHIP-020       |
| AC-SHIP2-011  | Training reproducible: seed fixed, two runs produce identical first 100 steps | FALSIFY-SHIP-021   |
| AC-SHIP2-012  | Weights + tokenizer + config published with CC-BY-4.0 data provenance     | FALSIFY-SHIP-022       |

### 5.3 Critical Path (MODEL-2)

```
[llama.yaml 370m entry] ──► AC-001 ──► AC-002 tokenizer
                                               │
                                               ▼
                                      AC-011 reproducibility check (dry run, 100 steps)
                                               │
                                               ▼
                                      AC-003 pretraining loop
                                               │
                                               ▼
                                      AC-004 hardware budget ── (MONITOR) ──► AC-005 save .apr
                                                                                    │
                                                              ┌─────────────────────┼─────────────────────┐
                                                              ▼                     ▼                     ▼
                                                       AC-006 qa gates      AC-007 run valid     AC-008 humaneval
                                                                                                         │
                                                                             ┌───────────────────────────┤
                                                                             ▼                           ▼
                                                                      AC-009 gguf export         AC-010 bench
                                                                             │
                                                                             ▼
                                                                      AC-012 publish
```

### 5.4 Contract Registry (MODEL-2)

Leverages 54 contracts from the albor POC, promoted into the monorepo:

| Kind             | Contract                                                   | Status  |
|------------------|------------------------------------------------------------|---------|
| model-family     | `contracts/model-families/llama.yaml` (add 370m variant)   | AMEND   |
| tokenizer        | `contracts/tokenizer-bpe-v1.yaml`                          | **NEW** |
| dataset          | `contracts/dataset-thestack-python-v1.yaml`                | **NEW** |
| training-loop    | `contracts/training-loop-pretrain-v1.yaml`                 | **NEW** |
| checkpoint       | `contracts/checkpoint-apr-native-v1.yaml`                  | **NEW** |
| eval-harness     | `contracts/eval-harness-humaneval-v1.yaml` (shared)        | SHARED  |
| publish-manifest | `contracts/publish-manifest-v1.yaml` (shared)              | SHARED  |

---

## 6. Compound Ship Gates

All gates are binary; any failure blocks publish.

| Gate             | Description                                                    | Blocks          |
|------------------|----------------------------------------------------------------|-----------------|
| GATE-SHIP-001    | MODEL-1: all 10 AC-SHIP1-* PASS                                | MODEL-1 publish |
| GATE-SHIP-002    | MODEL-2: all 12 AC-SHIP2-* PASS                                | MODEL-2 publish |
| GATE-SHIP-003    | Both models: `apr qa` Golden Output never regresses post-quantize | publish     |
| GATE-SHIP-004    | HumanEval harness produces identical score on two consecutive runs (seed=0) | AC-005, AC-008 |
| GATE-SHIP-005    | License metadata is present AND matches upstream declaration   | publish         |
| GATE-SHIP-006    | GGUF round-trip: APR → GGUF → load in llama.cpp → first-token match (tol ≤ 1e-3) | AC-004, AC-009 |
| GATE-SHIP-007    | No unwrap() in new code (enforced by `.clippy.toml`)           | merge           |
| GATE-SHIP-008    | Contract density: every new public fn has `#[contract]`        | merge           |
| GATE-SHIP-009    | CI green: `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test --workspace` | merge |
| GATE-SHIP-010    | `cargo deny check advisories` — zero vulnerabilities in weight/tokenizer dependencies | merge |
| GATE-SHIP-011    | PMAT quality score ≥ A- (project), TDG ≥ 90                    | merge           |
| GATE-SHIP-012    | Coverage ≥ 95% line on new modules (per `.pmat-gates.toml`)    | merge           |

---

## 7. Falsification Tests

Each test is named, executable, and has a defined failure signal.

### 7.1 MODEL-1 Falsification (12 tests)

| ID                 | Test                                                              | Failure Signal                         |
|--------------------|-------------------------------------------------------------------|----------------------------------------|
| FALSIFY-SHIP-001   | `realizar::Model::load_safetensors(path)` returns Ok              | Err(_) returned                        |
| FALSIFY-SHIP-002   | Run `apr run ... --prompt "def fib(n):"`; parse output as Python AST | SyntaxError                         |
| FALSIFY-SHIP-003   | Convert then compare per-layer cosine similarity                  | any layer cos < 0.999                  |
| FALSIFY-SHIP-004   | Shell out to `llama-cli` on exported GGUF; prompt → logits        | llama.cpp exit ≠ 0                     |
| FALSIFY-SHIP-005   | Run HumanEval 164× via `apr eval`; pass@1 computed                | pass@1 < 86.00%                        |
| FALSIFY-SHIP-006   | `apr qa <model>` exit code = 0                                    | any gate reports FAIL                  |
| FALSIFY-SHIP-007   | `apr bench --iterations 5 --max-tokens 128`; median tok/s         | median < 30                            |
| FALSIFY-SHIP-008   | Render chat template on canonical system+user; diff vs golden     | diff ≠ 0                               |
| FALSIFY-SHIP-009   | `apr inspect <model>.apr`; grep for `license: apache-2.0`         | missing or mismatched                  |
| FALSIFY-SHIP-010   | curl + sha256sum against manifest                                 | hash mismatch or 404                   |
| FALSIFY-SHIP-023   | Re-run AC-005 on second day; score drift                          | drift > 1.2 pp                         |
| FALSIFY-SHIP-024   | Prompt-injection torture suite (50 adversarial inputs)            | any panic or NaN in logits             |

### 7.2 MODEL-2 Falsification (10 tests)

| ID                 | Test                                                              | Failure Signal                         |
|--------------------|-------------------------------------------------------------------|----------------------------------------|
| FALSIFY-SHIP-011   | `llama.yaml` `370m` entry validates against `_schema.yaml`        | schema error                           |
| FALSIFY-SHIP-012   | Tokenize 10K docs, detokenize, byte-compare                       | any byte mismatch                      |
| FALSIFY-SHIP-013   | Training val CE at final step                                     | CE > 2.2                               |
| FALSIFY-SHIP-014   | Wall-clock from train start to final checkpoint                   | > 21 days                              |
| FALSIFY-SHIP-015   | Load checkpoint via `apr inspect`; count params                   | params ≠ 370M ± 1%                     |
| FALSIFY-SHIP-016   | `apr qa <model>.apr` exit code                                    | any gate FAIL                          |
| FALSIFY-SHIP-017   | 100 prompts → Python AST parse                                    | ≥2 SyntaxError (tolerate ≤1)           |
| FALSIFY-SHIP-018   | `apr eval --benchmark humaneval` pass@1                           | < 30.0%                                |
| FALSIFY-SHIP-019   | GGUF export; first-token probability vs APR                       | |Δp| > 1e-3 on top-1                   |
| FALSIFY-SHIP-020   | `apr bench` median tok/s on RTX 4090                              | < 100                                  |
| FALSIFY-SHIP-021   | Run training 100 steps × 2 with seed=0; diff loss trajectories    | any step diff > 1e-6                   |
| FALSIFY-SHIP-022   | `apr inspect`; check `license`, `data_source`, `data_license`     | any field missing                      |

---

## 8. Execution Plan

### 8.1 Phase DAG

```
                    ┌─────────────────────────────────────┐
                    │          Phase 0: Scaffold          │
                    │  (contracts, schema, test harness)  │
                    └──────────────────┬──────────────────┘
                                       │
                      ┌────────────────┴────────────────┐
                      │                                 │
                      ▼                                 ▼
         ┌─────────────────────────┐       ┌─────────────────────────┐
         │   Phase 1-A (MODEL-1)   │       │   Phase 1-B (MODEL-2)   │
         │  36h — packaging path   │       │  Week 1: tokenizer + dry │
         │                         │       │  run (AC-001,002,011)    │
         │  AC-001..004 load/conv  │       └───────────┬─────────────┘
         └───────────┬─────────────┘                   │
                     ▼                                 ▼
         ┌─────────────────────────┐       ┌─────────────────────────┐
         │  Phase 2-A: eval + qa   │       │  Phase 2-B: pretraining │
         │  AC-005..008            │       │  Weeks 2-3: AC-003,004  │
         └───────────┬─────────────┘       └───────────┬─────────────┘
                     ▼                                 ▼
         ┌─────────────────────────┐       ┌─────────────────────────┐
         │  Phase 3-A: publish     │       │  Phase 3-B: eval + qa   │
         │  AC-009,010  — SHIP-1   │       │  Week 4: AC-005..012    │
         └─────────────────────────┘       └───────────┬─────────────┘
                                                       ▼
                                           ┌─────────────────────────┐
                                           │  Phase 4-B: publish     │
                                           │  SHIP-2                 │
                                           └─────────────────────────┘
```

Phase 1-A and Phase 1-B are independent and run in parallel.

### 8.2 Effort Budget

| Phase | Model    | Effort       | Calendar  | Owner        |
|-------|----------|--------------|-----------|--------------|
| 0     | shared   | 6 h          | day 0     | eng          |
| 1-A   | MODEL-1  | 10 h         | day 1     | eng          |
| 2-A   | MODEL-1  | 12 h         | day 2     | eng          |
| 3-A   | MODEL-1  | 4 h          | day 3     | eng          |
| 1-B   | MODEL-2  | 40 h         | week 1    | eng          |
| 2-B   | MODEL-2  | compute-bound | weeks 2-3 | GPU node    |
| 3-B   | MODEL-2  | 16 h         | week 4    | eng          |
| 4-B   | MODEL-2  | 4 h          | week 4    | eng          |
| **Σ** |          | 92 h + 2 wk  | ~4 weeks  |              |

### 8.3 Integration with `apr run`

After SHIP, both models must satisfy:

```bash
apr run ./model-1.apr --prompt "def quicksort(arr):"       # MODEL-1
apr run ./model-2.apr --prompt "def binary_search(xs, t):" # MODEL-2
```

Both paths resolve through `realizar::Model` (see `crates/aprender-serve/CLAUDE.md` Realizar-First Architecture).
No code path in `aprender-core` may invoke generation.

---

## 9. Risk Matrix

| # | Risk                                           | Probability | Impact | Mitigation                                                           |
|---|------------------------------------------------|-------------|--------|----------------------------------------------------------------------|
| 1 | HumanEval eval non-deterministic               | MED         | HIGH   | seed=0, greedy; GATE-SHIP-004 enforces two-run identity              |
| 2 | GGUF export has tensor-layout bug (LAYOUT-001) | HIGH        | HIGH   | FALSIFY-SHIP-019 parity check; reuse `layout_contract.rs` validator  |
| 3 | MODEL-2 training diverges                      | MED         | HIGH   | AC-SHIP2-003 loss gate; fallback = reduce LR, resume from ckpt       |
| 4 | RTX 4090 insufficient for 370M in 21 days      | LOW         | HIGH   | AC-SHIP2-004 budget; overflow → rent 2× H100 week 3                  |
| 5 | Teacher (Qwen2.5-Coder-7B) license ambiguity   | LOW         | MED    | AC-SHIP1-009; confirm Apache-2.0 in `config.json`                    |
| 6 | `apr convert` quantize drops accuracy > 1pp    | MED         | MED    | FALSIFY-SHIP-003 cos-sim gate; fallback = Q5_K_M                     |
| 7 | Tokenizer round-trip bytes mismatch            | LOW         | HIGH   | FALSIFY-SHIP-012 on 10K corpus                                       |
| 8 | `cargo install aprender` breaks during release | MED         | HIGH   | CI `cargo install --path .` smoke test (GATE-SHIP-009)               |
| 9 | HF artifact hosting outage                     | LOW         | LOW    | dual-publish to HF + self-hosted bucket                              |
| 10| CUDA JIT regression (trueno#200/203)           | MED         | MED    | Pin trueno version per memory note 2026-03-22                        |

---

## 10. Failure Protocol (Hansei)

Any FALSIFY-SHIP-* failure triggers the following sequence. No retry is permitted before completion.

### 10.1 Five Whys

1. What check failed? (name the FALSIFY-*)
2. What invariant did it violate?
3. What code path was responsible?
4. Why was the code path wrong? (bug class: layout, numeric, eval harness, toolchain)
5. Why did no earlier gate catch it? (contract gap → file follow-up)

### 10.2 Decision Gate

| Condition                                              | Action                         |
|--------------------------------------------------------|--------------------------------|
| Single test failed, root cause known, fix ≤ 2 h        | Fix + re-run full gate         |
| Multiple tests failed OR root cause unknown            | Escalate to design review      |
| AC breaks but SHIP is blocking deadline                | Reduced-gate ship (below)      |
| Hardware / compute budget breached                     | Full failure escalation        |

### 10.3 Reduced-Gate Ship (emergency only)

Acceptable only with written Noah Gift approval. A reduced ship may drop:
- AC-SHIP1-007 / AC-SHIP2-010 (bench speed) — ship with "beta performance" label
- AC-SHIP1-010 / AC-SHIP2-012 (artifact publication) — ship to internal bucket only

All 8 `apr qa` gates (Golden Output, layout, tensor stats, license, etc.) MUST still pass.

### 10.4 Full Failure Escalation

If neither model ships within the budget, this specification is void. Retrospective must
answer: (a) which assumption was wrong, (b) what contract should have caught it earlier,
(c) what gets deleted from scope before restart.

---

## 12. Expedited Ship Plan (v2.0.0 — teacher-first)

**Goal:** publish ONE artifact within **10 engineering hours** of 2026-04-17 to falsify the null
hypothesis "the stack cannot produce shippable weights."

**Strategy:** ship the **teacher** (`qwen2.5-coder-7b-instruct-q4k.apr`) under a new artifact ID
`paiml/qwen2.5-coder-7b-apache-q4k-v1`. Defer distillation proof to v1.1. Defer MODEL-2 to v2.0.

### 12.1 Pre-requisite: plug the Golden Output gate gap

Before any publish, `apr qa` must be configured so that **Golden Output failure blocks publish**.
Today it is reported but non-fatal — exactly the hole that let the v1.0.0 plan rely on a
garbage checkpoint for 14 days before audit. Track as contract amendment to `apr-qa-v1.yaml`
(or equivalent); must land before §12.2.

### 12.2 Teacher-first critical path (10h budget)

```
[qwen2.5-coder-7b-instruct-q4k.apr]      # already in apr-leaderboard/checkpoints/, 7.5 GB
         │
         ▼
  EX-01  apr qa --require-golden-output   # must PASS after §12.1 gate fix  (1 h)
         │
         ▼
  EX-02  apr eval --benchmark humaneval   # reproduces ≥84.5 pass@1 (noise-band of 85.98)  (2 h)
         │
         ▼
  EX-03  Write contracts/publish-manifest-v1.yaml entry    (1 h)
           - sha256, size_bytes, license=Apache-2.0
           - provenance.pipeline=finetune
           - provenance.parent=Qwen/Qwen2.5-Coder-7B-Instruct
           - provenance.recipe=contracts/model-families/qwen2.yaml
         │
         ▼
  EX-04  Upload artifact to HF Hub AND self-hosted bucket  (2 h)
         │
         ▼
  EX-05  Verify manifest: sha256 match, URL 200, SPDX valid  (1 h)
         │
         ▼
  EX-06  apr pull <published_id> → local file; re-run EX-02 from downloaded artifact  (2 h)
         │
         ▼
  EX-07  Tag release in spec + announce  (1 h)
```

### 12.3 Expedited Acceptance Criteria

| ID            | Criterion                                                                        | Verification        |
|---------------|----------------------------------------------------------------------------------|---------------------|
| AC-EX-001     | Golden Output gate is a HARD BLOCKER in `apr qa`                                 | FALSIFY-EX-001      |
| AC-EX-002     | Teacher passes all 8 `apr qa` gates including Golden Output                      | FALSIFY-EX-002      |
| AC-EX-003     | `apr eval --benchmark humaneval` on teacher ≥84.5% pass@1 (85.98 − 1.5 noise)   | FALSIFY-EX-003      |
| AC-EX-004     | `publish-manifest-v1.yaml` instance for artifact passes `apr validate-manifest`  | FALSIFY-EX-004      |
| AC-EX-005     | `apr pull paiml/qwen2.5-coder-7b-apache-q4k-v1` resolves + SHA-256 matches       | FALSIFY-EX-005      |
| AC-EX-006     | `apr run <published>.apr --prompt "def fib(n):"` emits syntactically valid Python | FALSIFY-EX-006     |
| AC-EX-007     | Parallel eval lane: N-shard run on ≥2 hosts matches single-host pass@1 (Δ ≤ 0.01 pp) and completes in ≤ `single_host_wall_time / N × 1.25` | FALSIFY-SHARD-001..004 |

### 12.4 Explicit Scope Cut (v2.0.0)

Moved out of v1 ship:
- **Distilled student artifact** → v1.1 (requires diagnosis per `validation_result_v1_1` ACT-01..03,
  then re-distillation with contract-gated Golden Output at each epoch).
- **MODEL-2 (albor sovereign)** → v2.0 (3+ weeks of compute; no reason to couple to MODEL-1 ship).
- **GGUF round-trip export** (AC-SHIP1-004) → v1.1 (teacher already has GGUF on HF).

### 12.5 What falsifies the expedited plan

| Condition                                         | Action                                              |
|---------------------------------------------------|-----------------------------------------------------|
| AC-EX-002 FAIL on teacher                         | Pipeline regressed — block ship, investigate realizar |
| AC-EX-003 pass@1 < 84.5                           | Harness drift since 2026-03-28; do not ship until resolved |
| AC-EX-004 manifest invalid                        | Fix manifest schema compliance, retry                |
| AC-EX-005 SHA-256 mismatch                        | Re-upload; investigate CDN/transit corruption        |
| Any EX-* step takes > 2× budget                   | Escalate; triggers §13.2 retrospective update        |
| Shard merged pass@1 differs from single-host by > 0.01 pp | Parity FAIL — block ship, investigate shard determinism (FALSIFY-SHARD-003) |
| Any shard reports missing / duplicate task_ids    | Completeness or disjointness FAIL — block ship (FALSIFY-SHARD-001/002) |

### 12.6 Parallel Eval Lane (post-hoc lesson, 2026-04-17)

**Problem surfaced during v2.0.0 ship.** EX-02 (single-host HumanEval on yoga) ran
serially for ~2 hours while `gx10` (Blackwell GB10, `apr-cli` inference unaffected
by PMAT-587 JIT issues) and any Lambda-Labs GPU instance sat idle. 5-Whys (recorded
in contract `eval-sharding-v1.yaml::five_whys`):

1. **Why only yoga?** The orchestration script accepted a single `MODEL_PATH` and
   invoked `apr run --batch-jsonl` once, consuming all 164 tasks serially.
2. **Why serial batch?** `eval-pass-at-k.sh` (inherited from apr-leaderboard) has
   no shard dimension; it assumes one GPU.
3. **Why wasn't sharding added?** EX-02 was treated as a monolithic §12.2 step;
   decomposing "generate N completions" into `generate N/k × k hosts` was not
   considered because the 10h budget was written assuming yoga alone.
4. **Why was the budget yoga-alone?** `gx10` was mentally categorized as "training,
   blocked on JIT bug" without separating the inference path, which works today.
5. **Root cause.** Spec optimized for *matching existing tooling* (one-host eval
   harness) instead of *minimizing critical path*. A 2-way shard cuts EX-02 from
   ~2h → ~1h; a 3-way shard (yoga+gx10+Lambda) to ~40 min.

#### 12.6.1 Scope

This lane is **post-hoc for v2.0.0** (sunk cost on the in-flight serial run) and
**pre-requisite for v1.1 / v2.0** future evals (distilled student, MODEL-2 sovereign,
multi-seed reproducibility runs per FALSIFY-PUBLISH-RECIPE-001).

#### 12.6.2 Architecture

```
benchmark.jsonl  ──(round-robin split, stride N)──►  shard_0.jsonl … shard_{N-1}.jsonl
                                                           │ │ … │
                                                           ▼ ▼   ▼
                                                       host_0 host_1 … host_{N-1}
                                                       (yoga) (gx10) … (lambda)
                                                           │ │ … │
                                                           ▼ ▼   ▼
                                                   humaneval_shard_i.json (each host)
                                                           │ │ … │
                                                           └─┴───┘
                                                              ▼
                                          eval-shard-merge.py: concat problems[],
                                          recompute Chen pass@1 → humaneval_merged.json
```

- **Shard algorithm.** Round-robin stride: task `i` goes to host `i mod N`.
  Evens out per-task cost variance (long prompts, long generations) without
  needing a pre-estimated cost model.
- **Model sync.** `rsync -c` (content-checksum) pushes the .apr + tokenizer to
  each host once; subsequent runs are no-ops.
- **Merge.** Per-shard result JSONs share the `eval-pass-at-k.sh` schema
  (`problems[]`, `results.passed`, `results.total`). Merge = concat `problems`,
  sum totals, recompute pass@1 using Chen et al. unbiased estimator on merged
  array.

#### 12.6.3 Acceptance (AC-EX-007 discharge)

Run the 4 FALSIFY-SHARD tests in `contracts/eval-sharding-v1.yaml`:

- **FALSIFY-SHARD-001 (completeness):** `sum(shard_i.total) == benchmark.total`
  and every benchmark task_id appears in exactly one shard result.
- **FALSIFY-SHARD-002 (disjointness):** no task_id appears in two shards.
- **FALSIFY-SHARD-003 (determinism parity):** at temperature=0.0, completions for
  task T on host A == completions for task T on host B for a 16-task probe set.
- **FALSIFY-SHARD-004 (merged-score identity):** reshard an existing single-host
  humaneval_*.json result by task_id; merged pass@1 matches within 0.01 pp of the
  original.

Evidence location: `evidence/ship-two-001/shard-eval/`.

#### 12.6.4 Non-goals for this lane

- **Dynamic load-balancing.** Static stride-N is sufficient for ≤5 hosts and
  benchmarks under a few thousand tasks.
- **Remote-managed model caches.** `rsync -c` on each invocation is <2 min on
  gigabit for a 7.5 GB .apr; optimizing further is premature.
- **Fault-tolerant shard retry.** If one host dies mid-run, operator re-runs the
  missing shard manually — no automatic reassignment. (Revisit for v1.1 if
  experienced in practice.)

### 12.7 Dogfood Gate + Three-Format Ship (2026-04-18 amendment)

**Problem surfaced during EX-04.** The first-cut `ex-04-upload-hf.sh` called
`uv run --with huggingface-hub python3` instead of our own product. That is the
same failure class as §13.2 cause 7 ("Tooling investment vs tooling usage"):
we have `apr publish`, and we should be shipping through it.

Two product gaps had to be closed before EX-04 could run through `apr publish`:

1. `apr publish` did not natively consume `publish-manifest-v1.yaml` or upload
   arbitrary sidecar files (tokenizer.json, per-format manifests).
2. No contract stated that ships must be published in multiple ecosystem
   formats, and no contract stated the required safetensors dtype.

Both gaps are now closed by **contract `contracts/apr-cli-publish-extra-v1.yaml`
(F-PUBLISH-EXTRA-001)**, a peer of `publish-manifest-v1.yaml` that adds:

- `manifest_upload_roundtrip` — `apr publish --manifest <yaml>` validates, hashes
  the declared artifact locally, and aborts before network I/O on mismatch.
- `extra_file_passthrough` — `apr publish --extra-file <path>` (repeatable) uploads
  sidecars verbatim in CLI-argument order.
- `no_readme_when_manifest` — when `--manifest` is passed, the auto-generated
  `README.md` is suppressed; the manifest is the provenance document.
- `dogfood_shell_script` — `scripts/ship-two-001/ex-04-upload-hf.sh` MUST invoke
  `apr publish`; `uv run`, `huggingface_hub`, `huggingface-cli`, and `pip install`
  are forbidden in the ship script.
- `three_format_preference` — every SHIP-TWO-* release publishes `.apr`,
  `.safetensors`, and `.gguf` side-by-side in the same HF repo.
- `safetensors_dtype_fp16` — ship-bound `.safetensors` MUST be exported via
  `apr export --format safetensors --quantize fp16`. Default-fp32 export doubles
  disk/network cost; the `transformers` / `candle` / HF ecosystem reads fp16
  natively. Expected 7B sizes: `.apr` ≈ 7.5 GB, `.safetensors-fp16` ≈ 14 GB,
  `.gguf` ≈ 7.5 GB (a fp32 safetensors at ≈ 29 GB is forbidden for ships).

Discharged by falsification tests **FALSIFY-PUB-EXTRA-001 through -010**
(contract `apr-cli-publish-extra-v1.yaml` v1.2.0):

- **-001..-004** covered by `apr publish` unit tests
- **-005** dogfood gate (no Python in ship scripts)
- **-006** post-upload sha256 round-trip (discharged by EX-05)
- **-007** three-format HF repo (discharged by EX-05 + list-repo-files)
- **-008** no Python in `ex-05-verify-manifest.sh` (discharged)
- **-009** corrupt-manifest pre-flight abort (shows exit code 5 blocking upload)
- **-010** `preflight_validate_manifest` function present + invoked before any `publish_format`

Additionally, **FALSIFY-PM-007** (safetensors header dtype Poka-Yoke, contract
`publish-manifest-v1.yaml` v1.1.0) fires automatically inside every pre-flight
invocation for `.safetensors` format. Eight unit tests cover both the happy path
and the exact §12.7.2 ship-blocker scenario (`pm007_f32_weight_when_fp16_declared_fails`).

The unit test matrix (`cargo test -p apr-cli validate_manifest`) runs 21 tests on
every push; the end-to-end pre-flight gate runs against real 15 GiB artifacts
only at ship time (evidence at `evidence/ship-two-001/ex-04-preflight-gate-smoketest.json`).

#### 12.7.1 Revised EX-04 invocation

EX-04 is now **one command per format**, pointed at a per-format manifest in
`contracts/publish-manifests/`:

```
apr publish /mnt/nvme-raid0/models/ship-two-001/ \
    paiml/qwen2.5-coder-7b-apache-q4k-v1 \
    --manifest contracts/publish-manifests/paiml-qwen2.5-coder-7b-apache-q4k-v1-apr.yaml \
    --extra-file /mnt/nvme-raid0/models/ship-two-001/tokenizer.json
```

and repeats for `-safetensors.yaml` and `-gguf.yaml`. Each invocation runs the
pre-flight sha256 guard *before* opening any network socket, then uploads the
artifact + tokenizer + manifest.yaml.

#### 12.7.2 What falsifies the dogfood gate

| Condition                                                                                   | Action                                                  |
|---------------------------------------------------------------------------------------------|---------------------------------------------------------|
| `ex-04-upload-hf.sh` contains `uv run` / `huggingface_hub` / `huggingface-cli` / `pip install` | FALSIFY-PUB-EXTRA-005 FAIL — fix script, rerun           |
| `ex-05-verify-manifest.sh` contains `uv run` / `python3` / `pip` / `huggingface_hub`        | FALSIFY-PUB-EXTRA-008 FAIL — ex-05 must use `apr validate-manifest --live` |
| HF repo missing any of `.apr` / `.safetensors` / `.gguf` after EX-04                        | FALSIFY-PUB-EXTRA-007 FAIL — re-upload missing format    |
| Staged `.safetensors` header declares F32 for weight tensors when manifest says fp16        | **FALSIFY-PM-007 FAIL — pre-flight gate aborts with exit 2 BEFORE any network I/O; re-export with `--quantize fp16`** |
| Staged artifact's local sha256 ≠ per-format manifest sha256 at ship time                    | **FALSIFY-PUB-EXTRA-009 FAIL — pre-flight gate aborts with exit 5 BEFORE any network I/O** |
| `preflight_validate_manifest` removed or reordered after `publish_format`                   | FALSIFY-PUB-EXTRA-010 FAIL — Poka-Yoke bypassed; re-sequence |
| Any uploaded artifact's CDN-served sha256 ≠ per-format manifest sha256                      | FALSIFY-PUB-EXTRA-006 FAIL (post-upload) — investigate transit corruption |

**Ship-time Poka-Yoke:** prior to contract v1.2.0 (2026-04-18), the dtype mismatch
row above required post-hoc detection and a deprecation cycle on HF Hub. With
PM-007 + the pre-flight gate, it is structurally unreachable: an ex-04 invocation
with divergent bytes exits non-zero before the first HTTP connection opens.

---

## 13. Why Did This Take So Long? (Retrospective)

### 13.1 Timeline

- **2026-01-01 → 2026-04-17:** 3.5 months of work on the Sovereign AI Stack.
- 2141 commits to aprender (of which 181 = 8.5% are perf-path-to-1.5× commits).
- apr-leaderboard ran **12 distillation recipes** (a → l); multiple had broken checkpoint output.
- Commit `0fc5436 fix: LoRA merge was element-wise, not matrix multiply — root cause of distilled
  model garbage` on apr-leaderboard — **this exact failure class has been fixed before**.
- Commit `a20f234 docs: document Q4K roundtrip corruption blocker` — **also previously known**.
- Current distilled-v2 checkpoints are dated 2026-04-03; they sat **14 days** before any
  `apr qa`-driven audit.
- Spec SPEC-SHIP-TWO-001 v1.0.0 was written 2026-04-17, *after* the broken checkpoint had been
  sitting for 2 weeks and cited as "trained, needs packaging."

### 13.2 Root causes

1. **Contract came after POC, not before.** The 87.20% number lived in a recipe comment since
   Q1 and was promoted to a spec headline without a falsification test run. Design Principle 1
   (Contract-first) was violated by its own spec.
2. **`apr qa` gate matrix lets Golden Output fail silently.** A model that cannot generate "4"
   for "2+2=" passed `apr qa` overall because Golden Output was reported but non-blocking. Tensor
   Contract PASS became a false confidence signal. This is the structural defect that allowed
   broken weights to persist for 14 days.
3. **Perf work crowded out ship work.** 181 perf commits toward 1.5× Ollama parity between
   January and April; **zero** commits on publishing a *model* artifact (as opposed to a *crate*).
   Perf gains are visible in benchmarks; ship state was not gated anywhere.
4. **Monorepo reorg (APR-MONO) consumed weeks.** Phases 1–11 moved 70 crates, introduced shim
   layers, debugged publishing cycles — necessary ceremony but directly competed with model-ship
   bandwidth. Commits like "Phase 10d+10e done" / "Phase 11 (CI fix + publish babysit)" show
   sustained multi-week focus.
5. **Distillation recipe churn without shipping discipline.** Recipes a/b/c/d/e/f/g/h/i/j/k/l —
   twelve experiments, each generating a checkpoint — but no contract defining what makes a
   recipe's output "ship-quality." Each recipe was treated as a new chance; no recipe was ever
   contractually retired. `contracts/distillation-pipeline-v1.yaml` (per v1.0.0 §4.4) is listed as
   "NEW" but has never existed — the proof of this is that broken checkpoints sat unaudited for weeks.
6. **Two-model scope from the start.** v1.0.0 bundled distilled (quick) with sovereign
   (multi-week). This guaranteed the multi-week item would block the quick item from any
   expedited ship path. The fix is to ship MODEL-1 alone as a v1 and move MODEL-2 to v2.
7. **Tooling investment vs tooling usage.** `apr qa`, `apr trace`, `apr profile`, `apr diff` all
   exist and are well-built. They were not being *dogfooded on the shipping artifact* until
   2026-04-17. The audit that exposed this was a 10-minute `apr qa` invocation.
8. **"87.20%" was never in a results JSON.** 17 HumanEval result files exist in
   `apr-leaderboard/results/`; all 17 are teacher runs. The student has no recorded result. A
   spec claim not traceable to a results file is an unfalsified claim.

### 13.3 Lessons codified as contracts

| Contract (new)                                  | Prevents future occurrence of                                |
|-------------------------------------------------|--------------------------------------------------------------|
| `contracts/eval-harness-humaneval-v1.yaml` v1.1 | Headline pass@1 numbers without a results-JSON trail         |
| `contracts/publish-manifest-v1.yaml` v1.0       | Artifacts shipping without sha256 / license / provenance     |
| **APR-QA GATE AMENDMENT (§12.1)**               | Tensor Contract masking a Golden Output failure              |
| `contracts/distillation-pipeline-v1.yaml` (TBD) | Recipes run without per-epoch Golden Output gating           |

### 13.4 Publish-fastest rules going forward

1. **No claim without a JSON.** Every pass@1 number in any spec must be a `jq`-extractable field
   in a file under version control. If it isn't, it doesn't exist.
2. **Golden Output is a ship-blocking gate.** `apr qa` must exit non-zero if Golden Output fails,
   even when all structural gates pass.
3. **Ship-first, proof-second.** Ship the teacher in 10 hours; use the published artifact as the
   reference against which distillation is measured. Do not wait to ship because distillation
   isn't done.
4. **One artifact per release.** v1 = teacher. v1.1 = distilled (if/when it works). v2 =
   sovereign. Coupling them couples their risks.
5. **Dogfood before declaring "trained."** A checkpoint is not "trained" until `apr qa` and
   `apr eval` agree it is. Until then it is "saved."

---

## 11. References

### 11.1 Existing Contracts

- `contracts/model-families/qwen2.yaml` — Qwen2 architecture descriptor
- `contracts/model-families/llama.yaml` — LLaMA architecture descriptor
- `contracts/model-families/_schema.yaml` — family schema validator
- `contracts/tensor-layout-v1.yaml` — row-major APR invariant (LAYOUT-001/002)
- `contracts/chat-templates-v1.yaml` — chat-template engine spec
- `contracts/apr-cli-commands-v1.yaml` — 57-command CLI contract
- `contracts/publish-manifest-v1.yaml` v1.1.0 — artifact-shipping schema (sha256, provenance, license) + **FALSIFY-PM-007** safetensors header dtype Poka-Yoke
- `contracts/apr-cli-publish-extra-v1.yaml` v1.2.0 — **F-PUBLISH-EXTRA-001** (§12.7): manifest consumption, `--extra-file` passthrough, three-format ship, safetensors fp16 dtype, **preflight_validate_manifest** (FALSIFY-PUB-EXTRA-009/-010)
- `contracts/eval-harness-humaneval-v1.yaml` — pass@1 harness / AC-EX-003 floor
- `contracts/apr-model-qa-v1.yaml` — `apr qa` gate matrix / AC-EX-001/-002 (Golden Output hard-block)

### 11.2 Related Specifications

- `docs/specifications/aprender-train/hugging-face-distill-learn-pipeline-spec.md`
- `docs/specifications/aprender-train/comprehensive-qa-falsification.md`
- `docs/specifications/aprender-train/model-eval-framework-spec.md`
- `docs/specifications/aprender-monorepo-consolidation.md`

### 11.3 External

- HumanEval: Chen et al. 2021, *Evaluating Large Language Models Trained on Code*
- Qwen2.5-Coder: Hui et al. 2024, *Qwen2.5-Coder Technical Report*
- The Stack v2: BigCode, CC-BY-4.0

---

**END OF SPECIFICATION**
