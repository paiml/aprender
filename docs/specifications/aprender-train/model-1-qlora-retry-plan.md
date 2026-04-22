# MODEL-1 QLoRA Retry Plan (v3)

- **Parent spec:** [SHIP-TWO-001 / ship-two-models-spec.md](./ship-two-models-spec.md) §1.5, §12.4, AC-SHIP1-005
- **Status:** PROPOSED
- **Date:** 2026-04-18
- **Author:** PAIML Engineering
- **Task:** #86 (MODEL-1 retry, out of scope for current v2.10.0 teacher-only ship)
- **Scope flag:** This plan describes a FUTURE retry. It does NOT authorize compute spend.

---

## 1. Root Cause Summary (v2 failure, 2026-04-18 probe)

Pulled verbatim from `memory/project_ship_two_001_model1_qlora_divergence.md`:

- **Adapter never converged.** `checkpoints/instruct-qlora-7b/best/metadata.json` records
  `train_loss=15.41`, `val_loss=31.99`, `train_perplexity=val_perplexity=1e6`
  (saturated at the 1M cap). Only the epoch-0 artifact exists; `best/` and `epoch-0/`
  adapter safetensors are byte-identical — training halted at epoch 0 of 3.
- **Recipe vs. actual drift.** `distill-32b-7b-text.yaml` specifies `rank: 32` and
  `learning_rate: 0.0002`, but the recorded run used `lora_rank: 16`. LR=2e-4 is too
  hot for the actual rank-16 run, especially combined with soft-label temperature=4.0
  applied at runtime (not pinned in the recipe).
- **Non-training hypotheses FALSIFIED.** Embedded BPE loads cleanly (152064 vocab),
  `embed_tokens.weight` is byte-identical to the teacher, lm_head Q4K stats match
  teacher f32 within quant noise, `apr qa` Tensor Contract passes all 339 tensors.
  The `ylkoylkoylko…` mode-collapse is purely an adapter-space pathology.

---

## 2. Falsification Gates (MUST PASS before any merge/upload)

Each gate is binary; any FAIL blocks the retry. Gates must be evaluated IN ORDER.

| ID | Measurement | Pass rule | Evidence path | Actor |
|----|-------------|-----------|---------------|-------|
| **G1: Per-epoch train loss** | `train_loss` written to each `epoch-N/metadata.json` | `< 2.5` by end of epoch 0; strictly decreasing across epochs 0→1→2 | `checkpoints/instruct-qlora-7b-v3/epoch-{0,1,2}/metadata.json` | entrenar training script |
| **G2: Per-epoch val loss monotone OR early-stop** | `val_loss` per epoch | Monotone-decreasing OR early-stop triggers (patience=1) when val rises | `checkpoints/instruct-qlora-7b-v3/epoch-*/metadata.json` | entrenar scheduler |
| **G3: Val perplexity not saturated** | `val_perplexity` per epoch | `< 1e5` at end of epoch 0 (v2 hit 1e6 cap — any run still at cap = DIVERGED) | same metadata.json | entrenar metrics logger |
| **G4: Adapter byte-distinct across epochs** | `sha256` of adapter safetensors per epoch | `epoch-0 ≠ epoch-1 ≠ epoch-2`; `best/ == epoch-N` for the `N` with lowest val_loss | `evidence/ship-two-001/model-1-v3-adapter-hashes.txt` | PAIML eng (manual `sha256sum`) |
| **G5: Merged-and-quantized student passes `apr qa`** | `apr qa qwen2.5-coder-7b-distilled-v3-q4k.apr --require-golden-output` | All 8 gates PASS, including hard-gated Golden Output (per `apr-model-qa-v1.yaml` v1.1.0, FALSIFY-EX-001) | `evidence/ship-two-001/model-1-v3-qa.json` | PAIML eng via `apr qa --json` |
| **G6: HumanEval acceptance** | `scripts/eval-pass-at-k.sh` on merged APR | `pass@1 ≥ 30.0%` on the 164-task HumanEval set (AC-SHIP1-005 literal) | `apr-leaderboard/results/humaneval_model-1-v3_<date>.json` | PAIML eng on yoga (RTX 4090) |

G1–G5 gate adapter/merge; G6 gates acceptance. G6 is the AC-SHIP1-005 discharge signal.

---

## 3. Hyperparameter Changes from v2

| Parameter | v2 actual | v3 proposed | Rationale |
|-----------|-----------|-------------|-----------|
| `lora_rank` | 16 (recorded) | **32** | Match recipe (`distill-32b-7b-text.yaml::finetune.rank`); v2 drifted from recipe unexplainedly |
| `lora_alpha` | 32 (recorded) | **64** | Maintain `alpha = 2 × rank` ratio (standard QLoRA practice) |
| `learning_rate` | 2e-4 | **5e-5** | 4× reduction; 2e-4 is aggressive for 7B QLoRA even at rank-32, catastrophic at rank-16 |
| `distill_temperature` (soft labels) | 4.0 | **2.0** | Halve entropy of soft targets; 4.0 contributed to gradient noise (root cause §1). Recipe does not pin this — MUST be pinned in v3 CLI invocation |
| `warmup_steps` | unknown / default | **100 steps (or 3% of total, whichever greater)** | entrenar default is typically 0; linear warmup stabilizes first-epoch gradients and would have caught v2's epoch-0 blowup |
| `gradient_clip_norm` | unknown / default | **1.0** | Standard; limits rare-token gradient spikes that caused PPL→1e6 saturation |
| `epochs` | 3 planned / 0 completed | **3, with per-epoch G1–G3 gates** | No change in count; add gating so we halt on divergence instead of wasting compute |
| `batch_size` | unknown | **inherit from recipe** (unchanged) | Not suspected; out of scope for this retry |

Values marked "unknown" are inferred from entrenar defaults and were not recorded in the
v2 `metadata.json`. V3 MUST emit all hyperparameters into `metadata.json` for auditability
(contract: `apr-train-qlora-metadata-v1.yaml`, to be drafted alongside this retry).

---

## 4. Compute Budget

Estimates are for a single 3-epoch QLoRA run on the existing
`teacher-completions.jsonl` (~500K tokens, 2048-char max prompts). These are
ESTIMATES (no benchmarked v2 wall-clock is recorded).

| Host | GPU | Estimated wall-clock (3 epochs, rank 32) | Notes |
|------|-----|-----------------------------------------|-------|
| **yoga (primary)** | RTX 4090 24 GiB | ~6–9 hours | cuBLAS path; pre-compiled kernels; NO JIT issues. RECOMMENDED LANE. |
| **gx10 (contingent)** | Blackwell GB10 | UNAVAILABLE until trueno 0.4.36 | PMAT-587 Blackwell JIT pre-warming bug forces fused NF4 fallback (15.5 tok/s); entrenar training path currently DEGRADED per CLAUDE.md "SSC Training Infrastructure Status". Re-evaluate when trueno#200/203 land. |
| **Lambda Labs H100 (backup)** | H100 80 GiB | ~2–3 hours | Backup if yoga is occupied; requires hourly spend authorization. |

**Decision:** Run on yoga. Do NOT use gx10 until Blackwell JIT bug is resolved.
If yoga is saturated by Ship Two eval lane, queue the retry rather than move to gx10.

---

## 5. Acceptance for AC-SHIP1-005

Literal restatement from `ship-two-models-spec.md` §1.5 / MEMORY root-cause doc:

> **AC-SHIP1-005 (distilled student ≥30% HumanEval pass@1).**

Evaluation harness: `/home/noah/src/apr-leaderboard/scripts/eval-pass-at-k.sh`
(the same 164-task HumanEval harness used for the teacher 85.98% baseline
recorded in `results/humaneval_20260328_121327.json`).

Invocation (yoga, RTX 4090):

```bash
cd /home/noah/src/apr-leaderboard
./scripts/eval-pass-at-k.sh \
    checkpoints/qwen2.5-coder-7b-distilled-v3-q4k.apr \
    humaneval \
    --max-tokens 512 \
    --temperature 0.0
```

**PASS:** resulting `pass@1 ≥ 30.0` on 164 tasks, recorded JSON posted to
`evidence/ship-two-001/ac-ship1-005-v3.json`. **FAIL:** any pass@1 below 30.0
falsifies the retry — MODEL-1 stays shelved and ship remains teacher-only.

---

## 6. Decision Log

- **2026-04-17:** AC-SHIP1-005 FALSIFIED on v2 (see `memory/project_ship_two_001_model1_falsified.md`, task #55).
- **2026-04-18:** Root cause of v2 failure localized to QLoRA divergence
  (`memory/project_ship_two_001_model1_qlora_divergence.md`, task #84).
- **2026-04-18:** Ship decision for v2.10.0 = **teacher-only**. MODEL-1 retry
  placed **OUT OF SCOPE** for the current ship per spec §12.4 "Explicit Scope Cut."
- **This document (2026-04-18, task #86):** Plan drafted. PROPOSED status only;
  no compute authorized until (a) teacher-only v2.10.0 ships and (b) trueno 0.4.36
  lands (to unlock gx10 as a fallback lane) OR a yoga window opens.
- **Next action:** Review this plan, open a contract `apr-train-qlora-metadata-v1.yaml`
  to enforce G1–G4 at training time, then schedule the retry.

---

## Appendix A — Referenced Artifacts

- Broken v2 adapter: `/home/noah/src/apr-leaderboard/checkpoints/instruct-qlora-7b/best/`
- Broken v2 APR: `/home/noah/src/apr-leaderboard/checkpoints/qwen2.5-coder-7b-distilled-v2-q4k.apr`
- Recipe: `/home/noah/src/apr-leaderboard/configs/distill/distill-32b-7b-text.yaml`
- Teacher (ship-anchor): `/home/noah/src/apr-leaderboard/checkpoints/qwen2.5-coder-7b-instruct-q4k.apr`
- Teacher eval: `/home/noah/src/apr-leaderboard/results/humaneval_20260328_121327.json`
- QA contract: `contracts/apr-model-qa-v1.yaml` v1.1.0 (FALSIFY-EX-001)
- Eval harness: `/home/noah/src/apr-leaderboard/scripts/eval-pass-at-k.sh`
