# Roadmap: aprender/albor-370m (MODEL-2) — Path to Ship

**Document ID:** SPEC-SHIP-MODEL-2-ROADMAP
**Version:** 1.0.0
**Parent specs:**
- [aprender/albor-370m full spec (MODEL-2)](./ship-model-2-spec.md) — historical record, §5-§82
- [Ship Two Models Index](./ship-two-models-spec.md)
- [Shared methodology](./ship-shared-methodology.md)

**Purpose:** Focused, forward-looking spec for the only open model. The 3,290-line MODEL-2 spec is the source of truth for what *happened*; this roadmap is what we *work against*. Update this doc as items land. Reference §N markers point back to the full spec for context.

---

## 1. Ship goal

Publish `aprender/albor-370m` as the second model in the family:

- HuggingFace artifact `paiml/albor-370m-v1` with at least one usable format (APR, with optional SafeTensors + GGUF parity)
- `apr run paiml/albor-370m` produces coherent Python continuations
- HumanEval pass@1 ≥ 30% (below MODEL-1 teacher's 86.59% but proves the student is functional, not gibberish)
- All 10 AC-SHIP2-* falsifiers LIVE-discharged or PARTIAL with documented gap

**Current ship %: 79%.** Bounded path to 100%: see §6.

## 2. Current state (§82 snapshot, 2026-05-15)

| Item | Value |
|---|---|
| Best checkpoint | `/mnt/nvme-raid0/runs/model-2-p2a-5000steps-20260515-205805/ckpt/epoch-020.apr` (2.35 GiB F32) |
| Best val_loss | **4.7111** at epoch 20 (broke §34 from-scratch ceiling 9.38 → 5.36 → 4.71 in 16-day arc) |
| Compute used | 27 epochs / 2700 steps / ~40 min on RTX 4090 (lambda-vector) |
| Inference speed | 325.1 tok/s via `apr bench` (AC-SHIP2-009 DISCHARGED) |
| Sample at val_loss=4.71 | Repetitive token gibberish (`def fibonacci(n):` → ` č č č č ...`) — needs val_loss < 4 for any P1-B/C eval to be meaningful |

## 3. AC-SHIP2-* falsifier status

| ID | Falsifier | Status | Source |
|---|---|---|---|
| AC-SHIP2-001 | `apr run` exit-0 on checkpoint | PARTIAL (gibberish but exits cleanly) | §5.2 |
| AC-SHIP2-002 | `apr diff` against teacher | NOT-YET (needs distillation, not pretrain-only) | §5.2 |
| AC-SHIP2-003 | val_loss < 2.2 (strict) | **PARTIAL** — 4.71 > 2.2 strict, but ceiling broken | §82 |
| AC-SHIP2-004 | Training ≤ 21 days | **DISCHARGED** — §78 ran 8 min, §82 ran 40 min | §78 |
| AC-SHIP2-005 | ≥ 1 valid checkpoint | **DISCHARGED** — 5 valid in §78, 27 valid in §82 | §78 |
| AC-SHIP2-006 | `apr qa` runs end-to-end | **FUNCTIONAL** — infra passes; only golden_output fails (expected for pretrain-only) | §82 |
| AC-SHIP2-007 | `apr inspect --quality` ≥ 90 | NOT-YET | §5.2 |
| AC-SHIP2-008 | `apr lint` zero High severity | NOT-YET | §5.2 |
| AC-SHIP2-009 | `apr bench` ≥ 100 tok/s | **DISCHARGED** — 325.1 tok/s at §82 | §82 |
| AC-SHIP2-010 | `llama-cli` end-to-end interop | **UNBLOCKED** — P0-G + P0-H both merged (PR #1706 + #1709, 2026-05-16); needs re-export + re-test | §82 |

**Tally:** 3 DISCHARGED · 1 FUNCTIONAL · 1 UNBLOCKED · 2 PARTIAL · 3 NOT-YET = 79% ship %.

## 4. Open EV-ranked work queue

Priority order (Δship-% ÷ effort × P(success)):

### P0 — closes immediately if dispatched

| Item | Action | Δship | Effort | P | Notes |
|---|---|---|---|---|---|
| **P0-I** Verify P0-G+P0-H end-to-end on a fresh checkpoint | Build apr from post-merge main; re-export `epoch-020.apr` → GGUF → `llama-cli`; expect load + decode | +2 | 30 min | 95% | Validates §82 AC-SHIP2-010 fully. Trivial. |
| **P2-B2** Wire `apr pretrain --warn-on-wrap-around` env-default to enable in CI | (already merged in #1707; just verify default behaviour) | +0 (already counted) | 5 min | 99% | Sanity check. |

### P1 — short tasks that compose toward 90%

| Item | Action | Δship | Effort | P | Notes |
|---|---|---|---|---|---|
| **P1-A2** Re-run Chinchilla gate against epoch-020 to confirm warning fires (D=22M vs N=494M = 0.04× < 20× target) | Already merged in #1708; verify by running apr pretrain --num-steps 5000 with init and capturing stderr | +0 (already counted) | 10 min | 99% | Confirms gate works in practice. |
| **P1-B** HumanEval pass@1 on epoch-020 | `apr eval humaneval` against the current checkpoint | +3 if pass > 5% | 5-8h gx10 | **5%** (DEAD at val_loss 4.71 — model output is repetitive) | Defer until val_loss < 4. |
| **P1-C** Python validity (100 prompts, syntax-only) | Generate from 100 prompts, parse with `ast.parse`, count zero-error | +3 if pass > 30% | 2h | **10%** (still likely gibberish) | Defer until val_loss < 4. |

### P2 — multi-hour training compute to drive val_loss down

| Item | Action | Δship | Effort | P | Notes |
|---|---|---|---|---|---|
| **P2-A2** Longer P2-A run: 20K-50K steps, same Qwen-0.5B init + qwen-v2 corpus | `apr pretrain --num-steps 20000 --init <qwen-0.5b> --dataset qwen-v2` | +5 if val_loss < 3.5; +8 if val_loss < 2.5 | 3-8h GPU | 40% (corpus capacity is the binding constraint, not steps) | Highest-EV next dispatch. |
| **P2-C** Wider corpus: codeparrot-python permissive + the-stack-v2 Python | Author corpus merge contract; retokenize; rerun | +5 if val_loss < 3.0 | 6-12h CPU prep + 8-16h GPU train | 35% | Addresses §49 corpus diversity hypothesis directly. |
| **P2-D** True distillation from MODEL-1 (replace pretrain with apr distill loop) | Requires shipping `apr distill` per §35 (currently STUB) | +10 | 16-40h (multi-week scope) | 25% | Architectural change; defer until P2-A/C exhausted. |

### P3 — polish + publish

| Item | Action | Δship | Effort | P | Notes |
|---|---|---|---|---|---|
| **P3-A** `apr inspect --quality` on best checkpoint | Run the quality scorer (needs implementation if not present) | +1 | 1h | 80% | Discharges AC-SHIP2-007. |
| **P3-B** `apr lint` zero High severity | Currently passes presumably | +1 | 30 min | 90% | Discharges AC-SHIP2-008. |
| **P3-C** Publish to HuggingFace as `paiml/albor-370m-v1` | Once val_loss < 3 + smoke OK: `apr publish paiml/albor-370m-v1 --formats apr,safetensors,gguf` | +5 (final ship gate) | 1-2h | 95% | Triggers full ship close. |
| **P3-D** Post-publish QA + /dogfood verdict | Per `feedback_post_publish_qa_required.md` | +0 (gating) | 1h | 99% | Mandatory after every publish. |

## 5. Methodology lessons in flight (apply to MODEL-2 work)

These are the load-bearing ones from §82 cycle that govern remaining work:

- **#24** Mid-run progress logs aren't completion records — manifest.json is the contract (§77).
- **#25** Pretrained-init fine-tune dominates from-scratch on small compute (44.9% loss reduction same-budget, §78).
- **#26** Three-class root-cause taxonomy: data starvation / optimization defects / infrastructure masking (§79).
- **#27** Prioritize by Δship-% ÷ effort × P(success) (§80).
- **#28** Class 3 packaging defects come in waves (sharpened by #29).
- **#29** Class 3 waves are 4-5 deep, not 1-2: every downstream tool falsifies its own invariant (§82).

When dispatching P2-A2 / P2-C, predict val_loss + sample-quality bands BEFORE the run; verify after (lesson #18 predict-then-verify).

## 6. Bounded path to 100%

```
                                                        published
                                                        on HF +
                                                        /dogfood GO
                                                            ▲
                                                            │ +5
                                                            │
                       AC-SHIP2-007/008                     │
                       +2                                   │
                              ▲                             │
                              │                             │
   val_loss < 3       P1-B pass@1                          │
   +5                 > 10%                                │
        ▲             +3                                   │
        │             ▲                                    │
        │             │                                    │
   P2-A2            P1-C valid                            │
   succeeds         > 30%                                  │
        ▲                                                  │
        │ P0-I verifies AC-SHIP2-010                       │
   79% ─┴────► 81% ────► 86% ────► 91% ────► 93% ────► 98% ─┴────► 100%
   (today)          (P0-I)    (P2-A2)   (P1-B)   (P3-A/B)   (P3-C/D)
```

**Realistic 4-week shipping plan:**

- **Week 1**: Dispatch P0-I (verify) + P2-A2 (20K-step retrain). Lock val_loss < 3.5 or pivot to P2-C.
- **Week 2**: Run P1-B + P1-C against the new checkpoint. Eval gates `apr inspect --quality` + `apr lint`.
- **Week 3**: P3-A/B polish; author HF manifest; smoke-test on yoga (RTX 4060) for accessibility.
- **Week 4**: P3-C publish + P3-D /dogfood verdict. Ship %: 100.

If P2-A2/C don't crack val_loss < 3 with the available compute budget, pivot to P2-D (true distillation from MODEL-1) and accept multi-week delay.

## 7. Compute lanes for the queue

Per `feedback_compute_pre_authorized.md`:

- **lambda-vector RTX 4090** — pre-authorized for named dispatches. Use for P2-A2.
- **gx10 sm_121a** — sm_121a Blackwell, only via llama.cpp / cuBLAS forward path. Use for HumanEval evaluation (CPU forward at 5.8h wall per §67).
- **yoga RTX 4060** — fits 370M; smoke test only.
- **jetson** — still blocked (5 prerequisites, see `project_ship_two_001_jetson_blocked.md`).

All P2-* dispatches need an explicit pre-flight Chinchilla check (P1-A merged): D ≈ 20·N for compute-optimal.

## 8. How to update this roadmap

When an item lands:

1. Move it from "Open" to a "Closed" appendix at the bottom of this file.
2. Update the ship % in the index file + full MODEL-2 spec.
3. If the move uncovers a new defect or pivots strategy, author a new section in `ship-model-2-spec.md` with the next §N number (currently next is §83) and link from this roadmap.

When the bounded path needs to change (e.g. P2-A2 lifts val_loss < 2.5 sooner than expected): rewrite §6 here, don't try to amend the full spec.
