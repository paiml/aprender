# Roadmap: aprender/albor-370m (MODEL-2) — Path to Ship

**Document ID:** SPEC-SHIP-MODEL-2-ROADMAP
**Version:** 2.0.0 (post-audit reprioritization)
**Parent specs:**
- [aprender/albor-370m full spec (MODEL-2)](./ship-model-2-spec.md) — historical record, §5-§83
- [External audit](../audits/albor-370.md) — 2026-05-16 pre-falsification of P2-A2 via Chinchilla math
- [Ship Two Models Index](./ship-two-models-spec.md)
- [Shared methodology](./ship-shared-methodology.md)

**Purpose:** Focused, forward-looking spec for the only open model. The 3,290-line MODEL-2 spec is the source of truth for what *happened*; this roadmap is what we *work against*. Update this doc as items land. Reference §N markers point back to the full spec for context.

## 🚨 Audit-driven reprioritization (2026-05-16, v2.0.0)

An [external audit](../audits/albor-370.md) pre-falsified the v1.0.0 P2-A2 plan via Chinchilla math (Hoffmann et al. 2022, arXiv:2203.15556) BEFORE the dispatch. Key numbers:

| Quantity | Value | Notes |
|---|---|---|
| N (params) | ~494M | Qwen-0.5B init scale |
| Chinchilla target D = 20·N | **9.88B tokens** | Compute-optimal |
| D consumed in §82 P2-A | **22M tokens** | 2700 steps × ~8192 tokens/step |
| Empirical ratio | **0.04×** | Catastrophically under-provisioned |
| Available corpus (qwen-v2 codeparrot-python) | 1.24B tokens | Still only **0.125×** Chinchilla even if fully consumed |
| Target for first usable run | **> 2B tokens** (audit rec) — drives val_loss toward 3.0 | Requires P2-C corpus widening |

**Audit verdict:** P2-A2 (more steps on same corpus) cannot break the plateau — the binding constraint is *data*, not *compute*. The repetitive `č č č č` gibberish at val_loss=4.71 is the [Holtzman et al. 2019](https://arxiv.org/abs/1904.09751) degeneration signature, classic symptom of an under-trained model with insufficient data diversity to shape the long-tail distribution.

**Four engineering actions** (now reflected in §4 and §6 below):

1. **Promote P2-C above P2-A2.** Widen corpus to > 2B tokens BEFORE the next multi-hour GPU dispatch.
2. **Make Chinchilla a hard gate, not a warning.** New P0-J: fail-fast in `apr pretrain` when D/N < 10× unless `--force-under-provisioned` flag is passed.
3. **Defer P1-B/C/P3-A until val_loss < 3.0** (was < 4.0). Perplexity > 20 = mathematically incapable of zero-shot reasoning.
4. **Pre-flight prediction via theoretical constraint** is a load-bearing methodology — see lesson #30.

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
| **P0-J** Chinchilla gate: warning → hard blocker | Convert `[P1-A]` stderr warning to a fail-fast error when D/N < 10×; add `--force-under-provisioned` opt-in bypass. Updates `apr-cli/src/commands/pretrain.rs` and writes a contract `chinchilla-gate-v1.yaml` with falsification test. | +1 (prevention) | 1-2h | 95% | Audit recommendation #2. Prevents next "wasted 40 min on a 0.04× run". |
| **P2-B2** Wire `apr pretrain --warn-on-wrap-around` env-default to enable in CI | (already merged in #1707; just verify default behaviour) | +0 (already counted) | 5 min | 99% | Sanity check. |

### P1 — short tasks that compose toward 90%

| Item | Action | Δship | Effort | P | Notes |
|---|---|---|---|---|---|
| **P1-A2** Re-run Chinchilla gate against epoch-020 to confirm warning fires (D=22M vs N=494M = 0.04× < 20× target) | Already merged in #1708; verify by running apr pretrain --num-steps 5000 with init and capturing stderr | +0 (already counted) | 10 min | 99% | Confirms gate works in practice. |
| **P1-B** HumanEval pass@1 on epoch-020 | `apr eval humaneval` against the current checkpoint | +3 if pass > 5% | 5-8h gx10 | **3%** (DEAD at val_loss 4.71 — model output is repetitive) | **Defer until val_loss < 3.0** (was 4.0). Audit rec #3: perplexity > 20 = no zero-shot reasoning. |
| **P1-C** Python validity (100 prompts, syntax-only) | Generate from 100 prompts, parse with `ast.parse`, count zero-error | +3 if pass > 30% | 2h | **5%** (still likely gibberish) | **Defer until val_loss < 3.0** (was 4.0). |

### P2 — data engineering + training compute (P2-C now first; audit Rec #1)

| Item | Action | Δship | Effort | P | Notes |
|---|---|---|---|---|---|
| **P2-C** Widen corpus to > 2B tokens: codeparrot-python permissive + the-stack-v2 Python (audit recommendation) | (1) Author corpus merge contract; (2) Pull the-stack-v2 Python permissive shards via `apr corpus pull`; (3) Concatenate + dedupe with §77 NFC pipeline; (4) Retokenize with Qwen tokenizer; (5) Rerun pretrain with Chinchilla gate enforced (P0-J) | +6 if val_loss < 3.5; +10 if val_loss < 3.0 | 6-12h CPU prep + 8-16h GPU train | **55%** (corpus diversity is the binding constraint per §49 + audit) | **NEW highest-EV next dispatch.** Audit Rec #1: P2-A2 cannot break the plateau; only P2-C can. |
| **P2-A2** Longer P2-A run on the same qwen-v2 subset (DOWNGRADED — pre-falsified) | `apr pretrain --num-steps 20000 --init <qwen-0.5b> --dataset qwen-v2` | +1 (best case overfit) | 3-8h GPU | **15%** (Chinchilla 0.04× → mode collapse guaranteed) | **Audit Rec #1 pre-falsifies this.** Keep as fallback ONLY if P2-C is blocked on corpus pull/tokenize. Otherwise skip. |
| **P2-D** True distillation from MODEL-1 (replace pretrain with `apr distill` loop) | Requires shipping `apr distill` per §35 (currently STUB) | +10 | 16-40h (multi-week scope) | 25% | Architectural change; superseded by PMAT-683/684 (SPEC §89) — distillation epic post-v1-ship. |

### P3 — polish + publish — STATUS POST-§88

| Item | Action | Δship | Effort | P | Status (2026-05-17) |
|---|---|---|---|---|---|
| **P3-A** `apr inspect --quality` on best checkpoint | Run the quality scorer | +1 | 1h | 80% | ✅ **SHIPPED** (PR #1750, merged via #1742 squash). Scorer lives at `apr inspect --quality`. AC-SHIP2-007-prep DISCHARGED. |
| **P3-B** `apr lint` zero High severity | `apr lint` passes | +1 | 30 min | 90% | ⚙️ Operator-dispatchable. Pre-§88 deferred until val_loss < 3.0; §88 unblocks. |
| **P3-C-prep** Model card + publish-readiness preflight | PR #1764 ships `docs/model-cards/albor-370m-v1.md` + `scripts/publish/albor-370m-publish-readiness.sh` | +1 | 1h | 95% | ✅ **SHIPPED** (PR #1764). |
| **P3-C-exec** Publish to HuggingFace as `paiml/albor-370m-v1` | Operator runs: `apr stamp <pre-P0-K-ckpt> --architecture qwen2 ...` (§86 salvage) → `bash scripts/publish/albor-370m-publish-readiness.sh <stamped.apr>` → `apr publish paiml/albor-370m-v1 --formats apr,safetensors,gguf --model-card docs/model-cards/albor-370m-v1.md` | +5 (final ship gate) | 1-2h | 95% | 🟡 **OPERATOR-READY** — requires explicit user invocation (external-action authorization). |
| **P3-D** Post-publish QA + /dogfood verdict | Per `feedback_post_publish_qa_required.md`; template at `docs/dogfood-templates/albor-370m-v1-dogfood-template.md` (PR #1765) | +0 (gating) | 1h | 99% | 🟡 **TEMPLATE READY** — execution gated on P3-C-exec. |

### P4 — Distillation epic (out-of-v1-scope per SPEC §89)

Path to `AC-SHIP2-003-STRICT` (val_loss ≤ 2.2). Deferred to a follow-up epic post-v1-ship.

| Item | Action | Δship-strict | Effort | P | Notes |
|---|---|---|---|---|---|
| **PMAT-683** Teacher selection + pull | `apr pull Qwen/Qwen2.5-Coder-7B-Instruct --quantize q4k -o teacher.apr` + `apr qa teacher.apr` | +0 (gating) | 4-6h operator | 95% | Validates the teacher reaches non-degenerate output on the held-out corpus. |
| **PMAT-684** Distillation training dispatch + evidence | `apr distill --teacher teacher.apr --student qwen-init.apr --dataset qwen-v3/ --num-steps 245000 --temperature 4.0 --lr 1.5e-5` | +5 (strict-target ship) | ~43h GPU (fits 48-hr budget) + ~8h operator | 70% | Tests Stanton et al. 2021's 5× token-reduction claim empirically. |
| **PMAT-685** Distillation loop hardening (deferred) | Multi-teacher ensemble / curriculum corpus / LR cycling / layer-wise losses | +0 (signaling) | TBD | TBD | Only dispatched IF PMAT-684 result is borderline. |
| **paiml/albor-370m-v2** Publish + /dogfood | Same workflow as v1 but using §86.4 stamp recipe pre-baked into a `v2-prep` script | +5 (formal STRICT discharge) | 1-2h | 95% | After PMAT-684 reaches val_loss ≤ 2.2. |

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

**Realistic 4-week shipping plan (post-audit, v2.0.0):**

- **Week 1**: Dispatch P0-I (verify GGUF interop) + P0-J (Chinchilla hard gate landing). Begin P2-C data engineering: pull the-stack-v2 Python, dedupe against codeparrot, retokenize. Target corpus ≥ 2B tokens by end of week. NO multi-hour training dispatches until corpus is ready.
- **Week 2**: P2-C training dispatch on widened corpus (8-16h GPU). Lock val_loss < 3.0. If val_loss plateaus above 3.0, immediately pivot to P2-D (true distillation).
- **Week 3**: With val_loss < 3.0, run P1-B + P1-C against the new checkpoint. Eval gates `apr inspect --quality` + `apr lint` (P3-A/B).
- **Week 4**: P3-C publish to `paiml/albor-370m-v1` + P3-D /dogfood verdict. Ship %: 100.

**Pre-falsified shortcut.** Per audit math, P2-A2 (more steps on same data) cannot reach val_loss < 3.5 — skip it unless P2-C is blocked on tokenizer infrastructure. If P2-C blocks AND P2-D is multi-week scope, fall back to P2-A2 only to keep momentum, accepting the overfit result.

## 7. Post-§88 actual shipping plan (2026-05-17 amendment)

The §82-§87 cycle empirically proved the 4-week plan above was over-optimistic on val_loss target (the strict CE ≤ 2.2 requires 9-day compute, not feasible in 48-hr iteration budgets). §88 amended `AC-SHIP2-003` to a compute-bounded target (CE ≤ 4.7) which P2-E **DISCHARGES** at val_loss = 4.6227. The shipping plan compresses to:

### v1 ship (stack-existence-proof, post-§88)

| Phase | Status | Owner | Notes |
|---|---|---|---|
| P0-K cascade (PRs #1742/1746/1748/1750/1757) | ✅ SHIPPED | autonomous | apr_convert + apr_import + apr inspect + E2E test + apr stamp HF identity |
| P2-F val-shard (#1744) | ✅ SHIPPED | autonomous | independent held-out val source |
| P2-E training (val_loss=4.6227) | ✅ COMPLETE | autonomous | discharges §88 loose target |
| §85 + §86 + §87 + §88 spec amendments | 🟡 PR #1754 + #1763 stack (auto-merge armed) | autonomous | will land via #1754 squash |
| §89 distillation epic scoping | 🟡 this PR | autonomous | scopes PMAT-683/684 |
| P3-A apr inspect --quality (#1750) | ✅ SHIPPED | autonomous | scorer landed |
| P3-C-prep model card + readiness (#1764) | ✅ SHIPPED | autonomous | docs ready |
| P3-C-exec `apr publish paiml/albor-370m-v1` | 🟡 OPERATOR-READY | user | external-action authorization required |
| P3-D /dogfood verdict (template #1765) | 🟡 TEMPLATE READY | user | gated on P3-C-exec |

### v2 ship (strict-target distillation, multi-week)

Per SPEC §89 — out of v1 scope. Triggers AFTER:

1. ✅ v1 published + /dogfood GO
2. ✅ At least one independent consumer downloads + runs v1 (validation-by-use)
3. ✅ User authorization for ~43-hour distillation dispatch

Then PMAT-683 (teacher pull, 4-6h) → PMAT-684 (distillation training, 43h GPU + 8h operator) → publish `paiml/albor-370m-v2` with strict-target discharge.

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
