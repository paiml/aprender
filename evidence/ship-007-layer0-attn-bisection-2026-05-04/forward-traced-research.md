# SHIP-007 layer-0 attention bisection — `forward_traced_with_plan` research

**Date:** 2026-05-04
**Scope:** Pre-implementation research for FALSIFY-ATTN-SUB-002 (`trace-attn-sub-stages-v1.yaml` v1.1.0).
**Author:** Loop iteration during PR cascade (#1448/#1449/#1450/#1451) CI wait.

## What's in `forward_traced_with_plan` today

Source: `crates/aprender-serve/src/apr_transformer/inference.rs`.

Current `emit(SaveTensorStage::*)` calls in the attention-block portion:

| Line | Stage |
|------|-------|
| 70   | `Embedding` |
| 87   | `AttnNorm` |
| 98   | `QkvMatmul` |
| 101  | `QkvBias` (conditional on bias) |
| 174  | `Attention` (post softmax·V, pre O-proj) |
| 182  | `AttnOut` (post O-projection) |
| 192  | `PostAttnResidual` |

## Pre-existing gap (parent contract drift)

The parent contract `apr-cli-trace-save-tensor-v1.yaml` v1.4.0 (FUNCTIONAL) claims
all 18 stages are wired. **Empirically false**: `QPostRope` and `KPostRope` are
in the `SaveTensorStage` enum (lines 47-50) but have **no `emit()` call** in
`forward_traced_with_plan`. The RoPE-rotated tensors `q_all` and `k_all` are
computed at lines 130-131 but never captured.

This means:

- The parent contract has a **silent FUNCTIONAL drift** for QPostRope + KPostRope.
- A user passing `--save-tensor q_post_rope` would get a clean exit without
  a file written — silent failure.

## What FALSIFY-ATTN-SUB-002 should wire

Per `trace-attn-sub-stages-v1.yaml` v1.1.0 ordering proof_obligation:

```text
QkvBias → QPostRope → KPostRope → AttnScores → AttnSoftmax → Attention → AttnOut
```

Two of the four missing wires are NEW (introduced by #1451): `AttnScores`,
`AttnSoftmax`. Two are PRE-EXISTING gaps the cascade must close as a side
effect: `QPostRope`, `KPostRope`.

## Proposed wire plan

| New capture | Insertion point | Tensor | Shape (BOS, seq=1) |
|---|---|---|---|
| `QPostRope`   | After line 133 (post inner-loop Q/K/V copy) | `q_all` | `[seq × hidden_dim]` |
| `KPostRope`   | After line 133 | `k_all` | `[seq × kv_dim]` |
| `AttnScores`  | Inside head loop after scale (line 152), accumulated across heads | flattened `[num_heads × seq × seq]` | `[num_heads]` for BOS |
| `AttnSoftmax` | Inside head loop after softmax (line 160), accumulated across heads | same shape as scores | `[num_heads]` for BOS |

`AttnScores` and `AttnSoftmax` require a small refactor: the inner loop
allocates `scores` and `probs` per (head, position). To capture as full tensors
we need to:

1. Pre-allocate `scores_all` and `softmax_all` Vec<f32> of size
   `num_heads × seq × seq` (zero-initialized).
2. After computing `scores` (line 152), copy into
   `scores_all[head * seq * seq + i * seq + 0..=i]`.
3. After softmax (line 160), copy `probs` into the same offset of `softmax_all`.
4. After the heads loop completes, emit both tensors.

## Five Whys (why this scope, not a smaller one)

1. **Why wire 4 stages, not 2?**
   QPostRope + KPostRope are pre-existing gaps. A 2-stage PR ships a release
   where 2 advertised stages still silently no-op. Toyota Way mandates
   fixing the upstream defect when the change is in the same file.

2. **Why discover this only now, not when authoring v1.0.0?**
   Per `feedback_no_guessing.md`: should have checked the live source. The
   parent contract description was the source of truth I trusted — its
   FUNCTIONAL claim was the drift.

3. **Why not amend the parent contract first?**
   The parent contract `apr-cli-trace-save-tensor-v1.yaml` v1.4.0 is at
   FUNCTIONAL — bumping it conflicts the spec amendment cadence with our
   in-flight #1450. Better to record this drift in evidence + amend in the
   next cycle once the cascade lands.

4. **Why an evidence file instead of a 5th stacked PR right now?**
   Four PRs (#1448-#1451) are already in flight with auto-merge armed.
   Adding a 5th stacked PR while CI churns slows down both review and
   merge throughput. The implementation work is captured here so the next
   loop iteration can spawn it as an independent PR off main once #1451
   merges.

5. **Why not also wire `AttnVOut` (post softmax·V, pre O-proj)?**
   That's already wired as `Attention` (line 174). Re-named per the
   contract's existing semantic; no new wiring needed.

## Next-iteration deliverables

Once #1451 lands:

1. Open `feat(aprender-serve): wire 4 attention sub-stages in forward_traced_with_plan`
   off main (NOT stacked).
2. Add `emit(QPostRope, ...)` + `emit(KPostRope, ...)` after line 133.
3. Refactor scores/softmax accumulator: pre-allocate
   `scores_all` + `softmax_all` Vec<f32>, populate per (head, i, j),
   emit after head loop.
4. Add 4 unit tests using a tiny synthetic transformer (vocab=2, hidden=8,
   2 heads, 1 layer) — verify emit calls fire for each new stage.
5. Add 1 backward-compat test confirming `attn_output` byte-identity vs
   pre-impl run on canonical 7B teacher BOS forward.
6. Promote FALSIFY-ATTN-SUB-002 PARTIAL_ALGORITHM_LEVEL → FUNCTIONAL once
   the wires are live.

## Cross-references

- Parent contract: `contracts/apr-cli-trace-save-tensor-v1.yaml` v1.4.0 FUNCTIONAL
- Sibling contract: `contracts/trace-attn-sub-stages-v1.yaml` v1.1.0 PROPOSED (#1450)
- Enum extension: PR #1451
- Spec amendment: §47 (next cycle, will record §46 → §47 the SHIP-007 cascade)
- Memory: `2026-05-03 SHIP-007 finding`
