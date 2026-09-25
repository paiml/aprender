# GDN kernels: oxide or ptx_exemption (OXIDE-001 O-4, aprender#3522)

KERNEL-001 migrates the GDN set first. This file records one decision per GDN kernel, and each decision
comes from receipts under `evidence/kernels/<k>/<host>.json`. A kernel with no receipt has no decision.
It is **RED**, not deferred: without a receipt nobody knows whether oxide wins, so the hand PTX stays and
is still unjustified.

This file sits outside `evidence/kernels/`, because the `kernel-receipt` extractor refuses any file there
that is neither a manifest nor a receipt (WrongCorpus).

The set is every `impl Kernel` under `crates/aprender-gpu/src/kernels/gdn/`, measured at 7af9e6637, except
`HelperProbe`, which is a test helper. `layernorm/` and `conv1d.rs` are outside the GDN set.

## Decided (1 of 15)

The thresholds are cos ≥ 0.9999, max|Δ| < 1e-3 against f64, and oxide/hand ≤ 1.2, taking the worst ratio
over heads 16/32/48.

| kernel | shipped PTX | decision | lambda (sm_89) | yoga (sm_89) | gx10 (sm_121) |
|---|---|---|---|---|---|
| `GatedRmsNormKernel` | `kernels/gdn/gated_rmsnorm.rs` | **oxide** | 0.913 | 0.948 | 1.000 |

- Receipts: `evidence/kernels/gdn_gated_rmsnorm/{noah-Lambda-Vector,yoga,gx10-a5b5}.json`, 2 entries per
  host (`exp`, `ex2`).
- Parity and timing pass on every row, and every row is on a clean tree.
- Oxide wins or ties on all three hosts, so no `ptx_exemption` is justified.
- The shipped kernel is still the hand PTX. No crate on `main` depends on cuda-oxide yet, so replacing it is
  a separate migration row. This decision only settles which variant that row should ship.

## Undecided: RED, no receipt (14 of 15)

Each row needs an O-1-style port: an oxide `#[kernel]` beside the hand PTX, an f64 CPU reference, a
`receipt.sh` run on lambda, yoga and gx10, and a manifest under `evidence/kernels/`, with
`EXPECTED_KERNELS` bumped. The row order is the issue's order: memory-bound first.

| kernel | shipped PTX |
|---|---|
| `CausalConv1dSiluKernel` | `kernels/gdn/causal_conv1d.rs` |
| `CausalConv1dSiluSeqKernel` | `kernels/gdn/causal_conv1d_seq.rs` |
| `GdnGatesKernel` | `kernels/gdn/gdn_gates.rs` |
| `GdnGatesRowsKernel` | `kernels/gdn/rows.rs` |
| `SigmoidGateKernel` | `kernels/gdn/sigmoid_gate.rs` |
| `PerHeadL2NormKernel` | `kernels/gdn/l2_norm.rs` |
| `PerHeadL2NormRowsKernel` | `kernels/gdn/rows.rs` |
| `PartialNeoxRopeKernel` | `kernels/gdn/partial_rope.rs` |
| `PartialNeoxRopeRowsKernel` | `kernels/gdn/rows.rs` |
| `SplitInterleavedKernel` | `kernels/gdn/split_interleave.rs` |
| `DeltaRuleRecurrenceKernel` | `kernels/gdn/delta_rule.rs` |
| `DeltaRuleChunkScanKernel` | `kernels/gdn/delta_rule_scan.rs` |
| `DecodeAttention256Kernel` | `kernels/gdn/decode_attention.rs` |
| `PrefillFlashAttention256Kernel` | `kernels/gdn/prefill_flash_attention.rs` |

## #3062

#3064 armed EPIC #3062's O2 (`register_budget`). Its GDN migration question now lives here and in #3522
(0.70.1). The rest of #3062 (cuda-core, cutile) is not a GDN kernel decision and stays at its own milestone.
