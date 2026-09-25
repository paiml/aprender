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

**Timing method: CUDA-graph replay** (`time_graph_us`: 100 launches captured on a created stream, median of
5 replays). The first receipts timed eager back-to-back launches on the legacy null stream. At ~2 µs per
kernel that measures host submission: on yoga the l2-norm eager ratio flipped 0.92 ↔ 1.38 run to run with no
foreign process, and on lambda one run gave eager 0.557 / 1.216 / 1.051 across three head counts while the
graph ratio sat at 0.860 / 0.864 / 0.866. Eager is still recorded as `timing.eager_ratio_max`, never gated.
A receipt whose `timing` has no `"method":"cuda-graph-100"` is an eager one.

| kernel | shipped PTX | decision | lambda (sm_89) | yoga (sm_89) | gx10 (sm_121) |
|---|---|---|---|---|---|
| `GatedRmsNormKernel` | `kernels/gdn/gated_rmsnorm.rs` | **oxide** | 0.887 | 0.908 | 1.000 (eager) |

- Receipts: `evidence/kernels/gdn_gated_rmsnorm/{noah-Lambda-Vector,yoga,gx10-a5b5}.json`, 2 entries per
  host (`exp`, `ex2`).
- Parity and timing pass on every row, and every row is on a clean tree.
- Oxide wins or ties on all three hosts, so no `ptx_exemption` is justified.
- lambda and yoga are graph-timed (eager was 0.913 / 0.948). gx10 is still the eager receipt: its re-take is
  blocked, see the gx10 note below.
- The shipped kernel is still the hand PTX. No crate on `main` depends on cuda-oxide yet, so replacing it is
  a separate migration row. This decision only settles which variant that row should ship.

## Measured, RED until gx10 (1 of 15)

| kernel | shipped PTX | lambda (sm_89) | yoga (sm_89) | gx10 (sm_121) |
|---|---|---|---|---|
| `PerHeadL2NormKernel` | `kernels/gdn/l2_norm.rs` | sqrt 0.866 · rsqrt 0.839 | sqrt 0.889 · rsqrt 0.860 | — |

- Port: `experiments/cuda-oxide/l2-norm/`, two entries (`l2_norm_sqrt` = `1/sqrt`, `l2_norm_rsqrt` =
  `rsqrt.approx`, the hand PTX's form), out of place: a safe kernel cannot alias its input.
- Parity on both hosts: cos 1.0, max|Δ| ≤ 2.98e-8, every head's ‖h‖ within 2.5e-7 of 1.
- The receipts and manifest (`evidence/kernels/gdn_l2_norm/`, `EXPECTED_KERNELS` = 2) land together once gx10
  has a receipt, because the committed-tree gate fails a required host with no receipt.
- **gx10 note (2026-09-25):** `apr-review-serve.service` (infra#1088) runs `apr serve` inside
  `flock /tmp/apr-gpu.lock`, so gx10's GPU lock is held for the service's lifetime and a lock-respecting
  receipt never starts. Measuring beside it would record `foreign_gpu_procs` and fail kernel-timing anyway.

## Undecided: RED, no receipt (13 of 15)

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
