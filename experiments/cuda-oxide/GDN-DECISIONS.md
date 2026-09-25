# GDN kernels: oxide or ptx_exemption (OXIDE-001 O-4, aprender#3522)

KERNEL-001 migrates the GDN set first. This file records one decision per GDN kernel, and each decision
comes from receipts under `evidence/kernels/<k>/<host>.json`. A kernel with no receipt has no decision.
It is **RED**, not deferred: without a receipt nobody knows whether oxide wins, so the hand PTX stays and
is still unjustified.

This file sits outside `evidence/kernels/`, because the `kernel-receipt` extractor refuses any file there
that is neither a manifest nor a receipt (WrongCorpus).

The set is every `impl Kernel` under `crates/aprender-gpu/src/kernels/gdn/`, measured at 7af9e6637, except
`HelperProbe`, which is a test helper. `layernorm/` and `conv1d.rs` are outside the GDN set.

## Decided (4 of 15)

The thresholds are cos ≥ 0.9999, max|Δ| < 1e-3 against f64, and oxide/hand ≤ 1.2, taking the worst ratio
over the timed shapes (heads 16/32/48 for the per-head kernels, n = 2048/4096/6144 for the elementwise ones,
channels = 2048/4096/6144 with K = 4 for the conv).

**Timing method: CUDA-graph replay** (`time_graph_us`: 100 launches captured on a created stream, median of
5 replays). The first receipts timed eager back-to-back launches on the legacy null stream. At ~2 µs per
kernel that measures host submission: on yoga the l2-norm eager ratio flipped 0.92 ↔ 1.38 run to run with no
foreign process, and on lambda one run gave eager 0.557 / 1.216 / 1.051 across three head counts while the
graph ratio sat at 0.860 / 0.864 / 0.866. Eager is still recorded as `timing.eager_ratio_max`, never gated.
A receipt whose `timing` has no `"method":"cuda-graph-100"` is an eager one.

| kernel | shipped PTX | decision | lambda (sm_89) | yoga (sm_89) | gx10 (sm_121) |
|---|---|---|---|---|---|
| `GatedRmsNormKernel` | `kernels/gdn/gated_rmsnorm.rs` | **oxide** | 0.887 | 0.908 | 0.902 |
| `PerHeadL2NormKernel` | `kernels/gdn/l2_norm.rs` | **oxide** (`rsqrt`) | 0.839 | 0.860 | 0.834 |
| `SigmoidGateKernel` | `kernels/gdn/sigmoid_gate.rs` | **oxide** (`ex2`), within budget, not a win | 1.058 | 1.076 | 1.089 |
| `CausalConv1dSiluKernel` | `kernels/gdn/causal_conv1d.rs` | **oxide** (`ex2`), within budget, not a win | 1.080 | 1.091 | 1.139 |

**`GatedRmsNormKernel`**
- Receipts: `evidence/kernels/gdn_gated_rmsnorm/{noah-Lambda-Vector,yoga,gx10-a5b5}.json`, 2 entries per
  host (`exp`, `ex2`).
- Parity and timing pass on every row, and every row is on a clean tree.
- Oxide wins on all three hosts, so no `ptx_exemption` is justified.
- All three hosts are graph-timed. The eager ratios were 0.913 / 0.948 / 1.004: on gx10 eager read a tie
  while graph replay shows the same ~10% win the sm_89 hosts show.
- The shipped kernel is still the hand PTX. No crate on `main` depends on cuda-oxide yet, so replacing it is
  a separate migration row. This decision only settles which variant that row should ship.

**`PerHeadL2NormKernel`**
- Port: `experiments/cuda-oxide/l2-norm/`, two entries, out of place, because a safe kernel cannot alias its
  input. `l2_norm_sqrt` computes `1/sqrt`; `l2_norm_rsqrt` uses `rsqrt.approx`, the hand PTX's form.
- Receipts: `evidence/kernels/gdn_l2_norm/{noah-Lambda-Vector,yoga,gx10-a5b5}.json`, 2 entries per host.
- Parity on every host: cos 1.0, max|Δ| ≤ 2.98e-8, and every head's ‖h‖ within 2.5e-7 of 1.
- The table shows `rsqrt`. `sqrt` wins too: 0.866 / 0.889 / 0.862.
- The decision is `rsqrt` because it is the faster entry on all three hosts and the same instruction the
  shipped kernel uses, so the swap changes the authoring and not the math.
- gx10 was taken after infra#1111 made `apr-review-serve` yield the GPU lock to a blocking waiter. Of three
  runs, the first overlapped the serve process's exit (it listed a foreign GPU process); the committed
  receipt is the third run, with `foreign_gpu_procs` empty.

**`SigmoidGateKernel`**
- Port: `experiments/cuda-oxide/sigmoid-gate/`, two entries, out of place for the same reason as l2-norm.
  `sigmoid_gate_exp` uses `exp`; `sigmoid_gate_ex2` uses `exp2(-g·log2 e)`, the hand PTX's `ex2.approx` form.
- `n` is baked into the hand PTX, so there is one golden baseline per timed width (`gdn_sigmoid_gate_ptx_golden`).
- Receipts: `evidence/kernels/gdn_sigmoid_gate/{noah-Lambda-Vector,yoga,gx10-a5b5}.json`, 2 entries per host,
  all at eedc0b8d8 on a clean tree with no foreign GPU process.
- Parity on every host: cos 1.0, max|Δ| ≤ 2.4e-7. The parity widths include n = 1000, which is not a multiple
  of the block, so the tail threads are exercised.
- **This is the first GDN port that is slower than the hand PTX: 6–10% on every host and both variants.**
  It passes the ≤ 1.2 gate, so the decision is oxide, but no speedup is claimed. The time is flat across n
  (about 0.9–1.1 µs per launch at every width), so the gap is a fixed per-launch cost, not bandwidth.
- The PTX differs in these ways; which of them costs the ~70 ns has not been measured:
  - 6 params (ptr + len per slice) against 2.
  - three 64-bit bounds compares with `trap` arms against one immediate `n` compare.
  - a `stacksave`/`stackrestore` pair.
  - `%ntid.x` read instead of an immediate block size.
  - 17 registers (sm_89) or 19 (sm_121) against 15.
- The table shows `ex2`, the hand PTX's own form. `exp` gives 1.058 / 1.067 / 1.097.
- Runs: yoga and gx10 ran 3 times each, and the committed receipt is run 3, the same convention as l2-norm.
  - gx10 runs 1–2 agree with run 3 (≤ 1.090).
  - yoga run 1 agrees (≤ 1.067). **yoga run 2 had one NO-GO row: `exp` at n = 4096, ratio 1.236.** No foreign
    GPU process was listed, but its eager times on that run were erratic (13.4 µs and 23.9 µs against a
    typical 3–4 µs), which points to host interference. The other five rows of run 2 were ≤ 1.088. That
    single row is disclosed here rather than hidden. A re-run that goes over 1.2 again reopens the row.
  - lambda ran once (1.058 for both variants).

**`CausalConv1dSiluKernel`**
- Port: `experiments/cuda-oxide/causal-conv1d/`, two entries, out of place. The hand PTX shifts the state
  window in place. The port reads `state` and writes `state_out`, a second `DisjointSlice` with
  `LinearTiles<3>`. Each output slice is claimed with its own `thread::index_1d_u32`, because the index
  token is `!Copy`. Parity checks the SiLU output and the shifted window.
- `channels` and `K` are baked into the hand PTX, so there is one golden baseline per timed width
  (`gdn_conv1d_ptx_golden`, K = 4).
- Receipts: `evidence/kernels/gdn_causal_conv1d_silu/{noah-Lambda-Vector,yoga,gx10-a5b5}.json`, 2 entries per
  host. All are at 78a1769df on a clean tree with no foreign GPU process.
- Parity on every host: cos 1.0, max|Δ| ≤ 1.2e-7 (output and state).
- **Slower than the hand PTX: 8–14%, worst on gx10.** It passes the gate, so the decision is oxide with no
  speedup claimed. `exp` is within 0.004 of `ex2` on every host. The table shows `ex2`, the hand PTX's form,
  as for the sigmoid gate.
- **The first form failed on gx10, and the fix is recorded here.**
  - At 204bbb660 the port indexed every element (`state[c*3+k]`, `weight[c*4+k]`), so its PTX carried 8
    bounds checks with `trap` arms.
  - gx10 timed that form at 1.20–1.22 in all three runs, a timing NO-GO. Lambda measured 1.12 and yoga
    1.13 for the same form.
  - 78a1769df takes one checked chunk per operand (`get(..)` + `first_chunk::<N>()`, early return on
    `None`), claims `ThreadRunMut32::Full` runs and writes through `at_const`.
  - The rewrite has 0 traps. gx10 dropped to 1.12–1.14, lambda to 1.08 and yoga to 1.09. The kernel math
    did not change.
  - The checks cost more on sm_121 than on sm_89, and this port measures that directly. It is the same
    fixed per-launch overhead the sigmoid-gate notes list.
- The PTX still differs from the hand PTX: 10 params against 4, a `stacksave` pair, and 20 registers
  (sm_89) or 22 (sm_121) against 17.
- Runs: yoga and gx10 ran 3 times each, and the committed receipt is run 3. Every run on both hosts was GO
  (yoga ≤ 1.092, gx10 ≤ 1.144). Lambda ran once.

## Undecided: RED, no receipt (11 of 15)

Each row needs an O-1-style port: an oxide `#[kernel]` beside the hand PTX, an f64 CPU reference, a
`receipt.sh` run on lambda, yoga and gx10, and a manifest under `evidence/kernels/`, with
`EXPECTED_KERNELS` bumped. The row order is the issue's order: memory-bound first.

| kernel | shipped PTX |
|---|---|
| `CausalConv1dSiluSeqKernel` | `kernels/gdn/causal_conv1d_seq.rs` |
| `GdnGatesKernel` | `kernels/gdn/gdn_gates.rs` |
| `GdnGatesRowsKernel` | `kernels/gdn/rows.rs` |
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
