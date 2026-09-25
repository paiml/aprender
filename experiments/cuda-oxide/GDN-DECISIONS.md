# GDN kernels: oxide or ptx_exemption (OXIDE-001 O-4, aprender#3522)

KERNEL-001 migrates the GDN set first. This file records one decision per GDN kernel, and each decision
comes from receipts under `evidence/kernels/<k>/<host>.json`. A kernel with no receipt has no decision.
It is **RED**, not deferred: without a receipt nobody knows whether oxide wins, so the hand PTX stays and
is still unjustified.

This file sits outside `evidence/kernels/`, because the `kernel-receipt` extractor refuses any file there
that is neither a manifest nor a receipt (WrongCorpus).

The set is every `impl Kernel` under `crates/aprender-gpu/src/kernels/gdn/`, measured at 7af9e6637, except
`HelperProbe`, which is a test helper. `layernorm/` and `conv1d.rs` are outside the GDN set.

## Decided (13 of 15)

The thresholds are cos ≥ 0.9999, max|Δ| < 1e-3 against f64, and oxide/hand ≤ 1.2, taking the worst ratio
over the timed shapes (heads 16/32/48 for the per-head kernels, n = 2048/4096/6144 for the elementwise ones,
channels = 2048/4096/6144 with K = 4 for the convs, and t_count = 64 for the sequence conv).

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
| `CausalConv1dSiluSeqKernel` | `kernels/gdn/causal_conv1d_seq.rs` | **oxide** (`ex2`), at parity | 1.003 | 1.002 | 0.984 |
| `GdnGatesKernel` | `kernels/gdn/gdn_gates.rs` | **oxide** (`ex2` + `lg2_approx_f32`) | 0.911 | 0.918 | 0.901 |
| `GdnGatesRowsKernel` | `kernels/gdn/rows.rs` | **oxide** (`ex2`) | 0.954 | 0.961 | 0.928 |
| `PerHeadL2NormRowsKernel` | `kernels/gdn/rows.rs` | **oxide** (`rsqrt`) | 0.882 | 1.075 | 1.100 |
| `PartialNeoxRopeKernel` | `kernels/gdn/partial_rope.rs` | **oxide** (`approx`) | 0.879 | 0.854 | 0.884 |
| `PartialNeoxRopeRowsKernel` | `kernels/gdn/rows.rs` | **oxide** (`approx`), within budget, not a win | 1.033 | 1.037 | 1.061 |
| `SplitInterleavedKernel` | `kernels/gdn/split_interleave.rs` | **oxide**, within budget, not a win | 1.061 | 1.072 | 1.027 |
| `DeltaRuleRecurrenceKernel` | `kernels/gdn/delta_rule.rs` | **oxide** (`ex2`), at parity | 1.021 | 1.079 | 1.035 |
| `DeltaRuleChunkScanKernel` | `kernels/gdn/delta_rule_scan.rs` | **ptx_exemption**: oxide NO-GO on sm_89 | 1.814 | 1.986 | 1.158 |

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

**`CausalConv1dSiluSeqKernel`**
- Port: `experiments/cuda-oxide/causal-conv1d-seq/`, two entries, out of place. One thread per channel walks
  `t_count` tokens down a strided input column. It is the first GDN port with a strided-column write, and
  the safe form needs a 2-D launch contract (`domain = 2`, `block = (256,1,1)`). The input column is read
  through `MatrixView32::new(input, stride).col(c, rows)`. Output goes through
  `DisjointSlice<f32, RuntimeRowMajorTiles<T_TILE, 1>>`, and the host binds the row width with
  `RowWidth::new`. The shifted state goes to `state_out` with `RuntimeRowMajorTiles<1, 3>`.
- **Port constraint:** the tile height is a const generic, so the port fixes `T_TILE = 64` and clamps
  `t_count` to it. A shipping port needs one monomorphisation per chunk size, or a clipped 2-D run type,
  which this cuda-oxide rev does not have. This decision covers chunks of ≤ 64 tokens only.
- `channels`, `K` and the strides are baked into the hand PTX, so there is one golden baseline per timed
  width (`gdn_conv1d_seq_ptx_golden`, stride = channels).
- Parity cases include stride ≠ channels (1000/1024), a partial tile (t = 23) and a single token. Parity on
  every host: cos 1.0, max|Δ| ≤ 2.4e-7 (output and state).
- **Findings from writing the port:**
  - The first form read `w[i]` inside the token loop. LLVM cannot prove that the weight slice and the
    output do not alias, so it reloaded the weights from global memory on every step. Hoisting them into
    locals (`k0..k3`) fixed that.
  - LLVM contracts `sum += a*b` into `fma.rn.f32`. The hand PTX uses mul + add. Parity is unaffected.
    The per-token conv's header comment claimed otherwise and has been corrected: its PTX has 13 `fma.rn.f32`.
- Registers: oxide 30 (lambda), 29/32 (yoga, exp/ex2) and 28/31 (gx10), against 26 for the hand PTX on
  sm_89 and 29 on sm_121.
- **Timing harness fix: the first harness was biased against oxide on yoga.**
  - yoga's per-launch time steps between two levels during a run, about 15.2–15.5 µs and 11.4–11.7 µs,
    for both kernels. It looks like a clock-state change.
  - The first harness always timed oxide first and then the hand PTX. When the step fell inside a row,
    oxide took the slow reading and the hand PTX the fast one. At 888c736c3 that happened on 2 of 6 yoga
    runs: `ex2` at c = 2048 read 15.483/11.510 = 1.345, and `ex2` at c = 4096 read 14.920/11.590 = 1.287.
    The eager ratios for those same rows, timed after the step, were 0.993.
  - f5345368d times each row in 3 rounds, alternating which kernel goes first, and keeps the round with
    the median ratio. A step can land in only one round, and the median drops that round.
  - Under the fixed harness, all 3 yoga runs are GO with a worst ratio of 1.004. In several rows both
    kernels read the slow level, and the ratio was still 0.98.
  - The earlier harnesses (sigmoid gate, per-token conv) have the same fixed order. That may explain the
    sigmoid gate's yoga run-2 NO-GO row, but it has not been re-measured.
- **Oxide is at parity with the hand PTX:** within ±0.5% for `ex2` and about 1.5% faster for `exp` on
  every host. The table shows `ex2`, the hand PTX's form. `exp` gives 0.993 / 0.984 / 0.984.
- **Phantom GPU contexts:** from 2026-09-25, lambda's `nvidia-smi --query-compute-apps` has listed
  pid 3650689 (936 MiB, name `[Not Found]`). That pid is not in `/proc`, even as root, so it is a context
  the driver kept after its process died. Nothing is left to launch work on it. The `kernel-timing` shape
  fails any receipt with a non-empty `foreign_gpu_procs`, so that context blocked every lambda receipt
  until a GPU reset, and a reset is not ours to do on the operator's desktop.
  - From 11af9a79c, `receipt.sh` checks each row's pid against `/proc`.
  - A row whose pid is missing is written to `phantom_gpu_ctx` instead of `foreign_gpu_procs`.
  - A live process, including one in a container (host `/proc` lists those), stays foreign.
  - A pid that does not parse stays foreign, so the check fails closed.
  - Only this experiment's `receipt.sh` has the split so far.
- Receipts: `evidence/kernels/gdn_causal_conv1d_silu_seq/{noah-Lambda-Vector,yoga,gx10-a5b5}.json`, all at
  11af9a79c on a clean tree.
  - yoga and gx10 ran 3 times each at both f5345368d and 11af9a79c, 12 runs in all. Every run is GO, and
    the worst ratio is 1.004.
  - The committed receipt is run 3 at 11af9a79c, with no foreign GPU process. gx10's run 1 listed
    `apr.cur` both times.
  - Lambda ran with no live foreign process. Its only entry is the phantom context above.

**`GdnGatesKernel`**

- One thread per head: `dt = softplus(alpha + dt_bias) * a` (threshold 20) and `beta = sigmoid(beta_raw)`.
  Six pointer params in the hand PTX; the oxide port takes twelve (slice ptr + len each), which costs nothing
  measurable at this size.
- Variant B needs `cuda_device::float::lg2_approx_f32` explicitly. `f32::log2` lowers to libdevice's exact
  `log2f`, an inlined ~12-FMA polynomial, not `lg2.approx.f32`. With the intrinsic, B's PTX has the hand
  kernel's 2 × `ex2.approx.f32` + 1 × `lg2.approx.f32`. The sigmoid's `1 / y` lowers to `rcp.rn.f32`, where
  the hand kernel has `div.rn.f32`. Both round correctly, so the values match.
- Variant A (`exp`/`ln`) is GO too (worst 0.967, yoga) and uses 20 registers on gx10 against the hand's 14.
  B uses 15 on every host.
- Parity is exact to within 7.6e-6 at heads 1/16/48/1000, including the softplus branch on both sides of 20.
- Receipts are at 125ede286 on a clean tree, with the alternating-order 3-round median timing and the
  `phantom_gpu_ctx` split from row 5. yoga and gx10 ran 3 times each, and all 18 rows per host are GO. The
  committed receipt is the fixed run 3, with no foreign GPU process (gx10's run 1 listed `apr.cur`). Lambda
  was one run, with only the phantom context (pid 3650689).

**Rows kernels and RoPE (`GdnGatesRowsKernel`, `PerHeadL2NormRowsKernel`, `PartialNeoxRopeKernel`,
`PartialNeoxRopeRowsKernel`)**
- Ports: `experiments/cuda-oxide/{gdn-gates-rows,l2-norm-rows,partial-rope,partial-rope-rows}/`, two entries
  each (`exp`/`ex2`, `sqrt`/`rsqrt`, `libm`/`approx`). Each rows port reads the `[T][…]` projection by the
  runtime row stride that the host binds (`RuntimeRowMajorTiles` + `RowWidth`), as the hand kernel's baked
  stride does.
- Receipts: `evidence/kernels/{gdn_gates_rows,gdn_l2_norm_rows,gdn_partial_rope,gdn_partial_rope_rows}/`,
  3 hosts, 2 entries each, every row parity + timing PASS on a clean tree. The yoga and gx10 receipts are
  run 3 of 3. Lambda is one run at 625fe7928 with only the phantom context (pid 3650689).
- Each decision picks the entry that is faster on all three hosts, which is also the hand PTX's instruction.
- `PerHeadL2NormRowsKernel` wins on lambda (0.882) and loses 7–10% on yoga and gx10. `PartialNeoxRopeRowsKernel`
  loses 3–6% everywhere. Both pass the ≤ 1.2 gate, so they are oxide, but no speedup is claimed.

**`SplitInterleavedKernel`**
- Port: `experiments/cuda-oxide/split-interleave/`, one entry (`copy`), with no math to vary.
- Receipts: `evidence/kernels/gdn_split_interleave/`, 3 hosts, parity bit-exact, timing 1.03–1.07.
- yoga's first three runs (79ec2c259) all listed a foreign GPU process: a CI `realizar` test binary (pid
  1643163, 120 MiB) that stayed resident. They measured 1.078, but `kernel-timing` refuses a row with
  `foreignGpuProcs`. yoga was rerun 3 times at 625fe7928 once its GPU was empty. All 3 runs were GO at
  1.057–1.072, and the committed receipt is run 3 (1.072). The earlier contaminated number agreed, so the
  rerun did not change the decision.

**`DeltaRuleRecurrenceKernel`**
- Port: `experiments/cuda-oxide/delta-rule/`, one thread per state row. The row is a `[f32; 128]` held in
  registers by `#[unroll]` loops (0 `.local`), and the state is a `RuntimeRowMajorTiles<1, DK>` tile, which
  gives the hand kernel's exclusive-row ownership as a type. Parity uses 3 chained steps on one state and
  checks both the output and the state.
- Receipts: `evidence/kernels/gdn_delta_rule/`, 3 hosts, all PASS. `exp` and `ex2` agree within 0.4% on every
  host, so the decision is `ex2`, the hand PTX's instruction.

**`DeltaRuleChunkScanKernel`: NO-GO, the hand PTX stays**
- Port: `experiments/cuda-oxide/delta-rule-scan/`. It is the per-token kernel above, looped over up to
  CHUNK = 64 tokens per launch (the output is a `RuntimeRowMajorTiles<CHUNK, 1>` column).
- It has **no shared staging**. The hand kernel copies each token's k and q heads into shared memory behind
  `bar.sync`. cuda-oxide's `SharedArray` is a `static mut`, and every access to it is `unsafe`, so the safe
  port reads k and q from global memory as broadcasts instead.
- Parity is exact on every host (cos 1.0, max|Δ| ≤ 6e-8) over a chained 1/17/64-token schedule, checking both
  the output and the state.
- Timing, graph-replayed over one 64-token launch, worst over nv 16/32/48, `ex2` / `exp`:
  - lambda: 1.814 / 1.844, **NO-GO**
  - yoga: 1.986 / 1.844, **NO-GO**
  - gx10: 1.158 / 1.159, GO
- Cause: `ptxas -v` shows the oxide kernel at 255 registers with **212 B/thread of spills on sm_89 and 224 B on
  sm_121**. The hand kernel spills 60 B on sm_89 and 0 B on sm_121, at 158 registers. The single-token oxide
  kernel does not spill. With the global k/q loads hoisted across the 128 live state registers, ptxas runs
  out of registers. gx10's unified memory hides the spill traffic, and neither sm_89 card does.
- yoga was NO-GO on all 3 runs. Run 1 had no foreign GPU process and measured 1.76–1.93. The committed run 3
  overlapped a CI `realizar` test binary. All three runs agree, so run 3 is kept as the receipt.
- Safe oxide has no lever for this. Shared staging needs `unsafe`, and blocking the i-loop turns the row
  index dynamic, which moves the state to `.local`. So this row is a `ptx_exemption` until cuda-oxide has a
  safe shared-memory primitive. It is not deferred.
- The manifest says `authoring: ptx`, with `ptx_exemption.receipt` naming the losing lambda receipt, which
  has no foreign process. This follows `contracts/kernel-receipt-v1.yaml`: an oxide kernel that loses on
  timing keeps the hand PTX under an exemption that names the losing receipt.
- As an `OxideKernel` it would also fail `kernel-safety` `missingEntry`, and not because of `unsafe`. The two
  entries come from one `macro_rules!` that emits the whole `#[cuda_module]` (a `#[device]` helper is a
  `.func` call, which turned the ex2 flag into a runtime branch). The extractor's `syn` walk does not
  expand macros. A future oxide port of this row must spell its entries out.

## Undecided: RED, no receipt (2 of 15)

Each row needs an O-1-style port: an oxide `#[kernel]` beside the hand PTX, an f64 CPU reference, a
`receipt.sh` run on lambda, yoga and gx10, and a manifest under `evidence/kernels/`, with
`EXPECTED_KERNELS` bumped. The row order is the issue's order: memory-bound first.

| kernel | shipped PTX |
|---|---|
| `DecodeAttention256Kernel` | `kernels/gdn/decode_attention.rs` |
| `PrefillFlashAttention256Kernel` | `kernels/gdn/prefill_flash_attention.rs` |

## #3062

#3064 armed EPIC #3062's O2 (`register_budget`). Its GDN migration question now lives here and in #3522
(0.70.1). The rest of #3062 (cuda-core, cutile) is not a GDN kernel decision and stays at its own milestone.
