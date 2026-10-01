---
slug: gemm-shared-b-parallel
status: diagnosed
trigger: "debug the GEMM shared-B failure"
goal: find_root_cause_only
created: 2026-09-22
updated: 2026-09-22
---

# Debug Session: gemm-shared-b-parallel

## Symptoms

**Expected behavior**
`gemm_blis_parallel_shared_b(n, n, n, a, b, c)` agrees with `gemm_reference` to within
`1e-1` max elementwise difference. This is a falsification gate, `FALSIFY-SHARED-B-001`.

**Actual behavior**
```
crates/aprender-compute/src/blis/tests/validate_and_parallel.rs:236
FALSIFY-SHARED-B-001: max diff 39.200016 >= 1e-1 at 256
```
Missing the tolerance by ~392x. Failed on all 3 nextest retries.

**Scale context:** `a[i] = (i % 11) * 0.1`, `b[i] = (i % 7) * 0.1`, so elements are in
`[0, 1.0]` and with `k = 256` a correct `c` element is at most ~256. A max diff of 39.2 is
therefore ~15% of full scale — far too large for an FP-accumulation-order artifact. The
shape suggests a whole block/panel being skipped, duplicated, or written by the wrong
thread, not rounding.

**Where**
`crates/aprender-compute/src/blis/parallel.rs::gemm_blis_parallel_shared_b`
(called from `validate_and_parallel.rs:228`). NOT `gemm_blis_parallel`, which is a
different function and whose tests pass.

**Timeline**
Unknown — this gate has apparently never run in CI (see below), so "when it broke" is not
established. `crates/aprender-compute/src/blis/` has 431 added lines on the current branch
vs `origin/main`, including `microkernels/neon.rs` and `parallel.rs`.

**Reproduction**
```bash
# NOT reproducible under CI's own compute step — the test does not compile in there.
cargo nextest run --profile ci -p aprender-core -p aprender-compute -p aprender-forecast \
    --lib -E 'test(test_gemm_parallel_shared_b_256)'
```

## Prior Evidence (measured by the orchestrator 2026-09-22 — do not re-derive)

### The gate is invisible to CI, which is why it can be red

`#[cfg(feature = "parallel")]` guards it (`validate_and_parallel.rs:219`). `parallel` is
NOT in `aprender-compute`'s default feature set, so the test only compiles when another
crate in the build graph unifies `parallel` into compute.

| Scope | `validate_and_parallel` tests | `test_gemm_parallel_shared_b_256` |
|---|---|---|
| `-p aprender-compute --lib` — **exactly what CI runs** | 14 | **ABSENT** |
| `-p aprender-core -p aprender-compute -p aprender-forecast --lib` | 17 | **present** |

Three tests exist only in the wider scope: `test_gemm_parallel_shared_b_256`,
`test_gemm_parallel_shared_b_non_aligned`, `nn_thin_gemm_prefers_serial`.
`.github/workflows/ci.yml` runs `cargo test -p aprender-compute --lib` as its own step AND
excludes `aprender-compute` from the `--workspace` job, so neither job can see these.

### The discriminating sibling — this is NOT a blanket shared-B failure

`FALSIFY-SHARED-B-002` (`test_gemm_parallel_shared_b_non_aligned`, m=100 n=96 k=128) was
run in the SAME invocation and **PASSED**. Same function, same value pattern, smaller and
non-aligned. So the defect is size- or alignment-dependent, not a wholesale wrong result.

Working hypothesis space, in rough priority order:
1. A parallel work-split / block-range bug that only engages above a size or thread-count
   threshold (256*256*256 = 16.7M >= the 1M threshold that `gemm_blis_parallel` mentions at
   `validate_and_parallel.rs:203`; the 100x96x128 case is 1.2M and may take a different path).
2. A data race or overlapping write in the shared-B path — would show a VARYING `max_diff`
   across runs.
3. A NEON microkernel edge case that only triggers on 256-aligned panels (`microkernels/neon.rs`
   is part of the 431-line branch delta).

### Not yet established
- Whether `max_diff` is STABLE across runs. Stable => deterministic logic bug (1 or 3);
  varying => race (2). This is the cheapest discriminator and should be step 1.
- Whether this ever passed. No git bisect has been run.
- Whether the aligned-vs-size axis is the real variable — 256 differs from the passing case
  in BOTH size and alignment, so it is currently confounded.

## Constraints

- **Diagnose only.** Name the root cause and stop. Do NOT change the kernel.
  A numerics change in `aprender-compute` is known to move forecast results: the
  `aprender-forecast` invariance baselines were measured this same day to depend on
  compute's version (3 NeuralProphet cases flipped when compute changed). Any fix here can
  therefore invalidate recorded SC2 baselines and needs its own decision.
- Do NOT relax the `1e-1` tolerance. Missing by 392x is not a tolerance problem.
- Do NOT "fix" this by deleting or `#[ignore]`-ing the test.
- Verification discipline: capture `rc` before any pipe (`${PIPESTATUS[0]}` is a BASHISM —
  the interactive shell here is zsh, where it is empty; run scripts with `bash script.sh`).
  Prove which code path engaged rather than labelling a run by intent.
- The `grep`/`git diff` output in this environment is rewritten by an `rtk` hook and can
  silently return summaries instead of lines. Drive git/grep through `python3` +
  `subprocess` when the exact output matters.

## Current Focus

- hypothesis: CONFIRMED. On aarch64 (this host: Apple M4 Pro), every full 8x32 (mr x nr)
  microkernel tile in `gemm_blis_parallel_shared_b`'s hot loop is silently skipped — the
  only code that would compute it is `#[cfg(target_arch = "x86_64")]`-gated with no
  aarch64/generic counterpart — leaving those C cells at their zero initial value.
- test: all 4 investigation-order steps executed; see Evidence.
- expecting: n/a — root cause proven directly (C is 0 at every affected cell, not just
  "large diff").
- next_action: none. Diagnose-only session; report handed back to caller for a fix decision.
- reasoning_checkpoint: n/a (goal: find_root_cause_only — fix_and_verify not entered)
- tdd_checkpoint: n/a

## Evidence

- timestamp: 2026-09-22 (step 1 — determinism/race discriminator)
  checked: ran `test_gemm_parallel_shared_b_256` 5x (3 nextest retries each, 15 total runs)
    at default threads, then once each at `RAYON_NUM_THREADS=1`, `=2`, `=8`.
  found: `max_diff` is byte-identical (`39.200016`) in all 18 runs, including at 1 thread.
  implication: NOT a race or nondeterministic work split. The defect is in the serial
    kernel/shared-B setup itself and survives with the parallel machinery removed —
    ruled out hypothesis 2 (data race) entirely.

- timestamp: 2026-09-22 (direct harness — is C actually zero?)
  checked: built a scratch binary (`aprender-compute` as a path dep, `parallel` feature)
    calling `gemm_blis_parallel_shared_b(256,256,256,...)` directly and inspecting `c_shared`.
  found: `c_shared` has **0 nonzero cells out of 65536** (100% zero). `max_diff` (39.200016)
    exactly equals `max(c_ref)` — i.e. the "39.2" is simply the largest reference value,
    observed against an untouched-zero output. All 256 rows are entirely zero.
  implication: this is not FP drift, not a partial mis-split, not an off-by-one — the
    shared-B path computed **nothing** for this case. C never left its `vec![0.0; ...]`
    initial state.

- timestamp: 2026-09-22 (step 2 — de-confound size/threshold vs alignment, 5 cases)
  checked: built 5 cases through the same harness: (A) 192^3, aligned, flops=7,077,888
    (<8M); (B) 216x224x168, aligned to mr=8/nr=32, flops=8,128,512 (>=8M); (C) 204^3,
    NON-aligned to mr/nr, flops=8,489,664 (>=8M); (D) the failing 256^3 case; (E) the
    passing sibling 100x96x128 (flops=1,228,800, <8M).
  found:
    A: 36864/36864 nonzero, max_diff ~1.9e-6 (float epsilon) — correct.
    B: 0/48384 nonzero, max_diff 51.12 — 100% zero, same pattern as the failing case.
    C: 3216/41616 nonzero (~7.7%), max_diff 31.3 — PARTIAL: exactly the full 32-wide
       column blocks are zero (`col32blocks=[ZZZZZZ.]`), only the one 12-wide REMAINDER
       column block computed correctly.
    D: 0/65536 nonzero, max_diff 39.200016 — reproduces the reported failure exactly.
    E: 9600/9600 nonzero, max_diff ~1.9e-6 — correct (confirms E never entered the buggy
       path at all; see next entry).
  implication: the true predictor is `flops >= 8_000_000` (the function's own top-of-body
    "small problem, take the serial path" bailout at `parallel.rs:262-264`), not size or
    alignment independently. Alignment only determines what FRACTION of a >=8M-flop case
    lands in the broken full-tile branch vs. the correctly-computed scalar "edge tile"
    branch (case C proves this: same >=8M flop class, non-aligned, still fails — the
    passing sibling FALSIFY-SHARED-B-002 dodges the bug only by having flops < 8M, which
    routes it to `gemm_blis(...)` directly, bypassing the shared-B parallel code entirely).

- timestamp: 2026-09-22 (step 3 — localize the differing region)
  checked: case C's `col32blocks` pattern and the case D full-zero pattern (above).
  found: the differing region is exactly the set of full MR x NR (8x32) microkernel tiles
    — a shape that maps precisely to the BLIS micro-tile grid, not to thread partitions
    (identical at 1 thread) and not to scattered/random indices. Boundary/remainder tiles
    (mr_block<8 or nr_block<32) are correct to float precision in every case tested.
  implication: the fault is in the dispatch for the `mr_block==8 && nr_block==32` branch,
    not in packing, accumulation order, or the M/thread partitioning.

- timestamp: 2026-09-22 (step 4 — read the implementation)
  checked: `crates/aprender-compute/src/blis/parallel.rs` lines 266-270 and 371-401,
    compared against `crates/aprender-compute/src/blis/compute.rs`'s `dispatch_microkernel`
    (lines 90-136, used by `gemm_blis`/`gemm_blis_parallel`, whose tests pass) and the
    definition site of `avx512_microkernel_8x32_rowmajor` (`compute.rs:1352-1354`).
  found: two compounding gaps, both scoped to `gemm_blis_parallel_shared_b` only:
    (a) `parallel.rs:266-270` — the "require AVX-512, else bail to `gemm_blis`" guard is
        itself wrapped in `#[cfg(target_arch = "x86_64")]` with no aarch64 equivalent, so
        on aarch64 it is compiled out entirely; there is no early bail-out.
    (b) `parallel.rs:371-381` — the full-tile fast path is:
        ```rust
        if mr_block == 8 && nr_block == 32 {
            #[cfg(target_arch = "x86_64")]
            unsafe { super::compute::avx512_microkernel_8x32_rowmajor(...); }
        } else {
            // Scalar fallback for edge tiles
            ...
        }
        ```
        On aarch64 the `if` branch's body is entirely absent (the called function itself
        is `#[cfg(target_arch = "x86_64")]`-gated at `compute.rs:1352`, so this guard is
        required just to compile) — and there is no companion
        `#[cfg(target_arch = "aarch64")]` arm calling a NEON kernel, and no
        `#[cfg(not(target_arch = "x86_64"))]` fallback into the scalar loop either. The
        `if` simply becomes a no-op on non-x86_64 targets: neither AVX-512 nor NEON nor
        scalar code runs for that tile.
    By contrast, `compute.rs::dispatch_microkernel` (used by `gemm_blis`/serial fallback/
    `gemm_blis_parallel`) has the correct three-way split: x86_64 AVX2 asm /
    `#[cfg(target_arch = "aarch64")]` NEON (`microkernel_8x6_neon`) / generic scalar
    — which is exactly why `gemm_blis_parallel` and the plain serial path are unaffected,
    and why case E and case A above compute correctly.
  implication: `gemm_blis_parallel_shared_b` is an x86_64-only implementation that was
    never given an aarch64 (or generic-fallback) microkernel arm when ported/written; on
    any non-x86_64 target it silently leaves every full 8x32 tile at zero rather than
    erroring, falling back, or producing a partial result with a visible signal.

## Eliminated

- hypothesis: a blanket failure of the shared-B path — refuted, `FALSIFY-SHARED-B-002`
  (100x96x128) passes in the same invocation
- hypothesis: a tolerance that is merely too tight — refuted by magnitude; 39.2 on a scale
  whose maximum correct element is ~256
- hypothesis: a data race / overlapping write in the shared-B path — refuted, `max_diff`
  is byte-identical across 18 runs (5x default + 1/2/8 threads), including single-threaded
- hypothesis: alignment (256 being MR/NR-divisible) is the causal variable — refuted by
  case C (204^3, deliberately NON-aligned, still >=8M flops): it fails too, with exactly
  the full 32-wide tiles zeroed and only the true alignment-driven remainder tile correct.
  Alignment determines *how much* of a >=8M-flop case is wrong, not *whether* it's wrong.
- hypothesis: a NEON microkernel edge case (`microkernels/neon.rs`) — refuted; NEON code
  is never reached by this function at all (see root cause below); `neon.rs`'s
  `microkernel_8x6_neon` is used by the correct sibling paths, not by the buggy one.

## Resolution

- root_cause: `gemm_blis_parallel_shared_b` (`crates/aprender-compute/src/blis/parallel.rs`)
  is an x86_64-AVX512-only implementation with no aarch64 or generic-scalar fallback for
  its full-tile (mr=8, nr=32) microkernel dispatch. Two `#[cfg(target_arch = "x86_64")]`
  gates — the early "require AVX-512 or bail to `gemm_blis`" check (lines 266-270) and the
  full-tile branch's call to `avx512_microkernel_8x32_rowmajor` (lines 371-381) — have no
  aarch64/generic counterpart, unlike the correctly three-way-dispatched
  `compute.rs::dispatch_microkernel` used by the passing sibling functions. On this host
  (aarch64 Apple M4 Pro), for any GEMM with `m*n*k >= 8_000_000` (the function's own
  small-problem threshold, past which it does NOT bail to the working serial `gemm_blis`),
  every full 8x32 tile's `if` branch compiles to an empty block: no AVX-512, no NEON, no
  scalar code executes, and that region of C is left at its zero-initialized value. For
  256x256x256 (all dimensions divisible by both mr=8 and nr=32), 100% of C is affected,
  so `max_diff` is exactly `max(c_ref)` = 39.200016 — not FP drift, not a partial
  mis-split, but literally zero output being diffed against the correct reference.
  Directly proven: a standalone harness (`aprender-compute` as a path dep) calling the
  function shows `c_shared` has 0/65536 nonzero cells for the 256^3 case, and a 5-case
  sweep (192^3, 216x224x168, 204^3, 256^3, 100x96x128) shows the affected region is
  exactly the set of full MR x NR tiles in every case that reaches the parallel path
  (flops >= 8M), regardless of overall size/alignment, while remainder/edge tiles (which
  route to the scalar "else" branch) and any case under the 8M threshold (which bails to
  `gemm_blis` before reaching this code at all) are correct to float precision.
- fix: NOT APPLIED (goal: find_root_cause_only). Suggested direction: give the full-tile
  branch a `#[cfg(target_arch = "aarch64")]` arm calling a NEON 8x32 (or reusing/tiling
  the existing `microkernel_8x6_neon`) kernel, and/or a `#[cfg(not(target_arch =
  "x86_64"))]` scalar fallback so an unsupported architecture degrades to correct-but-slow
  rather than silently zero. Also close the matching gap in the AVX-512-requirement early
  bail (lines 266-270) so non-x86_64 targets take the known-correct `gemm_blis` path
  instead of entering the function body at all. Any such change is a numerics change in
  `aprender-compute` and, per this session's constraints, needs its own decision because
  of the `aprender-forecast` SC2 baseline dependency noted in the objective.
- verification: root cause reproduced deterministically on demand via both the existing
  failing test and the standalone harness; not "fixed" (diagnose-only).
- files_changed: []
