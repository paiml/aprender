# R3 test skeletons: FALSIFY-NEON-Q4K-001..003, 006, 007, 008 (draft, la-73, 2026-09-30; L25 amended 2026-10-03)

Contract: `contracts-draft/neon-q4k-q6k-v1.yaml`. The code below is a draft and is
**not compiled here**. It lands with the R3 kernels, as a sibling of the existing
parity tests in `crates/aprender-serve/src/quantize/`.

## 1. What already exists (read at d63d8935e4)

SIMD-vs-scalar parity tests already exist for all three dispatchers:

| Kernel | Existing test | Inputs | Tolerance |
|--------|---------------|--------|-----------|
| q4k-f32 | `fused_k_tests.rs::test_fused_q4k_dot_simd_matches_scalar`, `fused_k_tests_q4k.rs::test_fused_q4k_dot_simd_vs_scalar_varied_activations` | 1 fixed block, all 12 scale bytes `0x11` | relative, no floor |
| q4k-q8k | `fused_k_tests.rs::test_fused_q4k_q8k_dot_simd_matches_scalar`, `fused_k_tests_q4k.rs:330/367`, `tests/fused.rs:32` | fixed blocks | relative |
| q6k-f32 | `create.rs::test_fused_q6k_dot_simd_matches_scalar` | 1 fixed block, all 16 scales `1` | **1 %** |

The contract needs three things these tests cannot give it:

1. **On aarch64 they pass without testing anything (L25).** Today `*_dot_simd`
   falls through to scalar on aarch64, so each assertion compares scalar with
   itself. It will keep passing if the NEON arm is never reached, which is the
   exact defect `dispatch_honesty` names. Each skeleton therefore reads
   `kernel_path` first. If it is not a NEON path on aarch64, the test **fails**
   once R3 lands, and is `NotRun` before then (see §3).
2. **Uniform scales hide scale-index bugs.** When every sub-block has the same
   scale and min, a NEON kernel that reads scale `j` for sub-block `j+1` gives
   the same answer. The skeleton seeds every byte, scales included.
3. **The tolerance is wrong.** The contract says `1e-3 * max(1, |scalar|)`.
   q6k-f32 uses 1 %, which is 10× looser, and none of the tests has the floor
   of 1.

None of these tests runs on aarch64 in CI, because CI is x86-only. They only
witness C4 when run on gx10, which is GPU-deferred while a train is active
(Next item 1).

## 2. Skeleton: FALSIFY-NEON-Q4K-001 (parity, 10 000 seeded blocks per kernel)

```rust
// crates/aprender-serve/src/quantize/neon_parity_tests.rs  (lands with R3)
use super::{fused_q4k_dot, fused_q4k_dot_simd, fused_q4k_q8k_dot, fused_q4k_q8k_dot_simd,
            fused_q6k_dot, fused_q6k_dot_simd};
use proptest::prelude::*;

const EPS: f32 = 1e-3; // contract neon_scalar_parity; avx512-q4k-v1 precedent

fn close(neon: f32, scalar: f32) -> bool {
    (neon - scalar).abs() < EPS * scalar.abs().max(1.0)
}

/// d and dmin come from a finite, NORMAL, non-zero f32 range through half::f16.
/// (2026-10-03 L25: the old `b[1] &= 0x3B` mask kept d finite but allowed
/// d = 0 and subnormals, and dmin = 0 makes the min term invisible.)
/// The 12 packed scale/min bytes and the 128 qs bytes stay fully random, so
/// the scales and mins vary across the 8 sub-blocks.
fn f16_le(lo: f32, hi: f32) -> impl Strategy<Value = [u8; 2]> {
    (lo..hi).prop_map(|v| half::f16::from_f32(v).to_le_bytes())
}

fn q4k_one() -> impl Strategy<Value = Vec<u8>> {
    (f16_le(1e-3, 0.5), f16_le(1e-3, 0.5), prop::collection::vec(any::<u8>(), 140))
        .prop_map(|(d, dmin, rest)| [d.as_slice(), dmin.as_slice(), rest.as_slice()].concat())
}

fn q6k_one() -> impl Strategy<Value = Vec<u8>> {
    (prop::collection::vec(any::<u8>(), 208), f16_le(1e-3, 0.5))
        .prop_map(|(body, d)| [body.as_slice(), d.as_slice()].concat()) // d = last 2 bytes
}

/// Rows of n in {1, 2, 3, 7} super-blocks, so the block loop and the
/// accumulator reset run (contract precondition, L25 2026-10-03).
fn rows<S: Strategy<Value = Vec<u8>>>(one: fn() -> S) -> impl Strategy<Value = (usize, Vec<u8>)> {
    prop::sample::select(vec![1usize, 2, 3, 7]).prop_flat_map(move |n| {
        (Just(n), prop::collection::vec(one(), n).prop_map(|v| v.concat()))
    })
}

fn acts(n: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-4.0f32..4.0, n * 256)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 10_000, ..ProptestConfig::default() })]

    #[test]
    fn falsify_neon_q4k_001_q4k_f32((b, x) in rows(q4k_one).prop_flat_map(|(n, b)| (Just(b), acts(n)))) {
        require_neon_path("q4k-f32")?;
        let s = fused_q4k_dot(&b, &x).expect("scalar oracle");
        let n = fused_q4k_dot_simd(&b, &x).expect("neon");
        prop_assert!(close(n, s), "neon {n} vs scalar {s}");
    }

    #[test]
    fn falsify_neon_q4k_001_q6k_f32((b, x) in rows(q6k_one).prop_flat_map(|(n, b)| (Just(b), acts(n)))) {
        require_neon_path("q6k-f32")?;
        let s = fused_q6k_dot(&b, &x).expect("scalar oracle");
        let n = fused_q6k_dot_simd(&b, &x).expect("neon");
        prop_assert!(close(n, s), "neon {n} vs scalar {s}");
    }

    #[test]
    fn falsify_neon_q4k_001_q4k_q8k((b, q, d) in rows(q4k_one).prop_flat_map(|(n, b)| (
                                        Just(b),
                                        prop::collection::vec(-127i8..=127, n * 256), // Q8_K range
                                        prop::collection::vec(1e-4f32..1.0, n)))) {
        require_neon_path("q4k-q8k")?;
        let s = fused_q4k_q8k_dot(&b, &d, &q).expect("scalar oracle");
        let n = fused_q4k_q8k_dot_simd(&b, &d, &q).expect("neon");
        prop_assert!(close(n, s), "neon {n} vs scalar {s}");
    }
}
```

**Mutation (from the contract):** swap the low and high nibble order in the q4k NEON
kernel; `falsify_neon_q4k_001_q4k_f32` must turn RED. A second mutation for the
scale gap in §1.2: shift the scale index by one sub-block. The uniform-scale tests
stay green under it, and this one must turn RED. A third mutation (contract,
2026-10-03): drop the `dmin * min` term. It bites only because `dmin` is now non-zero by
construction.

**Magnitude floor (contract precondition).** Below |dot| = 1 the epsilon is an absolute
1e-3, so the fixture distribution is itself under test:

```rust
#[test]
fn falsify_neon_q4k_001_magnitude_floor() {
    use proptest::test_runner::TestRunner;
    let mut r = TestRunner::deterministic();
    let strat = rows(q4k_one).prop_flat_map(|(n, b)| (Just(b), acts(n)));
    let big = (0..10_000)
        .map(|_| strat.new_tree(&mut r).expect("tree").current())
        .filter(|(b, x)| fused_q4k_dot(b, x).expect("scalar").abs() >= 1.0)
        .count();
    eprintln!("magnitude floor: {big}/10000 cases with |dot| >= 1"); // printed, per contract
    assert!(big >= 9_000, "fixture too small: epsilon is absolute for {} cases", 10_000 - big);
}
```
The same test runs for q6k-f32 and q4k-q8k. It is arch-independent (scalar only), so it also
runs on x86 CI and gives the fixture a check that is not NotRun there.

## 3. Skeleton: FALSIFY-NEON-Q4K-002 (dispatch honesty, and the no-vacuous-pass guard)

`kernel_path(k)` does not exist yet. R3 adds it next to each dispatcher, and it
returns the name of the arm the call actually takes, decided by the same `cfg` and
feature checks as the dispatch.

```rust
/// Rejects the case, which is NotRun and not PASS, off aarch64, and fails
/// on aarch64 if the call does not reach NEON. Every parity case calls this
/// first, so none of them can pass by comparing scalar with scalar.
fn require_neon_path(k: &str) -> Result<(), TestCaseError> {
    if !cfg!(target_arch = "aarch64") {
        return Err(TestCaseError::reject("NotRun: not aarch64")); // counted, never PASS
    }
    let path = super::kernel_path(k);
    if !path.starts_with(&format!("{k}/neon")) {
        return Err(TestCaseError::fail(format!("dispatch_honesty: {k} reached {path}")));
    }
    Ok(())
}

#[cfg(target_arch = "aarch64")] // §3 option (a)
#[test]
fn falsify_neon_q4k_002_q4k_q8k_path_matches_dotprod() {
    // L25 (2026-10-03): the earlier draft asserted only inside
    // `if !dotprod`, so on gx10 (dotprod present) it asserted nothing and passed.
    // Both branches now assert.
    let want = if std::arch::is_aarch64_feature_detected!("dotprod") {
        "q4k-q8k/neon-sdot"
    } else {
        "q4k-q8k/neon-widen"
    };
    assert_eq!(super::kernel_path("q4k-q8k"), want);
}
```

**Open point (L25):** proptest counts too many rejects as a failure ("too many
global rejects"). On x86 that turns NotRun into FAIL, which is loud but wrong. There
are two options. (a) Put the whole module behind `#[cfg(target_arch = "aarch64")]`,
so x86 lists 0 tests. The gate that consumes the result must then treat an absent
test as `not_measured`, never as a pass. (b) Keep the reject and rely on a runner
filter. **L3 recommends (a):** an absent test is visible in the listing, while a
reject-to-FAIL is noise that people learn to ignore. With (a), the plain
`#[test]` above drops its early return.

**Mutation:** delete the `cfg(aarch64)` arm in `fused_q4k_dot_simd`. Then
`kernel_path` reads `q4k-f32/scalar` and every parity case fails with a
dispatch_honesty message instead of passing against itself.

## 4. Skeleton: FALSIFY-NEON-Q4K-003 (cross-arch golden)

- **Record** on x86, from the scalar oracle only: 64 seeded blocks per kernel, at a
  fixed seed. Write `(kernel, block_hex, acts_seed, dot_f32_bits)` rows to
  `crates/aprender-serve/tests/golden/neon-q4k-q6k-v1.tsv`.
- **Check** on aarch64: recompute with `*_dot_simd` and compare to the stored
  value with `close`.
- The golden file is generated by a `#[ignore]` test on x86 and committed. The
  check test is aarch64-only, per §3 option (a).
- **Mutation:** flip one scale byte in one golden row; the check must turn RED on
  that row only.
- **Pinned golden (contract, 2026-10-03).** The check test first asserts
  `sha256(golden.tsv) == GOLDEN_SHA256` (sha2 is already a serve dependency). If the file was
  regenerated in the same run, or edited without updating the constant, the test fails
  before it compares anything. A golden that the test writes and then reads is a
  self-comparison.

## 4a. Skeleton: FALSIFY-NEON-Q4K-006 (each NEON arm sits in a compiled file)

This is a script, not a `#[test]`: it needs the aarch64 cross target, so it runs with
NEON-Q4K-000, off the release path. It works on a scratch copy of the tree, never the worktree:

```bash
#!/usr/bin/env bash
set -euo pipefail
# usage: neon_site_compiled.sh <scratch-tree>   (a copy; this script edits it)
tree="${1:?scratch tree}"
q="${tree}/crates/aprender-serve/src/quantize"
plant='#[cfg(target_arch = "aarch64")] compile_error!("NEON-Q4K-006 planted");'
check() { cargo check -q -p aprender-serve --lib --target aarch64-unknown-linux-gnu \
            --manifest-path "${tree}/Cargo.toml" >/dev/null 2>&1; }
rc=0
for site in fused_k.rs fused_q5k_q6k.rs q4k_dot_avx2.rs; do       # live: must go RED
  cp "${q}/${site}" "${q}/${site}.bak"
  printf '%s\n' "${plant}" >> "${q}/${site}"
  if check; then echo "FAIL ${site}: planted error compiled away (dead site)"; rc=1
  else echo "ok ${site}: compiled"; fi
  mv "${q}/${site}.bak" "${q}/${site}"
done
cp "${q}/fused_q4k.rs" "${q}/fused_q4k.rs.bak"                    # orphan control
printf '%s\n' "${plant}" >> "${q}/fused_q4k.rs"
if check; then echo "ok fused_q4k.rs: orphan (control holds)"
else echo "FAIL control: fused_q4k.rs is compiled now; update the R3 cites"; rc=1; fi
mv "${q}/fused_q4k.rs.bak" "${q}/fused_q4k.rs"
exit "${rc}"
```
The orphan control proves the probe can tell the two cases apart. Once the PROPOSE-TICKET
delete lands, the control is dropped in the same commit.

## 4b. Skeleton: FALSIFY-NEON-Q4K-007 (widen fallback, tested on a dotprod host)

R3 exposes the widen kernel to tests directly, because dispatch on gx10 never reaches it:

```rust
#[cfg(target_arch = "aarch64")]
proptest! {
    #![proptest_config(ProptestConfig { cases: 10_000, ..ProptestConfig::default() })]
    #[test]
    fn falsify_neon_q4k_007_widen_matches_scalar((b, q, d) in /* same strategy as 001_q4k_q8k */) {
        let s = fused_q4k_q8k_dot(&b, &d, &q).expect("scalar oracle");
        // SAFETY: NEON is baseline on aarch64; the widen kernel needs no extension.
        let w = unsafe { super::fused_q4k_q8k_dot_neon_widen(&b, &d, &q) }.expect("widen");
        prop_assert!(close(w, s), "widen {w} vs scalar {s}");
    }
}
```
**Mutation:** widen the i8 lanes as unsigned (`vmovl_u8` in place of `vmovl_s8`). Negative
activations then read as values of 128 or more, and the test must turn RED.
Note: aprender-serve sets `[lints.rust] unsafe_code = "allow"` (Cargo.toml:31-32, read
2026-10-03), and clippy requires a `// SAFETY:` comment on every `unsafe` block (PMAT-134).
The entry follows the AVX2 precedent (`unsafe fn` plus a SAFETY comment at the call site).

## 4c. Skeleton: FALSIFY-NEON-Q4K-008 (every matvec row, at production row lengths)

Every CPU matvec writes 0.0 for a row whose dot returns Err, so no caller sees a NEON Err. The §2 tests would see
one, but only on rows of 1 to 7 blocks. This test calls each row's dot directly at 8 and 36 blocks per row, then
checks that the matvec wrote that value to that row. It runs all five decode entries and the two prefill (multirow)
entries, F and G, for 3 tokens at once (contract `matvec_row_parity`; R3 §13 row 10).

```rust
// crates/aprender-serve/src/quantize/neon_parity_tests.rs  (lands with R3; adds to the §2 imports)
use super::{fused_q4k_multirow_matmul_f32_into, fused_q4k_parallel_matvec_f32_into,
            fused_q4k_parallel_matvec_into, fused_q4k_q8k_ffn_up_gate_into,
            fused_q6k_multirow_matmul_into, fused_q6k_parallel_matvec_into,
            quantize_activations_q8k_into, with_fp32_activations};
use crate::error::Result;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;

/// Contract matvec_row_parity: 8 and 36 super-blocks per row, and out_dim on both
/// sides of the sequential/parallel split (300 ends in a 44-row tail tile).
const SHAPES: [(usize, usize); 4] = [(2048, 48), (2048, 300), (9216, 48), (9216, 300)];
const PLANT_ROW: usize = 280; // inside the tail tile, rows 256..300
const M: usize = 3; // tokens for the prefill entries F and G; G takes its multirow route only for m > 1

fn draw<T: std::fmt::Debug, S: Strategy<Value = T>>(r: &mut TestRunner, s: S) -> T {
    s.new_tree(r).expect("tree").current()
}

/// rows x n blocks from the §2 generators, row-major.
fn matrix<S: Strategy<Value = Vec<u8>>>(r: &mut TestRunner, one: fn() -> S, rows: usize, n: usize) -> Vec<u8> {
    (0..rows * n).flat_map(|_| draw(r, one())).collect()
}

/// Calls each row's dot directly, so a NEON Err panics here instead of becoming 0.0.
/// Then checks that the matvec wrote that value to that row.
fn check_rows(entry: &str, got: &[f32], w: &[u8], row_bytes: usize, dot: impl Fn(&[u8]) -> Result<f32>) {
    assert_eq!(w.len(), got.len() * row_bytes, "{entry}: fixture shape");
    for (r, (g, row)) in got.iter().zip(w.chunks(row_bytes)).enumerate() {
        let v = dot(row).unwrap_or_else(|e| panic!("{entry}: dot Err on row {r}: {e}"));
        assert!(v.abs() >= 0.01, "{entry}: fixture row {r} has a near-zero dot {v}");
        assert!(close(*g, v), "{entry}: row {r} = {g}, its dot = {v}");
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn falsify_neon_q4k_008_every_matvec_row() {
    for k in ["q4k-f32", "q6k-f32", "q4k-q8k"] {
        require_neon_path(k).expect("dispatch_honesty"); // §3
    }
    assert_ne!(std::env::var("DIRECT_FP32_GEMV").as_deref(), Ok("1"), "entries C and G would run f32");
    let mut r = TestRunner::deterministic();
    for (in_dim, out_dim) in SHAPES {
        let n = in_dim / 256;
        let mut up = matrix(&mut r, q4k_one, out_dim, n);
        if out_dim > PLANT_ROW {
            // Mutation 2's trigger: the first 16 qs bytes of this row's first block (qs start at byte 16).
            let at = PLANT_ROW * n * 144 + 16;
            up[at..at + 16].fill(0x5A);
        }
        let gate = matrix(&mut r, q4k_one, out_dim, n);
        let w6 = matrix(&mut r, q6k_one, out_dim, n);
        let x = draw(&mut r, acts(n));
        let (mut d, mut q) = (vec![0.0f32; n], vec![0i8; in_dim]);
        quantize_activations_q8k_into(&x, &mut d, &mut q).expect("q8k");
        let f32_dot = |row: &[u8]| fused_q4k_dot_simd(row, &x);
        let q8k_dot = |row: &[u8]| fused_q4k_q8k_dot_simd(row, &d, &q);
        let nan = || vec![f32::NAN; out_dim]; // a row the entry never writes cannot pass

        let mut y = nan();
        with_fp32_activations(|| fused_q4k_parallel_matvec_into(&up, &x, in_dim, out_dim, &mut y)).expect("A");
        check_rows("A", &y, &up, n * 144, f32_dot);
        let mut y = nan();
        fused_q4k_parallel_matvec_f32_into(&up, &x, in_dim, out_dim, &mut y).expect("B");
        check_rows("B", &y, &up, n * 144, f32_dot);
        let mut y = nan();
        fused_q4k_parallel_matvec_into(&up, &x, in_dim, out_dim, &mut y).expect("C");
        check_rows("C", &y, &up, n * 144, q8k_dot);
        let (mut yu, mut yg) = (nan(), nan());
        fused_q4k_q8k_ffn_up_gate_into(&up, &gate, &d, &q, in_dim, out_dim, &mut yu, &mut yg).expect("D");
        check_rows("D up", &yu, &up, n * 144, q8k_dot);
        check_rows("D gate", &yg, &gate, n * 144, q8k_dot);
        let mut y = nan();
        fused_q6k_parallel_matvec_into(&w6, &x, in_dim, out_dim, &mut y).expect("E");
        check_rows("E", &y, &w6, n * 210, |row: &[u8]| fused_q6k_dot_simd(row, &x));

        // Prefill: M tokens at once, output token-major (token t is y[t * out_dim..][..out_dim]).
        let xs: Vec<f32> = (0..M).flat_map(|_| draw(&mut r, acts(n))).collect();
        let mut y = vec![f32::NAN; M * out_dim];
        fused_q6k_multirow_matmul_into(&w6, &xs, M, in_dim, out_dim, &mut y).expect("F");
        for (t, (xt, yt)) in xs.chunks(in_dim).zip(y.chunks(out_dim)).enumerate() {
            check_rows(&format!("F t{t}"), yt, &w6, n * 210, |row: &[u8]| fused_q6k_dot_simd(row, xt));
        }
        let mut y = vec![f32::NAN; M * out_dim];
        fused_q4k_multirow_matmul_f32_into(&up, &xs, M, in_dim, out_dim, &mut y).expect("G"); // outside the scope
        for (t, (xt, yt)) in xs.chunks(in_dim).zip(y.chunks(out_dim)).enumerate() {
            let (mut dt, mut qt) = (vec![0.0f32; n], vec![0i8; in_dim]);
            quantize_activations_q8k_into(xt, &mut dt, &mut qt).expect("q8k");
            check_rows(&format!("G t{t}"), yt, &up, n * 144, |row: &[u8]| fused_q4k_q8k_dot_simd(row, &dt, &qt));
        }
    }
}
```
**Mutation (from the contract):**
1. Make a NEON arm return Err for rows of 8 or more super-blocks. The §2 tests (1 to 7 blocks) stay GREEN. This
   test turns RED in its first shape (in_dim 2048, 8 blocks), since every arm is reached there.
2. Make a Q4_K NEON arm return Err for the planted block. The fixture plants it on every run (row 280, inside the
   44-row tail tile of the out_dim 300 cases), so the mutation needs no fixture edit. The test turns RED on row 280.
3. In the parallel branch of `fused_q4k_q8k_parallel_matvec_into` (q5k_q6k_matvec.rs:319), write 0.0 in place of one
   tail-tile row. Entry C turns RED on that row. This one tests the row check itself: `close(0.0, v)` fails whenever
   |v| >= 0.01, which the fixture asserts.
4. In `generic_multirow_matmul_into` (generic_matvec.rs:238-239), write 0.0 for the last row of the part tile for
   token m - 1. Entry F turns RED on that row. No decode entry reaches that loop.

**Why the reference is each row's own dot.** `close` bounds the error relative to the reference, while f32
summation-order error grows with the row's sum of |w·x|. A 36-block row whose terms cancel could miss the bound
against the scalar oracle with no kernel defect. So value parity stays with §2, and taking it past 7 blocks needs
an error term first. On aarch64, `fused_q4k_q8k_dot_with_bsums_simd` falls through to `fused_q4k_q8k_dot_simd`
(bsum_precompute.rs:237), so entries C and D reach that one dispatcher whether bsums is set or not. A NEON bsums
variant would change their reference.

**Why the prefill entries, with no C4 caller yet.** The only production caller of the multirow kernels,
`matmul_rows` (forward_qwen35.rs:1416), is reached only from `forward_prefill_qwen35` (:1200), and only its tests
call that (forward_qwen35_prefill_tests.rs). FALSIFY-4228-004 and -005 check the batched prefill bit-identical to
`forward_single_qwen35`, and a NEON Err would pass them: on aarch64 both sides call the same dot on the same row and
swallow the Err into the same 0.0. F and G close that before the batched prefill is wired into a run. G reads only
`DIRECT_FP32_GEMV`, never the fp32 scope (q4k_q8k_multirow.rs:178-179), so it runs outside the scope and its
reference is each token's own Q8_K dot. `fused_q5k_multirow_matmul_into` is left out: Q5_K is not an R3 kernel.

## 5. What this needs before it can run

| Need | State |
|------|-------|
| `kernel_path(k)` API | R3 code (not written; mint deferred, cop 10:20Z 09-28). It is also emitted in the forward trace, because the R1 receipt checker reads it: FALSIFY-BPM-012 refuses a C4 receipt in which one quantized matmul tensor has the same `kernel_id` in the backend and reference runs |
| `kernel_path` shape | reuse `apr-kernel-path-v1` (OBS-15, unmerged #4574): `kernel_path(k)` becomes the entry's `kernel_id`, with `arch = aarch64`. On x86 it names the x86 arm (e.g. `q4k-f32/avx2`), because the C0 reference is read too. Each trace entry also carries `tensor`, the GGUF name, since BPM-012 compares per tensor. See P1 spec §3a |
| `fused_q4k_q8k_dot_neon_widen` test entry | R3 code (NEON-Q4K-007) |
| The seven matvec entries (five decode, two prefill), `with_fp32_activations` and `quantize_activations_q8k_into` in reach of a `quantize/` test (NEON-Q4K-008) | exist at 316dee2cd4: re-exported at quantize/mod.rs:132 and :140-146 (the multirow ones at :141 and :145); the quantizer is at mod.rs:241 |
| Orphan `quantize/fused_q4k.rs` deleted | PROPOSE-TICKET 07:49Z; NEON-Q4K-006 keeps it as a control until then |
| Orphan `quantize/fused_q.rs` deleted | PROPOSE-TICKET (R3 §13 row 9). It is not a dot site, so NEON-Q4K-006 needs no second control |
| An aarch64 run (gx10) | GPU-deferred while a train is active; runs on CPU only, so it may be admissible earlier. Cop to rule |
| An aarch64 CI lane | none today; FALSIFY-NEON-Q4K-000 proposes a report-only `cargo check --target aarch64` step |
