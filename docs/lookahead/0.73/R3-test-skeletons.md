# R3 test skeletons: FALSIFY-NEON-Q4K-001..003, 006, 007 (draft, la-73, 2026-09-30; L25 amended 2026-10-03)

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

## 5. What this needs before it can run

| Need | State |
|------|-------|
| `kernel_path(k)` API | R3 code (not written; mint deferred, cop 10:20Z 09-28). It is also emitted in the forward trace, because the R1 receipt checker reads it: FALSIFY-BPM-012 refuses a C4 receipt whose backend and reference `kernel_path` match |
| `kernel_path` shape | reuse `apr-kernel-path-v1` (OBS-15, unmerged #4574): `kernel_path(k)` becomes the entry's `kernel_id`, with `arch = aarch64`. See P1 spec §3a |
| `fused_q4k_q8k_dot_neon_widen` test entry | R3 code (NEON-Q4K-007) |
| Orphan `quantize/fused_q4k.rs` deleted | PROPOSE-TICKET 07:49Z; NEON-Q4K-006 keeps it as a control until then |
| Orphan `quantize/fused_q.rs` deleted | PROPOSE-TICKET (R3 §13 row 9). It is not a dot site, so NEON-Q4K-006 needs no second control |
| An aarch64 run (gx10) | GPU-deferred while a train is active; runs on CPU only, so it may be admissible earlier. Cop to rule |
| An aarch64 CI lane | none today; FALSIFY-NEON-Q4K-000 proposes a report-only `cargo check --target aarch64` step |
