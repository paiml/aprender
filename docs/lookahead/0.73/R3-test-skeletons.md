# R3 test skeletons: FALSIFY-NEON-Q4K-001..003 (draft, la-73, 2026-09-30)

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

/// Every byte is seeded, including d/dmin and the 12 packed scale bytes.
/// d and dmin are clamped to finite f16 values so the oracle stays finite.
fn q4k_block() -> impl Strategy<Value = Vec<u8>> {
    (prop::array::uniform32(any::<u8>()), prop::collection::vec(any::<u8>(), 112))
        .prop_map(|(head, tail)| {
            let mut b = [head.as_slice(), tail.as_slice()].concat(); // 144 bytes
            b[1] &= 0x3B; // d: finite, |d| < 1.0
            b[3] &= 0x3B; // dmin: same
            b
        })
}

fn q6k_block() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 210).prop_map(|mut b| {
        b[209] &= 0x3B; // d (last 2 bytes): finite
        b
    })
}

fn acts() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-4.0f32..4.0, 256)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 10_000, ..ProptestConfig::default() })]

    #[test]
    fn falsify_neon_q4k_001_q4k_f32(b in q4k_block(), x in acts()) {
        require_neon_path("q4k-f32")?;
        let s = fused_q4k_dot(&b, &x).expect("scalar oracle");
        let n = fused_q4k_dot_simd(&b, &x).expect("neon");
        prop_assert!(close(n, s), "neon {n} vs scalar {s}");
    }

    #[test]
    fn falsify_neon_q4k_001_q6k_f32(b in q6k_block(), x in acts()) {
        require_neon_path("q6k-f32")?;
        let s = fused_q6k_dot(&b, &x).expect("scalar oracle");
        let n = fused_q6k_dot_simd(&b, &x).expect("neon");
        prop_assert!(close(n, s), "neon {n} vs scalar {s}");
    }

    #[test]
    fn falsify_neon_q4k_001_q4k_q8k(b in q4k_block(),
                                    q in prop::collection::vec(any::<i8>(), 256),
                                    d in 1e-4f32..1.0) {
        require_neon_path("q4k-q8k")?;
        let s = fused_q4k_q8k_dot(&b, &[d], &q).expect("scalar oracle");
        let n = fused_q4k_q8k_dot_simd(&b, &[d], &q).expect("neon");
        prop_assert!(close(n, s), "neon {n} vs scalar {s}");
    }
}
```

**Mutation (from the contract):** swap the low and high nibble order in the q4k NEON
kernel; `falsify_neon_q4k_001_q4k_f32` must turn RED. A second mutation for the
scale gap in §1.2: shift the scale index by one sub-block. The uniform-scale tests
stay green under it, and this one must turn RED.

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

#[test]
fn falsify_neon_q4k_002_q4k_q8k_without_dotprod_is_widen() {
    if !cfg!(target_arch = "aarch64") { return; } // reported NotRun by the runner filter, see note
    #[cfg(target_arch = "aarch64")]
    if !std::arch::is_aarch64_feature_detected!("dotprod") {
        assert_eq!(super::kernel_path("q4k-q8k"), "q4k-q8k/neon-widen");
    }
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

## 5. What this needs before it can run

| Need | State |
|------|-------|
| `kernel_path(k)` API | R3 code (not written; mint deferred, cop 10:20Z 09-28) |
| An aarch64 run (gx10) | GPU-deferred while a train is active; runs on CPU only, so it may be admissible earlier. Cop to rule |
| An aarch64 CI lane | none today; FALSIFY-NEON-Q4K-000 proposes a report-only `cargo check --target aarch64` step |
