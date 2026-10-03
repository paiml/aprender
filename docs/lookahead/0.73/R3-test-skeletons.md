# R3 test skeletons: FALSIFY-NEON-Q4K-001..003, 006, 007, 008 (draft, la-73, 2026-09-30; L25 and item (d) amended 2026-10-03)

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
3. **The tolerance has no basis.** q6k-f32 uses 1 % and the others a bare
   relative error. The contract now derives one (item d, 2026-10-03):
   `2 * gamma(K) * S` over the row's terms, which holds at every row length (§2).

None of these tests runs on aarch64 in CI, because CI is x86-only. They only
witness C4 when run on gx10, which is GPU-deferred while a train is active
(Next item 1).

## 2. Skeleton: FALSIFY-NEON-Q4K-001 (parity, 10 000 seeded rows per kernel, and the masked sweep)

```rust
// crates/aprender-serve/src/quantize/neon_parity_tests.rs  (lands with R3)
use super::{
    dequantize_q4_k, dequantize_q6_k, extract_scale_min, fused_q4k_dot, fused_q4k_dot_simd,
    fused_q4k_q8k_dot, fused_q4k_q8k_dot_simd, fused_q6k_dot, fused_q6k_dot_simd, read_f16,
};
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::TestRunner;

/// Contract neon_scalar_parity (item d, 2026-10-03): |neon - scalar| <= 2 * gamma(K) * S,
/// where S is the row's sum of |term|. Each kernel is within gamma(K) * S of the exact sum
/// (Higham 2002, section 3.1, Lemma 3.1), so no fixture turns this RED without a defect.
const U: f64 = 1.0 / 16_777_216.0; // 2^-24, the f32 unit roundoff

fn gamma(k: usize) -> f64 {
    let ku = k as f64 * U;
    ku / (1.0 - ku)
}

/// The difference of two f32 is exact in f64, and a NaN on either side fails.
fn within(neon: f32, scalar: f32, tol: f64) -> bool {
    (f64::from(neon) - f64::from(scalar)).abs() <= tol
}

/// q4k-f32: K = 256n + 2, and S sums |d*sc_j*q_i*x_i| + |dmin*m_j*x_i|. Layout as
/// fused_q4k_dot (fused_k.rs:60): in 64-value chunk c, the 32 low nibbles are
/// sub-block 2c and the 32 high nibbles sub-block 2c + 1.
fn tol_q4k_f32(b: &[u8], x: &[f32]) -> f64 {
    let mut s = 0.0f64;
    for (blk, xb) in b.chunks(144).zip(x.chunks(256)) {
        let d = f64::from(read_f16(&blk[0..2]));
        let dmin = f64::from(read_f16(&blk[2..4]));
        let scales: &[u8; 12] = blk[4..16].try_into().expect("12 scale bytes");
        for (c, qs) in blk[16..144].chunks(32).enumerate() {
            for h in 0..2 {
                let (sc, m) = extract_scale_min(scales, 2 * c + h);
                for (l, &byte) in qs.iter().enumerate() {
                    let q = f64::from(if h == 0 { byte & 0x0F } else { byte >> 4 });
                    let xi = f64::from(xb[64 * c + 32 * h + l]);
                    s += (d * f64::from(sc) * q * xi).abs() + (dmin * f64::from(m) * xi).abs();
                }
            }
        }
    }
    2.0 * gamma(256 * (b.len() / 144) + 2) * s
}

/// q6k-f32: K = 256n + 2, and S sums |d*sc*(q - 32)*x_i|. Layout as fused_q6k_dot
/// (fused_q5k_q6k.rs:15): value 16g + l of 128-value half h takes scale 8h + g.
fn tol_q6k_f32(b: &[u8], x: &[f32]) -> f64 {
    let mut s = 0.0f64;
    for (blk, xb) in b.chunks(210).zip(x.chunks(256)) {
        let d = f64::from(read_f16(&blk[208..210]));
        for h in 0..2 {
            let (ql, qh) = (&blk[64 * h..64 * h + 64], &blk[128 + 32 * h..160 + 32 * h]);
            for l in 0..32 {
                let q = [
                    (ql[l] & 0xF) | ((qh[l] & 3) << 4),
                    (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4),
                    (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4),
                    (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4),
                ];
                for (p, &qp) in q.iter().enumerate() {
                    let i = 32 * p + l; // index within the half
                    let sc = f64::from(i8::from_le_bytes([blk[192 + 8 * h + i / 16]]));
                    s += (d * sc * (f64::from(qp) - 32.0) * f64::from(xb[128 * h + i])).abs();
                }
            }
        }
    }
    2.0 * gamma(256 * (b.len() / 210) + 2) * s
}

/// q4k-q8k: K = 16n + 2, and S sums, per sub-block j, |q8s*d*sc_j*S_j| + |q8s*dmin*m_j*Q_j|
/// over the exact integers S_j = sum q*q8 and Q_j = sum q8. The kernel must form them
/// exactly before any f32 rounding, as the oracle does (q4k_dot_avx2.rs:232).
fn tol_q4k_q8k(b: &[u8], d8: &[f32], q8: &[i8]) -> f64 {
    let mut s = 0.0f64;
    for ((blk, &q8s), qb) in b.chunks(144).zip(d8).zip(q8.chunks(256)) {
        let d = f64::from(read_f16(&blk[0..2]));
        let dmin = f64::from(read_f16(&blk[2..4]));
        let scales: &[u8; 12] = blk[4..16].try_into().expect("12 scale bytes");
        let q8s = f64::from(q8s);
        for (c, qs) in blk[16..144].chunks(32).enumerate() {
            for h in 0..2 {
                let (sc, m) = extract_scale_min(scales, 2 * c + h);
                let (mut sj, mut qj) = (0i32, 0i32);
                for (l, &byte) in qs.iter().enumerate() {
                    let v = i32::from(qb[64 * c + 32 * h + l]);
                    sj += i32::from(if h == 0 { byte & 0x0F } else { byte >> 4 }) * v;
                    qj += v;
                }
                s += (q8s * d * f64::from(sc) * f64::from(sj)).abs()
                    + (q8s * dmin * f64::from(m) * f64::from(qj)).abs();
            }
        }
    }
    2.0 * gamma(16 * (b.len() / 144) + 2) * s
}

fn draw<T: std::fmt::Debug, S: Strategy<Value = T>>(r: &mut TestRunner, s: S) -> T {
    s.new_tree(r).expect("tree").current()
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
    (
        f16_le(1e-3, 0.5),
        f16_le(1e-3, 0.5),
        prop::collection::vec(any::<u8>(), 140),
    )
        .prop_map(|(d, dmin, rest)| [d.as_slice(), dmin.as_slice(), rest.as_slice()].concat())
}

fn q6k_one() -> impl Strategy<Value = Vec<u8>> {
    (prop::collection::vec(any::<u8>(), 208), f16_le(1e-3, 0.5))
        .prop_map(|(body, d)| [body.as_slice(), d.as_slice()].concat()) // d = last 2 bytes
}

/// Rows of n in {1, 2, 3, 7, 8, 36} super-blocks: the block loop and the accumulator
/// reset run (L25), and 8 and 36 are the production row lengths (item d).
fn rows<S: Strategy<Value = Vec<u8>>>(one: fn() -> S) -> impl Strategy<Value = (usize, Vec<u8>)> {
    prop::sample::select(vec![1usize, 2, 3, 7, 8, 36]).prop_flat_map(move |n| {
        (
            Just(n),
            prop::collection::vec(one(), n).prop_map(|v| v.concat()),
        )
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
        let t = tol_q4k_f32(&b, &x);
        prop_assert!(within(n, s, t), "neon {n} vs scalar {s}, bound {t}");
    }

    #[test]
    fn falsify_neon_q4k_001_q6k_f32((b, x) in rows(q6k_one).prop_flat_map(|(n, b)| (Just(b), acts(n)))) {
        require_neon_path("q6k-f32")?;
        let s = fused_q6k_dot(&b, &x).expect("scalar oracle");
        let n = fused_q6k_dot_simd(&b, &x).expect("neon");
        let t = tol_q6k_f32(&b, &x);
        prop_assert!(within(n, s, t), "neon {n} vs scalar {s}, bound {t}");
    }

    #[test]
    fn falsify_neon_q4k_001_q4k_q8k((b, q, d) in rows(q4k_one).prop_flat_map(|(n, b)| (
                                        Just(b),
                                        prop::collection::vec(-127i8..=127, n * 256), // Q8_K range
                                        prop::collection::vec(1e-4f32..1.0, n)))) {
        require_neon_path("q4k-q8k")?;
        let s = fused_q4k_q8k_dot(&b, &d, &q).expect("scalar oracle");
        let n = fused_q4k_q8k_dot_simd(&b, &d, &q).expect("neon");
        let t = tol_q4k_q8k(&b, &d, &q);
        prop_assert!(within(n, s, t), "neon {n} vs scalar {s}, bound {t}");
    }
}
```

**Mutation (from the contract):** swap the low and high nibble order in the q4k NEON
kernel; `falsify_neon_q4k_001_q4k_f32` must turn RED. A second mutation for the
scale gap in §1.2: shift the scale index by one sub-block. The uniform-scale tests
stay green under it, and this one must turn RED. A third mutation (contract,
2026-10-03): drop the `dmin * min` term. It bites only because `dmin` is now non-zero by
construction. A fourth (item d): skip one 32-value group (one 16-value group for Q6_K) in
rows of 8 or more blocks. The full-row cases at 36 blocks stay GREEN in most draws (85%
simulated), and `falsify_neon_q4k_001_masked_sweep` must turn RED on that group.

**Masked sweep and power guard (contract precondition, item d).** At 36 blocks the bound is
about 1.1e-3 of S, and one 32-value group carries about 1/288 of S. So on full random rows a
NEON arm that skips one group stays GREEN 85% of the time (simulated). The sweep gives each
scale group a case of its own. x is zero outside the group and takes the signs of the
dequantized weights inside it, so the group's dot carries all of its S:

```rust
/// Rows whose scales are all non-zero: a group whose weights are all zero has no dot to
/// lose, and the power guard would refuse it. Q4_K, by the get_scale_min_k4 packing
/// (extract_scale_min, simd.rs:38): bit 0 of scales[0..4] and of scales[8..12] makes every
/// sc_j odd. Q6_K: a zero i8 scale becomes 1. Returns the row and its dequantized weights.
fn sweep_row(k: &str, n: usize, r: &mut TestRunner) -> (Vec<u8>, Vec<f32>) {
    if k == "q6k-f32" {
        let mut b: Vec<u8> = (0..n).flat_map(|_| draw(r, q6k_one())).collect();
        for blk in b.chunks_mut(210) {
            blk[192..208]
                .iter_mut()
                .filter(|v| **v == 0)
                .for_each(|v| *v = 1);
        }
        let w = dequantize_q6_k(&b).expect("dequant");
        (b, w)
    } else {
        let mut b: Vec<u8> = (0..n).flat_map(|_| draw(r, q4k_one())).collect();
        for blk in b.chunks_mut(144) {
            blk[4..8].iter_mut().for_each(|v| *v |= 1);
            blk[12..16].iter_mut().for_each(|v| *v |= 1);
        }
        let w = dequantize_q4_k(&b).expect("dequant");
        (b, w)
    }
}

/// One case per scale group (32 values for Q4_K and q4k-q8k, 16 for Q6_K) of every block,
/// at n in {1, 8, 36}. Asserts the power guard on every case and, if `compare`, the bound
/// between the SIMD dot and the scalar oracle. Returns the smallest |scalar| / bound.
fn masked_sweep(k: &str, compare: bool) -> f64 {
    let mut r = TestRunner::deterministic();
    let group = if k == "q6k-f32" { 16 } else { 32 };
    let mut min = f64::INFINITY;
    for n in [1usize, 8, 36] {
        let (b, w) = sweep_row(k, n, &mut r);
        let mags = draw(&mut r, prop::collection::vec(1i8..=127, n * 256));
        let d8 = draw(&mut r, prop::collection::vec(1e-4f32..1.0, n));
        for g in 0..n * 256 / group {
            let q8: Vec<i8> = (0..n * 256)
                .map(|i| match (i / group == g, w[i] < 0.0) {
                    (false, _) => 0,
                    (true, true) => -mags[i],
                    (true, false) => mags[i],
                })
                .collect();
            let x: Vec<f32> = q8.iter().map(|&v| f32::from(v)).collect(); // the same signs
            let (s, bound) = match k {
                "q4k-f32" => (fused_q4k_dot(&b, &x), tol_q4k_f32(&b, &x)),
                "q6k-f32" => (fused_q6k_dot(&b, &x), tol_q6k_f32(&b, &x)),
                _ => (fused_q4k_q8k_dot(&b, &d8, &q8), tol_q4k_q8k(&b, &d8, &q8)),
            };
            let s = s.expect("scalar oracle");
            assert!(
                bound > 0.0 && f64::from(s.abs()) >= 10.0 * bound,
                "power guard: {k} n {n} group {g}: |{s}| < 10 * {bound}"
            );
            min = min.min(f64::from(s.abs()) / bound);
            if compare {
                let v = match k {
                    "q4k-f32" => fused_q4k_dot_simd(&b, &x),
                    "q6k-f32" => fused_q6k_dot_simd(&b, &x),
                    _ => fused_q4k_q8k_dot_simd(&b, &d8, &q8),
                }
                .expect("neon");
                assert!(
                    within(v, s, bound),
                    "{k} n {n} group {g}: neon {v} vs scalar {s}, bound {bound}"
                );
            }
        }
    }
    min
}

/// Scalar and the bound only, so it also runs on x86 CI, where the parity cases are NotRun.
/// Under §3 option (a) it and the helpers it calls stay outside the aarch64-only cfg.
/// It prints the smallest margin per kernel.
#[test]
fn falsify_neon_q4k_001_sweep_power_guard() {
    for k in ["q4k-f32", "q6k-f32", "q4k-q8k"] {
        eprintln!(
            "{k}: min |scalar| / bound over the masked sweep = {:.0}",
            masked_sweep(k, false)
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn falsify_neon_q4k_001_masked_sweep() {
    for k in ["q4k-f32", "q6k-f32", "q4k-q8k"] {
        require_neon_path(k).expect("dispatch_honesty"); // §3
        masked_sweep(k, true);
    }
}
```
A skipped group misses by its whole dot. At n = 36 that is at least 184 bounds for Q4_K and
4253 for q4k-q8k (simulated), and exactly 1 / (2γ(K)), about 910, for Q6_K. The guard asks
for 10. On x86, `masked_sweep(k, true)` would check the same bound against the AVX2 arms, a
pre-gx10 check that the bound never fails a real SIMD order. That is a check of the bound,
never NEON evidence, and its q4k-q8k half first needs the AVX2 arm shown to form exact
integer sub-block sums (not read here).

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
are two options. (a) Put the parity cases behind `#[cfg(target_arch = "aarch64")]` (an
inner module), so x86 lists none of them. The §2 sweep power guard and the helpers it
calls stay outside: they are scalar only. The gate that consumes the result must then treat an absent
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
  value under the §2 bound (`within` and the kernel's `tol_*`, with the activations
  regenerated from `acts_seed`).
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
orphan="${q}/neon_q4k_006_orphan.rs"                               # control: no mod or include! names it
printf '%s\n' "${plant}" > "${orphan}"
if check; then echo "ok control: an unnamed file is not compiled"
else echo "FAIL control: the planted orphan compiled; the probe cannot tell dead from live"; rc=1; fi
rm -f "${orphan:?}"
if [ -e "${q}/fused_q4k.rs" ]; then                                # tripwire while S3 is open
  cp "${q}/fused_q4k.rs" "${q}/fused_q4k.rs.bak"
  printf '%s\n' "${plant}" >> "${q}/fused_q4k.rs"
  if check; then echo "ok fused_q4k.rs: still an orphan (S3 open)"
  else echo "FAIL fused_q4k.rs is compiled now; update the R3 cites"; rc=1; fi
  mv "${q}/fused_q4k.rs.bak" "${q}/fused_q4k.rs"
else echo "n/a fused_q4k.rs: deleted (S3)"; fi
exit "${rc}"
```
The control is a file the probe plants itself, so it survives the S3 delete
(`ticket-bodies-side-fixes.md`). It proves that `check` can pass: a `check` that always fails, such as
one on a tree whose aarch64 build is already broken, would otherwise report every live site as compiled.
The `fused_q4k.rs` tripwire runs only while that file exists. Until S3 lands, it catches anyone who
wires the orphan back in without updating the R3 cites.

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
        prop_assert!(within(w, s, tol_q4k_q8k(&b, &d, &q)), "widen {w} vs scalar {s}");
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
one on their own fixtures (1 to 36 blocks since item d), but not an Err on an input they never draw, such as the
planted block below. This test calls each row's dot directly at 8 and 36 blocks per row, then
checks that the matvec wrote that value to that row. It runs all five decode entries and the two prefill (multirow)
entries, F and G, for 3 tokens at once (contract `matvec_row_parity`; R3 §13 row 10).

```rust
// crates/aprender-serve/src/quantize/neon_parity_tests.rs  (lands with R3; adds to the §2 imports and uses §2's draw)
use super::{fused_q4k_multirow_matmul_f32_into, fused_q4k_parallel_matvec_f32_into,
            fused_q4k_parallel_matvec_into, fused_q4k_q8k_ffn_up_gate_into,
            fused_q6k_multirow_matmul_into, fused_q6k_parallel_matvec_into,
            quantize_activations_q8k_into, with_fp32_activations};
use crate::error::Result;

/// Contract matvec_row_parity: 8 and 36 super-blocks per row, and out_dim on both
/// sides of the sequential/parallel split (300 ends in a 44-row tail tile).
const SHAPES: [(usize, usize); 4] = [(2048, 48), (2048, 300), (9216, 48), (9216, 300)];
const PLANT_ROW: usize = 280; // inside the tail tile, rows 256..300
const M: usize = 3; // tokens for the prefill entries F and G; G takes its multirow route only for m > 1

/// rows x n blocks from the §2 generators, row-major.
fn matrix<S: Strategy<Value = Vec<u8>>>(r: &mut TestRunner, one: fn() -> S, rows: usize, n: usize) -> Vec<u8> {
    (0..rows * n).flat_map(|_| draw(r, one())).collect()
}

/// A row against its own dot. The matvec should reproduce that dot exactly, so this keeps
/// matvec_row_parity's relative 1e-3, which absorbs only a future reordering (a NEON bsums
/// variant). It is not a parity bound; that is §2's.
fn close_row(got: f32, dot: f32) -> bool {
    (got - dot).abs() < 1e-3 * dot.abs().max(1.0)
}

/// Calls each row's dot directly, so a NEON Err panics here instead of becoming 0.0.
/// Then checks that the matvec wrote that value to that row.
fn check_rows(entry: &str, got: &[f32], w: &[u8], row_bytes: usize, dot: impl Fn(&[u8]) -> Result<f32>) {
    assert_eq!(w.len(), got.len() * row_bytes, "{entry}: fixture shape");
    for (r, (g, row)) in got.iter().zip(w.chunks(row_bytes)).enumerate() {
        let v = dot(row).unwrap_or_else(|e| panic!("{entry}: dot Err on row {r}: {e}"));
        assert!(v.abs() >= 0.01, "{entry}: fixture row {r} has a near-zero dot {v}");
        assert!(close_row(*g, v), "{entry}: row {r} = {g}, its dot = {v}");
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
1. Make a NEON arm return Err for rows of 8 or more super-blocks. This test turns RED in its first shape (in_dim
   2048, 8 blocks), since every arm is reached there. Since item (d) the §2 tests, which draw 8 and 36 blocks, turn
   RED too.
2. Make a Q4_K NEON arm return Err for the planted block. The fixture plants it on every run (row 280, inside the
   44-row tail tile of the out_dim 300 cases), so the mutation needs no fixture edit. The test turns RED on row 280. The §2
   fixtures never draw that block, so only this test sees the mutation.
3. In the parallel branch of `fused_q4k_q8k_parallel_matvec_into` (q5k_q6k_matvec.rs:319), write 0.0 in place of one
   tail-tile row. Entry C turns RED on that row. This one tests the row check itself: `close_row(0.0, v)` fails whenever
   |v| >= 0.01, which the fixture asserts.
4. In `generic_multirow_matmul_into` (generic_matvec.rs:238-239), write 0.0 for the last row of the part tile for
   token m - 1. Entry F turns RED on that row. No decode entry reaches that loop.

**Why the reference is each row's own dot.** This test asks one thing: does each row hold its own dot? Value
parity against the scalar oracle is §2's, which since item (d) runs 8 and 36 blocks under the derived bound.
On aarch64, `fused_q4k_q8k_dot_with_bsums_simd` falls through to `fused_q4k_q8k_dot_simd`
(bsum_precompute.rs:237), so entries C and D reach that one dispatcher whether bsums is set or not. A NEON bsums
variant would change their reference.

**Why the prefill entries, with no C4 caller yet.** The only production caller of the multirow kernels,
`matmul_rows` (forward_qwen35.rs:1416), is reached only from `forward_prefill_qwen35` (:1200), and only its tests
call that (forward_qwen35_prefill_tests.rs). FALSIFY-4228-004 and -005 check the batched prefill bit-identical to
`forward_single_qwen35`, and a NEON Err would pass them: on aarch64 both sides call the same dot on the same row and
swallow the Err into the same 0.0. F and G close that before the batched prefill is wired into a run. G reads only
`DIRECT_FP32_GEMV`, never the fp32 scope (q4k_q8k_multirow.rs:178-179), so it runs outside the scope and its
reference is each token's own Q8_K dot. Running outside the scope stays right after ticket S2
(`ticket-bodies-side-fixes.md`): once the multirow follows the scope, G inside it would take the per-row f32
path and never reach the Q8_K multirow kernel it tests. `fused_q5k_multirow_matmul_into` is left out: Q5_K is not an R3 kernel.

## 5. What this needs before it can run

| Need | State |
|------|-------|
| `kernel_path(k)` API | R3 code (not written; mint deferred, cop 10:20Z 09-28). It is also emitted in the forward trace, because the R1 receipt checker reads it: FALSIFY-BPM-012 refuses a C4 receipt in which one quantized matmul tensor shares a `kernel_id` between the backend and reference runs |
| `kernel_path` shape | reuse `apr-kernel-path-v1` (OBS-15, unmerged #4574): `kernel_path(k)` becomes the entry's `kernel_id`, with `arch = aarch64`. On x86 it names the x86 arm (e.g. `q4k-f32/avx2`), because the C0 reference is read too. Each trace entry also carries `tensor`, the GGUF name, since BPM-012 compares per tensor, and a tensor gets one entry per kernel it reached (a crushed activation block switches one Q4_K call to f32). See P1 spec §3a |
| `fused_q4k_q8k_dot_neon_widen` test entry | R3 code (NEON-Q4K-007) |
| The seven matvec entries (five decode, two prefill), `with_fp32_activations` and `quantize_activations_q8k_into` in reach of a `quantize/` test (NEON-Q4K-008) | exist at 316dee2cd4: re-exported at quantize/mod.rs:132 and :140-146 (the multirow ones at :141 and :145); the quantizer is at mod.rs:241 |
| Orphan `quantize/fused_q4k.rs` deleted | Ticket S3 (`ticket-bodies-side-fixes.md`; PROPOSE-TICKET 07:49Z). NEON-Q4K-006 plants its own control file, so the delete does not weaken it; its `fused_q4k.rs` tripwire runs only while the file exists |
| Orphan `quantize/fused_q.rs` deleted | Ticket S3 (R3 §13 row 9; PROPOSE-TICKET 18:21Z). It is not a dot site, so NEON-Q4K-006 needs no second control |
| An aarch64 run (gx10) | GPU-deferred while a train is active; runs on CPU only, so it may be admissible earlier. Cop to rule |
| An aarch64 CI lane | none today; FALSIFY-NEON-Q4K-000 proposes a report-only `cargo check --target aarch64` step |
