//! 0.73 R3 parity tests: `fused_q{4,6}k_dot_simd` against the scalar oracles.
//!
//! Contract `contracts/neon-q4k-q6k-v1.yaml`. On aarch64 these exercise the NEON
//! kernels; on x86_64 the same tests exercise the AVX2 kernels, and the
//! `kernel_path` test asserts the path is NOT neon there.

use super::fused_k::{fused_q4k_dot, fused_q4k_dot_kernel_path, fused_q4k_dot_simd};
use super::fused_q5k_q6k::{fused_q6k_dot, fused_q6k_dot_kernel_path, fused_q6k_dot_simd};
use super::types::QK_K;

const BLOCKS: usize = 10_000;
const EPS: f32 = 1e-3;

/// Deterministic xorshift64*, so every run checks the same 10k blocks.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn byte(&mut self) -> u8 {
        (self.next_u64() >> 56) as u8
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn f16_bytes(v: f32) -> [u8; 2] {
    half::f16::from_f32(v).to_le_bytes()
}

fn q4k_block(rng: &mut Rng) -> Vec<u8> {
    let mut b = Vec::with_capacity(144);
    b.extend_from_slice(&f16_bytes(0.001 + rng.unit().abs() * 0.1));
    b.extend_from_slice(&f16_bytes(rng.unit().abs() * 0.05));
    b.extend((0..140).map(|_| rng.byte()));
    b
}

fn q6k_block(rng: &mut Rng) -> Vec<u8> {
    let mut b: Vec<u8> = (0..208).map(|_| rng.byte()).collect();
    b.extend_from_slice(&f16_bytes(0.001 + rng.unit().abs() * 0.01));
    b
}

fn activations(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.unit()).collect()
}

fn within(got: f32, want: f32) -> bool {
    (got - want).abs() < EPS * want.abs().max(1.0)
}

/// FALSIFY-NEON-Q4K-001: the dispatched kernel matches the scalar oracle on 10k seeded blocks.
#[test]
fn falsify_neon_q4k_001_parity_10k_blocks() {
    let mut rng = Rng(0x0073_0003_0001);
    for i in 0..BLOCKS {
        let w = q4k_block(&mut rng);
        let x = activations(&mut rng, QK_K);
        let want = fused_q4k_dot(&w, &x).expect("scalar");
        let got = fused_q4k_dot_simd(&w, &x).expect("simd");
        assert!(within(got, want), "block {i}: simd {got} vs scalar {want}");
    }
}

/// FALSIFY-NEON-Q4K-004: the Q6_K analogue of 001.
#[test]
fn falsify_neon_q4k_004_q6k_parity_10k_blocks() {
    let mut rng = Rng(0x0073_0003_0004);
    for i in 0..BLOCKS {
        let w = q6k_block(&mut rng);
        let x = activations(&mut rng, QK_K);
        let want = fused_q6k_dot(&w, &x).expect("scalar");
        let got = fused_q6k_dot_simd(&w, &x).expect("simd");
        assert!(within(got, want), "block {i}: simd {got} vs scalar {want}");
    }
}

/// Multi-super-block rows (the matvec shape), so the per-block offsets are covered.
#[test]
fn neon_q4k_q6k_parity_multi_block_rows() {
    let mut rng = Rng(0x0073_0003_0010);
    for n in [2usize, 7, 16] {
        let w4: Vec<u8> = (0..n).flat_map(|_| q4k_block(&mut rng)).collect();
        let w6: Vec<u8> = (0..n).flat_map(|_| q6k_block(&mut rng)).collect();
        let x = activations(&mut rng, n * QK_K);
        let (a, b) = (
            fused_q4k_dot(&w4, &x).expect("q4k"),
            fused_q4k_dot_simd(&w4, &x).expect("q4k simd"),
        );
        assert!(within(b, a), "q4k n={n}: {b} vs {a}");
        let (a, b) = (
            fused_q6k_dot(&w6, &x).expect("q6k"),
            fused_q6k_dot_simd(&w6, &x).expect("q6k simd"),
        );
        assert!(within(b, a), "q6k n={n}: {b} vs {a}");
    }
}

/// The parity check must be able to fail: a nibble-order swap (the planted mutant of
/// FALSIFY-NEON-Q4K-001) is caught on nearly every block.
#[test]
fn falsify_neon_q4k_001_planted_nibble_swap_is_caught() {
    let mut rng = Rng(0x0073_0003_0101);
    let caught = (0..200)
        .filter(|_| {
            let w = q4k_block(&mut rng);
            let x = activations(&mut rng, QK_K);
            let mut m = w.clone();
            m[16..].iter_mut().for_each(|b| *b = b.rotate_left(4));
            let want = fused_q4k_dot(&w, &x).expect("scalar");
            !within(fused_q4k_dot_simd(&m, &x).expect("simd"), want)
        })
        .count();
    assert!(
        caught >= 190,
        "nibble-swap mutant caught on only {caught}/200 blocks"
    );
}

/// FALSIFY-NEON-Q4K-004 planted mutant: dropping qh bit 5 is caught.
#[test]
fn falsify_neon_q4k_004_planted_qh_bit5_drop_is_caught() {
    let mut rng = Rng(0x0073_0003_0104);
    let caught = (0..200)
        .filter(|_| {
            let w = q6k_block(&mut rng);
            let x = activations(&mut rng, QK_K);
            let mut m = w.clone();
            m[128..192].iter_mut().for_each(|b| *b &= !0x20);
            let want = fused_q6k_dot(&w, &x).expect("scalar");
            !within(fused_q6k_dot_simd(&m, &x).expect("simd"), want)
        })
        .count();
    assert!(
        caught >= 150,
        "qh-bit5 mutant caught on only {caught}/200 blocks"
    );
}

/// Shape errors come back as errors from the dispatched kernel, as from the scalar one.
#[test]
fn neon_q4k_q6k_reject_bad_shapes() {
    assert!(fused_q4k_dot_simd(&[0u8; 143], &[0.0; QK_K]).is_err());
    assert!(fused_q4k_dot_simd(&[0u8; 144], &[0.0; QK_K - 1]).is_err());
    assert!(fused_q6k_dot_simd(&[0u8; 209], &[0.0; QK_K]).is_err());
    assert!(fused_q6k_dot_simd(&[0u8; 210], &[0.0; QK_K + 1]).is_err());
    assert_eq!(fused_q4k_dot_simd(&[], &[]).expect("empty q4k"), 0.0);
    assert_eq!(fused_q6k_dot_simd(&[], &[]).expect("empty q6k"), 0.0);
}

/// FALSIFY-NEON-Q4K-002: `kernel_path` names the kernel the dispatcher reaches.
#[test]
fn falsify_neon_q4k_002_kernel_path_is_honest() {
    let (p4, p6) = (fused_q4k_dot_kernel_path(), fused_q6k_dot_kernel_path());
    if cfg!(target_arch = "aarch64") {
        assert_eq!((p4, p6), ("q4k-f32/neon", "q6k-f32/neon"));
    } else {
        // Not aarch64: the NEON claim is not tested here, and must not be made.
        assert!(!p4.contains("neon") && !p6.contains("neon"), "{p4} {p6}");
    }
}
