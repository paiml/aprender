//! BLIS Microkernels - High-Performance SIMD Compute Kernels
//!
//! This module contains the microkernel implementations for different architectures:
//! - Scalar reference (correctness validation)
//! - AVX2 intrinsics
//! - AVX2 hand-tuned ASM with software pipelining
//! - ARM NEON
//!
//! # Performance Targets
//!
//! - 70%+ FMA utilization on Haswell+ CPUs
//! - 4-way K unrolling for software pipelining
//! - 10-12 instruction latency hiding
//!
//! # References
//!
//! - Goto, K., & Van de Geijn, R. A. (2008). Anatomy of High-Performance Matrix Multiplication.
//! - Agner Fog (2024). Optimizing subroutines in assembly language, Section 12.7.
//! - Intel(R) 64 and IA-32 Architectures Optimization Reference Manual.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "x86_64")]
pub mod codegen;
mod neon;

// Re-export all public microkernel functions
#[cfg(target_arch = "x86_64")]
pub use avx2::{
    microkernel_8x6_avx2, microkernel_8x6_avx2_asm, microkernel_8x6_true_asm,
    microkernel_8x8_avx2_fma,
};
#[cfg(target_arch = "x86_64")]
pub use avx512::{microkernel_16x8_avx512, microkernel_32x6_avx512};
#[cfg(target_arch = "aarch64")]
pub use neon::microkernel_8x8_neon;

use super::{MR, NR};

/// Scalar microkernel for correctness validation
///
/// Computes C[MR x NR] += A[MR x K] * B[K x NR]
/// where A is packed column-major and B is packed row-major.
///
/// This serves as the reference for validating SIMD microkernels.
#[inline(never)]
pub fn microkernel_scalar(
    k: usize,
    a: &[f32],     // MR x K, column-major (MR stride)
    b: &[f32],     // K x NR, row-major (NR stride)
    c: &mut [f32], // MR x NR, column-major
    ldc: usize,    // Leading dimension of C
) {
    // Accumulate MR x NR output tile
    for p in 0..k {
        for jr in 0..NR {
            let b_val = b[p * NR + jr];
            for ir in 0..MR {
                let a_val = a[p * MR + ir];
                c[jr * ldc + ir] += a_val * b_val;
            }
        }
    }
}
