//! #2880: per-ISA kernel reachability for the CPU quantized dot dispatchers.
//!
//! The numbers of a narrower kernel are correct, so no correctness test can see
//! a dispatcher that runs scalar on a SIMD host (#2604, #2605, #2567 were all
//! found by perf probes). Each `*_dot_simd` dispatcher therefore `match`es on
//! [`selected_dot_kernel`], so the path this module reports IS the path that
//! runs, and `reachability_gate` turns an undeclared scalar path into a RED test.
//!
//! The oracle is "not scalar where a SIMD kernel exists", not "the widest ISA":
//! on Zen 4 the AVX-512 VNNI 4-row Q4_K kernel measured 1.48–1.98× SLOWER than
//! the lean AVX2 one (#2880 slice 1), so width is not speed.

/// A quantized dot product the CPU decode path dispatches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DotOp {
    /// Q4_K weights × f32 activations (`fused_q4k_dot_simd`)
    Q4kF32,
    /// Q4_K weights × Q8_K activations (`fused_q4k_q8k_dot_simd`)
    Q4kQ8k,
    /// Q5_K weights × f32 activations (`fused_q5k_dot_simd`)
    Q5kF32,
    /// Q6_K weights × f32 activations (`fused_q6k_dot_simd`)
    Q6kF32,
}

impl DotOp {
    /// Every op, for the gate and for reports.
    pub const ALL: [DotOp; 4] = [DotOp::Q4kF32, DotOp::Q4kQ8k, DotOp::Q5kF32, DotOp::Q6kF32];
}

/// The kernel family a dispatcher runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DotKernel {
    /// Portable scalar loop
    Scalar,
    /// x86_64 AVX2 (+FMA where the kernel needs it)
    Avx2,
    /// x86_64 AVX-512F + AVX-512 VNNI
    Avx512Vnni,
    /// aarch64 baseline NEON
    Neon,
}

/// The kernel `op`'s dispatcher runs on this host. The dispatchers match on
/// this, so it cannot disagree with them.
#[must_use]
pub fn selected_dot_kernel(op: DotOp) -> DotKernel {
    #[cfg(target_arch = "x86_64")]
    {
        let avx2_fma = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        match op {
            DotOp::Q4kQ8k => {
                if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vnni") {
                    DotKernel::Avx512Vnni
                } else if is_x86_feature_detected!("avx2") {
                    DotKernel::Avx2
                } else {
                    DotKernel::Scalar
                }
            },
            DotOp::Q4kF32 | DotOp::Q6kF32 if avx2_fma => DotKernel::Avx2,
            _ => DotKernel::Scalar,
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        match op {
            DotOp::Q4kQ8k | DotOp::Q6kF32 => DotKernel::Neon,
            DotOp::Q4kF32 | DotOp::Q5kF32 => DotKernel::Scalar,
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = op;
        DotKernel::Scalar
    }
}

/// Does this host have a SIMD ISA the dot kernels target at all?
#[must_use]
pub fn host_has_simd() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

/// Scalar paths on a SIMD host that are KNOWN and owned: (op, target_arch, owner).
/// A gap not listed here fails the gate; a listed gap that has been closed
/// also fails it, so this list cannot go stale.
pub const KNOWN_SCALAR_GAPS: &[(DotOp, &str, &str)] = &[
    (DotOp::Q5kF32, "x86_64", "#2880 (no Q5_K SIMD kernel)"),
    (DotOp::Q5kF32, "aarch64", "#2880 (no Q5_K SIMD kernel)"),
    (DotOp::Q4kF32, "aarch64", "#2880 (no Q4_K×f32 NEON kernel)"),
];

#[cfg(test)]
mod tests {
    use super::*;

    fn declared_gap(op: DotOp) -> Option<&'static str> {
        KNOWN_SCALAR_GAPS
            .iter()
            .find(|(o, arch, _)| *o == op && *arch == std::env::consts::ARCH)
            .map(|(_, _, owner)| *owner)
    }

    #[test]
    fn reachability_gate() {
        let mut red = Vec::new();
        for op in DotOp::ALL {
            let k = selected_dot_kernel(op);
            println!("kernel-path {} {op:?} -> {k:?}", std::env::consts::ARCH);
            let gap = declared_gap(op);
            if host_has_simd() && k == DotKernel::Scalar && gap.is_none() {
                red.push(format!(
                    "{op:?} runs Scalar on a SIMD host and no owner is declared"
                ));
            }
            if let (true, Some(owner)) = (k != DotKernel::Scalar, gap) {
                red.push(format!(
                    "{op:?} gap ({owner}) is closed ({k:?}) — remove it"
                ));
            }
        }
        assert!(red.is_empty(), "kernel reachability: {red:#?}");
    }
}
