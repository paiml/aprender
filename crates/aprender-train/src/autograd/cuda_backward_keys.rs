//! Backward-cache GEMM keys and kernel dims, as one source both the pre-warm and
//! the runtime read (R15a C6, contract `lora-target-selection-v1` equation
//! `lora_backward_prewarm`, FALSIFY-LORA_TARGET_SELECTION_V1_012).
//!
//! The backward GEMM kernels bake their dims into the PTX, and the runtime takes a
//! cached module before it compiles one. So a pre-warm that builds a kernel with
//! other dims than the runtime would, under the runtime's key, is not a cache miss:
//! it is a wrong kernel that runs. At 96e2f7b2a1 the pre-warm built
//! `tiled_unrolled(m, k, n)` under the key the runtime builds `tiled_unrolled(m, n, k)`
//! for, so every pre-warmed shape with `k != n` ran with `n` and `k` swapped on the
//! PTX path. Here the key and the dims come out of one function.
//!
//! Like [`super::cuda_forward_keys`], this module is NOT behind the `cuda` feature,
//! so its tests run without a GPU.

/// A backward GEMM as the kernel cache holds it: the key the runtime looks up and
/// the `(m, n, k)` its kernel is built with (`tiled_unrolled(m, n, k, tile)`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct BackwardGemm {
    /// `gemm_backward_{a,b}_{m}_{k}_{n}`.
    pub key: String,
    /// Rows of the GEMM.
    pub m: u32,
    /// The kernel's `n`.
    pub n: u32,
    /// The kernel's `k`.
    pub k: u32,
}

/// `gemm_backward_a(m, k, n)` as the runtime calls it (cuda_backward/gemm.rs).
#[must_use]
pub fn gemm_backward_a(m: u32, k: u32, n: u32) -> BackwardGemm {
    BackwardGemm { key: format!("gemm_backward_a_{m}_{k}_{n}"), m, n, k }
}

/// `gemm_backward_b(m, k, n)` as the runtime calls it (cuda_backward/gemm.rs).
#[must_use]
pub fn gemm_backward_b(m: u32, k: u32, n: u32) -> BackwardGemm {
    BackwardGemm { key: format!("gemm_backward_b_{m}_{k}_{n}"), m, n, k }
}

/// Which of the two backward kernels a [`BackwardGemm`] key names.
#[must_use]
pub fn is_backward_a(gemm: &BackwardGemm) -> bool {
    gemm.key.starts_with("gemm_backward_a_")
}

/// The backward GEMMs the LoRA backward of the NF4 block asks for, for adapters of
/// dims `(d_out, d_in)` at sequence length `s` and rank `r`
/// (`lora_backward_weights` and `lora_backward_input` in transformer/cuda_block.rs),
/// sorted by key with no repeat.
#[must_use]
pub fn lora_backward_gemms(dims: &[(u32, u32)], s: u32, r: u32) -> Vec<BackwardGemm> {
    let mut gemms: Vec<BackwardGemm> = dims
        .iter()
        .flat_map(|&(d_out, d_in)| {
            [
                gemm_backward_b(s, r, d_out),
                gemm_backward_a(s, d_out, r),
                gemm_backward_b(s, d_in, r),
                gemm_backward_a(s, r, d_in),
            ]
        })
        .collect();
    gemms.sort();
    gemms.dedup();
    gemms
}

#[cfg(test)]
mod tests;
