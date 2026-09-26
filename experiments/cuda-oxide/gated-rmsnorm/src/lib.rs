//! #3522 O-1 — `gated_rmsnorm` in pure Rust via cuda-oxide, beside the hand PTX
//! (`crates/aprender-gpu/src/kernels/gdn/gated_rmsnorm.rs`, entry `gdn_gated_rmsnorm`).
//!
//! Same math as the CPU reference `realizar::gguf::inference::forward::forward_qwen35::gated_rmsnorm`:
//!
//! ```text
//! for each head chunk of head_dim:
//!     rms_scale = 1 / sqrt(sum(input^2) / head_dim + eps)
//!     out[i]    = input[i] * rms_scale * weight[i] * silu(gate[i])
//! ```
//!
//! No `unsafe` and no unchecked indexing: every read is a bounds-checked slice index and the one write per
//! thread goes through `DisjointSlice::get_mut`, so the ontology extractor grades it `unsafeFree` and
//! `boundsChecked` from the source, not from a claim.

use cuda_device::cuda_module;

#[cuda_module]
pub mod kernels {
    use cuda_device::{kernel, thread, warp, DisjointSlice};

    /// Grid `(num_heads, 1, 1)`, block `(head_dim, 1, 1)`; `head_dim` a multiple of 32, at most 1024.
    /// Every warp of a head reduces the head's sum of squares itself (a shuffle-xor butterfly leaves the
    /// total in every lane), so no shared memory is needed; each thread then writes its one element.
    #[kernel]
    pub fn gated_rmsnorm(
        input: &[f32],
        gate: &[f32],
        weight: &[f32],
        mut out: DisjointSlice<f32>,
        head_dim: u32,
        eps: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let d = head_dim as usize;
        let base = thread::blockIdx_x() as usize * d;
        let mut sq = 0.0f32;
        let mut j = warp::lane_id() as usize;
        while j < d {
            let v = input[base + j];
            sq += v * v;
            j += 32;
        }
        sq += warp::shuffle_xor_f32_sync(0xFFFF_FFFF, sq, 16);
        sq += warp::shuffle_xor_f32_sync(0xFFFF_FFFF, sq, 8);
        sq += warp::shuffle_xor_f32_sync(0xFFFF_FFFF, sq, 4);
        sq += warp::shuffle_xor_f32_sync(0xFFFF_FFFF, sq, 2);
        sq += warp::shuffle_xor_f32_sync(0xFFFF_FFFF, sq, 1);
        let rms_scale = 1.0f32 / (sq / d as f32 + eps).sqrt();
        if let Some(o) = out.get_mut(idx) {
            let g = gate[i];
            let silu = g / (1.0f32 + (-g).exp());
            *o = input[i] * rms_scale * weight[i - base] * silu;
        }
    }
}
