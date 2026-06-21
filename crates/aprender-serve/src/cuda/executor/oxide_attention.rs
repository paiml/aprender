//! PMAT-883: cuda-oxide pure-Rust incremental (KV-cache) attention — INTEGRATION
//! SCAFFOLD (default OFF, NOT wired into the live decode path).
//!
//! This module embeds the source-of-record PTX emitted by the cuda-oxide
//! `attn_warp_rawptr` kernel (see `experiments/cuda-oxide/incremental-attention/`,
//! PMAT-882/883) and provides the *promotion-candidate* launch wrapper. It is
//! gated behind the default-OFF `oxide-attention` feature so it compiles and is
//! testable WITHOUT altering production decode (which still uses the hand-PTX
//! `MultiWarpIncrementalAttentionKernel` via `incremental_attention_async`).
//!
//! # Status (PMAT-883)
//!
//! - PTX artifact: bit-parity-correct (cos=1.000000, maxdiff < 1e-5) on GB10
//!   sm_121 across all 9 decode configs (seq{128,1024,4096} x heads{8,16,32}),
//!   3-way verified (oxide-PTX == hand-PTX == CPU). See the 3-way gate in the
//!   experiment harness and `PMAT-883-STATUS.md`.
//! - Perf: 0.34-1.01x vs hand-PTX (faster at short/mid ctx, tied at long ctx).
//! - This wrapper is NOT registered in any dispatch. Flipping the live default is
//!   a SEPARATE, reviewed step (see "Promotion criteria" below).
//!
//! # CRITICAL: layout difference vs the live path
//!
//! The live `incremental_attention_async` uses a SEPARATE-HEAD K/V cache layout
//! `[num_kv_heads, max_len, head_dim]` (kv_stride = max_len*head_dim). The oxide
//! `attn_warp_rawptr` kernel uses the INTERLEAVED `[seq, kv_dim]` layout
//! (kv_dim = num_kv_heads*head_dim), which matches the CPU `causal_attention_cached`
//! reference but NOT the live GPU cache. Promotion therefore requires EITHER:
//!   (a) author an interleaved-layout GPU KV cache for the oxide path (preferred --
//!       it is the same layout the CPU reference + the oxide kernel already use), OR
//!   (b) add an oxide variant that consumes the existing separate-head layout
//!       (mechanical: change `krow` indexing to `kv_head*kv_stride + pos*head_dim`).
//! Until that is done + the on-device 3-way gate re-passes against the LIVE cache
//! layout, this wrapper is correct ONLY for interleaved inputs (as in the gate).
//!
//! # Promotion criteria (the ONLY way this becomes the live default)
//!
//! 1. Resolve the layout (a or b above) and re-pass the on-device 3-way gate vs
//!    the LIVE separate-head cache.
//! 2. End-to-end decode tok/s on a real GQA model >= the hand-PTX default on GB10.
//! 3. A `cfg(feature = "oxide-attention")` dispatch branch in
//!    `incremental_attention_async` selecting this wrapper, behind a runtime env
//!    guard (e.g. `APR_OXIDE_ATTENTION=1`), default OFF.
//! 4. CPU/GPU parity test passes with the feature ON (`gpu_cpu_trace_compare`).
//!
//! # ABI (raw-pointer C-style, matches the emitted PTX entry)
//!
//! ```text
//! attn_warp_rawptr(
//!     q: *const f32,        // [n_heads * head_dim]
//!     k: *const f32,        // [kv_len * kv_dim]   (interleaved [seq, kv_dim])
//!     v: *const f32,        // [kv_len * kv_dim]
//!     out: *mut f32,        // [n_heads * head_dim]
//!     kv_len: u32, head_dim: u32, n_heads: u32, n_kv_heads: u32, scale: f32)
//! Launch: grid = (n_heads, 1, 1), block = (32*NW = 1024, 1, 1).
//! ```

#![cfg(all(feature = "cuda", feature = "oxide-attention"))]

use crate::cuda::executor::CudaExecutor;
use trueno_gpu::driver::{CudaModule, GpuBuffer, LaunchConfig};
use trueno_gpu::GpuError;

/// Source-of-record PTX for the oxide `attn_warp_rawptr` kernel (sm_121, GB10).
/// Emitted by `experiments/cuda-oxide/incremental-attention/emit_ptx.sh`.
pub const OXIDE_ATTN_PTX: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../experiments/cuda-oxide/generated/attn_warp.sm121.ptx"
));

/// Entry-point name in the emitted PTX.
pub const OXIDE_ATTN_ENTRY: &str = "attn_warp_rawptr";

/// Warps-per-head baked into the kernel (NW=32 -> block = 32*32 = 1024 threads).
pub const OXIDE_ATTN_NW: u32 = 32;

/// Compile the embedded oxide attention PTX into a loadable CUDA module.
///
/// Mirrors how the live executor loads PTX (`CudaModule::from_ptx`, which applies
/// the GH-480 sm_121 backward-branch patch + disk cache). Self-contained PTX
/// (libdevice `expf` already inlined), so it carries no cuda-oxide build dep.
///
/// # Errors
/// Returns `GpuError::ModuleLoad` if the PTX fails to JIT.
pub fn compile_oxide_attention(exec: &CudaExecutor) -> Result<CudaModule, GpuError> {
    CudaModule::from_ptx(exec.context(), OXIDE_ATTN_PTX)
}

/// Promotion-candidate launch of the oxide incremental-attention kernel.
///
/// SCAFFOLD: expects INTERLEAVED `[seq, kv_dim]` K/V buffers (NOT the live
/// separate-head cache -- see module docs). All buffers must already be device
/// pointers; `out` is fully written (no pre-zero required).
///
/// # Errors
/// Returns `GpuError` if the launch fails.
#[allow(clippy::too_many_arguments)]
pub fn launch_oxide_attention(
    exec: &CudaExecutor,
    module: &mut CudaModule,
    q: &GpuBuffer<f32>,
    k_interleaved: &GpuBuffer<f32>,
    v_interleaved: &GpuBuffer<f32>,
    out: &GpuBuffer<f32>,
    kv_len: u32,
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    scale: f32,
) -> Result<(), GpuError> {
    let config = LaunchConfig::grid_2d(n_heads, 1, 32 * OXIDE_ATTN_NW, 1);

    let mut ptr_q = q.as_ptr();
    let mut ptr_k = k_interleaved.as_ptr();
    let mut ptr_v = v_interleaved.as_ptr();
    let mut ptr_out = out.as_ptr();
    let mut kv_len_v = kv_len;
    let mut head_dim_v = head_dim;
    let mut n_heads_v = n_heads;
    let mut n_kv_heads_v = n_kv_heads;
    let mut scale_v = scale;

    // SAFETY: device pointers + scalars laid out in the exact param order of the
    // emitted PTX entry; bounds enforced by the kernel's `head < n_heads` guard.
    unsafe {
        exec.compute_stream().launch_kernel(
            module,
            OXIDE_ATTN_ENTRY,
            &config,
            &mut [
                std::ptr::from_mut(&mut ptr_q) as *mut std::ffi::c_void,
                std::ptr::from_mut(&mut ptr_k) as *mut std::ffi::c_void,
                std::ptr::from_mut(&mut ptr_v) as *mut std::ffi::c_void,
                std::ptr::from_mut(&mut ptr_out) as *mut std::ffi::c_void,
                std::ptr::from_mut(&mut kv_len_v) as *mut std::ffi::c_void,
                std::ptr::from_mut(&mut head_dim_v) as *mut std::ffi::c_void,
                std::ptr::from_mut(&mut n_heads_v) as *mut std::ffi::c_void,
                std::ptr::from_mut(&mut n_kv_heads_v) as *mut std::ffi::c_void,
                std::ptr::from_mut(&mut scale_v) as *mut std::ffi::c_void,
            ],
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The embedded PTX is the self-contained source-of-record (sm_121, 1 entry).
    /// This test needs NO GPU -- it validates the committed artifact's shape so a
    /// bad re-emit is caught in CI on any host.
    #[test]
    fn embedded_ptx_is_self_contained_single_entry() {
        let body: String = OXIDE_ATTN_PTX
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(body.contains(".target sm_121"), "must target sm_121");
        assert_eq!(
            body.matches(".visible .entry").count(),
            1,
            "exactly one entry"
        );
        assert!(
            body.contains(OXIDE_ATTN_ENTRY),
            "entry name must be {OXIDE_ATTN_ENTRY}"
        );
        assert!(
            !body.contains("__nv_") && !body.contains(".extern .func"),
            "PTX must be self-contained (libdevice inlined, no extern calls)"
        );
    }
}
