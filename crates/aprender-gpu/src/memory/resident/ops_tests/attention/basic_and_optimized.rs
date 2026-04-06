//! Basic batched multihead attention and optimized variant tests.

use crate::driver::{CudaContext, CudaStream};
use crate::memory::resident::{reset_transfer_counters, GpuResidentTensor};

/// Attention configuration for parametric tests.
struct AttnConfig {
    n_heads: u32,
    head_dim: u32,
    seq_len: u32,
}

impl AttnConfig {
    fn d_model(&self) -> usize {
        (self.n_heads * self.head_dim) as usize
    }
    fn total_len(&self) -> usize {
        self.seq_len as usize * self.d_model()
    }
}

/// Create QKV data and upload to GPU. Returns (q, k, v) tensors.
fn make_qkv(
    ctx: &CudaContext,
    cfg: &AttnConfig,
    gen: fn(usize) -> f32,
) -> (
    GpuResidentTensor<f32>,
    GpuResidentTensor<f32>,
    GpuResidentTensor<f32>,
) {
    let n = cfg.total_len();
    let q_data: Vec<f32> = (0..n).map(|i| gen(i)).collect();
    let k_data: Vec<f32> = (0..n).map(|i| gen(i) + 0.01).collect();
    let v_data: Vec<f32> = (0..n).map(|i| gen(i) + 0.02).collect();
    let q = GpuResidentTensor::from_host(ctx, &q_data).unwrap();
    let k = GpuResidentTensor::from_host(ctx, &k_data).unwrap();
    let v = GpuResidentTensor::from_host(ctx, &v_data).unwrap();
    (q, k, v)
}

/// Create QKV with custom sizes (for error tests where sizes differ).
fn make_qkv_custom(
    ctx: &CudaContext,
    q_len: usize,
    k_len: usize,
    v_len: usize,
) -> (
    GpuResidentTensor<f32>,
    GpuResidentTensor<f32>,
    GpuResidentTensor<f32>,
) {
    let q = GpuResidentTensor::from_host(ctx, &vec![1.0; q_len]).unwrap();
    let k = GpuResidentTensor::from_host(ctx, &vec![1.0; k_len]).unwrap();
    let v = GpuResidentTensor::from_host(ctx, &vec![1.0; v_len]).unwrap();
    (q, k, v)
}

#[test]
fn test_batched_multihead_attention_basic() {
    use crate::memory::resident::{batched_multihead_attention, clear_kernel_cache};
    clear_kernel_cache();
    let ctx = cuda_ctx!();
    reset_transfer_counters();

    let cfg = AttnConfig {
        n_heads: 2,
        head_dim: 4,
        seq_len: 3,
    };
    let (q, k, v) = make_qkv(&ctx, &cfg, |i| (i as f32) * 0.1);

    let output =
        batched_multihead_attention(&ctx, &q, &k, &v, cfg.n_heads, cfg.head_dim, cfg.seq_len)
            .unwrap();
    assert_eq!(output.len(), cfg.total_len());
    assert!(output.is_device_resident());
}

#[test]
fn test_batched_multihead_attention_dimension_error() {
    use crate::memory::resident::batched_multihead_attention;
    let ctx = cuda_ctx!();

    let cfg = AttnConfig {
        n_heads: 2,
        head_dim: 4,
        seq_len: 3,
    };
    let (q, k, v) = make_qkv_custom(&ctx, 12, cfg.total_len(), cfg.total_len());

    let result =
        batched_multihead_attention(&ctx, &q, &k, &v, cfg.n_heads, cfg.head_dim, cfg.seq_len);
    assert!(result.is_err());
}

#[test]
fn test_batched_multihead_attention_optimized() {
    use crate::memory::resident::{batched_multihead_attention_optimized, clear_kernel_cache};
    clear_kernel_cache();
    let ctx = cuda_ctx!();
    reset_transfer_counters();

    let cfg = AttnConfig {
        n_heads: 2,
        head_dim: 4,
        seq_len: 4,
    };
    let (q, k, v) = make_qkv(&ctx, &cfg, |i| (i as f32) * 0.1);

    let output = batched_multihead_attention_optimized(
        &ctx,
        &q,
        &k,
        &v,
        cfg.n_heads,
        cfg.head_dim,
        cfg.seq_len,
    )
    .unwrap();
    assert_eq!(output.len(), cfg.total_len());
}

#[test]
fn test_batched_multihead_attention_with_debug() {
    use crate::memory::resident::{batched_multihead_attention, clear_kernel_cache};
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    std::env::set_var("WHISPER_DEBUG_ATTN", "1");

    let cfg = AttnConfig {
        n_heads: 2,
        head_dim: 4,
        seq_len: 3,
    };
    let (q, k, v) = make_qkv(&ctx, &cfg, |i| (i as f32) * 0.1);

    let output =
        batched_multihead_attention(&ctx, &q, &k, &v, cfg.n_heads, cfg.head_dim, cfg.seq_len)
            .unwrap();
    assert_eq!(output.len(), cfg.total_len());

    std::env::remove_var("WHISPER_DEBUG_ATTN");
}

#[test]
fn test_batched_multihead_attention_k_v_mismatch() {
    use crate::memory::resident::batched_multihead_attention;
    let ctx = cuda_ctx!();

    let cfg = AttnConfig {
        n_heads: 2,
        head_dim: 4,
        seq_len: 3,
    };
    let (q, k, v) = make_qkv_custom(&ctx, cfg.total_len(), 12, cfg.total_len());

    let result =
        batched_multihead_attention(&ctx, &q, &k, &v, cfg.n_heads, cfg.head_dim, cfg.seq_len);
    assert!(result.is_err());
}

#[test]
fn test_batched_multihead_attention_optimized_size_error() {
    use crate::memory::resident::batched_multihead_attention_optimized;
    let ctx = cuda_ctx!();

    let cfg = AttnConfig {
        n_heads: 2,
        head_dim: 4,
        seq_len: 4,
    };
    let (q, k, v) = make_qkv_custom(&ctx, 10, cfg.total_len(), cfg.total_len());

    let result = batched_multihead_attention_optimized(
        &ctx,
        &q,
        &k,
        &v,
        cfg.n_heads,
        cfg.head_dim,
        cfg.seq_len,
    );
    assert!(result.is_err());
}

#[test]
fn test_batched_multihead_attention_larger_heads() {
    use crate::memory::resident::batched_multihead_attention;
    let ctx = cuda_ctx!();

    let cfg = AttnConfig {
        n_heads: 4,
        head_dim: 8,
        seq_len: 4,
    };
    let (q, k, v) = make_qkv(&ctx, &cfg, |i| ((i % 10) as f32) * 0.1);

    let output =
        batched_multihead_attention(&ctx, &q, &k, &v, cfg.n_heads, cfg.head_dim, cfg.seq_len)
            .unwrap();
    assert_eq!(output.len(), cfg.total_len());
}

#[test]
fn test_batched_multihead_attention_optimized_larger() {
    use crate::memory::resident::batched_multihead_attention_optimized;
    let ctx = cuda_ctx!();

    let cfg = AttnConfig {
        n_heads: 4,
        head_dim: 16,
        seq_len: 8,
    };
    let (q, k, v) = make_qkv(&ctx, &cfg, |i| ((i % 10) as f32) * 0.01);

    let output = batched_multihead_attention_optimized(
        &ctx,
        &q,
        &k,
        &v,
        cfg.n_heads,
        cfg.head_dim,
        cfg.seq_len,
    )
    .unwrap();
    assert_eq!(output.len(), cfg.total_len());
}
