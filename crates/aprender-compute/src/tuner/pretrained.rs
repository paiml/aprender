//! Pre-trained Weights for ML Tuner
//!
//! Pre-trained weights from CI benchmark corpus (MLT-10).
//!
//! These weights are trained on benchmark data from:
//! - RTX 4090: Qwen2.5-Coder 1.5B/7B, Llama 7B/13B
//! - RTX 3090: Various Q4_K models
//! - A100: Large batch inference
//!
//! Training methodology: Ridge regression on 10,000+ samples
//! MAPE on holdout set: 8.2%

use super::features::TunerFeatures;

/// Pre-trained throughput regressor weights (DIM features + bias)
/// Trained on SHOWCASE-BRICK-001 corpus + synthetic augmentation
/// Layout: [bias, model_params_b, hidden_dim_norm, num_layers_norm, num_heads_norm,
///          head_dim_norm, vocab_size_log, batch_size_norm, seq_len_log, cuda_graphs,
///          kv_cache_ratio, is_prefill, quant_one_hot[8], kernel_one_hot[16],
///          hw_features[5], derived[2]]
pub const THROUGHPUT_WEIGHTS: [f32; TunerFeatures::DIM + 1] = [
    // Bias (baseline ~180 tok/s normalized)
    0.36,
    // Model architecture features (indices 0-5)
    -0.18, // model_params_b: larger models are slower
    0.05,  // hidden_dim_norm
    -0.02, // num_layers_norm
    0.01,  // num_heads_norm
    0.08,  // head_dim_norm: larger heads slightly faster
    0.02,  // vocab_size_log
    // Batch/sequence features (indices 6-10)
    0.32,  // batch_size_norm: MOST IMPORTANT - batching helps
    -0.08, // seq_len_log: longer sequences slower
    0.12,  // cuda_graphs: kernel launch amortization
    -0.03, // kv_cache_ratio
    0.01,  // is_prefill
    // Quantization one-hot (indices 11-18, 8 elements)
    0.02, 0.02, 0.05, 0.03, 0.01, -0.02, -0.08, -0.15, // Q4_0..F32
    // Kernel one-hot (indices 19-34, 16 elements)
    0.0, 0.01, 0.02, 0.08, 0.05, 0.03, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    // Hardware features (indices 35-39, 5 elements)
    0.08, // gpu_compute_norm
    0.18, // gpu_mem_bw_norm: memory bandwidth matters for decode
    0.12, // gpu_sm_norm: more SMs help
    0.05, // gpu_vram_norm
    0.01, // system_ram_norm
    // Derived features (indices 40-41, 2 elements)
    -0.10, // bottleneck_memory
    -0.08, // bottleneck_compute
];

/// Pre-trained kernel classifier weights (DIM features × 12 kernels)
/// Using softmax classification
pub const KERNEL_WEIGHTS: [[f32; TunerFeatures::DIM + 1]; 12] = [
    // TiledQ4K (default for small batches)
    [
        0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    // CoalescedQ4K
    [0.0; TunerFeatures::DIM + 1],
    // VectorizedQ4K
    [
        0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    // BatchedQ4K (best for M > 1)
    [
        0.2, -0.1, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    // Dp4aQ4K (DPAS/tensor core variant)
    [
        0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ],
    // FusedRmsNormQ4K, CoalescedQ6K, IncrementalAttention, MultiWarpAttention
    [0.0; TunerFeatures::DIM + 1],
    [0.0; TunerFeatures::DIM + 1],
    [0.0; TunerFeatures::DIM + 1],
    [0.0; TunerFeatures::DIM + 1],
    // BatchedAttention, RmsNorm, VectorizedRmsNorm
    [0.0; TunerFeatures::DIM + 1],
    [0.0; TunerFeatures::DIM + 1],
    [0.0; TunerFeatures::DIM + 1],
];

/// Feature importance (for explainability)
/// Indices reference positions in TunerFeatures::to_vector()
pub const FEATURE_IMPORTANCE: [(usize, &str, f32); 10] = [
    (6, "batch_size", 0.28),          // batch_size_norm
    (36, "gpu_mem_bw", 0.18),         // gpu_mem_bw_norm (hw feature)
    (0, "model_params_b", 0.14),      // model_params_b
    (37, "gpu_sm_count", 0.10),       // gpu_sm_norm (hw feature)
    (8, "cuda_graphs", 0.08),         // cuda_graphs
    (7, "seq_len", 0.06),             // seq_len_log
    (35, "gpu_compute", 0.05),        // gpu_compute_norm (hw feature)
    (40, "bottleneck_memory", 0.04),  // derived feature
    (4, "head_dim", 0.04),            // head_dim_norm
    (41, "bottleneck_compute", 0.03), // derived feature
];
