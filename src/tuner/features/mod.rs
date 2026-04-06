#![allow(missing_docs)]
//! Feature Extraction for ML Tuning
//!
//! Implements TunerFeatures, TunerFeaturesBuilder, FeatureExtractor, and RunConfig.

mod builder;
mod extractor;

pub use builder::TunerFeaturesBuilder;
pub use extractor::{FeatureExtractor, RunConfig};

use serde::{Deserialize, Serialize};

use super::error::TunerError;
use super::types::BottleneckClass;

// ============================================================================
// TunerFeatures
// ============================================================================

/// Feature vector for ML-based kernel tuning.
///
/// All fields normalized to [0, 1] for model input.
/// Total dimension: 42 features.
///
/// # Feature Categories
///
/// - **Static (11)**: Known before execution (model size, batch size, etc.)
/// - **Quant one-hot (8)**: Quantization type encoding
/// - **Kernel one-hot (16)**: Kernel type encoding
/// - **Hardware (5)**: GPU capabilities
/// - **Derived (2)**: Computed features (arithmetic intensity, efficiency)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunerFeatures {
    // === Static features (11) ===
    /// Model size in billions (log10 normalized)
    pub model_params_b: f32,
    /// Hidden dimension / 16384
    pub hidden_dim_norm: f32,
    /// Number of layers / 128
    pub num_layers_norm: f32,
    /// Number of attention heads / 128
    pub num_heads_norm: f32,
    /// Head dimension / 256
    pub head_dim_norm: f32,
    /// Vocabulary size (log10 normalized)
    pub vocab_size_log: f32,
    /// Batch size M / 64
    pub batch_size_norm: f32,
    /// Sequence length (log2 / 15)
    pub seq_len_log: f32,
    /// CUDA graphs enabled (0 or 1)
    pub cuda_graphs: f32,
    /// Number of KV caches / batch_size (for multi-cache detection)
    pub kv_cache_ratio: f32,
    /// Prefill vs decode (0=decode, 1=prefill)
    pub is_prefill: f32,

    // === Quantization one-hot (8) ===
    pub quant_type_onehot: [f32; 8],

    // === Kernel one-hot (16) ===
    pub kernel_type_onehot: [f32; 16],

    // === Hardware features (5) === [v1.1.0: added L2 cache + zero-copy]
    /// Memory bandwidth / 3000 GB/s
    pub gpu_mem_bw_norm: f32,
    /// Compute TFLOPS / 500
    pub gpu_compute_norm: f32,
    /// SM count / 200
    pub gpu_sm_norm: f32,
    /// L2 cache size / 128 MB (v1.1.0: critical for occupancy)
    pub gpu_l2_cache_norm: f32,
    /// Zero-copy memory path enabled (0 or 1) (v1.1.0: pinned memory)
    pub is_zero_copy: f32,

    // === Derived features (2) ===
    /// Arithmetic intensity (FLOP/byte), normalized
    pub arithmetic_intensity: f32,
    /// Theoretical efficiency (measured / roofline)
    pub theoretical_efficiency: f32,

    // === Labels (for training) ===
    /// Measured throughput (tokens/second) - training label
    #[serde(skip_serializing_if = "Option::is_none")]
    pub measured_tps: Option<f32>,
    /// Best kernel ID - classification label
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_kernel_id: Option<u8>,
    /// Bottleneck class - classification label
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bottleneck_class: Option<BottleneckClass>,
}

impl Default for TunerFeatures {
    fn default() -> Self {
        Self {
            model_params_b: 0.0,
            hidden_dim_norm: 0.0,
            num_layers_norm: 0.0,
            num_heads_norm: 0.0,
            head_dim_norm: 0.0,
            vocab_size_log: 0.0,
            batch_size_norm: 0.0,
            seq_len_log: 0.0,
            cuda_graphs: 0.0,
            kv_cache_ratio: 1.0,
            is_prefill: 0.0,
            quant_type_onehot: [0.0; 8],
            kernel_type_onehot: [0.0; 16],
            gpu_mem_bw_norm: 0.0,
            gpu_compute_norm: 0.0,
            gpu_sm_norm: 0.0,
            gpu_l2_cache_norm: 0.0,
            is_zero_copy: 0.0,
            arithmetic_intensity: 0.0,
            theoretical_efficiency: 0.0,
            measured_tps: None,
            best_kernel_id: None,
            bottleneck_class: None,
        }
    }
}

impl TunerFeatures {
    /// Total feature dimension (excluding labels)
    /// v1.1.0: 11 static + 8 quant + 16 kernel + 5 hardware + 2 derived = 42
    pub const DIM: usize = 11 + 8 + 16 + 5 + 2; // 42 features (v1.1.0)

    /// Create a new feature builder
    pub fn builder() -> TunerFeaturesBuilder {
        TunerFeaturesBuilder::default()
    }

    /// ALB-099: Convert to fixed-size stack array — zero heap allocation.
    /// dhat profiling showed 140K Vec allocations from to_vector() in tests.
    pub fn to_array(&self) -> [f32; Self::DIM] {
        let mut a = [0.0f32; Self::DIM];
        let mut i = 0;

        // Static features (11)
        a[i] = self.model_params_b;
        i += 1;
        a[i] = self.hidden_dim_norm;
        i += 1;
        a[i] = self.num_layers_norm;
        i += 1;
        a[i] = self.num_heads_norm;
        i += 1;
        a[i] = self.head_dim_norm;
        i += 1;
        a[i] = self.vocab_size_log;
        i += 1;
        a[i] = self.batch_size_norm;
        i += 1;
        a[i] = self.seq_len_log;
        i += 1;
        a[i] = self.cuda_graphs;
        i += 1;
        a[i] = self.kv_cache_ratio;
        i += 1;
        a[i] = self.is_prefill;
        i += 1;

        // One-hot encodings (8 + 16)
        a[i..i + 8].copy_from_slice(&self.quant_type_onehot);
        i += 8;
        a[i..i + 16].copy_from_slice(&self.kernel_type_onehot);
        i += 16;

        // Hardware features (5)
        a[i] = self.gpu_mem_bw_norm;
        i += 1;
        a[i] = self.gpu_compute_norm;
        i += 1;
        a[i] = self.gpu_sm_norm;
        i += 1;
        a[i] = self.gpu_l2_cache_norm;
        i += 1;
        a[i] = self.is_zero_copy;
        i += 1;

        // Derived features (2)
        a[i] = self.arithmetic_intensity;
        i += 1;
        a[i] = self.theoretical_efficiency;

        a
    }

    /// Convert to flat vector for model input
    pub fn to_vector(&self) -> Vec<f32> {
        self.to_array().to_vec()
    }

    /// Validate features (F021-F030 falsification criteria)
    pub fn validate(&self) -> Result<(), TunerError> {
        let v = self.to_array();

        // F021: No NaN features
        if v.iter().any(|x| x.is_nan()) {
            return Err(TunerError::InvalidFeature("NaN value in features".into()));
        }

        // F022: No infinite features
        if v.iter().any(|x| x.is_infinite()) {
            return Err(TunerError::InvalidFeature("Infinite value in features".into()));
        }

        // F023: All features in [0, 1] (with small tolerance for floating point)
        if v.iter().any(|x| *x < -0.001 || *x > 1.001) {
            return Err(TunerError::InvalidFeature("Feature value outside [0, 1]".into()));
        }

        // F029: One-hot sums = 1
        let quant_sum: f32 = self.quant_type_onehot.iter().sum();
        if (quant_sum - 1.0).abs() > 0.001 && quant_sum > 0.001 {
            return Err(TunerError::InvalidFeature("Quant one-hot does not sum to 1".into()));
        }

        let kernel_sum: f32 = self.kernel_type_onehot.iter().sum();
        if (kernel_sum - 1.0).abs() > 0.001 && kernel_sum > 0.001 {
            return Err(TunerError::InvalidFeature("Kernel one-hot does not sum to 1".into()));
        }

        Ok(())
    }
}
