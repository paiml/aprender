#![allow(missing_docs)]
//! Builder for `TunerFeatures` with automatic normalization.

use crate::hardware::HardwareCapability;

use crate::tuner::types::{KernelType, QuantType};

use super::TunerFeatures;

// ============================================================================
// TunerFeaturesBuilder
// ============================================================================

/// Builder for TunerFeatures with automatic normalization.
#[derive(Default)]
pub struct TunerFeaturesBuilder {
    model_params_b: Option<f32>,
    hidden_dim: Option<u32>,
    num_layers: Option<u32>,
    num_heads: Option<u32>,
    head_dim: Option<u32>,
    vocab_size: Option<u32>,
    batch_size: Option<u32>,
    seq_len: Option<u32>,
    cuda_graphs: bool,
    kv_caches: Option<u32>,
    is_prefill: bool,
    quant_type: Option<QuantType>,
    kernel_type: Option<KernelType>,
    gpu_mem_bw_gbs: Option<f32>,
    gpu_compute_tflops: Option<f32>,
    gpu_sm_count: Option<u32>,
    gpu_l2_cache_mb: Option<f32>, // v1.1.0
    is_zero_copy: bool,           // v1.1.0
    measured_tps: Option<f32>,
}

impl TunerFeaturesBuilder {
    /// Set model size in billions of parameters
    pub fn model_params_b(mut self, params: f32) -> Self {
        self.model_params_b = Some(params);
        self
    }

    /// Set hidden dimension
    pub fn hidden_dim(mut self, dim: u32) -> Self {
        self.hidden_dim = Some(dim);
        self
    }

    /// Set number of layers
    pub fn num_layers(mut self, layers: u32) -> Self {
        self.num_layers = Some(layers);
        self
    }

    /// Set number of attention heads
    pub fn num_heads(mut self, heads: u32) -> Self {
        self.num_heads = Some(heads);
        self
    }

    /// Set head dimension
    pub fn head_dim(mut self, dim: u32) -> Self {
        self.head_dim = Some(dim);
        self
    }

    /// Set vocabulary size
    pub fn vocab_size(mut self, size: u32) -> Self {
        self.vocab_size = Some(size);
        self
    }

    /// Set batch size (M)
    pub fn batch_size(mut self, m: u32) -> Self {
        self.batch_size = Some(m);
        self
    }

    /// Set sequence length
    pub fn seq_len(mut self, len: u32) -> Self {
        self.seq_len = Some(len);
        self
    }

    /// Enable CUDA graphs
    pub fn cuda_graphs(mut self, enabled: bool) -> Self {
        self.cuda_graphs = enabled;
        self
    }

    /// Set number of KV caches
    pub fn kv_caches(mut self, count: u32) -> Self {
        self.kv_caches = Some(count);
        self
    }

    /// Set prefill mode
    pub fn is_prefill(mut self, prefill: bool) -> Self {
        self.is_prefill = prefill;
        self
    }

    /// Set quantization type
    pub fn quant_type(mut self, qt: QuantType) -> Self {
        self.quant_type = Some(qt);
        self
    }

    /// Set kernel type
    pub fn kernel_type(mut self, kt: KernelType) -> Self {
        self.kernel_type = Some(kt);
        self
    }

    /// Set GPU memory bandwidth in GB/s
    pub fn gpu_mem_bw_gbs(mut self, bw: f32) -> Self {
        self.gpu_mem_bw_gbs = Some(bw);
        self
    }

    /// Set GPU compute in TFLOPS
    pub fn gpu_compute_tflops(mut self, tflops: f32) -> Self {
        self.gpu_compute_tflops = Some(tflops);
        self
    }

    /// Set GPU SM count
    pub fn gpu_sm_count(mut self, count: u32) -> Self {
        self.gpu_sm_count = Some(count);
        self
    }

    /// Set measured throughput (for training data)
    pub fn measured_tps(mut self, tps: f32) -> Self {
        self.measured_tps = Some(tps);
        self
    }

    /// Set L2 cache size in MB (v1.1.0)
    pub fn gpu_l2_cache_mb(mut self, l2_mb: f32) -> Self {
        self.gpu_l2_cache_mb = Some(l2_mb);
        self
    }

    /// Set zero-copy memory path enabled (v1.1.0)
    pub fn is_zero_copy(mut self, enabled: bool) -> Self {
        self.is_zero_copy = enabled;
        self
    }

    /// Set hardware capability (auto-fills GPU features)
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    // SAFETY: GPU bandwidth/TFLOPS values fit in f32 (practical max ~10K); f64→f32 truncation negligible.
    pub fn hardware(mut self, hw: &HardwareCapability) -> Self {
        if let Some(gpu) = &hw.gpu {
            self.gpu_mem_bw_gbs = Some(gpu.memory_bw_gbps as f32);
            self.gpu_compute_tflops = Some(gpu.peak_tflops_fp32 as f32);
            // SM count not directly available; estimate from compute capability
            self.gpu_sm_count = None;
        }
        self
    }

    /// Build the feature vector with normalization
    #[allow(clippy::cast_precision_loss)]
    // SAFETY: All u32 values are model hyperparams (hidden_dim ≤ 16384, layers ≤ 128,
    //         heads ≤ 128, vocab ≤ 1M, batch ≤ 64, seq_len ≤ 32768) — well within f32 mantissa.
    /// Build features, panicking if raw inputs violate physical constraints.
    ///
    /// Use `try_build()` for fallible construction.
    pub fn build(self) -> TunerFeatures {
        self.try_build().expect("TunerFeatures: invalid raw input (see try_build for details)")
    }

    /// Build features, returning Err if raw inputs violate physical constraints.
    ///
    /// F024: model_params_b, gpu_mem_bw, gpu_compute must be non-negative.
    /// F024: Pre-normalization validation — physical quantities must be non-negative.
    fn validate_non_negative(&self) -> Result<(), crate::tuner::TunerError> {
        if let Some(p) = self.model_params_b {
            if p < 0.0 {
                return Err(crate::tuner::TunerError::InvalidFeature(format!(
                    "model_params_b must be non-negative, got {p}"
                )));
            }
        }
        if let Some(bw) = self.gpu_mem_bw_gbs {
            if bw < 0.0 {
                return Err(crate::tuner::TunerError::InvalidFeature(format!(
                    "gpu_mem_bw_gbs must be non-negative, got {bw}"
                )));
            }
        }
        if let Some(tf) = self.gpu_compute_tflops {
            if tf < 0.0 {
                return Err(crate::tuner::TunerError::InvalidFeature(format!(
                    "gpu_compute_tflops must be non-negative, got {tf}"
                )));
            }
        }
        Ok(())
    }

    pub fn try_build(self) -> Result<TunerFeatures, crate::tuner::TunerError> {
        self.validate_non_negative()?;

        let batch_size = self.batch_size.unwrap_or(1);
        let kv_caches = self.kv_caches.unwrap_or(batch_size);

        // Create one-hot encodings
        let mut quant_onehot = [0.0f32; 8];
        if let Some(qt) = self.quant_type {
            quant_onehot[qt.to_index()] = 1.0;
        }

        let mut kernel_onehot = [0.0f32; 16];
        if let Some(kt) = self.kernel_type {
            kernel_onehot[kt.to_index()] = 1.0;
        }

        // Calculate derived features
        // C-14 (Meyer DbC): 0 = unknown, no architecture-specific magic number.
        let hidden_dim = self.hidden_dim.unwrap_or(0) as f32;
        let batch_size_f = batch_size as f32;
        let quant_bytes = self.quant_type.map(|q| q.bytes_per_param()).unwrap_or(0.5625);

        // Arithmetic intensity for GEMV: 2*N*K FLOPs / (N*K*bytes + K + N) bytes
        // Simplified: ~2 / bytes_per_param for memory-bound inference
        let arithmetic_intensity = (2.0 / quant_bytes).min(10.0) / 10.0;

        // Theoretical efficiency starts at 0 (unknown until measured)
        let theoretical_efficiency = 0.0;

        Ok(TunerFeatures {
            // Normalized static features
            model_params_b: self
                .model_params_b
                .map(|p| (p.max(f32::EPSILON).log10() + 1.0) / 3.0) // log10(0.1)=-1, log10(100)=2 → [0, 1]
                .unwrap_or(0.0)
                .clamp(0.0, 1.0),
            hidden_dim_norm: (hidden_dim / 16384.0).clamp(0.0, 1.0),
            // N-07 (Meyer DbC): 0 when unknown — no model-specific defaults.
            num_layers_norm: (self.num_layers.unwrap_or(0) as f32 / 128.0).clamp(0.0, 1.0),
            num_heads_norm: (self.num_heads.unwrap_or(0) as f32 / 128.0).clamp(0.0, 1.0),
            head_dim_norm: (self.head_dim.unwrap_or(0) as f32 / 256.0).clamp(0.0, 1.0),
            vocab_size_log: self
                .vocab_size
                .map(|v| (v.max(1) as f32).log10() / 6.0) // log10(1M)=6
                .unwrap_or(0.0)
                .clamp(0.0, 1.0),
            batch_size_norm: (batch_size_f / 64.0).clamp(0.0, 1.0),
            seq_len_log: self
                .seq_len
                .map(|s| (s.max(1) as f32).log2() / 15.0) // log2(32K)≈15
                .unwrap_or(0.0)
                .clamp(0.0, 1.0),
            cuda_graphs: if self.cuda_graphs { 1.0 } else { 0.0 },
            kv_cache_ratio: (kv_caches as f32 / batch_size_f).clamp(0.0, 1.0),
            is_prefill: if self.is_prefill { 1.0 } else { 0.0 },

            // One-hot encodings
            quant_type_onehot: quant_onehot,
            kernel_type_onehot: kernel_onehot,

            // Hardware features (5) [v1.1.0]
            gpu_mem_bw_norm: (self.gpu_mem_bw_gbs.unwrap_or(1000.0) / 3000.0).clamp(0.0, 1.0),
            gpu_compute_norm: (self.gpu_compute_tflops.unwrap_or(100.0) / 500.0).clamp(0.0, 1.0),
            gpu_sm_norm: (self.gpu_sm_count.unwrap_or(128) as f32 / 200.0).clamp(0.0, 1.0),
            gpu_l2_cache_norm: (self.gpu_l2_cache_mb.unwrap_or(48.0) / 128.0).clamp(0.0, 1.0), // v1.1.0
            is_zero_copy: if self.is_zero_copy { 1.0 } else { 0.0 }, // v1.1.0

            // Derived features
            arithmetic_intensity,
            theoretical_efficiency,

            // Labels
            measured_tps: self.measured_tps,
            best_kernel_id: None,
            bottleneck_class: None,
        })
    }
}
