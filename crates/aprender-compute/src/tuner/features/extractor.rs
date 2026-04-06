#![allow(missing_docs)]
//! Feature extraction and runtime configuration.
//!
//! Implements `FeatureExtractor` and `RunConfig`.

use crate::brick::{BrickCategory, BrickProfiler};
use crate::hardware::HardwareCapability;
use serde::{Deserialize, Serialize};

use crate::tuner::types::{BottleneckClass, KernelType, QuantType};

use super::TunerFeatures;

// ============================================================================
// FeatureExtractor
// ============================================================================

/// Extracts features from BrickProfiler and runtime configuration.
#[derive(Debug)]
pub struct FeatureExtractor {
    /// Hardware capability (cached)
    pub(crate) hardware: Option<HardwareCapability>,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureExtractor {
    /// Create a new feature extractor
    pub fn new() -> Self {
        Self { hardware: None }
    }

    /// Create with hardware capability
    pub fn with_hardware(hardware: HardwareCapability) -> Self {
        Self { hardware: Some(hardware) }
    }

    /// Extract features from profiler and configuration
    pub fn extract(&self, profiler: &BrickProfiler, config: &RunConfig) -> TunerFeatures {
        let mut builder = TunerFeatures::builder()
            .model_params_b(config.model_params_b)
            .hidden_dim(config.hidden_dim)
            .num_layers(config.num_layers)
            .num_heads(config.num_heads)
            .batch_size(config.batch_size)
            .seq_len(config.seq_len)
            .cuda_graphs(config.cuda_graphs)
            .quant_type(config.quant_type)
            .kernel_type(config.kernel_type);

        // Add hardware features if available
        if let Some(hw) = &self.hardware {
            builder = builder.hardware(hw);
        }

        // Add measured throughput if available
        if let Some(tps) = profiler.tokens_per_sec() {
            builder = builder.measured_tps(tps);
        }

        let mut features = builder.build();

        // Update derived features from profiler
        if let Some(efficiency) = self.calculate_efficiency(profiler, config) {
            features.theoretical_efficiency = efficiency;
        }

        // Classify bottleneck from profiler data
        features.bottleneck_class = Some(self.classify_bottleneck(profiler));

        features
    }

    /// Calculate efficiency from profiler data
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    // SAFETY: GPU bandwidth f64→f32 truncation is negligible for roofline efficiency calculation.
    pub fn calculate_efficiency(
        &self,
        profiler: &BrickProfiler,
        config: &RunConfig,
    ) -> Option<f32> {
        let measured_tps = profiler.tokens_per_sec()?;
        let hw = self.hardware.as_ref()?;
        let gpu = hw.gpu.as_ref()?;

        // Calculate theoretical max based on roofline
        let bytes_per_token = config.model_params_b * 1e9 * config.quant_type.bytes_per_param();
        let theoretical_tps = (gpu.memory_bw_gbps as f32) * 1e9 / bytes_per_token;

        Some((measured_tps / theoretical_tps).clamp(0.0, 1.0))
    }

    /// Classify bottleneck from profiler brick breakdown.
    ///
    /// PAR-200: Uses category_stats() for efficient aggregation.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    // SAFETY: percentage() returns f64 in 0–100; f64→f32 truncation is negligible for threshold comparisons.
    pub fn classify_bottleneck(&self, profiler: &BrickProfiler) -> BottleneckClass {
        let cats = profiler.category_stats();
        let total_ns = profiler.total_ns();

        if total_ns == 0 {
            return BottleneckClass::Unknown;
        }

        // Get category percentages
        let attention_pct =
            cats[BrickCategory::Attention as usize].percentage(total_ns) as f32 / 100.0;
        let ffn_pct = cats[BrickCategory::Ffn as usize].percentage(total_ns) as f32 / 100.0;
        let norm_pct = cats[BrickCategory::Norm as usize].percentage(total_ns) as f32 / 100.0;

        // Classify based on dominant component
        if attention_pct > 0.35 {
            BottleneckClass::AttentionBound
        } else if ffn_pct > 0.50 {
            // FFN is memory-bound (large GEMV operations)
            BottleneckClass::MemoryBound
        } else if norm_pct > 0.20 {
            // High norm percentage indicates launch overhead
            BottleneckClass::LaunchBound
        } else {
            BottleneckClass::MemoryBound // Default for inference
        }
    }
}

// ============================================================================
// RunConfig
// ============================================================================

/// Runtime configuration for feature extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    pub model_params_b: f32,
    pub hidden_dim: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub batch_size: u32,
    pub seq_len: u32,
    pub cuda_graphs: bool,
    pub quant_type: QuantType,
    pub kernel_type: KernelType,
}

/// Default hidden dimension for 1.5B parameter model
const DEFAULT_HIDDEN_DIM: u32 = 1536;

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            model_params_b: 1.5,
            hidden_dim: DEFAULT_HIDDEN_DIM,
            num_layers: 28,
            num_heads: 12,
            batch_size: 1,
            seq_len: 1,
            cuda_graphs: false,
            quant_type: QuantType::Q4K,
            kernel_type: KernelType::VectorizedQ4K,
        }
    }
}
