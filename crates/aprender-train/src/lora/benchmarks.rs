//! Memory benchmarks: LoRA vs QLoRA
//!
//! Compares memory usage between full-precision LoRA and 4-bit quantized QLoRA
//! across realistic transformer model dimensions.

use crate::lora::{LoRALayer, QLoRALayer};
use crate::Tensor;

/// Memory usage statistics for a single layer
#[derive(Debug, Clone)]
pub struct LayerMemoryStats {
    /// Layer name/identifier
    pub name: String,
    /// Base weight memory (unquantized, bytes)
    pub base_unquantized_bytes: usize,
    /// Base weight memory (quantized, bytes) - only for QLoRA
    pub base_quantized_bytes: Option<usize>,
    /// LoRA adapter memory (bytes)
    pub lora_adapters_bytes: usize,
    /// Total memory usage (bytes)
    pub total_bytes: usize,
    /// Compression ratio (only for QLoRA)
    pub compression_ratio: Option<f32>,
}

impl LayerMemoryStats {
    /// Create stats from LoRALayer
    pub fn from_lora(name: String, layer: &LoRALayer) -> Self {
        let base_bytes = layer.d_out() * layer.d_in() * 4; // f32
        let lora_a_bytes = layer.rank() * layer.d_in() * 4;
        let lora_b_bytes = layer.d_out() * layer.rank() * 4;
        let lora_bytes = lora_a_bytes + lora_b_bytes;

        Self {
            name,
            base_unquantized_bytes: base_bytes,
            base_quantized_bytes: None,
            lora_adapters_bytes: lora_bytes,
            total_bytes: base_bytes + lora_bytes,
            compression_ratio: None,
        }
    }

    /// Create stats from QLoRALayer
    pub fn from_qlora(name: String, layer: &QLoRALayer) -> Self {
        let stats = layer.memory_stats();

        Self {
            name,
            base_unquantized_bytes: stats.base_unquantized_bytes,
            base_quantized_bytes: Some(stats.base_quantized_bytes),
            lora_adapters_bytes: stats.lora_bytes,
            total_bytes: stats.total_bytes,
            compression_ratio: Some(stats.compression_ratio),
        }
    }

    /// Calculate memory savings percentage vs unquantized
    pub fn savings_percent(&self) -> f32 {
        let unquantized_total = self.base_unquantized_bytes + self.lora_adapters_bytes;
        ((unquantized_total - self.total_bytes) as f32 / unquantized_total as f32) * 100.0
    }
}

/// Benchmark results comparing LoRA and QLoRA
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Model configuration name
    pub model_name: String,
    /// LoRA layer statistics
    pub lora_stats: Vec<LayerMemoryStats>,
    /// QLoRA layer statistics
    pub qlora_stats: Vec<LayerMemoryStats>,
    /// Total LoRA memory (bytes)
    pub total_lora_bytes: usize,
    /// Total QLoRA memory (bytes)
    pub total_qlora_bytes: usize,
    /// Overall memory savings (bytes)
    pub savings_bytes: usize,
    /// Overall savings percentage
    pub savings_percent: f32,
}

impl BenchmarkResults {
    /// Create benchmark results
    pub fn new(
        model_name: String,
        lora_stats: Vec<LayerMemoryStats>,
        qlora_stats: Vec<LayerMemoryStats>,
    ) -> Self {
        let total_lora_bytes: usize = lora_stats.iter().map(|s| s.total_bytes).sum();
        let total_qlora_bytes: usize = qlora_stats.iter().map(|s| s.total_bytes).sum();
        let savings_bytes = total_lora_bytes.saturating_sub(total_qlora_bytes);
        let savings_percent = (savings_bytes as f32 / total_lora_bytes as f32) * 100.0;

        Self {
            model_name,
            lora_stats,
            qlora_stats,
            total_lora_bytes,
            total_qlora_bytes,
            savings_bytes,
            savings_percent,
        }
    }

    /// Generate human-readable report
    pub fn report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!("\n{}\n", "=".repeat(70)));
        report.push_str(&format!("Memory Benchmark: {}\n", self.model_name));
        report.push_str(&format!("{}\n\n", "=".repeat(70)));

        // Per-layer breakdown
        report.push_str("Layer-by-Layer Comparison:\n");
        report.push_str(&format!("{:-<70}\n", ""));

        for (lora, qlora) in self.lora_stats.iter().zip(self.qlora_stats.iter()) {
            report.push_str(&format!("\n{}:\n", lora.name));
            report.push_str(&format!("  LoRA:  {:>8} KB total\n", lora.total_bytes / 1024));
            report.push_str(&format!(
                "  QLoRA: {:>8} KB total ({:.1}% savings)\n",
                qlora.total_bytes / 1024,
                qlora.savings_percent()
            ));

            if let Some(ratio) = qlora.compression_ratio {
                report.push_str(&format!("  Base weight compression: {ratio:.1}x\n"));
            }
        }

        // Overall summary
        report.push_str(&format!("\n{:-<70}\n", ""));
        report.push_str(&format!("Total LoRA memory:  {:>8} KB\n", self.total_lora_bytes / 1024));
        report.push_str(&format!("Total QLoRA memory: {:>8} KB\n", self.total_qlora_bytes / 1024));
        report.push_str(&format!(
            "Memory savings:     {:>8} KB ({:.1}%)\n",
            self.savings_bytes / 1024,
            self.savings_percent
        ));
        report.push_str(&format!("{}\n", "=".repeat(70)));

        report
    }
}

/// Benchmark a single layer, returning (LoRA stats, QLoRA stats).
fn benchmark_single_layer(
    name: &str,
    d_out: usize,
    d_in: usize,
    rank: usize,
    alpha: f32,
) -> (LayerMemoryStats, LayerMemoryStats) {
    let base_weight = Tensor::from_vec(vec![1.0; d_out * d_in], false);

    let lora = LoRALayer::new(base_weight.clone(), d_out, d_in, rank, alpha);
    let lora_stats = LayerMemoryStats::from_lora(name.to_string(), &lora);

    let qlora = QLoRALayer::new(base_weight, d_out, d_in, rank, alpha);
    let qlora_stats = LayerMemoryStats::from_qlora(name.to_string(), &qlora);

    (lora_stats, qlora_stats)
}

/// Run memory benchmark for a specific model configuration
///
/// # Arguments
/// * `model_name` - Name of the model configuration
/// * `layers` - Layer configurations [(name, d_out, d_in, rank, alpha)]
pub fn benchmark_model(
    model_name: &str,
    layers: &[(&str, usize, usize, usize, f32)],
) -> BenchmarkResults {
    let mut lora_stats = Vec::new();
    let mut qlora_stats = Vec::new();

    for (name, d_out, d_in, rank, alpha) in layers {
        let (lora, qlora) = benchmark_single_layer(name, *d_out, *d_in, *rank, *alpha);
        lora_stats.push(lora);
        qlora_stats.push(qlora);
    }

    BenchmarkResults::new(model_name.to_string(), lora_stats, qlora_stats)
}

/// Benchmark suite for common transformer sizes
pub fn run_transformer_benchmarks() -> Vec<BenchmarkResults> {
    vec![
        // Small model (e.g., BERT-small, DistilBERT)
        benchmark_model(
            "Small Transformer (256-dim, 6 layers)",
            &[
                ("layer_0_qkvo", 256, 256, 8, 16.0),
                ("layer_1_qkvo", 256, 256, 8, 16.0),
                ("layer_2_qkvo", 256, 256, 8, 16.0),
                ("layer_3_qkvo", 256, 256, 8, 16.0),
                ("layer_4_qkvo", 256, 256, 8, 16.0),
                ("layer_5_qkvo", 256, 256, 8, 16.0),
            ],
        ),
        // Medium model (e.g., BERT-base)
        benchmark_model(
            "Medium Transformer (768-dim, 12 layers)",
            &[
                ("layer_0_q", 768, 768, 16, 32.0),
                ("layer_0_k", 768, 768, 16, 32.0),
                ("layer_0_v", 768, 768, 16, 32.0),
                ("layer_0_o", 768, 768, 16, 32.0),
                ("layer_6_q", 768, 768, 16, 32.0),
                ("layer_6_k", 768, 768, 16, 32.0),
                ("layer_6_v", 768, 768, 16, 32.0),
                ("layer_6_o", 768, 768, 16, 32.0),
                ("layer_11_q", 768, 768, 16, 32.0),
                ("layer_11_k", 768, 768, 16, 32.0),
                ("layer_11_v", 768, 768, 16, 32.0),
                ("layer_11_o", 768, 768, 16, 32.0),
            ],
        ),
        // Large model (e.g., GPT-2, LLaMA-7B)
        benchmark_model(
            "Large Transformer (4096-dim, 2 layers)",
            &[
                ("layer_0_q", 4096, 4096, 64, 128.0),
                ("layer_0_k", 4096, 4096, 64, 128.0),
                ("layer_0_v", 4096, 4096, 64, 128.0),
                ("layer_0_o", 4096, 4096, 64, 128.0),
                ("layer_1_q", 4096, 4096, 64, 128.0),
                ("layer_1_k", 4096, 4096, 64, 128.0),
                ("layer_1_v", 4096, 4096, 64, 128.0),
                ("layer_1_o", 4096, 4096, 64, 128.0),
            ],
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // ========================================================================
    // PROPERTY TESTS - Memory calculation correctness
    // ========================================================================

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(100))]

        /// QLoRA should always use less memory than LoRA for large enough dimensions
        #[test]
        fn prop_qlora_always_less_memory(
            d in 16usize..128,
            rank in 1usize..8,
            alpha in 1.0f32..32.0,
        ) {
            let size = d * d;
            let base_weight = Tensor::from_vec(vec![1.0; size], false);

            let lora = LoRALayer::new(base_weight.clone(), d, d, rank, alpha);
            let qlora = QLoRALayer::new(base_weight, d, d, rank, alpha);

            let lora_stats = LayerMemoryStats::from_lora("test".to_string(), &lora);
            let qlora_stats = LayerMemoryStats::from_qlora("test".to_string(), &qlora);

            // QLoRA total should be less than LoRA for matrices >= 16x16
            prop_assert!(
                qlora_stats.total_bytes < lora_stats.total_bytes,
                "QLoRA {} should be < LoRA {}",
                qlora_stats.total_bytes,
                lora_stats.total_bytes
            );
        }

        /// Savings percentage should be positive for non-trivial sizes
        #[test]
        fn prop_savings_percent_positive(
            d in 16usize..64,
            rank in 1usize..4,
        ) {
            let size = d * d;
            let base_weight = Tensor::from_vec(vec![1.0; size], false);
            let qlora = QLoRALayer::new(base_weight, d, d, rank, 4.0);

            let stats = LayerMemoryStats::from_qlora("test".to_string(), &qlora);
            let savings = stats.savings_percent();

            // For d >= 16, should always have positive savings
            prop_assert!(
                savings > 0.0,
                "Savings {} should be positive for d={}",
                savings, d
            );
        }

        /// Benchmark results should be internally consistent
        #[test]
        fn prop_benchmark_results_consistent(
            d in 16usize..64,
            num_layers in 1usize..4,
            rank in 1usize..4,
        ) {
            let layers: Vec<(&str, usize, usize, usize, f32)> =
                (0..num_layers)
                    .map(|i| {
                        // Use static strings to satisfy lifetime
                        match i % 4 {
                            0 => ("q_proj", d, d, rank, rank as f32 * 2.0),
                            1 => ("k_proj", d, d, rank, rank as f32 * 2.0),
                            2 => ("v_proj", d, d, rank, rank as f32 * 2.0),
                            _ => ("o_proj", d, d, rank, rank as f32 * 2.0),
                        }
                    })
                    .collect();

            let results = benchmark_model("Test", &layers);

            // Total should equal sum of individual
            let sum_lora: usize = results.lora_stats.iter().map(|s| s.total_bytes).sum();
            let sum_qlora: usize = results.qlora_stats.iter().map(|s| s.total_bytes).sum();

            prop_assert_eq!(results.total_lora_bytes, sum_lora);
            prop_assert_eq!(results.total_qlora_bytes, sum_qlora);

            // Savings should be non-negative
            prop_assert!(results.savings_percent >= 0.0);
        }
    }

    // ========================================================================
    // UNIT TESTS
    // ========================================================================

    #[test]
    fn test_layer_memory_stats_lora() {
        let base_weight = Tensor::from_vec(vec![1.0; 256 * 256], false);
        let lora = LoRALayer::new(base_weight, 256, 256, 16, 32.0);

        let stats = LayerMemoryStats::from_lora("test_layer".to_string(), &lora);

        assert_eq!(stats.name, "test_layer");
        assert_eq!(stats.base_unquantized_bytes, 256 * 256 * 4); // 256KB
        assert!(stats.base_quantized_bytes.is_none());
        assert_eq!(stats.lora_adapters_bytes, (16 * 256 + 256 * 16) * 4); // 32KB
        assert!(stats.compression_ratio.is_none());
    }

    #[test]
    fn test_layer_memory_stats_qlora() {
        let base_weight = Tensor::from_vec(vec![1.0; 256 * 256], false);
        let qlora = QLoRALayer::new(base_weight, 256, 256, 16, 32.0);

        let stats = LayerMemoryStats::from_qlora("test_layer".to_string(), &qlora);

        assert_eq!(stats.name, "test_layer");
        assert_eq!(stats.base_unquantized_bytes, 256 * 256 * 4);
        assert!(stats.base_quantized_bytes.is_some());
        assert!(
            stats.base_quantized_bytes.expect("operation should succeed")
                < stats.base_unquantized_bytes
        );
        assert!(stats.compression_ratio.is_some());
        assert!(stats.compression_ratio.expect("operation should succeed") > 6.0);
    }

    #[test]
    fn test_savings_percent() {
        let base_weight = Tensor::from_vec(vec![1.0; 256 * 256], false);
        let qlora = QLoRALayer::new(base_weight, 256, 256, 16, 32.0);

        let stats = LayerMemoryStats::from_qlora("test".to_string(), &qlora);
        let savings = stats.savings_percent();

        // Should save significant memory (>60%)
        assert!(savings > 60.0, "Expected >60% savings, got {savings:.1}%");
    }

    #[test]
    fn test_benchmark_model_small() {
        let results = benchmark_model(
            "Test Small Model",
            &[("layer_0", 256, 256, 8, 16.0), ("layer_1", 256, 256, 8, 16.0)],
        );

        assert_eq!(results.model_name, "Test Small Model");
        assert_eq!(results.lora_stats.len(), 2);
        assert_eq!(results.qlora_stats.len(), 2);
        assert!(results.total_qlora_bytes < results.total_lora_bytes);
        assert!(results.savings_percent > 50.0);
    }

    #[test]
    fn test_benchmark_report_format() {
        let results = benchmark_model("Test Model", &[("layer_0", 256, 256, 8, 16.0)]);

        let report = results.report();

        // Report should contain key sections
        assert!(report.contains("Memory Benchmark: Test Model"));
        assert!(report.contains("Layer-by-Layer Comparison"));
        assert!(report.contains("layer_0"));
        assert!(report.contains("LoRA:"));
        assert!(report.contains("QLoRA:"));
        assert!(report.contains("Memory savings:"));
    }

    #[test]
    fn test_transformer_benchmarks() {
        let benchmarks = run_transformer_benchmarks();

        assert_eq!(benchmarks.len(), 3); // Small, Medium, Large

        // All should show significant savings
        for benchmark in &benchmarks {
            assert!(
                benchmark.savings_percent > 50.0,
                "{} should save >50% memory, got {:.1}%",
                benchmark.model_name,
                benchmark.savings_percent
            );
        }

        // Verify model names
        assert!(benchmarks[0].model_name.contains("Small"));
        assert!(benchmarks[1].model_name.contains("Medium"));
        assert!(benchmarks[2].model_name.contains("Large"));
    }

    #[test]
    fn test_benchmark_results_calculations() {
        let lora_stats = vec![
            LayerMemoryStats {
                name: "layer_0".to_string(),
                base_unquantized_bytes: 1000,
                base_quantized_bytes: None,
                lora_adapters_bytes: 200,
                total_bytes: 1200,
                compression_ratio: None,
            },
            LayerMemoryStats {
                name: "layer_1".to_string(),
                base_unquantized_bytes: 1000,
                base_quantized_bytes: None,
                lora_adapters_bytes: 200,
                total_bytes: 1200,
                compression_ratio: None,
            },
        ];

        let qlora_stats = vec![
            LayerMemoryStats {
                name: "layer_0".to_string(),
                base_unquantized_bytes: 1000,
                base_quantized_bytes: Some(150),
                lora_adapters_bytes: 200,
                total_bytes: 350,
                compression_ratio: Some(6.67),
            },
            LayerMemoryStats {
                name: "layer_1".to_string(),
                base_unquantized_bytes: 1000,
                base_quantized_bytes: Some(150),
                lora_adapters_bytes: 200,
                total_bytes: 350,
                compression_ratio: Some(6.67),
            },
        ];

        let results = BenchmarkResults::new("Test".to_string(), lora_stats, qlora_stats);

        assert_eq!(results.total_lora_bytes, 2400);
        assert_eq!(results.total_qlora_bytes, 700);
        assert_eq!(results.savings_bytes, 1700);
        assert!((results.savings_percent - 70.83).abs() < 0.1);
    }

    #[test]
    fn test_large_model_memory_savings() {
        // Test realistic large model scenario
        let results = benchmark_model(
            "Large Model Test",
            &[("layer_0_q", 4096, 4096, 64, 128.0), ("layer_0_v", 4096, 4096, 64, 128.0)],
        );

        // 4096x4096 = 16M values = 64MB per layer unquantized
        // With 2 layers = 128MB base + adapters
        // QLoRA should reduce this significantly

        let lora_mb = results.total_lora_bytes as f32 / (1024.0 * 1024.0);
        let qlora_mb = results.total_qlora_bytes as f32 / (1024.0 * 1024.0);

        assert!(lora_mb > 100.0, "LoRA should use >100MB");
        assert!(qlora_mb < 50.0, "QLoRA should use <50MB");
        assert!(results.savings_percent > 60.0, "Should save >60% memory on large models");
    }
}
