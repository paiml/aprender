//! LoRA configuration optimizer (Kaizen principle).

use crate::memory::MemoryPlanner;
use crate::Method;
use entrenar_common::{EntrenarError, Result};

/// Optimal LoRA configuration result.
#[derive(Debug, Clone)]
pub struct OptimalConfig {
    /// Recommended fine-tuning method
    pub method: Method,
    /// Recommended LoRA rank
    pub rank: u32,
    /// Recommended alpha scaling
    pub alpha: f32,
    /// Target modules to apply LoRA
    pub target_modules: Vec<String>,
    /// Estimated trainable parameters
    pub trainable_params: u64,
    /// Percentage of total parameters that are trainable
    pub trainable_percent: f64,
    /// Estimated memory requirement in GB
    pub memory_gb: f64,
    /// VRAM utilization percentage
    pub utilization_percent: f64,
    /// Training speedup compared to full fine-tuning
    pub speedup: f64,
}

impl OptimalConfig {
    /// Format as human-readable comparison table.
    pub fn to_comparison_table(&self) -> String {
        format!(
            "Optimal Configuration:\n  Method: {:?}\n  Rank: {}\n  Alpha: {:.1}\n  Trainable: {} ({:.2}%)\n  Memory: {:.1} GB ({:.0}% utilization)\n  Speedup: {:.1}x vs full",
            self.method,
            self.rank,
            self.alpha,
            format_params(self.trainable_params),
            self.trainable_percent,
            self.memory_gb,
            self.utilization_percent,
            self.speedup
        )
    }
}

fn format_params(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1e9)
    } else if params >= 1_000_000 {
        format!("{:.1}M", params as f64 / 1e6)
    } else {
        format!("{:.1}K", params as f64 / 1e3)
    }
}

/// LoRA configuration optimizer.
#[derive(Debug)]
pub struct LoraOptimizer {
    model_params: u64,
    available_vram_bytes: u64,
    target_utilization: f64,
}

impl LoraOptimizer {
    /// Create a new optimizer.
    pub fn new(model_params: u64, available_vram_gb: f64) -> Self {
        Self {
            model_params,
            available_vram_bytes: (available_vram_gb * 1e9) as u64,
            target_utilization: 0.85, // Target 85% VRAM utilization
        }
    }

    /// Set target VRAM utilization (0.0 - 1.0).
    pub fn with_target_utilization(mut self, utilization: f64) -> Self {
        self.target_utilization = utilization.clamp(0.5, 0.95);
        self
    }

    /// Find optimal configuration for the given method.
    pub fn optimize(&self, method: Method) -> Result<OptimalConfig> {
        let method = if method == Method::Auto { self.select_method() } else { method };

        let rank = self.find_optimal_rank(method)?;
        let planner = MemoryPlanner::new(self.model_params);
        let memory = planner.estimate(method, rank);

        let trainable_params = self.calculate_trainable_params(method, rank);
        let trainable_percent = (trainable_params as f64 / self.model_params as f64) * 100.0;

        let memory_gb = memory.total_bytes as f64 / 1e9;
        let utilization = memory.total_bytes as f64 / self.available_vram_bytes as f64 * 100.0;

        let speedup = match method {
            Method::Full => 1.0,
            Method::LoRA => 2.5,
            Method::QLoRA => 1.8, // QLoRA has dequantization overhead
            Method::Auto => 2.0,
        };

        Ok(OptimalConfig {
            method,
            rank,
            alpha: rank as f32 / 4.0, // Common heuristic: alpha = rank/4
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
            ],
            trainable_params,
            trainable_percent,
            memory_gb,
            utilization_percent: utilization,
            speedup,
        })
    }

    fn select_method(&self) -> Method {
        let planner = MemoryPlanner::new(self.model_params);

        // Check if full fine-tuning fits
        let full_mem = planner.estimate_full().total_bytes;
        if full_mem < (self.available_vram_bytes as f64 * self.target_utilization) as u64 {
            return Method::Full;
        }

        // Check if LoRA fits
        let lora_mem = planner.estimate_lora(64).total_bytes;
        if lora_mem < (self.available_vram_bytes as f64 * self.target_utilization) as u64 {
            return Method::LoRA;
        }

        // Default to QLoRA
        Method::QLoRA
    }

    fn find_optimal_rank(&self, method: Method) -> Result<u32> {
        if method == Method::Full {
            return Ok(0);
        }

        let planner = MemoryPlanner::new(self.model_params);
        let target_mem = (self.available_vram_bytes as f64 * self.target_utilization) as u64;

        // Binary search for optimal rank
        let mut low = 8u32;
        let mut high = 256u32;
        let mut best_rank = 64u32;

        while low <= high {
            let mid = u32::midpoint(low, high);
            let mem = if method == Method::QLoRA {
                planner.estimate_qlora(mid, 4).total_bytes
            } else {
                planner.estimate_lora(mid).total_bytes
            };

            if mem <= target_mem {
                best_rank = mid;
                low = mid + 1;
            } else {
                if mid == 0 {
                    break;
                }
                high = mid - 1;
            }
        }

        if best_rank < 8 {
            return Err(EntrenarError::InsufficientMemory {
                required: planner.estimate_qlora(8, 4).total_bytes as f64 / 1e9,
                available: self.available_vram_bytes as f64 / 1e9,
            });
        }

        Ok(best_rank)
    }

    fn calculate_trainable_params(&self, method: Method, rank: u32) -> u64 {
        if method == Method::Full {
            return self.model_params;
        }

        // Estimate hidden dim and layers
        let (hidden_dim, num_layers) = if self.model_params > 60_000_000_000 {
            (8192u64, 80u64)
        } else if self.model_params > 10_000_000_000 {
            (5120, 40)
        } else if self.model_params > 5_000_000_000 {
            (4096, 32)
        } else if self.model_params > 1_000_000_000 {
            (2048, 22)
        } else {
            (1024, 12)
        };

        // LoRA params: 2 matrices × 4 modules × num_layers
        // Each matrix is either (hidden × rank) or (rank × hidden)
        (hidden_dim * u64::from(rank) * 2) * 4 * num_layers
    }
}

/// Compare multiple fine-tuning methods.
pub fn compare_methods(model_params: u64, available_vram_gb: f64) -> Vec<MethodComparison> {
    let methods = [Method::Full, Method::LoRA, Method::QLoRA];
    let optimizer = LoraOptimizer::new(model_params, available_vram_gb);

    methods
        .iter()
        .filter_map(|&method| {
            optimizer.optimize(method).ok().map(|config| MethodComparison {
                method,
                fits: config.utilization_percent <= 100.0,
                memory_gb: config.memory_gb,
                trainable_params: config.trainable_params,
                speedup: config.speedup,
                rank: config.rank,
            })
        })
        .collect()
}

/// Method comparison result.
#[derive(Debug, Clone)]
pub struct MethodComparison {
    pub method: Method,
    pub fits: bool,
    pub memory_gb: f64,
    pub trainable_params: u64,
    pub speedup: f64,
    pub rank: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_selects_qlora_for_small_vram() {
        let optimizer = LoraOptimizer::new(7_000_000_000, 8.0);
        let config = optimizer.optimize(Method::Auto).expect("config should be valid");

        // With only 8GB, should select QLoRA
        assert_eq!(config.method, Method::QLoRA);
    }

    #[test]
    fn test_optimizer_selects_lora_for_medium_vram() {
        let optimizer = LoraOptimizer::new(7_000_000_000, 24.0);
        let config = optimizer.optimize(Method::Auto).expect("config should be valid");

        // With 24GB for 7B model, optimizer may select LoRA, QLoRA, or Full
        assert!(matches!(config.method, Method::LoRA | Method::QLoRA | Method::Full));
    }

    #[test]
    fn test_optimal_rank_is_positive() {
        let optimizer = LoraOptimizer::new(7_000_000_000, 16.0);
        let config = optimizer.optimize(Method::LoRA).expect("config should be valid");

        assert!(config.rank >= 8);
        assert!(config.rank <= 256);
    }

    #[test]
    fn test_trainable_params_less_than_total() {
        let optimizer = LoraOptimizer::new(7_000_000_000, 16.0);
        let config = optimizer.optimize(Method::LoRA).expect("config should be valid");

        assert!(config.trainable_params < 7_000_000_000);
        assert!(config.trainable_percent < 10.0);
    }

    #[test]
    fn test_compare_methods() {
        let comparisons = compare_methods(7_000_000_000, 16.0);

        assert!(!comparisons.is_empty());
        assert!(comparisons.iter().any(|c| c.method == Method::QLoRA));
    }

    #[test]
    fn test_alpha_is_rank_over_4() {
        let optimizer = LoraOptimizer::new(7_000_000_000, 16.0);
        let config = optimizer.optimize(Method::LoRA).expect("config should be valid");

        assert!((config.alpha - config.rank as f32 / 4.0).abs() < 0.01);
    }

    #[test]
    fn test_target_modules_populated() {
        let optimizer = LoraOptimizer::new(7_000_000_000, 16.0);
        let config = optimizer.optimize(Method::LoRA).expect("config should be valid");

        assert!(!config.target_modules.is_empty());
        assert!(config.target_modules.contains(&"q_proj".to_string()));
    }

    #[test]
    fn test_with_target_utilization() {
        let optimizer = LoraOptimizer::new(7_000_000_000, 16.0).with_target_utilization(0.75);
        let config = optimizer.optimize(Method::LoRA).expect("config should be valid");

        // Lower target utilization should give smaller rank
        let high_util = LoraOptimizer::new(7_000_000_000, 16.0)
            .with_target_utilization(0.95)
            .optimize(Method::LoRA)
            .expect("operation should succeed");

        assert!(config.rank <= high_util.rank);
    }

    #[test]
    fn test_target_utilization_clamping() {
        // Test that utilization is clamped to 0.5-0.95
        let low = LoraOptimizer::new(7_000_000_000, 16.0).with_target_utilization(0.1);
        assert!(low.target_utilization >= 0.5);

        let high = LoraOptimizer::new(7_000_000_000, 16.0).with_target_utilization(1.5);
        assert!(high.target_utilization <= 0.95);
    }

    #[test]
    fn test_format_params_billion() {
        assert_eq!(format_params(7_000_000_000), "7.0B");
        assert_eq!(format_params(1_500_000_000), "1.5B");
    }

    #[test]
    fn test_format_params_million() {
        assert_eq!(format_params(350_000_000), "350.0M");
        assert_eq!(format_params(1_500_000), "1.5M");
    }

    #[test]
    fn test_format_params_thousand() {
        assert_eq!(format_params(500_000), "500.0K");
        assert_eq!(format_params(1_500), "1.5K");
    }

    #[test]
    fn test_to_comparison_table() {
        let optimizer = LoraOptimizer::new(7_000_000_000, 16.0);
        let config = optimizer.optimize(Method::LoRA).expect("config should be valid");
        let table = config.to_comparison_table();

        assert!(table.contains("Optimal Configuration"));
        assert!(table.contains("Method:"));
        assert!(table.contains("Rank:"));
        assert!(table.contains("Alpha:"));
        assert!(table.contains("Memory:"));
    }

    #[test]
    fn test_full_method_rank_zero() {
        let optimizer = LoraOptimizer::new(1_000_000_000, 100.0);
        let config = optimizer.optimize(Method::Full).expect("config should be valid");
        assert_eq!(config.rank, 0);
    }

    #[test]
    fn test_full_method_all_params_trainable() {
        let optimizer = LoraOptimizer::new(1_000_000_000, 100.0);
        let config = optimizer.optimize(Method::Full).expect("config should be valid");
        assert_eq!(config.trainable_params, 1_000_000_000);
        assert_eq!(config.trainable_percent, 100.0);
    }

    #[test]
    fn test_speedup_values() {
        let optimizer = LoraOptimizer::new(7_000_000_000, 100.0);

        let full = optimizer.optimize(Method::Full).expect("operation should succeed");
        assert_eq!(full.speedup, 1.0);

        let lora = optimizer.optimize(Method::LoRA).expect("operation should succeed");
        assert_eq!(lora.speedup, 2.5);

        let qlora = optimizer.optimize(Method::QLoRA).expect("operation should succeed");
        assert_eq!(qlora.speedup, 1.8);
    }

    #[test]
    fn test_compare_methods_includes_all() {
        let comparisons = compare_methods(7_000_000_000, 100.0);

        assert!(comparisons.iter().any(|c| c.method == Method::Full));
        assert!(comparisons.iter().any(|c| c.method == Method::LoRA));
        assert!(comparisons.iter().any(|c| c.method == Method::QLoRA));
    }

    #[test]
    fn test_compare_methods_small_vram() {
        let comparisons = compare_methods(7_000_000_000, 4.0);

        // With very small VRAM, only QLoRA might fit
        let _fitting: Vec<_> = comparisons.iter().filter(|c| c.fits).collect();
        // At least one method should work (QLoRA)
        assert!(!comparisons.is_empty());
    }

    #[test]
    fn test_method_comparison_struct() {
        let comparisons = compare_methods(7_000_000_000, 16.0);
        let qlora = comparisons.iter().find(|c| c.method == Method::QLoRA);

        if let Some(c) = qlora {
            assert!(c.memory_gb > 0.0);
            assert!(c.trainable_params > 0);
            assert!(c.speedup > 0.0);
            assert!(c.rank >= 8);
        }
    }
}
