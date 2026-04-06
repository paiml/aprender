//! Memory planning for LoRA configurations (Heijunka principle).

use crate::Method;

/// Memory requirement estimation.
#[derive(Debug, Clone)]
pub struct MemoryRequirement {
    /// Base model memory in bytes
    pub model_bytes: u64,
    /// Adapter memory in bytes
    pub adapter_bytes: u64,
    /// Optimizer state memory in bytes
    pub optimizer_bytes: u64,
    /// Activation memory in bytes
    pub activation_bytes: u64,
    /// Total memory in bytes
    pub total_bytes: u64,
    /// Memory savings compared to full fine-tuning (percentage)
    pub savings_percent: f64,
}

impl MemoryRequirement {
    /// Format as human-readable string.
    pub fn to_human_readable(&self) -> String {
        format!(
            "Memory Requirement:\n  Model: {:.1} GB\n  Adapter: {:.1} GB\n  Optimizer: {:.1} GB\n  Activations: {:.1} GB\n  Total: {:.1} GB\n  Savings: {:.0}%",
            self.model_bytes as f64 / 1e9,
            self.adapter_bytes as f64 / 1e9,
            self.optimizer_bytes as f64 / 1e9,
            self.activation_bytes as f64 / 1e9,
            self.total_bytes as f64 / 1e9,
            self.savings_percent
        )
    }
}

/// Memory planner for different fine-tuning methods.
#[derive(Debug)]
pub struct MemoryPlanner {
    model_params: u64,
    hidden_dim: u64,
    num_layers: u32,
    batch_size: u32,
    seq_len: u32,
}

impl MemoryPlanner {
    /// Create a new memory planner.
    pub fn new(model_params: u64) -> Self {
        // Estimate architecture from param count
        let (hidden_dim, num_layers) = estimate_architecture(model_params);

        Self {
            model_params,
            hidden_dim,
            num_layers,
            batch_size: 32,
            seq_len: 512,
        }
    }

    /// Set batch size.
    pub fn with_batch_size(mut self, batch_size: u32) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set sequence length.
    pub fn with_seq_len(mut self, seq_len: u32) -> Self {
        self.seq_len = seq_len;
        self
    }

    /// Estimate memory for full fine-tuning.
    pub fn estimate_full(&self) -> MemoryRequirement {
        let model_bytes = self.model_params * 2; // FP16
        let optimizer_bytes = self.model_params * 8; // Adam: 2 FP32 states
        let activation_bytes = self.estimate_activations();

        let total_bytes = model_bytes + optimizer_bytes + activation_bytes;

        MemoryRequirement {
            model_bytes,
            adapter_bytes: 0,
            optimizer_bytes,
            activation_bytes,
            total_bytes,
            savings_percent: 0.0,
        }
    }

    /// Estimate memory for LoRA fine-tuning.
    pub fn estimate_lora(&self, rank: u32) -> MemoryRequirement {
        let model_bytes = self.model_params * 2; // FP16 (frozen)

        // LoRA adapters: 2 matrices per target module (typically 4 modules per layer)
        // A: d × r, B: r × d for each module
        let adapter_params =
            (self.hidden_dim * u64::from(rank) * 2) * 4 * u64::from(self.num_layers);
        let adapter_bytes = adapter_params * 2; // FP16

        // Optimizer only for adapter params
        let optimizer_bytes = adapter_params * 8; // Adam states

        let activation_bytes = self.estimate_activations();

        let total_bytes = model_bytes + adapter_bytes + optimizer_bytes + activation_bytes;
        let full_total = self.estimate_full().total_bytes;
        let savings_percent = (1.0 - total_bytes as f64 / full_total as f64) * 100.0;

        MemoryRequirement {
            model_bytes,
            adapter_bytes,
            optimizer_bytes,
            activation_bytes,
            total_bytes,
            savings_percent,
        }
    }

    /// Estimate memory for QLoRA fine-tuning.
    pub fn estimate_qlora(&self, rank: u32, bits: u8) -> MemoryRequirement {
        // Base model in quantized format
        let model_bytes = self.model_params * u64::from(bits) / 8;

        // LoRA adapters in FP16
        let adapter_params =
            (self.hidden_dim * u64::from(rank) * 2) * 4 * u64::from(self.num_layers);
        let adapter_bytes = adapter_params * 2;

        // Optimizer only for adapter params
        let optimizer_bytes = adapter_params * 8;

        let activation_bytes = self.estimate_activations();

        let total_bytes = model_bytes + adapter_bytes + optimizer_bytes + activation_bytes;
        let full_total = self.estimate_full().total_bytes;
        let savings_percent = (1.0 - total_bytes as f64 / full_total as f64) * 100.0;

        MemoryRequirement {
            model_bytes,
            adapter_bytes,
            optimizer_bytes,
            activation_bytes,
            total_bytes,
            savings_percent,
        }
    }

    /// Estimate memory for a given method.
    pub fn estimate(&self, method: Method, rank: u32) -> MemoryRequirement {
        match method {
            Method::Full => self.estimate_full(),
            Method::LoRA => self.estimate_lora(rank),
            Method::QLoRA => self.estimate_qlora(rank, 4),
            Method::Auto => {
                // Try QLoRA first, then LoRA, then full
                self.estimate_qlora(rank, 4)
            }
        }
    }

    fn estimate_activations(&self) -> u64 {
        // Activations per layer: batch × seq × hidden × 2 (forward + backward)
        let per_layer =
            u64::from(self.batch_size) * u64::from(self.seq_len) * self.hidden_dim * 2 * 2; // FP16

        per_layer * u64::from(self.num_layers)
    }
}

fn estimate_architecture(params: u64) -> (u64, u32) {
    // Rough estimates based on common model sizes
    if params > 60_000_000_000 {
        (8192, 80) // 70B class
    } else if params > 10_000_000_000 {
        (5120, 40) // 13B class
    } else if params > 5_000_000_000 {
        (4096, 32) // 7B class
    } else if params > 1_000_000_000 {
        (2048, 22) // 1-3B class
    } else if params > 300_000_000 {
        (1024, 12) // 350M class (BERT-base)
    } else {
        (768, 12) // Small models
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_planner_7b() {
        let planner = MemoryPlanner::new(7_000_000_000);

        let full = planner.estimate_full();
        let lora = planner.estimate_lora(64);
        let qlora = planner.estimate_qlora(64, 4);

        // Full should use most memory
        assert!(full.total_bytes > lora.total_bytes);
        assert!(lora.total_bytes > qlora.total_bytes);

        // QLoRA should have significant savings
        assert!(qlora.savings_percent > 50.0);
    }

    #[test]
    fn test_lora_adapter_memory_scales_with_rank() {
        let planner = MemoryPlanner::new(7_000_000_000);

        let lora_16 = planner.estimate_lora(16);
        let lora_64 = planner.estimate_lora(64);
        let lora_128 = planner.estimate_lora(128);

        assert!(lora_16.adapter_bytes < lora_64.adapter_bytes);
        assert!(lora_64.adapter_bytes < lora_128.adapter_bytes);
    }

    #[test]
    fn test_qlora_4bit_vs_8bit() {
        let planner = MemoryPlanner::new(7_000_000_000);

        let qlora_4 = planner.estimate_qlora(64, 4);
        let qlora_8 = planner.estimate_qlora(64, 8);

        // 4-bit should use less model memory
        assert!(qlora_4.model_bytes < qlora_8.model_bytes);
    }

    #[test]
    fn test_batch_size_affects_activations() {
        let planner_small = MemoryPlanner::new(7_000_000_000).with_batch_size(8);
        let planner_large = MemoryPlanner::new(7_000_000_000).with_batch_size(64);

        let small = planner_small.estimate_full();
        let large = planner_large.estimate_full();

        assert!(small.activation_bytes < large.activation_bytes);
    }

    #[test]
    fn test_architecture_estimation() {
        let (hidden, layers) = estimate_architecture(7_000_000_000);
        assert_eq!(hidden, 4096);
        assert_eq!(layers, 32);

        let (hidden, layers) = estimate_architecture(350_000_000);
        assert_eq!(hidden, 1024);
        assert_eq!(layers, 12);
    }

    #[test]
    fn test_architecture_estimation_all_tiers() {
        // 70B class
        let (hidden, layers) = estimate_architecture(70_000_000_000);
        assert_eq!(hidden, 8192);
        assert_eq!(layers, 80);

        // 13B class
        let (hidden, layers) = estimate_architecture(13_000_000_000);
        assert_eq!(hidden, 5120);
        assert_eq!(layers, 40);

        // 1-3B class
        let (hidden, layers) = estimate_architecture(2_000_000_000);
        assert_eq!(hidden, 2048);
        assert_eq!(layers, 22);

        // Small models
        let (hidden, layers) = estimate_architecture(100_000_000);
        assert_eq!(hidden, 768);
        assert_eq!(layers, 12);
    }

    #[test]
    fn test_with_seq_len() {
        let planner = MemoryPlanner::new(7_000_000_000).with_seq_len(1024);
        let full_1024 = planner.estimate_full();

        let planner_short = MemoryPlanner::new(7_000_000_000).with_seq_len(256);
        let full_256 = planner_short.estimate_full();

        // Longer sequences require more activation memory
        assert!(full_1024.activation_bytes > full_256.activation_bytes);
    }

    #[test]
    fn test_estimate_method_dispatch() {
        let planner = MemoryPlanner::new(7_000_000_000);

        let full = planner.estimate(Method::Full, 64);
        assert_eq!(full.adapter_bytes, 0);

        let lora = planner.estimate(Method::LoRA, 64);
        assert!(lora.adapter_bytes > 0);

        let qlora = planner.estimate(Method::QLoRA, 64);
        assert!(qlora.model_bytes < lora.model_bytes);

        let auto = planner.estimate(Method::Auto, 64);
        assert!(auto.savings_percent > 0.0);
    }

    #[test]
    fn test_to_human_readable() {
        let planner = MemoryPlanner::new(7_000_000_000);
        let req = planner.estimate_full();
        let readable = req.to_human_readable();

        assert!(readable.contains("Memory Requirement"));
        assert!(readable.contains("GB"));
        assert!(readable.contains("Model:"));
        assert!(readable.contains("Total:"));
    }

    #[test]
    fn test_full_has_zero_savings() {
        let planner = MemoryPlanner::new(7_000_000_000);
        let full = planner.estimate_full();
        assert_eq!(full.savings_percent, 0.0);
    }

    #[test]
    fn test_lora_has_positive_savings() {
        let planner = MemoryPlanner::new(7_000_000_000);
        let lora = planner.estimate_lora(64);
        assert!(lora.savings_percent > 0.0);
    }

    #[test]
    fn test_qlora_saves_more_than_lora() {
        let planner = MemoryPlanner::new(7_000_000_000);
        let lora = planner.estimate_lora(64);
        let qlora = planner.estimate_qlora(64, 4);
        assert!(qlora.savings_percent > lora.savings_percent);
    }
}
