//! Edge deployment efficiency metrics.

use serde::{Deserialize, Serialize};

use super::budget::{BudgetViolation, WasmBudget};

/// Edge deployment efficiency metrics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdgeEfficiency {
    /// Binary/model size in bytes
    pub binary_size_bytes: u64,
    /// Startup/initialization latency in milliseconds
    pub startup_latency_ms: u64,
    /// 99th percentile inference latency in milliseconds
    pub inference_latency_p99_ms: f64,
    /// Runtime memory footprint in bytes
    pub memory_footprint_bytes: u64,
}

impl EdgeEfficiency {
    /// Create new edge efficiency metrics
    pub fn new(
        binary_size_bytes: u64,
        startup_latency_ms: u64,
        inference_latency_p99_ms: f64,
        memory_footprint_bytes: u64,
    ) -> Self {
        Self {
            binary_size_bytes,
            startup_latency_ms,
            inference_latency_p99_ms,
            memory_footprint_bytes,
        }
    }

    /// Check if deployment meets WASM budget constraints
    pub fn meets_wasm_budget(&self, budget: &WasmBudget) -> bool {
        self.binary_size_bytes <= budget.max_binary_size
            && self.startup_latency_ms <= budget.max_startup_ms
            && self.memory_footprint_bytes <= budget.max_memory_bytes
    }

    /// Get binary size in MB
    pub fn binary_size_mb(&self) -> f64 {
        self.binary_size_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get memory footprint in MB
    pub fn memory_footprint_mb(&self) -> f64 {
        self.memory_footprint_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Calculate maximum throughput based on latency
    pub fn max_throughput_per_sec(&self) -> f64 {
        if self.inference_latency_p99_ms > 0.0 {
            1000.0 / self.inference_latency_p99_ms
        } else {
            0.0
        }
    }

    /// Check which budget constraints are violated
    pub fn budget_violations(&self, budget: &WasmBudget) -> Vec<BudgetViolation> {
        let mut violations = Vec::new();

        if self.binary_size_bytes > budget.max_binary_size {
            violations.push(BudgetViolation::BinarySize {
                actual: self.binary_size_bytes,
                limit: budget.max_binary_size,
            });
        }

        if self.startup_latency_ms > budget.max_startup_ms {
            violations.push(BudgetViolation::StartupLatency {
                actual: self.startup_latency_ms,
                limit: budget.max_startup_ms,
            });
        }

        if self.memory_footprint_bytes > budget.max_memory_bytes {
            violations.push(BudgetViolation::MemoryFootprint {
                actual: self.memory_footprint_bytes,
                limit: budget.max_memory_bytes,
            });
        }

        violations
    }
}

impl Default for EdgeEfficiency {
    fn default() -> Self {
        Self {
            binary_size_bytes: 0,
            startup_latency_ms: 0,
            inference_latency_p99_ms: 0.0,
            memory_footprint_bytes: 0,
        }
    }
}
