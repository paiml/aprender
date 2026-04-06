//! Core Analyzer trait and types for trueno-explain
//!
//! Implements the Toyota Way principle of Genchi Genbutsu (Go and See)
//! by making invisible compiler transformations visible.

use crate::error::Result;
use serde::{Deserialize, Serialize};

/// Muda (waste) categories mapped to technical inefficiencies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MudaType {
    /// Muda of Transport: Register spills (moving data unnecessarily)
    Transport,
    /// Muda of Waiting: Uncoalesced memory access (stalls)
    Waiting,
    /// Muda of Overprocessing: Redundant instructions or excessive precision
    Overprocessing,
}

/// A warning about detected waste (Muda) in generated code
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MudaWarning {
    /// The category of waste detected
    pub muda_type: MudaType,
    /// Human-readable description of the issue
    pub description: String,
    /// Performance impact of the waste
    pub impact: String,
    /// Source line number if available
    pub line: Option<usize>,
    /// Suggested fix for the issue
    pub suggestion: Option<String>,
}

/// Register usage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RegisterUsage {
    /// Number of 32-bit floating point registers
    pub f32_regs: u32,
    /// Number of 64-bit floating point registers
    pub f64_regs: u32,
    /// Number of 32-bit integer/bit registers
    pub b32_regs: u32,
    /// Number of 64-bit integer/bit registers
    pub b64_regs: u32,
    /// Number of predicate (1-bit) registers
    pub pred_regs: u32,
}

impl RegisterUsage {
    /// Total register count
    #[must_use]
    pub fn total(&self) -> u32 {
        self.f32_regs + self.f64_regs + self.b32_regs + self.b64_regs + self.pred_regs
    }

    /// Estimate occupancy based on register usage (simplified model)
    /// SM 7.0+: 65536 registers per SM, max 255 per thread
    #[must_use]
    pub fn estimated_occupancy(&self) -> f32 {
        let total = self.total();
        if total == 0 {
            return 1.0;
        }
        // Simplified: assume 2048 threads max per SM
        // registers_per_thread * threads <= 65536
        let max_threads = (65536 / total.max(1)).min(2048);
        max_threads as f32 / 2048.0
    }
}

/// Memory access pattern analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct MemoryPattern {
    /// Number of global memory load operations
    pub global_loads: u32,
    /// Number of global memory store operations
    pub global_stores: u32,
    /// Number of shared memory load operations
    pub shared_loads: u32,
    /// Number of shared memory store operations
    pub shared_stores: u32,
    /// Ratio of coalesced memory accesses (0.0-1.0)
    pub coalesced_ratio: f32,
}

/// Roofline model metrics for performance estimation
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RooflineMetric {
    /// FLOPs per byte transferred (compute vs memory ratio)
    pub arithmetic_intensity: f32,
    /// Theoretical peak compute performance in GFLOPS
    pub theoretical_peak_gflops: f32,
    /// True if kernel is memory-bound, false if compute-bound
    pub memory_bound: bool,
}

/// Complete analysis report for a kernel or function
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnalysisReport {
    /// Kernel or function name
    pub name: String,
    /// Target IR type (PTX, x86 ASM, WGSL)
    pub target: String,
    /// Register usage statistics
    pub registers: RegisterUsage,
    /// Memory access patterns
    pub memory: MemoryPattern,
    /// Roofline performance model
    pub roofline: RooflineMetric,
    /// Detected waste (Muda) warnings
    pub warnings: Vec<MudaWarning>,
    /// Total instruction count
    pub instruction_count: u32,
    /// Estimated GPU occupancy (0.0-1.0)
    pub estimated_occupancy: f32,
}

/// Core trait for all analyzers (PTX, SIMD, WGSL)
pub trait Analyzer {
    /// The type of IR being analyzed (e.g., "PTX", "x86 ASM", "WGSL")
    fn target_name(&self) -> &str;

    /// Analyze the provided code and return a structured report
    ///
    /// # Errors
    ///
    /// Returns `ExplainError::PtxParseError` if the code cannot be parsed.
    fn analyze(&self, code: &str) -> Result<AnalysisReport>;

    /// Identify specific performance bottlenecks (Muda)
    fn detect_muda(&self, code: &str) -> Vec<MudaWarning>;

    /// Estimate theoretical peak performance
    fn estimate_roofline(&self, analysis: &AnalysisReport) -> RooflineMetric;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_usage_total() {
        let usage = RegisterUsage {
            f32_regs: 10,
            f64_regs: 5,
            b32_regs: 8,
            b64_regs: 4,
            pred_regs: 2,
        };
        assert_eq!(usage.total(), 29);
    }

    #[test]
    fn test_register_usage_total_empty() {
        let usage = RegisterUsage::default();
        assert_eq!(usage.total(), 0);
    }

    #[test]
    fn test_occupancy_low_registers() {
        let usage = RegisterUsage {
            f32_regs: 16,
            ..Default::default()
        };
        // 16 registers -> 65536/16 = 4096, capped at 2048 -> 100%
        assert!((usage.estimated_occupancy() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_occupancy_high_registers() {
        let usage = RegisterUsage {
            f32_regs: 128,
            ..Default::default()
        };
        // 128 registers -> 65536/128 = 512 threads -> 512/2048 = 25%
        assert!((usage.estimated_occupancy() - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_occupancy_zero_registers() {
        let usage = RegisterUsage::default();
        assert!((usage.estimated_occupancy() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_muda_warning_serialization() {
        let warning = MudaWarning {
            muda_type: MudaType::Transport,
            description: "5 register spills detected".to_string(),
            impact: "High latency local memory access".to_string(),
            line: Some(42),
            suggestion: Some("Reduce live variables".to_string()),
        };

        let json = serde_json::to_string(&warning).unwrap();
        let parsed: MudaWarning = serde_json::from_str(&json).unwrap();
        assert_eq!(warning, parsed);
    }

    #[test]
    fn test_analysis_report_serialization() {
        let report = AnalysisReport {
            name: "test_kernel".to_string(),
            target: "PTX".to_string(),
            registers: RegisterUsage {
                f32_regs: 24,
                b32_regs: 18,
                ..Default::default()
            },
            memory: MemoryPattern {
                global_loads: 100,
                coalesced_ratio: 0.95,
                ..Default::default()
            },
            warnings: vec![],
            instruction_count: 150,
            estimated_occupancy: 0.875,
            ..Default::default()
        };

        let json = serde_json::to_string_pretty(&report).unwrap();
        assert!(json.contains("test_kernel"));
        assert!(json.contains("PTX"));
    }
}
