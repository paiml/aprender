//! PTX Parser and Analyzer
//!
//! Implements the Analyzer trait for NVIDIA PTX assembly.

use crate::analyzer::{
    AnalysisReport, Analyzer, MemoryPattern, MudaType, MudaWarning, RegisterUsage, RooflineMetric,
};
use crate::error::Result;
use regex::Regex;

/// PTX code analyzer
pub struct PtxAnalyzer {
    /// Warn if register count exceeds this threshold
    pub register_warning_threshold: u32,
    /// Warn if coalescing ratio falls below this threshold
    pub coalescing_warning_threshold: f32,
}

impl Default for PtxAnalyzer {
    fn default() -> Self {
        Self {
            register_warning_threshold: 128,
            coalescing_warning_threshold: 0.8,
        }
    }
}

impl PtxAnalyzer {
    /// Create a new PTX analyzer with default thresholds
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse register declarations from PTX
    fn parse_registers(&self, ptx: &str) -> RegisterUsage {
        let mut usage = RegisterUsage::default();

        // Match patterns like: .reg .f32 %f<24>;
        let reg_pattern =
            Regex::new(r"\.reg\s+\.(\w+)\s+%\w+<(\d+)>").expect("valid regex pattern");

        for cap in reg_pattern.captures_iter(ptx) {
            let reg_type = &cap[1];
            let count: u32 = cap[2].parse().unwrap_or(0);

            match reg_type {
                "f32" => usage.f32_regs += count,
                "f64" => usage.f64_regs += count,
                "b32" | "u32" | "s32" => usage.b32_regs += count,
                "b64" | "u64" | "s64" => usage.b64_regs += count,
                "pred" => usage.pred_regs += count,
                _ => {}
            }
        }

        usage
    }

    /// Parse memory operations from PTX
    fn parse_memory_ops(&self, ptx: &str) -> MemoryPattern {
        let mut pattern = MemoryPattern::default();

        // Count global loads
        let global_load = Regex::new(r"ld\.global").expect("valid regex pattern");
        pattern.global_loads = global_load.find_iter(ptx).count() as u32;

        // Count global stores
        let global_store = Regex::new(r"st\.global").expect("valid regex pattern");
        pattern.global_stores = global_store.find_iter(ptx).count() as u32;

        // Count shared loads
        let shared_load = Regex::new(r"ld\.shared").expect("valid regex pattern");
        pattern.shared_loads = shared_load.find_iter(ptx).count() as u32;

        // Count shared stores
        let shared_store = Regex::new(r"st\.shared").expect("valid regex pattern");
        pattern.shared_stores = shared_store.find_iter(ptx).count() as u32;

        // Estimate coalescing based on access patterns
        // Coalesced access indicators:
        // 1. tid/ctaid references (thread and block IDs - used for index computation)
        // 2. mad.lo with tid (computing linear index from thread/block IDs)
        // 3. mul.wide with small constant (stride-1 access)
        // 4. shfl instructions (warp shuffle - implicit coalescing)
        // Note: Include both x and y dimensions since 2D kernels use both
        let tid_pattern =
            Regex::new(r"%tid\.[xy]|%ntid\.[xy]|%ctaid\.[xy]").expect("valid regex pattern");
        let tid_refs = tid_pattern.find_iter(ptx).count();

        // mad.lo often computes coalesced indices: mad.lo %r, %ctaid, %ntid, %tid
        let mad_pattern = Regex::new(r"mad\.lo").expect("valid regex pattern");
        let mad_refs = mad_pattern.find_iter(ptx).count();

        // mul.lo also used for index computation
        let mul_lo_pattern = Regex::new(r"mul\.lo").expect("valid regex pattern");
        let mul_lo_refs = mul_lo_pattern.find_iter(ptx).count();

        // mul.wide with small constants indicates stride-based access
        let stride_pattern = Regex::new(r"mul\.wide\.[us]32").expect("valid regex pattern");
        let stride_refs = stride_pattern.find_iter(ptx).count();

        // Warp shuffles indicate warp-level data sharing (inherently coalesced)
        let shfl_pattern = Regex::new(r"shfl\.(down|up|bfly|idx)").expect("valid regex pattern");
        let shfl_refs = shfl_pattern.find_iter(ptx).count();

        // rem/div operations often used for lane computation in coalesced patterns
        let lane_pattern = Regex::new(r"rem\.u32|div\.u32").expect("valid regex pattern");
        let lane_refs = lane_pattern.find_iter(ptx).count();

        let total_accesses = pattern.global_loads + pattern.global_stores;
        if total_accesses > 0 {
            // Improved heuristic: weight different indicators
            // Each indicator suggests thread-based indexing which implies coalescing potential
            let coalescing_score = tid_refs as f32
                + (mad_refs as f32 * 0.6)  // mad.lo strongly indicates index computation
                + (mul_lo_refs as f32 * 0.4) // mul.lo also used for indices
                + (stride_refs as f32 * 0.3) // stride patterns
                + (shfl_refs as f32 * 0.3)  // warp shuffles
                + (lane_refs as f32 * 0.2); // lane computation
            pattern.coalesced_ratio = (coalescing_score / total_accesses as f32).min(1.0);
        } else {
            pattern.coalesced_ratio = 1.0;
        }

        pattern
    }

    /// Count total instructions
    fn count_instructions(&self, ptx: &str) -> u32 {
        // Count lines that look like instructions (not directives or labels)
        let instruction_pattern = Regex::new(r"^\s+(add|sub|mul|div|mad|fma|ld|st|mov|setp|bra|ret|cvt|and|or|xor|shl|shr|min|max|abs|neg|sqrt|rsqrt|sin|cos|ex2|lg2|rcp|selp|set|bar)").expect("valid regex pattern");

        ptx.lines()
            .filter(|line| instruction_pattern.is_match(line))
            .count() as u32
    }

    /// Extract kernel name from PTX
    fn extract_kernel_name(&self, ptx: &str) -> String {
        let entry_pattern = Regex::new(r"\.entry\s+(\w+)").expect("valid regex pattern");
        entry_pattern
            .captures(ptx)
            .map(|c| c[1].to_string())
            .unwrap_or_else(|| "unknown".to_string())
    }

    /// Detect spills (Muda of Transport)
    fn detect_spills(&self, ptx: &str) -> Option<MudaWarning> {
        // Spills manifest as .local memory usage
        let local_pattern = Regex::new(r"\.local").expect("valid regex pattern");
        let spill_count = local_pattern.find_iter(ptx).count();

        if spill_count > 0 {
            Some(MudaWarning {
                muda_type: MudaType::Transport,
                description: format!("{} potential register spills detected", spill_count),
                impact: "High latency local memory access".to_string(),
                line: None,
                suggestion: Some(
                    "Reduce live variables or increase register allocation".to_string(),
                ),
            })
        } else {
            None
        }
    }

    /// Detect uncoalesced access (Muda of Waiting)
    fn detect_uncoalesced(&self, memory: &MemoryPattern) -> Option<MudaWarning> {
        if memory.coalesced_ratio < self.coalescing_warning_threshold {
            Some(MudaWarning {
                muda_type: MudaType::Waiting,
                description: format!(
                    "Memory coalescing ratio {:.1}% below threshold {:.1}%",
                    memory.coalesced_ratio * 100.0,
                    self.coalescing_warning_threshold * 100.0
                ),
                impact: "Serialized memory transactions, reduced bandwidth".to_string(),
                line: None,
                suggestion: Some(
                    "Ensure adjacent threads access adjacent memory addresses".to_string(),
                ),
            })
        } else {
            None
        }
    }

    /// Detect excessive register usage
    fn detect_register_pressure(&self, registers: &RegisterUsage) -> Option<MudaWarning> {
        let total = registers.total();
        if total > self.register_warning_threshold {
            Some(MudaWarning {
                muda_type: MudaType::Overprocessing,
                description: format!(
                    "High register usage: {} registers (threshold: {})",
                    total, self.register_warning_threshold
                ),
                impact: "Reduced occupancy, fewer concurrent warps".to_string(),
                line: None,
                suggestion: Some(
                    "Consider loop tiling or reducing intermediate values".to_string(),
                ),
            })
        } else {
            None
        }
    }
}

impl Analyzer for PtxAnalyzer {
    fn target_name(&self) -> &str {
        "PTX"
    }

    fn analyze(&self, ptx: &str) -> Result<AnalysisReport> {
        let registers = self.parse_registers(ptx);
        let memory = self.parse_memory_ops(ptx);
        let instruction_count = self.count_instructions(ptx);
        let name = self.extract_kernel_name(ptx);
        let warnings = self.detect_muda(ptx);
        let estimated_occupancy = registers.estimated_occupancy();

        let mut report = AnalysisReport {
            name,
            target: self.target_name().to_string(),
            registers,
            memory,
            warnings,
            instruction_count,
            estimated_occupancy,
            ..Default::default()
        };

        report.roofline = self.estimate_roofline(&report);
        Ok(report)
    }

    fn detect_muda(&self, ptx: &str) -> Vec<MudaWarning> {
        let mut warnings = Vec::new();

        if let Some(w) = self.detect_spills(ptx) {
            warnings.push(w);
        }

        let memory = self.parse_memory_ops(ptx);
        if let Some(w) = self.detect_uncoalesced(&memory) {
            warnings.push(w);
        }

        let registers = self.parse_registers(ptx);
        if let Some(w) = self.detect_register_pressure(&registers) {
            warnings.push(w);
        }

        warnings
    }

    fn estimate_roofline(&self, analysis: &AnalysisReport) -> RooflineMetric {
        // Simplified roofline model
        // Arithmetic intensity = FLOPs / Bytes transferred
        let mem_ops = analysis.memory.global_loads + analysis.memory.global_stores;
        let bytes = mem_ops * 4; // Assume f32

        let flops = analysis.instruction_count; // Rough approximation

        let arithmetic_intensity = if bytes > 0 {
            flops as f32 / bytes as f32
        } else {
            0.0
        };

        // SM 7.0 theoretical peak: ~15 TFLOPS (varies by GPU)
        let theoretical_peak_gflops = 15000.0;

        // Memory bound if AI < ridge point (typically ~10 for modern GPUs)
        let memory_bound = arithmetic_intensity < 10.0;

        RooflineMetric {
            arithmetic_intensity,
            theoretical_peak_gflops,
            memory_bound,
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod property_tests;
