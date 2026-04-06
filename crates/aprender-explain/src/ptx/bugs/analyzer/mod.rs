use regex::Regex;

use super::types::{PtxBugClass, PtxBugReport};

/// Whitelist entry for suppressing known acceptable warnings
#[derive(Debug, Clone)]
pub struct WhitelistEntry {
    /// Kernel name pattern (supports prefix matching with *)
    pub kernel_pattern: String,
    /// Bug class to suppress
    pub bug_class: PtxBugClass,
    /// Reason for whitelisting
    pub reason: String,
}

/// PTX bug hunting analyzer (inspired by probar `gpu_pixels`)
#[derive(Debug, Default, Clone)]
pub struct PtxBugAnalyzer {
    /// Enable strict mode (more warnings, catches PARITY-114 pattern)
    pub strict: bool,
    /// Whitelist for suppressing known acceptable warnings
    pub whitelist: Vec<WhitelistEntry>,
}

impl PtxBugAnalyzer {
    /// Create analyzer with default (non-strict) mode
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create analyzer with strict mode enabled
    #[must_use]
    pub fn strict() -> Self {
        Self { strict: true, whitelist: Vec::new() }
    }

    /// Add a whitelist entry to suppress warnings
    #[must_use]
    pub fn with_whitelist(
        mut self,
        kernel_pattern: &str,
        bug_class: PtxBugClass,
        reason: &str,
    ) -> Self {
        self.whitelist.push(WhitelistEntry {
            kernel_pattern: kernel_pattern.to_string(),
            bug_class,
            reason: reason.to_string(),
        });
        self
    }

    /// Create analyzer with default whitelist for quantized kernels
    #[must_use]
    pub fn with_quantized_whitelist() -> Self {
        Self::new()
            .with_whitelist(
                "q4k*",
                PtxBugClass::HighRegisterPressure,
                "Quantized kernels require high registers for dequantization",
            )
            .with_whitelist(
                "q5k*",
                PtxBugClass::HighRegisterPressure,
                "Quantized kernels require high registers for dequantization",
            )
            .with_whitelist(
                "q6k*",
                PtxBugClass::HighRegisterPressure,
                "Quantized kernels require high registers for dequantization",
            )
            .with_whitelist(
                "q8k*",
                PtxBugClass::HighRegisterPressure,
                "Quantized kernels require high registers for dequantization",
            )
    }

    /// Create analyzer with comprehensive whitelist for all high-performance kernels
    ///
    /// This whitelist covers expected register pressure and predicate usage in:
    /// - Tensor Core kernels (WMMA requires many registers for matrix fragments)
    /// - Attention kernels (`FlashAttention` needs registers for tiling state)
    /// - Quantized kernels (dequantization requires intermediate values)
    ///
    /// These are documented performance tradeoffs, not bugs.
    #[must_use]
    pub fn with_performance_whitelist() -> Self {
        Self::new()
            // Tensor Core / WMMA kernels - high register usage is expected
            // WMMA m16n16k16 requires 8 registers per fragment × 3 fragments = 24+ registers
            // Plus accumulator, addresses, loop counters, etc.
            .with_whitelist(
                "gemm_tensor_core*",
                PtxBugClass::HighRegisterPressure,
                "Tensor Core WMMA requires many registers for matrix fragments",
            )
            .with_whitelist(
                "gemm_tensor_core*",
                PtxBugClass::PredicateOverflow,
                "Tensor Core kernels use predicates for bounds checking and masking",
            )
            .with_whitelist(
                "gemm_wmma*",
                PtxBugClass::HighRegisterPressure,
                "WMMA FP16 requires registers for A/B/C/D matrix fragments",
            )
            .with_whitelist(
                "gemm_wmma*",
                PtxBugClass::PredicateOverflow,
                "WMMA kernels use predicates for tile boundary handling",
            )
            // Attention kernels - FlashAttention tiling requires state
            .with_whitelist(
                "flash_attention*",
                PtxBugClass::HighRegisterPressure,
                "FlashAttention tiling requires registers for Q/K/V/O tiles and softmax state",
            )
            .with_whitelist(
                "attention*",
                PtxBugClass::HighRegisterPressure,
                "Attention kernels require registers for Q/K/V tiles and reduction",
            )
            // Quantized kernels - dequantization math
            .with_whitelist(
                "q4k*",
                PtxBugClass::HighRegisterPressure,
                "Q4_K dequantization requires registers for scale/min extraction",
            )
            .with_whitelist(
                "q5k*",
                PtxBugClass::HighRegisterPressure,
                "Q5_K dequantization requires registers for 5-bit value reconstruction",
            )
            .with_whitelist(
                "q6k*",
                PtxBugClass::HighRegisterPressure,
                "Q6_K dequantization requires registers for 6-bit value reconstruction",
            )
            .with_whitelist(
                "q8k*",
                PtxBugClass::HighRegisterPressure,
                "Q8_K dequantization requires registers for scale application",
            )
    }

    /// Check if a bug should be suppressed by whitelist
    fn is_whitelisted(&self, kernel_name: Option<&String>, bug_class: &PtxBugClass) -> bool {
        let Some(kernel) = kernel_name else {
            return false;
        };

        for entry in &self.whitelist {
            if &entry.bug_class != bug_class {
                continue;
            }
            // Pattern matching: "q4k*" matches "q4k_gemm_ggml"
            if entry.kernel_pattern.ends_with('*') {
                let prefix = &entry.kernel_pattern[..entry.kernel_pattern.len() - 1];
                if kernel.starts_with(prefix) {
                    return true;
                }
            } else if &entry.kernel_pattern == kernel {
                return true;
            }
        }
        false
    }

    /// Analyze PTX for bugs
    #[must_use]
    pub fn analyze(&self, ptx: &str) -> PtxBugReport {
        let mut bugs = Vec::new();
        let lines: Vec<&str> = ptx.lines().collect();

        // Extract kernel name
        let kernel_name = self.extract_kernel_name(ptx);

        // Execute all pattern detectors
        bugs.extend(self.detect_shared_mem_u64(ptx, &lines));
        bugs.extend(self.detect_loop_branch_to_end(ptx, &lines));
        bugs.extend(self.detect_missing_barrier_sync(ptx, &lines));
        bugs.extend(self.detect_early_exit_before_barrier(ptx));
        bugs.extend(self.detect_register_spills(ptx, &lines));
        bugs.extend(self.detect_missing_entry_point(ptx, &lines));
        bugs.extend(self.detect_redundant_moves(ptx, &lines));
        bugs.extend(self.detect_unoptimized_memory(ptx, &lines));
        bugs.extend(self.detect_high_register_pressure(ptx, &lines));
        bugs.extend(self.detect_predicate_overflow(ptx, &lines));
        bugs.extend(self.detect_placeholder_code(ptx, &lines));
        // New extended detectors
        bugs.extend(self.detect_empty_loop_body(ptx, &lines));
        bugs.extend(self.detect_missing_bounds_check(ptx, &lines));
        bugs.extend(self.detect_dead_code(ptx, &lines));

        // Filter out whitelisted bugs
        bugs.retain(|bug| !self.is_whitelisted(kernel_name.as_ref(), &bug.class));

        PtxBugReport { kernel_name, bugs, lines_analyzed: lines.len(), strict_mode: self.strict }
    }

    /// Extract kernel name from PTX
    fn extract_kernel_name(&self, ptx: &str) -> Option<String> {
        let entry_pattern = Regex::new(r"\.(?:visible\s+)?\.entry\s+(\w+)")
            .expect("invariant: regex pattern is valid");
        entry_pattern
            .captures(ptx)
            .map(|c| c.get(1).expect("invariant: capture group 1 exists").as_str().to_string())
    }
}

mod detectors;
