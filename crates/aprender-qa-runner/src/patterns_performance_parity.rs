
// Shared builders — accept pre-computed description to avoid repeated if/else.

/// Constructor for PerformanceCheckResult
impl PerformanceCheckResult {
    /// Create a new performance check result from gate, measurements, and description
    fn new(gate: SpecGate, passed: bool, measured: f64, threshold: f64, description: String) -> Self {
        Self { gate_id: gate.id().to_string(), passed, measured, threshold, description }
    }
}

/// Constructor for ParityCheckResult
impl ParityCheckResult {
    /// Create a new parity check result from gate, diff metrics, and description
    fn new(gate: SpecGate, passed: bool, max_diff: f64, threshold: f64, description: String) -> Self {
        Self { gate_id: gate.id().to_string(), passed, max_diff, threshold, description }
    }
}

/// Constructor for IntegrityCheckResult
impl IntegrityCheckResult {
    /// Create a new integrity check result from gate, pass status, and evidence
    fn new(gate: SpecGate, passed: bool, description: String, evidence: Option<String>) -> Self {
        Self { gate_id: gate.id().to_string(), passed, description, evidence }
    }
}

/// Generic threshold-based performance check (higher_is_better=true → pass when val >= threshold).
fn perf_threshold_check(
    gate: SpecGate, label: &str, value: f64, threshold: f64, higher_is_better: bool,
) -> PerformanceCheckResult {
    let passed = if higher_is_better { value >= threshold } else { value <= threshold };
    let cmp = ["<", ">=", ">", "<="];
    let idx = usize::from(higher_is_better) + (usize::from(!passed) * 2);
    let suffix = if passed { "" } else { " (threshold exceeded)" };
    PerformanceCheckResult::new(
        gate, passed, value, threshold,
        format!("{label} {value:.1} {} {threshold:.1}{suffix}", cmp[idx.min(3)]),
    )
}

/// Performance validator
pub struct PerformanceValidator;

/// Performance validation checks for F-PERF gate IDs
impl PerformanceValidator {
    /// F-PERF-001: Check minimum TPS
    #[must_use]
    pub fn check_tps(measured_tps: f64, threshold: f64) -> PerformanceCheckResult {
        perf_threshold_check(SpecGate::PerfMinimumTps, "TPS", measured_tps, threshold, true)
    }

    /// F-PERF-002: Check time to first token
    #[must_use]
    pub fn check_ttft(ttft_ms: u64, max_ttft_ms: u64) -> PerformanceCheckResult {
        perf_threshold_check(SpecGate::PerfTtft, "TTFT(ms)", ttft_ms as f64, max_ttft_ms as f64, false)
    }

    /// F-PERF-003: Check memory leak (RSS growth over N requests)
    #[must_use]
    pub fn check_memory_leak(
        initial_rss_mb: f64,
        final_rss_mb: f64,
        max_growth_percent: f64,
    ) -> PerformanceCheckResult {
        let growth = if initial_rss_mb > 0.0 {
            ((final_rss_mb - initial_rss_mb) / initial_rss_mb) * 100.0
        } else {
            0.0
        };
        perf_threshold_check(SpecGate::PerfMemoryLeak, "Memory leak(%)", growth, max_growth_percent, false)
    }

    /// F-PERF-004: Check GPU utilization
    #[must_use]
    pub fn check_gpu_utilization(utilization: f64, min_utilization: f64) -> PerformanceCheckResult {
        perf_threshold_check(SpecGate::PerfGpuUtilization, "GPU util(%)", utilization, min_utilization, true)
    }
}

// ============================================================================
// CROSS-PLATFORM PARITY (F-PAR-001..003)
// ============================================================================

/// Result of parity check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityCheckResult {
    /// Gate ID
    pub gate_id: String,
    /// Whether check passed
    pub passed: bool,
    /// Maximum difference found
    pub max_diff: f64,
    /// Threshold for difference
    pub threshold: f64,
    /// Description
    pub description: String,
}

/// Cross-platform parity checker
pub struct ParityChecker;

/// Cross-platform parity checks for F-PAR gate IDs
impl ParityChecker {
    /// F-PAR-001: Check CPU/GPU equivalence
    #[must_use]
    pub fn check_cpu_gpu_equivalence(
        cpu_output: &[f32],
        gpu_output: &[f32],
        epsilon: f64,
    ) -> ParityCheckResult {
        let max_diff = cpu_output
            .iter()
            .zip(gpu_output.iter())
            .map(|(a, b)| f64::from((a - b).abs()))
            .fold(0.0f64, f64::max);
        let passed = max_diff <= epsilon;
        let status = ["MISMATCH", "OK"][usize::from(passed)];
        ParityCheckResult::new(
            SpecGate::ParCpuGpuEquivalence, passed, max_diff, epsilon,
            format!("CPU/GPU {status}: diff {max_diff:.2e} vs eps {epsilon:.2e}"),
        )
    }

    /// F-PAR-002: Check format parity (GGUF vs SafeTensors)
    #[must_use]
    pub fn check_format_parity(
        gguf_tokens: &[u32],
        safetensors_tokens: &[u32],
    ) -> ParityCheckResult {
        let diff_count = gguf_tokens
            .iter()
            .zip(safetensors_tokens.iter())
            .filter(|(a, b)| a != b)
            .count();
        let passed = diff_count == 0;
        ParityCheckResult::new(
            SpecGate::ParFormatParity, passed, diff_count as f64, 0.0,
            match diff_count {
                0 => "GGUF/SafeTensors output identical".to_string(),
                n => format!("{n} token differences found"),
            },
        )
    }

    /// F-PAR-003: Check quantization impact on perplexity
    #[must_use]
    pub fn check_quantization_impact(
        f16_perplexity: f64,
        quantized_perplexity: f64,
        max_degradation_percent: f64,
    ) -> ParityCheckResult {
        let degradation = if f16_perplexity > 0.0 {
            ((quantized_perplexity - f16_perplexity) / f16_perplexity) * 100.0
        } else {
            0.0
        };
        let passed = degradation <= max_degradation_percent;
        let verdict = ["EXCEEDED", "within bounds"][usize::from(passed)];
        ParityCheckResult::new(
            SpecGate::ParQuantizationImpact, passed, degradation, max_degradation_percent,
            format!("Perplexity degradation {degradation:.1}% {verdict} (max {max_degradation_percent}%)"),
        )
    }
}

// ============================================================================
// FUNDAMENTAL INTEGRITY CHECKS (F-INT-001..005)
// ============================================================================

/// Result of integrity check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityCheckResult {
    /// Gate ID
    pub gate_id: String,
    /// Whether check passed
    pub passed: bool,
    /// Description
    pub description: String,
    /// Evidence/details
    pub evidence: Option<String>,
}

/// Fundamental integrity checker
pub struct IntegrityChecker;

/// Signal-based error classification table.
const SIGNAL_ERRORS: &[(i32, i32, &str)] = &[
    (11, 139, "Segmentation fault detected"),    // SIGSEGV
    (7,  135, "Bus error detected"),              // SIGBUS
    (6,  134, "Abort signal detected"),           // SIGABRT
];

/// Stderr patterns indicating memory safety violations.
const MEMORY_STDERR_PATTERNS: &[&str] = &["SIGSEGV", "Segmentation fault", "buffer overflow", "stack smashing"];

/// Fundamental integrity checks for F-INT gate IDs
impl IntegrityChecker {
    /// F-INT-001: Check for memory safety violations
    #[must_use]
    pub fn check_memory_safety(exit_signal: Option<i32>, stderr: &str) -> IntegrityCheckResult {
        // Check signal-based errors via table
        let signal_error = exit_signal.and_then(|sig| {
            SIGNAL_ERRORS.iter().find(|&&(lo, hi, _)| sig == lo || sig == hi).map(|e| e.2)
        });
        let stderr_bad = MEMORY_STDERR_PATTERNS.iter().any(|pat| stderr.contains(pat));

        let passed = signal_error.is_none() && !stderr_bad;
        let desc = signal_error.unwrap_or(
            if stderr_bad { "Memory safety violation in stderr" } else { "No memory safety violations" }
        );
        IntegrityCheckResult::new(
            SpecGate::IntMemorySafety, passed, desc.to_string(),
            (!passed).then(|| format!("Signal: {exit_signal:?}")),
        )
    }

    /// F-INT-002: Check process termination
    #[must_use]
    pub fn check_process_termination(
        exit_code: Option<i32>,
        timed_out: bool,
        has_output: bool,
    ) -> IntegrityCheckResult {
        let clean_exit = exit_code == Some(0) && has_output;
        let error_exit = exit_code.is_some() && exit_code != Some(0);
        let passed = !timed_out && (clean_exit || (error_exit && has_output));

        // Classify failure reason via match (different from if/else chains above)
        let desc = match (timed_out, exit_code, has_output) {
            (true, _, _) => "Process timed out (hang detected)",
            (_, None, _) => "Zombie process (no exit code)",
            (_, Some(c), false) if c != 0 => "Unclean exit without error output",
            _ => "Clean process termination",
        };
        IntegrityCheckResult::new(
            SpecGate::IntProcessTermination, passed, desc.to_string(),
            exit_code.map(|c| format!("Exit code: {c}")),
        )
    }

    /// F-INT-003: Check tensor validity (delegates to PatternDetector)
    #[must_use]
    pub fn check_tensor_validity(values: &[f32]) -> IntegrityCheckResult {
        let r = PatternDetector::new().check_tensor_validity(values);
        // Classify failure via match on struct fields
        let desc = match (r.is_valid, r.nan_count > 0, r.inf_count > 0) {
            (true, _, _) => "Tensor values valid".to_string(),
            (_, true, _) => format!("Found {} NaN values", r.nan_count),
            (_, _, true) => format!("Found {} Inf values", r.inf_count),
            _ => "Tensor validation failed".to_string(),
        };
        IntegrityCheckResult::new(
            SpecGate::IntTensorValidity, r.is_valid, desc,
            Some(format!("NaN: {}, Inf: {}, Mean: {:.4}", r.nan_count, r.inf_count, r.mean)),
        )
    }

    /// F-INT-004: Check format fidelity (round-trip)
    #[must_use]
    pub fn check_format_fidelity(original_hash: &str, roundtrip_hash: &str) -> IntegrityCheckResult {
        let passed = original_hash == roundtrip_hash;
        let msgs = ["Round-trip conversion altered weights", "Round-trip conversion bitwise identical"];
        let evidence = (!passed).then(|| format!(
            "Original: {}, After: {}",
            &original_hash[..8.min(original_hash.len())],
            &roundtrip_hash[..8.min(roundtrip_hash.len())]
        ));
        IntegrityCheckResult::new(SpecGate::IntFormatFidelity, passed, msgs[usize::from(passed)].to_string(), evidence)
    }

    /// F-INT-005: Check determinism (same seed = same output)
    #[must_use]
    pub fn check_determinism(run1_output: &str, run2_output: &str, seed: u64) -> IntegrityCheckResult {
        let passed = run1_output == run2_output;
        let labels = ["Non-deterministic", "Deterministic"];
        let evidence = (!passed).then(|| {
            let pos = run1_output.chars().zip(run2_output.chars())
                .position(|(a, b)| a != b)
                .unwrap_or_else(|| run1_output.len().min(run2_output.len()));
            format!("First difference at position {pos}")
        });
        IntegrityCheckResult::new(
            SpecGate::IntDeterminism, passed,
            format!("{} output with seed {seed}", labels[usize::from(passed)]),
            evidence,
        )
    }
}

/// Result of tensor validity check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorValidityResult {
    /// Number of NaN values
    pub nan_count: usize,
    /// Number of Inf values
    pub inf_count: usize,
    /// Number of zero values
    pub zero_count: usize,
    /// Total number of values
    pub total: usize,
    /// Mean value
    pub mean: f64,
    /// Whether tensor is valid
    pub is_valid: bool,
}

/// Result of companion file check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompanionCheckResult {
    /// Missing companion files
    pub missing: Vec<String>,
    /// Found companion files
    pub found: Vec<String>,
    /// Whether all companions are present
    pub all_present: bool,
}

/// A path safety violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathViolation {
    /// The dangerous pattern found
    pub pattern: String,
    /// Description of the risk
    pub description: String,
}

/// Result of path safety check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathSafetyResult {
    /// Whether path is safe
    pub is_safe: bool,
    /// Violations found
    pub violations: Vec<PathViolation>,
}

/// A dangerous prompt pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptPattern {
    /// The pattern found
    pub pattern: String,
    /// Description of the risk
    pub description: String,
}

/// Result of prompt safety check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptSafetyResult {
    /// Whether prompt is safe
    pub is_safe: bool,
    /// Dangerous patterns found
    pub found_patterns: Vec<PromptPattern>,
}

#[cfg(test)]
#[path = "patterns_tests.rs"]
mod tests;
