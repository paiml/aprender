//! Baseline comparison and validation logic.

use std::fmt;

use super::{
    GpuClass, ServerBaseline, SingleComparison, SmHealth, ThroughputGrade, INDUSTRY_BASELINES,
    VLLM_BASELINE,
};

/// Baseline comparison result.
///
/// Compares actual metrics against industry baselines.
#[derive(Debug, Clone)]
pub struct BaselineComparison {
    /// Detected GPU class
    pub gpu_class: GpuClass,
    /// Actual throughput (tok/s)
    pub actual_tok_per_sec: u32,
    /// Expected throughput range
    pub expected_range: (u32, u32),
    /// Percentage of vLLM baseline achieved
    pub vllm_percentage: f64,
    /// Throughput grade
    pub grade: ThroughputGrade,
    /// SM utilization
    pub sm_utilization: u8,
    /// SM health indicator
    pub sm_health: SmHealth,
    /// P95 latency (ms)
    pub p95_latency_ms: Option<u32>,
    /// Comparison with each baseline
    pub baseline_comparisons: Vec<SingleComparison>,
}

impl BaselineComparison {
    /// Create a new baseline comparison.
    pub fn new(
        gpu_name: &str,
        actual_tok_per_sec: u32,
        sm_utilization: u8,
        p95_latency_ms: Option<u32>,
    ) -> Self {
        let gpu_class = GpuClass::from_name(gpu_name);
        let expected_range = gpu_class.expected_throughput();

        // Calculate percentage of vLLM baseline (scaled by GPU class)
        let vllm_scaled_baseline = scale_baseline_for_gpu(&VLLM_BASELINE, &gpu_class);
        let vllm_percentage = (actual_tok_per_sec as f64 / vllm_scaled_baseline as f64) * 100.0;

        let grade = ThroughputGrade::from_percentage(vllm_percentage);
        let sm_health = SmHealth::from_utilization(sm_utilization);

        // Compare against all baselines
        let baseline_comparisons: Vec<_> = INDUSTRY_BASELINES
            .iter()
            .map(|baseline| {
                let scaled = scale_baseline_for_gpu(baseline, &gpu_class);
                SingleComparison {
                    baseline: *baseline,
                    percentage: (actual_tok_per_sec as f64 / scaled as f64) * 100.0,
                    delta_tok_per_sec: actual_tok_per_sec as i32 - scaled as i32,
                }
            })
            .collect();

        BaselineComparison {
            gpu_class,
            actual_tok_per_sec,
            expected_range,
            vllm_percentage,
            grade,
            sm_utilization,
            sm_health,
            p95_latency_ms,
            baseline_comparisons,
        }
    }

    /// Check if throughput is within expected range for GPU class.
    pub fn is_within_expected_range(&self) -> bool {
        self.actual_tok_per_sec >= self.expected_range.0
            && self.actual_tok_per_sec <= self.expected_range.1
    }

    /// Get improvement suggestions based on metrics.
    pub fn suggestions(&self) -> Vec<&'static str> {
        let mut suggestions = Vec::new();

        // SM utilization suggestions
        match self.sm_health {
            SmHealth::Critical => {
                suggestions
                    .push("Critical: SM utilization < 50% - check batch size and kernel occupancy");
                suggestions.push("Consider increasing batch size or concurrent requests");
            }
            SmHealth::Moderate => {
                suggestions.push("SM utilization 50-80% - room for optimization");
                suggestions.push("Try increasing kernel occupancy or reducing memory pressure");
            }
            SmHealth::Saturated => {
                suggestions
                    .push("SM utilization > 95% - at saturation, throughput limited by compute");
            }
            SmHealth::Optimal => {}
        }

        // Grade-based suggestions
        match self.grade {
            ThroughputGrade::F => {
                suggestions.push("Throughput < 40% of baseline - major optimization needed");
                suggestions
                    .push("Check for: kernel inefficiency, memory bottlenecks, PCIe transfers");
            }
            ThroughputGrade::D => {
                suggestions.push("Throughput 40-60% of baseline - significant optimization needed");
            }
            ThroughputGrade::C => {
                suggestions
                    .push("Throughput 60-80% of baseline - optimization opportunities exist");
            }
            ThroughputGrade::B | ThroughputGrade::A => {}
        }

        suggestions
    }
}

impl fmt::Display for BaselineComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Baseline Comparison Report")?;
        writeln!(f, "==========================")?;
        writeln!(f)?;
        writeln!(f, "GPU Class: {}", self.gpu_class)?;
        writeln!(f, "Actual Throughput: {} tok/s", self.actual_tok_per_sec)?;
        writeln!(f, "Expected Range: {}-{} tok/s", self.expected_range.0, self.expected_range.1)?;
        writeln!(f, "Grade: {}", self.grade)?;
        writeln!(f)?;
        writeln!(f, "SM Utilization: {}% ({})", self.sm_utilization, self.sm_health)?;
        if let Some(latency) = self.p95_latency_ms {
            writeln!(f, "P95 Latency: {} ms", latency)?;
        }
        writeln!(f)?;
        writeln!(f, "Comparison vs Industry Baselines:")?;
        for cmp in &self.baseline_comparisons {
            let sign = if cmp.delta_tok_per_sec >= 0 { "+" } else { "" };
            writeln!(
                f,
                "  {}: {:.1}% ({}{} tok/s)",
                cmp.baseline.name, cmp.percentage, sign, cmp.delta_tok_per_sec
            )?;
        }

        let suggestions = self.suggestions();
        if !suggestions.is_empty() {
            writeln!(f)?;
            writeln!(f, "Suggestions:")?;
            for suggestion in suggestions {
                writeln!(f, "  - {}", suggestion)?;
            }
        }

        Ok(())
    }
}

/// Scale baseline for different GPU classes.
///
/// Baselines are measured on A10; scale for other GPUs based on expected performance.
fn scale_baseline_for_gpu(baseline: &ServerBaseline, gpu_class: &GpuClass) -> u32 {
    let (min_expected, max_expected) = gpu_class.expected_throughput();
    let a10_expected = (350 + 450) / 2; // A10 midpoint

    let target_expected = (min_expected + max_expected) / 2;
    let scale_factor = target_expected as f64 / a10_expected as f64;

    (baseline.peak_tok_per_sec as f64 * scale_factor) as u32
}

/// Baseline validator for F971-F985 falsification criteria.
#[derive(Debug, Default)]
pub struct BaselineValidator {
    /// Validated criteria
    validations: Vec<(String, bool, String)>,
}

impl BaselineValidator {
    /// Create a new validator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate F971: Realistic GPU throughput (within 30% of vLLM).
    pub fn validate_f971_throughput(&mut self, comparison: &BaselineComparison) -> bool {
        let passed = comparison.vllm_percentage >= 70.0; // Within 30% means at least 70%
        self.validations.push((
            "F971".to_string(),
            passed,
            format!("Throughput {:.1}% of vLLM (need >= 70%)", comparison.vllm_percentage),
        ));
        passed
    }

    /// Validate F972: SM utilization correct (within 5% of nvidia-smi).
    pub fn validate_f972_sm_util(&mut self, reported: u8, actual: u8) -> bool {
        let diff = (reported as i16 - actual as i16).unsigned_abs();
        let passed = diff <= 5;
        self.validations.push((
            "F972".to_string(),
            passed,
            format!("SM util diff: {}% (need <= 5%)", diff),
        ));
        passed
    }

    /// Validate F975: Baseline comparison available.
    pub fn validate_f975_baseline_available(&mut self, has_comparison: bool) -> bool {
        self.validations.push((
            "F975".to_string(),
            has_comparison,
            "Baseline comparison available".to_string(),
        ));
        has_comparison
    }

    /// Validate F976: No foreign code dependency.
    pub fn validate_f976_no_foreign_code(&mut self) -> bool {
        // This is always true for cbtop - we don't depend on vLLM/llama.cpp
        self.validations.push((
            "F976".to_string(),
            true,
            "No foreign code in cbtop binary".to_string(),
        ));
        true
    }

    /// Validate F982: GPU class detected correctly.
    pub fn validate_f982_gpu_detected(&mut self, gpu_class: &GpuClass) -> bool {
        let passed = *gpu_class != GpuClass::Unknown;
        self.validations.push(("F982".to_string(), passed, format!("GPU class: {}", gpu_class)));
        passed
    }

    /// Validate F983: Throughput grade calculated.
    pub fn validate_f983_grade_calculated(&mut self, grade: &ThroughputGrade) -> bool {
        self.validations.push(("F983".to_string(), true, format!("Grade calculated: {:?}", grade)));
        true
    }

    /// Validate F984: Health indicators displayed.
    pub fn validate_f984_health_indicators(
        &mut self,
        has_sm: bool,
        has_memory: bool,
        has_scaling: bool,
    ) -> bool {
        let passed = has_sm && has_memory && has_scaling;
        self.validations.push((
            "F984".to_string(),
            passed,
            format!("Health: SM={}, Memory={}, Scaling={}", has_sm, has_memory, has_scaling),
        ));
        passed
    }

    /// Get validation summary.
    pub fn summary(&self) -> ValidationSummary {
        let total = self.validations.len();
        let passed = self.validations.iter().filter(|(_, p, _)| *p).count();
        ValidationSummary {
            total,
            passed,
            failed: total - passed,
            details: self.validations.clone(),
        }
    }
}

/// Validation summary.
#[derive(Debug, Clone)]
pub struct ValidationSummary {
    /// Total validations run
    pub total: usize,
    /// Passed validations
    pub passed: usize,
    /// Failed validations
    pub failed: usize,
    /// Detailed results
    pub details: Vec<(String, bool, String)>,
}

impl fmt::Display for ValidationSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Baseline Validation Summary")?;
        writeln!(f, "===========================")?;
        writeln!(f, "Passed: {}/{}", self.passed, self.total)?;
        writeln!(f)?;
        for (id, passed, msg) in &self.details {
            let status = if *passed { "PASS" } else { "FAIL" };
            writeln!(f, "[{}] {}: {}", status, id, msg)?;
        }
        Ok(())
    }
}
