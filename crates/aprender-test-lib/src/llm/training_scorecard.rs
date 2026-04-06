//! Training Step Scorecard — scientific profiler consumer for training runs.
//!
//! Parses entrenar's StepProfiler JSON output and produces graded efficiency
//! analysis with bottleneck identification and regression detection.
//!
//! Contract: training-step-scorecard-v1.yaml (PMAT-485)
//!
//! # Methodology
//!
//! - Hoefler & Belli SC'15: report median, CI, wall coverage
//! - Popperian falsification: 7 testable predictions
//! - Five-whys root cause: bottleneck classification maps to actionable fixes

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ─── Grade mapping ───

/// Efficiency grade (A through F).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Grade {
    F,
    D,
    C,
    B,
    A,
}

impl Grade {
    /// Map efficiency score [0.0, 1.0] to grade.
    #[must_use]
    pub fn from_efficiency(eff: f64) -> Self {
        if eff >= 0.60 {
            Self::A
        } else if eff >= 0.40 {
            Self::B
        } else if eff >= 0.20 {
            Self::C
        } else if eff >= 0.10 {
            Self::D
        } else {
            Self::F
        }
    }

    /// Grade as letter string.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::A => "A",
            Self::B => "B",
            Self::C => "C",
            Self::D => "D",
            Self::F => "F",
        }
    }
}

impl std::fmt::Display for Grade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ─── Bottleneck classification ───

/// Training bottleneck classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Bottleneck {
    /// Host-device transfer dominates (>30% of step time).
    Transfer,
    /// Kernel launch overhead dominates (>40% unaccounted time).
    Launch,
    /// GPU ALU utilization high (>50%). This is the GOOD state.
    Compute,
    /// Memory bandwidth bound (default).
    MemoryBw,
}

impl Bottleneck {
    /// Classify bottleneck from per-operation profiling data.
    #[must_use]
    pub fn classify(transfer_pct: f64, launch_overhead: f64, compute_util: f64) -> Self {
        if transfer_pct > 0.30 {
            Self::Transfer
        } else if launch_overhead > 0.40 {
            Self::Launch
        } else if compute_util > 0.50 {
            Self::Compute
        } else {
            Self::MemoryBw
        }
    }

    /// Human-readable recommendation for this bottleneck.
    #[must_use]
    pub fn recommendation(&self) -> &'static str {
        match self {
            Self::Transfer => "Reduce host-device transfers. Check H2D/D2H in profiler.",
            Self::Launch => "Enable kernel fusion (NF4_FUSED_GEMM=1, NF4_TC_BWD_GEMM=1).",
            Self::Compute => "GPU ALU bound — good! Consider algorithmic improvements.",
            Self::MemoryBw => "Enable tensor cores (NF4_TC_GEMM=1) or FP16 (FP16_GEMM=1).",
        }
    }
}

impl std::fmt::Display for Bottleneck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transfer => f.write_str("transfer"),
            Self::Launch => f.write_str("launch"),
            Self::Compute => f.write_str("compute"),
            Self::MemoryBw => f.write_str("memory_bw"),
        }
    }
}

// ─── Per-layer summary ───

/// Per-layer profiling summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSummary {
    pub layer: usize,
    pub fwd_ms: f64,
    pub bwd_ms: f64,
    pub ratio: f64,
    pub top_op: String,
    pub top_op_pct: f64,
}

// ─── Regression detection ───

/// A detected regression in a metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Regression {
    pub metric: String,
    pub delta: f64,
    pub severity: String,
}

// ─── Scorecard output ───

/// Training step scorecard — the primary output type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingScorecard {
    pub grade: String,
    pub efficiency: f64,
    pub bottleneck: Bottleneck,
    pub throughput_tok_s: f64,
    pub step_time_ms: f64,
    pub forward_backward_ratio: f64,
    pub wall_coverage: f64,
    pub per_layer_summary: Vec<LayerSummary>,
    pub hotspot_ops: Vec<(String, f64, f64)>, // (op_name, total_ms, pct)
    pub regressions: Vec<Regression>,
    pub recommendations: Vec<String>,
}

// ─── StepProfiler JSON input ───

/// Phase timing from entrenar StepProfiler.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PhaseData {
    pub total_ms: f64,
    pub pct: f64,
    pub avg_ms: f64,
}

/// Per-layer timing from entrenar StepProfiler.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerLayerData {
    pub layer: usize,
    pub fwd_ms: f64,
    pub bwd_ms: f64,
    #[serde(default)]
    pub ops: BTreeMap<String, f64>,
}

/// Parsed StepProfiler JSON from entrenar.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StepProfilerInput {
    #[serde(default)]
    pub steps: usize,
    #[serde(default)]
    pub avg_step_ms: f64,
    #[serde(default)]
    pub wall_coverage: f64,
    #[serde(default)]
    pub phases: BTreeMap<String, PhaseData>,
    #[serde(default)]
    pub per_layer: Vec<PerLayerData>,
    #[serde(default)]
    pub bottleneck: Option<String>,
    #[serde(default)]
    pub gemm_pct: f64,
    // Legacy text-parsed fields
    #[serde(default)]
    pub total_ms: f64,
}

/// Hardware roofline parameters.
#[derive(Debug, Clone)]
pub struct HardwareSpec {
    /// Peak memory bandwidth in GB/s.
    pub peak_bw_gb_s: f64,
    /// Bytes per token (model-dependent).
    pub bytes_per_token: f64,
}

impl HardwareSpec {
    /// RTX 4060 Laptop (yoga) for Qwen 1.5B NF4 training.
    ///
    /// Derived from measured roofline: 256 GB/s DRAM BW, Qwen 1.5B NF4 training
    /// reads ~158 MB of weight data per effective token (NF4 packed weights ×
    /// forward + backward passes, amortized over batch). At peak BW this gives
    /// ~1620 tok/s theoretical, making 194 tok/s ≈ 12% efficiency (matches
    /// measured 12.2% from profile-yoga-20260402.json).
    #[must_use]
    pub fn rtx_4060l() -> Self {
        Self {
            peak_bw_gb_s: 256.0,
            bytes_per_token: 158_000_000.0, // ~158 MB/token (NF4 fwd+bwd amortized)
        }
    }

    /// GB10 (gx10) for Qwen 1.5B NF4 training.
    #[must_use]
    pub fn gb10() -> Self {
        Self {
            peak_bw_gb_s: 273.0,
            bytes_per_token: 158_000_000.0,
        }
    }

    /// Peak throughput estimate (memory-bound).
    #[must_use]
    pub fn peak_throughput_tok_s(&self) -> f64 {
        (self.peak_bw_gb_s * 1e9) / self.bytes_per_token
    }
}

// ─── Scorecard computation ───

/// Compute training scorecard from StepProfiler JSON.
///
/// # Arguments
/// - `input`: Parsed StepProfiler JSON
/// - `throughput_tok_s`: Measured throughput in tokens/second
/// - `hw`: Hardware specification for roofline analysis
/// - `baseline`: Optional baseline scorecard for regression detection
#[must_use]
pub fn compute_training_scorecard(
    input: &StepProfilerInput,
    throughput_tok_s: f64,
    hw: &HardwareSpec,
    baseline: Option<&TrainingScorecard>,
) -> TrainingScorecard {
    // Efficiency
    let peak = hw.peak_throughput_tok_s();
    let efficiency = if peak > 0.0 {
        (throughput_tok_s / peak).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let grade = Grade::from_efficiency(efficiency);

    // Bottleneck classification
    let transfer_pct = transfer_percentage(input);
    let launch_overhead = launch_overhead_percentage(input);
    let compute_util = efficiency; // Simplified: efficiency ≈ compute utilization
    let bottleneck = Bottleneck::classify(transfer_pct, launch_overhead, compute_util);

    // Forward/backward ratio
    let (avg_ratio, per_layer_summary) = compute_layer_summaries(input);

    // Hotspot ops
    let hotspot_ops = compute_hotspot_ops(input);

    // Regression detection
    let regressions = if let Some(base) = baseline {
        detect_regressions(base, throughput_tok_s, input)
    } else {
        vec![]
    };

    // Recommendations
    let mut recommendations = vec![];
    if grade <= Grade::C {
        recommendations.push(bottleneck.recommendation().to_string());
    }
    if grade <= Grade::D && launch_overhead > 0.30 {
        recommendations.push(
            "Kernel launch overhead >30%. Enable NF4_TC_BWD_GEMM=1 to eliminate 196 launches/step."
                .to_string(),
        );
    }
    if avg_ratio < 1.0 && avg_ratio > 0.0 {
        recommendations.push(
            "Backward faster than forward — possible NaN backward skips. Check PMAT-462."
                .to_string(),
        );
    }

    TrainingScorecard {
        grade: grade.to_string(),
        efficiency,
        bottleneck,
        throughput_tok_s,
        step_time_ms: input.avg_step_ms,
        forward_backward_ratio: avg_ratio,
        wall_coverage: input.wall_coverage,
        per_layer_summary,
        hotspot_ops,
        regressions,
        recommendations,
    }
}

fn transfer_percentage(input: &StepProfilerInput) -> f64 {
    let mut transfer = 0.0;
    for (name, phase) in &input.phases {
        if name.contains("h2d") || name.contains("d2h") || name.contains("grad_h2d") {
            transfer += phase.pct;
        }
    }
    transfer / 100.0
}

fn launch_overhead_percentage(input: &StepProfilerInput) -> f64 {
    let total_ops: f64 = input.phases.values().map(|p| p.total_ms).sum();
    let step = input.avg_step_ms;
    if step > 0.0 {
        1.0 - (total_ops / step).min(1.0)
    } else {
        0.0
    }
}

fn compute_layer_summaries(input: &StepProfilerInput) -> (f64, Vec<LayerSummary>) {
    let mut summaries = vec![];
    let mut ratio_sum = 0.0;
    let mut ratio_count = 0;

    for layer in &input.per_layer {
        let ratio = if layer.fwd_ms > 0.0 {
            layer.bwd_ms / layer.fwd_ms
        } else {
            0.0
        };

        // Find top op
        let (top_op, top_ms) = layer
            .ops
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, v)| (k.clone(), *v))
            .unwrap_or_default();

        let total_ops: f64 = layer.ops.values().sum();
        let top_pct = if total_ops > 0.0 {
            top_ms / total_ops * 100.0
        } else {
            0.0
        };

        if ratio > 0.0 && layer.bwd_ms > 0.0 {
            ratio_sum += ratio;
            ratio_count += 1;
        }

        summaries.push(LayerSummary {
            layer: layer.layer,
            fwd_ms: layer.fwd_ms,
            bwd_ms: layer.bwd_ms,
            ratio,
            top_op,
            top_op_pct: top_pct,
        });
    }

    let avg_ratio = if ratio_count > 0 {
        ratio_sum / ratio_count as f64
    } else {
        0.0
    };

    (avg_ratio, summaries)
}

fn compute_hotspot_ops(input: &StepProfilerInput) -> Vec<(String, f64, f64)> {
    let mut op_totals: BTreeMap<String, f64> = BTreeMap::new();
    for layer in &input.per_layer {
        for (op, ms) in &layer.ops {
            *op_totals.entry(op.clone()).or_default() += ms;
        }
    }

    let total: f64 = op_totals.values().sum();
    let mut ops: Vec<_> = op_totals
        .into_iter()
        .map(|(name, ms)| {
            let pct = if total > 0.0 { ms / total * 100.0 } else { 0.0 };
            (name, ms, pct)
        })
        .collect();

    ops.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ops.truncate(10);
    ops
}

fn detect_regressions(
    baseline: &TrainingScorecard,
    throughput: f64,
    input: &StepProfilerInput,
) -> Vec<Regression> {
    let mut regressions = vec![];

    // Throughput regression (10% threshold)
    if baseline.throughput_tok_s > 0.0 {
        let delta = (throughput - baseline.throughput_tok_s) / baseline.throughput_tok_s;
        if delta < -0.10 {
            regressions.push(Regression {
                metric: "throughput_tok_s".to_string(),
                delta,
                severity: if delta < -0.20 { "critical" } else { "warning" }.to_string(),
            });
        }
    }

    // Step time regression (10% threshold)
    if baseline.step_time_ms > 0.0 && input.avg_step_ms > 0.0 {
        let delta = (input.avg_step_ms - baseline.step_time_ms) / baseline.step_time_ms;
        if delta > 0.10 {
            regressions.push(Regression {
                metric: "step_time_ms".to_string(),
                delta,
                severity: if delta > 0.20 { "critical" } else { "warning" }.to_string(),
            });
        }
    }

    // Wall coverage regression (5% threshold)
    if baseline.wall_coverage > 0.0 && input.wall_coverage > 0.0 {
        let delta = input.wall_coverage - baseline.wall_coverage;
        if delta < -0.05 {
            regressions.push(Regression {
                metric: "wall_coverage".to_string(),
                delta,
                severity: "warning".to_string(),
            });
        }
    }

    regressions
}

/// Format scorecard as Markdown table.
#[must_use]
pub fn format_training_scorecard_markdown(sc: &TrainingScorecard) -> String {
    let mut s = String::new();
    s.push_str("## Training Scorecard\n\n");
    s.push_str("| Metric | Value |\n|--------|-------|\n");
    s.push_str(&format!("| Grade | **{}** |\n", sc.grade));
    s.push_str(&format!("| Efficiency | {:.1}% |\n", sc.efficiency * 100.0));
    s.push_str(&format!("| Bottleneck | {} |\n", sc.bottleneck));
    s.push_str(&format!(
        "| Throughput | {:.0} tok/s |\n",
        sc.throughput_tok_s
    ));
    s.push_str(&format!("| Step time | {:.1} ms |\n", sc.step_time_ms));
    s.push_str(&format!(
        "| Fwd/Bwd ratio | {:.2} |\n",
        sc.forward_backward_ratio
    ));
    s.push_str(&format!(
        "| Wall coverage | {:.1}% |\n",
        sc.wall_coverage * 100.0
    ));

    if !sc.hotspot_ops.is_empty() {
        s.push_str("\n### Hotspot Operations\n\n");
        s.push_str("| Op | Total ms | % |\n|-----|---------|---|\n");
        for (op, ms, pct) in &sc.hotspot_ops {
            s.push_str(&format!("| {} | {:.1} | {:.1}% |\n", op, ms, pct));
        }
    }

    if !sc.regressions.is_empty() {
        s.push_str("\n### Regressions\n\n");
        for r in &sc.regressions {
            s.push_str(&format!(
                "- **{}**: {:.1}% ({}) \n",
                r.metric,
                r.delta * 100.0,
                r.severity
            ));
        }
    }

    if !sc.recommendations.is_empty() {
        s.push_str("\n### Recommendations\n\n");
        for r in &sc.recommendations {
            s.push_str(&format!("- {r}\n"));
        }
    }

    s
}

// ─── Parity comparison (PMAT-487) ───

/// Parity profile from a single runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityProfile {
    pub runtime: String,
    pub throughput_tok_s: f64,
    pub step_time_ms: f64,
    pub kernel_launches_per_step: usize,
    pub ops: BTreeMap<String, f64>, // op_name → mean_ms
}

/// Parity gap between APR and a baseline runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityGap {
    pub op: String,
    pub apr_ms: f64,
    pub baseline_ms: f64,
    pub ratio: f64, // apr_ms / baseline_ms (>1 = APR slower)
}

/// Parity comparison result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityComparison {
    pub apr_runtime: String,
    pub baseline_runtime: String,
    pub throughput_ratio: f64, // baseline_tok_s / apr_tok_s
    pub launch_ratio: f64,     // apr_launches / baseline_launches
    pub top_gaps: Vec<ParityGap>,
    pub parity_achieved: bool, // all ops within 10%
}

/// Compare APR against a baseline runtime for parity analysis.
///
/// Identifies the top operation gaps (where APR is slowest relative to baseline)
/// and determines whether parity is achieved (all ops within 10%).
#[must_use]
pub fn compute_parity_comparison(
    apr: &ParityProfile,
    baseline: &ParityProfile,
) -> ParityComparison {
    let throughput_ratio = if apr.throughput_tok_s > 0.0 {
        baseline.throughput_tok_s / apr.throughput_tok_s
    } else {
        0.0
    };

    let launch_ratio = if baseline.kernel_launches_per_step > 0 {
        apr.kernel_launches_per_step as f64 / baseline.kernel_launches_per_step as f64
    } else {
        0.0
    };

    // Find per-op gaps
    let mut gaps = Vec::new();
    for (op, &apr_ms) in &apr.ops {
        let baseline_ms = baseline.ops.get(op).copied().unwrap_or(0.0);
        if apr_ms > 0.0 && baseline_ms > 0.0 {
            let ratio = apr_ms / baseline_ms;
            gaps.push(ParityGap {
                op: op.clone(),
                apr_ms,
                baseline_ms,
                ratio,
            });
        }
    }

    gaps.sort_by(|a, b| {
        b.ratio
            .partial_cmp(&a.ratio)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let parity_achieved = gaps.iter().all(|g| g.ratio <= 1.10);
    gaps.truncate(5); // top 5

    ParityComparison {
        apr_runtime: apr.runtime.clone(),
        baseline_runtime: baseline.runtime.clone(),
        throughput_ratio,
        launch_ratio,
        top_gaps: gaps,
        parity_achieved,
    }
}

/// Format parity comparison as Markdown.
#[must_use]
pub fn format_parity_markdown(comp: &ParityComparison) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "## Parity: {} vs {}\n\n",
        comp.apr_runtime, comp.baseline_runtime
    ));
    s.push_str(&format!(
        "- **Throughput gap**: {:.1}x ({} is {:.1}x faster)\n",
        comp.throughput_ratio, comp.baseline_runtime, comp.throughput_ratio
    ));
    s.push_str(&format!(
        "- **Kernel launches**: {:.1}x (APR launches {:.1}x more)\n",
        comp.launch_ratio, comp.launch_ratio
    ));
    s.push_str(&format!(
        "- **Parity achieved**: {}\n",
        if comp.parity_achieved { "YES" } else { "NO" }
    ));

    if !comp.top_gaps.is_empty() {
        s.push_str("\n### Top Gaps\n\n");
        s.push_str("| Op | APR (ms) | Baseline (ms) | Ratio |\n");
        s.push_str("|-----|---------|--------------|-------|\n");
        for g in &comp.top_gaps {
            s.push_str(&format!(
                "| {} | {:.1} | {:.1} | {:.1}x |\n",
                g.op, g.apr_ms, g.baseline_ms, g.ratio
            ));
        }
    }

    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grade_from_efficiency() {
        assert_eq!(Grade::from_efficiency(0.65), Grade::A);
        assert_eq!(Grade::from_efficiency(0.45), Grade::B);
        assert_eq!(Grade::from_efficiency(0.25), Grade::C);
        assert_eq!(Grade::from_efficiency(0.12), Grade::D);
        assert_eq!(Grade::from_efficiency(0.05), Grade::F);
    }

    #[test]
    fn test_grade_monotonic() {
        // F-TSC-001: Grade monotonically non-decreasing with efficiency
        let mut prev = Grade::F;
        for eff in (0..=100).map(|i| i as f64 / 100.0) {
            let grade = Grade::from_efficiency(eff);
            assert!(grade >= prev, "Grade decreased at efficiency {eff}");
            prev = grade;
        }
    }

    #[test]
    fn test_bottleneck_classification() {
        // F-TSC-002: Known classifications
        assert_eq!(Bottleneck::classify(0.35, 0.10, 0.10), Bottleneck::Transfer);
        assert_eq!(Bottleneck::classify(0.10, 0.50, 0.10), Bottleneck::Launch);
        assert_eq!(Bottleneck::classify(0.10, 0.10, 0.60), Bottleneck::Compute);
        assert_eq!(Bottleneck::classify(0.10, 0.10, 0.10), Bottleneck::MemoryBw);
    }

    #[test]
    fn test_bottleneck_mutually_exclusive() {
        // Every possible input produces exactly one classification
        for t in [0.0, 0.15, 0.31] {
            for l in [0.0, 0.20, 0.41] {
                for c in [0.0, 0.25, 0.51] {
                    let _b = Bottleneck::classify(t, l, c);
                    // No panic = exactly one classification
                }
            }
        }
    }

    #[test]
    fn test_scorecard_yoga_profile() {
        // F-TSC-001: Yoga APR profile at 194 tok/s, 12.2% efficiency → grade D
        let input = StepProfilerInput {
            avg_step_ms: 105.0,
            wall_coverage: 0.85,
            ..Default::default()
        };
        let hw = HardwareSpec::rtx_4060l();
        let sc = compute_training_scorecard(&input, 194.0, &hw, None);

        assert_eq!(sc.grade, "D");
        assert!(sc.efficiency < 0.20);
        assert!(sc.efficiency > 0.0);
    }

    #[test]
    fn test_regression_detection() {
        // F-TSC-003: 15% throughput regression triggers alert
        let baseline = TrainingScorecard {
            grade: "D".to_string(),
            efficiency: 0.12,
            bottleneck: Bottleneck::MemoryBw,
            throughput_tok_s: 200.0,
            step_time_ms: 100.0,
            forward_backward_ratio: 2.0,
            wall_coverage: 0.90,
            per_layer_summary: vec![],
            hotspot_ops: vec![],
            regressions: vec![],
            recommendations: vec![],
        };

        let input = StepProfilerInput {
            avg_step_ms: 115.0,
            wall_coverage: 0.88,
            ..Default::default()
        };

        let sc =
            compute_training_scorecard(&input, 170.0, &HardwareSpec::rtx_4060l(), Some(&baseline));
        assert!(
            !sc.regressions.is_empty(),
            "Should detect 15% throughput regression"
        );
        assert_eq!(sc.regressions[0].metric, "throughput_tok_s");
    }

    #[test]
    fn test_no_false_regression() {
        // 5% improvement should NOT trigger regression
        let baseline = TrainingScorecard {
            grade: "D".to_string(),
            efficiency: 0.12,
            bottleneck: Bottleneck::MemoryBw,
            throughput_tok_s: 200.0,
            step_time_ms: 100.0,
            forward_backward_ratio: 2.0,
            wall_coverage: 0.90,
            per_layer_summary: vec![],
            hotspot_ops: vec![],
            regressions: vec![],
            recommendations: vec![],
        };

        let input = StepProfilerInput {
            avg_step_ms: 95.0,
            wall_coverage: 0.91,
            ..Default::default()
        };

        let sc =
            compute_training_scorecard(&input, 210.0, &HardwareSpec::rtx_4060l(), Some(&baseline));
        assert!(
            sc.regressions.is_empty(),
            "5% improvement should not regress"
        );
    }

    #[test]
    fn test_scorecard_json_roundtrip() {
        // F-TSC-005: Scorecard serializes to valid JSON
        let input = StepProfilerInput::default();
        let sc = compute_training_scorecard(&input, 100.0, &HardwareSpec::rtx_4060l(), None);
        let json = serde_json::to_string(&sc).unwrap();
        let _: TrainingScorecard = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_recommendations_for_low_grade() {
        // F-TSC-007: Grade D with "launch" bottleneck recommends fusion
        let mut phases = BTreeMap::new();
        phases.insert(
            "forward".to_string(),
            PhaseData {
                total_ms: 10.0,
                pct: 10.0,
                avg_ms: 10.0,
            },
        );
        let input = StepProfilerInput {
            avg_step_ms: 200.0, // Large gap → launch overhead
            phases,
            ..Default::default()
        };
        let sc = compute_training_scorecard(&input, 50.0, &HardwareSpec::rtx_4060l(), None);
        assert!(!sc.recommendations.is_empty());
    }
}
