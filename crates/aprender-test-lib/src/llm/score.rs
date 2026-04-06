//! Scoring engine for LLM inference runtime comparison.
//!
//! Computes weighted composite scores from `LoadTestResult` metrics using
//! absolute thresholds. Each metric gets a 0-100 score, combined into a
//! weighted composite with letter grade (A+ through F).
//!
//! Design: absolute thresholds (not relative best=100) so scores are stable
//! when runtimes are added/removed. Best-in-class gets a +5 bonus per metric.
//!
//! References:
//!   - Metron fluidity-index (arXiv 2407.07000) — jitter penalty on ITL
//!   - MLPerf Inference — per-scenario metrics
//!   - pmat perfection-score — weighted category aggregation

use super::loadtest::{LoadTestResult, TailAnalysis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Scoring contract (deserialized from YAML or built-in defaults)
// =============================================================================

/// A single metric threshold definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricThreshold {
    /// Value at which score = 100.
    pub excellent: f64,
    /// Value at which score = 75.
    pub good: f64,
    /// Whether higher values are better (true) or lower values are better (false).
    pub higher_is_better: bool,
}

/// Scoring contract with thresholds, weights, and grade boundaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringContract {
    /// Per-metric threshold definitions.
    pub thresholds: HashMap<String, MetricThreshold>,
    /// Interactive (c=1) weights: metric_name → weight (must sum to 1.0).
    pub interactive_weights: HashMap<String, f64>,
    /// Throughput (c>1) weights: metric_name → weight (must sum to 1.0).
    pub throughput_weights: HashMap<String, f64>,
    /// Best-in-class bonus points (capped at 100).
    pub best_in_class_bonus: u8,
    /// Grade boundaries: (min_score, grade_label).
    pub grades: Vec<(f64, String)>,
}

impl Default for ScoringContract {
    fn default() -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert(
            "decode_tok_s".into(),
            MetricThreshold {
                excellent: 160.0,
                good: 120.0,
                higher_is_better: true,
            },
        );
        thresholds.insert(
            "ttft_p50_ms".into(),
            MetricThreshold {
                excellent: 12.0,
                good: 50.0,
                higher_is_better: false,
            },
        );
        thresholds.insert(
            "itl_p50_ms".into(),
            MetricThreshold {
                excellent: 6.0,
                good: 10.0,
                higher_is_better: false,
            },
        );
        thresholds.insert(
            "ttft_p99_ms".into(),
            MetricThreshold {
                excellent: 15.0,
                good: 50.0,
                higher_is_better: false,
            },
        );
        thresholds.insert(
            "error_rate".into(),
            MetricThreshold {
                excellent: 0.0,
                good: 0.01,
                higher_is_better: false,
            },
        );
        thresholds.insert(
            "aggregate_tok_s".into(),
            MetricThreshold {
                excellent: 600.0,
                good: 300.0,
                higher_is_better: true,
            },
        );
        thresholds.insert(
            "throughput_scaling".into(),
            MetricThreshold {
                excellent: 3.8,
                good: 2.0,
                higher_is_better: true,
            },
        );

        let mut interactive_weights = HashMap::new();
        interactive_weights.insert("decode_tok_s".into(), 0.30);
        interactive_weights.insert("ttft_p50_ms".into(), 0.30);
        interactive_weights.insert("itl_p50_ms".into(), 0.15);
        interactive_weights.insert("ttft_p99_ms".into(), 0.15);
        interactive_weights.insert("error_rate".into(), 0.10);

        let mut throughput_weights = HashMap::new();
        throughput_weights.insert("aggregate_tok_s".into(), 0.30);
        throughput_weights.insert("decode_tok_s".into(), 0.15);
        throughput_weights.insert("ttft_p50_ms".into(), 0.15);
        throughput_weights.insert("itl_p50_ms".into(), 0.15);
        throughput_weights.insert("throughput_scaling".into(), 0.15);
        throughput_weights.insert("error_rate".into(), 0.10);

        let grades = vec![
            (95.0, "A+".into()),
            (90.0, "A".into()),
            (85.0, "A-".into()),
            (80.0, "B+".into()),
            (70.0, "B".into()),
            (60.0, "C+".into()),
            (50.0, "C".into()),
            (40.0, "D".into()),
            (30.0, "D-".into()),
            (0.0, "F".into()),
        ];

        Self {
            thresholds,
            interactive_weights,
            throughput_weights,
            best_in_class_bonus: 5,
            grades,
        }
    }
}

// =============================================================================
// Score results
// =============================================================================

/// Score for a single metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricScore {
    /// Raw value from the benchmark.
    pub value: f64,
    /// Computed score (0-100).
    pub score: u8,
    /// Whether this runtime was best-in-class for this metric.
    pub best: bool,
    /// Jitter penalty applied (only for ITL metric).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jitter_penalty: Option<u8>,
}

/// Scores for a single runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeScore {
    /// Runtime name (e.g., "realizr", "llamacpp", "vllm", "ollama").
    pub name: String,
    /// Source JSON file.
    pub source_file: String,
    /// Per-metric scores.
    pub metrics: HashMap<String, MetricScore>,
    /// Weighted composite score (0-100).
    pub composite: f64,
    /// Letter grade.
    pub grade: String,
}

/// Complete scorecard output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scorecard {
    /// Contract version used.
    pub contract_version: String,
    /// Timestamp of computation.
    pub timestamp: String,
    /// Concurrency level.
    pub concurrency: usize,
    /// Scored runtimes, sorted by composite descending.
    pub runtimes: Vec<RuntimeScore>,
}

// =============================================================================
// Score computation
// =============================================================================

/// Compute a single metric score from absolute thresholds.
///
/// - Higher-is-better: value >= excellent → 100, value >= good → 75-100, else 0-75
/// - Lower-is-better: value <= excellent → 100, value <= good → 75-100, else 0-75
pub fn compute_metric_score(value: f64, threshold: &MetricThreshold) -> u8 {
    if threshold.higher_is_better {
        score_higher_is_better(value, threshold.excellent, threshold.good)
    } else {
        score_lower_is_better(value, threshold.excellent, threshold.good)
    }
}

fn score_higher_is_better(value: f64, excellent: f64, good: f64) -> u8 {
    if value >= excellent {
        100
    } else if value <= 0.0 {
        0
    } else if value >= good {
        let pct = (value - good) / (excellent - good);
        (75.0 + 25.0 * pct).round() as u8
    } else {
        (75.0 * value / good).round().min(74.0) as u8
    }
}

fn score_lower_is_better(value: f64, excellent: f64, good: f64) -> u8 {
    if value <= excellent {
        100
    } else if value <= good {
        let pct = (good - value) / (good - excellent);
        (75.0 + 25.0 * pct).round() as u8
    } else if good > 0.0 {
        (75.0 * good / value).round().min(74.0).max(0.0) as u8
    } else {
        0
    }
}

/// Compute jitter penalty from tail analysis.
///
/// Deducts up to 30 points from ITL score based on spike count and CV.
/// Based on Metron fluidity-index insight: P50 hides stalls.
pub fn compute_jitter_penalty(tail: &TailAnalysis) -> u8 {
    let spike_penalty = (tail.jitter.spike_count as f64 * 2.0).min(20.0);
    let cv_penalty = (tail.jitter.itl_cv * 100.0).min(10.0);
    (spike_penalty + cv_penalty).round().min(30.0) as u8
}

/// Assign a letter grade from composite score.
pub fn assign_grade(composite: f64, grades: &[(f64, String)]) -> String {
    for (min_score, label) in grades {
        if composite >= *min_score {
            return label.clone();
        }
    }
    "F".into()
}

/// Extract metric values from a `LoadTestResult`.
fn extract_metrics(result: &LoadTestResult, is_throughput: bool) -> HashMap<String, f64> {
    let mut metrics = HashMap::new();
    metrics.insert("decode_tok_s".into(), result.decode_tok_per_sec);
    metrics.insert("ttft_p50_ms".into(), result.ttft_p50_ms);
    metrics.insert("itl_p50_ms".into(), result.itl_p50_ms);
    metrics.insert("ttft_p99_ms".into(), result.ttft_p99_ms);
    metrics.insert("error_rate".into(), result.error_rate);

    if is_throughput {
        metrics.insert("aggregate_tok_s".into(), result.tokens_per_sec);
    }
    metrics
}

/// Strip concurrency suffix (-c1, -c4, etc.) from runtime name for cross-concurrency matching.
fn strip_concurrency_suffix(name: &str) -> String {
    // Match patterns like "-c1", "-c4", "-c16" at end of string
    if let Some(pos) = name.rfind("-c") {
        let suffix = &name[pos + 2..];
        if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
            return name[..pos].to_string();
        }
    }
    name.to_string()
}

/// Compute scorecard from a set of benchmark results.
///
/// All results must share the same concurrency level. For throughput scoring,
/// provide `c1_results` to compute throughput_scaling ratios.
pub fn compute_scorecard(
    results: &[(LoadTestResult, String)], // (result, source_filename)
    c1_results: Option<&[(LoadTestResult, String)]>, // c=1 baselines for scaling ratio
    contract: &ScoringContract,
) -> Scorecard {
    let concurrency = results.first().map(|r| r.0.concurrency).unwrap_or(1);
    let is_throughput = concurrency > 1;
    let weights = if is_throughput {
        &contract.throughput_weights
    } else {
        &contract.interactive_weights
    };

    // Build c=1 decode lookup for throughput_scaling.
    // Strip concurrency suffixes (-c1, -c4, etc.) for cross-concurrency matching.
    let c1_decode: HashMap<String, f64> = c1_results
        .map(|c1| {
            c1.iter()
                .map(|(r, _)| {
                    (
                        strip_concurrency_suffix(&r.runtime_name),
                        r.decode_tok_per_sec,
                    )
                })
                .collect()
        })
        .unwrap_or_default();

    // Phase 1: Extract raw metric values per runtime
    let mut runtime_metrics: Vec<(String, String, HashMap<String, f64>, Option<&TailAnalysis>)> =
        Vec::new();

    for (result, source_file) in results {
        let mut metrics = extract_metrics(result, is_throughput);

        // Compute throughput_scaling if we have c=1 data
        if is_throughput {
            let base_name = strip_concurrency_suffix(&result.runtime_name);
            if let Some(&c1_decode_val) = c1_decode.get(&base_name) {
                if c1_decode_val > 0.0 {
                    metrics.insert(
                        "throughput_scaling".into(),
                        result.tokens_per_sec / c1_decode_val,
                    );
                }
            }
        }

        runtime_metrics.push((
            result.runtime_name.clone(),
            source_file.clone(),
            metrics,
            result.tail_analysis.as_ref(),
        ));
    }

    // Phase 2: Find best value per metric (for best-in-class bonus)
    let mut best_per_metric: HashMap<String, (f64, usize)> = HashMap::new(); // metric → (best_value, runtime_idx)
    for (idx, (_, _, metrics, _)) in runtime_metrics.iter().enumerate() {
        for (metric_name, &value) in metrics {
            if let Some(threshold) = contract.thresholds.get(metric_name) {
                let is_better = match best_per_metric.get(metric_name) {
                    None => true,
                    Some(&(best_val, _)) => {
                        if threshold.higher_is_better {
                            value > best_val
                        } else {
                            value < best_val
                        }
                    }
                };
                if is_better {
                    best_per_metric.insert(metric_name.clone(), (value, idx));
                }
            }
        }
    }

    // Phase 3: Compute scores per runtime
    let mut scored_runtimes: Vec<RuntimeScore> = Vec::new();

    for (idx, (name, source_file, metrics, tail)) in runtime_metrics.iter().enumerate() {
        let mut metric_scores: HashMap<String, MetricScore> = HashMap::new();
        let mut weighted_sum = 0.0;

        for (metric_name, weight) in weights {
            if let Some(&value) = metrics.get(metric_name) {
                if let Some(threshold) = contract.thresholds.get(metric_name) {
                    let mut score = compute_metric_score(value, threshold);

                    // Apply jitter penalty to ITL
                    let jitter_penalty = if metric_name == "itl_p50_ms" {
                        if let Some(tail_analysis) = tail {
                            let penalty = compute_jitter_penalty(tail_analysis);
                            score = score.saturating_sub(penalty);
                            Some(penalty)
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    // Best-in-class bonus
                    let is_best = best_per_metric
                        .get(metric_name)
                        .is_some_and(|&(_, best_idx)| best_idx == idx);

                    if is_best {
                        score = score.saturating_add(contract.best_in_class_bonus).min(100);
                    }

                    weighted_sum += *weight * f64::from(score);

                    metric_scores.insert(
                        metric_name.clone(),
                        MetricScore {
                            value,
                            score,
                            best: is_best,
                            jitter_penalty,
                        },
                    );
                }
            }
        }

        let composite = weighted_sum.round().min(100.0);
        let grade = assign_grade(composite, &contract.grades);

        scored_runtimes.push(RuntimeScore {
            name: name.clone(),
            source_file: source_file.clone(),
            metrics: metric_scores,
            composite,
            grade,
        });
    }

    // Sort by composite descending
    scored_runtimes.sort_by(|a, b| {
        b.composite
            .partial_cmp(&a.composite)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Scorecard {
        contract_version: "2.0.0".into(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        concurrency,
        runtimes: scored_runtimes,
    }
}

/// Format scorecard as a terminal table.
pub fn format_table(scorecard: &Scorecard) -> String {
    let scenario = if scorecard.concurrency == 1 {
        "Interactive (c=1)"
    } else {
        &format!("Throughput (c={})", scorecard.concurrency)
    };

    let mut lines = Vec::new();
    lines.push(format!("Inference Runtime Scorecard — {scenario}"));
    lines.push(String::new());

    if scorecard.concurrency == 1 {
        // Interactive columns
        lines.push(format!(
            "{:<12} {:>8} {:>8} {:>8} {:>8} {:>8}  {:>10}",
            "Runtime", "Decode", "TTFT", "ITL", "Tail", "Error", "Composite"
        ));
        lines.push(format!(
            "{:<12} {:>8} {:>8} {:>8} {:>8} {:>8}  {:>10}",
            "", "tok/s", "P50", "P50", "P99", "Rate", "Score"
        ));
        lines.push("-".repeat(74));

        for rt in &scorecard.runtimes {
            let decode = format_metric_cell(rt.metrics.get("decode_tok_s"));
            let ttft = format_metric_cell(rt.metrics.get("ttft_p50_ms"));
            let itl = format_metric_cell(rt.metrics.get("itl_p50_ms"));
            let tail = format_metric_cell(rt.metrics.get("ttft_p99_ms"));
            let error = format_metric_cell(rt.metrics.get("error_rate"));

            lines.push(format!(
                "{:<12} {:>8} {:>8} {:>8} {:>8} {:>8}  {:>5.1} {}",
                rt.name, decode, ttft, itl, tail, error, rt.composite, rt.grade
            ));
        }
    } else {
        // Throughput columns
        lines.push(format!(
            "{:<12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}  {:>10}",
            "Runtime", "Aggr", "Decode", "TTFT", "ITL", "Scale", "Error", "Composite"
        ));
        lines.push(format!(
            "{:<12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}  {:>10}",
            "", "tok/s", "tok/s", "P50", "P50", "ratio", "Rate", "Score"
        ));
        lines.push("-".repeat(88));

        for rt in &scorecard.runtimes {
            let aggr = format_metric_cell(rt.metrics.get("aggregate_tok_s"));
            let decode = format_metric_cell(rt.metrics.get("decode_tok_s"));
            let ttft = format_metric_cell(rt.metrics.get("ttft_p50_ms"));
            let itl = format_metric_cell(rt.metrics.get("itl_p50_ms"));
            let scale = format_metric_cell(rt.metrics.get("throughput_scaling"));
            let error = format_metric_cell(rt.metrics.get("error_rate"));

            lines.push(format!(
                "{:<12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}  {:>5.1} {}",
                rt.name, aggr, decode, ttft, itl, scale, error, rt.composite, rt.grade
            ));
        }
    }

    lines.push(String::new());
    lines.push("* = best in class".into());
    lines.join("\n")
}

fn format_metric_cell(metric: Option<&MetricScore>) -> String {
    match metric {
        Some(m) => {
            let star = if m.best { "*" } else { " " };
            format!("{:>3}{}", m.score, star)
        }
        None => "   -".into(),
    }
}

/// Format scorecard as Markdown.
pub fn format_markdown(scorecard: &Scorecard) -> String {
    let scenario = if scorecard.concurrency == 1 {
        "Interactive (c=1)"
    } else {
        &format!("Throughput (c={})", scorecard.concurrency)
    };

    let mut lines = Vec::new();
    lines.push(format!("## Scorecard — {scenario}"));
    lines.push(String::new());

    if scorecard.concurrency == 1 {
        lines.push(
            "| Runtime | Decode | TTFT P50 | ITL P50 | Tail P99 | Error | **Composite** |".into(),
        );
        lines.push(
            "|---------|--------|----------|---------|----------|-------|---------------|".into(),
        );

        for rt in &scorecard.runtimes {
            let decode = format_md_cell(rt.metrics.get("decode_tok_s"));
            let ttft = format_md_cell(rt.metrics.get("ttft_p50_ms"));
            let itl = format_md_cell(rt.metrics.get("itl_p50_ms"));
            let tail = format_md_cell(rt.metrics.get("ttft_p99_ms"));
            let error = format_md_cell(rt.metrics.get("error_rate"));

            lines.push(format!(
                "| {} | {} | {} | {} | {} | {} | **{:.1} ({})** |",
                rt.name, decode, ttft, itl, tail, error, rt.composite, rt.grade
            ));
        }
    } else {
        lines.push("| Runtime | Aggregate | Decode | TTFT P50 | ITL P50 | Scaling | Error | **Composite** |".into());
        lines.push("|---------|-----------|--------|----------|---------|---------|-------|---------------|".into());

        for rt in &scorecard.runtimes {
            let aggr = format_md_cell(rt.metrics.get("aggregate_tok_s"));
            let decode = format_md_cell(rt.metrics.get("decode_tok_s"));
            let ttft = format_md_cell(rt.metrics.get("ttft_p50_ms"));
            let itl = format_md_cell(rt.metrics.get("itl_p50_ms"));
            let scale = format_md_cell(rt.metrics.get("throughput_scaling"));
            let error = format_md_cell(rt.metrics.get("error_rate"));

            lines.push(format!(
                "| {} | {} | {} | {} | {} | {} | {} | **{:.1} ({})** |",
                rt.name, aggr, decode, ttft, itl, scale, error, rt.composite, rt.grade
            ));
        }
    }

    lines.join("\n")
}

fn format_md_cell(metric: Option<&MetricScore>) -> String {
    match metric {
        Some(m) => {
            let star = if m.best { " **" } else { "" };
            let end = if m.best { "**" } else { "" };
            format!("{star}{}{end}", m.score)
        }
        None => "-".into(),
    }
}

// =============================================================================
// Per-layer scoring
// =============================================================================

/// Per-layer decode time thresholds (microseconds per layer).
/// Derived from current baselines: vLLM 223us, llama.cpp 249us, realizr 255us.
const LAYER_US_EXCELLENT: f64 = 220.0; // vLLM-class
const LAYER_US_GOOD: f64 = 300.0; // Acceptable for interactive use

/// Score for a single runtime's per-layer decode efficiency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerScore {
    /// Runtime name.
    pub name: String,
    /// Microseconds per layer during decode.
    pub us_per_layer: f64,
    /// Number of layers.
    pub num_layers: u32,
    /// Score (0-100).
    pub score: u8,
    /// Letter grade.
    pub grade: String,
    /// Whether this runtime is best-in-class.
    pub best: bool,
}

/// Layer scorecard across runtimes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerScorecard {
    /// Timestamp.
    pub timestamp: String,
    /// Scored runtimes, sorted by us_per_layer ascending (best first).
    pub runtimes: Vec<LayerScore>,
}

/// Compute per-layer decode scores from results that have `decode_us_per_layer`.
pub fn compute_layer_scorecard(
    results: &[(LoadTestResult, String)],
    grades: &[(f64, String)],
) -> LayerScorecard {
    let layer_threshold = MetricThreshold {
        excellent: LAYER_US_EXCELLENT,
        good: LAYER_US_GOOD,
        higher_is_better: false, // lower us/layer is better
    };

    let mut scored: Vec<LayerScore> = results
        .iter()
        .filter_map(|(r, _)| {
            let us = r.decode_us_per_layer?;
            let layers = r.num_layers?;
            if us <= 0.0 {
                return None;
            }
            let score = compute_metric_score(us, &layer_threshold);
            let grade = assign_grade(f64::from(score), grades);
            Some(LayerScore {
                name: r.runtime_name.clone(),
                us_per_layer: us,
                num_layers: layers,
                score,
                grade,
                best: false,
            })
        })
        .collect();

    // Sort ascending (best = lowest us/layer)
    scored.sort_by(|a, b| {
        a.us_per_layer
            .partial_cmp(&b.us_per_layer)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Mark best
    if let Some(first) = scored.first_mut() {
        first.best = true;
    }

    LayerScorecard {
        timestamp: chrono::Utc::now().to_rfc3339(),
        runtimes: scored,
    }
}

/// Format layer scorecard as a terminal table.
pub fn format_layer_table(scorecard: &LayerScorecard) -> String {
    let mut lines = Vec::new();
    lines.push("Per-Layer Decode Efficiency".into());
    lines.push(String::new());
    lines.push(format!(
        "{:<20} {:>12} {:>8} {:>8} {:>8}",
        "Runtime", "us/layer", "Layers", "Score", "Grade"
    ));
    lines.push("-".repeat(60));

    for rt in &scorecard.runtimes {
        let star = if rt.best { "*" } else { " " };
        lines.push(format!(
            "{:<20} {:>11.1}{} {:>8} {:>8} {:>8}",
            rt.name, rt.us_per_layer, star, rt.num_layers, rt.score, rt.grade
        ));
    }

    lines.push(String::new());
    lines.push(format!(
        "Thresholds: excellent <= {LAYER_US_EXCELLENT}us, good <= {LAYER_US_GOOD}us"
    ));
    lines.join("\n")
}

/// Format layer scorecard as Markdown.
pub fn format_layer_markdown(scorecard: &LayerScorecard) -> String {
    let mut lines = Vec::new();
    lines.push("## Per-Layer Decode Efficiency".into());
    lines.push(String::new());
    lines.push("| Runtime | us/layer | Layers | Score | Grade |".into());
    lines.push("|---------|----------|--------|-------|-------|".into());

    for rt in &scorecard.runtimes {
        let star = if rt.best { " **" } else { "" };
        let end = if rt.best { "**" } else { "" };
        lines.push(format!(
            "| {} | {star}{:.1}{end} | {} | {} | {} |",
            rt.name, rt.us_per_layer, rt.num_layers, rt.score, rt.grade
        ));
    }

    lines.join("\n")
}

// =============================================================================
// Per-training-step scoring (PMAT-485)
// =============================================================================

/// Bottleneck classification for training profiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingBottleneck {
    /// Memory bandwidth limited (DRAM loads dominate)
    MemoryBw,
    /// Compute limited (GPU utilization > 50%)
    Compute,
    /// Kernel launch overhead (too many small kernels)
    Launch,
    /// PCIe/H2D/D2H transfer limited
    Transfer,
}

impl std::fmt::Display for TrainingBottleneck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MemoryBw => f.write_str("memory_bw"),
            Self::Compute => f.write_str("compute"),
            Self::Launch => f.write_str("launch"),
            Self::Transfer => f.write_str("transfer"),
        }
    }
}

/// Score for a single runtime's per-step training efficiency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStepScore {
    /// Runtime name (e.g., "apr", "unsloth", "pytorch").
    pub name: String,
    /// Average milliseconds per training step.
    pub ms_per_step: f64,
    /// Training throughput in tokens/second.
    pub tokens_per_sec: f64,
    /// Wall coverage: fraction of step time accounted for by profiled phases.
    pub wall_coverage: f64,
    /// Classified bottleneck type.
    pub bottleneck: TrainingBottleneck,
    /// Number of hotspot layers (>1.5x average).
    pub hotspot_layers: u32,
    /// Score 0-100.
    pub score: u8,
    /// Letter grade.
    pub grade: String,
    /// Best in class flag.
    pub best: bool,
}

/// Training step scorecard across runtimes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStepScorecard {
    /// Timestamp of analysis.
    pub timestamp: String,
    /// Model being trained.
    pub model_name: String,
    /// Per-runtime scores, sorted best-first.
    pub runtimes: Vec<TrainingStepScore>,
}

/// Thresholds for training throughput scoring (Qwen 1.5B on RTX 4060L).
const TRAINING_TOK_S_EXCELLENT: f64 = 6000.0; // unsloth-level
const TRAINING_TOK_S_GOOD: f64 = 1500.0; // pytorch-gradacc level

/// Classify bottleneck from profiling phase percentages.
pub fn classify_bottleneck(
    forward_pct: f64,
    transfer_pct: f64,
    _compute_util: f64,
) -> TrainingBottleneck {
    if transfer_pct > 30.0 {
        TrainingBottleneck::Transfer
    } else if forward_pct < 20.0 {
        TrainingBottleneck::Launch
    } else if _compute_util > 50.0 {
        TrainingBottleneck::Compute
    } else {
        TrainingBottleneck::MemoryBw
    }
}

/// Compute training step scores from profiling data.
///
/// `results` is a list of (name, tokens_per_sec, wall_coverage, bottleneck, hotspot_layers, ms_per_step).
pub fn compute_training_step_scorecard(
    results: &[(String, f64, f64, TrainingBottleneck, u32, f64)],
    model_name: &str,
    grades: &[(f64, String)],
) -> TrainingStepScorecard {
    let threshold = MetricThreshold {
        excellent: TRAINING_TOK_S_EXCELLENT,
        good: TRAINING_TOK_S_GOOD,
        higher_is_better: true,
    };

    let mut runtimes: Vec<TrainingStepScore> = results
        .iter()
        .map(|(name, tok_s, wc, bn, hotspots, ms)| {
            let raw_score = compute_metric_score(*tok_s, &threshold);
            let grade = assign_grade(raw_score.into(), grades);
            TrainingStepScore {
                name: name.clone(),
                ms_per_step: *ms,
                tokens_per_sec: *tok_s,
                wall_coverage: *wc,
                bottleneck: *bn,
                hotspot_layers: *hotspots,
                score: raw_score,
                grade,
                best: false,
            }
        })
        .collect();

    // Sort descending by tokens/sec (best first)
    runtimes.sort_by(|a, b| {
        b.tokens_per_sec
            .partial_cmp(&a.tokens_per_sec)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(first) = runtimes.first_mut() {
        first.best = true;
    }

    TrainingStepScorecard {
        timestamp: chrono::Utc::now().to_rfc3339(),
        model_name: model_name.to_string(),
        runtimes,
    }
}

/// Format training step scorecard as aligned table.
pub fn format_training_step_table(scorecard: &TrainingStepScorecard) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "Training Step Scorecard — {}",
        scorecard.model_name
    ));
    lines.push(format!(
        "{:<12} {:>10} {:>10} {:>5} {:>5} {:>10} {:>3}",
        "Runtime", "tok/s", "ms/step", "WC%", "Score", "Bottleneck", "Grd"
    ));
    lines.push("-".repeat(62));

    for rt in &scorecard.runtimes {
        let star = if rt.best { "*" } else { " " };
        lines.push(format!(
            "{:<12}{star}{:>9.0} {:>10.1} {:>4.0}% {:>5} {:>10} {:>3}",
            rt.name,
            rt.tokens_per_sec,
            rt.ms_per_step,
            rt.wall_coverage * 100.0,
            rt.score,
            rt.bottleneck,
            rt.grade,
        ));
    }

    lines.join("\n")
}

// =============================================================================
// Per-prompt-profile scoring
// =============================================================================

/// Prompt profile categories derived from average prompt tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PromptCategory {
    /// ~10 tokens (TTFT-only measurement)
    Micro,
    /// ~23-32 tokens (quick latency)
    Short,
    /// ~100-128 tokens (standard comparison)
    Medium,
    /// ~512+ tokens (sustained decode)
    Long,
}

impl PromptCategory {
    /// Classify from average prompt tokens per request.
    pub fn from_avg_prompt_tokens(avg: f64) -> Self {
        if avg < 15.0 {
            Self::Micro
        } else if avg < 64.0 {
            Self::Short
        } else if avg < 256.0 {
            Self::Medium
        } else {
            Self::Long
        }
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Micro => "micro",
            Self::Short => "short",
            Self::Medium => "medium",
            Self::Long => "long",
        }
    }
}

impl std::fmt::Display for PromptCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Score for a single runtime at a specific prompt profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileEntry {
    /// Runtime name.
    pub name: String,
    /// Prompt profile category.
    pub profile: PromptCategory,
    /// Average prompt tokens.
    pub avg_prompt_tokens: f64,
    /// Composite score for this profile.
    pub composite: f64,
    /// Grade.
    pub grade: String,
    /// Key metric values for this profile.
    pub decode_tok_s: f64,
    pub ttft_p50_ms: f64,
    pub itl_p50_ms: f64,
}

/// Profile scorecard showing how each runtime scores across prompt lengths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileScorecard {
    /// Timestamp.
    pub timestamp: String,
    /// Concurrency level.
    pub concurrency: usize,
    /// Entries grouped by runtime, then by profile.
    pub entries: Vec<ProfileEntry>,
    /// Per-runtime consistency score: how much does the score degrade from short to long?
    /// 100 = no degradation, 0 = catastrophic degradation.
    pub consistency: Vec<ConsistencyScore>,
}

/// How consistent a runtime's performance is across prompt profiles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyScore {
    /// Runtime name.
    pub name: String,
    /// Best profile score.
    pub best_score: f64,
    /// Worst profile score.
    pub worst_score: f64,
    /// Consistency: worst/best * 100 (100 = perfectly consistent).
    pub consistency: f64,
    /// Grade.
    pub grade: String,
}

/// Compute per-prompt-profile scores.
///
/// Groups results by runtime and prompt category, computes composite scores
/// per profile, and derives a consistency metric.
pub fn compute_profile_scorecard(
    results: &[(LoadTestResult, String)],
    contract: &ScoringContract,
) -> ProfileScorecard {
    let concurrency = results.first().map(|r| r.0.concurrency).unwrap_or(1);

    // Group by (runtime_name, prompt_category)
    let mut grouped: HashMap<(String, PromptCategory), Vec<&LoadTestResult>> = HashMap::new();
    for (result, _) in results {
        let avg_prompt = if result.total_requests > 0 {
            result.prompt_tokens_total as f64 / result.total_requests as f64
        } else {
            0.0
        };
        let category = PromptCategory::from_avg_prompt_tokens(avg_prompt);
        grouped
            .entry((result.runtime_name.clone(), category))
            .or_default()
            .push(result);
    }

    let weights = if concurrency > 1 {
        &contract.throughput_weights
    } else {
        &contract.interactive_weights
    };

    // Compute score for each (runtime, profile) pair using latest result
    let mut entries: Vec<ProfileEntry> = Vec::new();
    for ((name, profile), results_in_group) in &grouped {
        // Use the latest result (last by timestamp, already sorted)
        if let Some(result) = results_in_group.last() {
            let avg_prompt = if result.total_requests > 0 {
                result.prompt_tokens_total as f64 / result.total_requests as f64
            } else {
                0.0
            };

            // Compute composite from available metrics
            let mut weighted_sum = 0.0;
            for (metric_name, weight) in weights {
                let value = match metric_name.as_str() {
                    "decode_tok_s" => result.decode_tok_per_sec,
                    "ttft_p50_ms" => result.ttft_p50_ms,
                    "itl_p50_ms" => result.itl_p50_ms,
                    "ttft_p99_ms" => result.ttft_p99_ms,
                    "error_rate" => result.error_rate,
                    "aggregate_tok_s" => result.tokens_per_sec,
                    _ => continue,
                };
                if let Some(threshold) = contract.thresholds.get(metric_name) {
                    let score = compute_metric_score(value, threshold);
                    weighted_sum += weight * f64::from(score);
                }
            }

            let composite = weighted_sum.round().min(100.0);
            let grade = assign_grade(composite, &contract.grades);

            entries.push(ProfileEntry {
                name: name.clone(),
                profile: *profile,
                avg_prompt_tokens: avg_prompt,
                composite,
                grade,
                decode_tok_s: result.decode_tok_per_sec,
                ttft_p50_ms: result.ttft_p50_ms,
                itl_p50_ms: result.itl_p50_ms,
            });
        }
    }

    // Sort by runtime name, then profile order
    entries.sort_by(|a, b| {
        a.name.cmp(&b.name).then_with(|| {
            let order = |p: &PromptCategory| match p {
                PromptCategory::Micro => 0,
                PromptCategory::Short => 1,
                PromptCategory::Medium => 2,
                PromptCategory::Long => 3,
            };
            order(&a.profile).cmp(&order(&b.profile))
        })
    });

    // Compute consistency per runtime
    let mut runtime_scores: HashMap<String, Vec<f64>> = HashMap::new();
    for entry in &entries {
        runtime_scores
            .entry(entry.name.clone())
            .or_default()
            .push(entry.composite);
    }

    let mut consistency: Vec<ConsistencyScore> = runtime_scores
        .into_iter()
        .filter(|(_, scores)| scores.len() >= 2)
        .map(|(name, scores)| {
            let best = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let worst = scores.iter().cloned().fold(f64::INFINITY, f64::min);
            let cons = if best > 0.0 {
                (worst / best * 100.0).round()
            } else {
                0.0
            };
            let grade = assign_grade(cons, &contract.grades);
            ConsistencyScore {
                name,
                best_score: best,
                worst_score: worst,
                consistency: cons,
                grade,
            }
        })
        .collect();

    consistency.sort_by(|a, b| {
        b.consistency
            .partial_cmp(&a.consistency)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    ProfileScorecard {
        timestamp: chrono::Utc::now().to_rfc3339(),
        concurrency,
        entries,
        consistency,
    }
}

/// Format profile scorecard as a terminal table.
pub fn format_profile_table(scorecard: &ProfileScorecard) -> String {
    let mut lines = Vec::new();
    lines.push("Per-Prompt-Profile Scores".into());
    lines.push(String::new());
    lines.push(format!(
        "{:<20} {:>8} {:>8} {:>10} {:>10} {:>8}  {:>10}",
        "Runtime", "Profile", "Tokens", "Decode", "TTFT", "ITL", "Score"
    ));
    lines.push("-".repeat(82));

    for entry in &scorecard.entries {
        lines.push(format!(
            "{:<20} {:>8} {:>8.0} {:>9.1} {:>9.1} {:>8.1}  {:>5.1} {}",
            entry.name,
            entry.profile.label(),
            entry.avg_prompt_tokens,
            entry.decode_tok_s,
            entry.ttft_p50_ms,
            entry.itl_p50_ms,
            entry.composite,
            entry.grade,
        ));
    }

    if !scorecard.consistency.is_empty() {
        lines.push(String::new());
        lines.push("Profile Consistency (worst/best score across profiles)".into());
        lines.push(format!(
            "{:<20} {:>8} {:>8} {:>10} {:>8}",
            "Runtime", "Best", "Worst", "Consistency", "Grade"
        ));
        lines.push("-".repeat(58));
        for cs in &scorecard.consistency {
            lines.push(format!(
                "{:<20} {:>8.1} {:>8.1} {:>9.0}% {:>8}",
                cs.name, cs.best_score, cs.worst_score, cs.consistency, cs.grade
            ));
        }
    }

    lines.join("\n")
}

/// Format profile scorecard as Markdown.
pub fn format_profile_markdown(scorecard: &ProfileScorecard) -> String {
    let mut lines = Vec::new();
    lines.push("## Per-Prompt-Profile Scores".into());
    lines.push(String::new());
    lines.push(
        "| Runtime | Profile | Tokens | Decode tok/s | TTFT P50 ms | ITL P50 ms | **Score** |"
            .into(),
    );
    lines.push(
        "|---------|---------|--------|-------------|-------------|------------|-----------|"
            .into(),
    );

    for entry in &scorecard.entries {
        lines.push(format!(
            "| {} | {} | {:.0} | {:.1} | {:.1} | {:.1} | **{:.1} ({})** |",
            entry.name,
            entry.profile.label(),
            entry.avg_prompt_tokens,
            entry.decode_tok_s,
            entry.ttft_p50_ms,
            entry.itl_p50_ms,
            entry.composite,
            entry.grade,
        ));
    }

    if !scorecard.consistency.is_empty() {
        lines.push(String::new());
        lines.push("### Profile Consistency".into());
        lines.push(String::new());
        lines.push("| Runtime | Best | Worst | Consistency | Grade |".into());
        lines.push("|---------|------|-------|-------------|-------|".into());
        for cs in &scorecard.consistency {
            lines.push(format!(
                "| {} | {:.1} | {:.1} | {:.0}% | {} |",
                cs.name, cs.best_score, cs.worst_score, cs.consistency, cs.grade
            ));
        }
    }

    lines.join("\n")
}

// =============================================================================
// Correctness scoring
// =============================================================================

/// Score for a single runtime's correctness (assertion pass rate).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectnessScore {
    /// Runtime name.
    pub name: String,
    /// Pass rate (0.0-1.0).
    pub pass_rate: f64,
    /// Total tests/validations.
    pub total: u64,
    /// Passed.
    pub passed: u64,
    /// Score (0-100).
    pub score: u8,
    /// Letter grade.
    pub grade: String,
    /// Best in class.
    pub best: bool,
}

/// Correctness scorecard across runtimes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectnessScorecard {
    pub timestamp: String,
    pub runtimes: Vec<CorrectnessScore>,
}

/// Compute correctness scores from LoadTestResult quality data.
pub fn compute_correctness_scorecard(
    results: &[(LoadTestResult, String)],
    grades: &[(f64, String)],
) -> CorrectnessScorecard {
    let threshold = MetricThreshold {
        excellent: 1.0,
        good: 0.95,
        higher_is_better: true,
    };

    let mut scored: Vec<CorrectnessScore> = results
        .iter()
        .filter_map(|(r, _)| {
            let q = r.quality.as_ref()?;
            if q.total_validated == 0 {
                return None;
            }
            let score = compute_metric_score(q.pass_rate, &threshold);
            let grade = assign_grade(f64::from(score), grades);
            Some(CorrectnessScore {
                name: r.runtime_name.clone(),
                pass_rate: q.pass_rate,
                total: q.total_validated,
                passed: q.passed,
                score,
                grade,
                best: false,
            })
        })
        .collect();

    scored.sort_by(|a, b| {
        b.pass_rate
            .partial_cmp(&a.pass_rate)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(first) = scored.first_mut() {
        first.best = true;
    }

    CorrectnessScorecard {
        timestamp: chrono::Utc::now().to_rfc3339(),
        runtimes: scored,
    }
}

/// Format correctness scorecard as a terminal table.
pub fn format_correctness_table(scorecard: &CorrectnessScorecard) -> String {
    let mut lines = Vec::new();
    lines.push("Correctness Scores".into());
    lines.push(String::new());
    lines.push(format!(
        "{:<24} {:>10} {:>8} {:>8} {:>8} {:>8}",
        "Runtime", "Pass Rate", "Passed", "Total", "Score", "Grade"
    ));
    lines.push("-".repeat(70));

    for rt in &scorecard.runtimes {
        let star = if rt.best { "*" } else { " " };
        lines.push(format!(
            "{:<24} {:>9.1}%{} {:>8} {:>8} {:>8} {:>8}",
            rt.name,
            rt.pass_rate * 100.0,
            star,
            rt.passed,
            rt.total,
            rt.score,
            rt.grade
        ));
    }

    lines.push(String::new());
    lines.push("Thresholds: excellent = 100%, good = 95%".into());
    lines.join("\n")
}

/// Format correctness scorecard as Markdown.
pub fn format_correctness_markdown(scorecard: &CorrectnessScorecard) -> String {
    let mut lines = Vec::new();
    lines.push("## Correctness Scores".into());
    lines.push(String::new());
    lines.push("| Runtime | Pass Rate | Passed | Total | Score | Grade |".into());
    lines.push("|---------|-----------|--------|-------|-------|-------|".into());

    for rt in &scorecard.runtimes {
        lines.push(format!(
            "| {} | {:.1}% | {} | {} | {} | {} |",
            rt.name,
            rt.pass_rate * 100.0,
            rt.passed,
            rt.total,
            rt.score,
            rt.grade
        ));
    }
    lines.join("\n")
}

// =============================================================================
// Output length profile scoring
// =============================================================================

/// Output token count categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputLengthCategory {
    /// < 32 tokens.
    Short,
    /// 32-128 tokens.
    Medium,
    /// > 128 tokens.
    Long,
}

impl OutputLengthCategory {
    /// Classify from completion token count.
    pub fn from_tokens(tokens: u32) -> Self {
        if tokens < 32 {
            Self::Short
        } else if tokens <= 128 {
            Self::Medium
        } else {
            Self::Long
        }
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Short => "short",
            Self::Medium => "medium",
            Self::Long => "long",
        }
    }
}

impl std::fmt::Display for OutputLengthCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Score entry for a runtime at a specific output length bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputLengthEntry {
    pub name: String,
    pub category: OutputLengthCategory,
    pub request_count: usize,
    pub avg_output_tokens: f64,
    pub decode_tok_s: f64,
    pub itl_p50_ms: f64,
    pub score: u8,
    pub grade: String,
}

/// Output length scorecard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputLengthScorecard {
    pub timestamp: String,
    pub entries: Vec<OutputLengthEntry>,
}

/// Compute output-length-profile scores from request_details.
pub fn compute_output_length_scorecard(
    results: &[(LoadTestResult, String)],
    contract: &ScoringContract,
) -> OutputLengthScorecard {
    let mut entries = Vec::new();

    for (result, _) in results {
        if result.request_details.is_empty() {
            continue;
        }

        // Bucket requests by output length
        let mut buckets: HashMap<OutputLengthCategory, Vec<&super::loadtest::RequestDetail>> =
            HashMap::new();
        for rd in &result.request_details {
            let cat = OutputLengthCategory::from_tokens(rd.completion_tokens);
            buckets.entry(cat).or_default().push(rd);
        }

        for (cat, reqs) in &buckets {
            if reqs.is_empty() {
                continue;
            }
            let avg_tokens = reqs
                .iter()
                .map(|r| f64::from(r.completion_tokens))
                .sum::<f64>()
                / reqs.len() as f64;
            let avg_itl = reqs.iter().map(|r| r.itl_ms).sum::<f64>() / reqs.len() as f64;
            let decode = if avg_itl > 0.0 { 1000.0 / avg_itl } else { 0.0 };

            // Score using ITL threshold (decode is derived from ITL)
            let itl_threshold =
                contract
                    .thresholds
                    .get("itl_p50_ms")
                    .cloned()
                    .unwrap_or(MetricThreshold {
                        excellent: 6.0,
                        good: 10.0,
                        higher_is_better: false,
                    });
            let score = compute_metric_score(avg_itl, &itl_threshold);
            let grade = assign_grade(f64::from(score), &contract.grades);

            entries.push(OutputLengthEntry {
                name: result.runtime_name.clone(),
                category: *cat,
                request_count: reqs.len(),
                avg_output_tokens: avg_tokens,
                decode_tok_s: decode,
                itl_p50_ms: avg_itl,
                score,
                grade,
            });
        }
    }

    // Sort by name, then category order
    entries.sort_by(|a, b| {
        a.name.cmp(&b.name).then_with(|| {
            let order = |c: &OutputLengthCategory| match c {
                OutputLengthCategory::Short => 0,
                OutputLengthCategory::Medium => 1,
                OutputLengthCategory::Long => 2,
            };
            order(&a.category).cmp(&order(&b.category))
        })
    });

    OutputLengthScorecard {
        timestamp: chrono::Utc::now().to_rfc3339(),
        entries,
    }
}

/// Format output-length scorecard as a terminal table.
pub fn format_output_length_table(scorecard: &OutputLengthScorecard) -> String {
    let mut lines = Vec::new();
    lines.push("Per-Output-Length Scores".into());
    lines.push(String::new());
    lines.push(format!(
        "{:<24} {:>8} {:>8} {:>8} {:>10} {:>8} {:>8}",
        "Runtime", "Output", "Count", "AvgTok", "Decode", "ITL", "Score"
    ));
    lines.push("-".repeat(80));

    for e in &scorecard.entries {
        lines.push(format!(
            "{:<24} {:>8} {:>8} {:>8.1} {:>9.1} {:>8.1} {:>5} {}",
            e.name,
            e.category.label(),
            e.request_count,
            e.avg_output_tokens,
            e.decode_tok_s,
            e.itl_p50_ms,
            e.score,
            e.grade
        ));
    }
    lines.join("\n")
}

/// Format output-length scorecard as Markdown.
pub fn format_output_length_markdown(scorecard: &OutputLengthScorecard) -> String {
    let mut lines = Vec::new();
    lines.push("## Per-Output-Length Scores".into());
    lines.push(String::new());
    lines.push(
        "| Runtime | Output | Count | Avg Tokens | Decode tok/s | ITL ms | Score | Grade |".into(),
    );
    lines.push(
        "|---------|--------|-------|------------|-------------|--------|-------|-------|".into(),
    );

    for e in &scorecard.entries {
        lines.push(format!(
            "| {} | {} | {} | {:.1} | {:.1} | {:.1} | {} | {} |",
            e.name,
            e.category.label(),
            e.request_count,
            e.avg_output_tokens,
            e.decode_tok_s,
            e.itl_p50_ms,
            e.score,
            e.grade
        ));
    }
    lines.join("\n")
}

// =============================================================================
// Memory footprint scoring
// =============================================================================

/// Memory efficiency score for a single runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryScore {
    pub name: String,
    /// Peak VRAM used (MB).
    pub vram_used_mb: f64,
    /// Total VRAM (MB).
    pub vram_total_mb: f64,
    /// Decode tok/s per GB of VRAM used.
    pub tok_per_sec_per_gb: f64,
    /// Score (0-100).
    pub score: u8,
    /// Grade.
    pub grade: String,
    /// Best in class.
    pub best: bool,
}

/// Memory scorecard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryScorecard {
    pub timestamp: String,
    pub runtimes: Vec<MemoryScore>,
}

// tok/s per GB thresholds
const MEMORY_EFFICIENCY_EXCELLENT: f64 = 40.0; // ~140 tok/s in 3.5 GB
const MEMORY_EFFICIENCY_GOOD: f64 = 20.0; // ~160 tok/s in 8 GB

/// Compute memory efficiency scores from GPU telemetry.
pub fn compute_memory_scorecard(
    results: &[(LoadTestResult, String)],
    grades: &[(f64, String)],
) -> MemoryScorecard {
    let threshold = MetricThreshold {
        excellent: MEMORY_EFFICIENCY_EXCELLENT,
        good: MEMORY_EFFICIENCY_GOOD,
        higher_is_better: true,
    };

    let mut scored: Vec<MemoryScore> = results
        .iter()
        .filter_map(|(r, _)| {
            let telem = r.gpu_telemetry.as_ref()?;
            let vram_gb = telem.memory_used_mb.max / 1024.0;
            if vram_gb <= 0.0 {
                return None;
            }
            let efficiency = r.decode_tok_per_sec / vram_gb;
            let score = compute_metric_score(efficiency, &threshold);
            let grade = assign_grade(f64::from(score), grades);
            Some(MemoryScore {
                name: r.runtime_name.clone(),
                vram_used_mb: telem.memory_used_mb.max,
                vram_total_mb: telem.memory_total_mb,
                tok_per_sec_per_gb: efficiency,
                score,
                grade,
                best: false,
            })
        })
        .collect();

    scored.sort_by(|a, b| {
        b.tok_per_sec_per_gb
            .partial_cmp(&a.tok_per_sec_per_gb)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(first) = scored.first_mut() {
        first.best = true;
    }

    MemoryScorecard {
        timestamp: chrono::Utc::now().to_rfc3339(),
        runtimes: scored,
    }
}

/// Format memory scorecard as a terminal table.
pub fn format_memory_table(scorecard: &MemoryScorecard) -> String {
    let mut lines = Vec::new();
    lines.push("Memory Efficiency".into());
    lines.push(String::new());
    lines.push(format!(
        "{:<24} {:>10} {:>10} {:>12} {:>8} {:>8}",
        "Runtime", "VRAM MB", "Total MB", "tok/s/GB", "Score", "Grade"
    ));
    lines.push("-".repeat(76));

    for rt in &scorecard.runtimes {
        let star = if rt.best { "*" } else { " " };
        lines.push(format!(
            "{:<24} {:>10.0} {:>10.0} {:>11.1}{} {:>8} {:>8}",
            rt.name,
            rt.vram_used_mb,
            rt.vram_total_mb,
            rt.tok_per_sec_per_gb,
            star,
            rt.score,
            rt.grade
        ));
    }

    lines.push(String::new());
    lines.push(format!(
        "Thresholds: excellent >= {MEMORY_EFFICIENCY_EXCELLENT} tok/s/GB, good >= {MEMORY_EFFICIENCY_GOOD} tok/s/GB"
    ));
    lines.join("\n")
}

/// Format memory scorecard as Markdown.
pub fn format_memory_markdown(scorecard: &MemoryScorecard) -> String {
    let mut lines = Vec::new();
    lines.push("## Memory Efficiency".into());
    lines.push(String::new());
    lines.push("| Runtime | VRAM (MB) | Total (MB) | tok/s/GB | Score | Grade |".into());
    lines.push("|---------|-----------|------------|----------|-------|-------|".into());

    for rt in &scorecard.runtimes {
        lines.push(format!(
            "| {} | {:.0} | {:.0} | {:.1} | {} | {} |",
            rt.name, rt.vram_used_mb, rt.vram_total_mb, rt.tok_per_sec_per_gb, rt.score, rt.grade
        ));
    }
    lines.join("\n")
}

// =============================================================================
// Cold start scoring
// =============================================================================

/// Cold start score for a single runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdStartScore {
    pub name: String,
    /// Cold start time (ms).
    pub cold_start_ms: f64,
    /// Score (0-100).
    pub score: u8,
    /// Grade.
    pub grade: String,
    /// Best in class.
    pub best: bool,
}

/// Cold start scorecard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdStartScorecard {
    pub timestamp: String,
    pub runtimes: Vec<ColdStartScore>,
}

const COLD_START_EXCELLENT_MS: f64 = 500.0;
const COLD_START_GOOD_MS: f64 = 3000.0;

/// Compute cold start scores from `cold_start_ms` field.
pub fn compute_cold_start_scorecard(
    results: &[(LoadTestResult, String)],
    grades: &[(f64, String)],
) -> ColdStartScorecard {
    let threshold = MetricThreshold {
        excellent: COLD_START_EXCELLENT_MS,
        good: COLD_START_GOOD_MS,
        higher_is_better: false,
    };

    let mut scored: Vec<ColdStartScore> = results
        .iter()
        .filter_map(|(r, _)| {
            let cs = r.cold_start_ms?;
            if cs <= 0.0 {
                return None;
            }
            let score = compute_metric_score(cs, &threshold);
            let grade = assign_grade(f64::from(score), grades);
            Some(ColdStartScore {
                name: r.runtime_name.clone(),
                cold_start_ms: cs,
                score,
                grade,
                best: false,
            })
        })
        .collect();

    scored.sort_by(|a, b| {
        a.cold_start_ms
            .partial_cmp(&b.cold_start_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(first) = scored.first_mut() {
        first.best = true;
    }

    ColdStartScorecard {
        timestamp: chrono::Utc::now().to_rfc3339(),
        runtimes: scored,
    }
}

/// Format cold start scorecard as a terminal table.
pub fn format_cold_start_table(scorecard: &ColdStartScorecard) -> String {
    let mut lines = Vec::new();
    lines.push("Cold Start Time".into());
    lines.push(String::new());
    lines.push(format!(
        "{:<24} {:>12} {:>8} {:>8}",
        "Runtime", "Start (ms)", "Score", "Grade"
    ));
    lines.push("-".repeat(56));

    for rt in &scorecard.runtimes {
        let star = if rt.best { "*" } else { " " };
        lines.push(format!(
            "{:<24} {:>11.0}{} {:>8} {:>8}",
            rt.name, rt.cold_start_ms, star, rt.score, rt.grade
        ));
    }

    lines.push(String::new());
    lines.push(format!(
        "Thresholds: excellent <= {COLD_START_EXCELLENT_MS}ms, good <= {COLD_START_GOOD_MS}ms"
    ));
    lines.join("\n")
}

/// Format cold start scorecard as Markdown.
pub fn format_cold_start_markdown(scorecard: &ColdStartScorecard) -> String {
    let mut lines = Vec::new();
    lines.push("## Cold Start Time".into());
    lines.push(String::new());
    lines.push("| Runtime | Start (ms) | Score | Grade |".into());
    lines.push("|---------|------------|-------|-------|".into());

    for rt in &scorecard.runtimes {
        lines.push(format!(
            "| {} | {:.0} | {} | {} |",
            rt.name, rt.cold_start_ms, rt.score, rt.grade
        ));
    }
    lines.join("\n")
}

// =============================================================================
// Power efficiency scoring
// =============================================================================

/// Power efficiency score for a single runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerEfficiencyScore {
    pub name: String,
    /// Mean power draw (watts).
    pub mean_power_w: f64,
    /// Energy per token (mJ).
    pub energy_per_token_mj: f64,
    /// Decode tokens per second per watt.
    pub tok_per_watt: f64,
    /// Score (0-100).
    pub score: u8,
    /// Grade.
    pub grade: String,
    /// Best in class.
    pub best: bool,
}

/// Power efficiency scorecard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerEfficiencyScorecard {
    pub timestamp: String,
    pub runtimes: Vec<PowerEfficiencyScore>,
}

const POWER_EFFICIENCY_EXCELLENT: f64 = 3.0; // tok/s/W (Jetson-class)
const POWER_EFFICIENCY_GOOD: f64 = 1.5; // tok/s/W (desktop GPU)

/// Compute power efficiency scores from GPU telemetry.
pub fn compute_power_efficiency_scorecard(
    results: &[(LoadTestResult, String)],
    grades: &[(f64, String)],
) -> PowerEfficiencyScorecard {
    let threshold = MetricThreshold {
        excellent: POWER_EFFICIENCY_EXCELLENT,
        good: POWER_EFFICIENCY_GOOD,
        higher_is_better: true,
    };

    let mut scored: Vec<PowerEfficiencyScore> = results
        .iter()
        .filter_map(|(r, _)| {
            let telem = r.gpu_telemetry.as_ref()?;
            if telem.power_draw_w.mean <= 0.0 || r.decode_tok_per_sec <= 0.0 {
                return None;
            }
            let tok_per_watt = r.decode_tok_per_sec / telem.power_draw_w.mean;
            let score = compute_metric_score(tok_per_watt, &threshold);
            let grade = assign_grade(f64::from(score), grades);
            Some(PowerEfficiencyScore {
                name: r.runtime_name.clone(),
                mean_power_w: telem.power_draw_w.mean,
                energy_per_token_mj: telem.energy_per_token_mj,
                tok_per_watt,
                score,
                grade,
                best: false,
            })
        })
        .collect();

    scored.sort_by(|a, b| {
        b.tok_per_watt
            .partial_cmp(&a.tok_per_watt)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(first) = scored.first_mut() {
        first.best = true;
    }

    PowerEfficiencyScorecard {
        timestamp: chrono::Utc::now().to_rfc3339(),
        runtimes: scored,
    }
}

/// Format power efficiency scorecard as a terminal table.
pub fn format_power_table(scorecard: &PowerEfficiencyScorecard) -> String {
    let mut lines = Vec::new();
    lines.push("Power Efficiency".into());
    lines.push(String::new());
    lines.push(format!(
        "{:<24} {:>10} {:>12} {:>10} {:>8} {:>8}",
        "Runtime", "Power (W)", "mJ/token", "tok/s/W", "Score", "Grade"
    ));
    lines.push("-".repeat(76));

    for rt in &scorecard.runtimes {
        let star = if rt.best { "*" } else { " " };
        lines.push(format!(
            "{:<24} {:>10.1} {:>12.1} {:>9.2}{} {:>8} {:>8}",
            rt.name,
            rt.mean_power_w,
            rt.energy_per_token_mj,
            rt.tok_per_watt,
            star,
            rt.score,
            rt.grade
        ));
    }

    lines.push(String::new());
    lines.push(format!(
        "Thresholds: excellent >= {POWER_EFFICIENCY_EXCELLENT} tok/s/W, good >= {POWER_EFFICIENCY_GOOD} tok/s/W"
    ));
    lines.join("\n")
}

/// Format power efficiency scorecard as Markdown.
pub fn format_power_markdown(scorecard: &PowerEfficiencyScorecard) -> String {
    let mut lines = Vec::new();
    lines.push("## Power Efficiency".into());
    lines.push(String::new());
    lines.push("| Runtime | Power (W) | mJ/token | tok/s/W | Score | Grade |".into());
    lines.push("|---------|-----------|----------|---------|-------|-------|".into());

    for rt in &scorecard.runtimes {
        lines.push(format!(
            "| {} | {:.1} | {:.1} | {:.2} | {} | {} |",
            rt.name, rt.mean_power_w, rt.energy_per_token_mj, rt.tok_per_watt, rt.score, rt.grade
        ));
    }
    lines.join("\n")
}

// =============================================================================
// Concurrency scaling curve scoring
// =============================================================================

/// Concurrency scaling score for a single runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrencyScalingScore {
    pub name: String,
    /// Decode tok/s at c=1.
    pub c1_decode_tok_s: f64,
    /// Peak aggregate tok/s (at best concurrency).
    pub peak_aggregate_tok_s: f64,
    /// Concurrency level of peak throughput.
    pub peak_concurrency: usize,
    /// Scaling efficiency: peak_aggregate / (c1_decode * peak_concurrency).
    /// 1.0 = perfect linear scaling.
    pub scaling_efficiency: f64,
    /// Score (0-100).
    pub score: u8,
    /// Grade.
    pub grade: String,
    /// Best in class.
    pub best: bool,
}

/// Concurrency scaling scorecard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrencyScalingScorecard {
    pub timestamp: String,
    pub runtimes: Vec<ConcurrencyScalingScore>,
}

const SCALING_EFFICIENCY_EXCELLENT: f64 = 0.90; // Near-linear (vLLM PagedAttention)
const SCALING_EFFICIENCY_GOOD: f64 = 0.50; // Acceptable batching

/// Compute concurrency scaling scores by grouping results by runtime across concurrency levels.
pub fn compute_concurrency_scaling_scorecard(
    results: &[(LoadTestResult, String)],
    grades: &[(f64, String)],
) -> ConcurrencyScalingScorecard {
    let threshold = MetricThreshold {
        excellent: SCALING_EFFICIENCY_EXCELLENT,
        good: SCALING_EFFICIENCY_GOOD,
        higher_is_better: true,
    };

    // Group by runtime base name (strip -cN suffix)
    let mut by_runtime: HashMap<String, Vec<&LoadTestResult>> = HashMap::new();
    for (r, _) in results {
        let base = strip_concurrency_suffix(&r.runtime_name);
        by_runtime.entry(base).or_default().push(r);
    }

    let mut scored: Vec<ConcurrencyScalingScore> = Vec::new();

    for (base_name, runs) in &by_runtime {
        // Need at least 2 concurrency levels
        let mut by_c: HashMap<usize, Vec<&LoadTestResult>> = HashMap::new();
        for r in runs {
            by_c.entry(r.concurrency).or_default().push(r);
        }
        if by_c.len() < 2 {
            continue;
        }

        // Get c=1 baseline (best decode tok/s)
        let c1_decode = by_c
            .get(&1)
            .and_then(|runs| {
                runs.iter()
                    .map(|r| r.decode_tok_per_sec)
                    .fold(None, |max: Option<f64>, v| {
                        Some(max.map_or(v, |m: f64| m.max(v)))
                    })
            })
            .unwrap_or(0.0);

        if c1_decode <= 0.0 {
            continue;
        }

        // Find peak aggregate across all concurrency levels
        let mut peak_agg = 0.0f64;
        let mut peak_c = 1usize;
        for (&c, runs) in &by_c {
            for r in runs {
                if r.tokens_per_sec > peak_agg {
                    peak_agg = r.tokens_per_sec;
                    peak_c = c;
                }
            }
        }

        if peak_c == 0 {
            continue;
        }

        let efficiency = peak_agg / (c1_decode * peak_c as f64);
        let score = compute_metric_score(efficiency, &threshold);
        let grade = assign_grade(f64::from(score), grades);

        scored.push(ConcurrencyScalingScore {
            name: base_name.clone(),
            c1_decode_tok_s: c1_decode,
            peak_aggregate_tok_s: peak_agg,
            peak_concurrency: peak_c,
            scaling_efficiency: efficiency,
            score,
            grade,
            best: false,
        });
    }

    scored.sort_by(|a, b| {
        b.scaling_efficiency
            .partial_cmp(&a.scaling_efficiency)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(first) = scored.first_mut() {
        first.best = true;
    }

    ConcurrencyScalingScorecard {
        timestamp: chrono::Utc::now().to_rfc3339(),
        runtimes: scored,
    }
}

/// Format concurrency scaling scorecard as a terminal table.
pub fn format_scaling_table(scorecard: &ConcurrencyScalingScorecard) -> String {
    let mut lines = Vec::new();
    lines.push("Concurrency Scaling Efficiency".into());
    lines.push(String::new());
    lines.push(format!(
        "{:<24} {:>10} {:>12} {:>8} {:>12} {:>8} {:>8}",
        "Runtime", "c=1 tok/s", "Peak Aggr", "Peak c", "Efficiency", "Score", "Grade"
    ));
    lines.push("-".repeat(88));

    for rt in &scorecard.runtimes {
        let star = if rt.best { "*" } else { " " };
        lines.push(format!(
            "{:<24} {:>10.1} {:>12.1} {:>8} {:>11.1}%{} {:>8} {:>8}",
            rt.name,
            rt.c1_decode_tok_s,
            rt.peak_aggregate_tok_s,
            rt.peak_concurrency,
            rt.scaling_efficiency * 100.0,
            star,
            rt.score,
            rt.grade
        ));
    }

    lines.push(String::new());
    lines.push(format!(
        "Thresholds: excellent >= {:.0}%, good >= {:.0}%",
        SCALING_EFFICIENCY_EXCELLENT * 100.0,
        SCALING_EFFICIENCY_GOOD * 100.0
    ));
    lines.push("Efficiency = peak_aggregate / (c1_decode × peak_concurrency)".into());
    lines.join("\n")
}

/// Format concurrency scaling scorecard as Markdown.
pub fn format_scaling_markdown(scorecard: &ConcurrencyScalingScorecard) -> String {
    let mut lines = Vec::new();
    lines.push("## Concurrency Scaling Efficiency".into());
    lines.push(String::new());
    lines.push(
        "| Runtime | c=1 tok/s | Peak Aggregate | Peak c | Efficiency | Score | Grade |".into(),
    );
    lines.push(
        "|---------|-----------|---------------|--------|------------|-------|-------|".into(),
    );

    for rt in &scorecard.runtimes {
        lines.push(format!(
            "| {} | {:.1} | {:.1} | {} | {:.1}% | {} | {} |",
            rt.name,
            rt.c1_decode_tok_s,
            rt.peak_aggregate_tok_s,
            rt.peak_concurrency,
            rt.scaling_efficiency * 100.0,
            rt.score,
            rt.grade
        ));
    }
    lines.join("\n")
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_higher_is_better_at_excellent() {
        let t = MetricThreshold {
            excellent: 160.0,
            good: 120.0,
            higher_is_better: true,
        };
        assert_eq!(compute_metric_score(160.0, &t), 100);
        assert_eq!(compute_metric_score(200.0, &t), 100); // capped
    }

    #[test]
    fn test_higher_is_better_at_good() {
        let t = MetricThreshold {
            excellent: 160.0,
            good: 120.0,
            higher_is_better: true,
        };
        assert_eq!(compute_metric_score(120.0, &t), 75);
    }

    #[test]
    fn test_higher_is_better_below_good() {
        let t = MetricThreshold {
            excellent: 160.0,
            good: 120.0,
            higher_is_better: true,
        };
        let score = compute_metric_score(60.0, &t);
        assert_eq!(score, 38); // 75 * 60/120 = 37.5 → 38
    }

    #[test]
    fn test_higher_is_better_zero() {
        let t = MetricThreshold {
            excellent: 160.0,
            good: 120.0,
            higher_is_better: true,
        };
        assert_eq!(compute_metric_score(0.0, &t), 0);
    }

    #[test]
    fn test_lower_is_better_at_excellent() {
        let t = MetricThreshold {
            excellent: 12.0,
            good: 50.0,
            higher_is_better: false,
        };
        assert_eq!(compute_metric_score(12.0, &t), 100);
        assert_eq!(compute_metric_score(5.0, &t), 100); // better than excellent
    }

    #[test]
    fn test_lower_is_better_at_good() {
        let t = MetricThreshold {
            excellent: 12.0,
            good: 50.0,
            higher_is_better: false,
        };
        assert_eq!(compute_metric_score(50.0, &t), 75);
    }

    #[test]
    fn test_lower_is_better_above_good() {
        let t = MetricThreshold {
            excellent: 12.0,
            good: 50.0,
            higher_is_better: false,
        };
        let score = compute_metric_score(100.0, &t);
        assert_eq!(score, 38); // 75 * 50/100 = 37.5 → 38
    }

    #[test]
    fn test_error_rate_zero_is_perfect() {
        let t = MetricThreshold {
            excellent: 0.0,
            good: 0.01,
            higher_is_better: false,
        };
        assert_eq!(compute_metric_score(0.0, &t), 100);
    }

    #[test]
    fn test_error_rate_low_still_high_score() {
        // F-SCORE-007: 0.7% error should score >= 80
        let t = MetricThreshold {
            excellent: 0.0,
            good: 0.01,
            higher_is_better: false,
        };
        let score = compute_metric_score(0.007, &t);
        assert!(
            score >= 80,
            "0.7% error rate scored {score}, expected >= 80"
        );
    }

    #[test]
    fn test_jitter_penalty_clean() {
        let tail = TailAnalysis {
            itl_p999_ms: 7.0,
            itl_p9999_ms: 7.0,
            ttft_p999_ms: 15.0,
            ttft_p9999_ms: 15.0,
            latency_p999_ms: 250.0,
            latency_p9999_ms: 250.0,
            tail_ratio_itl: 1.0,
            tail_ratio_ttft: 1.0,
            tail_ratio_latency: 1.0,
            jitter: super::super::loadtest::JitterAnalysis {
                itl_cv: 0.01,
                itl_iqr_ms: 0.1,
                spike_count: 0,
                spike_threshold_ms: 35.0,
                spikes: vec![],
            },
            drift: super::super::loadtest::DriftAnalysis {
                itl_slope_ms_per_min: 0.0,
                ttft_slope_ms_per_min: 0.0,
                degradation_detected: false,
            },
        };
        assert_eq!(compute_jitter_penalty(&tail), 1); // just 0.01*100 = 1
    }

    #[test]
    fn test_jitter_penalty_spiky() {
        // F-SCORE-003: spiky runtime should get significant penalty
        let tail = TailAnalysis {
            itl_p999_ms: 50.0,
            itl_p9999_ms: 100.0,
            ttft_p999_ms: 15.0,
            ttft_p9999_ms: 15.0,
            latency_p999_ms: 300.0,
            latency_p9999_ms: 350.0,
            tail_ratio_itl: 7.0,
            tail_ratio_ttft: 1.0,
            tail_ratio_latency: 1.2,
            jitter: super::super::loadtest::JitterAnalysis {
                itl_cv: 0.15,
                itl_iqr_ms: 5.0,
                spike_count: 10,
                spike_threshold_ms: 35.0,
                spikes: vec![],
            },
            drift: super::super::loadtest::DriftAnalysis {
                itl_slope_ms_per_min: 0.0,
                ttft_slope_ms_per_min: 0.0,
                degradation_detected: false,
            },
        };
        let penalty = compute_jitter_penalty(&tail);
        assert!(penalty >= 25, "spiky penalty={penalty}, expected >= 25");
        assert!(penalty <= 30, "spiky penalty={penalty}, expected <= 30");
    }

    #[test]
    fn test_grade_assignment() {
        let grades = ScoringContract::default().grades;
        assert_eq!(assign_grade(97.0, &grades), "A+");
        assert_eq!(assign_grade(92.0, &grades), "A");
        assert_eq!(assign_grade(85.0, &grades), "A-");
        assert_eq!(assign_grade(80.0, &grades), "B+");
        assert_eq!(assign_grade(75.0, &grades), "B");
        assert_eq!(assign_grade(60.0, &grades), "C+");
        assert_eq!(assign_grade(50.0, &grades), "C");
        assert_eq!(assign_grade(40.0, &grades), "D");
        assert_eq!(assign_grade(30.0, &grades), "D-");
        assert_eq!(assign_grade(10.0, &grades), "F");
    }

    #[test]
    fn test_no_single_metric_dominates() {
        // F-SCORE-002: zeroing any one metric cannot drop composite below 40
        let contract = ScoringContract::default();
        for zeroed_metric in contract.interactive_weights.keys() {
            let mut weighted_sum = 0.0;
            for (metric, weight) in &contract.interactive_weights {
                let score = if metric == zeroed_metric { 0.0 } else { 100.0 };
                weighted_sum += weight * score;
            }
            assert!(
                weighted_sum >= 40.0,
                "Zeroing {zeroed_metric} drops composite to {weighted_sum}"
            );
        }
    }

    #[test]
    fn test_weights_sum_to_one() {
        let contract = ScoringContract::default();
        let interactive_sum: f64 = contract.interactive_weights.values().sum();
        assert!(
            (interactive_sum - 1.0).abs() < 0.001,
            "Interactive weights sum to {interactive_sum}"
        );
        let throughput_sum: f64 = contract.throughput_weights.values().sum();
        assert!(
            (throughput_sum - 1.0).abs() < 0.001,
            "Throughput weights sum to {throughput_sum}"
        );
    }

    #[test]
    fn test_score_independence_from_field() {
        // F-SCORE-001: Adding/removing a runtime changes scores by at most the bonus amount
        let contract = ScoringContract::default();

        // Create two fake results
        let result_a = make_test_result("runtime_a", 150.0, 15.0, 7.0, 20.0, 0.0, 1);
        let result_b = make_test_result("runtime_b", 130.0, 30.0, 8.0, 40.0, 0.0, 1);
        let result_c = make_test_result("runtime_c", 100.0, 60.0, 12.0, 80.0, 0.01, 1);

        let card_abc = compute_scorecard(
            &[
                (result_a.clone(), "a.json".into()),
                (result_b.clone(), "b.json".into()),
                (result_c, "c.json".into()),
            ],
            None,
            &contract,
        );

        let card_ab = compute_scorecard(
            &[(result_a, "a.json".into()), (result_b, "b.json".into())],
            None,
            &contract,
        );

        let score_a_with_bc = card_abc
            .runtimes
            .iter()
            .find(|r| r.name == "runtime_a")
            .unwrap()
            .composite;
        let score_a_with_b = card_ab
            .runtimes
            .iter()
            .find(|r| r.name == "runtime_a")
            .unwrap()
            .composite;

        let diff = (score_a_with_bc - score_a_with_b).abs();
        assert!(
            diff <= f64::from(contract.best_in_class_bonus),
            "Score changed by {diff} when removing runtime_c (max allowed: {})",
            contract.best_in_class_bonus
        );
    }

    fn make_test_result(
        name: &str,
        decode: f64,
        ttft: f64,
        itl: f64,
        ttft_p99: f64,
        error_rate: f64,
        concurrency: usize,
    ) -> LoadTestResult {
        LoadTestResult {
            total_requests: 100,
            successful: (100.0 * (1.0 - error_rate)) as u64,
            failed: (100.0 * error_rate) as u64,
            throughput_rps: decode / 32.0,
            latency_p50_ms: ttft + itl * 31.0,
            latency_p95_ms: ttft + itl * 31.0 * 1.1,
            latency_p99_ms: ttft + itl * 31.0 * 1.2,
            ttft_p50_ms: ttft,
            tokens_per_sec: decode * concurrency as f64,
            avg_tok_per_req: 32.0,
            itl_p50_ms: itl,
            decode_tok_per_sec: decode,
            prefill_tok_per_sec: 1000.0 / ttft * 23.0,
            timestamp: "2026-03-11T00:00:00Z".into(),
            runtime_name: name.into(),
            elapsed_secs: 60.0,
            concurrency,
            ttft_p90_ms: ttft * 1.1,
            ttft_p95_ms: ttft * 1.2,
            ttft_p99_ms: ttft_p99,
            tpot_p50_ms: itl,
            tpot_p90_ms: itl * 1.1,
            tpot_p95_ms: itl * 1.2,
            tpot_p99_ms: itl * 1.3,
            latency_min_ms: ttft + itl * 30.0,
            latency_max_ms: ttft + itl * 35.0,
            latency_stddev_ms: itl * 0.5,
            error_rate,
            prompt_tokens_total: 2300,
            completion_tokens_total: 3200,
            truncated_pct: 0.0,
            sse_batch_ratio: 1.0,
            goodput_pct: 100.0,
            output_tokens_dist: None,
            decode_us_per_layer: None,
            num_layers: Some(28),
            brick_trace_summary: None,
            request_details: vec![],
            quality: None,
            tail_analysis: None,
            gpu_telemetry: None,
            dataset_stats: None,
            cold_start_ms: None,
        }
    }

    fn make_test_result_with_layers(
        name: &str,
        decode: f64,
        ttft: f64,
        us_per_layer: f64,
        prompt_tokens: u64,
    ) -> LoadTestResult {
        let mut r = make_test_result(name, decode, ttft, 7.0, 20.0, 0.0, 1);
        r.decode_us_per_layer = Some(us_per_layer);
        r.prompt_tokens_total = prompt_tokens;
        r
    }

    #[test]
    fn test_layer_scoring_best_first() {
        let contract = ScoringContract::default();
        let results = vec![
            (
                make_test_result_with_layers("fast", 160.0, 12.0, 220.0, 2300),
                "a.json".into(),
            ),
            (
                make_test_result_with_layers("slow", 100.0, 50.0, 350.0, 2300),
                "b.json".into(),
            ),
        ];
        let card = compute_layer_scorecard(&results, &contract.grades);
        assert_eq!(card.runtimes.len(), 2);
        assert_eq!(card.runtimes[0].name, "fast");
        assert!(card.runtimes[0].best);
        assert!(card.runtimes[0].score > card.runtimes[1].score);
    }

    #[test]
    fn test_layer_scoring_excellent_threshold() {
        let contract = ScoringContract::default();
        let results = vec![(
            make_test_result_with_layers("vllm", 160.0, 12.0, 220.0, 2300),
            "a.json".into(),
        )];
        let card = compute_layer_scorecard(&results, &contract.grades);
        assert_eq!(card.runtimes[0].score, 100);
    }

    #[test]
    fn test_prompt_category_classification() {
        assert_eq!(
            PromptCategory::from_avg_prompt_tokens(10.0),
            PromptCategory::Micro
        );
        assert_eq!(
            PromptCategory::from_avg_prompt_tokens(23.0),
            PromptCategory::Short
        );
        assert_eq!(
            PromptCategory::from_avg_prompt_tokens(102.0),
            PromptCategory::Medium
        );
        assert_eq!(
            PromptCategory::from_avg_prompt_tokens(512.0),
            PromptCategory::Long
        );
    }

    #[test]
    fn test_profile_consistency_perfect() {
        let contract = ScoringContract::default();
        // Same runtime, same metrics, different prompt lengths
        let r_short = make_test_result_with_layers("runtime_a", 150.0, 15.0, 240.0, 2300);
        let mut r_medium = make_test_result_with_layers("runtime_a", 150.0, 15.0, 240.0, 10200);
        r_medium.prompt_tokens_total = 10200; // 102 avg prompt tokens
        let results = vec![
            (r_short, "short.json".into()),
            (r_medium, "medium.json".into()),
        ];
        let card = compute_profile_scorecard(&results, &contract);
        assert!(card.entries.len() >= 2);
        // Same metrics → consistency should be 100%
        if let Some(cs) = card.consistency.first() {
            assert_eq!(cs.consistency, 100.0);
        }
    }

    #[test]
    fn test_correctness_scoring() {
        let contract = ScoringContract::default();
        let mut r = make_test_result("runtime_a", 150.0, 15.0, 7.0, 20.0, 0.0, 1);
        r.quality = Some(super::super::loadtest::QualityResult {
            validation_level: "basic".into(),
            total_validated: 100,
            passed: 100,
            failed: 0,
            pass_rate: 1.0,
            failures: vec![],
        });
        let results = vec![(r, "a.json".into())];
        let card = compute_correctness_scorecard(&results, &contract.grades);
        assert_eq!(card.runtimes.len(), 1);
        assert_eq!(card.runtimes[0].score, 100);
    }

    #[test]
    fn test_correctness_partial() {
        let contract = ScoringContract::default();
        let mut r = make_test_result("runtime_a", 150.0, 15.0, 7.0, 20.0, 0.0, 1);
        r.quality = Some(super::super::loadtest::QualityResult {
            validation_level: "basic".into(),
            total_validated: 100,
            passed: 90,
            failed: 10,
            pass_rate: 0.9,
            failures: vec![],
        });
        let results = vec![(r, "a.json".into())];
        let card = compute_correctness_scorecard(&results, &contract.grades);
        assert!(
            card.runtimes[0].score < 75,
            "90% pass rate should score below good"
        );
    }

    #[test]
    fn test_output_length_classification() {
        assert_eq!(
            OutputLengthCategory::from_tokens(10),
            OutputLengthCategory::Short
        );
        assert_eq!(
            OutputLengthCategory::from_tokens(32),
            OutputLengthCategory::Medium
        );
        assert_eq!(
            OutputLengthCategory::from_tokens(128),
            OutputLengthCategory::Medium
        );
        assert_eq!(
            OutputLengthCategory::from_tokens(200),
            OutputLengthCategory::Long
        );
    }

    #[test]
    fn test_memory_scoring() {
        let contract = ScoringContract::default();
        let mut r = make_test_result("runtime_a", 140.0, 15.0, 7.0, 20.0, 0.0, 1);
        r.gpu_telemetry = Some(super::super::loadtest::GpuTelemetry {
            samples: 10,
            gpu_utilization_pct: super::super::loadtest::TelemetryStat {
                mean: 80.0,
                max: 95.0,
                min: 60.0,
            },
            memory_used_mb: super::super::loadtest::TelemetryStat {
                mean: 3200.0,
                max: 3500.0,
                min: 3000.0,
            },
            memory_total_mb: 8192.0,
            power_draw_w: super::super::loadtest::TelemetryStat {
                mean: 80.0,
                max: 100.0,
                min: 60.0,
            },
            temperature_c: super::super::loadtest::TelemetryStat {
                mean: 70.0,
                max: 80.0,
                min: 50.0,
            },
            clock_gpu_mhz: super::super::loadtest::TelemetryStat {
                mean: 1500.0,
                max: 1500.0,
                min: 1500.0,
            },
            throttle_events: 0,
            energy_total_wh: 1.0,
            energy_per_token_mj: 5.0,
            energy_per_request_mj: 160.0,
        });
        let results = vec![(r, "a.json".into())];
        let card = compute_memory_scorecard(&results, &contract.grades);
        assert_eq!(card.runtimes.len(), 1);
        // 140 tok/s / 3.42 GB = ~40.9 tok/s/GB → excellent
        assert!(
            card.runtimes[0].score >= 95,
            "High efficiency should score well: {}",
            card.runtimes[0].score
        );
    }

    #[test]
    fn test_cold_start_scoring() {
        let contract = ScoringContract::default();
        let mut r_fast = make_test_result("realizr", 140.0, 15.0, 7.0, 20.0, 0.0, 1);
        r_fast.cold_start_ms = Some(300.0);
        let mut r_slow = make_test_result("vllm", 160.0, 12.0, 6.0, 15.0, 0.0, 1);
        r_slow.cold_start_ms = Some(15000.0);
        let results = vec![(r_fast, "a.json".into()), (r_slow, "b.json".into())];
        let card = compute_cold_start_scorecard(&results, &contract.grades);
        assert_eq!(card.runtimes.len(), 2);
        assert_eq!(card.runtimes[0].name, "realizr"); // fastest first
        assert!(card.runtimes[0].score > card.runtimes[1].score);
    }

    #[test]
    fn test_power_efficiency_scoring() {
        let contract = ScoringContract::default();
        let mut r = make_test_result("runtime_a", 140.0, 15.0, 7.0, 20.0, 0.0, 1);
        r.gpu_telemetry = Some(super::super::loadtest::GpuTelemetry {
            samples: 10,
            gpu_utilization_pct: super::super::loadtest::TelemetryStat {
                mean: 80.0,
                max: 95.0,
                min: 60.0,
            },
            memory_used_mb: super::super::loadtest::TelemetryStat {
                mean: 3200.0,
                max: 3500.0,
                min: 3000.0,
            },
            memory_total_mb: 8192.0,
            power_draw_w: super::super::loadtest::TelemetryStat {
                mean: 80.0,
                max: 100.0,
                min: 60.0,
            },
            temperature_c: super::super::loadtest::TelemetryStat {
                mean: 70.0,
                max: 80.0,
                min: 50.0,
            },
            clock_gpu_mhz: super::super::loadtest::TelemetryStat {
                mean: 1500.0,
                max: 1500.0,
                min: 1500.0,
            },
            throttle_events: 0,
            energy_total_wh: 1.0,
            energy_per_token_mj: 5.0,
            energy_per_request_mj: 160.0,
        });
        let results = vec![(r, "a.json".into())];
        let card = compute_power_efficiency_scorecard(&results, &contract.grades);
        assert_eq!(card.runtimes.len(), 1);
        // 140 tok/s / 80W = 1.75 tok/s/W → above good
        assert!(
            card.runtimes[0].score >= 75,
            "1.75 tok/s/W should be above good: {}",
            card.runtimes[0].score
        );
    }

    #[test]
    fn test_concurrency_scaling() {
        let contract = ScoringContract::default();
        let r_c1 = make_test_result("runtime_a-c1", 150.0, 15.0, 7.0, 20.0, 0.0, 1);
        let mut r_c4 = make_test_result("runtime_a-c4", 140.0, 30.0, 8.0, 40.0, 0.0, 4);
        r_c4.tokens_per_sec = 540.0; // aggregate = 540
        let results = vec![(r_c1, "c1.json".into()), (r_c4, "c4.json".into())];
        let card = compute_concurrency_scaling_scorecard(&results, &contract.grades);
        assert_eq!(card.runtimes.len(), 1);
        // 540 / (150 * 4) = 0.90 → excellent
        assert!(card.runtimes[0].scaling_efficiency > 0.85);
        assert!(
            card.runtimes[0].score >= 90,
            "Near-linear scaling: {}",
            card.runtimes[0].score
        );
    }

    #[test]
    fn test_profile_consistency_degradation() {
        let contract = ScoringContract::default();
        // Good on short, bad on medium (TTFT degrades)
        let r_short = make_test_result_with_layers("runtime_a", 150.0, 15.0, 240.0, 2300);
        let mut r_medium = make_test_result_with_layers("runtime_a", 140.0, 80.0, 240.0, 10200);
        r_medium.prompt_tokens_total = 10200;
        let results = vec![
            (r_short, "short.json".into()),
            (r_medium, "medium.json".into()),
        ];
        let card = compute_profile_scorecard(&results, &contract);
        if let Some(cs) = card.consistency.first() {
            assert!(
                cs.consistency < 90.0,
                "Expected degradation, got {}%",
                cs.consistency
            );
            assert!(cs.worst_score < cs.best_score);
        }
    }
}
