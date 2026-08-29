//! Backend comparison module
//!
//! Compares kernel performance characteristics across different backends or configurations.

use crate::analyzer::AnalysisReport;
use serde::{Deserialize, Serialize};

/// Comparison result for a single metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComparison {
    /// Metric name
    pub name: String,
    /// Value in first report
    pub value_a: f32,
    /// Value in second report
    pub value_b: f32,
    /// Winner ("A", "B", or "Tie")
    pub winner: String,
    /// Notes about the comparison
    pub notes: String,
}

/// Full comparison report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    /// First report name
    pub report_a_name: String,
    /// Second report name
    pub report_b_name: String,
    /// Individual metric comparisons
    pub metrics: Vec<MetricComparison>,
    /// Overall recommendation
    pub recommendation: String,
}

/// Direction that makes a metric "better" when comparing two reports.
#[derive(Copy, Clone)]
enum Direction {
    LowerIsBetter,
    HigherIsBetter,
}

fn pick_winner(value_a: f32, value_b: f32, dir: Direction) -> &'static str {
    let (a_better, b_better) = match dir {
        Direction::LowerIsBetter => (value_a < value_b, value_b < value_a),
        Direction::HigherIsBetter => (value_a > value_b, value_b > value_a),
    };
    if a_better {
        "A"
    } else if b_better {
        "B"
    } else {
        "Tie"
    }
}

fn metric_comparison(
    name: &str,
    value_a: f32,
    value_b: f32,
    dir: Direction,
    notes: &str,
) -> MetricComparison {
    MetricComparison {
        name: name.to_string(),
        value_a,
        value_b,
        winner: pick_winner(value_a, value_b, dir).to_string(),
        notes: notes.to_string(),
    }
}

fn collect_metrics(a: &AnalysisReport, b: &AnalysisReport) -> Vec<MetricComparison> {
    vec![
        metric_comparison(
            "Register Count",
            a.registers.total() as f32,
            b.registers.total() as f32,
            Direction::LowerIsBetter,
            "Lower is better (higher occupancy)",
        ),
        metric_comparison(
            "Instruction Count",
            a.instruction_count as f32,
            b.instruction_count as f32,
            Direction::LowerIsBetter,
            "Lower is better (less work)",
        ),
        metric_comparison(
            "Estimated Occupancy",
            a.estimated_occupancy * 100.0,
            b.estimated_occupancy * 100.0,
            Direction::HigherIsBetter,
            "Higher is better (GPU utilization)",
        ),
        metric_comparison(
            "Muda Warnings",
            a.warnings.len() as f32,
            b.warnings.len() as f32,
            Direction::LowerIsBetter,
            "Lower is better (less waste)",
        ),
        metric_comparison(
            "Memory Coalescing",
            a.memory.coalesced_ratio * 100.0,
            b.memory.coalesced_ratio * 100.0,
            Direction::HigherIsBetter,
            "Higher is better (bandwidth efficiency)",
        ),
    ]
}

fn recommendation_text(metrics: &[MetricComparison], name_a: &str, name_b: &str) -> String {
    let a_wins = metrics.iter().filter(|m| m.winner == "A").count();
    let b_wins = metrics.iter().filter(|m| m.winner == "B").count();
    match a_wins.cmp(&b_wins) {
        std::cmp::Ordering::Greater => format!("{name_a} wins {a_wins} to {b_wins} metrics"),
        std::cmp::Ordering::Less => format!("{name_b} wins {b_wins} to {a_wins} metrics"),
        std::cmp::Ordering::Equal => "Both configurations are comparable".to_string(),
    }
}

/// Compare two analysis reports
#[must_use]
pub fn compare_analyses(report_a: &AnalysisReport, report_b: &AnalysisReport) -> ComparisonReport {
    let metrics = collect_metrics(report_a, report_b);
    let recommendation = recommendation_text(&metrics, &report_a.name, &report_b.name);
    ComparisonReport {
        report_a_name: report_a.name.clone(),
        report_b_name: report_b.name.clone(),
        metrics,
        recommendation,
    }
}

/// Format comparison report as text
#[must_use]
pub fn format_comparison_text(report: &ComparisonReport) -> String {
    let mut output = String::new();

    output.push_str(&format!(
        "╔══ Comparison: {} vs {} ══╗\n\n",
        report.report_a_name, report.report_b_name
    ));

    output.push_str(&format!(
        "{:<25} {:>12} {:>12} {:>8}\n",
        "Metric", report.report_a_name, report.report_b_name, "Winner"
    ));
    output.push_str(&format!("{}\n", "─".repeat(60)));

    for metric in &report.metrics {
        let winner_icon = match metric.winner.as_str() {
            "A" => "◀",
            "B" => "▶",
            _ => "═",
        };
        output.push_str(&format!(
            "{:<25} {:>12.1} {:>12.1} {:>6} {}\n",
            metric.name, metric.value_a, metric.value_b, winner_icon, metric.winner
        ));
    }

    output.push_str(&format!("\n{}\n", report.recommendation));

    output
}

/// Format comparison report as JSON
#[must_use]
pub fn format_comparison_json(report: &ComparisonReport) -> String {
    serde_json::to_string_pretty(report).unwrap_or_else(|_| "{}".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::{MemoryPattern, MudaWarning, RegisterUsage, RooflineMetric};

    fn make_report(
        name: &str,
        regs: u32,
        inst: u32,
        occ: f32,
        warns: usize,
        coal: f32,
    ) -> AnalysisReport {
        AnalysisReport {
            name: name.to_string(),
            target: "PTX".to_string(),
            registers: RegisterUsage {
                f32_regs: regs,
                ..Default::default()
            },
            memory: MemoryPattern {
                coalesced_ratio: coal,
                ..Default::default()
            },
            roofline: RooflineMetric::default(),
            warnings: (0..warns)
                .map(|_| MudaWarning {
                    muda_type: crate::analyzer::MudaType::Transport,
                    description: "test".to_string(),
                    impact: "test".to_string(),
                    line: None,
                    suggestion: None,
                })
                .collect(),
            instruction_count: inst,
            estimated_occupancy: occ,
        }
    }

    #[test]
    fn test_compare_identical() {
        let report_a = make_report("A", 32, 100, 0.75, 0, 0.95);
        let report_b = make_report("B", 32, 100, 0.75, 0, 0.95);

        let comparison = compare_analyses(&report_a, &report_b);

        // All ties
        assert!(comparison.metrics.iter().all(|m| m.winner == "Tie"));
    }

    #[test]
    fn test_compare_clear_winner() {
        let report_a = make_report("Optimized", 16, 50, 0.90, 0, 0.98);
        let report_b = make_report("Baseline", 64, 200, 0.50, 3, 0.70);

        let comparison = compare_analyses(&report_a, &report_b);

        // A should win on all metrics
        let a_wins = comparison
            .metrics
            .iter()
            .filter(|m| m.winner == "A")
            .count();
        assert!(a_wins >= 4, "Optimized should win most metrics");
        assert!(comparison.recommendation.contains("Optimized"));
    }

    #[test]
    fn test_compare_mixed() {
        // A has fewer registers but more warnings
        let report_a = make_report("LowReg", 16, 100, 0.90, 5, 0.80);
        let report_b = make_report("HighReg", 64, 100, 0.50, 0, 0.95);

        let comparison = compare_analyses(&report_a, &report_b);

        // Should have mixed results
        let a_wins = comparison
            .metrics
            .iter()
            .filter(|m| m.winner == "A")
            .count();
        let b_wins = comparison
            .metrics
            .iter()
            .filter(|m| m.winner == "B")
            .count();
        assert!(a_wins > 0 && b_wins > 0, "Should have mixed winners");
    }

    #[test]
    fn test_format_text() {
        let report_a = make_report("A", 32, 100, 0.75, 1, 0.90);
        let report_b = make_report("B", 48, 150, 0.60, 2, 0.85);

        let comparison = compare_analyses(&report_a, &report_b);
        let text = format_comparison_text(&comparison);

        assert!(text.contains("Comparison"));
        assert!(text.contains("Register Count"));
        assert!(text.contains("Instruction Count"));
    }

    #[test]
    fn test_format_json() {
        let report_a = make_report("A", 32, 100, 0.75, 0, 0.90);
        let report_b = make_report("B", 32, 100, 0.75, 0, 0.90);

        let comparison = compare_analyses(&report_a, &report_b);
        let json = format_comparison_json(&comparison);

        assert!(json.contains("\"report_a_name\": \"A\""));
        assert!(json.contains("\"report_b_name\": \"B\""));
    }
}
