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

/// Compare two analysis reports
#[must_use]
pub fn compare_analyses(report_a: &AnalysisReport, report_b: &AnalysisReport) -> ComparisonReport {
    let mut metrics = Vec::new();

    // Compare register usage (lower is better)
    let regs_a = report_a.registers.total() as f32;
    let regs_b = report_b.registers.total() as f32;
    metrics.push(MetricComparison {
        name: "Register Count".to_string(),
        value_a: regs_a,
        value_b: regs_b,
        winner: if regs_a < regs_b {
            "A".to_string()
        } else if regs_b < regs_a {
            "B".to_string()
        } else {
            "Tie".to_string()
        },
        notes: "Lower is better (higher occupancy)".to_string(),
    });

    // Compare instruction count (lower is better for same work)
    let inst_a = report_a.instruction_count as f32;
    let inst_b = report_b.instruction_count as f32;
    metrics.push(MetricComparison {
        name: "Instruction Count".to_string(),
        value_a: inst_a,
        value_b: inst_b,
        winner: if inst_a < inst_b {
            "A".to_string()
        } else if inst_b < inst_a {
            "B".to_string()
        } else {
            "Tie".to_string()
        },
        notes: "Lower is better (less work)".to_string(),
    });

    // Compare occupancy (higher is better)
    let occ_a = report_a.estimated_occupancy;
    let occ_b = report_b.estimated_occupancy;
    metrics.push(MetricComparison {
        name: "Estimated Occupancy".to_string(),
        value_a: occ_a * 100.0,
        value_b: occ_b * 100.0,
        winner: if occ_a > occ_b {
            "A".to_string()
        } else if occ_b > occ_a {
            "B".to_string()
        } else {
            "Tie".to_string()
        },
        notes: "Higher is better (GPU utilization)".to_string(),
    });

    // Compare warning counts (lower is better)
    let warns_a = report_a.warnings.len() as f32;
    let warns_b = report_b.warnings.len() as f32;
    metrics.push(MetricComparison {
        name: "Muda Warnings".to_string(),
        value_a: warns_a,
        value_b: warns_b,
        winner: if warns_a < warns_b {
            "A".to_string()
        } else if warns_b < warns_a {
            "B".to_string()
        } else {
            "Tie".to_string()
        },
        notes: "Lower is better (less waste)".to_string(),
    });

    // Compare memory coalescing (higher is better)
    let coal_a = report_a.memory.coalesced_ratio;
    let coal_b = report_b.memory.coalesced_ratio;
    metrics.push(MetricComparison {
        name: "Memory Coalescing".to_string(),
        value_a: coal_a * 100.0,
        value_b: coal_b * 100.0,
        winner: if coal_a > coal_b {
            "A".to_string()
        } else if coal_b > coal_a {
            "B".to_string()
        } else {
            "Tie".to_string()
        },
        notes: "Higher is better (bandwidth efficiency)".to_string(),
    });

    // Count wins
    let a_wins = metrics.iter().filter(|m| m.winner == "A").count();
    let b_wins = metrics.iter().filter(|m| m.winner == "B").count();

    let recommendation = match a_wins.cmp(&b_wins) {
        std::cmp::Ordering::Greater => {
            format!("{} wins {} to {} metrics", report_a.name, a_wins, b_wins)
        }
        std::cmp::Ordering::Less => {
            format!("{} wins {} to {} metrics", report_b.name, b_wins, a_wins)
        }
        std::cmp::Ordering::Equal => "Both configurations are comparable".to_string(),
    };

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
        "Metric", &report.report_a_name, &report.report_b_name, "Winner"
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
