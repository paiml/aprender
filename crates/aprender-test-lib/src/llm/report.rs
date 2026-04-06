//! Report generation for LLM test results.
//!
//! Produces JSON and Markdown output, and can update a historical
//! `performance.md` table with new results.

use super::benchmark::{AggregateStats, Regression};
use super::loadtest::LoadTestResult;
use std::path::Path;

/// Serialize a load test result to a pretty-printed JSON string.
pub fn to_json(result: &LoadTestResult) -> String {
    serde_json::to_string_pretty(result).unwrap_or_else(|_| "{}".to_string())
}

/// Produce a single Markdown table row for a load test result.
pub fn to_markdown_row(result: &LoadTestResult) -> String {
    let decode = if result.decode_tok_per_sec > 0.0 {
        format!("{:.1}", result.decode_tok_per_sec)
    } else {
        "-".to_string()
    };
    let itl = if result.itl_p50_ms > 0.0 {
        format!("{:.1}", result.itl_p50_ms)
    } else {
        "-".to_string()
    };
    let tpot = if result.tpot_p50_ms > 0.0 {
        format!("{:.1}", result.tpot_p50_ms)
    } else {
        "-".to_string()
    };
    let err_rate = if result.error_rate > 0.0 {
        format!("{:.1}%", result.error_rate * 100.0)
    } else {
        "0%".to_string()
    };
    let prefill = if result.prefill_tok_per_sec > 0.0 {
        format!("{:.1}", result.prefill_tok_per_sec)
    } else {
        "-".to_string()
    };
    format!(
        "| {} | {} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {} | {} | {} | {} | {} | {} |",
        result.timestamp.split('T').next().unwrap_or(&result.timestamp),
        result.runtime_name,
        result.concurrency,
        result.throughput_rps,
        result.latency_p50_ms,
        result.latency_p95_ms,
        result.latency_p99_ms,
        result.ttft_p50_ms,
        result.tokens_per_sec,
        result.avg_tok_per_req,
        itl,
        decode,
        prefill,
        tpot,
        err_rate,
        result.total_requests,
    )
}

/// Header for the performance Markdown table.
const TABLE_HEADER: &str = "\
| Date | Runtime | Concurrency | RPS | P50 (ms) | P95 (ms) | P99 (ms) | TTFT P50 (ms) | Tok/s | Avg tok/req | ITL P50 (ms) | Decode tok/s | Prefill tok/s | TPOT P50 (ms) | Err% | Requests |
|------|---------|-------------|-----|----------|----------|----------|---------------|-------|-------------|--------------|--------------|---------------|---------------|------|----------|";

/// Generate a complete Markdown table from multiple results.
pub fn to_markdown_table(results: &[LoadTestResult]) -> String {
    let mut lines = vec![
        "## Performance Results".to_string(),
        String::new(),
        TABLE_HEADER.to_string(),
    ];
    for r in results {
        lines.push(to_markdown_row(r));
    }
    lines.push(String::new());
    lines.join("\n")
}

/// Append new results to an existing `performance.md` file.
///
/// If the file doesn't exist, creates it with the table header.
/// If it exists, appends new rows after the existing table.
pub fn update_performance_md(
    path: &Path,
    results: &[LoadTestResult],
) -> Result<(), std::io::Error> {
    let existing = if path.exists() {
        std::fs::read_to_string(path)?
    } else {
        String::new()
    };

    let new_rows: Vec<String> = results.iter().map(|r| to_markdown_row(r)).collect();

    let content = if existing.is_empty() {
        // Create fresh file
        let mut lines = vec![
            "# LLM Inference Performance".to_string(),
            String::new(),
            TABLE_HEADER.to_string(),
        ];
        lines.extend(new_rows);
        lines.push(String::new());
        lines.join("\n")
    } else if existing.contains(TABLE_HEADER.lines().next().unwrap_or("")) {
        // Append rows to existing table
        let trimmed = existing.trim_end();
        let mut out = trimmed.to_string();
        for row in &new_rows {
            out.push('\n');
            out.push_str(row);
        }
        out.push('\n');
        out
    } else {
        // File exists but no table — append a new table section
        let mut out = existing;
        if !out.ends_with('\n') {
            out.push('\n');
        }
        out.push('\n');
        out.push_str(&to_markdown_table(results));
        out
    };

    std::fs::write(path, content)
}

/// Compare current aggregate results against a baseline and detect regressions.
///
/// For throughput-like metrics (higher is better), a decrease exceeding `threshold_pct`
/// is a regression. For latency-like metrics (lower is better), an increase exceeding
/// `threshold_pct` is a regression.
pub fn compare_to_baseline(
    current: &AggregateStats,
    baseline: &LoadTestResult,
    threshold_pct: f64,
) -> Vec<Regression> {
    let mut regressions = Vec::new();

    // Throughput: higher is better → decrease is regression
    check_regression_higher_better(
        "throughput_rps",
        baseline.throughput_rps,
        current.throughput_rps.mean,
        threshold_pct,
        &mut regressions,
    );

    // Tokens per second: higher is better
    check_regression_higher_better(
        "tokens_per_sec",
        baseline.tokens_per_sec,
        current.tokens_per_sec.mean,
        threshold_pct,
        &mut regressions,
    );

    // Latency P50: lower is better → increase is regression
    check_regression_lower_better(
        "latency_p50_ms",
        baseline.latency_p50_ms,
        current.latency_p50.mean,
        threshold_pct,
        &mut regressions,
    );

    // TPOT P50: lower is better
    if baseline.tpot_p50_ms > 0.0 {
        check_regression_lower_better(
            "tpot_p50_ms",
            baseline.tpot_p50_ms,
            current.tpot_p50.mean,
            threshold_pct,
            &mut regressions,
        );
    }

    regressions
}

fn check_regression_higher_better(
    metric: &str,
    baseline: f64,
    current: f64,
    threshold_pct: f64,
    regressions: &mut Vec<Regression>,
) {
    if baseline <= 0.0 {
        return;
    }
    let change_pct = ((current - baseline) / baseline) * 100.0;
    // Negative change_pct means decrease → regression
    let exceeds = change_pct < -threshold_pct;
    if exceeds {
        regressions.push(Regression {
            metric: metric.to_string(),
            baseline_value: baseline,
            current_value: current,
            change_pct,
            exceeds_threshold: true,
        });
    }
}

fn check_regression_lower_better(
    metric: &str,
    baseline: f64,
    current: f64,
    threshold_pct: f64,
    regressions: &mut Vec<Regression>,
) {
    if baseline <= 0.0 {
        return;
    }
    let change_pct = ((current - baseline) / baseline) * 100.0;
    // Positive change_pct means increase → regression
    let exceeds = change_pct > threshold_pct;
    if exceeds {
        regressions.push(Regression {
            metric: metric.to_string(),
            baseline_value: baseline,
            current_value: current,
            change_pct,
            exceeds_threshold: true,
        });
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::super::benchmark::StatSummary;
    use super::*;

    fn sample_result(runtime: &str) -> LoadTestResult {
        LoadTestResult {
            total_requests: 100,
            successful: 95,
            failed: 5,
            throughput_rps: 10.5,
            latency_p50_ms: 150.3,
            latency_p95_ms: 300.7,
            latency_p99_ms: 500.2,
            ttft_p50_ms: 80.1,
            tokens_per_sec: 200.0,
            avg_tok_per_req: 15.0,
            itl_p50_ms: 5.0,
            decode_tok_per_sec: 200.0,
            prefill_tok_per_sec: 0.0,
            timestamp: "2026-03-01T04:00:00Z".to_string(),
            runtime_name: runtime.to_string(),
            elapsed_secs: 10.0,
            concurrency: 4,
            ttft_p90_ms: 90.0,
            ttft_p95_ms: 95.0,
            ttft_p99_ms: 99.0,
            tpot_p50_ms: 6.0,
            tpot_p90_ms: 8.0,
            tpot_p95_ms: 9.0,
            tpot_p99_ms: 12.0,
            latency_min_ms: 50.0,
            latency_max_ms: 800.0,
            latency_stddev_ms: 120.0,
            error_rate: 0.05,
            prompt_tokens_total: 950,
            completion_tokens_total: 1425,
            truncated_pct: 0.0,
            sse_batch_ratio: 0.0,
            goodput_pct: 0.0,
            decode_us_per_layer: None,
            num_layers: None,
            output_tokens_dist: None,
            brick_trace_summary: None,
            request_details: Vec::new(),
            quality: None,
            tail_analysis: None,
            gpu_telemetry: None,
            dataset_stats: None,
            cold_start_ms: None,
        }
    }

    fn sample_aggregate(throughput: f64, latency: f64, tps: f64, tpot: f64) -> AggregateStats {
        AggregateStats {
            throughput_rps: StatSummary {
                mean: throughput,
                stddev: 0.5,
                ci_95_lower: throughput - 1.0,
                ci_95_upper: throughput + 1.0,
                values: vec![throughput],
            },
            latency_p50: StatSummary {
                mean: latency,
                stddev: 5.0,
                ci_95_lower: latency - 10.0,
                ci_95_upper: latency + 10.0,
                values: vec![latency],
            },
            tokens_per_sec: StatSummary {
                mean: tps,
                stddev: 10.0,
                ci_95_lower: tps - 20.0,
                ci_95_upper: tps + 20.0,
                values: vec![tps],
            },
            ttft_p50: StatSummary {
                mean: 50.0,
                stddev: 2.0,
                ci_95_lower: 48.0,
                ci_95_upper: 52.0,
                values: vec![50.0],
            },
            tpot_p50: StatSummary {
                mean: tpot,
                stddev: 0.5,
                ci_95_lower: tpot - 1.0,
                ci_95_upper: tpot + 1.0,
                values: vec![tpot],
            },
        }
    }

    #[test]
    fn test_to_json() {
        let result = sample_result("realizar");
        let json = to_json(&result);
        assert!(json.contains("\"runtime_name\": \"realizar\""));
        assert!(json.contains("\"total_requests\": 100"));
    }

    #[test]
    fn test_to_markdown_row() {
        let result = sample_result("ollama");
        let row = to_markdown_row(&result);
        assert!(row.starts_with("| 2026-03-01"));
        assert!(row.contains("ollama"));
        assert!(row.contains("10.5"));
        assert!(row.contains("150.3"));
        // New columns
        assert!(row.contains("6.0")); // TPOT P50
        assert!(row.contains("5.0%")); // Error rate
    }

    #[test]
    fn test_to_markdown_row_zero_error_rate() {
        let mut result = sample_result("test");
        result.error_rate = 0.0;
        let row = to_markdown_row(&result);
        assert!(row.contains("0%"));
    }

    #[test]
    fn test_to_markdown_table() {
        let results = vec![sample_result("realizar"), sample_result("ollama")];
        let table = to_markdown_table(&results);
        assert!(table.contains("## Performance Results"));
        assert!(table.contains("| Date |"));
        assert!(table.contains("TPOT P50"));
        assert!(table.contains("Err%"));
        assert!(table.contains("realizar"));
        assert!(table.contains("ollama"));
    }

    #[test]
    fn test_update_performance_md_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("performance.md");
        let results = vec![sample_result("realizar")];
        update_performance_md(&path, &results).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("# LLM Inference Performance"));
        assert!(content.contains("realizar"));
    }

    #[test]
    fn test_update_performance_md_append() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("performance.md");

        // First write
        update_performance_md(&path, &[sample_result("realizar")]).unwrap();

        // Second write (append)
        update_performance_md(&path, &[sample_result("ollama")]).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("realizar"));
        assert!(content.contains("ollama"));
        // Should NOT have duplicate headers
        assert_eq!(content.matches("# LLM Inference Performance").count(), 1);
    }

    #[test]
    fn test_update_performance_md_no_table_existing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("performance.md");
        std::fs::write(&path, "# Some other content\n\nHello world.\n").unwrap();

        update_performance_md(&path, &[sample_result("llamacpp")]).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("# Some other content"));
        assert!(content.contains("llamacpp"));
        assert!(content.contains("## Performance Results"));
    }

    #[test]
    fn test_to_json_roundtrip() {
        let result = sample_result("test");
        let json = to_json(&result);
        let back: LoadTestResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.runtime_name, "test");
        assert_eq!(back.total_requests, 100);
    }

    #[test]
    fn test_markdown_row_date_extraction() {
        let mut result = sample_result("x");
        result.timestamp = "2026-12-25T12:00:00Z".to_string();
        let row = to_markdown_row(&result);
        assert!(row.contains("2026-12-25"));
        assert!(!row.contains("T12:00:00Z"));
    }

    #[test]
    fn test_compare_to_baseline_no_regression() {
        let baseline = sample_result("baseline");
        let current = sample_aggregate(10.5, 150.3, 200.0, 6.0);
        let regressions = compare_to_baseline(&current, &baseline, 10.0);
        assert!(regressions.is_empty());
    }

    #[test]
    fn test_compare_to_baseline_throughput_regression() {
        let baseline = sample_result("baseline");
        // Throughput dropped from 10.5 to 8.0 → -23.8% → exceeds 10% threshold
        let current = sample_aggregate(8.0, 150.3, 200.0, 6.0);
        let regressions = compare_to_baseline(&current, &baseline, 10.0);
        assert_eq!(regressions.len(), 1);
        assert_eq!(regressions[0].metric, "throughput_rps");
        assert!(regressions[0].exceeds_threshold);
        assert!(regressions[0].change_pct < 0.0);
    }

    #[test]
    fn test_compare_to_baseline_latency_regression() {
        let baseline = sample_result("baseline");
        // Latency increased from 150.3 to 200.0 → +33% → exceeds 10% threshold
        let current = sample_aggregate(10.5, 200.0, 200.0, 6.0);
        let regressions = compare_to_baseline(&current, &baseline, 10.0);
        assert_eq!(regressions.len(), 1);
        assert_eq!(regressions[0].metric, "latency_p50_ms");
        assert!(regressions[0].change_pct > 0.0);
    }

    #[test]
    fn test_compare_to_baseline_multiple_regressions() {
        let baseline = sample_result("baseline");
        // Both throughput down and latency up
        let current = sample_aggregate(5.0, 300.0, 100.0, 15.0);
        let regressions = compare_to_baseline(&current, &baseline, 10.0);
        assert!(regressions.len() >= 2);
    }
}
