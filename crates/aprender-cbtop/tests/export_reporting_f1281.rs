#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Falsification Tests for PMAT-036: Multi-Format Export System
//!
//! F1281-F1290: Export reporting falsification tests

use cbtop::{
    BenchmarkMetric, BenchmarkReport, ComparisonEntry, ComparisonReport, ExportFormat,
    ReportBuilder, ReportExporter, ReportType,
};

// =============================================================================
// F1281: JSON Export Tests
// =============================================================================

/// F1281.1: JSON export valid (parses correctly)
#[test]
fn f1281_json_export_valid() {
    let mut report = BenchmarkReport::new("Test");
    report.add_metric("latency", 10.5, "ms");
    report.add_metric("throughput", 1000.0, "ops/s");

    let json = ReportExporter::benchmark_to_json(&report);

    assert!(json.starts_with("{"));
    assert!(json.ends_with("}"));
    assert!(json.contains("\"title\":\"Test\""));
    assert!(json.contains("\"latency\""));
    assert!(json.contains("10.5"));
}

/// F1281.2: JSON metrics array
#[test]
fn f1281_json_metrics_array() {
    let mut report = BenchmarkReport::new("Test");
    report.add_metric("a", 1.0, "x");
    report.add_metric("b", 2.0, "y");

    let json = ReportExporter::benchmark_to_json(&report);
    assert!(json.contains("\"metrics\":["));
}

// =============================================================================
// F1282: CSV Export Tests
// =============================================================================

/// F1282.1: CSV export valid (columns aligned)
#[test]
fn f1282_csv_export_valid() {
    let mut report = BenchmarkReport::new("Test");
    report.add_metric("latency", 10.5, "ms");
    report.add_metric("throughput", 1000.0, "ops/s");

    let csv = ReportExporter::benchmark_to_csv(&report);

    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines[0], "metric,value,unit");
    assert!(lines[1].contains("latency"));
    assert!(lines[1].contains("10.5"));
    assert!(lines[1].contains("ms"));
}

/// F1282.2: CSV has header row
#[test]
fn f1282_csv_header() {
    let report = BenchmarkReport::new("Test");
    let csv = ReportExporter::benchmark_to_csv(&report);

    assert!(csv.starts_with("metric,value,unit"));
}

// =============================================================================
// F1283: Markdown Export Tests
// =============================================================================

/// F1283.1: Markdown formatting (headers rendered)
#[test]
fn f1283_markdown_headers() {
    let mut report = BenchmarkReport::new("My Report");
    report.add_metric("latency", 10.0, "ms");

    let md = ReportExporter::benchmark_to_markdown(&report);

    assert!(md.contains("# My Report"));
    assert!(md.contains("## Metrics"));
    assert!(md.contains("| Metric |"));
}

/// F1283.2: Markdown table formatting
#[test]
fn f1283_markdown_table() {
    let mut report = BenchmarkReport::new("Test");
    report.add_metric("latency", 10.0, "ms");

    let md = ReportExporter::benchmark_to_markdown(&report);

    assert!(md.contains("|--------|"));
    assert!(md.contains("| latency |"));
}

// =============================================================================
// F1284: HTML Export Tests
// =============================================================================

/// F1284.1: HTML well-formed (valid HTML5)
#[test]
fn f1284_html_wellformed() {
    let mut report = BenchmarkReport::new("Test Report");
    report.add_metric("latency", 10.0, "ms");

    let html = ReportExporter::benchmark_to_html(&report);

    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("<html>"));
    assert!(html.contains("</html>"));
    assert!(html.contains("<head>"));
    assert!(html.contains("</head>"));
    assert!(html.contains("<body>"));
    assert!(html.contains("</body>"));
}

/// F1284.2: HTML title set
#[test]
fn f1284_html_title() {
    let report = BenchmarkReport::new("Test Report");
    let html = ReportExporter::benchmark_to_html(&report);

    assert!(html.contains("<title>Test Report</title>"));
    assert!(html.contains("<h1>Test Report</h1>"));
}

// =============================================================================
// F1285: Metrics Included Tests
// =============================================================================

/// F1285.1: All fields present
#[test]
fn f1285_all_fields_present() {
    let mut report =
        BenchmarkReport::new("Test").with_workload("gemm").with_backend("avx2").with_size(1024);
    report.add_metric("latency", 10.0, "ms");

    let json = ReportExporter::benchmark_to_json(&report);

    assert!(json.contains("\"workload\":\"gemm\""));
    assert!(json.contains("\"backend\":\"avx2\""));
    assert!(json.contains("\"size\":1024"));
}

/// F1285.2: Metadata included
#[test]
fn f1285_metadata_included() {
    let mut report = BenchmarkReport::new("Test");
    report.add_metadata("version", "1.0");
    report.add_metadata("host", "server01");

    let json = ReportExporter::benchmark_to_json(&report);

    assert!(json.contains("\"version\":\"1.0\""));
    assert!(json.contains("\"host\":\"server01\""));
}

// =============================================================================
// F1286: Comparison Report Tests
// =============================================================================

/// F1286.1: Diff computed
#[test]
fn f1286_diff_computed() {
    let entry = ComparisonEntry::new("latency", 100.0, 110.0, "ms");

    let change = entry.percent_change();
    assert!((change - 10.0).abs() < 0.01); // 10% increase
}

/// F1286.2: Comparison report JSON
#[test]
fn f1286_comparison_json() {
    let mut report = ComparisonReport::new("Compare").with_labels("baseline", "current");
    report.add_entry("latency", 100.0, 110.0, "ms");

    let json = ReportExporter::comparison_to_json(&report);

    assert!(json.contains("\"title\":\"Compare\""));
    assert!(json.contains("\"baseline\":\"baseline\""));
    assert!(json.contains("\"change\":10.00"));
}

// =============================================================================
// F1287: Regression Flagging Tests
// =============================================================================

/// F1287.1: Threshold violations flagged
#[test]
fn f1287_regression_flagged() {
    let mut report = ComparisonReport::new("Test").with_threshold(5.0);
    report.add_entry("latency", 100.0, 120.0, "ms"); // 20% regression

    assert!(report.has_regressions());
    assert_eq!(report.get_regressions().len(), 1);
}

/// F1287.2: No regression below threshold
#[test]
fn f1287_no_regression_below_threshold() {
    let mut report = ComparisonReport::new("Test").with_threshold(10.0);
    report.add_entry("latency", 100.0, 105.0, "ms"); // 5% change

    assert!(!report.has_regressions());
}

/// F1287.3: Comparison entry is_regression
#[test]
fn f1287_is_regression() {
    let entry = ComparisonEntry::new("latency", 100.0, 115.0, "ms");
    assert!(entry.is_regression(10.0)); // 15% > 10%
    assert!(!entry.is_regression(20.0)); // 15% < 20%
}

// =============================================================================
// F1288: File Write Tests
// =============================================================================

/// F1288.1: Path creates file
#[test]
fn f1288_file_write() {
    let content = "test content";
    let path = std::env::temp_dir().join("cbtop_test_export.txt");

    let result = ReportExporter::write_to_file(content, &path);
    assert!(result.is_ok());

    // Verify content
    let read_content = std::fs::read_to_string(&path).unwrap();
    assert_eq!(read_content, content);

    // Cleanup
    std::fs::remove_file(path).ok();
}

// =============================================================================
// F1289: Format Selection Tests
// =============================================================================

/// F1289.1: Enum dispatch works
#[test]
fn f1289_format_dispatch() {
    let report = BenchmarkReport::new("Test");

    let json = ReportExporter::export_benchmark(&report, ExportFormat::Json);
    assert!(json.contains("{"));

    let csv = ReportExporter::export_benchmark(&report, ExportFormat::Csv);
    assert!(csv.contains("metric,value,unit"));

    let md = ReportExporter::export_benchmark(&report, ExportFormat::Markdown);
    assert!(md.contains("#"));

    let html = ReportExporter::export_benchmark(&report, ExportFormat::Html);
    assert!(html.contains("<!DOCTYPE"));
}

/// F1289.2: Format from string
#[test]
fn f1289_format_from_str() {
    assert_eq!(ExportFormat::parse("json"), Some(ExportFormat::Json));
    assert_eq!(ExportFormat::parse("csv"), Some(ExportFormat::Csv));
    assert_eq!(ExportFormat::parse("md"), Some(ExportFormat::Markdown));
    assert_eq!(ExportFormat::parse("html"), Some(ExportFormat::Html));
    assert_eq!(ExportFormat::parse("invalid"), None);
}

// =============================================================================
// F1290: Report Builder Tests
// =============================================================================

/// F1290.1: Fluent API works
#[test]
fn f1290_fluent_api() {
    let output = ReportBuilder::new(ReportType::Benchmark)
        .title("My Report")
        .metric("latency", 10.0, "ms")
        .metric("throughput", 1000.0, "ops/s")
        .format(ExportFormat::Json)
        .build();

    assert!(output.contains("My Report"));
    assert!(output.contains("latency"));
    assert!(output.contains("throughput"));
}

/// F1290.2: Builder with metadata
#[test]
fn f1290_builder_metadata() {
    let output = ReportBuilder::new(ReportType::Summary)
        .title("Summary")
        .metadata("version", "1.0")
        .format(ExportFormat::Json)
        .build();

    assert!(output.contains("\"version\":\"1.0\""));
}

/// F1290.3: Comparison builder
#[test]
fn f1290_comparison_builder() {
    let output = ReportBuilder::new(ReportType::Comparison)
        .title("Compare")
        .comparison("latency", 100.0, 110.0, "ms")
        .format(ExportFormat::Markdown)
        .build();

    assert!(output.contains("# Compare"));
    assert!(output.contains("latency"));
}

// =============================================================================
// Additional Tests
// =============================================================================

/// Test export format names
#[test]
fn test_format_names() {
    assert_eq!(ExportFormat::Json.name(), "json");
    assert_eq!(ExportFormat::Csv.name(), "csv");
    assert_eq!(ExportFormat::Markdown.name(), "markdown");
    assert_eq!(ExportFormat::Html.name(), "html");
}

/// Test export format extensions
#[test]
fn test_format_extensions() {
    assert_eq!(ExportFormat::Json.extension(), "json");
    assert_eq!(ExportFormat::Csv.extension(), "csv");
    assert_eq!(ExportFormat::Markdown.extension(), "md");
    assert_eq!(ExportFormat::Html.extension(), "html");
}

/// Test report type names
#[test]
fn test_report_type_names() {
    assert_eq!(ReportType::Benchmark.name(), "benchmark");
    assert_eq!(ReportType::Comparison.name(), "comparison");
    assert_eq!(ReportType::Regression.name(), "regression");
    assert_eq!(ReportType::Summary.name(), "summary");
}

/// Test benchmark metric creation
#[test]
fn test_benchmark_metric() {
    let metric = BenchmarkMetric::new("test", 42.0, "units");
    assert_eq!(metric.name, "test");
    assert_eq!(metric.value, 42.0);
    assert_eq!(metric.unit, "units");
}

/// Test comparison percent change edge cases
#[test]
fn test_percent_change_edge_cases() {
    // Zero baseline
    let entry = ComparisonEntry::new("test", 0.0, 10.0, "x");
    let change = entry.percent_change();
    assert_eq!(change, 100.0);

    // Both zero
    let entry2 = ComparisonEntry::new("test", 0.0, 0.0, "x");
    let change2 = entry2.percent_change();
    assert_eq!(change2, 0.0);
}

/// Test comparison markdown with regressions
#[test]
fn test_comparison_markdown_regressions() {
    let mut report = ComparisonReport::new("Test").with_threshold(5.0);
    report.add_entry("latency", 100.0, 120.0, "ms");

    let md = ReportExporter::comparison_to_markdown(&report);

    assert!(md.contains("## Regressions Detected"));
    assert!(md.contains("latency"));
}

/// Test benchmark report builder pattern
#[test]
fn test_benchmark_builder_pattern() {
    let report = BenchmarkReport::new("Test")
        .with_timestamp("2024-01-01")
        .with_workload("gemm")
        .with_backend("cuda")
        .with_size(1024);

    assert_eq!(report.title, "Test");
    assert_eq!(report.timestamp, "2024-01-01");
    assert_eq!(report.workload, "gemm");
    assert_eq!(report.backend, "cuda");
    assert_eq!(report.size, 1024);
}
