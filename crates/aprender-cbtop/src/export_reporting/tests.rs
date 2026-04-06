use super::*;

#[test]
fn test_export_format_names() {
    assert_eq!(ExportFormat::Json.name(), "json");
    assert_eq!(ExportFormat::Csv.name(), "csv");
    assert_eq!(ExportFormat::Markdown.name(), "markdown");
    assert_eq!(ExportFormat::Html.name(), "html");
}

#[test]
fn test_export_format_extensions() {
    assert_eq!(ExportFormat::Json.extension(), "json");
    assert_eq!(ExportFormat::Markdown.extension(), "md");
}

#[test]
fn test_format_from_str() {
    assert_eq!(ExportFormat::parse("json"), Some(ExportFormat::Json));
    assert_eq!(ExportFormat::parse("md"), Some(ExportFormat::Markdown));
    assert_eq!(ExportFormat::parse("invalid"), None);
}

#[test]
fn test_benchmark_to_json() {
    let mut report = BenchmarkReport::new("Test Report");
    report.add_metric("latency", 10.5, "ms");
    report.add_metric("throughput", 1000.0, "ops/s");

    let json = ReportExporter::benchmark_to_json(&report);
    assert!(json.contains("\"title\":\"Test Report\""));
    assert!(json.contains("\"latency\""));
    assert!(json.contains("10.5"));
}

#[test]
fn test_benchmark_to_csv() {
    let mut report = BenchmarkReport::new("Test Report");
    report.add_metric("latency", 10.5, "ms");

    let csv = ReportExporter::benchmark_to_csv(&report);
    assert!(csv.contains("metric,value,unit"));
    assert!(csv.contains("latency,10.5,ms"));
}

#[test]
fn test_benchmark_to_markdown() {
    let mut report = BenchmarkReport::new("Test Report");
    report.add_metric("latency", 10.5, "ms");

    let md = ReportExporter::benchmark_to_markdown(&report);
    assert!(md.contains("# Test Report"));
    assert!(md.contains("| Metric |"));
}

#[test]
fn test_benchmark_to_html() {
    let mut report = BenchmarkReport::new("Test Report");
    report.add_metric("latency", 10.5, "ms");

    let html = ReportExporter::benchmark_to_html(&report);
    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("<h1>Test Report</h1>"));
    assert!(html.contains("</html>"));
}

#[test]
fn test_comparison_percent_change() {
    let entry = ComparisonEntry::new("latency", 100.0, 110.0, "ms");
    assert!((entry.percent_change() - 10.0).abs() < 0.001);
}

#[test]
fn test_comparison_regression() {
    let entry = ComparisonEntry::new("latency", 100.0, 115.0, "ms");
    assert!(entry.is_regression(10.0));

    let ok_entry = ComparisonEntry::new("latency", 100.0, 103.0, "ms");
    assert!(!ok_entry.is_regression(10.0));
}

#[test]
fn test_comparison_to_json() {
    let mut report = ComparisonReport::new("Test Comparison");
    report.add_entry("latency", 100.0, 110.0, "ms");

    let json = ReportExporter::comparison_to_json(&report);
    assert!(json.contains("\"title\":\"Test Comparison\""));
    assert!(json.contains("\"change\":10.00"));
}

#[test]
fn test_report_builder() {
    let output = ReportBuilder::new(ReportType::Benchmark)
        .title("My Report")
        .metric("latency", 10.0, "ms")
        .format(ExportFormat::Json)
        .build();

    assert!(output.contains("My Report"));
    assert!(output.contains("latency"));
}

#[test]
fn test_comparison_builder() {
    let output = ReportBuilder::new(ReportType::Comparison)
        .title("Compare")
        .comparison("latency", 100.0, 110.0, "ms")
        .format(ExportFormat::Markdown)
        .build();

    assert!(output.contains("# Compare"));
}
