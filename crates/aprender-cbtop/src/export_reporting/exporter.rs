//! Report exporter for multiple output formats.

use std::io::Write;
use std::path::Path;

use super::types::{BenchmarkReport, ComparisonReport, ExportFormat};

/// Report exporter
#[derive(Debug)]
pub struct ReportExporter;

impl ReportExporter {
    /// Export benchmark report to JSON
    pub fn benchmark_to_json(report: &BenchmarkReport) -> String {
        let metrics_json: Vec<String> = report
            .metrics
            .iter()
            .map(|m| {
                format!(
                    r#"{{"name":"{}","value":{},"unit":"{}"}}"#,
                    m.name, m.value, m.unit
                )
            })
            .collect();

        let metadata_json: Vec<String> = report
            .metadata
            .iter()
            .map(|(k, v)| format!(r#""{}":"{}""#, k, v))
            .collect();

        format!(
            r#"{{"title":"{}","timestamp":"{}","workload":"{}","backend":"{}","size":{},"metrics":[{}],"metadata":{{{}}}}}"#,
            report.title,
            report.timestamp,
            report.workload,
            report.backend,
            report.size,
            metrics_json.join(","),
            metadata_json.join(",")
        )
    }

    /// Export benchmark report to CSV
    pub fn benchmark_to_csv(report: &BenchmarkReport) -> String {
        let mut lines = vec!["metric,value,unit".to_string()];
        for m in &report.metrics {
            lines.push(format!("{},{},{}", m.name, m.value, m.unit));
        }
        lines.join("\n")
    }

    /// Export benchmark report to Markdown
    pub fn benchmark_to_markdown(report: &BenchmarkReport) -> String {
        let mut md = format!("# {}\n\n", report.title);
        md.push_str(&format!("**Workload**: {}  \n", report.workload));
        md.push_str(&format!("**Backend**: {}  \n", report.backend));
        md.push_str(&format!("**Size**: {}  \n\n", report.size));

        md.push_str("## Metrics\n\n");
        md.push_str("| Metric | Value | Unit |\n");
        md.push_str("|--------|-------|------|\n");
        for m in &report.metrics {
            md.push_str(&format!("| {} | {:.4} | {} |\n", m.name, m.value, m.unit));
        }

        md
    }

    /// Export benchmark report to HTML
    pub fn benchmark_to_html(report: &BenchmarkReport) -> String {
        let mut html = String::from("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str(&format!("<title>{}</title>\n", report.title));
        html.push_str("<style>table{border-collapse:collapse;}th,td{border:1px solid #ccc;padding:8px;}</style>\n");
        html.push_str("</head>\n<body>\n");
        html.push_str(&format!("<h1>{}</h1>\n", report.title));
        html.push_str(&format!(
            "<p><strong>Workload:</strong> {}</p>\n",
            report.workload
        ));
        html.push_str(&format!(
            "<p><strong>Backend:</strong> {}</p>\n",
            report.backend
        ));
        html.push_str(&format!("<p><strong>Size:</strong> {}</p>\n", report.size));

        html.push_str("<h2>Metrics</h2>\n");
        html.push_str("<table>\n<tr><th>Metric</th><th>Value</th><th>Unit</th></tr>\n");
        for m in &report.metrics {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{:.4}</td><td>{}</td></tr>\n",
                m.name, m.value, m.unit
            ));
        }
        html.push_str("</table>\n</body>\n</html>");

        html
    }

    /// Export comparison report to JSON
    pub fn comparison_to_json(report: &ComparisonReport) -> String {
        let entries_json: Vec<String> = report
            .entries
            .iter()
            .map(|e| {
                format!(
                    r#"{{"metric":"{}","baseline":{},"current":{},"change":{:.2},"unit":"{}"}}"#,
                    e.metric,
                    e.baseline,
                    e.current,
                    e.percent_change(),
                    e.unit
                )
            })
            .collect();

        format!(
            r#"{{"title":"{}","baseline":"{}","current":"{}","has_regressions":{},"entries":[{}]}}"#,
            report.title,
            report.baseline_label,
            report.current_label,
            report.has_regressions(),
            entries_json.join(",")
        )
    }

    /// Export comparison report to Markdown
    pub fn comparison_to_markdown(report: &ComparisonReport) -> String {
        let mut md = format!("# {}\n\n", report.title);
        md.push_str(&format!(
            "Comparing **{}** vs **{}**\n\n",
            report.baseline_label, report.current_label
        ));

        md.push_str("| Metric | Baseline | Current | Change |\n");
        md.push_str("|--------|----------|---------|--------|\n");
        for e in &report.entries {
            let change = e.percent_change();
            let indicator = if change.abs() > report.regression_threshold {
                "\u{26a0}\u{fe0f}"
            } else {
                "\u{2705}"
            };
            md.push_str(&format!(
                "| {} | {:.4} {} | {:.4} {} | {:.2}% {} |\n",
                e.metric, e.baseline, e.unit, e.current, e.unit, change, indicator
            ));
        }

        if report.has_regressions() {
            md.push_str("\n## Regressions Detected\n\n");
            for e in report.get_regressions() {
                md.push_str(&format!(
                    "- **{}**: {:.2}% change\n",
                    e.metric,
                    e.percent_change()
                ));
            }
        }

        md
    }

    /// Export to specified format
    pub fn export_benchmark(report: &BenchmarkReport, format: ExportFormat) -> String {
        match format {
            ExportFormat::Json => Self::benchmark_to_json(report),
            ExportFormat::Csv => Self::benchmark_to_csv(report),
            ExportFormat::Markdown => Self::benchmark_to_markdown(report),
            ExportFormat::Html => Self::benchmark_to_html(report),
        }
    }

    /// Export comparison to specified format
    pub fn export_comparison(report: &ComparisonReport, format: ExportFormat) -> String {
        match format {
            ExportFormat::Json => Self::comparison_to_json(report),
            ExportFormat::Markdown => Self::comparison_to_markdown(report),
            ExportFormat::Csv | ExportFormat::Html => Self::comparison_to_json(report),
        }
    }

    /// Write report to file
    pub fn write_to_file<P: AsRef<Path>>(content: &str, path: P) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        file.write_all(content.as_bytes())?;
        Ok(())
    }
}
