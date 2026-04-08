//! HTML report output for pv lint.

use provable_contracts::lint::rules::RuleSeverity;
use provable_contracts::lint::LintReport;

pub fn render_html(report: &LintReport) -> String {
    let mut html = String::new();
    html.push_str("<!DOCTYPE html><html><head><meta charset='utf-8'>");
    html.push_str("<title>pv lint report</title>");
    html.push_str("<style>");
    html.push_str("body{font-family:system-ui;max-width:900px;margin:2em auto;padding:0 1em}");
    html.push_str("h1{border-bottom:2px solid #333}.pass{color:#2a7}");
    html.push_str(".fail{color:#c33}.warn{color:#c80}.info{color:#38c}");
    html.push_str("table{border-collapse:collapse;width:100%}");
    html.push_str("th,td{border:1px solid #ddd;padding:6px 10px;text-align:left}");
    html.push_str("th{background:#f5f5f5}.new{background:#fff3cd}");
    html.push_str("</style></head><body>");

    // Header
    let status_class = if report.passed { "pass" } else { "fail" };
    let status_text = if report.passed { "PASS" } else { "FAIL" };
    html.push_str(&format!(
        "<h1>pv lint report \u{2014} <span class='{status_class}'>{status_text}</span></h1>"
    ));
    html.push_str(&format!("<p>Duration: {}ms</p>", report.total_duration_ms));

    // Gates table
    html.push_str(
        "<h2>Gates</h2><table><tr><th>#</th><th>Gate</th><th>Status</th><th>Duration</th></tr>",
    );
    for (i, gate) in report.gates.iter().enumerate() {
        let status = if gate.skipped {
            "skip"
        } else if gate.passed {
            "pass"
        } else {
            "fail"
        };
        let cls = if gate.passed || gate.skipped {
            "pass"
        } else {
            "fail"
        };
        html.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td class='{cls}'>{status}</td><td>{}ms</td></tr>",
            i + 1,
            gate.name,
            gate.duration_ms
        ));
    }
    html.push_str("</table>");

    // Findings table
    let active: Vec<_> = report.findings.iter().filter(|f| !f.suppressed).collect();
    if !active.is_empty() {
        html.push_str(&format!("<h2>Findings ({})</h2>", active.len()));
        html.push_str(
            "<table><tr><th>Rule</th><th>Severity</th><th>File</th><th>Message</th></tr>",
        );
        for f in &active {
            let sev_cls = match f.severity {
                RuleSeverity::Error => "fail",
                RuleSeverity::Warning => "warn",
                _ => "info",
            };
            let sev = match f.severity {
                RuleSeverity::Error => "ERROR",
                RuleSeverity::Warning => "WARN",
                RuleSeverity::Info => "INFO",
                RuleSeverity::Off => "OFF",
            };
            let new_cls = if f.is_new { " class='new'" } else { "" };
            let new_badge = if f.is_new { " [NEW]" } else { "" };
            html.push_str(&format!(
                "<tr{new_cls}><td>{}</td><td class='{sev_cls}'>{sev}</td><td>{}</td><td>{}{new_badge}</td></tr>",
                f.rule_id, f.file, f.message
            ));
        }
        html.push_str("</table>");
    }

    // Summary
    let errors = active
        .iter()
        .filter(|f| f.severity == RuleSeverity::Error)
        .count();
    let warnings = active
        .iter()
        .filter(|f| f.severity == RuleSeverity::Warning)
        .count();
    html.push_str(&format!(
        "<h2>Summary</h2><p>{errors} errors, {warnings} warnings, {} suppressed</p>",
        report.findings.len() - active.len()
    ));

    html.push_str("</body></html>");
    html
}
