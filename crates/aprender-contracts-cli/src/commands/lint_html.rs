//! HTML report output for pv lint.

use provable_contracts::lint::LintReport;
use provable_contracts::lint::rules::RuleSeverity;

pub fn render_html(report: &LintReport) -> String {
    let mut html = String::new();
    html.push_str(HTML_HEADER);

    render_header(&mut html, report);
    render_gates(&mut html, report);

    let active: Vec<_> = report.findings.iter().filter(|f| !f.suppressed).collect();
    render_findings(&mut html, &active);
    render_summary(&mut html, report, &active);

    html.push_str("</body></html>");
    html
}

const HTML_HEADER: &str = concat!(
    "<!DOCTYPE html><html><head><meta charset='utf-8'>",
    "<title>pv lint report</title>",
    "<style>",
    "body{font-family:system-ui;max-width:900px;margin:2em auto;padding:0 1em}",
    "h1{border-bottom:2px solid #333}.pass{color:#2a7}",
    ".fail{color:#c33}.warn{color:#c80}.info{color:#38c}",
    "table{border-collapse:collapse;width:100%}",
    "th,td{border:1px solid #ddd;padding:6px 10px;text-align:left}",
    "th{background:#f5f5f5}.new{background:#fff3cd}",
    "</style></head><body>",
);

fn render_header(html: &mut String, report: &LintReport) {
    let status_class = if report.passed { "pass" } else { "fail" };
    let status_text = if report.passed { "PASS" } else { "FAIL" };
    html.push_str(&format!(
        "<h1>pv lint report \u{2014} <span class='{status_class}'>{status_text}</span></h1>"
    ));
    html.push_str(&format!("<p>Duration: {}ms</p>", report.total_duration_ms));
}

fn render_gates(html: &mut String, report: &LintReport) {
    html.push_str(
        "<h2>Gates</h2><table><tr><th>#</th><th>Gate</th><th>Status</th><th>Duration</th></tr>",
    );
    for (i, gate) in report.gates.iter().enumerate() {
        let (status, cls) = gate_status_labels(gate.passed, gate.skipped);
        html.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td class='{cls}'>{status}</td><td>{}ms</td></tr>",
            i + 1,
            gate.name,
            gate.duration_ms
        ));
    }
    html.push_str("</table>");
}

fn gate_status_labels(passed: bool, skipped: bool) -> (&'static str, &'static str) {
    if skipped {
        ("skip", "pass")
    } else if passed {
        ("pass", "pass")
    } else {
        ("fail", "fail")
    }
}

fn render_findings(html: &mut String, active: &[&provable_contracts::lint::finding::LintFinding]) {
    if active.is_empty() {
        return;
    }
    html.push_str(&format!("<h2>Findings ({})</h2>", active.len()));
    html.push_str("<table><tr><th>Rule</th><th>Severity</th><th>File</th><th>Message</th></tr>");
    for f in active {
        let (sev, sev_cls) = severity_labels(f.severity);
        let new_cls = if f.is_new { " class='new'" } else { "" };
        let new_badge = if f.is_new { " [NEW]" } else { "" };
        html.push_str(&format!(
            "<tr{new_cls}><td>{}</td><td class='{sev_cls}'>{sev}</td><td>{}</td><td>{}{new_badge}</td></tr>",
            f.rule_id, f.file, f.message
        ));
    }
    html.push_str("</table>");
}

fn severity_labels(sev: RuleSeverity) -> (&'static str, &'static str) {
    match sev {
        RuleSeverity::Error => ("ERROR", "fail"),
        RuleSeverity::Warning => ("WARN", "warn"),
        RuleSeverity::Info => ("INFO", "info"),
        RuleSeverity::Off => ("OFF", "info"),
    }
}

fn render_summary(
    html: &mut String,
    report: &LintReport,
    active: &[&provable_contracts::lint::finding::LintFinding],
) {
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
}
