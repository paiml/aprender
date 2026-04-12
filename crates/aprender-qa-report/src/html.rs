//! HTML Dashboard Generator
//!
//! Generates interactive HTML dashboards for MQS results.

use apr_qa_runner::EvidenceCollector;

use crate::error::Result;
use crate::mqs::{CategoryScores, MqsScore};
use crate::popperian::PopperianScore;

/// HTML dashboard generator
#[derive(Debug, Default)]
pub struct HtmlDashboard {
    /// Dashboard title
    title: String,
    /// Include interactive charts
    include_charts: bool,
}

impl HtmlDashboard {
    /// Create a new dashboard generator
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            include_charts: true,
        }
    }

    /// Disable interactive charts
    #[must_use]
    pub fn without_charts(mut self) -> Self {
        self.include_charts = false;
        self
    }

    /// Generate HTML dashboard
    ///
    /// # Errors
    ///
    /// Returns an error if HTML generation fails.
    pub fn generate(
        &self,
        mqs: &MqsScore,
        popperian: &PopperianScore,
        _evidence: &EvidenceCollector,
    ) -> Result<String> {
        let grade_color = Self::grade_color(&mqs.grade);
        let pass_rate = if mqs.total_tests > 0 {
            (mqs.tests_passed as f64 / mqs.total_tests as f64) * 100.0
        } else {
            0.0
        };

        let html = format!(
            r##"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --bg-color: #1a1a2e;
            --card-bg: #16213e;
            --text-color: #eee;
            --accent: #0f3460;
            --success: #00d26a;
            --warning: #ffc107;
            --danger: #ff4757;
            --grade-color: {grade_color};
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ margin-bottom: 1rem; color: #fff; }}
        .model-id {{ color: #888; font-size: 0.9em; }}
        .dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin-top: 2rem; }}
        .card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .card h3 {{ color: #aaa; font-size: 0.85em; text-transform: uppercase; margin-bottom: 0.5rem; }}
        .score-large {{
            font-size: 3rem;
            font-weight: bold;
            color: var(--grade-color);
        }}
        .grade {{ font-size: 4rem; font-weight: bold; color: var(--grade-color); }}
        .stat {{ display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.1); }}
        .stat:last-child {{ border-bottom: none; }}
        .stat-label {{ color: #888; }}
        .stat-value {{ font-weight: 600; }}
        .progress-bar {{ background: rgba(255,255,255,0.1); border-radius: 4px; height: 8px; overflow: hidden; margin-top: 0.5rem; }}
        .progress-fill {{ height: 100%; transition: width 0.3s; }}
        .success {{ background: var(--success); }}
        .warning {{ background: var(--warning); }}
        .danger {{ background: var(--danger); }}
        .gateway {{ display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem 0; }}
        .gateway-icon {{ width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; }}
        .gateway-pass {{ background: var(--success); color: #000; }}
        .gateway-fail {{ background: var(--danger); color: #fff; }}
        .category-bar {{ display: flex; align-items: center; gap: 1rem; margin: 0.75rem 0; }}
        .category-name {{ width: 60px; font-size: 0.85em; color: #888; }}
        .category-track {{ flex: 1; background: rgba(255,255,255,0.1); border-radius: 4px; height: 12px; overflow: hidden; }}
        .category-fill {{ height: 100%; background: linear-gradient(90deg, var(--success), #00ff88); border-radius: 4px; }}
        .category-value {{ width: 50px; text-align: right; font-size: 0.85em; }}
        .falsification {{ background: rgba(255,71,87,0.1); border-left: 3px solid var(--danger); padding: 0.75rem; margin: 0.5rem 0; border-radius: 0 4px 4px 0; }}
        .falsification-gate {{ font-weight: 600; color: var(--danger); }}
        .black-swan {{ background: rgba(255,71,87,0.2); }}
        .timestamp {{ color: #666; font-size: 0.8em; margin-top: 2rem; text-align: center; }}
        @media (max-width: 600px) {{
            body {{ padding: 1rem; }}
            .score-large {{ font-size: 2rem; }}
            .grade {{ font-size: 3rem; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p class="model-id">Model: {model_id}</p>

        <div class="dashboard">
            <!-- MQS Score Card -->
            <div class="card">
                <h3>MQS Score</h3>
                <div class="score-large">{normalized_score:.1}</div>
                <div class="stat">
                    <span class="stat-label">Raw Score</span>
                    <span class="stat-value">{raw_score}/1000</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Status</span>
                    <span class="stat-value">{qualification_status}</span>
                </div>
            </div>

            <!-- Grade Card -->
            <div class="card">
                <h3>Grade</h3>
                <div class="grade">{grade}</div>
                <div class="stat">
                    <span class="stat-label">Production Ready</span>
                    <span class="stat-value">{production_ready}</span>
                </div>
            </div>

            <!-- Pass Rate Card -->
            <div class="card">
                <h3>Test Results</h3>
                <div class="stat">
                    <span class="stat-label">Total Tests</span>
                    <span class="stat-value">{total_tests}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Passed</span>
                    <span class="stat-value" style="color: var(--success)">{tests_passed}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Failed</span>
                    <span class="stat-value" style="color: var(--danger)">{tests_failed}</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill {pass_rate_class}" style="width: {pass_rate:.1}%"></div>
                </div>
            </div>

            <!-- Gateways Card -->
            <div class="card">
                <h3>Gateway Checks</h3>
                {gateway_html}
            </div>

            <!-- Categories Card -->
            <div class="card" style="grid-column: span 2;">
                <h3>Category Breakdown</h3>
                {categories_html}
            </div>

            <!-- Popperian Card -->
            <div class="card">
                <h3>Popperian Analysis</h3>
                <div class="stat">
                    <span class="stat-label">Hypotheses Tested</span>
                    <span class="stat-value">{hypotheses_tested}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Corroborated</span>
                    <span class="stat-value">{corroborated}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Falsified</span>
                    <span class="stat-value">{falsified}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Black Swans</span>
                    <span class="stat-value">{black_swans}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Confidence</span>
                    <span class="stat-value">{confidence:.1}%</span>
                </div>
            </div>

            <!-- Falsifications Card -->
            <div class="card" style="grid-column: span 2;">
                <h3>Falsifications</h3>
                {falsifications_html}
            </div>
        </div>

        <p class="timestamp">Generated: {timestamp}</p>
    </div>
</body>
</html>"##,
            title = Self::escape_html(&self.title),
            model_id = Self::escape_html(&mqs.model_id),
            grade_color = grade_color,
            normalized_score = mqs.normalized_score,
            raw_score = mqs.raw_score,
            grade = Self::escape_html(&mqs.grade),
            qualification_status = if mqs.qualifies() {
                "Qualified"
            } else {
                "Not Qualified"
            },
            production_ready = if mqs.is_production_ready() {
                "Yes"
            } else {
                "No"
            },
            total_tests = mqs.total_tests,
            tests_passed = mqs.tests_passed,
            tests_failed = mqs.tests_failed,
            pass_rate = pass_rate,
            pass_rate_class = if pass_rate >= 90.0 {
                "success"
            } else if pass_rate >= 70.0 {
                "warning"
            } else {
                "danger"
            },
            gateway_html = self.render_gateways(mqs),
            categories_html = self.render_categories(mqs),
            hypotheses_tested = popperian.hypotheses_tested,
            corroborated = popperian.corroborated,
            falsified = popperian.falsified,
            black_swans = popperian.black_swan_count,
            confidence = popperian.confidence_level * 100.0,
            falsifications_html = self.render_falsifications(popperian),
            timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
        );

        Ok(html)
    }

    /// Render gateway checks HTML
    fn render_gateways(&self, mqs: &MqsScore) -> String {
        let mut html = String::new();
        for gateway in &mqs.gateways {
            let (icon_class, icon) = if gateway.passed {
                ("gateway-pass", "✓")
            } else {
                ("gateway-fail", "✗")
            };
            html.push_str(&format!(
                r#"<div class="gateway">
                    <div class="gateway-icon {}">{}</div>
                    <span>{}: {}</span>
                </div>"#,
                icon_class,
                icon,
                Self::escape_html(&gateway.id),
                Self::escape_html(&gateway.description)
            ));
        }
        if html.is_empty() {
            html = "<p>No gateway checks recorded</p>".to_string();
        }
        html
    }

    /// Render category breakdown HTML
    fn render_categories(&self, mqs: &MqsScore) -> String {
        let categories = [
            ("QUAL", mqs.categories.qual, CategoryScores::MAX_QUAL),
            ("PERF", mqs.categories.perf, CategoryScores::MAX_PERF),
            ("STAB", mqs.categories.stab, CategoryScores::MAX_STAB),
            ("COMP", mqs.categories.comp, CategoryScores::MAX_COMP),
            ("EDGE", mqs.categories.edge, CategoryScores::MAX_EDGE),
            ("REGR", mqs.categories.regr, CategoryScores::MAX_REGR),
        ];

        let mut html = String::new();
        for (name, score, max) in categories {
            let pct = if max > 0 {
                (score as f64 / max as f64) * 100.0
            } else {
                0.0
            };
            html.push_str(&format!(
                r#"<div class="category-bar">
                    <span class="category-name">{}</span>
                    <div class="category-track">
                        <div class="category-fill" style="width: {:.1}%"></div>
                    </div>
                    <span class="category-value">{}/{}</span>
                </div>"#,
                name, pct, score, max
            ));
        }
        html
    }

    /// Render falsifications HTML
    fn render_falsifications(&self, popperian: &PopperianScore) -> String {
        if popperian.falsifications.is_empty() {
            return "<p style=\"color: var(--success)\">No falsifications - all hypotheses corroborated!</p>".to_string();
        }

        let mut html = String::new();
        for f in popperian.falsifications.iter().take(10) {
            let class = if f.is_black_swan {
                "falsification black-swan"
            } else {
                "falsification"
            };
            html.push_str(&format!(
                r#"<div class="{}">
                    <span class="falsification-gate">{}</span>
                    {}: {}
                    {}
                </div>"#,
                class,
                Self::escape_html(&f.gate_id),
                Self::escape_html(&f.hypothesis),
                Self::escape_html(&f.evidence),
                if f.is_black_swan {
                    " <strong>(Black Swan)</strong>"
                } else {
                    ""
                }
            ));
        }

        if popperian.falsifications.len() > 10 {
            html.push_str(&format!(
                "<p>... and {} more falsifications</p>",
                popperian.falsifications.len() - 10
            ));
        }

        html
    }

    /// Get color for grade
    fn grade_color(grade: &str) -> &'static str {
        match grade {
            "A+" | "A" | "A-" => "#00d26a",
            "B+" | "B" | "B-" => "#7bed9f",
            "C+" | "C" | "C-" => "#ffc107",
            "D+" | "D" | "D-" => "#ff9f43",
            _ => "#ff4757",
        }
    }

    /// Escape HTML special characters
    fn escape_html(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&#x27;")
    }
}

#[cfg(test)]
#[path = "html_tests.rs"]
mod html_tests;
