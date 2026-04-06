/// Report generation for migration analysis
use crate::types::{ProjectAnalysis, WorkflowState};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Migration report data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationReport {
    pub project_name: String,
    pub analysis: ProjectAnalysis,
    pub workflow: WorkflowState,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl MigrationReport {
    pub fn new(project_name: String, analysis: ProjectAnalysis, workflow: WorkflowState) -> Self {
        Self { project_name, analysis, workflow, timestamp: chrono::Utc::now() }
    }

    /// Generate HTML report
    pub fn to_html(&self) -> String {
        let mut html = String::new();

        // HTML header
        html.push_str("<!DOCTYPE html>\n");
        html.push_str("<html lang=\"en\">\n");
        html.push_str("<head>\n");
        html.push_str("  <meta charset=\"UTF-8\">\n");
        html.push_str(
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n",
        );
        html.push_str(&format!("  <title>Migration Report - {}</title>\n", self.project_name));
        html.push_str("  <style>\n");
        html.push_str(include_str!("report_style.css"));
        html.push_str("  </style>\n");
        html.push_str("</head>\n");
        html.push_str("<body>\n");

        // Header
        html.push_str(&format!(
            "<header><h1>Migration Report: {}</h1><p>Generated: {}</p></header>\n",
            self.project_name,
            format_timestamp(self.timestamp, "%Y-%m-%d %H:%M:%S UTC")
        ));

        // Summary section
        html.push_str("<section class=\"summary\">\n");
        html.push_str("<h2>Summary</h2>\n");
        html.push_str("<div class=\"stats\">\n");
        html.push_str(&format!(
            "<div class=\"stat\"><span class=\"label\">Total Files</span><span class=\"value\">{}</span></div>\n",
            self.analysis.total_files
        ));
        html.push_str(&format!(
            "<div class=\"stat\"><span class=\"label\">Total Lines</span><span class=\"value\">{}</span></div>\n",
            self.analysis.total_lines
        ));
        if let Some(lang) = &self.analysis.primary_language {
            html.push_str(&format!(
                "<div class=\"stat\"><span class=\"label\">Primary Language</span><span class=\"value\">{}</span></div>\n",
                lang
            ));
        }
        if let Some(score) = self.analysis.tdg_score {
            html.push_str(&format!(
                "<div class=\"stat\"><span class=\"label\">TDG Score</span><span class=\"value\">{}</span></div>\n",
                format_tdg_score(score)
            ));
        }
        html.push_str("</div>\n");
        html.push_str("</section>\n");

        // Languages section
        if !self.analysis.languages.is_empty() {
            html.push_str("<section class=\"languages\">\n");
            html.push_str("<h2>Languages</h2>\n");
            html.push_str("<table>\n");
            html.push_str("<thead><tr><th>Language</th><th>Files</th><th>Lines</th><th>Percentage</th></tr></thead>\n");
            html.push_str("<tbody>\n");
            for lang_stat in &self.analysis.languages {
                html.push_str(&format!(
                    "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.1}%</td></tr>\n",
                    lang_stat.language,
                    lang_stat.file_count,
                    lang_stat.line_count,
                    lang_stat.percentage
                ));
            }
            html.push_str("</tbody>\n");
            html.push_str("</table>\n");
            html.push_str("</section>\n");
        }

        // Dependencies section
        if !self.analysis.dependencies.is_empty() {
            html.push_str("<section class=\"dependencies\">\n");
            html.push_str("<h2>Dependencies</h2>\n");
            html.push_str("<ul>\n");
            for dep in &self.analysis.dependencies {
                html.push_str(&format!(
                    "<li><strong>{}</strong>{} - {:?}</li>\n",
                    dep.manager,
                    format_dep_count(dep.count),
                    dep.file_path
                ));
            }
            html.push_str("</ul>\n");
            html.push_str("</section>\n");
        }

        // Workflow progress section
        html.push_str("<section class=\"workflow\">\n");
        html.push_str("<h2>Workflow Progress</h2>\n");
        html.push_str(&format!(
            "<p class=\"progress\">{:.0}% complete</p>\n",
            self.workflow.progress_percentage()
        ));
        html.push_str("<table>\n");
        html.push_str("<thead><tr><th>Phase</th><th>Status</th><th>Started</th><th>Completed</th></tr></thead>\n");
        html.push_str("<tbody>\n");
        for phase in crate::types::WorkflowPhase::all() {
            if let Some(info) = self.workflow.phases.get(&phase) {
                let status_class = match info.status {
                    crate::types::PhaseStatus::Completed => "completed",
                    crate::types::PhaseStatus::InProgress => "in-progress",
                    crate::types::PhaseStatus::Failed => "failed",
                    crate::types::PhaseStatus::NotStarted => "not-started",
                };
                html.push_str(&format!("<tr class=\"{}\">\n", status_class));
                html.push_str(&format!("<td>{}</td>\n", phase));
                html.push_str(&format!("<td>{}</td>\n", info.status));
                html.push_str(&format!(
                    "<td>{}</td>\n",
                    format_timestamp_or_dash(info.started_at, "%Y-%m-%d %H:%M:%S")
                ));
                html.push_str(&format!(
                    "<td>{}</td>\n",
                    format_timestamp_or_dash(info.completed_at, "%Y-%m-%d %H:%M:%S")
                ));
                html.push_str("</tr>\n");
            }
        }
        html.push_str("</tbody>\n");
        html.push_str("</table>\n");
        html.push_str("</section>\n");

        // Recommendations section
        html.push_str("<section class=\"recommendations\">\n");
        html.push_str("<h2>Recommendations</h2>\n");
        html.push_str("<ul>\n");
        if let Some(transpiler) = self.analysis.recommend_transpiler() {
            html.push_str(&format!(
                "<li>Use <strong>{}</strong> for transpilation</li>\n",
                transpiler
            ));
        }
        if self.analysis.has_ml_dependencies() {
            html.push_str("<li>Consider <strong>Aprender</strong> for ML algorithms and <strong>Realizar</strong> for inference</li>\n");
        }
        if needs_refactoring(self.analysis.tdg_score) {
            html.push_str("<li>TDG score below 85 - consider refactoring before migration</li>\n");
        }
        html.push_str("</ul>\n");
        html.push_str("</section>\n");

        // Footer
        html.push_str("<footer>\n");
        html.push_str("<p>Generated by Batuta - Sovereign AI Stack</p>\n");
        html.push_str(
            "<p><a href=\"https://github.com/paiml/Batuta\">github.com/paiml/Batuta</a></p>\n",
        );
        html.push_str("</footer>\n");

        html.push_str("</body>\n");
        html.push_str("</html>\n");

        html
    }

    /// Generate Markdown report
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        // Header
        md.push_str(&format!("# Migration Report: {}\n\n", self.project_name));
        md.push_str(&format!(
            "**Generated:** {}\n\n",
            format_timestamp(self.timestamp, "%Y-%m-%d %H:%M:%S UTC")
        ));

        // Summary
        md.push_str("## Summary\n\n");
        md.push_str(&format!("- **Total Files:** {}\n", self.analysis.total_files));
        md.push_str(&format!("- **Total Lines:** {}\n", self.analysis.total_lines));
        if let Some(lang) = &self.analysis.primary_language {
            md.push_str(&format!("- **Primary Language:** {}\n", lang));
        }
        if let Some(score) = self.analysis.tdg_score {
            md.push_str(&format!("- **TDG Score:** {}\n", format_tdg_score(score)));
        }
        md.push('\n');

        // Languages
        if !self.analysis.languages.is_empty() {
            md.push_str("## Languages\n\n");
            md.push_str("| Language | Files | Lines | Percentage |\n");
            md.push_str("|----------|-------|-------|------------|\n");
            for lang_stat in &self.analysis.languages {
                md.push_str(&format!(
                    "| {} | {} | {} | {:.1}% |\n",
                    lang_stat.language,
                    lang_stat.file_count,
                    lang_stat.line_count,
                    lang_stat.percentage
                ));
            }
            md.push('\n');
        }

        // Dependencies
        if !self.analysis.dependencies.is_empty() {
            md.push_str("## Dependencies\n\n");
            for dep in &self.analysis.dependencies {
                md.push_str(&format!(
                    "- **{}**{} - `{:?}`\n",
                    dep.manager,
                    format_dep_count(dep.count),
                    dep.file_path
                ));
            }
            md.push('\n');
        }

        // Workflow
        md.push_str("## Workflow Progress\n\n");
        md.push_str(&format!(
            "**Overall:** {:.0}% complete\n\n",
            self.workflow.progress_percentage()
        ));
        md.push_str("| Phase | Status | Started | Completed |\n");
        md.push_str("|-------|--------|---------|----------|\n");
        for phase in crate::types::WorkflowPhase::all() {
            if let Some(info) = self.workflow.phases.get(&phase) {
                md.push_str(&format!(
                    "| {} | {} | {} | {} |\n",
                    phase,
                    info.status,
                    format_timestamp_or_dash(info.started_at, "%Y-%m-%d %H:%M"),
                    format_timestamp_or_dash(info.completed_at, "%Y-%m-%d %H:%M")
                ));
            }
        }
        md.push('\n');

        // Recommendations
        md.push_str("## Recommendations\n\n");
        if let Some(transpiler) = self.analysis.recommend_transpiler() {
            md.push_str(&format!("- Use **{}** for transpilation\n", transpiler));
        }
        if self.analysis.has_ml_dependencies() {
            md.push_str(
                "- Consider **Aprender** for ML algorithms and **Realizar** for inference\n",
            );
        }
        if needs_refactoring(self.analysis.tdg_score) {
            md.push_str("- TDG score below 85 - consider refactoring before migration\n");
        }
        md.push('\n');

        // Footer
        md.push_str("---\n\n");
        md.push_str("*Generated by Batuta - Sovereign AI Stack*  \n");
        md.push_str("https://github.com/paiml/Batuta\n");

        md
    }

    /// Generate JSON report
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Generate plain text report
    pub fn to_text(&self) -> String {
        let mut text = String::new();

        // Header
        text.push_str(&format!("MIGRATION REPORT: {}\n", self.project_name));
        text.push_str(&format!(
            "Generated: {}\n",
            format_timestamp(self.timestamp, "%Y-%m-%d %H:%M:%S UTC")
        ));
        text.push_str(&"=".repeat(80));
        text.push_str("\n\n");

        // Summary
        text.push_str("SUMMARY\n");
        text.push_str(&"-".repeat(80));
        text.push('\n');
        text.push_str(&format!("Total Files: {}\n", self.analysis.total_files));
        text.push_str(&format!("Total Lines: {}\n", self.analysis.total_lines));
        if let Some(lang) = &self.analysis.primary_language {
            text.push_str(&format!("Primary Language: {}\n", lang));
        }
        if let Some(score) = self.analysis.tdg_score {
            text.push_str(&format!("TDG Score: {}\n", format_tdg_score(score)));
        }
        text.push('\n');

        // Languages
        if !self.analysis.languages.is_empty() {
            text.push_str("LANGUAGES\n");
            text.push_str(&"-".repeat(80));
            text.push('\n');
            for lang_stat in &self.analysis.languages {
                text.push_str(&format!(
                    "{:15} {:8} files  {:10} lines  {:5.1}%\n",
                    format!("{}", lang_stat.language),
                    lang_stat.file_count,
                    lang_stat.line_count,
                    lang_stat.percentage
                ));
            }
            text.push('\n');
        }

        // Dependencies
        if !self.analysis.dependencies.is_empty() {
            text.push_str("DEPENDENCIES\n");
            text.push_str(&"-".repeat(80));
            text.push('\n');
            for dep in &self.analysis.dependencies {
                text.push_str(&format!("{}{}\n", dep.manager, format_dep_count(dep.count)));
                text.push_str(&format!("  File: {:?}\n", dep.file_path));
            }
            text.push('\n');
        }

        // Workflow
        text.push_str("WORKFLOW PROGRESS\n");
        text.push_str(&"-".repeat(80));
        text.push('\n');
        text.push_str(&format!(
            "Overall: {:.0}% complete\n\n",
            self.workflow.progress_percentage()
        ));
        for phase in crate::types::WorkflowPhase::all() {
            if let Some(info) = self.workflow.phases.get(&phase) {
                text.push_str(&format!(
                    "{:15} {:12}",
                    format!("{}", phase),
                    format!("{}", info.status)
                ));
                if let Some(started) = info.started_at {
                    text.push_str(&format!(
                        "  Started: {}",
                        format_timestamp(started, "%Y-%m-%d %H:%M")
                    ));
                }
                if let Some(completed) = info.completed_at {
                    text.push_str(&format!(
                        "  Completed: {}",
                        format_timestamp(completed, "%Y-%m-%d %H:%M")
                    ));
                }
                text.push('\n');
            }
        }
        text.push('\n');

        // Recommendations
        text.push_str("RECOMMENDATIONS\n");
        text.push_str(&"-".repeat(80));
        text.push('\n');
        if let Some(transpiler) = self.analysis.recommend_transpiler() {
            text.push_str(&format!("• Use {} for transpilation\n", transpiler));
        }
        if self.analysis.has_ml_dependencies() {
            text.push_str("• Consider Aprender for ML algorithms and Realizar for inference\n");
        }
        if needs_refactoring(self.analysis.tdg_score) {
            text.push_str("• TDG score below 85 - consider refactoring before migration\n");
        }
        text.push('\n');

        text
    }

    /// Save report to file
    pub fn save(&self, path: &Path, format: ReportFormat) -> Result<()> {
        let content = match format {
            ReportFormat::Html => self.to_html(),
            ReportFormat::Markdown => self.to_markdown(),
            ReportFormat::Json => self.to_json()?,
            ReportFormat::Text => self.to_text(),
        };

        std::fs::write(path, content)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ReportFormat {
    Html,
    Markdown,
    Json,
    Text,
}

fn get_tdg_grade(score: f64) -> &'static str {
    const GRADES: &[(f64, &str)] =
        &[(95.0, "A+"), (90.0, "A"), (85.0, "B+"), (80.0, "B"), (70.0, "C")];
    GRADES.iter().find(|(threshold, _)| score >= *threshold).map_or("D", |(_, grade)| grade)
}

fn format_tdg_score(score: f64) -> String {
    format!("{:.1}/100 ({})", score, get_tdg_grade(score))
}

fn format_dep_count(count: Option<usize>) -> String {
    count.map_or_else(String::new, |c| format!(" ({} packages)", c))
}

fn format_timestamp(dt: chrono::DateTime<chrono::Utc>, fmt: &str) -> String {
    dt.format(fmt).to_string()
}

fn format_timestamp_or_dash(dt: Option<chrono::DateTime<chrono::Utc>>, fmt: &str) -> String {
    dt.map_or_else(|| "-".to_string(), |t| t.format(fmt).to_string())
}

fn needs_refactoring(tdg_score: Option<f64>) -> bool {
    tdg_score.is_some_and(|s| s < 85.0)
}

#[cfg(test)]
#[path = "report_tests.rs"]
mod tests;
