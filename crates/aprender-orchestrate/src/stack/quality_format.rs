//! Quality Report Formatting
//!
//! Text and JSON formatting for stack quality reports.

use anyhow::{anyhow, Result};
use std::collections::HashMap;

use super::quality::{ComponentQuality, IssueSeverity, StackLayer, StackQualityReport};

/// Layer display order for quality reports.
const LAYER_ORDER: [StackLayer; 8] = [
    StackLayer::Compute,
    StackLayer::Ml,
    StackLayer::Training,
    StackLayer::Transpilers,
    StackLayer::Orchestration,
    StackLayer::Quality,
    StackLayer::DataMlops,
    StackLayer::Presentation,
];

/// Format a single layer's components as text rows.
fn format_layer_components(output: &mut String, components: &[&ComponentQuality]) {
    output.push_str(&format!(
        "  {:20} {:8} {:8} {:8} {:6} {:7} {:6}\n",
        "Component", "Rust", "Repo", "README", "Hero", "SQI", "Grade"
    ));
    output.push_str(&format!(
        "  {:20} {:8} {:8} {:8} {:6} {:7} {:6}\n",
        "─".repeat(20),
        "─".repeat(8),
        "─".repeat(8),
        "─".repeat(8),
        "─".repeat(6),
        "─".repeat(7),
        "─".repeat(6)
    ));

    for comp in components {
        let hero_status = if comp.hero_image.valid { "✓" } else { "✗" };
        output.push_str(&format!(
            "  {:20} {:>3}/{:<4} {:>3}/{:<4} {:>2}/{:<4} {:^6} {:>6.1} {} {}\n",
            comp.name,
            comp.rust_score.value,
            comp.rust_score.max,
            comp.repo_score.value,
            comp.repo_score.max,
            comp.readme_score.value,
            comp.readme_score.max,
            hero_status,
            comp.sqi,
            comp.grade.symbol(),
            comp.grade.icon(),
        ));

        for issue in &comp.issues {
            let icon = match issue.severity {
                IssueSeverity::Error => "└── ❌",
                IssueSeverity::Warning => "└── ⚠️",
                IssueSeverity::Info => "└── ℹ️",
            };
            output.push_str(&format!("    {} {}\n", icon, issue.message));
        }
    }
}

/// Format quality report as text
pub fn format_report_text(report: &StackQualityReport) -> String {
    let mut output = String::new();

    output.push_str("PAIML Stack Quality Matrix\n");
    output.push_str(&"═".repeat(78));
    output.push_str("\n\n");

    // Group by layer
    let mut by_layer: HashMap<StackLayer, Vec<&ComponentQuality>> = HashMap::new();
    for comp in &report.components {
        by_layer.entry(comp.layer).or_default().push(comp);
    }

    for layer in LAYER_ORDER {
        if let Some(components) = by_layer.get(&layer) {
            output.push_str(&format!("{}\n", layer.display_name()));
            output.push_str(&"─".repeat(78));
            output.push('\n');
            format_layer_components(&mut output, components);
            output.push('\n');
        }
    }

    // Summary
    output.push_str(&"═".repeat(78));
    output.push_str("\nSUMMARY\n");
    output.push_str(&"═".repeat(78));
    output.push_str("\n\n");

    output.push_str(&format!(
        "Quality Distribution:\n  A+  {:3} components ({:.0}%)\n  A   {:3} components ({:.0}%)\n  A-  {:3} components ({:.0}%)\n  <A- {:3} components ({:.0}%)\n\n",
        report.summary.a_plus_count,
        (report.summary.a_plus_count as f64 / report.summary.total_components as f64) * 100.0,
        report.summary.a_count,
        (report.summary.a_count as f64 / report.summary.total_components as f64) * 100.0,
        report.summary.a_minus_count,
        (report.summary.a_minus_count as f64 / report.summary.total_components as f64) * 100.0,
        report.summary.below_threshold_count,
        (report.summary.below_threshold_count as f64 / report.summary.total_components as f64) * 100.0,
    ));

    output.push_str(&format!(
        "Stack Quality Index: {:.1} ({})\n\n",
        report.stack_quality_index, report.overall_grade
    ));

    if report.release_ready {
        output.push_str("Release Status: ✅ READY\n");
    } else {
        output.push_str("Release Status: ❌ BLOCKED\n");
        output
            .push_str(&format!("  Blocked components: {}\n", report.blocked_components.join(", ")));
    }

    if !report.recommendations.is_empty() {
        output.push_str("\nRecommended Actions:\n");
        for (i, rec) in report.recommendations.iter().enumerate() {
            output.push_str(&format!("  {}. {}\n", i + 1, rec));
        }
    }

    output
}

/// Format quality report as JSON
pub fn format_report_json(report: &StackQualityReport) -> Result<String> {
    serde_json::to_string_pretty(report).map_err(|e| anyhow!("JSON serialization error: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stack::hero_image::{HeroImageResult, ImageFormat};
    use crate::stack::quality::{QualityGrade, Score};
    use std::path::PathBuf;

    fn create_test_component(
        name: &str,
        rust: u32,
        repo: u32,
        readme: u32,
        has_hero: bool,
    ) -> ComponentQuality {
        let rust_score = Score::new(rust, 114, QualityGrade::from_rust_project_score(rust));
        let repo_score = Score::new(repo, 110, QualityGrade::from_repo_score(repo));
        let readme_score = Score::new(readme, 20, QualityGrade::from_readme_score(readme));
        let hero = if has_hero {
            HeroImageResult::found(PathBuf::from("hero.png"), ImageFormat::Png)
        } else {
            HeroImageResult::missing()
        };

        ComponentQuality::new(
            name,
            PathBuf::from("/test"),
            rust_score,
            repo_score,
            readme_score,
            hero,
        )
    }

    #[test]
    fn test_format_report_text() {
        let components = vec![create_test_component("trueno", 107, 98, 20, true)];
        let report = StackQualityReport::from_components(components);
        let text = format_report_text(&report);

        assert!(text.contains("trueno"));
        assert!(text.contains("PAIML"));
    }

    #[test]
    fn test_format_report_json() {
        let components = vec![create_test_component("trueno", 107, 98, 20, true)];
        let report = StackQualityReport::from_components(components);
        let json = format_report_json(&report).expect("unexpected failure");

        assert!(json.contains("trueno"));
        assert!(json.contains("stack_quality_index"));
    }

    #[test]
    fn test_format_report_text_with_layers() {
        let components = vec![
            create_test_component("trueno", 107, 98, 20, true), // Compute
            create_test_component("aprender", 95, 90, 16, true), // ML
            create_test_component("entrenar", 100, 92, 18, true), // Training
            create_test_component("depyler", 90, 88, 15, false), // Transpilers
        ];
        let report = StackQualityReport::from_components(components);
        let text = format_report_text(&report);

        // Verify layer headers
        assert!(text.contains("COMPUTE PRIMITIVES"));
        assert!(text.contains("ML ALGORITHMS"));
        assert!(text.contains("TRAINING & INFERENCE"));
        assert!(text.contains("TRANSPILERS"));
        assert!(text.contains("SUMMARY"));
    }

    #[test]
    fn test_format_report_text_with_issues() {
        use crate::stack::quality::QualityIssue;

        let mut comp = create_test_component("test", 70, 60, 10, false);
        comp.issues.push(QualityIssue::new(
            "low_score",
            "Score below threshold",
            IssueSeverity::Error,
        ));
        comp.issues.push(QualityIssue::new(
            "warning",
            "Missing documentation",
            IssueSeverity::Warning,
        ));
        comp.issues.push(QualityIssue::new(
            "info",
            "Consider adding examples",
            IssueSeverity::Info,
        ));

        let report = StackQualityReport::from_components(vec![comp]);
        let text = format_report_text(&report);

        assert!(text.contains("❌"));
        assert!(text.contains("⚠️"));
        assert!(text.contains("ℹ️"));
    }

    #[test]
    fn test_format_report_text_blocked_components() {
        let mut comp = create_test_component("blocked", 70, 60, 10, false);
        comp.release_ready = false;
        comp.grade = QualityGrade::B;

        let report = StackQualityReport::from_components(vec![comp]);
        let text = format_report_text(&report);

        assert!(text.contains("BLOCKED"));
        assert!(text.contains("blocked"));
    }
}
