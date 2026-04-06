use super::*;
use crate::types::{DependencyInfo, DependencyManager, Language, LanguageStats};
use std::path::PathBuf;
use tempfile::TempDir;

fn create_test_analysis() -> ProjectAnalysis {
    let mut analysis = ProjectAnalysis::new(PathBuf::from("/test/project"));
    analysis.total_files = 50;
    analysis.total_lines = 5000;
    analysis.primary_language = Some(Language::Python);
    analysis.tdg_score = Some(87.5);

    analysis.languages.push(LanguageStats {
        language: Language::Python,
        file_count: 40,
        line_count: 4000,
        percentage: 80.0,
    });
    analysis.languages.push(LanguageStats {
        language: Language::Rust,
        file_count: 10,
        line_count: 1000,
        percentage: 20.0,
    });

    analysis.dependencies.push(DependencyInfo {
        manager: DependencyManager::Pip,
        file_path: PathBuf::from("requirements.txt"),
        count: Some(25),
    });

    analysis
}

fn create_test_workflow() -> WorkflowState {
    let mut workflow = WorkflowState::new();
    workflow.start_phase(crate::types::WorkflowPhase::Analysis);
    workflow.complete_phase(crate::types::WorkflowPhase::Analysis);
    workflow.start_phase(crate::types::WorkflowPhase::Transpilation);
    workflow
}

// ============================================================================
// MIGRATION REPORT TESTS
// ============================================================================

#[test]
fn test_migration_report_new() {
    let analysis = create_test_analysis();
    let workflow = create_test_workflow();

    let report =
        MigrationReport::new("TestProject".to_string(), analysis.clone(), workflow.clone());

    assert_eq!(report.project_name, "TestProject");
    assert_eq!(report.analysis.total_files, 50);
    assert_eq!(report.workflow.progress_percentage(), 20.0);
}

#[test]
fn test_migration_report_clone() {
    let analysis = create_test_analysis();
    let workflow = create_test_workflow();
    let report1 = MigrationReport::new("Test".to_string(), analysis, workflow);

    let report2 = report1.clone();
    assert_eq!(report1.project_name, report2.project_name);
    assert_eq!(report1.analysis.total_files, report2.analysis.total_files);
}

// ============================================================================
// HTML REPORT TESTS
// ============================================================================

#[test]
fn test_to_html_contains_header() {
    let analysis = create_test_analysis();
    let workflow = create_test_workflow();
    let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

    let html = report.to_html();

    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("<html lang=\"en\">"));
    assert!(html.contains("TestProject"));
}

#[test]
fn test_to_html_contains_summary() {
    let analysis = create_test_analysis();
    let workflow = create_test_workflow();
    let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

    let html = report.to_html();

    assert!(html.contains("Summary"));
    assert!(html.contains("50")); // total_files
    assert!(html.contains("5000")); // total_lines
    assert!(html.contains("Python"));
}

#[test]
fn test_to_html_contains_tdg_score() {
    let analysis = create_test_analysis();
    let workflow = create_test_workflow();
    let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

    let html = report.to_html();

    assert!(html.contains("TDG Score"));
    assert!(html.contains("87.5"));
    assert!(html.contains("B+"));
}

#[test]
fn test_to_html_contains_languages_table() {
    let analysis = create_test_analysis();
    let workflow = create_test_workflow();
    let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

    let html = report.to_html();

    assert!(html.contains("<table>"));
    assert!(html.contains("Languages"));
    assert!(html.contains("Python"));
    assert!(html.contains("Rust"));
    assert!(html.contains("80.0%"));
}

#[test]
fn test_to_html_contains_dependencies() {
    let analysis = create_test_analysis();
    let workflow = create_test_workflow();
    let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

    let html = report.to_html();

    assert!(html.contains("Dependencies"));
    assert!(html.contains("pip"));
    assert!(html.contains("25 packages"));
}

#[test]
fn test_to_html_contains_workflow() {
    let analysis = create_test_analysis();
    let workflow = create_test_workflow();
    let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

    let html = report.to_html();

    assert!(html.contains("Workflow Progress"));
    assert!(html.contains("20%")); // progress_percentage
    assert!(html.contains("Analysis"));
    assert!(html.contains("Transpilation"));
}

#[test]
fn test_to_html_contains_recommendations() {
    let analysis = create_test_analysis();
    let workflow = create_test_workflow();
    let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

    let html = report.to_html();

    assert!(html.contains("Recommendations"));
    assert!(html.contains("Depyler"));
    assert!(html.contains("Aprender"));
}

#[test]
fn test_to_html_contains_footer() {
    let analysis = create_test_analysis();
    let workflow = create_test_workflow();
    let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

    let html = report.to_html();

    assert!(html.contains("</html>"));
    assert!(html.contains("Batuta"));
    assert!(html.contains("github.com/paiml/Batuta"));
}

// ============================================================================
// MARKDOWN REPORT TESTS
// ============================================================================

#[test]
fn test_to_markdown_contains_header() {
    let analysis = create_test_analysis();
    let workflow = create_test_workflow();
    let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

    let md = report.to_markdown();

    assert!(md.contains("# Migration Report: TestProject"));
    assert!(md.contains("**Generated:**"));
}

#[test]
fn test_to_markdown_contains_summary() {
    let analysis = create_test_analysis();
    let workflow = create_test_workflow();
    let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

    let md = report.to_markdown();

    assert!(md.contains("## Summary"));
    assert!(md.contains("**Total Files:** 50"));
    assert!(md.contains("**Total Lines:** 5000"));
    assert!(md.contains("**Primary Language:** Python"));
}

#[test]
fn test_to_markdown_contains_languages_table() {
    let analysis = create_test_analysis();
    let workflow = create_test_workflow();
    let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

    let md = report.to_markdown();

    assert!(md.contains("## Languages"));
    assert!(md.contains("| Language | Files | Lines | Percentage |"));
    assert!(md.contains("| Python | 40 | 4000 | 80.0% |"));
}

#[test]
fn test_to_markdown_contains_dependencies() {
    let analysis = create_test_analysis();
    let workflow = create_test_workflow();
    let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

    let md = report.to_markdown();

    assert!(md.contains("## Dependencies"));
    assert!(md.contains("**pip (requirements.txt)** (25 packages)"));
}

#[test]
fn test_to_markdown_contains_workflow() {
    let analysis = create_test_analysis();
    let workflow = create_test_workflow();
    let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

    let md = report.to_markdown();

    assert!(md.contains("## Workflow Progress"));
    assert!(md.contains("**Overall:** 20% complete"));
    assert!(md.contains("| Phase | Status | Started | Completed |"));
}

#[test]
fn test_to_markdown_contains_footer() {
    let analysis = create_test_analysis();
    let workflow = create_test_workflow();
    let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

    let md = report.to_markdown();

    assert!(md.contains("---"));
    assert!(md.contains("*Generated by Batuta - Sovereign AI Stack*"));
    assert!(md.contains("https://github.com/paiml/Batuta"));
}

// ============================================================================
// JSON REPORT TESTS
// ============================================================================

#[test]
fn test_to_json_valid() {
    let analysis = create_test_analysis();
    let workflow = create_test_workflow();
    let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

    let json = report.to_json().unwrap();

    assert!(json.contains("\"project_name\""));
    assert!(json.contains("\"TestProject\""));
    assert!(json.contains("\"analysis\""));
    assert!(json.contains("\"workflow\""));
}

#[test]
fn test_to_json_deserialize() {
    let analysis = create_test_analysis();
    let workflow = create_test_workflow();
    let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

    let json = report.to_json().unwrap();
    let deserialized: MigrationReport = serde_json::from_str(&json).unwrap();

    assert_eq!(report.project_name, deserialized.project_name);
    assert_eq!(report.analysis.total_files, deserialized.analysis.total_files);
}

// ============================================================================
// TEXT REPORT TESTS
// ============================================================================

#[test]
fn test_to_text_contains_header() {
    let analysis = create_test_analysis();
    let workflow = create_test_workflow();
    let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

    let text = report.to_text();

    assert!(text.contains("MIGRATION REPORT: TestProject"));
    assert!(text.contains("Generated:"));
    assert!(text.contains("========"));
}

#[test]
fn test_to_text_contains_summary() {
    let analysis = create_test_analysis();
    let workflow = create_test_workflow();
    let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

    let text = report.to_text();

    assert!(text.contains("SUMMARY"));
    assert!(text.contains("Total Files: 50"));
    assert!(text.contains("Total Lines: 5000"));
    assert!(text.contains("Primary Language: Python"));
}

#[test]
fn test_to_text_contains_languages() {
    let analysis = create_test_analysis();
    let workflow = create_test_workflow();
    let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

    let text = report.to_text();

    assert!(text.contains("LANGUAGES"));
    assert!(text.contains("Python"));
    assert!(text.contains("Rust"));
    assert!(text.contains("80.0%"));
}

#[test]
fn test_to_text_contains_workflow() {
    let analysis = create_test_analysis();
    let workflow = create_test_workflow();
    let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

    let text = report.to_text();

    assert!(text.contains("WORKFLOW PROGRESS"));
    assert!(text.contains("Overall: 20% complete"));
    assert!(text.contains("Analysis"));
}

// ============================================================================
// FILE SAVE TESTS
// ============================================================================

#[test]
fn test_save_html() {
    let temp_dir = TempDir::new().unwrap();
    let report_path = temp_dir.path().join("report.html");

    let analysis = create_test_analysis();
    let workflow = create_test_workflow();
    let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

    report.save(&report_path, ReportFormat::Html).unwrap();

    assert!(report_path.exists());
    let content = std::fs::read_to_string(&report_path).unwrap();
    assert!(content.contains("<!DOCTYPE html>"));
}

#[test]
fn test_save_markdown() {
    let temp_dir = TempDir::new().unwrap();
    let report_path = temp_dir.path().join("report.md");

    let analysis = create_test_analysis();
    let workflow = create_test_workflow();
    let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

    report.save(&report_path, ReportFormat::Markdown).unwrap();

    assert!(report_path.exists());
    let content = std::fs::read_to_string(&report_path).unwrap();
    assert!(content.contains("# Migration Report"));
}

#[test]
fn test_save_json() {
    let temp_dir = TempDir::new().unwrap();
    let report_path = temp_dir.path().join("report.json");

    let analysis = create_test_analysis();
    let workflow = create_test_workflow();
    let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

    report.save(&report_path, ReportFormat::Json).unwrap();

    assert!(report_path.exists());
    let content = std::fs::read_to_string(&report_path).unwrap();
    assert!(content.contains("\"project_name\""));
}

#[test]
fn test_save_text() {
    let temp_dir = TempDir::new().unwrap();
    let report_path = temp_dir.path().join("report.txt");

    let analysis = create_test_analysis();
    let workflow = create_test_workflow();
    let report = MigrationReport::new("TestProject".to_string(), analysis, workflow);

    report.save(&report_path, ReportFormat::Text).unwrap();

    assert!(report_path.exists());
    let content = std::fs::read_to_string(&report_path).unwrap();
    assert!(content.contains("MIGRATION REPORT"));
}

// ============================================================================
// HELPER FUNCTION TESTS
// ============================================================================

#[test]
fn test_get_tdg_grade_a_plus() {
    assert_eq!(get_tdg_grade(100.0), "A+");
    assert_eq!(get_tdg_grade(95.0), "A+");
}

#[test]
fn test_get_tdg_grade_a() {
    assert_eq!(get_tdg_grade(94.9), "A");
    assert_eq!(get_tdg_grade(90.0), "A");
}

#[test]
fn test_get_tdg_grade_b_plus() {
    assert_eq!(get_tdg_grade(89.9), "B+");
    assert_eq!(get_tdg_grade(85.0), "B+");
}

#[test]
fn test_get_tdg_grade_b() {
    assert_eq!(get_tdg_grade(84.9), "B");
    assert_eq!(get_tdg_grade(80.0), "B");
}

#[test]
fn test_get_tdg_grade_c() {
    assert_eq!(get_tdg_grade(79.9), "C");
    assert_eq!(get_tdg_grade(70.0), "C");
}

#[test]
fn test_get_tdg_grade_d() {
    assert_eq!(get_tdg_grade(69.9), "D");
    assert_eq!(get_tdg_grade(50.0), "D");
    assert_eq!(get_tdg_grade(0.0), "D");
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

#[test]
fn test_report_with_no_languages() {
    let mut analysis = ProjectAnalysis::new(PathBuf::from("/test"));
    analysis.total_files = 0;
    analysis.total_lines = 0;
    let workflow = WorkflowState::new();

    let report = MigrationReport::new("Empty".to_string(), analysis, workflow);

    let html = report.to_html();
    assert!(html.contains("Empty"));

    let md = report.to_markdown();
    assert!(md.contains("Empty"));

    let text = report.to_text();
    assert!(text.contains("Empty"));
}

#[test]
fn test_report_with_no_dependencies() {
    let mut analysis = ProjectAnalysis::new(PathBuf::from("/test"));
    analysis.total_files = 10;
    analysis.total_lines = 100;
    let workflow = WorkflowState::new();

    let report = MigrationReport::new("NoDeps".to_string(), analysis, workflow);

    let html = report.to_html();
    let md = report.to_markdown();
    let text = report.to_text();

    // Should not crash and should contain project name
    assert!(html.contains("NoDeps"));
    assert!(md.contains("NoDeps"));
    assert!(text.contains("NoDeps"));
}

#[test]
fn test_report_with_no_tdg_score() {
    let mut analysis = ProjectAnalysis::new(PathBuf::from("/test"));
    analysis.total_files = 10;
    analysis.tdg_score = None;
    let workflow = WorkflowState::new();

    let report = MigrationReport::new("NoTDG".to_string(), analysis, workflow);

    let html = report.to_html();
    // Should not contain TDG score section
    assert!(!html.contains("TDG Score"));
}

#[test]
fn test_report_format_clone_copy() {
    let format1 = ReportFormat::Html;
    let format2 = format1; // Copy

    // Both should be usable
    let _ = format!("{:?}", format1);
    let _ = format!("{:?}", format2);
}

#[test]
fn test_report_with_high_tdg_score() {
    let mut analysis = create_test_analysis();
    analysis.tdg_score = Some(96.0);
    let workflow = create_test_workflow();

    let report = MigrationReport::new("HighTDG".to_string(), analysis, workflow);

    let html = report.to_html();
    assert!(html.contains("96.0"));
    assert!(html.contains("A+"));

    // Recommendations should not suggest refactoring for high scores
    assert!(!html.contains("consider refactoring before migration"));
}

#[test]
fn test_report_with_low_tdg_score() {
    let mut analysis = create_test_analysis();
    analysis.tdg_score = Some(70.0);
    let workflow = create_test_workflow();

    let report = MigrationReport::new("LowTDG".to_string(), analysis, workflow);

    let html = report.to_html();
    assert!(html.contains("70.0"));
    assert!(html.contains('C'));

    // Recommendations should suggest refactoring for low scores
    assert!(html.contains("consider refactoring before migration"));
}

#[test]
fn test_report_with_ml_dependencies() {
    let analysis = create_test_analysis();
    let workflow = create_test_workflow();

    let report = MigrationReport::new("MLProject".to_string(), analysis, workflow);

    let html = report.to_html();
    // Should recommend ML tools for projects with Pip dependencies
    assert!(html.contains("Aprender"));
    assert!(html.contains("Realizar"));
}
