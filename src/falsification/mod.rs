//! Popperian Falsification Checklist Implementation
//!
//! Implements the 108-item Sovereign AI Assurance Protocol checklist.
//! Each checklist item is a falsifiable claim with explicit rejection criteria.
//!
//! # Toyota Way Integration
//!
//! - **Jidoka**: Automated gates stop pipeline on failure
//! - **Genchi Genbutsu**: Evidence-based verification
//! - **Kaizen**: Continuous improvement via metrics
//!
//! # Severity Levels
//!
//! - **Critical**: Project FAIL - blocks release
//! - **Major**: Requires remediation before release
//! - **Minor**: Documented limitation
//! - **Info**: Clarification needed
//!
//! # Implemented Sections (108 Items Total)
//!
//! - Section 1: Sovereign Data Governance (SDG-01 to SDG-15) - 15 items
//! - Section 2: ML Technical Debt Prevention (MTD-01 to MTD-10) - 10 items
//! - Section 3: Hypothesis-Driven Development (HDD-01 to EDD-03) - 13 items
//! - Section 4: Numerical Reproducibility (NR-01 to NR-15) - 15 items
//! - Section 5: Performance & Waste Elimination (PW-01 to PW-15) - 15 items
//! - Section 6: Safety & Formal Verification (SF-01 to SF-10) - 9 items
//! - Section 7: Jidoka Automated Gates (JA-01 to JA-12) - 9 items
//! - Section 8: Model Cards & Auditability (MA-01 to MA-10) - 10 items
//! - Section 9: Cross-Platform & API (CP-01 to CP-05) - 5 items
//! - Section 10: Architectural Invariants (AI-01 to AI-05) - 5 items CRITICAL

mod auditors;
mod cross_platform;
pub(crate) mod helpers;
mod hypothesis_driven;
mod invariants;
mod jidoka;
mod model_cards;
mod numerical_reproducibility;
mod performance_waste;
mod safety;
mod sovereign_data;
mod technical_debt;
mod types;

pub use auditors::*;
pub use cross_platform::evaluate_all as evaluate_cross_platform;
pub use hypothesis_driven::evaluate_all as evaluate_hypothesis_driven;
pub use invariants::*;
pub use jidoka::evaluate_all as evaluate_jidoka;
pub use model_cards::evaluate_all as evaluate_model_cards;
pub use numerical_reproducibility::evaluate_all as evaluate_numerical_reproducibility;
pub use performance_waste::evaluate_all as evaluate_performance_waste;
pub use safety::evaluate_all as evaluate_safety;
pub use sovereign_data::evaluate_all as evaluate_sovereign_data;
pub use technical_debt::evaluate_all as evaluate_technical_debt;
pub use types::*;

use std::path::Path;

/// Run the complete falsification checklist against a project.
pub fn evaluate_project(project_path: &Path) -> ChecklistResult {
    let mut result = ChecklistResult::new(project_path);

    // Section 1: Sovereign Data Governance (15 items)
    let sovereign_results = sovereign_data::evaluate_all(project_path);
    result.add_section("Sovereign Data Governance", sovereign_results);

    // Section 2: ML Technical Debt Prevention (10 items)
    let debt_results = technical_debt::evaluate_all(project_path);
    result.add_section("ML Technical Debt Prevention", debt_results);

    // Section 3: Hypothesis-Driven Development (13 items)
    let hdd_results = hypothesis_driven::evaluate_all(project_path);
    result.add_section("Hypothesis-Driven Development", hdd_results);

    // Section 4: Numerical Reproducibility (15 items)
    let nr_results = numerical_reproducibility::evaluate_all(project_path);
    result.add_section("Numerical Reproducibility", nr_results);

    // Section 5: Performance & Waste Elimination (15 items)
    let pw_results = performance_waste::evaluate_all(project_path);
    result.add_section("Performance & Waste Elimination", pw_results);

    // Section 6: Safety & Formal Verification (9 items)
    let safety_results = safety::evaluate_all(project_path);
    result.add_section("Safety & Formal Verification", safety_results);

    // Section 7: Jidoka Automated Gates (9 items)
    let jidoka_results = jidoka::evaluate_all(project_path);
    result.add_section("Jidoka Automated Gates", jidoka_results);

    // Section 8: Model Cards & Auditability (10 items)
    let mc_results = model_cards::evaluate_all(project_path);
    result.add_section("Model Cards & Auditability", mc_results);

    // Section 9: Cross-Platform & API (5 items)
    let cp_results = cross_platform::evaluate_all(project_path);
    result.add_section("Cross-Platform & API", cp_results);

    // Section 10: Architectural Invariants (CRITICAL - 5 items)
    let invariant_results = invariants::evaluate_all(project_path);
    result.add_section("Architectural Invariants", invariant_results);

    // Calculate final score
    result.finalize();

    result
}

/// Run only the critical architectural invariants.
pub fn evaluate_critical_only(project_path: &Path) -> ChecklistResult {
    let mut result = ChecklistResult::new(project_path);

    let invariant_results = invariants::evaluate_all(project_path);
    result.add_section("Architectural Invariants", invariant_results);

    result.finalize();
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // =========================================================================
    // FALS-MOD-001: Evaluate Project
    // =========================================================================

    #[test]
    fn test_fals_001_evaluate_project_returns_result() {
        let path = PathBuf::from(".");
        let result = evaluate_project(&path);
        assert!(!result.sections.is_empty());
    }

    #[test]
    fn test_fals_002_critical_only_returns_invariants() {
        let path = PathBuf::from(".");
        let result = evaluate_critical_only(&path);
        assert!(result.sections.contains_key("Architectural Invariants"));
    }

    // =========================================================================
    // FALS-INT-001: Integration Tests - Full Checklist on batuta
    // =========================================================================

    #[test]
    fn test_fals_int_001_batuta_passes_critical_invariants() {
        // batuta itself should pass the critical invariants
        let path = PathBuf::from(".");
        let result = evaluate_critical_only(&path);

        // Verify no critical failures using an imperative loop
        // (avoids creating an LLVM closure region that would be uncovered
        // when no items pass the filter)
        let mut critical_count = 0;
        for items in result.sections.values() {
            for item in items {
                if item.is_critical_failure() {
                    critical_count += 1;
                }
            }
        }

        assert!(!result.has_critical_failure, "batuta has {critical_count} critical failure(s)",);
    }

    #[test]
    fn test_fals_int_002_batuta_achieves_kaizen_grade() {
        // batuta should achieve at least Kaizen Required grade
        let path = PathBuf::from(".");
        let result = evaluate_critical_only(&path);

        assert!(
            result.grade.passes(),
            "Expected Kaizen Required or better, got {} ({:.1}%)",
            result.grade,
            result.score
        );
    }

    #[test]
    fn test_fals_int_003_all_items_have_tps_principle() {
        let path = PathBuf::from(".");
        let result = evaluate_critical_only(&path);

        for (section, items) in &result.sections {
            for item in items {
                assert!(
                    !item.tps_principle.is_empty(),
                    "Item {}.{} missing TPS principle",
                    section,
                    item.id
                );
            }
        }
    }

    #[test]
    fn test_fals_int_004_result_serializes_to_json() {
        let path = PathBuf::from(".");
        let result = evaluate_critical_only(&path);

        let json = serde_json::to_string(&result);
        // Eagerly evaluate error for coverage (avoids lazy assert! format closure)
        let serialize_err = json.as_ref().err().map(|e| format!("{e:?}"));
        assert!(json.is_ok(), "Failed to serialize result: {serialize_err:?}");

        // Verify deserialize roundtrip
        let json_str = json.expect("unexpected failure");
        let parsed: Result<ChecklistResult, _> = serde_json::from_str(&json_str);
        let parse_err = parsed.as_ref().err().map(|e| format!("{e:?}"));
        assert!(parsed.is_ok(), "Failed to deserialize result: {parse_err:?}");
    }

    #[test]
    fn test_fals_int_005_result_summary_format() {
        let path = PathBuf::from(".");
        let result = evaluate_critical_only(&path);

        let summary = result.summary();
        // Summary should contain grade, score, and pass count
        assert!(summary.contains('%'), "Summary missing percentage: {}", summary);
        assert!(
            summary.contains("passed") || summary.contains("RELEASE"),
            "Summary missing status: {}",
            summary
        );
    }

    // =========================================================================
    // FALS-INT-010: Edge Cases
    // =========================================================================

    #[test]
    fn test_fals_int_010_nonexistent_project() {
        let path = PathBuf::from("/nonexistent/project/path");
        let result = evaluate_critical_only(&path);

        // Should not panic, should return result with failures
        assert!(result.total_items > 0);
    }

    #[test]
    fn test_fals_int_011_empty_directory() {
        // Create a temp dir for testing
        let temp_dir = std::env::temp_dir().join("batuta_test_empty");
        let _ = std::fs::create_dir_all(&temp_dir);

        let result = evaluate_critical_only(&temp_dir);

        // Should not panic, should return result
        assert!(result.total_items > 0);

        let _ = std::fs::remove_dir(&temp_dir);
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_fals_mod_evaluate_project_all_sections() {
        let path = PathBuf::from(".");
        let result = evaluate_project(&path);

        // Should have all 10 sections
        assert!(result.sections.len() >= 10);
        assert!(result.sections.contains_key("Sovereign Data Governance"));
        assert!(result.sections.contains_key("ML Technical Debt Prevention"));
        assert!(result.sections.contains_key("Hypothesis-Driven Development"));
        assert!(result.sections.contains_key("Numerical Reproducibility"));
        assert!(result.sections.contains_key("Performance & Waste Elimination"));
        assert!(result.sections.contains_key("Safety & Formal Verification"));
        assert!(result.sections.contains_key("Jidoka Automated Gates"));
        assert!(result.sections.contains_key("Model Cards & Auditability"));
        assert!(result.sections.contains_key("Cross-Platform & API"));
        assert!(result.sections.contains_key("Architectural Invariants"));
    }

    #[test]
    fn test_fals_mod_result_finalize_counts() {
        let path = PathBuf::from(".");
        let result = evaluate_project(&path);

        // Total items should be sum of all section items
        let expected_total: usize = result.sections.values().map(|v| v.len()).sum();
        assert_eq!(result.total_items, expected_total);

        // Passed + failed + other <= total
        assert!(result.passed_items + result.failed_items <= result.total_items);
    }

    #[test]
    fn test_fals_mod_result_score_range() {
        let path = PathBuf::from(".");
        let result = evaluate_project(&path);

        // Score should be between 0 and 100
        assert!(result.score >= 0.0);
        assert!(result.score <= 100.0);
    }

    #[test]
    fn test_fals_mod_result_passes_method() {
        let path = PathBuf::from(".");
        let result = evaluate_critical_only(&path);

        // passes() should be consistent with grade.passes()
        assert_eq!(result.passes(), result.grade.passes() && !result.has_critical_failure);
    }

    #[test]
    fn test_fals_mod_evaluate_project_nonexistent() {
        let path = PathBuf::from("/nonexistent/project");
        let result = evaluate_project(&path);
        // Should not panic, should have all 10 sections
        assert!(result.sections.len() >= 10);
        assert!(result.total_items > 0);
    }

    #[test]
    fn test_fals_mod_evaluate_project_temp_dir() {
        let temp_dir = std::env::temp_dir().join("batuta_fals_mod_temp");
        let _ = std::fs::create_dir_all(&temp_dir);
        let result = evaluate_project(&temp_dir);
        assert!(result.sections.len() >= 10);
        // Score may vary but should be in valid range
        assert!(result.score >= 0.0 && result.score <= 100.0);
        let _ = std::fs::remove_dir(&temp_dir);
    }

    #[test]
    fn test_fals_mod_critical_only_nonexistent() {
        let path = PathBuf::from("/nonexistent/path/xyz");
        let result = evaluate_critical_only(&path);
        assert!(result.sections.contains_key("Architectural Invariants"));
        // Score may be low but should not panic
        assert!(result.score >= 0.0);
    }

    #[test]
    fn test_fals_mod_critical_failure_format_chain() {
        // Exercise the .map closure in the critical failure iterator chain
        // by constructing a result that has critical failures, so the
        // filter passes items through to the map closure.
        let mut result = ChecklistResult::new(std::path::Path::new("/test"));
        result.add_section(
            "Test",
            vec![
                CheckItem::new("CF-01", "Critical Test", "Critical claim")
                    .with_severity(Severity::Critical)
                    .with_tps("Jidoka")
                    .fail("Critical failure reason"),
                CheckItem::new("OK-01", "Pass Test", "Pass claim").with_tps("Kaizen").pass(),
            ],
        );
        result.finalize();

        // Exercise the same iterator chain used in test_fals_int_001
        let critical_failures: Vec<_> = result
            .sections
            .values()
            .flat_map(|items| items.iter())
            .filter(|i| i.is_critical_failure())
            .map(|i| format!("{}: {}", i.id, i.rejection_reason.as_deref().unwrap_or("")))
            .collect();

        assert_eq!(critical_failures.len(), 1);
        assert!(critical_failures[0].contains("CF-01"));
        assert!(critical_failures[0].contains("Critical failure reason"));
    }
}
