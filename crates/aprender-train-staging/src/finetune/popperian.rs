//! Popperian Falsification QA System
//!
//! Implements a 100-point scientific validation checklist based on
//! Karl Popper's philosophy of falsifiability.
//!
//! # References
//!
//! - Popper, K. (1959) "The Logic of Scientific Discovery"
//! - Popper, K. (1963) "Conjectures and Refutations"

use std::fmt;

/// Quality grade based on Popperian score
/// Ordering: F < C < B < BPlus < A < APlus (worst to best)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QAGrade {
    /// <70: Failing
    F,
    /// 70-79: Needs Improvement
    C,
    /// 80-84: Satisfactory
    B,
    /// 85-89: Good
    BPlus,
    /// 90-94: Very Good
    A,
    /// 95-100: Excellent
    APlus,
}

impl QAGrade {
    /// Create grade from score
    #[must_use]
    pub const fn from_score(score: u8) -> Self {
        match score {
            95..=100 => Self::APlus,
            90..=94 => Self::A,
            85..=89 => Self::BPlus,
            80..=84 => Self::B,
            70..=79 => Self::C,
            0..=69 => Self::F,
            101.. => Self::F,
        }
    }

    /// Check if grade is passing (C or better)
    #[must_use]
    pub const fn is_passing(&self) -> bool {
        !matches!(self, Self::F)
    }
}

impl fmt::Display for QAGrade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::APlus => write!(f, "A+ (Excellent)"),
            Self::A => write!(f, "A  (Very Good)"),
            Self::BPlus => write!(f, "B+ (Good)"),
            Self::B => write!(f, "B  (Satisfactory)"),
            Self::C => write!(f, "C  (Needs Improvement)"),
            Self::F => write!(f, "F  (Failing)"),
        }
    }
}

/// 100-Point Popperian Falsification Checklist
///
/// Each field represents a falsifiable hypothesis. If the test passes,
/// the hypothesis is corroborated (not proven). If it fails, the hypothesis
/// is falsified.
#[derive(Debug, Clone, Default)]
pub struct PopperianQA {
    // === REPRODUCIBILITY (20 points) ===
    // H1: Training is deterministic with fixed seed
    /// Same seed produces identical loss curves (±1e-6)
    pub r1_same_loss_curve: bool,
    /// Adapter weights match exactly across runs
    pub r2_same_final_weights: bool,
    /// Evaluation metrics are identical
    pub r3_same_eval_metrics: bool,
    /// All dependencies are version-locked
    pub r4_environment_locked: bool,

    // === COMPILATION (20 points) ===
    // H2: Generated tests are syntactically valid Rust
    /// rustfmt succeeds on generated code
    pub c1_parses_as_rust: bool,
    /// cargo check succeeds
    pub c2_type_checks: bool,
    /// No clippy warnings
    pub c3_no_unused_warnings: bool,
    /// Links correctly against target crate
    pub c4_links_correctly: bool,

    // === CORRECTNESS (20 points) ===
    // H3: Generated tests are semantically meaningful
    /// Tests pass on original implementation
    pub x1_tests_pass_on_correct: bool,
    /// Tests fail on mutated implementation
    pub x2_tests_fail_on_mutant: bool,
    /// Assertions are not trivial (not just `assert!(true)`)
    pub x3_assertions_meaningful: bool,
    /// No tautologies (not `assert_eq!(x, x)`)
    pub x4_no_tautologies: bool,

    // === COVERAGE (15 points) ===
    // H4: Tests exercise meaningful code paths
    /// Branch coverage delta ≥+5%
    pub v1_branch_coverage_delta: bool,
    /// Line coverage delta ≥+10%
    pub v2_line_coverage_delta: bool,
    /// Tests include edge cases (empty, null, max)
    pub v3_edge_cases_present: bool,

    // === EFFICIENCY (10 points) ===
    // H5: Training completes within resource bounds
    /// Peak VRAM < 8GB
    pub e1_vram_under_8gb: bool,
    /// Training completes in <4 hours
    pub e2_training_under_4hrs: bool,
    /// Inference < 1s per function
    pub e3_inference_under_1s: bool,

    // === EDGE CASES (10 points) ===
    // H6: Handles difficult inputs gracefully
    /// Handles generic functions
    pub g1_handles_generics: bool,
    /// Handles lifetime annotations
    pub g2_handles_lifetimes: bool,
    /// Handles async functions
    pub g3_handles_async: bool,
    /// Handles unsafe blocks
    pub g4_handles_unsafe: bool,
    /// Handles macro-heavy code
    pub g5_handles_macros: bool,

    // === DOCUMENTATION (5 points) ===
    // H7: Output is self-explanatory
    /// Test names describe intent (test_*)
    pub d1_test_names_descriptive: bool,
    /// Comments explain edge cases
    pub d2_comments_present: bool,
    /// Proptest strategies have clear names
    pub d3_proptest_strategies_clear: bool,
}

impl PopperianQA {
    /// Create new QA checklist with all items unchecked
    #[must_use]
    pub const fn new() -> Self {
        Self {
            r1_same_loss_curve: false,
            r2_same_final_weights: false,
            r3_same_eval_metrics: false,
            r4_environment_locked: false,
            c1_parses_as_rust: false,
            c2_type_checks: false,
            c3_no_unused_warnings: false,
            c4_links_correctly: false,
            x1_tests_pass_on_correct: false,
            x2_tests_fail_on_mutant: false,
            x3_assertions_meaningful: false,
            x4_no_tautologies: false,
            v1_branch_coverage_delta: false,
            v2_line_coverage_delta: false,
            v3_edge_cases_present: false,
            e1_vram_under_8gb: false,
            e2_training_under_4hrs: false,
            e3_inference_under_1s: false,
            g1_handles_generics: false,
            g2_handles_lifetimes: false,
            g3_handles_async: false,
            g4_handles_unsafe: false,
            g5_handles_macros: false,
            d1_test_names_descriptive: false,
            d2_comments_present: false,
            d3_proptest_strategies_clear: false,
        }
    }

    /// Calculate total score (0-100)
    #[must_use]
    pub fn score(&self) -> u8 {
        let weighted: &[(bool, u8)] = &[
            // Reproducibility (20 points)
            (self.r1_same_loss_curve, 5),
            (self.r2_same_final_weights, 5),
            (self.r3_same_eval_metrics, 5),
            (self.r4_environment_locked, 5),
            // Compilation (20 points)
            (self.c1_parses_as_rust, 5),
            (self.c2_type_checks, 5),
            (self.c3_no_unused_warnings, 5),
            (self.c4_links_correctly, 5),
            // Correctness (20 points)
            (self.x1_tests_pass_on_correct, 5),
            (self.x2_tests_fail_on_mutant, 5),
            (self.x3_assertions_meaningful, 5),
            (self.x4_no_tautologies, 5),
            // Coverage (15 points)
            (self.v1_branch_coverage_delta, 5),
            (self.v2_line_coverage_delta, 5),
            (self.v3_edge_cases_present, 5),
            // Efficiency (10 points)
            (self.e1_vram_under_8gb, 3),
            (self.e2_training_under_4hrs, 4),
            (self.e3_inference_under_1s, 3),
            // Edge Cases (10 points)
            (self.g1_handles_generics, 2),
            (self.g2_handles_lifetimes, 2),
            (self.g3_handles_async, 2),
            (self.g4_handles_unsafe, 2),
            (self.g5_handles_macros, 2),
            // Documentation (5 points)
            (self.d1_test_names_descriptive, 2),
            (self.d2_comments_present, 2),
            (self.d3_proptest_strategies_clear, 1),
        ];
        weighted.iter().filter(|(passed, _)| *passed).map(|(_, pts)| pts).sum()
    }

    /// Get quality grade
    #[must_use]
    pub fn grade(&self) -> QAGrade {
        QAGrade::from_score(self.score())
    }

    /// Check if all reproducibility criteria pass
    #[must_use]
    pub const fn reproducibility_passed(&self) -> bool {
        self.r1_same_loss_curve
            && self.r2_same_final_weights
            && self.r3_same_eval_metrics
            && self.r4_environment_locked
    }

    /// Check if all compilation criteria pass
    #[must_use]
    pub const fn compilation_passed(&self) -> bool {
        self.c1_parses_as_rust
            && self.c2_type_checks
            && self.c3_no_unused_warnings
            && self.c4_links_correctly
    }

    /// Check if all correctness criteria pass
    #[must_use]
    pub const fn correctness_passed(&self) -> bool {
        self.x1_tests_pass_on_correct
            && self.x2_tests_fail_on_mutant
            && self.x3_assertions_meaningful
            && self.x4_no_tautologies
    }

    /// Count passed items
    #[must_use]
    pub fn passed_count(&self) -> usize {
        let bools = [
            self.r1_same_loss_curve,
            self.r2_same_final_weights,
            self.r3_same_eval_metrics,
            self.r4_environment_locked,
            self.c1_parses_as_rust,
            self.c2_type_checks,
            self.c3_no_unused_warnings,
            self.c4_links_correctly,
            self.x1_tests_pass_on_correct,
            self.x2_tests_fail_on_mutant,
            self.x3_assertions_meaningful,
            self.x4_no_tautologies,
            self.v1_branch_coverage_delta,
            self.v2_line_coverage_delta,
            self.v3_edge_cases_present,
            self.e1_vram_under_8gb,
            self.e2_training_under_4hrs,
            self.e3_inference_under_1s,
            self.g1_handles_generics,
            self.g2_handles_lifetimes,
            self.g3_handles_async,
            self.g4_handles_unsafe,
            self.g5_handles_macros,
            self.d1_test_names_descriptive,
            self.d2_comments_present,
            self.d3_proptest_strategies_clear,
        ];
        bools.iter().filter(|&&b| b).count()
    }

    /// Total number of items
    #[must_use]
    pub const fn total_items(&self) -> usize {
        26
    }

    /// Generate markdown report
    #[must_use]
    pub fn report(&self) -> String {
        let mut out = String::new();
        report_header(&mut out, self);
        report_section(&mut out, "## Reproducibility (20 pts)\n", &self.reproducibility_items());
        report_section(&mut out, "\n## Compilation (20 pts)\n", &self.compilation_items());
        report_section(&mut out, "\n## Correctness (20 pts)\n", &self.correctness_items());
        report_section(&mut out, "\n## Coverage (15 pts)\n", &self.coverage_items());
        report_section(&mut out, "\n## Efficiency (10 pts)\n", &self.efficiency_items());
        report_section(&mut out, "\n## Edge Cases (10 pts)\n", &self.edge_case_items());
        report_section(&mut out, "\n## Documentation (5 pts)\n", &self.documentation_items());
        out
    }

    fn reproducibility_items(&self) -> Vec<(bool, &'static str)> {
        vec![
            (self.r1_same_loss_curve, "R1: Same loss curve"),
            (self.r2_same_final_weights, "R2: Same final weights"),
            (self.r3_same_eval_metrics, "R3: Same eval metrics"),
            (self.r4_environment_locked, "R4: Environment locked"),
        ]
    }

    fn compilation_items(&self) -> Vec<(bool, &'static str)> {
        vec![
            (self.c1_parses_as_rust, "C1: Parses as Rust"),
            (self.c2_type_checks, "C2: Type checks"),
            (self.c3_no_unused_warnings, "C3: No unused warnings"),
            (self.c4_links_correctly, "C4: Links correctly"),
        ]
    }

    fn correctness_items(&self) -> Vec<(bool, &'static str)> {
        vec![
            (self.x1_tests_pass_on_correct, "X1: Tests pass on correct"),
            (self.x2_tests_fail_on_mutant, "X2: Tests fail on mutant"),
            (self.x3_assertions_meaningful, "X3: Assertions meaningful"),
            (self.x4_no_tautologies, "X4: No tautologies"),
        ]
    }

    fn coverage_items(&self) -> Vec<(bool, &'static str)> {
        vec![
            (self.v1_branch_coverage_delta, "V1: Branch coverage +5%"),
            (self.v2_line_coverage_delta, "V2: Line coverage +10%"),
            (self.v3_edge_cases_present, "V3: Edge cases present"),
        ]
    }

    fn efficiency_items(&self) -> Vec<(bool, &'static str)> {
        vec![
            (self.e1_vram_under_8gb, "E1: VRAM < 8GB"),
            (self.e2_training_under_4hrs, "E2: Training < 4hrs"),
            (self.e3_inference_under_1s, "E3: Inference < 1s"),
        ]
    }

    fn edge_case_items(&self) -> Vec<(bool, &'static str)> {
        vec![
            (self.g1_handles_generics, "G1: Handles generics"),
            (self.g2_handles_lifetimes, "G2: Handles lifetimes"),
            (self.g3_handles_async, "G3: Handles async"),
            (self.g4_handles_unsafe, "G4: Handles unsafe"),
            (self.g5_handles_macros, "G5: Handles macros"),
        ]
    }

    fn documentation_items(&self) -> Vec<(bool, &'static str)> {
        vec![
            (self.d1_test_names_descriptive, "D1: Descriptive test names"),
            (self.d2_comments_present, "D2: Comments present"),
            (self.d3_proptest_strategies_clear, "D3: Clear proptest strategies"),
        ]
    }
}

/// Write report header with score, grade, and item count.
fn report_header(out: &mut String, qa: &PopperianQA) {
    out.push_str("# Popperian Falsification QA Report\n\n");
    out.push_str(&format!("**Score:** {}/100\n", qa.score()));
    out.push_str(&format!("**Grade:** {}\n", qa.grade()));
    out.push_str(&format!("**Items Passed:** {}/{}\n\n", qa.passed_count(), qa.total_items()));
}

/// Write a report section with heading and checklist items.
fn report_section(out: &mut String, heading: &str, items: &[(bool, &str)]) {
    out.push_str(heading);
    for &(passed, label) in items {
        let mark = if passed { "x" } else { " " };
        out.push_str(&format!("- [{mark}] {label}\n"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_score_95_to_100_arm() {
        for score in [95u8, 97, 100] {
            match score {
                95..=100 => assert_eq!(QAGrade::from_score(score), QAGrade::APlus),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_from_score_90_to_94_arm() {
        for score in [90u8, 92, 94] {
            match score {
                90..=94 => assert_eq!(QAGrade::from_score(score), QAGrade::A),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_from_score_85_to_89_arm() {
        for score in [85u8, 87, 89] {
            match score {
                85..=89 => assert_eq!(QAGrade::from_score(score), QAGrade::BPlus),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_from_score_80_to_84_arm() {
        for score in [80u8, 82, 84] {
            match score {
                80..=84 => assert_eq!(QAGrade::from_score(score), QAGrade::B),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_from_score_70_to_79_arm() {
        for score in [70u8, 75, 79] {
            match score {
                70..=79 => assert_eq!(QAGrade::from_score(score), QAGrade::C),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_from_score_0_to_69_arm() {
        for score in [0u8, 35, 69] {
            match score {
                0..=69 => assert_eq!(QAGrade::from_score(score), QAGrade::F),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_qa_grade_is_passing() {
        assert!(QAGrade::APlus.is_passing());
        assert!(QAGrade::A.is_passing());
        assert!(QAGrade::BPlus.is_passing());
        assert!(QAGrade::B.is_passing());
        assert!(QAGrade::C.is_passing());
        assert!(!QAGrade::F.is_passing());
    }

    #[test]
    fn test_popperian_qa_new() {
        let qa = PopperianQA::new();
        assert_eq!(qa.score(), 0);
        assert_eq!(qa.grade(), QAGrade::F);
        assert_eq!(qa.passed_count(), 0);
    }

    #[test]
    fn test_popperian_qa_full_score() {
        let qa = PopperianQA {
            r1_same_loss_curve: true,
            r2_same_final_weights: true,
            r3_same_eval_metrics: true,
            r4_environment_locked: true,
            c1_parses_as_rust: true,
            c2_type_checks: true,
            c3_no_unused_warnings: true,
            c4_links_correctly: true,
            x1_tests_pass_on_correct: true,
            x2_tests_fail_on_mutant: true,
            x3_assertions_meaningful: true,
            x4_no_tautologies: true,
            v1_branch_coverage_delta: true,
            v2_line_coverage_delta: true,
            v3_edge_cases_present: true,
            e1_vram_under_8gb: true,
            e2_training_under_4hrs: true,
            e3_inference_under_1s: true,
            g1_handles_generics: true,
            g2_handles_lifetimes: true,
            g3_handles_async: true,
            g4_handles_unsafe: true,
            g5_handles_macros: true,
            d1_test_names_descriptive: true,
            d2_comments_present: true,
            d3_proptest_strategies_clear: true,
        };
        assert_eq!(qa.score(), 100);
        assert_eq!(qa.grade(), QAGrade::APlus);
        assert_eq!(qa.passed_count(), 26);
    }

    #[test]
    fn test_popperian_qa_partial_score() {
        let mut qa = PopperianQA::new();
        // Set all reproducibility (20 pts)
        qa.r1_same_loss_curve = true;
        qa.r2_same_final_weights = true;
        qa.r3_same_eval_metrics = true;
        qa.r4_environment_locked = true;

        assert_eq!(qa.score(), 20);
        assert!(qa.reproducibility_passed());
        assert!(!qa.compilation_passed());
    }

    #[test]
    fn test_popperian_qa_category_checks() {
        let mut qa = PopperianQA::new();

        // Compilation only
        qa.c1_parses_as_rust = true;
        qa.c2_type_checks = true;
        qa.c3_no_unused_warnings = true;
        qa.c4_links_correctly = true;

        assert!(qa.compilation_passed());
        assert!(!qa.reproducibility_passed());
        assert!(!qa.correctness_passed());
    }

    #[test]
    fn test_popperian_qa_report_contains_sections() {
        let qa = PopperianQA::new();
        let report = qa.report();

        assert!(report.contains("# Popperian Falsification QA Report"));
        assert!(report.contains("## Reproducibility"));
        assert!(report.contains("## Compilation"));
        assert!(report.contains("## Correctness"));
        assert!(report.contains("## Coverage"));
        assert!(report.contains("## Efficiency"));
        assert!(report.contains("## Edge Cases"));
        assert!(report.contains("## Documentation"));
    }

    #[test]
    fn test_qa_grade_display_aplus_arm() {
        let g = QAGrade::APlus;
        match g {
            QAGrade::APlus => assert_eq!(g.to_string(), "A+ (Excellent)"),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_qa_grade_display_a_arm() {
        let g = QAGrade::A;
        match g {
            QAGrade::A => assert_eq!(g.to_string(), "A  (Very Good)"),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_qa_grade_display_bplus_arm() {
        let g = QAGrade::BPlus;
        match g {
            QAGrade::BPlus => assert_eq!(g.to_string(), "B+ (Good)"),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_qa_grade_display_b_arm() {
        let g = QAGrade::B;
        match g {
            QAGrade::B => assert_eq!(g.to_string(), "B  (Satisfactory)"),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_qa_grade_display_c_arm() {
        let g = QAGrade::C;
        match g {
            QAGrade::C => assert_eq!(g.to_string(), "C  (Needs Improvement)"),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_qa_grade_display_f_arm() {
        let g = QAGrade::F;
        match g {
            QAGrade::F => assert_eq!(g.to_string(), "F  (Failing)"),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_qa_grade_from_score_overflow() {
        // Tests the 101.. => Self::F arm
        assert_eq!(QAGrade::from_score(101), QAGrade::F);
        assert_eq!(QAGrade::from_score(255), QAGrade::F);
    }

    #[test]
    fn test_qa_grade_ordering() {
        assert!(QAGrade::APlus > QAGrade::A);
        assert!(QAGrade::A > QAGrade::BPlus);
        assert!(QAGrade::BPlus > QAGrade::B);
        assert!(QAGrade::B > QAGrade::C);
        assert!(QAGrade::C > QAGrade::F);
    }
}
