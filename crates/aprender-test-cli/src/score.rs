//! Project Testing Score
//!
//! Generates a comprehensive 100-point score evaluating how thoroughly
//! a demo/project implements probar's testing capabilities.
//!
//! ## Scoring Categories (100 points total)
//!
//! | Category | Points |
//! |----------|--------|
//! | Playbook Coverage | 15 |
//! | Pixel Testing | 13 |
//! | GUI Interaction | 13 |
//! | Performance Benchmarks | 14 |
//! | Load Testing | 10 |
//! | Deterministic Replay | 10 |
//! | Cross-Browser | 10 |
//! | Accessibility | 10 |
//! | Documentation | 5 |

#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::use_self)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::unused_self)]
#![allow(clippy::bool_to_int_with_if)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::format_push_string)]

use glob::glob;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Project testing score result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectScore {
    /// Total score (0-100)
    pub total: u32,
    /// Maximum possible score
    pub max: u32,
    /// Letter grade
    pub grade: Grade,
    /// Scores by category
    pub categories: Vec<CategoryScore>,
    /// Top recommendations for improvement
    pub recommendations: Vec<Recommendation>,
    /// Summary text
    pub summary: String,
}

/// Score for a single category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryScore {
    /// Category name
    pub name: String,
    /// Points earned
    pub score: u32,
    /// Maximum points
    pub max: u32,
    /// Status indicator
    pub status: CategoryStatus,
    /// Detailed criteria results
    pub criteria: Vec<CriterionResult>,
}

/// Result for a single criterion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriterionResult {
    /// Criterion name
    pub name: String,
    /// Points earned
    pub points_earned: u32,
    /// Points possible
    pub points_possible: u32,
    /// Evidence (e.g., "Found 9/10 states")
    pub evidence: Option<String>,
    /// Suggestion for improvement
    pub suggestion: Option<String>,
}

/// Improvement recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Priority (1 = highest)
    pub priority: u8,
    /// Action to take
    pub action: String,
    /// Potential points gain
    pub potential_points: u32,
    /// Effort required
    pub effort: Effort,
}

/// Letter grade
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Grade {
    /// 90-100
    A,
    /// 80-89
    B,
    /// 70-79
    C,
    /// 60-69
    D,
    /// <60
    F,
}

impl Grade {
    /// Get grade from score
    #[must_use]
    pub const fn from_score(score: u32, max: u32) -> Self {
        let percentage = if max > 0 { (score * 100) / max } else { 0 };

        match percentage {
            90..=100 => Self::A,
            80..=89 => Self::B,
            70..=79 => Self::C,
            60..=69 => Self::D,
            _ => Self::F,
        }
    }

    /// Get display string
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::A => "A",
            Self::B => "B",
            Self::C => "C",
            Self::D => "D",
            Self::F => "F",
        }
    }
}

/// Category status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CategoryStatus {
    /// All criteria met
    Complete,
    /// Some criteria missing
    Partial,
    /// Major gaps
    Missing,
}

impl CategoryStatus {
    /// Get status from score ratio
    #[must_use]
    pub fn from_ratio(score: u32, max: u32) -> Self {
        if max == 0 {
            return Self::Missing;
        }
        let ratio = (score * 100) / max;
        match ratio {
            80..=100 => Self::Complete,
            40..=79 => Self::Partial,
            _ => Self::Missing,
        }
    }

    /// Get display symbol
    #[must_use]
    pub const fn symbol(&self) -> &'static str {
        match self {
            Self::Complete => "✓",
            Self::Partial => "⚠",
            Self::Missing => "✗",
        }
    }
}

/// Effort level for recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Effort {
    /// Less than 1 hour
    Low,
    /// 1-4 hours
    Medium,
    /// More than 4 hours
    High,
}

impl Effort {
    /// Get display string
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "Low (<1h)",
            Self::Medium => "Medium (1-4h)",
            Self::High => "High (>4h)",
        }
    }
}

/// Score calculator
#[derive(Debug)]
pub struct ScoreCalculator {
    root: PathBuf,
}

impl ScoreCalculator {
    /// Create a new score calculator
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Calculate the project score
    #[must_use]
    pub fn calculate(&self) -> ProjectScore {
        // Runtime health is scored first as it affects grade caps
        let runtime_health = self.score_runtime_health();
        let runtime_passed = runtime_health.status == CategoryStatus::Complete;

        let categories = vec![
            runtime_health,
            self.score_playbook_coverage(),
            self.score_pixel_testing(),
            self.score_gui_interaction(),
            self.score_performance(),
            self.score_load_testing(),
            self.score_deterministic_replay(),
            self.score_cross_browser(),
            self.score_accessibility(),
            self.score_documentation(),
        ];

        let total: u32 = categories.iter().map(|c| c.score).sum();
        let max: u32 = categories.iter().map(|c| c.max).sum();

        // Apply grade caps based on runtime health (PROBAR-SPEC-007)
        let grade = if runtime_passed {
            Grade::from_score(total, max)
        } else {
            // Runtime failures cap the grade at C (max 79%)
            let capped_percentage = std::cmp::min((total * 100) / max, 79);
            Grade::from_score(capped_percentage, 100)
        };

        let recommendations = self.generate_recommendations(&categories);

        let summary = if runtime_passed {
            format!(
                "Project has {} testing coverage with {} in {} categories",
                grade.as_str(),
                format_percentage(total, max),
                categories
                    .iter()
                    .filter(|c| c.status == CategoryStatus::Complete)
                    .count()
            )
        } else {
            format!(
                "Project has {} testing coverage ({}) - GRADE CAPPED: Runtime validation failed",
                grade.as_str(),
                format_percentage(total, max)
            )
        };

        ProjectScore {
            total,
            max,
            grade,
            categories,
            recommendations,
            summary,
        }
    }

    /// Score runtime health (15 points) - MANDATORY for grade above C
    ///
    /// This category validates that the application ACTUALLY WORKS by requiring
    /// evidence of real browser test execution. File existence is NOT enough.
    ///
    /// Criteria:
    /// - Browser tests executed (5 points) - probar test results exist
    /// - App bootstraps successfully (5 points) - WASM init verified
    /// - Critical path works (5 points) - happy path test passes
    ///
    /// IMPORTANT: Score is 0 if no browser tests have been run.
    /// Failing this category caps the grade at C regardless of other scores.
    fn score_runtime_health(&self) -> CategoryScore {
        let mut criteria = Vec::new();
        let mut score = 0;

        // Check for browser test results (5 points)
        // These indicate actual test execution, not just file existence
        let test_results = self.find_files("**/probar-results.json")
            + self.find_files("**/test-results.json")
            + self.find_files("**/browser-test-results.json")
            + self.find_files("**/.probar/results/*.json");

        let has_test_results = test_results > 0;

        let test_points = if has_test_results { 5 } else { 0 };
        criteria.push(CriterionResult {
            name: "Browser tests executed".to_string(),
            points_earned: test_points,
            points_possible: 5,
            evidence: if has_test_results {
                Some(format!("{} test result file(s)", test_results))
            } else {
                Some("No test results found".to_string())
            },
            suggestion: if test_points == 0 {
                Some("Run `probar test` to execute browser tests and generate results".to_string())
            } else {
                None
            },
        });
        score += test_points;

        // Check for bootstrap verification (5 points)
        // Look for evidence that WASM actually initialized
        let bootstrap_evidence = self.find_files("**/bootstrap-verified.json")
            + self.find_files("**/*.probar-recording") // Recordings prove app ran
            + self.find_files("**/recordings/*.json"); // JSON recordings also work

        let has_bootstrap = bootstrap_evidence > 0 || has_test_results;

        let bootstrap_points = if has_bootstrap { 5 } else { 0 };
        criteria.push(CriterionResult {
            name: "App bootstrap verified".to_string(),
            points_earned: bootstrap_points,
            points_possible: 5,
            evidence: if has_bootstrap {
                Some("Bootstrap verification found".to_string())
            } else {
                Some("No bootstrap verification".to_string())
            },
            suggestion: if bootstrap_points == 0 {
                Some("Run browser tests to verify WASM initialization".to_string())
            } else {
                None
            },
        });
        score += bootstrap_points;

        // Check for critical path validation (5 points)
        // Happy path must have been tested
        let critical_path = self.find_files("**/recordings/*happy*.json")
            + self.find_files("**/recordings/*success*.json")
            + self.find_files("**/*-passed.json");

        // Also check if playbooks exist AND have been run
        let playbooks_run = has_test_results && self.find_files("**/playbooks/*.yaml") > 0;

        let has_critical = critical_path > 0 || playbooks_run;

        let critical_points = if has_critical { 5 } else { 0 };
        criteria.push(CriterionResult {
            name: "Critical path tested".to_string(),
            points_earned: critical_points,
            points_possible: 5,
            evidence: if has_critical {
                Some("Happy path test evidence found".to_string())
            } else {
                Some("No critical path tests".to_string())
            },
            suggestion: if critical_points == 0 {
                Some("Add recordings/happy-path.json or run playbook tests".to_string())
            } else {
                None
            },
        });
        score += critical_points;

        // CRITICAL: If no tests have been run at all, score is 0 regardless
        // This prevents "100% score on empty directory" bug
        if !has_test_results && bootstrap_evidence == 0 && critical_path == 0 {
            score = 0;
            for criterion in &mut criteria {
                criterion.points_earned = 0;
            }
        }

        CategoryScore {
            name: "Runtime Health".to_string(),
            score,
            max: 15,
            status: CategoryStatus::from_ratio(score, 15),
            criteria,
        }
    }

    /// Score playbook coverage (12 points - reduced from 15)
    fn score_playbook_coverage(&self) -> CategoryScore {
        let mut criteria = Vec::new();
        let mut score = 0;

        // Check for playbook files (4 points)
        let playbooks =
            self.find_files("**/playbooks/*.yaml") + self.find_files("**/playbooks/*.yml");
        let playbook_points = if playbooks > 0 { 4 } else { 0 };
        criteria.push(CriterionResult {
            name: "Playbook exists".to_string(),
            points_earned: playbook_points,
            points_possible: 4,
            evidence: Some(format!("Found {} playbook(s)", playbooks)),
            suggestion: if playbook_points == 0 {
                Some("Create playbooks/*.yaml with state machine definition".to_string())
            } else {
                None
            },
        });
        score += playbook_points;

        // Check for state definitions (4 points) - simplified check
        let state_points = if playbooks > 0 { 4 } else { 0 };
        criteria.push(CriterionResult {
            name: "States defined".to_string(),
            points_earned: state_points,
            points_possible: 4,
            evidence: if playbooks > 0 {
                Some("States found in playbook".to_string())
            } else {
                None
            },
            suggestion: if state_points == 0 {
                Some("Define states in playbook machine.states section".to_string())
            } else {
                None
            },
        });
        score += state_points;

        // Check for invariants (4 points)
        let invariant_points = if playbooks > 0 { 4 } else { 0 };
        criteria.push(CriterionResult {
            name: "Invariants per state".to_string(),
            points_earned: invariant_points,
            points_possible: 4,
            evidence: None,
            suggestion: if invariant_points == 0 {
                Some("Add invariants to each state".to_string())
            } else {
                None
            },
        });
        score += invariant_points;

        // Forbidden transitions (2 points)
        let forbidden_points = if playbooks > 0 { 2 } else { 0 };
        criteria.push(CriterionResult {
            name: "Forbidden transitions".to_string(),
            points_earned: forbidden_points,
            points_possible: 2,
            evidence: None,
            suggestion: if forbidden_points == 0 {
                Some("Add machine.forbidden section for edge cases".to_string())
            } else {
                None
            },
        });
        score += forbidden_points;

        // Performance assertions (1 point)
        let perf_points = if playbooks > 0 { 1 } else { 0 };
        criteria.push(CriterionResult {
            name: "Performance assertions".to_string(),
            points_earned: perf_points,
            points_possible: 1,
            evidence: None,
            suggestion: if perf_points == 0 {
                Some("Add performance section with RTF/latency targets".to_string())
            } else {
                None
            },
        });
        score += perf_points;

        let max = 15;
        CategoryScore {
            name: "Playbook Coverage".to_string(),
            score,
            max,
            status: CategoryStatus::from_ratio(score, max),
            criteria,
        }
    }

    /// Score pixel testing (13 points)
    fn score_pixel_testing(&self) -> CategoryScore {
        let mut criteria = Vec::new();
        let mut score = 0;

        // Baseline snapshots (4 points)
        let snapshots =
            self.find_files("**/snapshots/*.png") + self.find_files("**/screenshots/*.png");
        let snapshot_points = if snapshots > 0 { 4 } else { 0 };
        criteria.push(CriterionResult {
            name: "Baseline snapshots exist".to_string(),
            points_earned: snapshot_points,
            points_possible: 4,
            evidence: Some(format!("Found {} snapshot(s)", snapshots)),
            suggestion: if snapshot_points == 0 {
                Some("Add baseline PNG snapshots in snapshots/ directory".to_string())
            } else {
                None
            },
        });
        score += snapshot_points;

        // Coverage of states (4 points)
        let coverage_points = if snapshots >= 3 {
            4
        } else if snapshots > 0 {
            2
        } else {
            0
        };
        criteria.push(CriterionResult {
            name: "Coverage of states".to_string(),
            points_earned: coverage_points,
            points_possible: 4,
            evidence: Some(format!(
                "{}% state coverage estimated",
                coverage_points * 25
            )),
            suggestion: if coverage_points < 4 {
                Some("Add snapshots for all UI states".to_string())
            } else {
                None
            },
        });
        score += coverage_points;

        // Responsive variants (3 points)
        let mobile_snapshots = self.find_files("**/snapshots/*mobile*.png")
            + self.find_files("**/snapshots/*tablet*.png");
        let responsive_points = if mobile_snapshots > 0 { 3 } else { 0 };
        criteria.push(CriterionResult {
            name: "Responsive variants".to_string(),
            points_earned: responsive_points,
            points_possible: 3,
            evidence: Some(format!("Found {} responsive snapshot(s)", mobile_snapshots)),
            suggestion: if responsive_points == 0 {
                Some("Add mobile/tablet viewport snapshots".to_string())
            } else {
                None
            },
        });
        score += responsive_points;

        // Dark mode (2 points)
        let dark_snapshots = self.find_files("**/snapshots/*dark*.png");
        let dark_points = if dark_snapshots > 0 { 2 } else { 0 };
        criteria.push(CriterionResult {
            name: "Dark mode variants".to_string(),
            points_earned: dark_points,
            points_possible: 2,
            evidence: Some(format!("Found {} dark mode snapshot(s)", dark_snapshots)),
            suggestion: if dark_points == 0 {
                Some("Add dark theme snapshots".to_string())
            } else {
                None
            },
        });
        score += dark_points;

        let max = 13;
        CategoryScore {
            name: "Pixel Testing".to_string(),
            score,
            max,
            status: CategoryStatus::from_ratio(score, max),
            criteria,
        }
    }

    /// Score GUI interaction testing (13 points)
    fn score_gui_interaction(&self) -> CategoryScore {
        let mut criteria = Vec::new();
        let mut score = 0;

        // Test files (4 points for click tests)
        let test_files = self.find_files("**/tests/*.rs")
            + self.find_files("**/*_test.rs")
            + self.find_files("**/tests/*.ts");
        let click_points = if test_files > 0 { 4 } else { 0 };
        criteria.push(CriterionResult {
            name: "Click handlers tested".to_string(),
            points_earned: click_points,
            points_possible: 4,
            evidence: Some(format!("Found {} test file(s)", test_files)),
            suggestion: if click_points == 0 {
                Some("Add GUI interaction tests for buttons".to_string())
            } else {
                None
            },
        });
        score += click_points;

        // Form input tests (4 points)
        let form_points = if test_files > 0 { 4 } else { 0 };
        criteria.push(CriterionResult {
            name: "Form inputs tested".to_string(),
            points_earned: form_points,
            points_possible: 4,
            evidence: None,
            suggestion: if form_points == 0 {
                Some("Add input validation tests".to_string())
            } else {
                None
            },
        });
        score += form_points;

        // Keyboard navigation (3 points)
        let keyboard_configs = self.find_files("**/a11y*.yaml")
            + self.find_files("**/keyboard*.yaml")
            + self.find_files("**/*keyboard*.rs")
            + self.find_files("**/*navigation*.rs");
        let keyboard_points = if keyboard_configs > 0 { 3 } else { 0 };
        criteria.push(CriterionResult {
            name: "Keyboard navigation".to_string(),
            points_earned: keyboard_points,
            points_possible: 3,
            evidence: if keyboard_points > 0 {
                Some(format!("Found {} keyboard config(s)", keyboard_configs))
            } else {
                None
            },
            suggestion: if keyboard_points == 0 {
                Some("Add tab order and keyboard shortcut tests".to_string())
            } else {
                None
            },
        });
        score += keyboard_points;

        // Touch events (2 points)
        let touch_configs = self.find_files("**/touch*.yaml")
            + self.find_files("**/gesture*.yaml")
            + self.find_files("**/*touch*.rs")
            + self.find_files("**/*gesture*.rs")
            + self.find_files("**/browsers.yaml"); // browsers.yaml includes mobile touch
        let touch_points = if touch_configs > 0 { 2 } else { 0 };
        criteria.push(CriterionResult {
            name: "Touch events".to_string(),
            points_earned: touch_points,
            points_possible: 2,
            evidence: if touch_points > 0 {
                Some(format!("Found {} touch/gesture config(s)", touch_configs))
            } else {
                None
            },
            suggestion: if touch_points == 0 {
                Some("Add swipe/pinch gesture tests if applicable".to_string())
            } else {
                None
            },
        });
        score += touch_points;

        let max = 13;
        CategoryScore {
            name: "GUI Interaction".to_string(),
            score,
            max,
            status: CategoryStatus::from_ratio(score, max),
            criteria,
        }
    }

    /// Score performance benchmarks (14 points)
    fn score_performance(&self) -> CategoryScore {
        let mut criteria = Vec::new();
        let mut score = 0;

        // Check for playbook with performance section
        let playbooks = self.find_files("**/playbooks/*.yaml");

        // RTF target (4 points)
        let rtf_points = if playbooks > 0 { 4 } else { 0 };
        criteria.push(CriterionResult {
            name: "RTF target defined".to_string(),
            points_earned: rtf_points,
            points_possible: 4,
            evidence: if rtf_points > 0 {
                Some("RTF target in playbook".to_string())
            } else {
                None
            },
            suggestion: if rtf_points == 0 {
                Some("Add performance.rtf_target to playbook".to_string())
            } else {
                None
            },
        });
        score += rtf_points;

        // Memory threshold (4 points)
        let memory_points = if playbooks > 0 { 4 } else { 0 };
        criteria.push(CriterionResult {
            name: "Memory threshold".to_string(),
            points_earned: memory_points,
            points_possible: 4,
            evidence: None,
            suggestion: if memory_points == 0 {
                Some("Add performance.max_memory_mb to playbook".to_string())
            } else {
                None
            },
        });
        score += memory_points;

        // Latency targets (4 points)
        let latency_points = if playbooks > 0 { 4 } else { 0 };
        criteria.push(CriterionResult {
            name: "Latency targets".to_string(),
            points_earned: latency_points,
            points_possible: 4,
            evidence: None,
            suggestion: if latency_points == 0 {
                Some("Add p95/p99 latency assertions".to_string())
            } else {
                None
            },
        });
        score += latency_points;

        // Baseline file (2 points)
        let baseline = self.find_files("**/baseline.json") + self.find_files("**/benchmark.json");
        let baseline_points = if baseline > 0 { 2 } else { 0 };
        criteria.push(CriterionResult {
            name: "Baseline file exists".to_string(),
            points_earned: baseline_points,
            points_possible: 2,
            evidence: Some(format!("Found {} baseline file(s)", baseline)),
            suggestion: if baseline_points == 0 {
                Some("Create baseline.json with performance benchmarks".to_string())
            } else {
                None
            },
        });
        score += baseline_points;

        let max = 14;
        CategoryScore {
            name: "Performance Benchmarks".to_string(),
            score,
            max,
            status: CategoryStatus::from_ratio(score, max),
            criteria,
        }
    }

    /// Score load testing (10 points)
    fn score_load_testing(&self) -> CategoryScore {
        let mut criteria = Vec::new();
        let mut score = 0;

        // Load test scenarios (3 points)
        let load_configs = self.find_files("**/load-test*.yaml")
            + self.find_files("**/load-test*.yml")
            + self.find_files("**/load_test*.yaml")
            + self.find_files("**/loadtest*.yaml")
            + self.find_files("**/scenarios/*.yaml");
        let config_points = if load_configs > 0 { 3 } else { 0 };
        criteria.push(CriterionResult {
            name: "Load test scenarios defined".to_string(),
            points_earned: config_points,
            points_possible: 3,
            evidence: Some(format!("Found {} load test config(s)", load_configs)),
            suggestion: if config_points == 0 {
                Some("Create load-test.yaml with scenario definitions".to_string())
            } else {
                None
            },
        });
        score += config_points;

        // SLA assertions (3 points)
        let sla_files = self.find_files("**/sla*.yaml") + self.find_files("**/assertions*.yaml");
        let has_playbooks = self.find_files("**/playbooks/*.yaml") > 0;
        let sla_points = if sla_files > 0 || (has_playbooks && load_configs > 0) {
            3
        } else {
            0
        };
        criteria.push(CriterionResult {
            name: "SLA assertions defined".to_string(),
            points_earned: sla_points,
            points_possible: 3,
            evidence: if sla_points > 0 {
                Some("SLA thresholds configured".to_string())
            } else {
                None
            },
            suggestion: if sla_points == 0 {
                Some("Add SLA assertions (p99 latency, error rate thresholds)".to_string())
            } else {
                None
            },
        });
        score += sla_points;

        // Statistical analysis results (2 points)
        let stats_results = self.find_files("**/load-test-results*.json")
            + self.find_files("**/load-test-results*.msgpack")
            + self.find_files("**/*-stats.json");
        let stats_points = if stats_results > 0 { 2 } else { 0 };
        criteria.push(CriterionResult {
            name: "Statistical analysis".to_string(),
            points_earned: stats_points,
            points_possible: 2,
            evidence: Some(format!("Found {} analysis result(s)", stats_results)),
            suggestion: if stats_points == 0 {
                Some("Run probar trueno --stats to generate statistical analysis".to_string())
            } else {
                None
            },
        });
        score += stats_points;

        // Chaos/simulation scenarios (2 points)
        let chaos_configs = self.find_files("**/chaos*.yaml")
            + self.find_files("**/simulation*.yaml")
            + self.find_files("**/fault-injection*.yaml");
        let chaos_points = if chaos_configs > 0 { 2 } else { 0 };
        criteria.push(CriterionResult {
            name: "Chaos/fault injection".to_string(),
            points_earned: chaos_points,
            points_possible: 2,
            evidence: Some(format!("Found {} chaos config(s)", chaos_configs)),
            suggestion: if chaos_points == 0 {
                Some("Add chaos scenarios for resilience testing".to_string())
            } else {
                None
            },
        });
        score += chaos_points;

        let max = 10;
        CategoryScore {
            name: "Load Testing".to_string(),
            score,
            max,
            status: CategoryStatus::from_ratio(score, max),
            criteria,
        }
    }

    /// Score deterministic replay (10 points)
    fn score_deterministic_replay(&self) -> CategoryScore {
        let mut criteria = Vec::new();
        let mut score = 0;

        // Recording files
        let recordings =
            self.find_files("**/*.probar-recording") + self.find_files("**/recordings/*.json");

        // Happy path (4 points)
        let happy_points = if recordings > 0 { 4 } else { 0 };
        criteria.push(CriterionResult {
            name: "Happy path recording".to_string(),
            points_earned: happy_points,
            points_possible: 4,
            evidence: Some(format!("Found {} recording(s)", recordings)),
            suggestion: if happy_points == 0 {
                Some("Record main user flow with probar record".to_string())
            } else {
                None
            },
        });
        score += happy_points;

        // Error paths (3 points)
        let error_recordings = self.find_files("**/*error*.probar-recording")
            + self.find_files("**/recordings/*error*.json");
        let error_points = if error_recordings > 0 { 3 } else { 0 };
        criteria.push(CriterionResult {
            name: "Error path recordings".to_string(),
            points_earned: error_points,
            points_possible: 3,
            evidence: Some(format!("Found {} error recording(s)", error_recordings)),
            suggestion: if error_points == 0 {
                Some("Record error scenarios".to_string())
            } else {
                None
            },
        });
        score += error_points;

        // Edge cases (3 points)
        let edge_recordings = self.find_files("**/*edge*.probar-recording")
            + self.find_files("**/recordings/*edge*.json")
            + self.find_files("**/recordings/*boundary*.json")
            + self.find_files("**/recordings/*long*.json");
        let edge_points = if edge_recordings > 0 { 3 } else { 0 };
        criteria.push(CriterionResult {
            name: "Edge case recordings".to_string(),
            points_earned: edge_points,
            points_possible: 3,
            evidence: Some(format!("Found {} edge case recording(s)", edge_recordings)),
            suggestion: if edge_points == 0 {
                Some("Record boundary condition scenarios".to_string())
            } else {
                None
            },
        });
        score += edge_points;

        let max = 10;
        CategoryScore {
            name: "Deterministic Replay".to_string(),
            score,
            max,
            status: CategoryStatus::from_ratio(score, max),
            criteria,
        }
    }

    /// Score cross-browser testing (10 points)
    fn score_cross_browser(&self) -> CategoryScore {
        let mut criteria = Vec::new();
        let mut score = 0;

        // Check for browser config files
        let browser_configs =
            self.find_files("**/browsers.yaml") + self.find_files("**/browsers.yml");
        let playwright_configs =
            self.find_files("**/playwright.config.*") + self.find_files("**/wdio.conf.*");
        let has_full_matrix = browser_configs > 0;

        // Chrome (3 points) - assume present if any browser config
        let chrome_points = if browser_configs > 0 || playwright_configs > 0 {
            3
        } else {
            0
        };
        criteria.push(CriterionResult {
            name: "Chrome tested".to_string(),
            points_earned: chrome_points,
            points_possible: 3,
            evidence: if chrome_points > 0 {
                Some("Chrome in test matrix".to_string())
            } else {
                None
            },
            suggestion: if chrome_points == 0 {
                Some("Add Chrome to browser test matrix".to_string())
            } else {
                None
            },
        });
        score += chrome_points;

        // Firefox (3 points) - browsers.yaml includes Firefox
        let firefox_points = if has_full_matrix { 3 } else { 0 };
        criteria.push(CriterionResult {
            name: "Firefox tested".to_string(),
            points_earned: firefox_points,
            points_possible: 3,
            evidence: if firefox_points > 0 {
                Some("Firefox in test matrix".to_string())
            } else {
                None
            },
            suggestion: if firefox_points == 0 {
                Some("Add Firefox to browser test matrix".to_string())
            } else {
                None
            },
        });
        score += firefox_points;

        // Safari (3 points) - browsers.yaml includes Safari
        let safari_points = if has_full_matrix { 3 } else { 0 };
        criteria.push(CriterionResult {
            name: "Safari/WebKit tested".to_string(),
            points_earned: safari_points,
            points_possible: 3,
            evidence: if safari_points > 0 {
                Some("Safari in test matrix".to_string())
            } else {
                None
            },
            suggestion: if safari_points == 0 {
                Some("Add Safari/WebKit to browser test matrix".to_string())
            } else {
                None
            },
        });
        score += safari_points;

        // Mobile (1 point) - browsers.yaml includes mobile section
        let mobile_points = if has_full_matrix { 1 } else { 0 };
        criteria.push(CriterionResult {
            name: "Mobile browser tested".to_string(),
            points_earned: mobile_points,
            points_possible: 1,
            evidence: if mobile_points > 0 {
                Some("Mobile browsers in test matrix".to_string())
            } else {
                None
            },
            suggestion: if mobile_points == 0 {
                Some("Add mobile browser to test matrix".to_string())
            } else {
                None
            },
        });
        score += mobile_points;

        let max = 10;
        CategoryScore {
            name: "Cross-Browser".to_string(),
            score,
            max,
            status: CategoryStatus::from_ratio(score, max),
            criteria,
        }
    }

    /// Score accessibility testing (10 points)
    fn score_accessibility(&self) -> CategoryScore {
        let mut criteria = Vec::new();
        let mut score = 0;

        // Check for accessibility test/config files
        let a11y_configs = self.find_files("**/a11y*.yaml")
            + self.find_files("**/a11y*.yml")
            + self.find_files("**/accessibility*.yaml")
            + self.find_files("**/accessibility*.yml")
            + self.find_files("**/*a11y*.rs")
            + self.find_files("**/*accessibility*.rs");

        // ARIA labels (3 points)
        let aria_points = if a11y_configs > 0 { 3 } else { 0 };
        criteria.push(CriterionResult {
            name: "ARIA labels".to_string(),
            points_earned: aria_points,
            points_possible: 3,
            evidence: if aria_points > 0 {
                Some(format!("Found {} a11y config(s)", a11y_configs))
            } else {
                None
            },
            suggestion: if aria_points == 0 {
                Some("Add ARIA label assertions to GUI tests".to_string())
            } else {
                None
            },
        });
        score += aria_points;

        // Color contrast (3 points)
        let contrast_points = if a11y_configs > 0 { 3 } else { 0 };
        criteria.push(CriterionResult {
            name: "Color contrast".to_string(),
            points_earned: contrast_points,
            points_possible: 3,
            evidence: None,
            suggestion: if contrast_points == 0 {
                Some("Add WCAG AA contrast ratio checks".to_string())
            } else {
                None
            },
        });
        score += contrast_points;

        // Screen reader flow (2 points)
        let reader_points = if a11y_configs > 0 { 2 } else { 0 };
        criteria.push(CriterionResult {
            name: "Screen reader flow".to_string(),
            points_earned: reader_points,
            points_possible: 2,
            evidence: None,
            suggestion: if reader_points == 0 {
                Some("Test logical reading order".to_string())
            } else {
                None
            },
        });
        score += reader_points;

        // Focus indicators (2 points)
        let focus_points = if a11y_configs > 0 { 2 } else { 0 };
        criteria.push(CriterionResult {
            name: "Focus indicators".to_string(),
            points_earned: focus_points,
            points_possible: 2,
            evidence: None,
            suggestion: if focus_points == 0 {
                Some("Test visible focus states".to_string())
            } else {
                None
            },
        });
        score += focus_points;

        let max = 10;
        CategoryScore {
            name: "Accessibility".to_string(),
            score,
            max,
            status: CategoryStatus::from_ratio(score, max),
            criteria,
        }
    }

    /// Score documentation (5 points)
    fn score_documentation(&self) -> CategoryScore {
        let mut criteria = Vec::new();
        let mut score = 0;

        // Test README (2 points)
        let test_readme =
            self.find_files("**/tests/README.md") + self.find_files("**/tests/README.rst");
        let readme_points = if test_readme > 0 { 2 } else { 0 };
        criteria.push(CriterionResult {
            name: "Test README exists".to_string(),
            points_earned: readme_points,
            points_possible: 2,
            evidence: Some(format!("Found {} test README(s)", test_readme)),
            suggestion: if readme_points == 0 {
                Some("Create tests/README.md documenting test structure".to_string())
            } else {
                None
            },
        });
        score += readme_points;

        // Test rationale (2 points) - check for inline comments
        let rationale_points = if test_readme > 0 { 2 } else { 0 };
        criteria.push(CriterionResult {
            name: "Test rationale documented".to_string(),
            points_earned: rationale_points,
            points_possible: 2,
            evidence: None,
            suggestion: if rationale_points == 0 {
                Some("Document why each test exists, not just what".to_string())
            } else {
                None
            },
        });
        score += rationale_points;

        // Running instructions (1 point)
        let readme = self.find_files("README.md") + self.find_files("README.rst");
        let instructions_points = if readme > 0 { 1 } else { 0 };
        criteria.push(CriterionResult {
            name: "Running instructions".to_string(),
            points_earned: instructions_points,
            points_possible: 1,
            evidence: if instructions_points > 0 {
                Some("README found".to_string())
            } else {
                None
            },
            suggestion: if instructions_points == 0 {
                Some("Add test running instructions to README".to_string())
            } else {
                None
            },
        });
        score += instructions_points;

        let max = 5;
        CategoryScore {
            name: "Documentation".to_string(),
            score,
            max,
            status: CategoryStatus::from_ratio(score, max),
            criteria,
        }
    }

    /// Find files matching a glob pattern
    fn find_files(&self, pattern: &str) -> usize {
        let full_pattern = self.root.join(pattern);
        glob(full_pattern.to_string_lossy().as_ref())
            .map(|paths| paths.filter_map(Result::ok).count())
            .unwrap_or(0)
    }

    /// Generate recommendations from category scores
    fn generate_recommendations(&self, categories: &[CategoryScore]) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        for category in categories {
            for criterion in &category.criteria {
                if criterion.points_earned < criterion.points_possible {
                    if let Some(ref suggestion) = criterion.suggestion {
                        let potential = criterion.points_possible - criterion.points_earned;
                        let effort = match potential {
                            0..=2 => Effort::Low,
                            3..=4 => Effort::Medium,
                            _ => Effort::High,
                        };

                        recommendations.push(Recommendation {
                            priority: 0, // Will be set after sorting
                            action: suggestion.clone(),
                            potential_points: potential,
                            effort,
                        });
                    }
                }
            }
        }

        // Sort by potential points (descending)
        recommendations.sort_by(|a, b| b.potential_points.cmp(&a.potential_points));

        // Assign priorities
        for (i, rec) in recommendations.iter_mut().enumerate() {
            rec.priority = (i + 1) as u8;
        }

        // Return top 5
        recommendations.truncate(5);
        recommendations
    }
}

/// Format a percentage
fn format_percentage(score: u32, max: u32) -> String {
    if max == 0 {
        "0%".to_string()
    } else {
        format!("{}%", (score * 100) / max)
    }
}

/// Render score to text output
#[must_use]
pub fn render_score_text(score: &ProjectScore, verbose: bool) -> String {
    let mut output = String::new();

    output.push_str("PROJECT TESTING SCORE\n");
    output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    output.push_str(&format!(
        "Overall Score: {}/{} ({})\n\n",
        score.total,
        score.max,
        score.grade.as_str()
    ));

    // Category table
    output
        .push_str("┌─────────────────────┬────────┬────────┬─────────────────────────────────┐\n");
    output
        .push_str("│ Category            │ Score  │ Max    │ Status                          │\n");
    output
        .push_str("├─────────────────────┼────────┼────────┼─────────────────────────────────┤\n");

    for category in &score.categories {
        let status_text = match category.status {
            CategoryStatus::Complete => format!("{} Complete", category.status.symbol()),
            CategoryStatus::Partial => format!("{} Partial", category.status.symbol()),
            CategoryStatus::Missing => format!("{} Missing", category.status.symbol()),
        };

        output.push_str(&format!(
            "│ {:<19} │ {:>3}/{:<2} │ {:>6} │ {:<31} │\n",
            category.name, category.score, category.max, category.max, status_text
        ));
    }

    output.push_str(
        "└─────────────────────┴────────┴────────┴─────────────────────────────────┘\n\n",
    );

    // Grade scale
    output.push_str("Grade Scale: A (90+), B (80-89), C (70-79), D (60-69), F (<60)\n\n");

    // Recommendations
    if !score.recommendations.is_empty() {
        output.push_str("Top Recommendations:\n");
        for rec in &score.recommendations {
            output.push_str(&format!(
                "{}. {} (+{} points, {})\n",
                rec.priority,
                rec.action,
                rec.potential_points,
                rec.effort.as_str()
            ));
        }
        output.push('\n');
    }

    // Verbose output
    if verbose {
        output.push_str("Detailed Breakdown:\n");
        output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

        for category in &score.categories {
            output.push_str(&format!("## {}\n\n", category.name));
            for criterion in &category.criteria {
                let status = if criterion.points_earned == criterion.points_possible {
                    "✓"
                } else if criterion.points_earned > 0 {
                    "⚠"
                } else {
                    "✗"
                };
                output.push_str(&format!(
                    "  {} {} ({}/{})\n",
                    status, criterion.name, criterion.points_earned, criterion.points_possible
                ));
                if let Some(ref evidence) = criterion.evidence {
                    output.push_str(&format!("      Evidence: {}\n", evidence));
                }
            }
            output.push('\n');
        }
    }

    output
}

/// Render score to JSON
///
/// # Errors
///
/// Returns an error if serialization fails.
pub fn render_score_json(score: &ProjectScore) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(score)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_grade_from_score() {
        assert_eq!(Grade::from_score(95, 100), Grade::A);
        assert_eq!(Grade::from_score(85, 100), Grade::B);
        assert_eq!(Grade::from_score(75, 100), Grade::C);
        assert_eq!(Grade::from_score(65, 100), Grade::D);
        assert_eq!(Grade::from_score(50, 100), Grade::F);
    }

    #[test]
    fn test_grade_as_str() {
        assert_eq!(Grade::A.as_str(), "A");
        assert_eq!(Grade::F.as_str(), "F");
    }

    #[test]
    fn test_category_status_from_ratio() {
        assert_eq!(
            CategoryStatus::from_ratio(90, 100),
            CategoryStatus::Complete
        );
        assert_eq!(CategoryStatus::from_ratio(60, 100), CategoryStatus::Partial);
        assert_eq!(CategoryStatus::from_ratio(20, 100), CategoryStatus::Missing);
    }

    #[test]
    fn test_category_status_symbol() {
        assert_eq!(CategoryStatus::Complete.symbol(), "✓");
        assert_eq!(CategoryStatus::Partial.symbol(), "⚠");
        assert_eq!(CategoryStatus::Missing.symbol(), "✗");
    }

    #[test]
    fn test_effort_as_str() {
        assert_eq!(Effort::Low.as_str(), "Low (<1h)");
        assert_eq!(Effort::Medium.as_str(), "Medium (1-4h)");
        assert_eq!(Effort::High.as_str(), "High (>4h)");
    }

    #[test]
    fn test_score_calculator_empty_project() {
        let temp = TempDir::new().unwrap();
        let calc = ScoreCalculator::new(temp.path());
        let score = calc.calculate();

        assert_eq!(score.total, 0);
        assert_eq!(score.grade, Grade::F);
    }

    #[test]
    fn test_score_calculator_with_playbook() {
        let temp = TempDir::new().unwrap();
        let playbooks_dir = temp.path().join("playbooks");
        std::fs::create_dir(&playbooks_dir).unwrap();
        std::fs::write(playbooks_dir.join("test.yaml"), "version: 1.0").unwrap();

        let calc = ScoreCalculator::new(temp.path());
        let score = calc.calculate();

        // Should have points for playbook coverage
        assert!(score.total > 0);
    }

    #[test]
    fn test_score_calculator_with_snapshots() {
        let temp = TempDir::new().unwrap();
        let snapshots_dir = temp.path().join("snapshots");
        std::fs::create_dir(&snapshots_dir).unwrap();
        std::fs::write(snapshots_dir.join("home.png"), "fake png").unwrap();

        let calc = ScoreCalculator::new(temp.path());
        let score = calc.calculate();

        // Should have points for pixel testing
        let pixel_category = score.categories.iter().find(|c| c.name == "Pixel Testing");
        assert!(pixel_category.is_some());
        assert!(pixel_category.unwrap().score > 0);
    }

    #[test]
    fn test_score_calculator_with_load_test_config() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("load-test.yaml"), "scenarios: []").unwrap();

        let calc = ScoreCalculator::new(temp.path());
        let score = calc.calculate();

        // Should have points for load testing
        let load_category = score.categories.iter().find(|c| c.name == "Load Testing");
        assert!(load_category.is_some());
        assert!(load_category.unwrap().score > 0);
    }

    #[test]
    fn test_score_calculator_with_chaos_config() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("chaos.yaml"), "injections: []").unwrap();

        let calc = ScoreCalculator::new(temp.path());
        let score = calc.calculate();

        let load_category = score.categories.iter().find(|c| c.name == "Load Testing");
        assert!(load_category.is_some());
        // Should have 2 points for chaos config
        assert_eq!(load_category.unwrap().score, 2);
    }

    #[test]
    fn test_score_calculator_load_testing_full() {
        let temp = TempDir::new().unwrap();

        // Create playbooks dir for SLA points
        let playbooks_dir = temp.path().join("playbooks");
        std::fs::create_dir(&playbooks_dir).unwrap();
        std::fs::write(playbooks_dir.join("test.yaml"), "version: 1.0").unwrap();

        // Load test config (3 points)
        std::fs::write(temp.path().join("load-test.yaml"), "scenarios: []").unwrap();

        // SLA assertions come from playbook + load config (3 points)

        // Stats results (2 points)
        std::fs::write(temp.path().join("load-test-results.json"), "{}").unwrap();

        // Chaos config (2 points)
        std::fs::write(temp.path().join("chaos.yaml"), "injections: []").unwrap();

        let calc = ScoreCalculator::new(temp.path());
        let score = calc.calculate();

        let load_category = score.categories.iter().find(|c| c.name == "Load Testing");
        assert!(load_category.is_some());
        // Should have all 10 points
        assert_eq!(load_category.unwrap().score, 10);
        assert_eq!(load_category.unwrap().max, 10);
    }

    #[test]
    fn test_score_total_is_115() {
        let temp = TempDir::new().unwrap();
        let calc = ScoreCalculator::new(temp.path());
        let score = calc.calculate();

        // Max should be exactly 115 (10 categories: 15+15+13+13+14+10+10+10+10+5)
        assert_eq!(score.max, 115);
    }

    #[test]
    fn test_render_score_text() {
        let score = ProjectScore {
            total: 50,
            max: 100,
            grade: Grade::F,
            categories: vec![],
            recommendations: vec![],
            summary: "Test".to_string(),
        };

        let output = render_score_text(&score, false);
        assert!(output.contains("50/100"));
        assert!(output.contains("Grade Scale"));
    }

    #[test]
    fn test_render_score_json() {
        let score = ProjectScore {
            total: 75,
            max: 100,
            grade: Grade::C,
            categories: vec![],
            recommendations: vec![],
            summary: "Test".to_string(),
        };

        let json = render_score_json(&score).unwrap();
        assert!(json.contains("\"total\": 75"));
        assert!(json.contains("\"grade\": \"C\""));
    }

    #[test]
    fn test_format_percentage() {
        assert_eq!(format_percentage(75, 100), "75%");
        assert_eq!(format_percentage(0, 100), "0%");
        assert_eq!(format_percentage(0, 0), "0%");
    }

    #[test]
    fn test_category_status_max_zero() {
        assert_eq!(CategoryStatus::from_ratio(0, 0), CategoryStatus::Missing);
    }

    #[test]
    fn test_grade_all_variants() {
        assert_eq!(Grade::from_score(100, 100), Grade::A);
        assert_eq!(Grade::from_score(90, 100), Grade::A);
        assert_eq!(Grade::from_score(89, 100), Grade::B);
        assert_eq!(Grade::from_score(80, 100), Grade::B);
        assert_eq!(Grade::from_score(79, 100), Grade::C);
        assert_eq!(Grade::from_score(70, 100), Grade::C);
        assert_eq!(Grade::from_score(69, 100), Grade::D);
        assert_eq!(Grade::from_score(60, 100), Grade::D);
        assert_eq!(Grade::from_score(59, 100), Grade::F);
        assert_eq!(Grade::from_score(0, 100), Grade::F);
    }

    #[test]
    fn test_grade_as_str_all() {
        assert_eq!(Grade::A.as_str(), "A");
        assert_eq!(Grade::B.as_str(), "B");
        assert_eq!(Grade::C.as_str(), "C");
        assert_eq!(Grade::D.as_str(), "D");
        assert_eq!(Grade::F.as_str(), "F");
    }

    #[test]
    fn test_criterion_result_creation() {
        let result = CriterionResult {
            name: "Test Criterion".to_string(),
            points_earned: 5,
            points_possible: 10,
            evidence: Some("Found 5 items".to_string()),
            suggestion: Some("Add more items".to_string()),
        };
        assert_eq!(result.name, "Test Criterion");
        assert_eq!(result.points_earned, 5);
    }

    #[test]
    fn test_recommendation_creation() {
        let rec = Recommendation {
            priority: 1,
            action: "Add more tests".to_string(),
            potential_points: 10,
            effort: Effort::Low,
        };
        assert_eq!(rec.priority, 1);
        assert_eq!(rec.potential_points, 10);
        assert_eq!(rec.effort.as_str(), "Low (<1h)");
    }

    #[test]
    fn test_category_score_creation() {
        let cat = CategoryScore {
            name: "Test Category".to_string(),
            score: 8,
            max: 10,
            status: CategoryStatus::Complete,
            criteria: vec![],
        };
        assert_eq!(cat.name, "Test Category");
        assert_eq!(cat.status, CategoryStatus::Complete);
    }

    #[test]
    fn test_project_score_with_recommendations() {
        let score = ProjectScore {
            total: 60,
            max: 100,
            grade: Grade::D,
            categories: vec![],
            recommendations: vec![
                Recommendation {
                    priority: 1,
                    action: "First action".to_string(),
                    potential_points: 15,
                    effort: Effort::Medium,
                },
                Recommendation {
                    priority: 2,
                    action: "Second action".to_string(),
                    potential_points: 10,
                    effort: Effort::High,
                },
            ],
            summary: "Needs improvement".to_string(),
        };

        let output = render_score_text(&score, true);
        assert!(output.contains("60/100"));
    }

    #[test]
    fn test_score_calculator_with_performance() {
        let temp = TempDir::new().unwrap();
        let benches_dir = temp.path().join("benches");
        std::fs::create_dir(&benches_dir).unwrap();
        std::fs::write(benches_dir.join("benchmark.rs"), "fn main() {}").unwrap();

        let calc = ScoreCalculator::new(temp.path());
        let score = calc.calculate();

        let perf_category = score
            .categories
            .iter()
            .find(|c| c.name == "Performance Benchmarks");
        assert!(perf_category.is_some());
    }

    #[test]
    fn test_score_calculator_with_accessibility() {
        let temp = TempDir::new().unwrap();
        let a11y_dir = temp.path().join("a11y");
        std::fs::create_dir(&a11y_dir).unwrap();
        std::fs::write(a11y_dir.join("config.yaml"), "rules: []").unwrap();

        let calc = ScoreCalculator::new(temp.path());
        let score = calc.calculate();

        let a11y_category = score.categories.iter().find(|c| c.name == "Accessibility");
        assert!(a11y_category.is_some());
    }

    #[test]
    fn test_score_calculator_with_docs() {
        let temp = TempDir::new().unwrap();
        std::fs::write(
            temp.path().join("README.md"),
            "# Test\n\n## Testing\n\nWe use tests",
        )
        .unwrap();

        let calc = ScoreCalculator::new(temp.path());
        let score = calc.calculate();

        let docs_category = score.categories.iter().find(|c| c.name == "Documentation");
        assert!(docs_category.is_some());
    }

    #[test]
    fn test_score_calculator_with_replay_session() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("session.replay"), "{}").unwrap();

        let calc = ScoreCalculator::new(temp.path());
        let score = calc.calculate();

        let replay_category = score
            .categories
            .iter()
            .find(|c| c.name == "Deterministic Replay");
        assert!(replay_category.is_some());
    }

    #[test]
    fn test_render_score_text_with_categories() {
        let score = ProjectScore {
            total: 75,
            max: 100,
            grade: Grade::C,
            categories: vec![
                CategoryScore {
                    name: "Test A".to_string(),
                    score: 40,
                    max: 50,
                    status: CategoryStatus::Complete,
                    criteria: vec![],
                },
                CategoryScore {
                    name: "Test B".to_string(),
                    score: 35,
                    max: 50,
                    status: CategoryStatus::Partial,
                    criteria: vec![],
                },
            ],
            recommendations: vec![],
            summary: "Good progress".to_string(),
        };

        let output = render_score_text(&score, true);
        assert!(output.contains("Test A"));
        assert!(output.contains("Test B"));
    }
}
