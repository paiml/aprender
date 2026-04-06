#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::doc_markdown,
    clippy::missing_const_for_fn
)]
//! Grade scoring system for quality evaluation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Quality grade levels (A+ through F) per spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Grade {
    /// Failing (0-49)
    #[default]
    F,
    /// Incomplete (50-54)
    D,
    /// Sketch (55-59)
    CMinus,
    /// Draft (60-64)
    C,
    /// Prototype (65-69)
    CPlus,
    /// Development (70-74)
    BMinus,
    /// Alpha Quality (75-79)
    B,
    /// Beta Quality (80-84)
    BPlus,
    /// Release Candidate (85-89)
    AMinus,
    /// Production Ready (90-94)
    A,
    /// Production Excellence (95-100)
    APlus,
}

impl PartialOrd for Grade {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Grade {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare by minimum percentage (A > B > C > D > F)
        self.min_percentage()
            .partial_cmp(&other.min_percentage())
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl Grade {
    /// Create a grade from a percentage score (0-100 scale).
    #[must_use]
    pub fn from_percentage(percent: f32) -> Self {
        match percent {
            p if p >= 95.0 => Self::APlus,
            p if p >= 90.0 => Self::A,
            p if p >= 85.0 => Self::AMinus,
            p if p >= 80.0 => Self::BPlus,
            p if p >= 75.0 => Self::B,
            p if p >= 70.0 => Self::BMinus,
            p if p >= 65.0 => Self::CPlus,
            p if p >= 60.0 => Self::C,
            p if p >= 55.0 => Self::CMinus,
            p if p >= 50.0 => Self::D,
            _ => Self::F,
        }
    }

    /// Get the minimum percentage for this grade.
    #[must_use]
    pub const fn min_percentage(&self) -> f32 {
        match self {
            Self::APlus => 95.0,
            Self::A => 90.0,
            Self::AMinus => 85.0,
            Self::BPlus => 80.0,
            Self::B => 75.0,
            Self::BMinus => 70.0,
            Self::CPlus => 65.0,
            Self::C => 60.0,
            Self::CMinus => 55.0,
            Self::D => 50.0,
            Self::F => 0.0,
        }
    }

    /// Get grade as a letter string.
    #[must_use]
    pub const fn letter(&self) -> &'static str {
        match self {
            Self::APlus => "A+",
            Self::A => "A",
            Self::AMinus => "A-",
            Self::BPlus => "B+",
            Self::B => "B",
            Self::BMinus => "B-",
            Self::CPlus => "C+",
            Self::C => "C",
            Self::CMinus => "C-",
            Self::D => "D",
            Self::F => "F",
        }
    }

    /// Check if this is a passing grade (C or better = 60+).
    #[must_use]
    pub const fn is_passing(&self) -> bool {
        matches!(
            self,
            Self::APlus
                | Self::A
                | Self::AMinus
                | Self::BPlus
                | Self::B
                | Self::BMinus
                | Self::CPlus
                | Self::C
        )
    }

    /// Check if this grade is production ready (B+ or better = 80+).
    #[must_use]
    pub const fn is_production_ready(&self) -> bool {
        matches!(self, Self::APlus | Self::A | Self::AMinus | Self::BPlus)
    }
}

impl std::fmt::Display for Grade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.letter())
    }
}

/// A scored criterion with name, weight, and result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Criterion {
    /// Name of the criterion
    pub name: String,
    /// Description of what is being measured
    pub description: String,
    /// Weight (importance) of this criterion (0.0 - 1.0)
    pub weight: f32,
    /// Score achieved (0.0 - 100.0)
    pub score: f32,
    /// Whether this criterion passed
    pub passed: bool,
    /// Detailed feedback
    pub feedback: Option<String>,
}

impl Criterion {
    /// Create a new criterion.
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            weight: 1.0,
            score: 0.0,
            passed: false,
            feedback: None,
        }
    }

    /// Set the weight.
    #[must_use]
    pub fn weight(mut self, weight: f32) -> Self {
        self.weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set the score.
    #[must_use]
    pub fn score(mut self, score: f32) -> Self {
        self.score = score.clamp(0.0, 100.0);
        self.passed = self.score >= 60.0;
        self
    }

    /// Mark as passed with a perfect score.
    #[must_use]
    pub const fn pass(mut self) -> Self {
        self.score = 100.0;
        self.passed = true;
        self
    }

    /// Mark as failed with zero score.
    #[must_use]
    pub const fn fail(mut self) -> Self {
        self.score = 0.0;
        self.passed = false;
        self
    }

    /// Add feedback.
    #[must_use]
    pub fn feedback(mut self, feedback: impl Into<String>) -> Self {
        self.feedback = Some(feedback.into());
        self
    }

    /// Get the grade for this criterion.
    #[must_use]
    pub fn grade(&self) -> Grade {
        Grade::from_percentage(self.score)
    }

    /// Get weighted score (score * weight).
    #[must_use]
    pub fn weighted_score(&self) -> f32 {
        self.score * self.weight
    }
}

/// A report card containing multiple criteria scores.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReportCard {
    /// Title of the evaluation
    pub title: String,
    /// Individual criteria scores
    pub criteria: Vec<Criterion>,
    /// Category scores (aggregated)
    pub categories: HashMap<String, f32>,
}

impl ReportCard {
    /// Create a new report card.
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            criteria: Vec::new(),
            categories: HashMap::new(),
        }
    }

    /// Add a criterion.
    pub fn add_criterion(&mut self, criterion: Criterion) {
        self.criteria.push(criterion);
    }

    /// Add criterion with builder pattern.
    #[must_use]
    pub fn criterion(mut self, criterion: Criterion) -> Self {
        self.criteria.push(criterion);
        self
    }

    /// Add a category score.
    pub fn add_category(&mut self, name: impl Into<String>, score: f32) {
        self.categories.insert(name.into(), score.clamp(0.0, 100.0));
    }

    /// Calculate the overall weighted score.
    #[must_use]
    pub fn overall_score(&self) -> f32 {
        if self.criteria.is_empty() {
            return 0.0;
        }

        let total_weight: f32 = self.criteria.iter().map(|c| c.weight).sum();
        if total_weight == 0.0 {
            return 0.0;
        }

        let weighted_sum: f32 = self.criteria.iter().map(Criterion::weighted_score).sum();
        weighted_sum / total_weight
    }

    /// Get the overall grade.
    #[must_use]
    pub fn overall_grade(&self) -> Grade {
        Grade::from_percentage(self.overall_score())
    }

    /// Check if all criteria passed.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.criteria.iter().all(|c| c.passed)
    }

    /// Count passed criteria.
    #[must_use]
    pub fn passed_count(&self) -> usize {
        self.criteria.iter().filter(|c| c.passed).count()
    }

    /// Count failed criteria.
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.criteria.iter().filter(|c| !c.passed).count()
    }

    /// Get all failing criteria.
    #[must_use]
    pub fn failures(&self) -> Vec<&Criterion> {
        self.criteria.iter().filter(|c| !c.passed).collect()
    }

    /// Check if the overall grade is passing.
    #[must_use]
    pub fn is_passing(&self) -> bool {
        self.overall_grade().is_passing()
    }
}

/// Standard evaluation categories.
pub mod categories {
    /// Accessibility evaluation.
    pub const ACCESSIBILITY: &str = "accessibility";
    /// Performance evaluation.
    pub const PERFORMANCE: &str = "performance";
    /// Visual consistency.
    pub const VISUAL: &str = "visual";
    /// Code quality.
    pub const CODE_QUALITY: &str = "code_quality";
    /// Test coverage.
    pub const TESTING: &str = "testing";
    /// Documentation.
    pub const DOCUMENTATION: &str = "documentation";
    /// Security.
    pub const SECURITY: &str = "security";
}

// =============================================================================
// AppQualityScore - Per Spec Section 5.1
// =============================================================================

/// App Quality Score breakdown per spec (6 orthogonal metrics).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    // Structural (25 points)
    /// Cyclomatic complexity (McCabe, 1976)
    pub widget_complexity: f64,
    /// Nesting depth penalty
    pub layout_depth: f64,
    /// Widget count vs viewport
    pub component_count: f64,

    // Performance (20 points)
    /// 95th percentile frame time
    pub render_time_p95: f64,
    /// Peak memory vs baseline
    pub memory_usage: f64,
    /// WASM binary size
    pub bundle_size: f64,

    // Accessibility (20 points)
    /// WCAG 2.1 AA checklist
    pub wcag_aa_compliance: f64,
    /// Full keyboard support
    pub keyboard_navigation: f64,
    /// ARIA labels coverage
    pub screen_reader: f64,

    // Data Quality (15 points)
    /// Missing value ratio
    pub data_completeness: f64,
    /// Staleness penalty
    pub data_freshness: f64,
    /// Type errors
    pub schema_validation: f64,

    // Documentation (10 points)
    /// Required fields coverage
    pub manifest_completeness: f64,
    /// Model/data cards present
    pub card_coverage: f64,

    // Consistency (10 points)
    /// Design system compliance
    pub theme_adherence: f64,
    /// ID/class naming
    pub naming_conventions: f64,
}

impl ScoreBreakdown {
    /// Calculate structural score (contributes 25% to total).
    #[must_use]
    pub fn structural_score(&self) -> f64 {
        (self.widget_complexity + self.layout_depth + self.component_count) / 3.0 * 0.25
    }

    /// Calculate performance score (contributes 20% to total).
    #[must_use]
    pub fn performance_score(&self) -> f64 {
        (self.render_time_p95 + self.memory_usage + self.bundle_size) / 3.0 * 0.20
    }

    /// Calculate accessibility score (contributes 20% to total).
    #[must_use]
    pub fn accessibility_score(&self) -> f64 {
        (self.wcag_aa_compliance + self.keyboard_navigation + self.screen_reader) / 3.0 * 0.20
    }

    /// Calculate data quality score (contributes 15% to total).
    #[must_use]
    pub fn data_quality_score(&self) -> f64 {
        (self.data_completeness + self.data_freshness + self.schema_validation) / 3.0 * 0.15
    }

    /// Calculate documentation score (contributes 10% to total).
    #[must_use]
    pub fn documentation_score(&self) -> f64 {
        (self.manifest_completeness + self.card_coverage) / 2.0 * 0.10
    }

    /// Calculate consistency score (contributes 10% to total).
    #[must_use]
    pub fn consistency_score(&self) -> f64 {
        (self.theme_adherence + self.naming_conventions) / 2.0 * 0.10
    }

    /// Calculate total score (0-100).
    #[must_use]
    pub fn total(&self) -> f64 {
        self.structural_score()
            + self.performance_score()
            + self.accessibility_score()
            + self.data_quality_score()
            + self.documentation_score()
            + self.consistency_score()
    }
}

/// App Quality Score (0-100, F-A+).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AppQualityScore {
    /// Overall score (0-100)
    pub overall: f64,
    /// Grade (F through A+)
    pub grade: Grade,
    /// Detailed breakdown
    pub breakdown: ScoreBreakdown,
}

impl AppQualityScore {
    /// Create from a score breakdown.
    #[must_use]
    pub fn from_breakdown(breakdown: ScoreBreakdown) -> Self {
        let overall = breakdown.total();
        let grade = Grade::from_percentage(overall as f32);
        Self {
            overall,
            grade,
            breakdown,
        }
    }

    /// Check if score meets minimum grade requirement.
    #[must_use]
    pub fn meets_minimum(&self, min_grade: Grade) -> bool {
        self.grade >= min_grade
    }

    /// Check if production ready (B+ or better).
    #[must_use]
    pub fn is_production_ready(&self) -> bool {
        self.grade.is_production_ready()
    }
}

/// Builder for constructing AppQualityScore.
#[derive(Debug, Clone, Default)]
pub struct QualityScoreBuilder {
    breakdown: ScoreBreakdown,
}

impl QualityScoreBuilder {
    /// Create a new quality score builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set structural metrics.
    #[must_use]
    pub fn structural(mut self, complexity: f64, depth: f64, count: f64) -> Self {
        self.breakdown.widget_complexity = complexity.clamp(0.0, 100.0);
        self.breakdown.layout_depth = depth.clamp(0.0, 100.0);
        self.breakdown.component_count = count.clamp(0.0, 100.0);
        self
    }

    /// Set performance metrics.
    #[must_use]
    pub fn performance(mut self, render_time: f64, memory: f64, bundle: f64) -> Self {
        self.breakdown.render_time_p95 = render_time.clamp(0.0, 100.0);
        self.breakdown.memory_usage = memory.clamp(0.0, 100.0);
        self.breakdown.bundle_size = bundle.clamp(0.0, 100.0);
        self
    }

    /// Set accessibility metrics.
    #[must_use]
    pub fn accessibility(mut self, wcag: f64, keyboard: f64, screen_reader: f64) -> Self {
        self.breakdown.wcag_aa_compliance = wcag.clamp(0.0, 100.0);
        self.breakdown.keyboard_navigation = keyboard.clamp(0.0, 100.0);
        self.breakdown.screen_reader = screen_reader.clamp(0.0, 100.0);
        self
    }

    /// Set data quality metrics.
    #[must_use]
    pub fn data_quality(mut self, completeness: f64, freshness: f64, schema: f64) -> Self {
        self.breakdown.data_completeness = completeness.clamp(0.0, 100.0);
        self.breakdown.data_freshness = freshness.clamp(0.0, 100.0);
        self.breakdown.schema_validation = schema.clamp(0.0, 100.0);
        self
    }

    /// Set documentation metrics.
    #[must_use]
    pub fn documentation(mut self, manifest: f64, cards: f64) -> Self {
        self.breakdown.manifest_completeness = manifest.clamp(0.0, 100.0);
        self.breakdown.card_coverage = cards.clamp(0.0, 100.0);
        self
    }

    /// Set consistency metrics.
    #[must_use]
    pub fn consistency(mut self, theme: f64, naming: f64) -> Self {
        self.breakdown.theme_adherence = theme.clamp(0.0, 100.0);
        self.breakdown.naming_conventions = naming.clamp(0.0, 100.0);
        self
    }

    /// Build the final quality score.
    #[must_use]
    pub fn build(self) -> AppQualityScore {
        AppQualityScore::from_breakdown(self.breakdown)
    }
}

// =============================================================================
// QualityGates - Per Spec Section 5.3
// =============================================================================

/// Quality gate configuration (from .presentar-gates.toml).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityGates {
    /// Minimum required grade
    pub min_grade: Grade,
    /// Minimum required score (0-100)
    pub min_score: f64,
    /// Performance requirements
    pub performance: PerformanceGates,
    /// Accessibility requirements
    pub accessibility: AccessibilityGates,
    /// Data requirements
    pub data: DataGates,
    /// Documentation requirements
    pub documentation: DocumentationGates,
}

/// Performance quality gates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceGates {
    /// Maximum render time in milliseconds (60fps = 16ms)
    pub max_render_time_ms: u32,
    /// Maximum bundle size in KB
    pub max_bundle_size_kb: u32,
    /// Maximum memory usage in MB
    pub max_memory_mb: u32,
}

/// Accessibility quality gates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AccessibilityGates {
    /// WCAG level ("A", "AA", "AAA")
    pub wcag_level: String,
    /// Minimum contrast ratio
    pub min_contrast_ratio: f32,
    /// Require full keyboard navigation
    pub require_keyboard_nav: bool,
    /// Require ARIA labels
    pub require_aria_labels: bool,
}

/// Data quality gates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataGates {
    /// Maximum staleness in minutes
    pub max_staleness_minutes: u32,
    /// Require schema validation
    pub require_schema_validation: bool,
}

/// Documentation quality gates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentationGates {
    /// Require model cards
    pub require_model_cards: bool,
    /// Require data cards
    pub require_data_cards: bool,
    /// Minimum manifest fields
    pub min_manifest_fields: Vec<String>,
}

impl Default for QualityGates {
    fn default() -> Self {
        Self {
            min_grade: Grade::BPlus,
            min_score: 80.0,
            performance: PerformanceGates::default(),
            accessibility: AccessibilityGates::default(),
            data: DataGates::default(),
            documentation: DocumentationGates::default(),
        }
    }
}

impl Default for PerformanceGates {
    fn default() -> Self {
        Self {
            max_render_time_ms: 16,
            max_bundle_size_kb: 500,
            max_memory_mb: 100,
        }
    }
}

impl Default for AccessibilityGates {
    fn default() -> Self {
        Self {
            wcag_level: "AA".to_string(),
            min_contrast_ratio: 4.5,
            require_keyboard_nav: true,
            require_aria_labels: true,
        }
    }
}

impl Default for DataGates {
    fn default() -> Self {
        Self {
            max_staleness_minutes: 60,
            require_schema_validation: true,
        }
    }
}

impl Default for DocumentationGates {
    fn default() -> Self {
        Self {
            require_model_cards: true,
            require_data_cards: true,
            min_manifest_fields: vec![
                "name".to_string(),
                "version".to_string(),
                "description".to_string(),
            ],
        }
    }
}

/// Result of checking quality gates.
#[derive(Debug, Clone)]
pub struct GateCheckResult {
    /// Whether all gates passed
    pub passed: bool,
    /// List of gate violations
    pub violations: Vec<GateViolation>,
}

/// A single gate violation.
#[derive(Debug, Clone)]
pub struct GateViolation {
    /// Gate name
    pub gate: String,
    /// Expected value
    pub expected: String,
    /// Actual value
    pub actual: String,
    /// Severity
    pub severity: ViolationSeverity,
}

/// Severity of a gate violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationSeverity {
    /// Warning - gate not met but not blocking
    Warning,
    /// Error - gate not met and blocking
    Error,
}

impl QualityGates {
    /// Check if a quality score passes all gates.
    #[must_use]
    pub fn check(&self, score: &AppQualityScore) -> GateCheckResult {
        let mut violations = Vec::new();

        // Check minimum grade
        if score.grade < self.min_grade {
            violations.push(GateViolation {
                gate: "min_grade".to_string(),
                expected: self.min_grade.letter().to_string(),
                actual: score.grade.letter().to_string(),
                severity: ViolationSeverity::Error,
            });
        }

        // Check minimum score
        if score.overall < self.min_score {
            violations.push(GateViolation {
                gate: "min_score".to_string(),
                expected: format!("{:.1}", self.min_score),
                actual: format!("{:.1}", score.overall),
                severity: ViolationSeverity::Error,
            });
        }

        GateCheckResult {
            passed: violations.is_empty(),
            violations,
        }
    }
}

/// Builder for creating standard evaluation criteria.
#[derive(Debug, Clone, Default)]
pub struct EvaluationBuilder {
    report: ReportCard,
}

impl EvaluationBuilder {
    /// Create a new evaluation builder.
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            report: ReportCard::new(title),
        }
    }

    /// Add accessibility criterion.
    #[must_use]
    pub fn accessibility(mut self, score: f32, feedback: Option<&str>) -> Self {
        let mut criterion = Criterion::new(
            "Accessibility",
            "WCAG 2.1 AA compliance and screen reader support",
        )
        .weight(1.0)
        .score(score);

        if let Some(fb) = feedback {
            criterion = criterion.feedback(fb);
        }

        self.report.add_criterion(criterion);
        self.report
            .add_category(categories::ACCESSIBILITY.to_string(), score);
        self
    }

    /// Add performance criterion.
    #[must_use]
    pub fn performance(mut self, score: f32, feedback: Option<&str>) -> Self {
        let mut criterion = Criterion::new(
            "Performance",
            "Frame rate, memory usage, and responsiveness",
        )
        .weight(1.0)
        .score(score);

        if let Some(fb) = feedback {
            criterion = criterion.feedback(fb);
        }

        self.report.add_criterion(criterion);
        self.report
            .add_category(categories::PERFORMANCE.to_string(), score);
        self
    }

    /// Add visual criterion.
    #[must_use]
    pub fn visual(mut self, score: f32, feedback: Option<&str>) -> Self {
        let mut criterion =
            Criterion::new("Visual Consistency", "Theme adherence and visual polish")
                .weight(0.8)
                .score(score);

        if let Some(fb) = feedback {
            criterion = criterion.feedback(fb);
        }

        self.report.add_criterion(criterion);
        self.report
            .add_category(categories::VISUAL.to_string(), score);
        self
    }

    /// Add code quality criterion.
    #[must_use]
    pub fn code_quality(mut self, score: f32, feedback: Option<&str>) -> Self {
        let mut criterion = Criterion::new(
            "Code Quality",
            "Lint compliance, documentation, and maintainability",
        )
        .weight(0.8)
        .score(score);

        if let Some(fb) = feedback {
            criterion = criterion.feedback(fb);
        }

        self.report.add_criterion(criterion);
        self.report
            .add_category(categories::CODE_QUALITY.to_string(), score);
        self
    }

    /// Add testing criterion.
    #[must_use]
    pub fn testing(mut self, score: f32, feedback: Option<&str>) -> Self {
        let mut criterion = Criterion::new("Testing", "Test coverage and mutation testing score")
            .weight(1.0)
            .score(score);

        if let Some(fb) = feedback {
            criterion = criterion.feedback(fb);
        }

        self.report.add_criterion(criterion);
        self.report
            .add_category(categories::TESTING.to_string(), score);
        self
    }

    /// Add custom criterion.
    #[must_use]
    pub fn custom(mut self, criterion: Criterion) -> Self {
        self.report.add_criterion(criterion);
        self
    }

    /// Build the final report card.
    #[must_use]
    pub fn build(self) -> ReportCard {
        self.report
    }
}

// =============================================================================
// TOML Configuration Loading
// =============================================================================

/// Error type for quality gate configuration.
#[derive(Debug, Clone, PartialEq)]
pub enum GateConfigError {
    /// Failed to parse TOML
    ParseError(String),
    /// Failed to read file
    IoError(String),
    /// Invalid configuration value
    InvalidValue(String),
}

impl std::fmt::Display for GateConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseError(msg) => write!(f, "parse error: {msg}"),
            Self::IoError(msg) => write!(f, "IO error: {msg}"),
            Self::InvalidValue(msg) => write!(f, "invalid value: {msg}"),
        }
    }
}

impl std::error::Error for GateConfigError {}

impl QualityGates {
    /// Default config file name.
    pub const CONFIG_FILE: &'static str = ".presentar-gates.toml";

    /// Parse quality gates from a TOML string.
    ///
    /// # Errors
    ///
    /// Returns error if TOML is invalid or values are out of range.
    pub fn from_toml(toml_str: &str) -> Result<Self, GateConfigError> {
        toml::from_str(toml_str).map_err(|e| GateConfigError::ParseError(e.to_string()))
    }

    /// Serialize quality gates to a TOML string.
    #[must_use]
    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).unwrap_or_default()
    }

    /// Load quality gates from a file.
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or parsed.
    pub fn load_from_file(path: &std::path::Path) -> Result<Self, GateConfigError> {
        let contents =
            std::fs::read_to_string(path).map_err(|e| GateConfigError::IoError(e.to_string()))?;
        Self::from_toml(&contents)
    }

    /// Save quality gates to a file.
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be written.
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<(), GateConfigError> {
        let contents = self.to_toml();
        std::fs::write(path, contents).map_err(|e| GateConfigError::IoError(e.to_string()))
    }

    /// Load from default config file in current directory.
    ///
    /// Returns default config if file doesn't exist.
    #[must_use]
    pub fn load_default() -> Self {
        let path = std::path::Path::new(Self::CONFIG_FILE);
        Self::load_from_file(path).unwrap_or_default()
    }

    /// Check a score with extended validation (performance, bundle size, etc.).
    #[must_use]
    pub fn check_extended(
        &self,
        score: &AppQualityScore,
        render_time_ms: Option<u32>,
        bundle_size_kb: Option<u32>,
        memory_mb: Option<u32>,
    ) -> GateCheckResult {
        let mut result = self.check(score);

        // Check performance gates
        if let Some(render_time) = render_time_ms {
            if render_time > self.performance.max_render_time_ms {
                result.violations.push(GateViolation {
                    gate: "max_render_time_ms".to_string(),
                    expected: format!("<= {}ms", self.performance.max_render_time_ms),
                    actual: format!("{}ms", render_time),
                    severity: ViolationSeverity::Error,
                });
            }
        }

        if let Some(bundle) = bundle_size_kb {
            if bundle > self.performance.max_bundle_size_kb {
                result.violations.push(GateViolation {
                    gate: "max_bundle_size_kb".to_string(),
                    expected: format!("<= {}KB", self.performance.max_bundle_size_kb),
                    actual: format!("{}KB", bundle),
                    severity: ViolationSeverity::Error,
                });
            }
        }

        if let Some(memory) = memory_mb {
            if memory > self.performance.max_memory_mb {
                result.violations.push(GateViolation {
                    gate: "max_memory_mb".to_string(),
                    expected: format!("<= {}MB", self.performance.max_memory_mb),
                    actual: format!("{}MB", memory),
                    severity: ViolationSeverity::Warning,
                });
            }
        }

        result.passed = result
            .violations
            .iter()
            .all(|v| v.severity != ViolationSeverity::Error);
        result
    }

    /// Generate a sample TOML config file content.
    #[must_use]
    pub fn sample_config() -> String {
        r#"# Presentar Quality Gates Configuration
# Place this file at .presentar-gates.toml in your project root

# Minimum required grade (F, D, C, C+, B-, B, B+, A-, A, A+)
min_grade = "B+"

# Minimum required score (0-100)
min_score = 80.0

[performance]
# Maximum render time in milliseconds (60fps = 16ms)
max_render_time_ms = 16

# Maximum bundle size in KB
max_bundle_size_kb = 500

# Maximum memory usage in MB
max_memory_mb = 100

[accessibility]
# WCAG level: "A", "AA", or "AAA"
wcag_level = "AA"

# Minimum contrast ratio
min_contrast_ratio = 4.5

# Require full keyboard navigation
require_keyboard_nav = true

# Require ARIA labels
require_aria_labels = true

[data]
# Maximum data staleness in minutes
max_staleness_minutes = 60

# Require schema validation
require_schema_validation = true

[documentation]
# Require model cards for ML models
require_model_cards = true

# Require data cards for datasets
require_data_cards = true

# Minimum required manifest fields
min_manifest_fields = ["name", "version", "description"]
"#
        .to_string()
    }
}

impl Grade {
    /// Parse grade from string (e.g., "A+", "B-", "C").
    ///
    /// # Errors
    ///
    /// Returns error if string is not a valid grade.
    pub fn from_str(s: &str) -> Result<Self, GateConfigError> {
        match s.trim().to_uppercase().as_str() {
            "A+" => Ok(Self::APlus),
            "A" => Ok(Self::A),
            "A-" => Ok(Self::AMinus),
            "B+" => Ok(Self::BPlus),
            "B" => Ok(Self::B),
            "B-" => Ok(Self::BMinus),
            "C+" => Ok(Self::CPlus),
            "C" => Ok(Self::C),
            "C-" => Ok(Self::CMinus),
            "D" => Ok(Self::D),
            "F" => Ok(Self::F),
            _ => Err(GateConfigError::InvalidValue(format!(
                "Invalid grade: {s}. Valid values: A+, A, A-, B+, B, B-, C+, C, C-, D, F"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Grade Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_grade_from_percentage() {
        assert_eq!(Grade::from_percentage(100.0), Grade::APlus);
        assert_eq!(Grade::from_percentage(95.0), Grade::APlus);
        assert_eq!(Grade::from_percentage(92.0), Grade::A);
        assert_eq!(Grade::from_percentage(90.0), Grade::A);
        assert_eq!(Grade::from_percentage(87.0), Grade::AMinus);
        assert_eq!(Grade::from_percentage(85.0), Grade::AMinus);
        assert_eq!(Grade::from_percentage(82.0), Grade::BPlus);
        assert_eq!(Grade::from_percentage(80.0), Grade::BPlus);
        assert_eq!(Grade::from_percentage(77.0), Grade::B);
        assert_eq!(Grade::from_percentage(75.0), Grade::B);
        assert_eq!(Grade::from_percentage(72.0), Grade::BMinus);
        assert_eq!(Grade::from_percentage(70.0), Grade::BMinus);
        assert_eq!(Grade::from_percentage(67.0), Grade::CPlus);
        assert_eq!(Grade::from_percentage(65.0), Grade::CPlus);
        assert_eq!(Grade::from_percentage(62.0), Grade::C);
        assert_eq!(Grade::from_percentage(60.0), Grade::C);
        assert_eq!(Grade::from_percentage(57.0), Grade::CMinus);
        assert_eq!(Grade::from_percentage(55.0), Grade::CMinus);
        assert_eq!(Grade::from_percentage(52.0), Grade::D);
        assert_eq!(Grade::from_percentage(50.0), Grade::D);
        assert_eq!(Grade::from_percentage(49.0), Grade::F);
        assert_eq!(Grade::from_percentage(0.0), Grade::F);
    }

    #[test]
    fn test_grade_min_percentage() {
        assert_eq!(Grade::APlus.min_percentage(), 95.0);
        assert_eq!(Grade::A.min_percentage(), 90.0);
        assert_eq!(Grade::AMinus.min_percentage(), 85.0);
        assert_eq!(Grade::BPlus.min_percentage(), 80.0);
        assert_eq!(Grade::B.min_percentage(), 75.0);
        assert_eq!(Grade::BMinus.min_percentage(), 70.0);
        assert_eq!(Grade::CPlus.min_percentage(), 65.0);
        assert_eq!(Grade::C.min_percentage(), 60.0);
        assert_eq!(Grade::CMinus.min_percentage(), 55.0);
        assert_eq!(Grade::D.min_percentage(), 50.0);
        assert_eq!(Grade::F.min_percentage(), 0.0);
    }

    #[test]
    fn test_grade_letter() {
        assert_eq!(Grade::APlus.letter(), "A+");
        assert_eq!(Grade::A.letter(), "A");
        assert_eq!(Grade::AMinus.letter(), "A-");
        assert_eq!(Grade::BPlus.letter(), "B+");
        assert_eq!(Grade::B.letter(), "B");
        assert_eq!(Grade::BMinus.letter(), "B-");
        assert_eq!(Grade::CPlus.letter(), "C+");
        assert_eq!(Grade::C.letter(), "C");
        assert_eq!(Grade::CMinus.letter(), "C-");
        assert_eq!(Grade::D.letter(), "D");
        assert_eq!(Grade::F.letter(), "F");
    }

    #[test]
    fn test_grade_is_passing() {
        assert!(Grade::APlus.is_passing());
        assert!(Grade::A.is_passing());
        assert!(Grade::AMinus.is_passing());
        assert!(Grade::BPlus.is_passing());
        assert!(Grade::B.is_passing());
        assert!(Grade::BMinus.is_passing());
        assert!(Grade::CPlus.is_passing());
        assert!(Grade::C.is_passing());
        assert!(!Grade::CMinus.is_passing());
        assert!(!Grade::D.is_passing());
        assert!(!Grade::F.is_passing());
    }

    #[test]
    fn test_grade_is_production_ready() {
        assert!(Grade::APlus.is_production_ready());
        assert!(Grade::A.is_production_ready());
        assert!(Grade::AMinus.is_production_ready());
        assert!(Grade::BPlus.is_production_ready());
        assert!(!Grade::B.is_production_ready());
        assert!(!Grade::BMinus.is_production_ready());
        assert!(!Grade::F.is_production_ready());
    }

    #[test]
    fn test_grade_default() {
        assert_eq!(Grade::default(), Grade::F);
    }

    #[test]
    fn test_grade_display() {
        assert_eq!(format!("{}", Grade::APlus), "A+");
        assert_eq!(format!("{}", Grade::A), "A");
        assert_eq!(format!("{}", Grade::F), "F");
    }

    #[test]
    fn test_grade_ordering() {
        assert!(Grade::APlus > Grade::A);
        assert!(Grade::A > Grade::AMinus);
        assert!(Grade::AMinus > Grade::BPlus);
        assert!(Grade::BPlus > Grade::B);
        assert!(Grade::B > Grade::BMinus);
        assert!(Grade::BMinus > Grade::CPlus);
        assert!(Grade::CPlus > Grade::C);
        assert!(Grade::C > Grade::CMinus);
        assert!(Grade::CMinus > Grade::D);
        assert!(Grade::D > Grade::F);
    }

    // =========================================================================
    // Criterion Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_criterion_new() {
        let c = Criterion::new("Test", "Description");
        assert_eq!(c.name, "Test");
        assert_eq!(c.description, "Description");
        assert_eq!(c.weight, 1.0);
        assert_eq!(c.score, 0.0);
        assert!(!c.passed);
    }

    #[test]
    fn test_criterion_weight() {
        let c = Criterion::new("Test", "Desc").weight(0.5);
        assert_eq!(c.weight, 0.5);
    }

    #[test]
    fn test_criterion_weight_clamped() {
        let c1 = Criterion::new("Test", "Desc").weight(2.0);
        assert_eq!(c1.weight, 1.0);

        let c2 = Criterion::new("Test", "Desc").weight(-1.0);
        assert_eq!(c2.weight, 0.0);
    }

    #[test]
    fn test_criterion_score() {
        let c = Criterion::new("Test", "Desc").score(85.0);
        assert_eq!(c.score, 85.0);
        assert!(c.passed);
    }

    #[test]
    fn test_criterion_score_failing() {
        let c = Criterion::new("Test", "Desc").score(50.0);
        assert_eq!(c.score, 50.0);
        assert!(!c.passed);
    }

    #[test]
    fn test_criterion_score_clamped() {
        let c1 = Criterion::new("Test", "Desc").score(150.0);
        assert_eq!(c1.score, 100.0);

        let c2 = Criterion::new("Test", "Desc").score(-10.0);
        assert_eq!(c2.score, 0.0);
    }

    #[test]
    fn test_criterion_pass() {
        let c = Criterion::new("Test", "Desc").pass();
        assert_eq!(c.score, 100.0);
        assert!(c.passed);
    }

    #[test]
    fn test_criterion_fail() {
        let c = Criterion::new("Test", "Desc").fail();
        assert_eq!(c.score, 0.0);
        assert!(!c.passed);
    }

    #[test]
    fn test_criterion_feedback() {
        let c = Criterion::new("Test", "Desc").feedback("Good work!");
        assert_eq!(c.feedback, Some("Good work!".to_string()));
    }

    #[test]
    fn test_criterion_grade() {
        assert_eq!(Criterion::new("T", "D").score(95.0).grade(), Grade::APlus);
        assert_eq!(Criterion::new("T", "D").score(90.0).grade(), Grade::A);
        assert_eq!(Criterion::new("T", "D").score(85.0).grade(), Grade::AMinus);
        assert_eq!(Criterion::new("T", "D").score(80.0).grade(), Grade::BPlus);
        assert_eq!(Criterion::new("T", "D").score(75.0).grade(), Grade::B);
        assert_eq!(Criterion::new("T", "D").score(70.0).grade(), Grade::BMinus);
        assert_eq!(Criterion::new("T", "D").score(65.0).grade(), Grade::CPlus);
        assert_eq!(Criterion::new("T", "D").score(60.0).grade(), Grade::C);
        assert_eq!(Criterion::new("T", "D").score(55.0).grade(), Grade::CMinus);
        assert_eq!(Criterion::new("T", "D").score(50.0).grade(), Grade::D);
        assert_eq!(Criterion::new("T", "D").score(40.0).grade(), Grade::F);
    }

    #[test]
    fn test_criterion_weighted_score() {
        let c = Criterion::new("Test", "Desc").weight(0.5).score(80.0);
        assert_eq!(c.weighted_score(), 40.0);
    }

    // =========================================================================
    // ReportCard Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_report_card_new() {
        let report = ReportCard::new("My Report");
        assert_eq!(report.title, "My Report");
        assert!(report.criteria.is_empty());
    }

    #[test]
    fn test_report_card_add_criterion() {
        let mut report = ReportCard::new("Test");
        report.add_criterion(Criterion::new("C1", "D1").score(90.0));
        assert_eq!(report.criteria.len(), 1);
    }

    #[test]
    fn test_report_card_builder() {
        let report = ReportCard::new("Test")
            .criterion(Criterion::new("C1", "D1").score(90.0))
            .criterion(Criterion::new("C2", "D2").score(80.0));
        assert_eq!(report.criteria.len(), 2);
    }

    #[test]
    fn test_report_card_overall_score_empty() {
        let report = ReportCard::new("Test");
        assert_eq!(report.overall_score(), 0.0);
    }

    #[test]
    fn test_report_card_overall_score_equal_weights() {
        let report = ReportCard::new("Test")
            .criterion(Criterion::new("C1", "D1").weight(1.0).score(100.0))
            .criterion(Criterion::new("C2", "D2").weight(1.0).score(80.0));
        assert_eq!(report.overall_score(), 90.0);
    }

    #[test]
    fn test_report_card_overall_score_different_weights() {
        let report = ReportCard::new("Test")
            .criterion(Criterion::new("C1", "D1").weight(0.75).score(100.0))
            .criterion(Criterion::new("C2", "D2").weight(0.25).score(80.0));
        // (100*0.75 + 80*0.25) / (0.75 + 0.25) = (75 + 20) / 1.0 = 95
        assert_eq!(report.overall_score(), 95.0);
    }

    #[test]
    fn test_report_card_overall_grade() {
        let report = ReportCard::new("Test").criterion(Criterion::new("C1", "D1").score(90.0));
        assert_eq!(report.overall_grade(), Grade::A);
    }

    #[test]
    fn test_report_card_all_passed() {
        let report = ReportCard::new("Test")
            .criterion(Criterion::new("C1", "D1").pass())
            .criterion(Criterion::new("C2", "D2").pass());
        assert!(report.all_passed());
    }

    #[test]
    fn test_report_card_not_all_passed() {
        let report = ReportCard::new("Test")
            .criterion(Criterion::new("C1", "D1").pass())
            .criterion(Criterion::new("C2", "D2").fail());
        assert!(!report.all_passed());
    }

    #[test]
    fn test_report_card_passed_count() {
        let report = ReportCard::new("Test")
            .criterion(Criterion::new("C1", "D1").pass())
            .criterion(Criterion::new("C2", "D2").pass())
            .criterion(Criterion::new("C3", "D3").fail());
        assert_eq!(report.passed_count(), 2);
        assert_eq!(report.failed_count(), 1);
    }

    #[test]
    fn test_report_card_failures() {
        let report = ReportCard::new("Test")
            .criterion(Criterion::new("C1", "D1").pass())
            .criterion(Criterion::new("C2", "D2").fail());
        let failures = report.failures();
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].name, "C2");
    }

    #[test]
    fn test_report_card_is_passing() {
        let passing = ReportCard::new("Test").criterion(Criterion::new("C1", "D1").score(90.0));
        assert!(passing.is_passing());

        let failing = ReportCard::new("Test").criterion(Criterion::new("C1", "D1").score(50.0));
        assert!(!failing.is_passing());
    }

    #[test]
    fn test_report_card_add_category() {
        let mut report = ReportCard::new("Test");
        report.add_category("performance", 95.0);
        assert_eq!(report.categories.get("performance"), Some(&95.0));
    }

    // =========================================================================
    // EvaluationBuilder Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_evaluation_builder_new() {
        let builder = EvaluationBuilder::new("My Eval");
        let report = builder.build();
        assert_eq!(report.title, "My Eval");
    }

    #[test]
    fn test_evaluation_builder_accessibility() {
        let report = EvaluationBuilder::new("Test")
            .accessibility(95.0, Some("Good a11y"))
            .build();

        assert_eq!(report.criteria.len(), 1);
        assert_eq!(report.criteria[0].name, "Accessibility");
        assert_eq!(report.criteria[0].score, 95.0);
        assert_eq!(
            report.categories.get(categories::ACCESSIBILITY),
            Some(&95.0)
        );
    }

    #[test]
    fn test_evaluation_builder_performance() {
        let report = EvaluationBuilder::new("Test")
            .performance(88.0, None)
            .build();

        assert_eq!(report.criteria[0].name, "Performance");
        assert_eq!(report.criteria[0].score, 88.0);
    }

    #[test]
    fn test_evaluation_builder_full() {
        let report = EvaluationBuilder::new("Full Evaluation")
            .accessibility(95.0, None)
            .performance(90.0, None)
            .visual(85.0, None)
            .code_quality(92.0, None)
            .testing(98.0, None)
            .build();

        assert_eq!(report.criteria.len(), 5);
        assert!(report.overall_score() > 90.0);
        assert_eq!(report.overall_grade(), Grade::A);
    }

    #[test]
    fn test_evaluation_builder_custom() {
        let report = EvaluationBuilder::new("Test")
            .custom(Criterion::new("Custom", "My custom criterion").score(75.0))
            .build();

        assert_eq!(report.criteria[0].name, "Custom");
    }

    // =========================================================================
    // AppQualityScore Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_score_breakdown_default() {
        let breakdown = ScoreBreakdown::default();
        assert_eq!(breakdown.total(), 0.0);
    }

    #[test]
    fn test_score_breakdown_perfect() {
        let breakdown = ScoreBreakdown {
            widget_complexity: 100.0,
            layout_depth: 100.0,
            component_count: 100.0,
            render_time_p95: 100.0,
            memory_usage: 100.0,
            bundle_size: 100.0,
            wcag_aa_compliance: 100.0,
            keyboard_navigation: 100.0,
            screen_reader: 100.0,
            data_completeness: 100.0,
            data_freshness: 100.0,
            schema_validation: 100.0,
            manifest_completeness: 100.0,
            card_coverage: 100.0,
            theme_adherence: 100.0,
            naming_conventions: 100.0,
        };

        assert!((breakdown.total() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_score_breakdown_category_scores() {
        let breakdown = ScoreBreakdown {
            widget_complexity: 90.0,
            layout_depth: 90.0,
            component_count: 90.0,
            render_time_p95: 80.0,
            memory_usage: 80.0,
            bundle_size: 80.0,
            wcag_aa_compliance: 100.0,
            keyboard_navigation: 100.0,
            screen_reader: 100.0,
            data_completeness: 70.0,
            data_freshness: 70.0,
            schema_validation: 70.0,
            manifest_completeness: 60.0,
            card_coverage: 60.0,
            theme_adherence: 50.0,
            naming_conventions: 50.0,
        };

        // structural: 90 * 0.25 = 22.5
        assert!((breakdown.structural_score() - 22.5).abs() < 0.01);
        // performance: 80 * 0.20 = 16.0
        assert!((breakdown.performance_score() - 16.0).abs() < 0.01);
        // accessibility: 100 * 0.20 = 20.0
        assert!((breakdown.accessibility_score() - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_app_quality_score_from_breakdown() {
        let breakdown = ScoreBreakdown {
            widget_complexity: 90.0,
            layout_depth: 90.0,
            component_count: 90.0,
            render_time_p95: 90.0,
            memory_usage: 90.0,
            bundle_size: 90.0,
            wcag_aa_compliance: 90.0,
            keyboard_navigation: 90.0,
            screen_reader: 90.0,
            data_completeness: 90.0,
            data_freshness: 90.0,
            schema_validation: 90.0,
            manifest_completeness: 90.0,
            card_coverage: 90.0,
            theme_adherence: 90.0,
            naming_conventions: 90.0,
        };

        let score = AppQualityScore::from_breakdown(breakdown);
        assert!((score.overall - 90.0).abs() < 0.01);
        assert_eq!(score.grade, Grade::A);
    }

    #[test]
    fn test_app_quality_score_meets_minimum() {
        let score = QualityScoreBuilder::new()
            .structural(85.0, 85.0, 85.0)
            .performance(85.0, 85.0, 85.0)
            .accessibility(85.0, 85.0, 85.0)
            .data_quality(85.0, 85.0, 85.0)
            .documentation(85.0, 85.0)
            .consistency(85.0, 85.0)
            .build();

        assert!(score.meets_minimum(Grade::BPlus));
        assert!(!score.meets_minimum(Grade::A));
    }

    #[test]
    fn test_app_quality_score_production_ready() {
        let ready = QualityScoreBuilder::new()
            .structural(90.0, 90.0, 90.0)
            .performance(90.0, 90.0, 90.0)
            .accessibility(90.0, 90.0, 90.0)
            .data_quality(90.0, 90.0, 90.0)
            .documentation(90.0, 90.0)
            .consistency(90.0, 90.0)
            .build();

        assert!(ready.is_production_ready());

        let not_ready = QualityScoreBuilder::new()
            .structural(70.0, 70.0, 70.0)
            .performance(70.0, 70.0, 70.0)
            .accessibility(70.0, 70.0, 70.0)
            .data_quality(70.0, 70.0, 70.0)
            .documentation(70.0, 70.0)
            .consistency(70.0, 70.0)
            .build();

        assert!(!not_ready.is_production_ready());
    }

    #[test]
    fn test_quality_score_builder() {
        let score = QualityScoreBuilder::new()
            .structural(100.0, 100.0, 100.0)
            .performance(100.0, 100.0, 100.0)
            .accessibility(100.0, 100.0, 100.0)
            .data_quality(100.0, 100.0, 100.0)
            .documentation(100.0, 100.0)
            .consistency(100.0, 100.0)
            .build();

        assert!((score.overall - 100.0).abs() < 0.01);
        assert_eq!(score.grade, Grade::APlus);
    }

    #[test]
    fn test_quality_score_builder_clamping() {
        let score = QualityScoreBuilder::new()
            .structural(150.0, -10.0, 200.0)
            .build();

        // Values should be clamped to 0-100
        assert_eq!(score.breakdown.widget_complexity, 100.0);
        assert_eq!(score.breakdown.layout_depth, 0.0);
        assert_eq!(score.breakdown.component_count, 100.0);
    }

    // =========================================================================
    // QualityGates Tests
    // =========================================================================

    #[test]
    fn test_quality_gates_default() {
        let gates = QualityGates::default();
        assert_eq!(gates.min_grade, Grade::BPlus);
        assert_eq!(gates.min_score, 80.0);
        assert_eq!(gates.performance.max_render_time_ms, 16);
        assert_eq!(gates.accessibility.wcag_level, "AA");
    }

    #[test]
    fn test_quality_gates_check_passes() {
        let gates = QualityGates::default();
        let score = QualityScoreBuilder::new()
            .structural(90.0, 90.0, 90.0)
            .performance(90.0, 90.0, 90.0)
            .accessibility(90.0, 90.0, 90.0)
            .data_quality(90.0, 90.0, 90.0)
            .documentation(90.0, 90.0)
            .consistency(90.0, 90.0)
            .build();

        let result = gates.check(&score);
        assert!(result.passed);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_quality_gates_check_fails_grade() {
        let gates = QualityGates::default();
        let score = QualityScoreBuilder::new()
            .structural(60.0, 60.0, 60.0)
            .performance(60.0, 60.0, 60.0)
            .accessibility(60.0, 60.0, 60.0)
            .data_quality(60.0, 60.0, 60.0)
            .documentation(60.0, 60.0)
            .consistency(60.0, 60.0)
            .build();

        let result = gates.check(&score);
        assert!(!result.passed);
        assert!(!result.violations.is_empty());
        assert_eq!(result.violations[0].gate, "min_grade");
    }

    #[test]
    fn test_quality_gates_check_fails_score() {
        let gates = QualityGates {
            min_grade: Grade::C, // Lower grade threshold
            min_score: 95.0,     // But require high score
            ..QualityGates::default()
        };

        let score = QualityScoreBuilder::new()
            .structural(85.0, 85.0, 85.0)
            .performance(85.0, 85.0, 85.0)
            .accessibility(85.0, 85.0, 85.0)
            .data_quality(85.0, 85.0, 85.0)
            .documentation(85.0, 85.0)
            .consistency(85.0, 85.0)
            .build();

        let result = gates.check(&score);
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.gate == "min_score"));
    }

    #[test]
    fn test_performance_gates_default() {
        let gates = PerformanceGates::default();
        assert_eq!(gates.max_render_time_ms, 16);
        assert_eq!(gates.max_bundle_size_kb, 500);
        assert_eq!(gates.max_memory_mb, 100);
    }

    #[test]
    fn test_accessibility_gates_default() {
        let gates = AccessibilityGates::default();
        assert_eq!(gates.wcag_level, "AA");
        assert_eq!(gates.min_contrast_ratio, 4.5);
        assert!(gates.require_keyboard_nav);
        assert!(gates.require_aria_labels);
    }

    #[test]
    fn test_documentation_gates_default() {
        let gates = DocumentationGates::default();
        assert!(gates.require_model_cards);
        assert!(gates.require_data_cards);
        assert!(gates.min_manifest_fields.contains(&"name".to_string()));
        assert!(gates.min_manifest_fields.contains(&"version".to_string()));
    }

    #[test]
    fn test_violation_severity() {
        let violation = GateViolation {
            gate: "test".to_string(),
            expected: "A".to_string(),
            actual: "B".to_string(),
            severity: ViolationSeverity::Error,
        };
        assert_eq!(violation.severity, ViolationSeverity::Error);
    }

    // =========================================================================
    // TOML Configuration Tests
    // =========================================================================

    #[test]
    fn test_gate_config_error_display() {
        let err = GateConfigError::ParseError("invalid toml".to_string());
        assert!(err.to_string().contains("parse error"));

        let err = GateConfigError::IoError("file not found".to_string());
        assert!(err.to_string().contains("IO error"));

        let err = GateConfigError::InvalidValue("out of range".to_string());
        assert!(err.to_string().contains("invalid value"));
    }

    #[test]
    fn test_quality_gates_to_toml() {
        let gates = QualityGates::default();
        let toml_str = gates.to_toml();

        assert!(toml_str.contains("min_score"));
        assert!(toml_str.contains("[performance]"));
        assert!(toml_str.contains("[accessibility]"));
        assert!(toml_str.contains("max_render_time_ms"));
    }

    #[test]
    fn test_quality_gates_from_toml() {
        let toml_str = r#"
            min_grade = "A"
            min_score = 90.0

            [performance]
            max_render_time_ms = 8
            max_bundle_size_kb = 300
            max_memory_mb = 50

            [accessibility]
            wcag_level = "AAA"
            min_contrast_ratio = 7.0
            require_keyboard_nav = true
            require_aria_labels = true

            [data]
            max_staleness_minutes = 30
            require_schema_validation = true

            [documentation]
            require_model_cards = true
            require_data_cards = true
            min_manifest_fields = ["name", "version"]
        "#;

        let gates = QualityGates::from_toml(toml_str).unwrap();
        assert_eq!(gates.min_score, 90.0);
        assert_eq!(gates.performance.max_render_time_ms, 8);
        assert_eq!(gates.performance.max_bundle_size_kb, 300);
        assert_eq!(gates.accessibility.wcag_level, "AAA");
        assert_eq!(gates.accessibility.min_contrast_ratio, 7.0);
        assert_eq!(gates.data.max_staleness_minutes, 30);
    }

    #[test]
    fn test_quality_gates_roundtrip() {
        let original = QualityGates::default();
        let toml_str = original.to_toml();
        let parsed = QualityGates::from_toml(&toml_str).unwrap();

        assert_eq!(parsed.min_score, original.min_score);
        assert_eq!(
            parsed.performance.max_render_time_ms,
            original.performance.max_render_time_ms
        );
        assert_eq!(
            parsed.accessibility.wcag_level,
            original.accessibility.wcag_level
        );
    }

    #[test]
    fn test_quality_gates_from_toml_invalid() {
        let result = QualityGates::from_toml("this is not valid toml {{{");
        assert!(matches!(result, Err(GateConfigError::ParseError(_))));
    }

    #[test]
    fn test_quality_gates_sample_config() {
        let sample = QualityGates::sample_config();
        assert!(sample.contains("min_grade"));
        assert!(sample.contains("max_bundle_size_kb"));
        assert!(sample.contains("wcag_level"));
        assert!(sample.contains("[performance]"));
        assert!(sample.contains("[accessibility]"));
        assert!(sample.contains("[data]"));
        assert!(sample.contains("[documentation]"));
    }

    #[test]
    fn test_quality_gates_check_extended_passes() {
        let gates = QualityGates::default();
        let score = QualityScoreBuilder::new()
            .structural(90.0, 90.0, 90.0)
            .performance(90.0, 90.0, 90.0)
            .accessibility(90.0, 90.0, 90.0)
            .data_quality(90.0, 90.0, 90.0)
            .documentation(90.0, 90.0)
            .consistency(90.0, 90.0)
            .build();

        let result = gates.check_extended(&score, Some(10), Some(400), Some(50));
        assert!(result.passed);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_quality_gates_check_extended_render_time_fails() {
        let gates = QualityGates::default();
        let score = QualityScoreBuilder::new()
            .structural(90.0, 90.0, 90.0)
            .performance(90.0, 90.0, 90.0)
            .accessibility(90.0, 90.0, 90.0)
            .data_quality(90.0, 90.0, 90.0)
            .documentation(90.0, 90.0)
            .consistency(90.0, 90.0)
            .build();

        // Render time exceeds 16ms limit
        let result = gates.check_extended(&score, Some(25), Some(400), Some(50));
        assert!(!result.passed);
        assert!(result
            .violations
            .iter()
            .any(|v| v.gate == "max_render_time_ms"));
    }

    #[test]
    fn test_quality_gates_check_extended_bundle_size_fails() {
        let gates = QualityGates::default();
        let score = QualityScoreBuilder::new()
            .structural(90.0, 90.0, 90.0)
            .performance(90.0, 90.0, 90.0)
            .accessibility(90.0, 90.0, 90.0)
            .data_quality(90.0, 90.0, 90.0)
            .documentation(90.0, 90.0)
            .consistency(90.0, 90.0)
            .build();

        // Bundle size exceeds 500KB limit
        let result = gates.check_extended(&score, Some(10), Some(600), Some(50));
        assert!(!result.passed);
        assert!(result
            .violations
            .iter()
            .any(|v| v.gate == "max_bundle_size_kb"));
    }

    #[test]
    fn test_quality_gates_check_extended_memory_warning() {
        let gates = QualityGates::default();
        let score = QualityScoreBuilder::new()
            .structural(90.0, 90.0, 90.0)
            .performance(90.0, 90.0, 90.0)
            .accessibility(90.0, 90.0, 90.0)
            .data_quality(90.0, 90.0, 90.0)
            .documentation(90.0, 90.0)
            .consistency(90.0, 90.0)
            .build();

        // Memory exceeds limit but is only a warning
        let result = gates.check_extended(&score, Some(10), Some(400), Some(150));
        assert!(result.passed); // Memory is a warning, not error
        assert!(result.violations.iter().any(|v| v.gate == "max_memory_mb"));
        assert!(result
            .violations
            .iter()
            .any(|v| v.severity == ViolationSeverity::Warning));
    }

    #[test]
    fn test_grade_from_str() {
        assert_eq!(Grade::from_str("A+").unwrap(), Grade::APlus);
        assert_eq!(Grade::from_str("a+").unwrap(), Grade::APlus);
        assert_eq!(Grade::from_str("A").unwrap(), Grade::A);
        assert_eq!(Grade::from_str("A-").unwrap(), Grade::AMinus);
        assert_eq!(Grade::from_str("B+").unwrap(), Grade::BPlus);
        assert_eq!(Grade::from_str("B").unwrap(), Grade::B);
        assert_eq!(Grade::from_str("B-").unwrap(), Grade::BMinus);
        assert_eq!(Grade::from_str("C+").unwrap(), Grade::CPlus);
        assert_eq!(Grade::from_str("C").unwrap(), Grade::C);
        assert_eq!(Grade::from_str("C-").unwrap(), Grade::CMinus);
        assert_eq!(Grade::from_str("D").unwrap(), Grade::D);
        assert_eq!(Grade::from_str("F").unwrap(), Grade::F);
    }

    #[test]
    fn test_grade_from_str_invalid() {
        assert!(matches!(
            Grade::from_str("X"),
            Err(GateConfigError::InvalidValue(_))
        ));
        assert!(matches!(
            Grade::from_str("E"),
            Err(GateConfigError::InvalidValue(_))
        ));
        assert!(matches!(
            Grade::from_str(""),
            Err(GateConfigError::InvalidValue(_))
        ));
    }

    #[test]
    fn test_quality_gates_config_file_constant() {
        assert_eq!(QualityGates::CONFIG_FILE, ".presentar-gates.toml");
    }
}
