//! 100-Point Quality Scoring System (GH-6)
//!
//! Based on the Toyota Way principles of Jidoka (built-in quality) and
//! the Doctest Corpus QA Checklist for Publication.

use std::{collections::HashMap, fmt};

/// Severity levels for quality issues per QA checklist
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Severity {
    /// Critical issues block publication (2.0x weight)
    Critical,
    /// High priority issues (1.5x weight)
    High,
    /// Medium priority issues (1.0x weight)
    Medium,
    /// Low priority issues (0.5x weight)
    Low,
}

impl Severity {
    /// Get the weight multiplier for this severity
    #[must_use]
    pub fn weight(&self) -> f64 {
        match self {
            Self::Critical => 2.0,
            Self::High => 1.5,
            Self::Medium => 1.0,
            Self::Low => 0.5,
        }
    }

    /// Get the base point value for this severity
    #[must_use]
    pub fn base_points(&self) -> f64 {
        match self {
            Self::Critical => 2.0,
            Self::High => 1.5,
            Self::Medium => 1.0,
            Self::Low => 0.5,
        }
    }
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Critical => write!(f, "Critical"),
            Self::High => write!(f, "High"),
            Self::Medium => write!(f, "Medium"),
            Self::Low => write!(f, "Low"),
        }
    }
}

/// Letter grades for dataset quality
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LetterGrade {
    /// A (95-100): Publish immediately
    A,
    /// B (85-94): Publish with documented caveats
    B,
    /// C (70-84): Remediation required before publication
    C,
    /// D (50-69): Major rework needed
    D,
    /// F (<50): Do not publish
    F,
}

impl LetterGrade {
    /// Create a letter grade from a numeric score (0-100)
    #[must_use]
    pub fn from_score(score: f64) -> Self {
        match score {
            s if s >= 95.0 => Self::A,
            s if s >= 85.0 => Self::B,
            s if s >= 70.0 => Self::C,
            s if s >= 50.0 => Self::D,
            _ => Self::F,
        }
    }

    /// Get the publication decision for this grade
    #[must_use]
    pub fn publication_decision(&self) -> &'static str {
        match self {
            Self::A => "Publish immediately",
            Self::B => "Publish with documented caveats",
            Self::C => "Remediation required before publication",
            Self::D => "Major rework needed",
            Self::F => "Do not publish",
        }
    }

    /// Check if this grade allows publication
    #[must_use]
    pub fn is_publishable(&self) -> bool {
        matches!(self, Self::A | Self::B)
    }
}

impl fmt::Display for LetterGrade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::A => write!(f, "A"),
            Self::B => write!(f, "B"),
            Self::C => write!(f, "C"),
            Self::D => write!(f, "D"),
            Self::F => write!(f, "F"),
        }
    }
}

/// A scored quality check item from the 100-point checklist
#[derive(Debug, Clone)]
pub struct ChecklistItem {
    /// Unique identifier (e.g., "1", "25", "53")
    pub id: u8,
    /// Check description
    pub description: String,
    /// Pass/fail status
    pub passed: bool,
    /// Severity level
    pub severity: Severity,
    /// Suggestion for improvement if failed
    pub suggestion: Option<String>,
}

impl ChecklistItem {
    /// Create a new checklist item
    #[must_use]
    pub fn new(id: u8, description: impl Into<String>, severity: Severity, passed: bool) -> Self {
        Self {
            id,
            description: description.into(),
            passed,
            severity,
            suggestion: None,
        }
    }

    /// Add a suggestion for improvement
    #[must_use]
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    /// Get the points earned (0 if failed, severity points if passed)
    #[must_use]
    pub fn points_earned(&self) -> f64 {
        if self.passed {
            self.severity.base_points()
        } else {
            0.0
        }
    }

    /// Get the maximum possible points for this item
    #[must_use]
    pub fn max_points(&self) -> f64 {
        self.severity.base_points()
    }
}

/// Complete quality score with breakdown
#[derive(Debug, Clone)]
pub struct QualityScore {
    /// Numeric score (0-100)
    pub score: f64,
    /// Letter grade
    pub grade: LetterGrade,
    /// Total points earned
    pub points_earned: f64,
    /// Maximum possible points
    pub max_points: f64,
    /// Individual checklist items
    pub checklist: Vec<ChecklistItem>,
    /// Summary statistics by severity
    pub severity_breakdown: HashMap<Severity, SeverityStats>,
}

/// Statistics for a severity level
#[derive(Debug, Clone, Default)]
pub struct SeverityStats {
    /// Number of checks at this severity
    pub total: usize,
    /// Number of passed checks
    pub passed: usize,
    /// Number of failed checks
    pub failed: usize,
    /// Points earned at this severity
    pub points_earned: f64,
    /// Maximum possible points at this severity
    pub max_points: f64,
}

impl QualityScore {
    /// Create a quality score from checklist items
    #[must_use]
    pub fn from_checklist(checklist: Vec<ChecklistItem>) -> Self {
        let mut severity_breakdown: HashMap<Severity, SeverityStats> = HashMap::new();

        let mut points_earned = 0.0;
        let mut max_points = 0.0;

        for item in &checklist {
            let stats = severity_breakdown.entry(item.severity).or_default();

            stats.total += 1;
            stats.max_points += item.max_points();

            if item.passed {
                stats.passed += 1;
                stats.points_earned += item.points_earned();
                points_earned += item.points_earned();
            } else {
                stats.failed += 1;
            }

            max_points += item.max_points();
        }

        let score = if max_points > 0.0 {
            (points_earned / max_points * 100.0).clamp(0.0, 100.0)
        } else {
            100.0
        };

        let grade = LetterGrade::from_score(score);

        Self {
            score,
            grade,
            points_earned,
            max_points,
            checklist,
            severity_breakdown,
        }
    }

    /// Get failed items for actionable suggestions
    #[must_use]
    pub fn failed_items(&self) -> Vec<&ChecklistItem> {
        self.checklist.iter().filter(|item| !item.passed).collect()
    }

    /// Get critical failures (blocks publication)
    #[must_use]
    pub fn critical_failures(&self) -> Vec<&ChecklistItem> {
        self.checklist
            .iter()
            .filter(|item| !item.passed && item.severity == Severity::Critical)
            .collect()
    }

    /// Check if there are any critical failures
    #[must_use]
    pub fn has_critical_failures(&self) -> bool {
        self.checklist
            .iter()
            .any(|item| !item.passed && item.severity == Severity::Critical)
    }

    /// Generate a badge URL for shields.io
    #[must_use]
    pub fn badge_url(&self) -> String {
        let color = match self.grade {
            LetterGrade::A => "brightgreen",
            LetterGrade::B => "green",
            LetterGrade::C => "yellow",
            LetterGrade::D => "orange",
            LetterGrade::F => "red",
        };
        format!(
            "https://img.shields.io/badge/data_quality-{}_({:.0}%25)-{}",
            self.grade, self.score, color
        )
    }

    /// Generate JSON output for CI/CD integration
    #[must_use]
    pub fn to_json(&self) -> String {
        let failed_items: Vec<_> = self
            .failed_items()
            .iter()
            .map(|item| {
                format!(
                    r#"    {{"id": {}, "description": "{}", "severity": "{}", "suggestion": {}}}"#,
                    item.id,
                    item.description.replace('"', "\\\""),
                    item.severity,
                    item.suggestion
                        .as_ref()
                        .map(|s| format!("\"{}\"", s.replace('"', "\\\"")))
                        .unwrap_or_else(|| "null".to_string())
                )
            })
            .collect();

        format!(
            r#"{{
  "score": {:.2},
  "grade": "{}",
  "is_publishable": {},
  "decision": "{}",
  "points_earned": {:.2},
  "max_points": {:.2},
  "critical_failures": {},
  "failed_items": [
{}
  ],
  "badge_url": "{}"
}}"#,
            self.score,
            self.grade,
            self.grade.is_publishable(),
            self.grade.publication_decision(),
            self.points_earned,
            self.max_points,
            self.has_critical_failures(),
            failed_items.join(",\n"),
            self.badge_url()
        )
    }
}
