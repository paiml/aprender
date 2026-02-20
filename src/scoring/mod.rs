//! 100-Point Model Quality Scoring System (spec §7)
//!
//! Evaluates models across seven dimensions based on data science and ML best practices,
//! aligned with Toyota Way principles:
//!
//! | Dimension | Max Points | Toyota Way Principle |
//! |-----------|-----------|---------------------|
//! | Accuracy & Performance | 25 | Kaizen (continuous improvement) |
//! | Generalization & Robustness | 20 | Jidoka (quality built-in) |
//! | Model Complexity | 15 | Muda elimination (waste reduction) |
//! | Documentation & Provenance | 15 | Genchi Genbutsu (go and see) |
//! | Reproducibility | 15 | Standardization |
//! | Security & Safety | 10 | Poka-yoke (error-proofing) |
//!
//! # References
//!
//! - [Raschka 2018] Model Evaluation, Model Selection, and Algorithm Selection in ML
//! - [Hastie et al. 2009] The Elements of Statistical Learning
//! - [Mitchell et al. 2019] Model Cards for Model Reporting
//! - [Gebru et al. 2021] Datasheets for Datasets
//! - [Pineau et al. 2021] ML Reproducibility Checklist

use std::collections::HashMap;
use std::fmt;

/// 100-point model quality score
#[derive(Debug, Clone)]
pub struct QualityScore {
    /// Total score (0-100, normalized from 110 raw points)
    pub total: f32,

    /// Grade letter (A+, A, A-, B+, ...)
    pub grade: Grade,

    /// Raw score before normalization (0-110)
    pub raw_score: f32,

    /// Individual dimension scores
    pub dimensions: DimensionScores,

    /// Detailed findings and recommendations
    pub findings: Vec<Finding>,

    /// Critical issues that must be addressed
    pub critical_issues: Vec<CriticalIssue>,
}

impl QualityScore {
    /// Create a new quality score from dimension scores
    #[must_use]
    pub fn new(
        dimensions: DimensionScores,
        findings: Vec<Finding>,
        critical_issues: Vec<CriticalIssue>,
    ) -> Self {
        let raw_score = dimensions.total_raw();
        let total = (raw_score / 110.0) * 100.0;
        let grade = Grade::from_score(total);

        Self {
            total,
            grade,
            raw_score,
            dimensions,
            findings,
            critical_issues,
        }
    }

    /// Check if the model passes minimum quality threshold
    #[must_use]
    pub fn passes_threshold(&self, min_score: f32) -> bool {
        self.total >= min_score && self.critical_issues.is_empty()
    }

    /// Check if there are critical issues
    #[must_use]
    pub fn has_critical_issues(&self) -> bool {
        !self.critical_issues.is_empty()
    }

    /// Get warnings count
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| matches!(f, Finding::Warning { .. }))
            .count()
    }

    /// Get info count
    #[must_use]
    pub fn info_count(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| matches!(f, Finding::Info { .. }))
            .count()
    }
}

impl fmt::Display for QualityScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Model Quality Score: {:.1}/100 (Grade: {})",
            self.total, self.grade
        )?;
        writeln!(f, "\nDimension Breakdown:")?;
        writeln!(
            f,
            "  Accuracy & Performance:    {:.1}/25 ({:.0}%)",
            self.dimensions.accuracy_performance.score,
            self.dimensions.accuracy_performance.percentage
        )?;
        writeln!(
            f,
            "  Generalization & Robust:   {:.1}/20 ({:.0}%)",
            self.dimensions.generalization_robustness.score,
            self.dimensions.generalization_robustness.percentage
        )?;
        writeln!(
            f,
            "  Model Complexity:          {:.1}/15 ({:.0}%)",
            self.dimensions.model_complexity.score, self.dimensions.model_complexity.percentage
        )?;
        writeln!(
            f,
            "  Documentation & Provenance:{:.1}/15 ({:.0}%)",
            self.dimensions.documentation_provenance.score,
            self.dimensions.documentation_provenance.percentage
        )?;
        writeln!(
            f,
            "  Reproducibility:           {:.1}/15 ({:.0}%)",
            self.dimensions.reproducibility.score, self.dimensions.reproducibility.percentage
        )?;
        writeln!(
            f,
            "  Security & Safety:         {:.1}/10 ({:.0}%)",
            self.dimensions.security_safety.score, self.dimensions.security_safety.percentage
        )?;

        if !self.critical_issues.is_empty() {
            writeln!(f, "\nCritical Issues ({}):", self.critical_issues.len())?;
            for issue in &self.critical_issues {
                writeln!(f, "  - {issue}")?;
            }
        }

        if !self.findings.is_empty() {
            let warnings: Vec<_> = self
                .findings
                .iter()
                .filter(|f| matches!(f, Finding::Warning { .. }))
                .collect();
            if !warnings.is_empty() {
                writeln!(f, "\nWarnings ({}):", warnings.len())?;
                for finding in warnings {
                    writeln!(f, "  - {finding}")?;
                }
            }
        }

        Ok(())
    }
}

/// Letter grade for quality score
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Grade {
    /// A+ (97-100)
    APlus,
    /// A (93-96)
    A,
    /// A- (90-92)
    AMinus,
    /// B+ (87-89)
    BPlus,
    /// B (83-86)
    B,
    /// B- (80-82)
    BMinus,
    /// C+ (77-79)
    CPlus,
    /// C (73-76)
    C,
    /// C- (70-72)
    CMinus,
    /// D+ (67-69)
    DPlus,
    /// D (63-66)
    D,
    /// D- (60-62)
    DMinus,
    /// F (< 60)
    F,
}

impl Grade {
    /// Convert score to grade
    #[must_use]
    pub fn from_score(score: f32) -> Self {
        match score {
            s if s >= 97.0 => Self::APlus,
            s if s >= 93.0 => Self::A,
            s if s >= 90.0 => Self::AMinus,
            s if s >= 87.0 => Self::BPlus,
            s if s >= 83.0 => Self::B,
            s if s >= 80.0 => Self::BMinus,
            s if s >= 77.0 => Self::CPlus,
            s if s >= 73.0 => Self::C,
            s if s >= 70.0 => Self::CMinus,
            s if s >= 67.0 => Self::DPlus,
            s if s >= 63.0 => Self::D,
            s if s >= 60.0 => Self::DMinus,
            _ => Self::F,
        }
    }

    /// Get minimum score for this grade
    #[must_use]
    pub const fn min_score(&self) -> f32 {
        match self {
            Self::APlus => 97.0,
            Self::A => 93.0,
            Self::AMinus => 90.0,
            Self::BPlus => 87.0,
            Self::B => 83.0,
            Self::BMinus => 80.0,
            Self::CPlus => 77.0,
            Self::C => 73.0,
            Self::CMinus => 70.0,
            Self::DPlus => 67.0,
            Self::D => 63.0,
            Self::DMinus => 60.0,
            Self::F => 0.0,
        }
    }

    /// Check if grade is passing (C- or better)
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
                | Self::CMinus
        )
    }
}

impl fmt::Display for Grade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::APlus => "A+",
            Self::A => "A",
            Self::AMinus => "A-",
            Self::BPlus => "B+",
            Self::B => "B",
            Self::BMinus => "B-",
            Self::CPlus => "C+",
            Self::C => "C",
            Self::CMinus => "C-",
            Self::DPlus => "D+",
            Self::D => "D",
            Self::DMinus => "D-",
            Self::F => "F",
        };
        write!(f, "{s}")
    }
}

/// Individual dimension scores
#[derive(Debug, Clone)]
pub struct DimensionScores {
    /// Accuracy & Performance (25 points max)
    pub accuracy_performance: DimensionScore,
    /// Generalization & Robustness (20 points max)
    pub generalization_robustness: DimensionScore,
    /// Model Complexity (15 points max)
    pub model_complexity: DimensionScore,
    /// Documentation & Provenance (15 points max)
    pub documentation_provenance: DimensionScore,
    /// Reproducibility (15 points max)
    pub reproducibility: DimensionScore,
    /// Security & Safety (10 points max)
    pub security_safety: DimensionScore,
}

impl DimensionScores {
    /// Get total raw score (out of 110)
    #[must_use]
    pub fn total_raw(&self) -> f32 {
        self.accuracy_performance.score
            + self.generalization_robustness.score
            + self.model_complexity.score
            + self.documentation_provenance.score
            + self.reproducibility.score
            + self.security_safety.score
    }

    /// Create default dimension scores
    #[must_use]
    pub fn default_scores() -> Self {
        Self {
            accuracy_performance: DimensionScore::new(25.0),
            generalization_robustness: DimensionScore::new(20.0),
            model_complexity: DimensionScore::new(15.0),
            documentation_provenance: DimensionScore::new(15.0),
            reproducibility: DimensionScore::new(15.0),
            security_safety: DimensionScore::new(10.0),
        }
    }
}

impl Default for DimensionScores {
    fn default() -> Self {
        Self::default_scores()
    }
}

/// Score for a single dimension
#[derive(Debug, Clone)]
pub struct DimensionScore {
    /// Score achieved
    pub score: f32,
    /// Maximum possible score
    pub max_score: f32,
    /// Percentage achieved
    pub percentage: f32,
    /// Detailed breakdown (criterion, score, max)
    pub breakdown: Vec<ScoreBreakdown>,
}

impl DimensionScore {
    /// Create a new dimension score with no points
    #[must_use]
    pub fn new(max_score: f32) -> Self {
        Self {
            score: 0.0,
            max_score,
            percentage: 0.0,
            breakdown: Vec::new(),
        }
    }

    /// Add points for a criterion
    pub fn add_score(&mut self, criterion: impl Into<String>, score: f32, max: f32) {
        self.breakdown.push(ScoreBreakdown {
            criterion: criterion.into(),
            score,
            max,
        });
        self.score += score;
        self.update_percentage();
    }

    /// Update percentage based on current score
    fn update_percentage(&mut self) {
        self.percentage = if self.max_score > 0.0 {
            (self.score / self.max_score) * 100.0
        } else {
            0.0
        };
    }

    /// Check if dimension achieved perfect score
    #[must_use]
    pub fn is_perfect(&self) -> bool {
        (self.score - self.max_score).abs() < f32::EPSILON
    }

    /// Get completion ratio (0.0 to 1.0)
    #[must_use]
    pub fn completion_ratio(&self) -> f32 {
        if self.max_score > 0.0 {
            self.score / self.max_score
        } else {
            0.0
        }
    }
}

/// Breakdown item for dimension scoring
#[derive(Debug, Clone)]
pub struct ScoreBreakdown {
    /// Criterion name
    pub criterion: String,
    /// Score achieved
    pub score: f32,
    /// Maximum possible
    pub max: f32,
}

impl ScoreBreakdown {
    /// Get percentage achieved for this criterion
    #[must_use]
    pub fn percentage(&self) -> f32 {
        if self.max > 0.0 {
            (self.score / self.max) * 100.0
        } else {
            0.0
        }
    }
}

/// Finding from quality analysis
#[derive(Debug, Clone)]
pub enum Finding {
    /// Warning that should be addressed
    Warning {
        /// Warning message
        message: String,
        /// Recommended action
        recommendation: String,
    },
    /// Informational note
    Info {
        /// Info message
        message: String,
        /// Suggested improvement
        recommendation: String,
    },
}

impl fmt::Display for Finding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Warning {
                message,
                recommendation,
            } => {
                write!(f, "[WARN] {message} (Recommendation: {recommendation})")
            }
            Self::Info {
                message,
                recommendation,
            } => {
                write!(f, "[INFO] {message} (Suggestion: {recommendation})")
            }
        }
    }
}

include!("mod_part_02.rs");
include!("score.rs");
