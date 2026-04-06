//! Quality type definitions
//!
//! Contains core types for quality assessment:
//! - `QualityGrade` - Grade levels (A+ through F)
//! - `Score` - Score with context
//! - `IssueSeverity` - Issue severity levels
//! - `QualityIssue` - Quality issues found during analysis
//! - `StackLayer` - Layer in the PAIML stack architecture

use serde::{Deserialize, Serialize};

// ============================================================================
// Quality Grade
// ============================================================================

/// Quality grade levels based on score percentages
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum QualityGrade {
    /// A+ (95-100% of max score)
    APlus,
    /// A (90-94%)
    A,
    /// A- (85-89%) - PMAT minimum for production
    AMinus,
    /// B+ (80-84%)
    BPlus,
    /// B (70-79%)
    B,
    /// C (60-69%)
    C,
    /// D (50-59%)
    D,
    /// F (0-49%)
    F,
}

impl QualityGrade {
    /// Calculate grade from Rust project score (max 114)
    pub fn from_rust_project_score(score: u32) -> Self {
        match score {
            105..=114 => Self::APlus,
            95..=104 => Self::A,
            85..=94 => Self::AMinus,
            80..=84 => Self::BPlus,
            70..=79 => Self::B,
            60..=69 => Self::C,
            50..=59 => Self::D,
            _ => Self::F,
        }
    }

    /// Calculate grade from repository score (max 110)
    pub fn from_repo_score(score: u32) -> Self {
        match score {
            95..=110 => Self::APlus,
            90..=94 => Self::A,
            85..=89 => Self::AMinus,
            80..=84 => Self::BPlus,
            70..=79 => Self::B,
            60..=69 => Self::C,
            50..=59 => Self::D,
            _ => Self::F,
        }
    }

    /// Calculate grade from README score (max 20)
    pub fn from_readme_score(score: u32) -> Self {
        match score {
            18..=20 => Self::APlus,
            16..=17 => Self::A,
            14..=15 => Self::AMinus,
            12..=13 => Self::BPlus,
            10..=11 => Self::B,
            8..=9 => Self::C,
            6..=7 => Self::D,
            _ => Self::F,
        }
    }

    /// Calculate grade from Stack Quality Index (0-100)
    pub fn from_sqi(sqi: f64) -> Self {
        match sqi as u32 {
            95..=100 => Self::APlus,
            90..=94 => Self::A,
            85..=89 => Self::AMinus,
            80..=84 => Self::BPlus,
            70..=79 => Self::B,
            60..=69 => Self::C,
            50..=59 => Self::D,
            _ => Self::F,
        }
    }

    /// Check if grade is release-ready (A- or better)
    pub fn is_release_ready(&self) -> bool {
        matches!(self, Self::APlus | Self::A | Self::AMinus)
    }

    /// Check if grade is A+
    pub fn is_a_plus(&self) -> bool {
        matches!(self, Self::APlus)
    }

    /// Get display symbol for grade
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::APlus => "A+",
            Self::A => "A",
            Self::AMinus => "A-",
            Self::BPlus => "B+",
            Self::B => "B",
            Self::C => "C",
            Self::D => "D",
            Self::F => "F",
        }
    }

    /// Get status icon for grade
    pub fn icon(&self) -> &'static str {
        match self {
            Self::APlus => "✅",
            Self::A | Self::AMinus => "⚠️",
            _ => "❌",
        }
    }
}

impl std::fmt::Display for QualityGrade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.symbol())
    }
}

// ============================================================================
// Score Types
// ============================================================================

/// Score with context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Score {
    /// Actual score value
    pub value: u32,
    /// Maximum possible score
    pub max: u32,
    /// Calculated grade
    pub grade: QualityGrade,
}

impl Score {
    /// Create a new score
    pub fn new(value: u32, max: u32, grade: QualityGrade) -> Self {
        Self { value, max, grade }
    }

    /// Calculate percentage
    pub fn percentage(&self) -> f64 {
        if self.max == 0 {
            0.0
        } else {
            (self.value as f64 / self.max as f64) * 100.0
        }
    }

    /// Normalize to 0-100 scale
    pub fn normalized(&self) -> f64 {
        self.percentage()
    }
}

// ============================================================================
// Quality Issue
// ============================================================================

/// Severity of a quality issue
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Blocks release
    Error,
    /// Should be addressed
    Warning,
    /// Informational
    Info,
}

/// A quality issue found during analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    /// Issue type identifier
    pub issue_type: String,
    /// Human-readable message
    pub message: String,
    /// Severity level
    pub severity: IssueSeverity,
    /// Recommended action
    pub recommendation: Option<String>,
}

impl QualityIssue {
    /// Create a new quality issue
    pub fn new(
        issue_type: impl Into<String>,
        message: impl Into<String>,
        severity: IssueSeverity,
    ) -> Self {
        Self {
            issue_type: issue_type.into(),
            message: message.into(),
            severity,
            recommendation: None,
        }
    }

    /// Add recommendation
    pub fn with_recommendation(mut self, rec: impl Into<String>) -> Self {
        self.recommendation = Some(rec.into());
        self
    }

    /// Create error for score below threshold
    pub fn score_below_threshold(metric: &str, score: u32, threshold: u32) -> Self {
        Self::new(
            format!("{}_below_threshold", metric),
            format!("{} score {} below A- threshold ({})", metric, score, threshold),
            IssueSeverity::Error,
        )
        .with_recommendation(format!("Improve {} to at least {}", metric, threshold))
    }

    /// Create error for missing hero image
    pub fn missing_hero_image() -> Self {
        Self::new("missing_hero_image", "No hero image found", IssueSeverity::Error)
            .with_recommendation("Add hero.png to docs/ or include image at top of README.md")
    }
}

// ============================================================================
// Stack Layer
// ============================================================================

/// Layer in the PAIML stack architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StackLayer {
    /// Layer 0: Compute primitives (trueno, etc.)
    Compute,
    /// Layer 1: ML algorithms (aprender, etc.)
    Ml,
    /// Layer 2: Training & inference
    Training,
    /// Layer 3: Transpilers
    Transpilers,
    /// Layer 4: Orchestration
    Orchestration,
    /// Layer 5: Quality tools
    Quality,
    /// Layer 6: Data & MLOps
    DataMlops,
    /// Layer 7: Presentation
    Presentation,
}

impl StackLayer {
    /// Determine layer from component name
    pub fn from_component(name: &str) -> Self {
        match name {
            "trueno" | "trueno-viz" | "trueno-db" | "trueno-graph" | "trueno-rag"
            | "trueno-zram" => Self::Compute,
            "aprender" | "aprender-shell" | "aprender-tsp" => Self::Ml,
            "entrenar" | "realizar" | "whisper-apr" => Self::Training,
            "depyler" | "decy" | "ruchy" => Self::Transpilers,
            "batuta" | "repartir" | "pepita" | "pforge" | "forjar" => Self::Orchestration,
            "certeza" | "renacer" | "pmat" | "provable-contracts" | "tiny-model-ground-truth" => {
                Self::Quality
            }
            "alimentar" | "pacha" => Self::DataMlops,
            "presentar"
            | "sovereign-ai-stack-book"
            | "apr-cookbook"
            | "alm-cookbook"
            | "pres-cookbook" => Self::Presentation,
            _ => Self::Orchestration, // Default
        }
    }

    /// Get display name for layer
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Compute => "COMPUTE PRIMITIVES",
            Self::Ml => "ML ALGORITHMS",
            Self::Training => "TRAINING & INFERENCE",
            Self::Transpilers => "TRANSPILERS",
            Self::Orchestration => "ORCHESTRATION",
            Self::Quality => "QUALITY",
            Self::DataMlops => "DATA & MLOPS",
            Self::Presentation => "PRESENTATION",
        }
    }
}
