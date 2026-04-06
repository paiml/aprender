//! HuggingFace leaderboard types
//!
//! Defines leaderboard kinds, entries, and result containers for
//! interacting with HuggingFace open evaluation leaderboards.

use std::collections::HashMap;

use crate::eval::evaluator::Metric;

/// Known HuggingFace evaluation leaderboards
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LeaderboardKind {
    /// HF Audio Open ASR Leaderboard
    OpenASR,
    /// Open LLM Leaderboard v2
    OpenLLMv2,
    /// MTEB (Massive Text Embedding Benchmark)
    MTEB,
    /// BigCodeBench
    BigCodeBench,
    /// Custom leaderboard by dataset repository ID
    Custom(String),
}

impl LeaderboardKind {
    /// Get the HuggingFace dataset repository ID for this leaderboard
    #[must_use]
    pub fn dataset_repo_id(&self) -> &str {
        match self {
            Self::OpenASR => "hf-audio/open_asr_leaderboard",
            Self::OpenLLMv2 => "open-llm-leaderboard/results",
            Self::MTEB => "mteb/leaderboard",
            Self::BigCodeBench => "bigcode/bigcodebench-results",
            Self::Custom(id) => id,
        }
    }

    /// Get the primary ranking metric for this leaderboard
    #[must_use]
    pub fn primary_metric(&self) -> Metric {
        match self {
            Self::OpenASR => Metric::WER,
            Self::OpenLLMv2 => Metric::MMLUAccuracy,
            Self::MTEB => Metric::NDCGAtK(10),
            Self::BigCodeBench => Metric::PassAtK(1),
            Self::Custom(_) => Metric::Accuracy,
        }
    }

    /// Get all tracked metrics for this leaderboard
    #[must_use]
    pub fn tracked_metrics(&self) -> Vec<Metric> {
        match self {
            Self::OpenASR => vec![Metric::WER, Metric::RTFx],
            Self::OpenLLMv2 => vec![Metric::MMLUAccuracy, Metric::Accuracy],
            Self::MTEB => vec![Metric::NDCGAtK(10), Metric::Accuracy],
            Self::BigCodeBench => vec![Metric::PassAtK(1), Metric::PassAtK(10)],
            Self::Custom(_) => vec![Metric::Accuracy],
        }
    }
}

impl std::fmt::Display for LeaderboardKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenASR => write!(f, "Open ASR Leaderboard"),
            Self::OpenLLMv2 => write!(f, "Open LLM Leaderboard v2"),
            Self::MTEB => write!(f, "MTEB Leaderboard"),
            Self::BigCodeBench => write!(f, "BigCodeBench"),
            Self::Custom(id) => write!(f, "Custom ({id})"),
        }
    }
}

/// A single entry (row) from a HuggingFace leaderboard
#[derive(Clone, Debug)]
pub struct LeaderboardEntry {
    /// Model repository ID (e.g., "openai/whisper-large-v3")
    pub model_id: String,
    /// Raw string-keyed scores from HuggingFace
    pub scores: HashMap<String, f64>,
    /// Additional metadata (license, parameter count, etc.)
    pub metadata: HashMap<String, String>,
}

impl LeaderboardEntry {
    /// Create a new leaderboard entry
    #[must_use]
    pub fn new(model_id: impl Into<String>) -> Self {
        Self { model_id: model_id.into(), scores: HashMap::new(), metadata: HashMap::new() }
    }

    /// Get a score by column name
    #[must_use]
    pub fn get_score(&self, column: &str) -> Option<f64> {
        self.scores.get(column).copied()
    }
}

/// Container for leaderboard data fetched from HuggingFace
#[derive(Clone, Debug)]
pub struct HfLeaderboard {
    /// Leaderboard kind
    pub kind: LeaderboardKind,
    /// Entries (rows)
    pub entries: Vec<LeaderboardEntry>,
    /// Total number of entries available (may be more than fetched)
    pub total_count: usize,
}

impl HfLeaderboard {
    /// Create a new leaderboard container
    #[must_use]
    pub fn new(kind: LeaderboardKind) -> Self {
        Self { kind, entries: Vec::new(), total_count: 0 }
    }

    /// Find an entry by model ID
    #[must_use]
    pub fn find_model(&self, model_id: &str) -> Option<&LeaderboardEntry> {
        self.entries.iter().find(|e| e.model_id == model_id)
    }
}
