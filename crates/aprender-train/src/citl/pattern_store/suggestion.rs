//! Fix suggestion representation for retrieval results.

use super::FixPattern;

/// A suggested fix from the pattern store
#[derive(Debug, Clone)]
pub struct FixSuggestion {
    /// The fix pattern
    pub pattern: FixPattern,
    /// Retrieval score from the RAG pipeline
    pub score: f32,
    /// Rank in the result set
    pub rank: usize,
}

impl FixSuggestion {
    /// Create a new fix suggestion
    #[must_use]
    pub fn new(pattern: FixPattern, score: f32, rank: usize) -> Self {
        Self { pattern, score, rank }
    }

    /// Get the weighted score (retrieval score * success rate)
    #[must_use]
    pub fn weighted_score(&self) -> f32 {
        self.score * (0.5 + 0.5 * self.pattern.success_rate())
    }
}
