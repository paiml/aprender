//! Domain-specific data quality filtering (GH-453)
//!
//! Quality scoring and filtering for training data before fine-tuning.
//! Scores samples on multiple heuristic dimensions and filters by threshold.

use std::collections::HashSet;

/// Quality dimensions for scoring training samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QualityDimension {
    /// Text length within acceptable range
    Length,
    /// Vocabulary diversity (unique/total word ratio)
    Diversity,
    /// Repetition detection (repeated n-grams)
    Repetition,
    /// Structural quality (proper sentences, punctuation)
    Structure,
    /// Language consistency (ASCII ratio as proxy)
    Language,
}

/// Score for a single quality dimension.
#[derive(Debug, Clone, Copy)]
pub struct DimensionScore {
    /// The dimension being scored
    pub dimension: QualityDimension,
    /// Score in [0.0, 1.0] — higher is better
    pub score: f32,
    /// Whether this dimension passed the threshold
    pub passed: bool,
}

/// Overall quality report for a text sample.
#[derive(Debug, Clone)]
pub struct QualityReport {
    /// Per-dimension scores
    pub scores: Vec<DimensionScore>,
    /// Aggregate quality score (weighted mean)
    pub aggregate: f32,
    /// Whether the sample passed overall
    pub passed: bool,
}

/// Configuration for quality filtering.
#[derive(Debug, Clone)]
pub struct QualityConfig {
    /// Minimum aggregate quality score to pass
    pub min_quality: f32,
    /// Minimum text length (characters)
    pub min_length: usize,
    /// Maximum text length (characters)
    pub max_length: usize,
    /// Minimum vocabulary diversity ratio
    pub min_diversity: f32,
    /// Maximum repetition ratio (repeated 3-grams / total 3-grams)
    pub max_repetition: f32,
    /// Minimum ASCII ratio (language consistency proxy)
    pub min_ascii_ratio: f32,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            min_quality: 0.5,
            min_length: 10,
            max_length: 100_000,
            min_diversity: 0.3,
            max_repetition: 0.5,
            min_ascii_ratio: 0.8,
        }
    }
}

impl QualityConfig {
    /// Set minimum quality threshold.
    #[must_use]
    pub fn with_min_quality(mut self, q: f32) -> Self {
        self.min_quality = q.clamp(0.0, 1.0);
        self
    }

    /// Set length bounds.
    ///
    /// # Panics
    ///
    /// Panics if `min` is 0 (would cause division by zero in scoring).
    #[must_use]
    pub fn with_length_bounds(mut self, min: usize, max: usize) -> Self {
        assert!(min > 0, "min_length must be > 0");
        self.min_length = min;
        self.max_length = max.max(min);
        self
    }

    /// Set minimum diversity ratio.
    #[must_use]
    pub fn with_min_diversity(mut self, d: f32) -> Self {
        self.min_diversity = d.clamp(0.0, 1.0);
        self
    }
}

/// Score text quality across multiple dimensions.
#[must_use]
pub fn score_quality(text: &str, config: &QualityConfig) -> QualityReport {
    let scores = vec![
        score_length(text, config),
        score_diversity(text, config),
        score_repetition(text, config),
        score_structure(text),
        score_language(text, config),
    ];

    let aggregate = scores.iter().map(|s| s.score).sum::<f32>() / scores.len() as f32;
    let passed = aggregate >= config.min_quality && scores.iter().all(|s| s.score > 0.0);

    QualityReport {
        scores,
        aggregate,
        passed,
    }
}

/// Filter a batch of texts by quality, returning (passed, rejected) counts.
pub fn filter_by_quality<'a>(
    texts: &'a [String],
    config: &QualityConfig,
) -> (Vec<&'a String>, usize) {
    let mut passed = Vec::new();
    let mut rejected = 0;

    for text in texts {
        let report = score_quality(text, config);
        if report.passed {
            passed.push(text);
        } else {
            rejected += 1;
        }
    }

    (passed, rejected)
}

/// Score JSONL data quality, returning (filtered_lines, stats).
pub fn filter_jsonl_quality(input: &str, config: &QualityConfig) -> (String, FilterStats) {
    let mut output = String::new();
    let mut stats = FilterStats::default();

    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        stats.total += 1;

        // Extract text fields from JSON and score
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(line) {
            let text = extract_text_fields(&value);
            let report = score_quality(&text, config);

            if report.passed {
                output.push_str(line);
                output.push('\n');
                stats.passed += 1;
            } else {
                stats.rejected += 1;
                stats.rejection_reasons.push(format!(
                    "score={:.2}: {}",
                    report.aggregate,
                    report
                        .scores
                        .iter()
                        .filter(|s| !s.passed)
                        .map(|s| format!("{:?}={:.2}", s.dimension, s.score))
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        } else {
            stats.rejected += 1;
            stats.rejection_reasons.push("invalid JSON".to_string());
        }
    }

    (output, stats)
}

/// Statistics from quality filtering.
#[derive(Debug, Clone, Default)]
pub struct FilterStats {
    /// Total lines processed
    pub total: usize,
    /// Lines that passed quality filter
    pub passed: usize,
    /// Lines rejected
    pub rejected: usize,
    /// Reasons for rejection (one per rejected line)
    pub rejection_reasons: Vec<String>,
}

impl FilterStats {
    /// Pass rate as a fraction in [0.0, 1.0].
    #[must_use]
    pub fn pass_rate(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        self.passed as f32 / self.total as f32
    }
}

// ── Dimension Scorers ───────────────────────────────────────

fn score_length(text: &str, config: &QualityConfig) -> DimensionScore {
    let len = text.len();
    let score = if len < config.min_length {
        len as f32 / config.min_length as f32
    } else if len > config.max_length {
        config.max_length as f32 / len as f32
    } else {
        1.0
    };
    DimensionScore {
        dimension: QualityDimension::Length,
        score,
        passed: score >= 0.5,
    }
}

fn score_diversity(text: &str, config: &QualityConfig) -> DimensionScore {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return DimensionScore {
            dimension: QualityDimension::Diversity,
            score: 0.0,
            passed: false,
        };
    }

    let unique: HashSet<&str> = words.iter().copied().collect();
    let ratio = unique.len() as f32 / words.len() as f32;

    DimensionScore {
        dimension: QualityDimension::Diversity,
        score: ratio,
        passed: ratio >= config.min_diversity,
    }
}

fn score_repetition(text: &str, config: &QualityConfig) -> DimensionScore {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 3 {
        return DimensionScore {
            dimension: QualityDimension::Repetition,
            score: 1.0,
            passed: true,
        };
    }

    // Count repeated 3-grams
    let total_trigrams = words.len() - 2;
    let mut seen = HashSet::new();
    let mut repeated = 0;

    for window in words.windows(3) {
        let trigram = (window[0], window[1], window[2]);
        if !seen.insert(trigram) {
            repeated += 1;
        }
    }

    let repetition_ratio = repeated as f32 / total_trigrams as f32;
    let score = 1.0 - repetition_ratio;

    DimensionScore {
        dimension: QualityDimension::Repetition,
        score,
        passed: repetition_ratio <= config.max_repetition,
    }
}

fn score_structure(text: &str) -> DimensionScore {
    if text.is_empty() {
        return DimensionScore {
            dimension: QualityDimension::Structure,
            score: 0.0,
            passed: false,
        };
    }

    let mut score = 0.0;

    // Has sentence-ending punctuation
    let has_ending = text.ends_with('.') || text.ends_with('!') || text.ends_with('?');
    if has_ending {
        score += 0.3;
    }

    // Has capitalized first character
    if text.chars().next().is_some_and(|c| c.is_uppercase()) {
        score += 0.3;
    }

    // Has reasonable word count (not just 1-2 words)
    let word_count = text.split_whitespace().count();
    if word_count >= 3 {
        score += 0.2;
    }

    // Has some variety in punctuation (commas, semicolons, etc.)
    let punct_variety = text
        .chars()
        .filter(|c| matches!(c, ',' | ';' | ':' | '-' | '(' | ')'))
        .count();
    if punct_variety > 0 {
        score += 0.2;
    }

    DimensionScore {
        dimension: QualityDimension::Structure,
        score,
        passed: score >= 0.3,
    }
}

fn score_language(text: &str, config: &QualityConfig) -> DimensionScore {
    if text.is_empty() {
        return DimensionScore {
            dimension: QualityDimension::Language,
            score: 0.0,
            passed: false,
        };
    }

    let ascii_count = text.chars().filter(|c| c.is_ascii()).count();
    let ratio = ascii_count as f32 / text.chars().count() as f32;

    DimensionScore {
        dimension: QualityDimension::Language,
        score: ratio,
        passed: ratio >= config.min_ascii_ratio,
    }
}

/// Extract all string values from a JSON value into a single text.
fn extract_text_fields(value: &serde_json::Value) -> String {
    let mut parts = Vec::new();
    collect_strings(value, &mut parts);
    parts.join(" ")
}

fn collect_strings(value: &serde_json::Value, parts: &mut Vec<String>) {
    match value {
        serde_json::Value::String(s) => parts.push(s.clone()),
        serde_json::Value::Array(arr) => {
            for v in arr {
                collect_strings(v, parts);
            }
        }
        serde_json::Value::Object(obj) => {
            for v in obj.values() {
                collect_strings(v, parts);
            }
        }
        _ => {}
    }
}

use serde_json;

#[cfg(test)]
#[path = "quality_filter_tests.rs"]
mod tests;
