//! Sentiment analysis with lexicon-based scoring.
//!
//! This module provides dictionary-based sentiment analysis:
//! - Sentiment lexicon with positive/negative word scores
//! - Document-level sentiment scoring
//! - Polarity classification (positive/negative/neutral)
//!
//! # Quick Start
//!
//! ```
//! use aprender::text::sentiment::SentimentAnalyzer;
//!
//! let analyzer = SentimentAnalyzer::default();
//! let score = analyzer.score("This movie is great and wonderful!").expect("score should succeed");
//!
//! println!("Sentiment: {:.3}", score);  // Positive score > 0
//! ```

use crate::AprenderError;
use std::collections::HashMap;

/// Sentiment polarity classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarity {
    /// Positive sentiment
    Positive,
    /// Negative sentiment
    Negative,
    /// Neutral sentiment
    Neutral,
}

/// Lexicon-based sentiment analyzer.
///
/// Uses a dictionary of positive and negative words with associated scores
/// to compute document-level sentiment.
///
/// # Examples
///
/// ```
/// use aprender::text::sentiment::SentimentAnalyzer;
///
/// let analyzer = SentimentAnalyzer::default();
///
/// // Positive text
/// let score = analyzer.score("amazing wonderful great").expect("score should succeed");
/// assert!(score > 0.0);
///
/// // Negative text
/// let score = analyzer.score("terrible awful horrible").expect("score should succeed");
/// assert!(score < 0.0);
/// ```
#[derive(Debug)]
pub struct SentimentAnalyzer {
    /// Sentiment lexicon: word -> score mapping
    lexicon: HashMap<String, f64>,
    /// Threshold for neutral classification
    neutral_threshold: f64,
}

impl SentimentAnalyzer {
    /// Create a new sentiment analyzer with default lexicon.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::sentiment::SentimentAnalyzer;
    ///
    /// let analyzer = SentimentAnalyzer::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            lexicon: Self::default_lexicon(),
            neutral_threshold: 0.05,
        }
    }

    /// Create analyzer with custom lexicon.
    ///
    /// # Arguments
    ///
    /// * `lexicon` - Word to sentiment score mapping (positive = >0, negative = <0)
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::sentiment::SentimentAnalyzer;
    /// use std::collections::HashMap;
    ///
    /// let mut lexicon = HashMap::new();
    /// lexicon.insert("good".to_string(), 1.0);
    /// lexicon.insert("bad".to_string(), -1.0);
    ///
    /// let analyzer = SentimentAnalyzer::with_lexicon(lexicon);
    /// ```
    #[must_use]
    pub fn with_lexicon(lexicon: HashMap<String, f64>) -> Self {
        Self {
            lexicon,
            neutral_threshold: 0.05,
        }
    }

    /// Set the neutral classification threshold.
    ///
    /// Scores with absolute value below this threshold are classified as neutral.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::sentiment::SentimentAnalyzer;
    ///
    /// let analyzer = SentimentAnalyzer::new().with_neutral_threshold(0.1);
    /// ```
    #[must_use]
    pub fn with_neutral_threshold(mut self, threshold: f64) -> Self {
        self.neutral_threshold = threshold;
        self
    }

    /// Compute sentiment score for text.
    ///
    /// Returns a score where:
    /// - Positive values indicate positive sentiment
    /// - Negative values indicate negative sentiment
    /// - Values near zero indicate neutral sentiment
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to analyze
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::sentiment::SentimentAnalyzer;
    ///
    /// let analyzer = SentimentAnalyzer::default();
    ///
    /// let score = analyzer.score("I love this amazing product!").expect("score should succeed");
    /// assert!(score > 0.0);  // Positive
    ///
    /// let score = analyzer.score("This is terrible and awful.").expect("score should succeed");
    /// assert!(score < 0.0);  // Negative
    /// ```
    pub fn score(&self, text: &str) -> Result<f64, AprenderError> {
        if text.is_empty() {
            return Ok(0.0);
        }

        // Tokenize on whitespace and punctuation
        let tokens: Vec<String> = text
            .to_lowercase()
            .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();

        if tokens.is_empty() {
            return Ok(0.0);
        }

        // Sum sentiment scores
        let total_score: f64 = tokens
            .iter()
            .filter_map(|token| self.lexicon.get(token))
            .sum();

        // Normalize by token count
        let normalized_score = total_score / tokens.len() as f64;

        Ok(normalized_score)
    }

    /// Classify text polarity.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to classify
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::sentiment::{SentimentAnalyzer, Polarity};
    ///
    /// let analyzer = SentimentAnalyzer::default();
    ///
    /// let polarity = analyzer.classify("I love this!").expect("classify should succeed");
    /// assert_eq!(polarity, Polarity::Positive);
    ///
    /// let polarity = analyzer.classify("This is terrible.").expect("classify should succeed");
    /// assert_eq!(polarity, Polarity::Negative);
    /// ```
    pub fn classify(&self, text: &str) -> Result<Polarity, AprenderError> {
        let score = self.score(text)?;

        if score > self.neutral_threshold {
            Ok(Polarity::Positive)
        } else if score < -self.neutral_threshold {
            Ok(Polarity::Negative)
        } else {
            Ok(Polarity::Neutral)
        }
    }

    /// Get the lexicon size.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::text::sentiment::SentimentAnalyzer;
    ///
    /// let analyzer = SentimentAnalyzer::default();
    /// assert!(analyzer.lexicon_size() > 0);
    /// ```
    #[must_use]
    pub fn lexicon_size(&self) -> usize {
        self.lexicon.len()
    }

    /// Default sentiment lexicon with common positive/negative words.
    fn default_lexicon() -> HashMap<String, f64> {
        let mut lexicon = HashMap::new();

        // Positive words (score: +1.0 to +3.0)
        let positive_strong = vec![
            "excellent",
            "amazing",
            "wonderful",
            "fantastic",
            "outstanding",
            "superb",
            "brilliant",
            "perfect",
            "love",
            "loved",
        ];
        for word in positive_strong {
            lexicon.insert(word.to_string(), 3.0);
        }

        let positive_moderate = vec![
            "good",
            "great",
            "nice",
            "fine",
            "happy",
            "glad",
            "pleased",
            "enjoy",
            "like",
            "better",
            "best",
            "positive",
            "beautiful",
            "awesome",
        ];
        for word in positive_moderate {
            lexicon.insert(word.to_string(), 2.0);
        }

        let positive_mild = vec!["ok", "okay", "decent", "fair", "acceptable", "well"];
        for word in positive_mild {
            lexicon.insert(word.to_string(), 1.0);
        }

        // Negative words (score: -1.0 to -3.0)
        let negative_strong = vec![
            "terrible",
            "awful",
            "horrible",
            "disgusting",
            "worst",
            "hate",
            "hated",
            "dreadful",
            "atrocious",
            "pathetic",
        ];
        for word in negative_strong {
            lexicon.insert(word.to_string(), -3.0);
        }

        let negative_moderate = vec![
            "bad",
            "poor",
            "disappointing",
            "sad",
            "unhappy",
            "angry",
            "upset",
            "annoying",
            "worse",
            "negative",
            "ugly",
            "boring",
        ];
        for word in negative_moderate {
            lexicon.insert(word.to_string(), -2.0);
        }

        let negative_mild = vec!["mediocre", "meh", "dull", "weak", "minor", "slight"];
        for word in negative_mild {
            lexicon.insert(word.to_string(), -1.0);
        }

        // Intensifiers (modify surrounding words)
        lexicon.insert("very".to_string(), 1.5);
        lexicon.insert("really".to_string(), 1.5);
        lexicon.insert("extremely".to_string(), 2.0);
        lexicon.insert("absolutely".to_string(), 2.0);

        // Negations (would need more sophisticated handling)
        lexicon.insert("not".to_string(), -1.0);
        lexicon.insert("no".to_string(), -1.0);
        lexicon.insert("never".to_string(), -1.5);

        lexicon
    }
}

impl Default for SentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "sentiment_tests.rs"]
mod tests;
