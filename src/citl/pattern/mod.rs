//! Pattern library for storing and retrieving error-fix patterns.
//!
//! Uses approximate nearest neighbor search for efficient pattern matching.

use super::encoder::ErrorEmbedding;
use super::error::{CITLError, CITLResult};
use super::{Difficulty, ErrorCategory, ErrorCode};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read as IoRead, Write as IoWrite};
use std::path::Path;
use trueno::Vector;

/// Magic header for pattern library files.
const MAGIC: &[u8; 4] = b"CITL";
/// Current format version.
const FORMAT_VERSION: u8 = 1;

/// Pattern library with similarity search.
///
/// Stores learned error-fix patterns for retrieval-augmented generation.
/// In production, this would use HNSW from `aprender::index`.
#[derive(Debug)]
pub struct PatternLibrary {
    /// Stored patterns
    patterns: Vec<ErrorFixPattern>,
    /// Index for fast search (simplified linear scan for now)
    embeddings: Vec<Vec<f32>>,
    /// Pattern statistics
    stats: PatternStats,
}

impl PatternLibrary {
    /// Create a new empty pattern library.
    #[must_use]
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            embeddings: Vec::new(),
            stats: PatternStats::new(),
        }
    }

    /// Load pattern library from file.
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or parsed.
    pub fn load(path: &str) -> CITLResult<Self> {
        let path = Path::new(path);
        if !path.exists() {
            return Err(CITLError::PatternLibraryError {
                message: format!("Pattern library not found: {}", path.display()),
            });
        }

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read and verify magic header
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(CITLError::PatternLibraryError {
                message: "Invalid pattern library file: bad magic header".to_string(),
            });
        }

        // Read version
        let mut version = [0u8; 1];
        reader.read_exact(&mut version)?;
        if version[0] != FORMAT_VERSION {
            return Err(CITLError::PatternLibraryError {
                message: format!("Unsupported format version: {}", version[0]),
            });
        }

        // Read pattern count
        let mut count_bytes = [0u8; 4];
        reader.read_exact(&mut count_bytes)?;
        let count = u32::from_le_bytes(count_bytes) as usize;

        let mut patterns = Vec::with_capacity(count);
        let mut embeddings = Vec::with_capacity(count);

        for _ in 0..count {
            let (pattern, embedding) = read_pattern(&mut reader)?;
            patterns.push(pattern);
            embeddings.push(embedding);
        }

        Ok(Self {
            patterns,
            embeddings,
            stats: PatternStats::new(),
        })
    }

    /// Save pattern library to file.
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be written.
    pub fn save(&self, path: &str) -> CITLResult<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write magic header
        writer.write_all(MAGIC)?;

        // Write version
        writer.write_all(&[FORMAT_VERSION])?;

        // Write pattern count
        let count = self.patterns.len() as u32;
        writer.write_all(&count.to_le_bytes())?;

        // Write each pattern with its embedding
        for (pattern, embedding) in self.patterns.iter().zip(self.embeddings.iter()) {
            write_pattern(&mut writer, pattern, embedding)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Search for similar error patterns.
    ///
    /// Uses linear scan with partial sort for top-k.
    /// For large libraries (10K+), consider HNSW from `aprender::index`.
    #[must_use]
    pub fn search(&self, query: &ErrorEmbedding, k: usize) -> Vec<PatternMatch> {
        if self.patterns.is_empty() {
            return Vec::new();
        }

        let n = self.embeddings.len();
        let k = k.min(n);

        // Compute similarities with pre-allocated capacity
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(n);
        for (idx, embedding) in self.embeddings.iter().enumerate() {
            let similarity = cosine_similarity(&query.vector, embedding);
            scored.push((idx, similarity));
        }

        // Partial sort: only sort enough to get top-k (O(n) average vs O(n log n))
        if k < n {
            scored.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(k);
            // Sort the top-k for consistent ordering
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Build results
        scored
            .into_iter()
            .map(|(idx, similarity)| PatternMatch {
                pattern: self.patterns[idx].clone(),
                similarity,
                success_rate: self.stats.success_rate(idx),
            })
            .collect()
    }

    /// Add a new pattern from successful fix.
    pub fn add_pattern(&mut self, error: ErrorEmbedding, fix: FixTemplate) {
        let pattern = ErrorFixPattern {
            error_code: error.error_code.clone(),
            context_hash: error.context_hash,
            fix_template: fix,
            success_count: 1,
            failure_count: 0,
        };

        self.embeddings.push(error.vector);
        self.patterns.push(pattern);
    }

    /// Update pattern statistics after fix attempt.
    pub fn record_outcome(&mut self, pattern_idx: usize, success: bool) {
        if pattern_idx < self.patterns.len() {
            if success {
                self.patterns[pattern_idx].success_count += 1;
            } else {
                self.patterns[pattern_idx].failure_count += 1;
            }
            self.stats.record(pattern_idx, success);
        }
    }

    /// Get the number of patterns.
    #[must_use]
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Check if the library is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Get pattern by index.
    #[must_use]
    pub fn get(&self, idx: usize) -> Option<&ErrorFixPattern> {
        self.patterns.get(idx)
    }

    /// Get patterns by error code.
    #[must_use]
    pub fn get_by_code(&self, code: &str) -> Vec<&ErrorFixPattern> {
        self.patterns
            .iter()
            .filter(|p| p.error_code.code == code)
            .collect()
    }
}

impl Default for PatternLibrary {
    fn default() -> Self {
        Self::new()
    }
}

/// Error-fix pattern learned from successful repairs.
#[derive(Debug, Clone)]
pub struct ErrorFixPattern {
    /// Error code this pattern applies to
    pub error_code: ErrorCode,
    /// Context hash for deduplication
    pub context_hash: u64,
    /// Parameterized fix template
    pub fix_template: FixTemplate,
    /// Number of successful applications
    pub success_count: u64,
    /// Number of failed applications
    pub failure_count: u64,
}

impl ErrorFixPattern {
    /// Get the success rate of this pattern.
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            0.0
        } else {
            self.success_count as f64 / total as f64
        }
    }

    /// Get total applications of this pattern.
    #[must_use]
    pub fn total_applications(&self) -> u64 {
        self.success_count + self.failure_count
    }
}

/// Parameterized fix template.
///
/// Templates capture the essence of fixes while allowing variable binding.
#[derive(Debug, Clone)]
pub struct FixTemplate {
    /// Template pattern with placeholders (e.g., "$`expr.to_string()`")
    pub pattern: String,
    /// Placeholder definitions
    pub placeholders: Vec<Placeholder>,
    /// Error codes this template applies to
    pub applicable_codes: Vec<String>,
    /// Confidence score (updated via reinforcement)
    pub confidence: f32,
    /// Human-readable description
    pub description: String,
}

impl FixTemplate {
    /// Create a new fix template.
    #[must_use]
    pub fn new(pattern: &str, description: &str) -> Self {
        Self {
            pattern: pattern.to_string(),
            placeholders: Vec::new(),
            applicable_codes: Vec::new(),
            confidence: 0.5,
            description: description.to_string(),
        }
    }

    /// Add a placeholder.
    #[must_use]
    pub fn with_placeholder(mut self, placeholder: Placeholder) -> Self {
        self.placeholders.push(placeholder);
        self
    }

    /// Add applicable error code.
    #[must_use]
    pub fn with_code(mut self, code: &str) -> Self {
        self.applicable_codes.push(code.to_string());
        self
    }

    /// Set confidence.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    /// Apply the template with given bindings.
    #[must_use]
    pub fn apply(&self, bindings: &HashMap<String, String>) -> String {
        let mut result = self.pattern.clone();
        for (name, value) in bindings {
            result = result.replace(&format!("${name}"), value);
        }
        result
    }

    /// Check if template applies to given error code.
    #[must_use]
    pub fn applies_to(&self, code: &str) -> bool {
        self.applicable_codes.is_empty() || self.applicable_codes.iter().any(|c| c == code)
    }
}

/// Placeholder in a fix template.
#[derive(Debug, Clone)]
pub struct Placeholder {
    /// Placeholder name (without $)
    pub name: String,
    /// Description of what this placeholder represents
    pub description: String,
    /// Type constraint (e.g., "expression", "type", "identifier")
    pub constraint: PlaceholderConstraint,
}

impl Placeholder {
    /// Create a new placeholder.
    #[must_use]
    pub fn new(name: &str, description: &str, constraint: PlaceholderConstraint) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            constraint,
        }
    }

    /// Create an expression placeholder.
    #[must_use]
    pub fn expression(name: &str) -> Self {
        Self::new(name, "An expression", PlaceholderConstraint::Expression)
    }

    /// Create a type placeholder.
    #[must_use]
    pub fn type_name(name: &str) -> Self {
        Self::new(name, "A type name", PlaceholderConstraint::Type)
    }

    /// Create an identifier placeholder.
    #[must_use]
    pub fn identifier(name: &str) -> Self {
        Self::new(name, "An identifier", PlaceholderConstraint::Identifier)
    }
}

/// Constraint on placeholder values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaceholderConstraint {
    /// Any expression
    Expression,
    /// Type name
    Type,
    /// Identifier
    Identifier,
    /// Literal value
    Literal,
    /// Any text
    Any,
}

/// Pattern match result.
#[derive(Debug, Clone)]
pub struct PatternMatch {
    /// The matched pattern
    pub pattern: ErrorFixPattern,
    /// Similarity score (0.0 to 1.0)
    pub similarity: f32,
    /// Historical success rate
    pub success_rate: f64,
}

impl PatternMatch {
    /// Get combined score (similarity * `success_rate`).
    #[must_use]
    pub fn combined_score(&self) -> f64 {
        f64::from(self.similarity) * self.success_rate
    }
}

/// Pattern statistics tracker.
#[derive(Debug, Clone)]
pub(super) struct PatternStats {
    /// Success counts by pattern index
    successes: HashMap<usize, u64>,
    /// Failure counts by pattern index
    failures: HashMap<usize, u64>,
}

impl PatternStats {
    /// Create new stats tracker.
    #[must_use]
    pub(super) fn new() -> Self {
        Self {
            successes: HashMap::new(),
            failures: HashMap::new(),
        }
    }

    /// Record an outcome.
    pub(super) fn record(&mut self, pattern_idx: usize, success: bool) {
        if success {
            *self.successes.entry(pattern_idx).or_insert(0) += 1;
        } else {
            *self.failures.entry(pattern_idx).or_insert(0) += 1;
        }
    }

    /// Get success rate for a pattern.
    #[must_use]
    pub(super) fn success_rate(&self, pattern_idx: usize) -> f64 {
        let successes = *self.successes.get(&pattern_idx).unwrap_or(&0);
        let failures = *self.failures.get(&pattern_idx).unwrap_or(&0);
        let total = successes + failures;

        if total == 0 {
            0.5 // Default rate for unseen patterns
        } else {
            successes as f64 / total as f64
        }
    }
}

include!("mod_part_02.rs");
include!("mod_part_03.rs");
