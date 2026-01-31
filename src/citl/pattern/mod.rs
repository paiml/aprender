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

impl Default for PatternStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute cosine similarity between two vectors using trueno SIMD.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let va = Vector::from_slice(a);
    let vb = Vector::from_slice(b);

    // Use trueno's SIMD-accelerated operations
    let dot = va.dot(&vb).unwrap_or(0.0);
    let norm_a = va.norm_l2().unwrap_or(0.0);
    let norm_b = vb.norm_l2().unwrap_or(0.0);

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

// ==================== Binary Serialization Helpers ====================

/// Write a pattern and its embedding to a writer.
fn write_pattern<W: IoWrite>(
    writer: &mut W,
    pattern: &ErrorFixPattern,
    embedding: &[f32],
) -> CITLResult<()> {
    // Error code
    write_string(writer, &pattern.error_code.code)?;
    writer.write_all(&[pattern.error_code.category as u8])?;
    writer.write_all(&[pattern.error_code.difficulty as u8])?;

    // Context hash
    writer.write_all(&pattern.context_hash.to_le_bytes())?;

    // Success/failure counts
    writer.write_all(&pattern.success_count.to_le_bytes())?;
    writer.write_all(&pattern.failure_count.to_le_bytes())?;

    // Fix template
    write_fix_template(writer, &pattern.fix_template)?;

    // Embedding
    let dim = embedding.len() as u32;
    writer.write_all(&dim.to_le_bytes())?;
    for val in embedding {
        writer.write_all(&val.to_le_bytes())?;
    }

    Ok(())
}

/// Convert a byte to an `ErrorCategory`.
fn parse_error_category(byte: u8) -> ErrorCategory {
    match byte {
        1 => ErrorCategory::TraitBound,
        2 => ErrorCategory::Unresolved,
        3 => ErrorCategory::Ownership,
        4 => ErrorCategory::Borrowing,
        5 => ErrorCategory::Lifetime,
        6 => ErrorCategory::Async,
        7 => ErrorCategory::TypeInference,
        8 => ErrorCategory::MethodNotFound,
        9 => ErrorCategory::Import,
        _ => ErrorCategory::TypeMismatch, // 0 and unknown default to TypeMismatch
    }
}

/// Convert a byte to a Difficulty.
fn parse_difficulty(byte: u8) -> Difficulty {
    match byte {
        0 => Difficulty::Easy,
        2 => Difficulty::Hard,
        3 => Difficulty::Expert,
        _ => Difficulty::Medium, // 1 and unknown default to Medium
    }
}

/// Convert a byte to a `PlaceholderConstraint`.
fn parse_placeholder_constraint(byte: u8) -> PlaceholderConstraint {
    match byte {
        0 => PlaceholderConstraint::Expression,
        1 => PlaceholderConstraint::Type,
        2 => PlaceholderConstraint::Identifier,
        3 => PlaceholderConstraint::Literal,
        _ => PlaceholderConstraint::Any,
    }
}

/// Read an error code from a reader.
fn read_error_code<R: IoRead>(reader: &mut R) -> CITLResult<ErrorCode> {
    let code_str = read_string(reader)?;
    let mut category_byte = [0u8; 1];
    reader.read_exact(&mut category_byte)?;
    let category = parse_error_category(category_byte[0]);
    let mut difficulty_byte = [0u8; 1];
    reader.read_exact(&mut difficulty_byte)?;
    let difficulty = parse_difficulty(difficulty_byte[0]);
    Ok(ErrorCode::new(&code_str, category, difficulty))
}

/// Read counts (success/failure) from a reader.
fn read_counts<R: IoRead>(reader: &mut R) -> CITLResult<(u64, u64)> {
    let mut success_bytes = [0u8; 8];
    reader.read_exact(&mut success_bytes)?;
    let success_count = u64::from_le_bytes(success_bytes);

    let mut failure_bytes = [0u8; 8];
    reader.read_exact(&mut failure_bytes)?;
    let failure_count = u64::from_le_bytes(failure_bytes);

    Ok((success_count, failure_count))
}

/// Read an embedding vector from a reader.
fn read_embedding<R: IoRead>(reader: &mut R) -> CITLResult<Vec<f32>> {
    let mut dim_bytes = [0u8; 4];
    reader.read_exact(&mut dim_bytes)?;
    let dim = u32::from_le_bytes(dim_bytes) as usize;

    let mut embedding = Vec::with_capacity(dim);
    for _ in 0..dim {
        let mut val_bytes = [0u8; 4];
        reader.read_exact(&mut val_bytes)?;
        embedding.push(f32::from_le_bytes(val_bytes));
    }
    Ok(embedding)
}

/// Read a single placeholder from a reader.
fn read_placeholder<R: IoRead>(reader: &mut R) -> CITLResult<Placeholder> {
    let name = read_string(reader)?;
    let desc = read_string(reader)?;
    let mut constraint_byte = [0u8; 1];
    reader.read_exact(&mut constraint_byte)?;
    let constraint = parse_placeholder_constraint(constraint_byte[0]);
    Ok(Placeholder::new(&name, &desc, constraint))
}

/// Read a vector of placeholders from a reader.
fn read_placeholders<R: IoRead>(reader: &mut R) -> CITLResult<Vec<Placeholder>> {
    let mut ph_count_bytes = [0u8; 2];
    reader.read_exact(&mut ph_count_bytes)?;
    let ph_count = u16::from_le_bytes(ph_count_bytes) as usize;

    let mut placeholders = Vec::with_capacity(ph_count);
    for _ in 0..ph_count {
        placeholders.push(read_placeholder(reader)?);
    }
    Ok(placeholders)
}

/// Read a vector of strings from a reader.
fn read_string_vec<R: IoRead>(reader: &mut R) -> CITLResult<Vec<String>> {
    let mut count_bytes = [0u8; 2];
    reader.read_exact(&mut count_bytes)?;
    let count = u16::from_le_bytes(count_bytes) as usize;

    let mut strings = Vec::with_capacity(count);
    for _ in 0..count {
        strings.push(read_string(reader)?);
    }
    Ok(strings)
}

/// Read a pattern and its embedding from a reader.
fn read_pattern<R: IoRead>(reader: &mut R) -> CITLResult<(ErrorFixPattern, Vec<f32>)> {
    let error_code = read_error_code(reader)?;

    // Context hash
    let mut hash_bytes = [0u8; 8];
    reader.read_exact(&mut hash_bytes)?;
    let context_hash = u64::from_le_bytes(hash_bytes);

    let (success_count, failure_count) = read_counts(reader)?;
    let fix_template = read_fix_template(reader)?;
    let embedding = read_embedding(reader)?;

    let pattern = ErrorFixPattern {
        error_code,
        context_hash,
        fix_template,
        success_count,
        failure_count,
    };

    Ok((pattern, embedding))
}

/// Write a fix template to a writer.
fn write_fix_template<W: IoWrite>(writer: &mut W, template: &FixTemplate) -> CITLResult<()> {
    write_string(writer, &template.pattern)?;
    write_string(writer, &template.description)?;
    writer.write_all(&template.confidence.to_le_bytes())?;

    // Placeholders
    let placeholder_count = template.placeholders.len() as u16;
    writer.write_all(&placeholder_count.to_le_bytes())?;
    for ph in &template.placeholders {
        write_string(writer, &ph.name)?;
        write_string(writer, &ph.description)?;
        writer.write_all(&[ph.constraint as u8])?;
    }

    // Applicable codes
    let codes_count = template.applicable_codes.len() as u16;
    writer.write_all(&codes_count.to_le_bytes())?;
    for code in &template.applicable_codes {
        write_string(writer, code)?;
    }

    Ok(())
}

/// Read a fix template from a reader.
fn read_fix_template<R: IoRead>(reader: &mut R) -> CITLResult<FixTemplate> {
    let pattern = read_string(reader)?;
    let description = read_string(reader)?;

    let mut confidence_bytes = [0u8; 4];
    reader.read_exact(&mut confidence_bytes)?;
    let confidence = f32::from_le_bytes(confidence_bytes);

    let placeholders = read_placeholders(reader)?;
    let applicable_codes = read_string_vec(reader)?;

    Ok(FixTemplate {
        pattern,
        placeholders,
        applicable_codes,
        confidence,
        description,
    })
}

/// Write a length-prefixed string.
fn write_string<W: IoWrite>(writer: &mut W, s: &str) -> CITLResult<()> {
    let bytes = s.as_bytes();
    let len = bytes.len() as u16;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(bytes)?;
    Ok(())
}

/// Read a length-prefixed string.
fn read_string<R: IoRead>(reader: &mut R) -> CITLResult<String> {
    let mut len_bytes = [0u8; 2];
    reader.read_exact(&mut len_bytes)?;
    let len = u16::from_le_bytes(len_bytes) as usize;

    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;

    String::from_utf8(buf).map_err(|_| CITLError::PatternLibraryError {
        message: "Invalid UTF-8 string in pattern file".to_string(),
    })
}

// ==================== Common Fix Templates ====================

/// Pre-defined fix templates for common Rust errors.
///
/// These templates are used by the fix generator to suggest fixes.
#[allow(dead_code)]
pub(super) mod templates {
    use super::{FixTemplate, Placeholder};

    /// Template: Add .`to_string()` for String/&str conversion
    #[must_use]
    pub(crate) fn to_string_conversion() -> FixTemplate {
        FixTemplate::new("$expr.to_string()", "Convert &str to String")
            .with_placeholder(Placeholder::expression("expr"))
            .with_code("E0308")
            .with_confidence(0.9)
    }

    /// Template: Add .`as_str()` for String/&str conversion
    #[must_use]
    pub(crate) fn as_str_conversion() -> FixTemplate {
        FixTemplate::new("$expr.as_str()", "Convert String to &str")
            .with_placeholder(Placeholder::expression("expr"))
            .with_code("E0308")
            .with_confidence(0.85)
    }

    /// Template: Add .`clone()` for ownership
    #[must_use]
    pub(crate) fn clone_value() -> FixTemplate {
        FixTemplate::new("$expr.clone()", "Clone the value to avoid move")
            .with_placeholder(Placeholder::expression("expr"))
            .with_code("E0382")
            .with_confidence(0.8)
    }

    /// Template: Add & for reference
    #[must_use]
    pub(crate) fn add_reference() -> FixTemplate {
        FixTemplate::new("&$expr", "Add reference")
            .with_placeholder(Placeholder::expression("expr"))
            .with_code("E0308")
            .with_confidence(0.7)
    }

    /// Template: Add &mut for mutable reference
    #[must_use]
    pub(crate) fn add_mut_reference() -> FixTemplate {
        FixTemplate::new("&mut $expr", "Add mutable reference")
            .with_placeholder(Placeholder::expression("expr"))
            .with_code("E0308")
            .with_confidence(0.7)
    }

    /// Template: Dereference
    #[must_use]
    pub(crate) fn dereference() -> FixTemplate {
        FixTemplate::new("*$expr", "Dereference")
            .with_placeholder(Placeholder::expression("expr"))
            .with_code("E0308")
            .with_confidence(0.6)
    }

    /// Template: Add .`into()` for type conversion
    #[must_use]
    pub(crate) fn into_conversion() -> FixTemplate {
        FixTemplate::new("$expr.into()", "Use Into trait for conversion")
            .with_placeholder(Placeholder::expression("expr"))
            .with_code("E0308")
            .with_confidence(0.75)
    }

    /// Template: `Vec::new()` for empty vector
    #[must_use]
    pub(crate) fn vec_new() -> FixTemplate {
        FixTemplate::new("Vec::new()", "Create empty Vec")
            .with_code("E0308")
            .with_confidence(0.8)
    }

    /// Template: `String::new()` for empty string
    #[must_use]
    pub(crate) fn string_new() -> FixTemplate {
        FixTemplate::new("String::new()", "Create empty String")
            .with_code("E0308")
            .with_confidence(0.8)
    }

    /// Template: `Option::unwrap_or_default()`
    #[must_use]
    pub(crate) fn unwrap_or_default() -> FixTemplate {
        FixTemplate::new(
            "$expr.unwrap_or_default()",
            "Unwrap Option with default value",
        )
        .with_placeholder(Placeholder::expression("expr"))
        .with_code("E0308")
        .with_confidence(0.7)
    }

    // ==================== E0382 Templates (Use of Moved Value) ====================

    /// Template: Borrow instead of move
    #[must_use]
    pub(crate) fn borrow_instead_of_move() -> FixTemplate {
        FixTemplate::new("&$expr", "Borrow instead of moving the value")
            .with_placeholder(Placeholder::expression("expr"))
            .with_code("E0382")
            .with_confidence(0.85)
    }

    /// Template: Wrap in Rc for shared ownership
    #[must_use]
    pub(crate) fn rc_wrap() -> FixTemplate {
        FixTemplate::new("Rc::new($expr)", "Wrap in Rc for shared ownership")
            .with_placeholder(Placeholder::expression("expr"))
            .with_code("E0382")
            .with_confidence(0.7)
    }

    /// Template: Wrap in Arc for thread-safe shared ownership
    #[must_use]
    pub(crate) fn arc_wrap() -> FixTemplate {
        FixTemplate::new(
            "Arc::new($expr)",
            "Wrap in Arc for thread-safe shared ownership",
        )
        .with_placeholder(Placeholder::expression("expr"))
        .with_code("E0382")
        .with_confidence(0.65)
    }

    // ==================== E0277 Templates (Trait Bound Not Satisfied) ====================

    /// Template: Derive Debug trait
    #[must_use]
    pub(crate) fn derive_debug() -> FixTemplate {
        FixTemplate::new("#[derive(Debug)]", "Add Debug derive to type")
            .with_code("E0277")
            .with_confidence(0.9)
    }

    /// Template: Derive Clone trait
    #[must_use]
    pub(crate) fn derive_clone_trait() -> FixTemplate {
        FixTemplate::new("#[derive(Clone)]", "Add Clone derive to type")
            .with_code("E0277")
            .with_confidence(0.85)
    }

    /// Template: Implement Display trait
    #[must_use]
    pub(crate) fn impl_display() -> FixTemplate {
        FixTemplate::new(
            "impl std::fmt::Display for $type { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, \"{}\", self.$field) } }",
            "Implement Display trait"
        )
        .with_placeholder(Placeholder::type_name("type"))
        .with_placeholder(Placeholder::identifier("field"))
        .with_code("E0277")
        .with_confidence(0.7)
    }

    /// Template: Implement From trait
    #[must_use]
    pub(crate) fn impl_from() -> FixTemplate {
        FixTemplate::new(
            "impl From<$source> for $target { fn from(value: $source) -> Self { Self($value) } }",
            "Implement From trait for type conversion",
        )
        .with_placeholder(Placeholder::type_name("source"))
        .with_placeholder(Placeholder::type_name("target"))
        .with_placeholder(Placeholder::expression("value"))
        .with_code("E0277")
        .with_confidence(0.65)
    }

    // ==================== E0515 Templates (Cannot Return Reference to Local) ====================

    /// Template: Return owned value instead of reference
    #[must_use]
    pub(crate) fn return_owned() -> FixTemplate {
        FixTemplate::new("$expr", "Return owned value instead of reference")
            .with_placeholder(Placeholder::expression("expr"))
            .with_code("E0515")
            .with_confidence(0.8)
    }

    /// Template: Return a clone
    #[must_use]
    pub(crate) fn return_cloned() -> FixTemplate {
        FixTemplate::new("$expr.clone()", "Return a clone instead of reference")
            .with_placeholder(Placeholder::expression("expr"))
            .with_code("E0515")
            .with_confidence(0.75)
    }

    /// Template: Use Cow for efficient return
    #[must_use]
    pub(crate) fn use_cow() -> FixTemplate {
        FixTemplate::new(
            "Cow::Owned($expr)",
            "Use Cow for efficient owned/borrowed return",
        )
        .with_placeholder(Placeholder::expression("expr"))
        .with_code("E0515")
        .with_confidence(0.7)
    }

    /// Get all standard templates.
    #[must_use]
    pub(crate) fn all_templates() -> Vec<FixTemplate> {
        vec![
            // E0308 (Type Mismatch)
            to_string_conversion(),
            as_str_conversion(),
            add_reference(),
            add_mut_reference(),
            dereference(),
            into_conversion(),
            vec_new(),
            string_new(),
            unwrap_or_default(),
            // E0382 (Use of Moved Value)
            clone_value(),
            borrow_instead_of_move(),
            rc_wrap(),
            arc_wrap(),
            // E0277 (Trait Bound Not Satisfied)
            derive_debug(),
            derive_clone_trait(),
            impl_display(),
            impl_from(),
            // E0515 (Cannot Return Reference to Local)
            return_owned(),
            return_cloned(),
            use_cow(),
        ]
    }
}


#[cfg(test)]
mod tests;
