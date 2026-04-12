//! Oracle definitions for output verification
//!
//! Oracles are pure functions that verify model output correctness.
//! Each oracle implements Popperian falsification - it attempts to
//! disprove the hypothesis that the model output is correct.
//!
//! # Design
//!
//! An oracle returns `OracleResult::Corroborated` when it fails to
//! disprove correctness, and `OracleResult::Falsified` when it
//! successfully disproves the hypothesis.

use serde::{Deserialize, Serialize};

/// Result of oracle evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OracleResult {
    /// Hypothesis not falsified - output appears correct
    Corroborated {
        /// Evidence supporting corroboration
        evidence: String,
    },
    /// Hypothesis falsified - output is incorrect
    Falsified {
        /// Reason for falsification
        reason: String,
        /// Evidence of failure
        evidence: String,
    },
}

impl OracleResult {
    /// Check if the result is corroborated
    #[must_use]
    pub const fn is_corroborated(&self) -> bool {
        matches!(self, Self::Corroborated { .. })
    }

    /// Check if the result is falsified
    #[must_use]
    pub const fn is_falsified(&self) -> bool {
        matches!(self, Self::Falsified { .. })
    }
}

/// Oracle trait for output verification
pub trait Oracle: Send + Sync {
    /// Evaluate the output against the prompt
    fn evaluate(&self, prompt: &str, output: &str) -> OracleResult;

    /// Get the oracle name
    fn name(&self) -> &'static str;
}

/// Arithmetic oracle - verifies mathematical correctness
#[derive(Debug, Clone, Default)]
pub struct ArithmeticOracle;

impl ArithmeticOracle {
    /// Create a new arithmetic oracle
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Try to parse and evaluate a simple arithmetic expression.
    ///
    /// Handles both raw expressions ("2+2") and natural language ("What is 2+2?").
    /// For natural language, extracts the first `number operator number` pattern.
    fn eval_arithmetic(expr: &str) -> Option<i64> {
        let cleaned = expr.trim().trim_end_matches('=').trim_end_matches('?');

        // Try direct parse first (raw expressions like "2+2", "15-7")
        if let Some(result) = Self::try_eval_simple(cleaned) {
            return Some(result);
        }

        // Extract arithmetic expression from natural language.
        // Scan for the pattern: digits [+-*/] digits
        Self::extract_and_eval(cleaned)
    }

    /// Evaluate a cleaned expression with no natural language.
    fn try_eval_simple(expr: &str) -> Option<i64> {
        // Find the FIRST operator by string position, not by operator priority.
        // Skip position 0 for '-' to handle negative numbers like "-5+3".
        let first_op = ['+', '-', '*', '/']
            .iter()
            .filter_map(|&op| {
                expr.find(op).and_then(|pos| {
                    if pos == 0 && op == '-' {
                        expr[1..].find(op).map(|p| (p + 1, op))
                    } else {
                        Some((pos, op))
                    }
                })
            })
            .min_by_key(|&(pos, _)| pos);

        if let Some((pos, op)) = first_op {
            let left: i64 = expr[..pos].trim().parse().ok()?;
            let right: i64 = expr[pos + 1..].trim().parse().ok()?;
            return match op {
                '+' => left.checked_add(right),
                '-' => left.checked_sub(right),
                '*' => left.checked_mul(right),
                '/' if right != 0 => left.checked_div(right),
                _ => None,
            };
        }
        None
    }

    /// Extract a `number op number` pattern from natural language text.
    ///
    /// Scans for the first occurrence of `\d+\s*[+\-*/]\s*\d+` and evaluates it.
    /// Handles: "What is 2+2?", "Calculate 7*8", "What is 15-7?", "100/4"
    fn extract_and_eval(text: &str) -> Option<i64> {
        let bytes = text.as_bytes();
        let len = bytes.len();
        let mut i = 0;

        while i < len {
            // Find start of a number
            if !bytes[i].is_ascii_digit() {
                i += 1;
                continue;
            }

            // Consume digits (left operand)
            let left_start = i;
            while i < len && bytes[i].is_ascii_digit() {
                i += 1;
            }
            let left_end = i;

            // Skip optional whitespace
            while i < len && bytes[i] == b' ' {
                i += 1;
            }

            // Check for operator
            if i >= len || !matches!(bytes[i], b'+' | b'-' | b'*' | b'/') {
                continue;
            }
            let op = bytes[i] as char;
            i += 1;

            // Skip optional whitespace
            while i < len && bytes[i] == b' ' {
                i += 1;
            }

            // Consume digits (right operand)
            if i >= len || !bytes[i].is_ascii_digit() {
                continue;
            }
            let right_start = i;
            while i < len && bytes[i].is_ascii_digit() {
                i += 1;
            }

            // Parse and evaluate
            let left: i64 = text[left_start..left_end].parse().ok()?;
            let right: i64 = text[right_start..i].parse().ok()?;
            return match op {
                '+' => left.checked_add(right),
                '-' => left.checked_sub(right),
                '*' => left.checked_mul(right),
                '/' if right != 0 => left.checked_div(right),
                _ => None,
            };
        }
        None
    }
}

impl Oracle for ArithmeticOracle {
    /// Evaluate arithmetic correctness by checking if output contains expected value
    fn evaluate(&self, prompt: &str, output: &str) -> OracleResult {
        // Try to extract arithmetic expression from prompt
        let Some(expected) = Self::eval_arithmetic(prompt) else {
            // Distinguish two failure modes:
            // 1. Raw expression that is mathematically unevaluable (e.g. "5/0=",
            //    overflow) → Falsified. The hypothesis was never tested; Popperian:
            //    absence of test ≠ pass.
            // 2. Natural language with embedded arithmetic (e.g. "What is 2+2?")
            //    → Corroborated (skip). This is a parser limitation, not a math
            //    impossibility. The oracle can't extract the expression.
            if is_raw_arithmetic_expr(prompt) {
                return OracleResult::Falsified {
                    reason: format!("Arithmetic prompt cannot be evaluated: {prompt}"),
                    evidence: output.to_string(),
                };
            }
            return OracleResult::Corroborated {
                evidence: "Non-evaluable arithmetic prompt, skipped".to_string(),
            };
        };

        // Check if output contains the expected value as a word boundary
        // (not just substring — "4" matching "42" is a false positive)
        let expected_str = expected.to_string();
        let found = output
            .split(|c: char| !c.is_ascii_digit() && c != '-')
            .any(|word| word == expected_str);
        if found {
            OracleResult::Corroborated {
                evidence: format!("Found expected value {expected} in output"),
            }
        } else {
            OracleResult::Falsified {
                reason: format!("Expected {expected} not found in output"),
                evidence: format!("Output: {}", truncate(output, 100)),
            }
        }
    }

    /// Return the oracle identifier
    fn name(&self) -> &'static str {
        "arithmetic"
    }
}

/// Garbage detection oracle - verifies output is not garbage
#[derive(Debug, Clone, Default)]
pub struct GarbageOracle;

impl GarbageOracle {
    /// Create a new garbage oracle
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Oracle for GarbageOracle {
    /// Check output for garbage patterns including empty, control chars, NaN, and repetition
    fn evaluate(&self, _prompt: &str, output: &str) -> OracleResult {
        // Check for empty output
        if output.trim().is_empty() {
            return OracleResult::Falsified {
                reason: "Output is empty".to_string(),
                evidence: "Empty output".to_string(),
            };
        }

        // Check for control characters (except newline, tab)
        let control_chars: Vec<char> = output
            .chars()
            .filter(|c| c.is_control() && *c != '\n' && *c != '\t' && *c != '\r')
            .collect();
        if !control_chars.is_empty() {
            return OracleResult::Falsified {
                reason: "Output contains control characters".to_string(),
                evidence: format!("Found {} control chars", control_chars.len()),
            };
        }

        // Check for NaN/Inf (numerical explosion)
        // Use word-boundary matching to avoid false positives on "information", "Infinity", etc.
        if has_nan_or_inf(output) {
            return OracleResult::Falsified {
                reason: "Output contains NaN or Inf".to_string(),
                evidence: format!("Output: {}", truncate(output, 100)),
            };
        }

        // Check for repetitive patterns (e.g., "akakakakak")
        if is_repetitive(output) {
            return OracleResult::Falsified {
                reason: "Output is highly repetitive".to_string(),
                evidence: format!("Output: {}", truncate(output, 100)),
            };
        }

        // Check for replacement character (encoding issues)
        if output.contains('\u{FFFD}') {
            return OracleResult::Falsified {
                reason: "Output contains replacement characters".to_string(),
                evidence: "Found U+FFFD replacement character".to_string(),
            };
        }

        OracleResult::Corroborated {
            evidence: format!("Valid output ({} chars)", output.len()),
        }
    }

    /// Return the oracle identifier
    fn name(&self) -> &'static str {
        "garbage"
    }
}

/// Code syntax oracle - verifies output looks like code
#[derive(Debug, Clone, Default)]
pub struct CodeSyntaxOracle;

impl CodeSyntaxOracle {
    /// Create a new code syntax oracle
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Oracle for CodeSyntaxOracle {
    /// Verify output contains code-like patterns after garbage check
    #[allow(clippy::used_underscore_binding)]
    fn evaluate(&self, _prompt: &str, output: &str) -> OracleResult {
        // First check for garbage
        let garbage_oracle = GarbageOracle::new();
        if let OracleResult::Falsified { reason, evidence } =
            garbage_oracle.evaluate(_prompt, output)
        {
            return OracleResult::Falsified { reason, evidence };
        }

        // Check for code-like patterns
        let code_indicators = [
            "fn ",
            "def ",
            "function ",
            "class ",
            "struct ",
            "impl ",
            "pub ",
            "let ",
            "const ",
            "var ",
            "if ",
            "for ",
            "while ",
            "return ",
            "import ",
            "from ",
            "use ",
            "{",
            "}",
            "(",
            ")",
            ";",
            "=>",
            "->",
        ];

        let has_code_pattern = code_indicators.iter().any(|p| output.contains(p));

        // Very short output might just be a completion of a function signature
        if has_code_pattern || output.len() < 20 {
            OracleResult::Corroborated {
                evidence: "Output appears to be valid code".to_string(),
            }
        } else {
            // Output has no code-like patterns and is long enough to expect them.
            // Popperian: this falsifies the hypothesis "model generates code".
            OracleResult::Falsified {
                reason: "Output does not contain code-like patterns".to_string(),
                evidence: format!("Output ({} chars): {}", output.len(), truncate(output, 100)),
            }
        }
    }

    /// Return the oracle identifier
    fn name(&self) -> &'static str {
        "code_syntax"
    }
}

/// Combined oracle that runs multiple oracles
pub struct CompositeOracle {
    /// Oracle display name
    name: &'static str,
    /// Child oracles evaluated in order
    oracles: Vec<Box<dyn Oracle + Send + Sync>>,
}

impl std::fmt::Debug for CompositeOracle {
    /// Format the composite oracle showing name and child count
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompositeOracle")
            .field("name", &self.name)
            .field("oracle_count", &self.oracles.len())
            .finish()
    }
}

// Manual Clone implementation since Box<dyn Oracle> doesn't implement Clone
impl CompositeOracle {
    /// Create a new composite oracle
    #[must_use]
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            oracles: Vec::new(),
        }
    }

    /// Add an oracle to the composite
    pub fn add<O: Oracle + Clone + 'static>(&mut self, oracle: O) {
        self.oracles.push(Box::new(oracle));
    }
}

// We need a wrapper to make the oracles cloneable
/// Wrapper to enable cloning of boxed oracle trait objects
#[allow(dead_code)]
struct OracleWrapper<O: Oracle + Clone>(O);

impl<O: Oracle + Clone> Oracle for OracleWrapper<O> {
    /// Delegate evaluation to the wrapped oracle
    fn evaluate(&self, prompt: &str, output: &str) -> OracleResult {
        self.0.evaluate(prompt, output)
    }

    /// Return the wrapped oracle's name
    fn name(&self) -> &'static str {
        self.0.name()
    }
}

impl Oracle for CompositeOracle {
    /// Evaluate all child oracles, returning first falsification or overall corroboration
    fn evaluate(&self, prompt: &str, output: &str) -> OracleResult {
        for oracle in &self.oracles {
            if let result @ OracleResult::Falsified { .. } = oracle.evaluate(prompt, output) {
                return result;
            }
        }
        OracleResult::Corroborated {
            evidence: format!("All {} oracles passed", self.oracles.len()),
        }
    }

    /// Return the composite oracle's name
    fn name(&self) -> &'static str {
        self.name
    }
}

/// Select the appropriate oracle based on prompt characteristics
#[must_use]
pub fn select_oracle(prompt: &str) -> Box<dyn Oracle + Send + Sync> {
    if is_arithmetic_prompt(prompt) {
        Box::new(ArithmeticOracle::new())
    } else if is_code_prompt(prompt) {
        Box::new(CodeSyntaxOracle::new())
    } else {
        Box::new(GarbageOracle::new())
    }
}

/// Check if prompt is an arithmetic question
fn is_arithmetic_prompt(prompt: &str) -> bool {
    let prompt_lower = prompt.to_lowercase();
    (prompt_lower.contains('+')
        || prompt_lower.contains('-')
        || prompt_lower.contains('*')
        || prompt_lower.contains('/'))
        && prompt.chars().any(|c| c.is_ascii_digit())
}

/// Check if prompt is a raw arithmetic expression (digits, operators, whitespace, `=`, `?`).
///
/// Returns true for "5/0=", "2+2", "100*3?", "-5+3".
/// Returns false for "What is 2+2?" (contains letters beyond the expression).
fn is_raw_arithmetic_expr(prompt: &str) -> bool {
    let trimmed = prompt.trim();
    if trimmed.is_empty() {
        return false;
    }
    trimmed
        .chars()
        .all(|c| c.is_ascii_digit() || "+-*/=? ".contains(c))
        && trimmed.chars().any(|c| "+-*/".contains(c))
        && trimmed.chars().any(|c| c.is_ascii_digit())
}

/// Check if prompt is a code completion request
fn is_code_prompt(prompt: &str) -> bool {
    let lower = prompt.to_lowercase();
    lower.starts_with("def ")
        || lower.starts_with("fn ")
        || lower.starts_with("function ")
        || lower.starts_with("class ")
        || lower.starts_with("async ")
        || prompt.contains("```")
}

/// Check if output contains NaN or Inf as standalone tokens (word-boundary aware).
///
/// Avoids false positives on common words like "information", "Infinity",
/// "Infrastructure", etc. Matches: "NaN", "nan", "Inf", "inf", "-Inf", "-inf".
fn has_nan_or_inf(output: &str) -> bool {
    let tokens = [
        "NaN", "nan", "NAN", "Inf", "inf", "-Inf", "-inf", "+Inf", "+inf",
    ];
    output
        .split(|c: char| {
            c.is_whitespace()
                || matches!(
                    c,
                    ',' | ';'
                        | ':'
                        | '['
                        | ']'
                        | '('
                        | ')'
                        | '{'
                        | '}'
                        | '='
                        | '"'
                        | '\''
                        | '<'
                        | '>'
                        | '|'
                        | '/'
                )
        })
        .any(|word| tokens.contains(&word))
}

/// Check if a string contains a repeating substring pattern
///
/// For each candidate period `p` in `[2, min(20, len/3)]`, extracts the first
/// `p` bytes as a pattern and counts consecutive repetitions from the start.
/// Returns true if reps >= 3 AND coverage >= 70% of the string length.
fn check_substring_repetition(s: &str) -> bool {
    let bytes = s.as_bytes();
    let len = bytes.len();
    if len < 6 {
        return false;
    }
    let max_period = 20.min(len / 3);
    for p in 2..=max_period {
        let pattern = &bytes[..p];
        let mut reps = 1;
        let mut pos = p;
        while pos + p <= len && &bytes[pos..pos + p] == pattern {
            reps += 1;
            pos += p;
        }
        if reps >= 3 && (reps * p) * 100 >= len * 70 {
            return true;
        }
    }
    false
}

/// Check if output has character-level n-gram repetition
///
/// Checks the full output string and each individual word (for words
/// with length >= 6) to catch patterns like "foo VILLEVILLEVILLE bar".
fn has_char_ngram_repetition(output: &str) -> bool {
    if check_substring_repetition(output) {
        return true;
    }
    output
        .split_whitespace()
        .any(|word| word.len() >= 6 && check_substring_repetition(word))
}

/// Check if words contain a 2-word repeating pattern starting at any offset.
///
/// Tries each starting bigram position to catch repetition that begins
/// mid-output (e.g., "Normal start foo bar foo bar foo bar").
/// Returns true if any bigram repeats >= 3 consecutive times.
fn has_two_word_repetition(words: &[&str]) -> bool {
    if words.len() < 6 {
        return false;
    }
    // Try each starting position for a 2-word pattern
    let max_start = words.len().saturating_sub(5); // need at least 3 reps = 6 words
    for start in 0..=max_start {
        let p0 = words[start];
        let p1 = words[start + 1];
        let mut reps = 1;
        let mut pos = start + 2;
        while pos + 1 < words.len() && words[pos] == p0 && words[pos + 1] == p1 {
            reps += 1;
            pos += 2;
        }
        if reps >= 3 {
            return true;
        }
    }
    false
}

/// Check if all words in a slice are identical
fn all_words_identical(words: &[&str]) -> bool {
    let first = words.first();
    first.is_some() && words.iter().all(|w| Some(w) == first)
}

/// Check if output is highly repetitive
fn is_repetitive(output: &str) -> bool {
    // Character-level n-gram check catches patterns like "VILLEVILLEVILLE"
    // that word-level checks miss (single continuous token, no whitespace).
    if has_char_ngram_repetition(output) {
        return true;
    }

    let words: Vec<&str> = output.split_whitespace().collect();
    if words.len() < 5 {
        return false;
    }

    all_words_identical(&words) || has_two_word_repetition(&words)
}

/// Truncate string for display (UTF-8 safe)
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        // Find last valid char boundary at or before max_len
        // floor_char_boundary is nightly-only, so scan manually
        let mut end = max_len;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &s[..end])
    }
}

#[cfg(test)]
#[path = "oracle_tests.rs"]
mod oracle_tests;
