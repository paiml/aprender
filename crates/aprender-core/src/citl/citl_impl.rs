
impl CITL {
    /// Create a new CITL builder.
    #[must_use]
    pub fn builder() -> CITLBuilder {
        CITLBuilder::new()
    }

    /// Compile source code and return structured result.
    ///
    /// # Errors
    ///
    /// Returns error if compilation process fails.
    pub fn compile(&self, source: &str) -> CITLResult<CompilationResult> {
        self.compiler.compile(source, &CompileOptions::default())
    }

    /// Encode a diagnostic into an embedding for pattern matching.
    #[must_use]
    pub fn encode_error(&self, diagnostic: &CompilerDiagnostic, source: &str) -> ErrorEmbedding {
        self.encoder.encode(diagnostic, source)
    }

    /// Search for similar error patterns.
    #[must_use]
    pub fn search_patterns(&self, embedding: &ErrorEmbedding, k: usize) -> Vec<PatternMatch> {
        self.pattern_library.search(embedding, k)
    }

    /// Add a successful fix to the pattern library (self-training).
    pub fn add_pattern(&mut self, error: ErrorEmbedding, fix: FixTemplate, success: bool) {
        if self.config.enable_self_training && success {
            self.pattern_library.add_pattern(error, fix);
        }
    }

    /// Get the compiler interface.
    #[must_use]
    pub fn compiler(&self) -> &dyn CompilerInterface {
        self.compiler.as_ref()
    }

    /// Suggest a fix for a compiler diagnostic.
    ///
    /// Encodes the error and searches the pattern library for similar fixes.
    /// Returns the best matching fix above the confidence threshold.
    ///
    /// # Arguments
    /// * `diagnostic` - The compiler diagnostic to fix
    /// * `source` - The source code containing the error
    ///
    /// # Returns
    /// `Some(SuggestedFix)` if a fix is found above confidence threshold, `None` otherwise.
    #[must_use]
    pub fn suggest_fix(
        &self,
        diagnostic: &CompilerDiagnostic,
        source: &str,
    ) -> Option<SuggestedFix> {
        // Encode the error
        let embedding = self.encoder.encode(diagnostic, source);

        // Search for similar patterns
        let matches = self.pattern_library.search(&embedding, 5);

        // Find best match above confidence threshold
        for m in matches {
            if m.similarity >= self.config.confidence_threshold {
                let replacement =
                    self.instantiate_template(&m.pattern.fix_template, diagnostic, source);
                return Some(SuggestedFix {
                    replacement,
                    confidence: m.similarity,
                    description: m.pattern.fix_template.description.clone(),
                    start_offset: diagnostic.span.column_start.saturating_sub(1),
                    end_offset: diagnostic.span.column_end,
                    error_code: diagnostic.code.code.clone(),
                });
            }
        }

        None
    }

    /// Apply a suggested fix to source code.
    ///
    /// # Arguments
    /// * `source` - The original source code
    /// * `fix` - The suggested fix to apply
    ///
    /// # Returns
    /// The modified source code with the fix applied.
    #[must_use]
    pub fn apply_fix(&self, source: &str, fix: &SuggestedFix) -> String {
        if fix.start_offset >= source.len() || fix.end_offset > source.len() {
            return source.to_string();
        }

        let mut result = String::with_capacity(source.len() + fix.replacement.len());
        result.push_str(&source[..fix.start_offset]);
        result.push_str(&fix.replacement);
        result.push_str(&source[fix.end_offset..]);
        result
    }

    /// Iteratively fix all errors in source code.
    ///
    /// Implements the CITL feedback loop:
    /// 1. Compile source code
    /// 2. If errors, suggest and apply fixes
    /// 3. Repeat until success or max iterations
    ///
    /// # Arguments
    /// * `source` - The source code to fix
    ///
    /// # Returns
    /// `FixResult` containing the final state after fixing attempts.
    pub fn fix_all(&mut self, source: &str) -> FixResult {
        let mut current_source = source.to_string();
        let mut iterations = 0;
        let mut applied_fixes = Vec::new();

        // First check if code already compiles
        match self.compile(&current_source) {
            Ok(CompilationResult::Success { .. }) => {
                return FixResult {
                    success: true,
                    fixed_source: Some(current_source),
                    iterations: 0,
                    remaining_errors: Vec::new(),
                    applied_fixes,
                };
            }
            Ok(CompilationResult::Failure { errors, .. }) if errors.is_empty() => {
                return FixResult {
                    success: true,
                    fixed_source: Some(current_source),
                    iterations: 0,
                    remaining_errors: Vec::new(),
                    applied_fixes,
                };
            }
            _ => {}
        }

        while iterations < self.config.max_iterations {
            iterations += 1;

            // Compile and get errors
            let Ok(result) = self.compile(&current_source) else {
                break;
            };

            match result {
                CompilationResult::Success { .. } => {
                    return FixResult {
                        success: true,
                        fixed_source: Some(current_source),
                        iterations,
                        remaining_errors: Vec::new(),
                        applied_fixes,
                    };
                }
                CompilationResult::Failure { errors, .. } => {
                    if errors.is_empty() {
                        return FixResult {
                            success: true,
                            fixed_source: Some(current_source),
                            iterations,
                            remaining_errors: Vec::new(),
                            applied_fixes,
                        };
                    }

                    // Try to fix the first error
                    let error = &errors[0];
                    if let Some(fix) = self.suggest_fix(error, &current_source) {
                        applied_fixes.push(fix.description.clone());
                        current_source = self.apply_fix(&current_source, &fix);
                    } else {
                        // No fix found, return failure
                        return FixResult {
                            success: false,
                            fixed_source: None,
                            iterations,
                            remaining_errors: errors.iter().map(|e| e.code.code.clone()).collect(),
                            applied_fixes,
                        };
                    }
                }
            }
        }

        // Max iterations reached
        let remaining = match self.compile(&current_source) {
            Ok(CompilationResult::Failure { errors, .. }) => {
                errors.iter().map(|e| e.code.code.clone()).collect()
            }
            _ => Vec::new(),
        };

        FixResult {
            success: false,
            fixed_source: None,
            iterations,
            remaining_errors: remaining,
            applied_fixes,
        }
    }

    /// Instantiate a fix template with concrete values.
    #[allow(clippy::unused_self)]
    fn instantiate_template(
        &self,
        template: &FixTemplate,
        diagnostic: &CompilerDiagnostic,
        _source: &str,
    ) -> String {
        let mut result = template.pattern.clone();

        // Simple placeholder replacement
        // $expr -> the expression at the error location
        // $type -> expected type
        // $found -> found type

        if let Some(expected) = &diagnostic.expected {
            result = result.replace("$type", &expected.to_string());
        }
        if let Some(found) = &diagnostic.found {
            result = result.replace("$found", &found.to_string());
        }

        // $expr is typically the problematic code - for now use a placeholder
        result = result.replace("$expr", "expr");

        result
    }
}

/// A suggested fix for a compiler error.
#[derive(Debug, Clone)]
pub struct SuggestedFix {
    /// The replacement text
    pub replacement: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Human-readable description
    pub description: String,
    /// Start offset in source (byte position)
    pub start_offset: usize,
    /// End offset in source (byte position)
    pub end_offset: usize,
    /// Error code this fix addresses
    pub error_code: String,
}

impl SuggestedFix {
    /// Create a new suggested fix.
    #[must_use]
    pub fn new(replacement: String, confidence: f32, description: String) -> Self {
        Self {
            replacement,
            confidence,
            description,
            start_offset: 0,
            end_offset: 0,
            error_code: String::new(),
        }
    }

    /// Set the span for replacement.
    #[must_use]
    pub fn with_span(mut self, start: usize, end: usize) -> Self {
        self.start_offset = start;
        self.end_offset = end;
        self
    }

    /// Set the error code.
    #[must_use]
    pub fn with_error_code(mut self, code: &str) -> Self {
        self.error_code = code.to_string();
        self
    }
}

/// Result of an iterative fix attempt.
#[derive(Debug, Clone)]
pub struct FixResult {
    /// Whether all errors were fixed
    pub success: bool,
    /// The fixed source code (if successful)
    pub fixed_source: Option<String>,
    /// Number of fix iterations performed
    pub iterations: usize,
    /// Error codes that remain unfixed
    pub remaining_errors: Vec<String>,
    /// Descriptions of fixes that were applied
    pub applied_fixes: Vec<String>,
}

impl FixResult {
    /// Create a successful fix result.
    #[must_use]
    pub fn success(source: String, iterations: usize) -> Self {
        Self {
            success: true,
            fixed_source: Some(source),
            iterations,
            remaining_errors: Vec::new(),
            applied_fixes: Vec::new(),
        }
    }

    /// Create a failed fix result.
    #[must_use]
    pub fn failure(iterations: usize, remaining: Vec<String>) -> Self {
        Self {
            success: false,
            fixed_source: None,
            iterations,
            remaining_errors: remaining,
            applied_fixes: Vec::new(),
        }
    }

    /// Check if the fix was successful.
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.success
    }

    /// Add an applied fix description.
    #[must_use]
    pub fn with_applied_fix(mut self, description: String) -> Self {
        self.applied_fixes.push(description);
        self
    }
}

/// Known Rust error codes with categories and difficulties.
///
/// Based on real-world data from depyler oracle (N=2,100+ samples):
/// - E0308 (20.9%), E0599 (17.9%), E0433 (16.4%), E0432 (14.1%)
pub static RUST_ERROR_CODES: &[ErrorCode] = &[
    // Tier 1: Easy - Type System Basics (most common)
    ErrorCode {
        code: String::new(), // Will be initialized properly
        category: ErrorCategory::TypeMismatch,
        difficulty: Difficulty::Easy,
    },
];

/// Initialize the static error codes properly.
#[must_use]
pub fn rust_error_codes() -> HashMap<String, ErrorCode> {
    let mut codes = HashMap::new();

    // Tier 1: Easy
    codes.insert(
        "E0308".to_string(),
        ErrorCode::new("E0308", ErrorCategory::TypeMismatch, Difficulty::Easy),
    );
    codes.insert(
        "E0425".to_string(),
        ErrorCode::new("E0425", ErrorCategory::Unresolved, Difficulty::Easy),
    );
    codes.insert(
        "E0433".to_string(),
        ErrorCode::new("E0433", ErrorCategory::Unresolved, Difficulty::Easy),
    );
    codes.insert(
        "E0432".to_string(),
        ErrorCode::new("E0432", ErrorCategory::Import, Difficulty::Easy),
    );
    codes.insert(
        "E0412".to_string(),
        ErrorCode::new("E0412", ErrorCategory::Unresolved, Difficulty::Easy),
    );
    codes.insert(
        "E0599".to_string(),
        ErrorCode::new("E0599", ErrorCategory::MethodNotFound, Difficulty::Easy),
    );

    // Tier 2: Medium - Ownership & Borrowing
    codes.insert(
        "E0382".to_string(),
        ErrorCode::new("E0382", ErrorCategory::Ownership, Difficulty::Medium),
    );
    codes.insert(
        "E0502".to_string(),
        ErrorCode::new("E0502", ErrorCategory::Borrowing, Difficulty::Medium),
    );
    codes.insert(
        "E0499".to_string(),
        ErrorCode::new("E0499", ErrorCategory::Borrowing, Difficulty::Medium),
    );
    codes.insert(
        "E0596".to_string(),
        ErrorCode::new("E0596", ErrorCategory::Borrowing, Difficulty::Medium),
    );
    codes.insert(
        "E0507".to_string(),
        ErrorCode::new("E0507", ErrorCategory::Ownership, Difficulty::Medium),
    );
    codes.insert(
        "E0282".to_string(),
        ErrorCode::new("E0282", ErrorCategory::TypeInference, Difficulty::Medium),
    );

    // Tier 3: Hard - Lifetimes
    codes.insert(
        "E0597".to_string(),
        ErrorCode::new("E0597", ErrorCategory::Lifetime, Difficulty::Hard),
    );
    codes.insert(
        "E0621".to_string(),
        ErrorCode::new("E0621", ErrorCategory::Lifetime, Difficulty::Hard),
    );
    codes.insert(
        "E0106".to_string(),
        ErrorCode::new("E0106", ErrorCategory::Lifetime, Difficulty::Medium),
    );
    codes.insert(
        "E0495".to_string(),
        ErrorCode::new("E0495", ErrorCategory::Lifetime, Difficulty::Hard),
    );
    codes.insert(
        "E0623".to_string(),
        ErrorCode::new("E0623", ErrorCategory::Lifetime, Difficulty::Hard),
    );

    // Tier 4: Expert - Advanced
    codes.insert(
        "E0277".to_string(),
        ErrorCode::new("E0277", ErrorCategory::TraitBound, Difficulty::Hard),
    );
    codes.insert(
        "E0373".to_string(),
        ErrorCode::new("E0373", ErrorCategory::Async, Difficulty::Expert),
    );

    codes
}

#[cfg(test)]
mod tests;
