//! Synthetic data augmentation for code
//!
//! Implements Easy Data Augmentation (EDA) techniques adapted for code,
//! based on Wei & Zou (2019).
//!
//! # Operations
//!
//! - **Synonym Replacement (SR)**: Rename variables consistently
//! - **Random Insertion (RI)**: Insert comments or pass statements
//! - **Random Swap (RS)**: Reorder independent statements
//! - **Random Deletion (RD)**: Remove dead code or redundant statements

use rand::prelude::*;
use std::collections::HashSet;

/// Configuration for code EDA augmentation
#[derive(Debug, Clone)]
pub struct CodeEDAConfig {
    /// Probability of synonym replacement (variable renaming)
    pub sr_prob: f32,
    /// Probability of random insertion (comments/pass)
    pub ri_prob: f32,
    /// Probability of random swap (statement reorder)
    pub rs_prob: f32,
    /// Probability of random deletion
    pub rd_prob: f32,
    /// Minimum quality score threshold
    pub quality_threshold: f32,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for CodeEDAConfig {
    fn default() -> Self {
        Self {
            sr_prob: 0.1,
            ri_prob: 0.1,
            rs_prob: 0.1,
            rd_prob: 0.05,
            quality_threshold: 0.75,
            seed: 42,
        }
    }
}

/// Easy Data Augmentation for code
#[derive(Debug)]
pub struct CodeEDA {
    config: CodeEDAConfig,
    rng: StdRng,
}

impl CodeEDA {
    /// Create a new CodeEDA augmenter with default config
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(CodeEDAConfig::default())
    }

    /// Create a new CodeEDA augmenter with custom config
    #[must_use]
    pub fn with_config(config: CodeEDAConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        Self { config, rng }
    }

    /// Generate augmented versions of the input code
    ///
    /// Returns a vector of augmented code strings that pass quality threshold.
    pub fn augment(&mut self, code: &str, n_augmentations: usize) -> Vec<String> {
        let mut results = Vec::with_capacity(n_augmentations);

        for _ in 0..n_augmentations * 2 {
            // Generate more to account for quality filtering
            let augmented = self.apply_augmentations(code);
            let quality = self.quality_score(&augmented, code);

            if quality >= self.config.quality_threshold {
                results.push(augmented);
                if results.len() >= n_augmentations {
                    break;
                }
            }
        }

        results
    }

    /// Apply all augmentation operations probabilistically
    fn apply_augmentations(&mut self, code: &str) -> String {
        let mut result = code.to_string();

        if self.rng.random::<f32>() < self.config.sr_prob {
            result = self.synonym_replacement(&result);
        }
        if self.rng.random::<f32>() < self.config.ri_prob {
            result = self.random_insertion(&result);
        }
        if self.rng.random::<f32>() < self.config.rs_prob {
            result = self.random_swap(&result);
        }
        if self.rng.random::<f32>() < self.config.rd_prob {
            result = self.random_deletion(&result);
        }

        result
    }

    /// Synonym Replacement: Rename variables consistently
    fn synonym_replacement(&mut self, code: &str) -> String {
        let variables = self.extract_variables(code);
        if variables.is_empty() {
            return code.to_string();
        }

        // Pick a random variable to rename (sorted for determinism)
        let mut var_list: Vec<_> = variables.into_iter().collect();
        var_list.sort();
        let idx = self.rng.random_range(0..var_list.len());
        let old_var = &var_list[idx];

        // Generate new name
        let new_var = self.generate_variable_name(old_var);

        // Replace all occurrences (simple word boundary replacement)
        self.replace_identifier(code, old_var, &new_var)
    }

    /// Random Insertion: Add comments or pass statements
    fn random_insertion(&mut self, code: &str) -> String {
        let lines: Vec<&str> = code.lines().collect();
        if lines.is_empty() {
            return code.to_string();
        }

        let insert_idx = self.rng.random_range(0..=lines.len());
        let insert_type = self.rng.random_range(0..3);

        let insertion = match insert_type {
            0 => "    # augmented".to_string(),
            1 => "    pass  # placeholder".to_string(),
            _ => format!("    # line {}", insert_idx + 1),
        };

        let mut result_lines: Vec<String> = lines.iter().map(|s| (*s).to_string()).collect();
        result_lines.insert(insert_idx, insertion);
        result_lines.join("\n")
    }

    /// Random Swap: Reorder independent statements
    fn random_swap(&mut self, code: &str) -> String {
        let lines: Vec<&str> = code.lines().collect();
        if lines.len() < 2 {
            return code.to_string();
        }

        // Find swappable pairs (same indentation, no dependencies)
        let swappable = self.find_swappable_pairs(&lines);
        if swappable.is_empty() {
            return code.to_string();
        }

        let (i, j) = swappable[self.rng.random_range(0..swappable.len())];
        let mut result_lines: Vec<String> = lines.iter().map(|s| (*s).to_string()).collect();
        result_lines.swap(i, j);
        result_lines.join("\n")
    }

    /// Random Deletion: Remove redundant statements
    fn random_deletion(&mut self, code: &str) -> String {
        let lines: Vec<&str> = code.lines().collect();
        if lines.len() <= 2 {
            return code.to_string();
        }

        // Only delete comments or pass statements
        let deletable: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, line)| {
                let trimmed = line.trim();
                trimmed.starts_with('#') || trimmed == "pass"
            })
            .map(|(i, _)| i)
            .collect();

        if deletable.is_empty() {
            return code.to_string();
        }

        let del_idx = deletable[self.rng.random_range(0..deletable.len())];
        let result_lines: Vec<&str> = lines
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != del_idx)
            .map(|(_, line)| *line)
            .collect();
        result_lines.join("\n")
    }

    /// Extract variable names from Python code (simple heuristic)
    fn extract_variables(&self, code: &str) -> HashSet<String> {
        let mut vars = HashSet::new();

        // Match assignment patterns: var_name = ...
        for line in code.lines() {
            let trimmed = line.trim();
            if let Some(eq_pos) = trimmed.find('=') {
                if eq_pos > 0 && !trimmed[..eq_pos].contains('(') {
                    let lhs = trimmed[..eq_pos].trim();
                    // Skip if it's a comparison (==, !=, etc.)
                    if !lhs.ends_with(['!', '<', '>', '=']) {
                        // Handle tuple unpacking
                        for var in lhs.split(',') {
                            let var = var.trim();
                            if is_valid_identifier(var) && !is_keyword(var) {
                                vars.insert(var.to_string());
                            }
                        }
                    }
                }
            }
        }

        vars
    }

    /// Generate a new variable name based on old one
    fn generate_variable_name(&mut self, old: &str) -> String {
        let suffixes = ["_new", "_v2", "_alt", "_mod", "2"];
        let suffix = suffixes[self.rng.random_range(0..suffixes.len())];
        format!("{old}{suffix}")
    }

    /// Replace identifier with word boundary awareness
    fn replace_identifier(&self, code: &str, old: &str, new: &str) -> String {
        let mut result = String::with_capacity(code.len() + 32);
        let chars: Vec<char> = code.chars().collect();
        let old_chars: Vec<char> = old.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            if i + old_chars.len() <= chars.len() {
                let matches = chars[i..i + old_chars.len()]
                    .iter()
                    .zip(old_chars.iter())
                    .all(|(a, b)| a == b);

                if matches {
                    // Check word boundaries
                    let before_ok =
                        i == 0 || !chars[i - 1].is_alphanumeric() && chars[i - 1] != '_';
                    let after_ok = i + old_chars.len() >= chars.len()
                        || !chars[i + old_chars.len()].is_alphanumeric()
                            && chars[i + old_chars.len()] != '_';

                    if before_ok && after_ok {
                        result.push_str(new);
                        i += old_chars.len();
                        continue;
                    }
                }
            }
            result.push(chars[i]);
            i += 1;
        }

        result
    }

    /// Find pairs of lines that can be safely swapped
    fn find_swappable_pairs(&self, lines: &[&str]) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();

        for i in 0..lines.len().saturating_sub(1) {
            let indent_i = lines[i].len() - lines[i].trim_start().len();
            let indent_j = lines[i + 1].len() - lines[i + 1].trim_start().len();

            // Same indentation, both are simple statements
            if indent_i == indent_j {
                let line_i = lines[i].trim();
                let line_j = lines[i + 1].trim();

                // Skip control flow, function defs, class defs
                let is_simple_i = !line_i.starts_with("if ")
                    && !line_i.starts_with("for ")
                    && !line_i.starts_with("while ")
                    && !line_i.starts_with("def ")
                    && !line_i.starts_with("class ")
                    && !line_i.starts_with("return")
                    && !line_i.is_empty();

                let is_simple_j = !line_j.starts_with("if ")
                    && !line_j.starts_with("for ")
                    && !line_j.starts_with("while ")
                    && !line_j.starts_with("def ")
                    && !line_j.starts_with("class ")
                    && !line_j.starts_with("return")
                    && !line_j.is_empty();

                if is_simple_i && is_simple_j {
                    pairs.push((i, i + 1));
                }
            }
        }

        pairs
    }

    /// Calculate quality score for augmented code
    ///
    /// Returns score in [0.0, 1.0] based on:
    /// - Syntactic validity (must parse)
    /// - Token overlap with original
    #[must_use]
    pub fn quality_score(&self, augmented: &str, original: &str) -> f32 {
        // Basic syntactic check: balanced parentheses, quotes
        if !self.basic_syntax_check(augmented) {
            return 0.0;
        }

        // Token overlap score
        let orig_tokens: HashSet<_> = tokenize(original).collect();
        let aug_tokens: HashSet<_> = tokenize(augmented).collect();

        if orig_tokens.is_empty() {
            return 1.0;
        }

        let overlap = orig_tokens.intersection(&aug_tokens).count();
        overlap as f32 / orig_tokens.len() as f32
    }

    /// Calculate diversity score for a batch of augmented code
    ///
    /// Returns score in [0.0, 1.0], higher means more diverse
    #[must_use]
    pub fn diversity_score(&self, batch: &[String]) -> f32 {
        if batch.is_empty() {
            return 0.0;
        }

        let unique: HashSet<_> = batch.iter().collect();
        unique.len() as f32 / batch.len() as f32
    }

    /// Basic syntax validation
    fn basic_syntax_check(&self, code: &str) -> bool {
        let mut paren_depth = 0i32;
        let mut bracket_depth = 0i32;
        let mut brace_depth = 0i32;
        let mut in_string = false;
        let mut string_char = ' ';

        for c in code.chars() {
            if in_string {
                if c == string_char {
                    in_string = false;
                }
                continue;
            }

            match c {
                '"' | '\'' => {
                    in_string = true;
                    string_char = c;
                }
                '(' => paren_depth += 1,
                ')' => paren_depth -= 1,
                '[' => bracket_depth += 1,
                ']' => bracket_depth -= 1,
                '{' => brace_depth += 1,
                '}' => brace_depth -= 1,
                _ => {}
            }

            if paren_depth < 0 || bracket_depth < 0 || brace_depth < 0 {
                return false;
            }
        }

        paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 && !in_string
    }
}

impl Default for CodeEDA {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple tokenizer for code
fn tokenize(code: &str) -> impl Iterator<Item = &str> {
    code.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
}

/// Check if string is a valid Python identifier
fn is_valid_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let Some(first) = s.chars().next() else {
        return false;
    };
    (first.is_alphabetic() || first == '_') && s.chars().all(|c| c.is_alphanumeric() || c == '_')
}

/// Check if string is a Python keyword
fn is_keyword(s: &str) -> bool {
    matches!(
        s,
        "False"
            | "None"
            | "True"
            | "and"
            | "as"
            | "assert"
            | "async"
            | "await"
            | "break"
            | "class"
            | "continue"
            | "def"
            | "del"
            | "elif"
            | "else"
            | "except"
            | "finally"
            | "for"
            | "from"
            | "global"
            | "if"
            | "import"
            | "in"
            | "is"
            | "lambda"
            | "nonlocal"
            | "not"
            | "or"
            | "pass"
            | "raise"
            | "return"
            | "try"
            | "while"
            | "with"
            | "yield"
    )
}

/// Batch augmentation result
#[derive(Debug, Clone)]
pub struct AugmentationResult {
    /// Original code
    pub original: String,
    /// Augmented variants
    pub variants: Vec<String>,
    /// Quality scores for each variant
    pub quality_scores: Vec<f32>,
    /// Overall diversity score
    pub diversity_score: f32,
}

/// Batch augmenter for processing multiple code samples
#[derive(Debug)]
pub struct BatchAugmenter {
    eda: CodeEDA,
    /// Augmentation factor (e.g., 5.0 = 5x more samples)
    pub factor: f32,
}

impl BatchAugmenter {
    /// Create a new batch augmenter
    #[must_use]
    pub fn new(config: CodeEDAConfig, factor: f32) -> Self {
        Self {
            eda: CodeEDA::with_config(config),
            factor,
        }
    }

    /// Augment a batch of code samples
    pub fn augment_batch(&mut self, samples: &[String]) -> Vec<AugmentationResult> {
        #[allow(clippy::cast_sign_loss)]
        let n_aug = (self.factor.max(0.0) as usize).max(1);

        samples
            .iter()
            .map(|code| {
                let variants = self.eda.augment(code, n_aug);
                let quality_scores: Vec<f32> = variants
                    .iter()
                    .map(|v| self.eda.quality_score(v, code))
                    .collect();
                let diversity_score = self.eda.diversity_score(&variants);

                AugmentationResult {
                    original: code.clone(),
                    variants,
                    quality_scores,
                    diversity_score,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_eda_basic() {
        let mut eda = CodeEDA::new();
        let code = "x = 1\ny = 2\nz = x + y";
        let augmented = eda.augment(code, 3);

        assert!(!augmented.is_empty());
        for aug in &augmented {
            let quality = eda.quality_score(aug, code);
            assert!(quality >= 0.75);
        }
    }

    #[test]
    fn test_synonym_replacement() {
        let mut eda = CodeEDA::with_config(CodeEDAConfig {
            sr_prob: 1.0,
            ri_prob: 0.0,
            rs_prob: 0.0,
            rd_prob: 0.0,
            ..Default::default()
        });

        let code = "foo = 1\nbar = foo + 2";
        let augmented = eda.augment(code, 1);

        assert!(!augmented.is_empty());
        // Should have renamed a variable
        let aug = &augmented[0];
        assert!(aug.contains("_new") || aug.contains("_v2") || aug.contains("2"));
    }

    #[test]
    fn test_random_insertion() {
        let mut eda = CodeEDA::with_config(CodeEDAConfig {
            sr_prob: 0.0,
            ri_prob: 1.0,
            rs_prob: 0.0,
            rd_prob: 0.0,
            ..Default::default()
        });

        let code = "x = 1";
        let augmented = eda.augment(code, 1);

        assert!(!augmented.is_empty());
        // Should have added a line
        assert!(augmented[0].lines().count() > code.lines().count());
    }

    #[test]
    fn test_quality_score() {
        let eda = CodeEDA::new();

        // High quality: similar code
        let score = eda.quality_score("x = 1\ny = 2", "x = 1\ny = 2");
        assert!((score - 1.0).abs() < f32::EPSILON);

        // Medium quality: some overlap
        let score = eda.quality_score("x_new = 1\ny = 2", "x = 1\ny = 2");
        assert!(score > 0.5);

        // Zero quality: unbalanced parens
        let score = eda.quality_score("x = (1", "x = 1");
        assert!(score < f32::EPSILON);
    }

    #[test]
    fn test_diversity_score() {
        let eda = CodeEDA::new();

        // All unique
        let batch = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert!((eda.diversity_score(&batch) - 1.0).abs() < f32::EPSILON);

        // All same
        let batch = vec!["a".to_string(), "a".to_string(), "a".to_string()];
        assert!((eda.diversity_score(&batch) - 1.0 / 3.0).abs() < f32::EPSILON);

        // Empty
        let batch: Vec<String> = vec![];
        assert!(eda.diversity_score(&batch) < f32::EPSILON);
    }

    #[test]
    fn test_batch_augmenter() {
        let config = CodeEDAConfig::default();
        let mut augmenter = BatchAugmenter::new(config, 2.0);

        let samples = vec!["x = 1".to_string(), "y = 2".to_string()];
        let results = augmenter.augment_batch(&samples);

        assert_eq!(results.len(), 2);
        for result in &results {
            assert!(result.diversity_score >= 0.0);
        }
    }

    #[test]
    fn test_extract_variables() {
        let eda = CodeEDA::new();

        let vars = eda.extract_variables("x = 1\ny = 2\nif x == y: pass");
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert!(!vars.contains("if")); // Keyword
    }

    #[test]
    fn test_basic_syntax_check() {
        let eda = CodeEDA::new();

        assert!(eda.basic_syntax_check("x = (1 + 2)"));
        assert!(eda.basic_syntax_check("x = [1, 2, 3]"));
        assert!(eda.basic_syntax_check("x = {'a': 1}"));
        assert!(eda.basic_syntax_check("x = \"hello\""));

        assert!(!eda.basic_syntax_check("x = (1 + 2"));
        assert!(!eda.basic_syntax_check("x = [1, 2"));
        assert!(!eda.basic_syntax_check("x = \"hello"));
    }

    #[test]
    fn test_is_valid_identifier() {
        assert!(is_valid_identifier("foo"));
        assert!(is_valid_identifier("_bar"));
        assert!(is_valid_identifier("baz123"));
        assert!(is_valid_identifier("__init__"));

        assert!(!is_valid_identifier("123abc"));
        assert!(!is_valid_identifier(""));
        assert!(!is_valid_identifier("foo-bar"));
    }

    #[test]
    fn test_is_keyword() {
        assert!(is_keyword("if"));
        assert!(is_keyword("for"));
        assert!(is_keyword("return"));
        assert!(is_keyword("True"));

        assert!(!is_keyword("foo"));
        assert!(!is_keyword("bar"));
    }

    // ========== EDGE CASE TESTS (Extreme TDD) ==========

    #[test]
    fn test_augment_empty_code() {
        let mut eda = CodeEDA::new();
        let augmented = eda.augment("", 3);
        // Empty code should still produce valid augmentations
        for aug in &augmented {
            assert!(eda.basic_syntax_check(aug));
        }
    }

    #[test]
    fn test_augment_single_char() {
        let mut eda = CodeEDA::new();
        let augmented = eda.augment("x", 3);
        assert!(augmented.is_empty() || augmented.iter().all(|a| eda.basic_syntax_check(a)));
    }

    #[test]
    fn test_augment_whitespace_only() {
        let mut eda = CodeEDA::new();
        let augmented = eda.augment("   \n\t\n   ", 3);
        for aug in &augmented {
            assert!(eda.basic_syntax_check(aug));
        }
    }

    #[test]
    fn test_extract_variables_tuple_unpacking() {
        let eda = CodeEDA::new();
        let vars = eda.extract_variables("a, b, c = 1, 2, 3");
        assert!(vars.contains("a"));
        assert!(vars.contains("b"));
        assert!(vars.contains("c"));
    }

    #[test]
    fn test_extract_variables_no_assignments() {
        let eda = CodeEDA::new();
        let vars = eda.extract_variables("print('hello')\nfoo()");
        assert!(vars.is_empty());
    }

    #[test]
    fn test_extract_variables_with_comparison() {
        let eda = CodeEDA::new();
        let vars = eda.extract_variables("if x == y:\n    pass");
        // Should not extract from comparisons
        assert!(!vars.contains("x"));
    }

    #[test]
    fn test_synonym_replacement_no_variables() {
        let mut eda = CodeEDA::with_config(CodeEDAConfig {
            sr_prob: 1.0,
            ri_prob: 0.0,
            rs_prob: 0.0,
            rd_prob: 0.0,
            ..Default::default()
        });

        let code = "print('hello')";
        let augmented = eda.augment(code, 1);
        // Should not crash, just return original or valid augmentation
        assert!(augmented.is_empty() || eda.basic_syntax_check(&augmented[0]));
    }

    #[test]
    fn test_random_swap_single_line() {
        let mut eda = CodeEDA::with_config(CodeEDAConfig {
            sr_prob: 0.0,
            ri_prob: 0.0,
            rs_prob: 1.0,
            rd_prob: 0.0,
            ..Default::default()
        });

        let code = "x = 1";
        let augmented = eda.augment(code, 1);
        // Single line can't be swapped
        assert!(augmented.is_empty() || augmented[0] == code);
    }

    #[test]
    fn test_random_deletion_minimal_code() {
        let mut eda = CodeEDA::with_config(CodeEDAConfig {
            sr_prob: 0.0,
            ri_prob: 0.0,
            rs_prob: 0.0,
            rd_prob: 1.0,
            ..Default::default()
        });

        let code = "x = 1\ny = 2"; // Only 2 lines
        let augmented = eda.augment(code, 1);
        // Should not delete from minimal code
        assert!(augmented.is_empty() || augmented[0].lines().count() >= 2);
    }

    #[test]
    fn test_random_deletion_removes_comment() {
        let mut eda = CodeEDA::with_config(CodeEDAConfig {
            sr_prob: 0.0,
            ri_prob: 0.0,
            rs_prob: 0.0,
            rd_prob: 1.0,
            quality_threshold: 0.0, // Accept any quality
            ..Default::default()
        });

        let code = "x = 1\n# comment\ny = 2\nz = 3";
        let augmented = eda.augment(code, 1);
        if !augmented.is_empty() {
            // Should have removed the comment
            assert!(!augmented[0].contains("# comment") || augmented[0].lines().count() < 4);
        }
    }

    #[test]
    fn test_quality_score_empty_original() {
        let eda = CodeEDA::new();
        let score = eda.quality_score("x = 1", "");
        // Empty original means no tokens to compare
        assert!((score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_quality_score_nested_brackets() {
        let eda = CodeEDA::new();
        let code = "x = [[1, 2], [3, 4]]";
        let score = eda.quality_score(code, code);
        assert!((score - 1.0).abs() < f32::EPSILON);
        assert!(eda.basic_syntax_check(code));
    }

    #[test]
    fn test_quality_score_unbalanced_nested() {
        let eda = CodeEDA::new();
        let score = eda.quality_score("x = [[1, 2]", "x = 1");
        assert!(score < f32::EPSILON);
    }

    #[test]
    fn test_replace_identifier_word_boundary() {
        let eda = CodeEDA::new();

        // Should not replace 'x' in 'max'
        let result = eda.replace_identifier("max = x + max_value", "x", "y");
        assert!(result.contains("max"));
        assert!(result.contains("y"));
        assert!(result.contains("max_value")); // Should not become may_value
    }

    #[test]
    fn test_replace_identifier_at_start() {
        let eda = CodeEDA::new();
        let result = eda.replace_identifier("foo = 1", "foo", "bar");
        assert_eq!(result, "bar = 1");
    }

    #[test]
    fn test_replace_identifier_at_end() {
        let eda = CodeEDA::new();
        let result = eda.replace_identifier("x = foo", "foo", "bar");
        assert_eq!(result, "x = bar");
    }

    #[test]
    fn test_find_swappable_pairs_control_flow() {
        let eda = CodeEDA::new();
        let lines: Vec<&str> = vec!["if x:", "    y = 1"];
        let pairs = eda.find_swappable_pairs(&lines);
        // Control flow should not be swappable
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_find_swappable_pairs_different_indent() {
        let eda = CodeEDA::new();
        let lines: Vec<&str> = vec!["x = 1", "    y = 2"];
        let pairs = eda.find_swappable_pairs(&lines);
        // Different indentation should not be swappable
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_find_swappable_pairs_valid() {
        let eda = CodeEDA::new();
        let lines: Vec<&str> = vec!["x = 1", "y = 2", "z = 3"];
        let pairs = eda.find_swappable_pairs(&lines);
        // Adjacent pairs with same indent should be swappable
        assert!(!pairs.is_empty());
    }

    #[test]
    fn test_basic_syntax_check_escaped_quotes() {
        let eda = CodeEDA::new();
        // Note: Our simple parser doesn't handle escapes, but shouldn't crash
        let result = eda.basic_syntax_check(r#"x = "hello""#);
        assert!(result);
    }

    #[test]
    fn test_basic_syntax_check_mixed_brackets() {
        let eda = CodeEDA::new();
        assert!(eda.basic_syntax_check("x = ([1, 2], {3: 4})"));
        assert!(!eda.basic_syntax_check("x = ([1, 2}, {3: 4])"));
    }

    #[test]
    fn test_config_probabilities_boundary() {
        let config = CodeEDAConfig {
            sr_prob: 0.0,
            ri_prob: 0.0,
            rs_prob: 0.0,
            rd_prob: 0.0,
            quality_threshold: 0.0,
            seed: 42,
        };
        let mut eda = CodeEDA::with_config(config);
        let code = "x = 1";
        let augmented = eda.augment(code, 5);
        // With all probs at 0, should return original code
        for aug in &augmented {
            assert_eq!(aug, code);
        }
    }

    #[test]
    fn test_config_all_ops_enabled() {
        let config = CodeEDAConfig {
            sr_prob: 1.0,
            ri_prob: 1.0,
            rs_prob: 1.0,
            rd_prob: 1.0,
            quality_threshold: 0.0, // Accept any
            seed: 42,
        };
        let mut eda = CodeEDA::with_config(config);
        let code = "x = 1\n# comment\ny = 2\nz = 3";
        let augmented = eda.augment(code, 3);
        // Should produce varied augmentations
        assert!(!augmented.is_empty());
    }

    #[test]
    fn test_batch_augmenter_empty_samples() {
        let config = CodeEDAConfig::default();
        let mut augmenter = BatchAugmenter::new(config, 2.0);
        let samples: Vec<String> = vec![];
        let results = augmenter.augment_batch(&samples);
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_augmenter_factor_zero() {
        let config = CodeEDAConfig::default();
        let mut augmenter = BatchAugmenter::new(config, 0.0);
        let samples = vec!["x = 1".to_string()];
        let results = augmenter.augment_batch(&samples);
        // Factor 0.0 should be treated as at least 1
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_augmentation_result_fields() {
        let result = AugmentationResult {
            original: "x = 1".to_string(),
            variants: vec!["x_new = 1".to_string()],
            quality_scores: vec![0.8],
            diversity_score: 1.0,
        };
        assert_eq!(result.original, "x = 1");
        assert_eq!(result.variants.len(), 1);
        assert_eq!(result.quality_scores.len(), 1);
        assert!((result.diversity_score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_tokenize_special_chars() {
        let tokens: Vec<_> = tokenize("x = 1 + y * 2").collect();
        assert!(tokens.contains(&"x"));
        assert!(tokens.contains(&"1"));
        assert!(tokens.contains(&"y"));
        assert!(tokens.contains(&"2"));
        // Should not contain operators
        assert!(!tokens.contains(&"+"));
        assert!(!tokens.contains(&"*"));
    }

    #[test]
    fn test_is_valid_identifier_unicode() {
        // Rust's is_alphabetic() includes unicode letters
        assert!(is_valid_identifier("über")); // Valid: starts with alphabetic
        assert!(is_valid_identifier("x123"));
        assert!(!is_valid_identifier("123über")); // Invalid: starts with digit
    }

    #[test]
    fn test_code_eda_deterministic_with_seed() {
        let config = CodeEDAConfig {
            seed: 12345,
            ..Default::default()
        };
        let mut eda1 = CodeEDA::with_config(config.clone());
        let mut eda2 = CodeEDA::with_config(config);

        let code = "x = 1\ny = 2\nz = 3";
        let aug1 = eda1.augment(code, 3);
        let aug2 = eda2.augment(code, 3);

        assert_eq!(aug1, aug2, "Same seed should produce same augmentations");
    }

    #[test]
    fn test_code_eda_different_seeds() {
        let mut eda1 = CodeEDA::with_config(CodeEDAConfig {
            seed: 1,
            sr_prob: 1.0,
            ..Default::default()
        });
        let mut eda2 = CodeEDA::with_config(CodeEDAConfig {
            seed: 2,
            sr_prob: 1.0,
            ..Default::default()
        });

        let code = "foo = 1\nbar = foo + 2";
        let aug1 = eda1.augment(code, 1);
        let aug2 = eda2.augment(code, 1);

        // Different seeds may produce different results
        // (Not guaranteed, but likely with SR renaming)
        assert!(!aug1.is_empty());
        assert!(!aug2.is_empty());
    }

    // ========== KEYWORD EXHAUSTIVE TEST ==========

    #[test]
    fn test_all_python_keywords() {
        let keywords = [
            "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class",
            "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global",
            "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise",
            "return", "try", "while", "with", "yield",
        ];
        for kw in keywords {
            assert!(is_keyword(kw), "{kw} should be a keyword");
        }
    }

    #[test]
    fn test_non_keywords() {
        let non_keywords = ["foo", "bar", "baz", "x", "y", "z", "print", "len", "str"];
        for nk in non_keywords {
            assert!(!is_keyword(nk), "{nk} should not be a keyword");
        }
    }
}

/// Property-based tests for CodeEDA using proptest
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Generate valid Python-like code snippets
    fn python_code_strategy() -> impl Strategy<Value = String> {
        prop::collection::vec(
            prop_oneof![
                // Simple assignments
                "[a-z][a-z0-9_]{0,10} = [0-9]{1,5}".prop_map(|s| s),
                // Comments
                "# [a-zA-Z0-9 ]{0,20}".prop_map(|s| s),
                // Function calls
                "[a-z]+\\([0-9, ]*\\)".prop_map(|s| s),
            ],
            1..10,
        )
        .prop_map(|lines| lines.join("\n"))
    }

    proptest! {
        /// Augmented code always passes basic syntax check
        #[test]
        fn prop_augmented_code_is_syntactically_valid(
            seed in 0u64..10000,
            n_aug in 1usize..5,
        ) {
            let config = CodeEDAConfig {
                seed,
                quality_threshold: 0.5,
                ..Default::default()
            };
            let mut eda = CodeEDA::with_config(config);
            let code = "x = 1\ny = 2\nz = 3";
            let augmented = eda.augment(code, n_aug);

            for aug in &augmented {
                prop_assert!(eda.basic_syntax_check(aug));
            }
        }

        /// Quality score is always in [0.0, 1.0]
        #[test]
        fn prop_quality_score_bounded(
            code in "[a-z]+ = [0-9]+",
            aug in "[a-z]+ = [0-9]+",
        ) {
            let eda = CodeEDA::new();
            let score = eda.quality_score(&aug, &code);
            prop_assert!(score >= 0.0);
            prop_assert!(score <= 1.0);
        }

        /// Diversity score is always in [0.0, 1.0]
        #[test]
        fn prop_diversity_score_bounded(
            batch in prop::collection::vec("[a-z]+", 1..10),
        ) {
            let eda = CodeEDA::new();
            let score = eda.diversity_score(&batch);
            prop_assert!(score >= 0.0);
            prop_assert!(score <= 1.0);
        }

        /// Deterministic: same seed + same input = same output
        #[test]
        fn prop_deterministic_with_seed(
            seed in 0u64..10000,
            code in "[a-z]+ = [0-9]+\n[a-z]+ = [0-9]+",
        ) {
            let config = CodeEDAConfig {
                seed,
                ..Default::default()
            };
            let mut eda1 = CodeEDA::with_config(config.clone());
            let mut eda2 = CodeEDA::with_config(config);

            let aug1 = eda1.augment(&code, 3);
            let aug2 = eda2.augment(&code, 3);

            prop_assert_eq!(aug1, aug2);
        }

        /// Extracted variables are valid identifiers
        #[test]
        fn prop_extracted_vars_are_valid_identifiers(
            var in "[a-z][a-z0-9_]{0,10}",
        ) {
            let eda = CodeEDA::new();
            let code = format!("{var} = 42");
            let vars = eda.extract_variables(&code);

            for v in vars {
                prop_assert!(is_valid_identifier(&v));
                prop_assert!(!is_keyword(&v));
            }
        }

        /// Replace identifier preserves code length approximately
        #[test]
        fn prop_replace_identifier_similar_length(
            old in "[a-z]{3,6}",
            new in "[a-z]{3,6}",
        ) {
            let eda = CodeEDA::new();
            let code = format!("{old} = 1\n{old} + 2");
            let result = eda.replace_identifier(&code, &old, &new);

            // Length difference should be bounded by replacement diff * occurrences
            let len_diff = (result.len() as i64 - code.len() as i64).unsigned_abs();
            let replacement_diff = (new.len() as i64 - old.len() as i64).unsigned_abs();
            prop_assert!(len_diff <= replacement_diff * 2 + 1);
        }

        /// Balanced brackets: unbalanced code always scores 0
        #[test]
        fn prop_unbalanced_scores_zero(
            n_open in 1usize..5,
        ) {
            let eda = CodeEDA::new();
            let unbalanced = "(".repeat(n_open);
            let score = eda.quality_score(&unbalanced, "x = 1");
            prop_assert!(score < f32::EPSILON);
        }

        /// Swappable pairs have same indentation
        #[test]
        fn prop_swappable_pairs_same_indent(
            indent in 0usize..4,
            n_lines in 2usize..6,
        ) {
            let eda = CodeEDA::new();
            let space = " ".repeat(indent * 4);
            let lines: Vec<String> = (0..n_lines)
                .map(|i| format!("{space}x{i} = {i}"))
                .collect();
            let lines_ref: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();

            let pairs = eda.find_swappable_pairs(&lines_ref);

            for (i, j) in pairs {
                let indent_i = lines_ref[i].len() - lines_ref[i].trim_start().len();
                let indent_j = lines_ref[j].len() - lines_ref[j].trim_start().len();
                prop_assert_eq!(indent_i, indent_j);
            }
        }
    }
}
