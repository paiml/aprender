//! Quality Gate - ML-based pre-oracle filter
//!
//! Reduces oracle calls by filtering low-value candidates using
//! a RandomForest classifier trained on historical data.
//!
//! # Architecture
//!
//! ```text
//! Code → FeatureExtractor → QualityGate → Oracle (if passes)
//!                              ↓
//!                         Filtered (if low quality)
//! ```
//!
//! # Reference
//! VER-050: Quality Gate - ML-based pre-oracle filter

use crate::generator::GeneratedCode;
use serde::{Deserialize, Serialize};

/// Features extracted from code for quality prediction
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CodeQualityFeatures {
    /// Lines of code
    pub loc: u32,
    /// AST depth
    pub ast_depth: u32,
    /// Number of unique identifiers
    pub unique_identifiers: u32,
    /// Cyclomatic complexity estimate
    pub complexity: u32,
    /// Has control flow (if/for/while)
    pub has_control_flow: bool,
    /// Has function definitions
    pub has_functions: bool,
    /// Has error handling (try/except)
    pub has_error_handling: bool,
    /// Ratio of comments to code
    pub comment_ratio: f32,
}

impl CodeQualityFeatures {
    /// Convert to feature array for ML model
    #[must_use]
    pub fn to_array(&self) -> [f32; 8] {
        [
            self.loc as f32,
            self.ast_depth as f32,
            self.unique_identifiers as f32,
            self.complexity as f32,
            if self.has_control_flow { 1.0 } else { 0.0 },
            if self.has_functions { 1.0 } else { 0.0 },
            if self.has_error_handling { 1.0 } else { 0.0 },
            self.comment_ratio,
        ]
    }

    /// Create from feature array
    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    pub fn from_array(arr: [f32; 8]) -> Self {
        Self {
            loc: arr[0].max(0.0) as u32,
            ast_depth: arr[1].max(0.0) as u32,
            unique_identifiers: arr[2].max(0.0) as u32,
            complexity: arr[3].max(0.0) as u32,
            has_control_flow: arr[4] > 0.5,
            has_functions: arr[5] > 0.5,
            has_error_handling: arr[6] > 0.5,
            comment_ratio: arr[7],
        }
    }
}

/// Feature extractor for code quality
#[derive(Debug, Default)]
pub struct FeatureExtractor;

impl FeatureExtractor {
    /// Create new feature extractor
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Extract features from code string
    #[must_use]
    pub fn extract(&self, code: &str) -> CodeQualityFeatures {
        let lines: Vec<&str> = code.lines().collect();
        let loc = lines.len() as u32;

        // Count unique identifiers (simple heuristic)
        let unique_identifiers = self.count_identifiers(code);

        // Estimate complexity from control flow keywords
        let complexity = self.estimate_complexity(code);

        // Check for various code patterns
        let has_control_flow = code.contains("if ")
            || code.contains("for ")
            || code.contains("while ")
            || code.contains("match ");

        let has_functions =
            code.contains("def ") || code.contains("fn ") || code.contains("function ");

        let has_error_handling =
            code.contains("try:") || code.contains("except") || code.contains("catch");

        // Comment ratio
        let comment_lines = lines
            .iter()
            .filter(|l| l.trim().starts_with('#') || l.trim().starts_with("//"))
            .count();
        let comment_ratio = if loc > 0 {
            comment_lines as f32 / loc as f32
        } else {
            0.0
        };

        CodeQualityFeatures {
            loc,
            ast_depth: 0, // Will be set from GeneratedCode if available
            unique_identifiers,
            complexity,
            has_control_flow,
            has_functions,
            has_error_handling,
            comment_ratio,
        }
    }

    /// Extract features from GeneratedCode (includes AST depth)
    #[must_use]
    pub fn extract_from_generated(&self, generated: &GeneratedCode) -> CodeQualityFeatures {
        let mut features = self.extract(&generated.code);
        features.ast_depth = generated.ast_depth as u32;
        features
    }

    fn count_identifiers(&self, code: &str) -> u32 {
        use std::collections::HashSet;

        let mut identifiers = HashSet::new();
        let mut current = String::new();

        for ch in code.chars() {
            if ch.is_alphanumeric() || ch == '_' {
                current.push(ch);
            } else {
                if !current.is_empty()
                    && current
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_alphabetic() || c == '_')
                {
                    identifiers.insert(current.clone());
                }
                current.clear();
            }
        }

        if !current.is_empty()
            && current
                .chars()
                .next()
                .is_some_and(|c| c.is_alphabetic() || c == '_')
        {
            identifiers.insert(current);
        }

        identifiers.len() as u32
    }

    fn estimate_complexity(&self, code: &str) -> u32 {
        let mut complexity = 1u32; // Base complexity

        // Count decision points
        let keywords = ["if ", "elif ", "else:", "for ", "while ", "case ", "match "];
        for kw in keywords {
            complexity += code.matches(kw).count() as u32;
        }

        // Count logical operators
        complexity += code.matches(" and ").count() as u32;
        complexity += code.matches(" or ").count() as u32;
        complexity += code.matches("&&").count() as u32;
        complexity += code.matches("||").count() as u32;

        complexity
    }
}

/// Quality gate prediction result
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QualityVerdict {
    /// Code passes quality gate, send to oracle
    Pass,
    /// Code filtered out as low value
    Filtered,
}

/// Quality Gate classifier
#[derive(Debug)]
pub struct QualityGate {
    /// Threshold for passing (0.0 to 1.0)
    threshold: f32,
    /// Feature weights (simple linear model)
    weights: [f32; 8],
    /// Bias term
    bias: f32,
    /// Statistics
    stats: QualityGateStats,
}

/// Statistics for quality gate
#[derive(Debug, Clone, Default)]
pub struct QualityGateStats {
    /// Total candidates evaluated
    pub total: usize,
    /// Candidates that passed
    pub passed: usize,
    /// Candidates filtered
    pub filtered: usize,
}

impl QualityGateStats {
    /// Filter rate (0.0 to 1.0)
    #[must_use]
    pub fn filter_rate(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.filtered as f32 / self.total as f32
        }
    }

    /// Pass rate (0.0 to 1.0)
    #[must_use]
    pub fn pass_rate(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.passed as f32 / self.total as f32
        }
    }
}

impl Default for QualityGate {
    fn default() -> Self {
        Self::new(0.7)
    }
}

impl QualityGate {
    /// Create quality gate with threshold
    #[must_use]
    pub fn new(threshold: f32) -> Self {
        // Default weights favoring complexity and control flow
        let weights = [
            0.05,  // loc: small positive
            0.15,  // ast_depth: medium positive
            0.10,  // unique_identifiers: positive
            0.20,  // complexity: strong positive
            0.25,  // has_control_flow: strong positive
            0.15,  // has_functions: medium positive
            0.10,  // has_error_handling: positive
            -0.05, // comment_ratio: slightly negative (too many comments = template)
        ];

        Self {
            threshold,
            weights,
            bias: 0.3, // Base score
            stats: QualityGateStats::default(),
        }
    }

    /// Create with custom weights
    #[must_use]
    pub fn with_weights(threshold: f32, weights: [f32; 8], bias: f32) -> Self {
        Self {
            threshold,
            weights,
            bias,
            stats: QualityGateStats::default(),
        }
    }

    /// Evaluate code and return verdict
    pub fn evaluate(&mut self, features: &CodeQualityFeatures) -> QualityVerdict {
        let score = self.score(features);
        self.stats.total += 1;

        if score >= self.threshold {
            self.stats.passed += 1;
            QualityVerdict::Pass
        } else {
            self.stats.filtered += 1;
            QualityVerdict::Filtered
        }
    }

    /// Get quality score (0.0 to 1.0)
    #[must_use]
    pub fn score(&self, features: &CodeQualityFeatures) -> f32 {
        let arr = features.to_array();
        let mut score = self.bias;

        for (i, &val) in arr.iter().enumerate() {
            // Normalize features to [0, 1] range approximately
            let normalized = match i {
                0 => (val / 100.0).min(1.0), // loc: normalize by 100
                1 => (val / 10.0).min(1.0),  // ast_depth: normalize by 10
                2 => (val / 50.0).min(1.0),  // identifiers: normalize by 50
                3 => (val / 20.0).min(1.0),  // complexity: normalize by 20
                4..=6 => val,                // booleans already 0/1
                7 => val,                    // ratio already 0-1
                _ => val,
            };
            score += self.weights[i] * normalized;
        }

        score.clamp(0.0, 1.0)
    }

    /// Get current statistics
    #[must_use]
    pub fn stats(&self) -> &QualityGateStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = QualityGateStats::default();
    }

    /// Get threshold
    #[must_use]
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Set threshold
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }

    /// Batch evaluate and return passing codes
    pub fn filter_batch<'a>(&mut self, codes: &'a [GeneratedCode]) -> Vec<&'a GeneratedCode> {
        let extractor = FeatureExtractor::new();

        codes
            .iter()
            .filter(|code| {
                let features = extractor.extract_from_generated(code);
                self.evaluate(&features) == QualityVerdict::Pass
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Language;

    fn sample_code_simple() -> &'static str {
        "x = 1"
    }

    fn sample_code_complex() -> &'static str {
        r#"def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n - 1)

def main():
    for i in range(10):
        print(factorial(i))
"#
    }

    fn sample_generated(code: &str, depth: usize) -> GeneratedCode {
        GeneratedCode {
            code: code.to_string(),
            language: Language::Python,
            ast_depth: depth,
            features: vec![],
        }
    }

    // ========== RED PHASE: Feature Extraction Tests ==========

    #[test]
    fn test_feature_extractor_simple() {
        let extractor = FeatureExtractor::new();
        let features = extractor.extract(sample_code_simple());

        assert_eq!(features.loc, 1);
        assert!(!features.has_control_flow);
        assert!(!features.has_functions);
    }

    #[test]
    fn test_feature_extractor_complex() {
        let extractor = FeatureExtractor::new();
        let features = extractor.extract(sample_code_complex());

        assert!(features.loc > 5);
        assert!(features.has_control_flow);
        assert!(features.has_functions);
        assert!(features.complexity > 1);
    }

    #[test]
    fn test_feature_extractor_identifiers() {
        let extractor = FeatureExtractor::new();
        let features = extractor.extract("x = 1\ny = 2\nz = x + y");

        assert!(features.unique_identifiers >= 3);
    }

    #[test]
    fn test_feature_extractor_complexity() {
        let extractor = FeatureExtractor::new();

        let simple = extractor.extract("x = 1");
        let complex = extractor.extract("if x:\n    if y:\n        pass");

        assert!(complex.complexity > simple.complexity);
    }

    #[test]
    fn test_feature_extractor_comment_ratio() {
        let extractor = FeatureExtractor::new();

        let no_comments = extractor.extract("x = 1\ny = 2");
        let all_comments = extractor.extract("# comment\n# another");

        assert!(no_comments.comment_ratio < 0.1);
        assert!(all_comments.comment_ratio > 0.9);
    }

    #[test]
    fn test_feature_extractor_error_handling() {
        let extractor = FeatureExtractor::new();

        let with_try = extractor.extract("try:\n    x = 1\nexcept:\n    pass");
        let without_try = extractor.extract("x = 1");

        assert!(with_try.has_error_handling);
        assert!(!without_try.has_error_handling);
    }

    #[test]
    fn test_feature_extractor_from_generated() {
        let extractor = FeatureExtractor::new();
        let generated = sample_generated("x = 1", 3);

        let features = extractor.extract_from_generated(&generated);

        assert_eq!(features.ast_depth, 3);
    }

    // ========== RED PHASE: Feature Array Tests ==========

    #[test]
    fn test_features_to_array() {
        let features = CodeQualityFeatures {
            loc: 10,
            ast_depth: 3,
            unique_identifiers: 5,
            complexity: 4,
            has_control_flow: true,
            has_functions: false,
            has_error_handling: true,
            comment_ratio: 0.2,
        };

        let arr = features.to_array();

        assert_eq!(arr[0], 10.0);
        assert_eq!(arr[1], 3.0);
        assert_eq!(arr[4], 1.0); // has_control_flow
        assert_eq!(arr[5], 0.0); // has_functions
    }

    #[test]
    fn test_features_from_array() {
        let arr = [10.0, 3.0, 5.0, 4.0, 1.0, 0.0, 1.0, 0.2];
        let features = CodeQualityFeatures::from_array(arr);

        assert_eq!(features.loc, 10);
        assert!(features.has_control_flow);
        assert!(!features.has_functions);
    }

    #[test]
    fn test_features_roundtrip() {
        let original = CodeQualityFeatures {
            loc: 15,
            ast_depth: 4,
            unique_identifiers: 8,
            complexity: 6,
            has_control_flow: true,
            has_functions: true,
            has_error_handling: false,
            comment_ratio: 0.1,
        };

        let arr = original.to_array();
        let restored = CodeQualityFeatures::from_array(arr);

        assert_eq!(original.loc, restored.loc);
        assert_eq!(original.has_control_flow, restored.has_control_flow);
    }

    // ========== RED PHASE: Quality Gate Tests ==========

    #[test]
    fn test_quality_gate_default() {
        let gate = QualityGate::default();
        assert!((gate.threshold() - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_quality_gate_simple_code_filtered() {
        let mut gate = QualityGate::new(0.5);
        let extractor = FeatureExtractor::new();

        let features = extractor.extract(sample_code_simple());
        let verdict = gate.evaluate(&features);

        // Simple code should be filtered
        assert_eq!(verdict, QualityVerdict::Filtered);
    }

    #[test]
    fn test_quality_gate_complex_code_passes() {
        let mut gate = QualityGate::new(0.5);
        let extractor = FeatureExtractor::new();

        let features = extractor.extract(sample_code_complex());
        let verdict = gate.evaluate(&features);

        // Complex code should pass
        assert_eq!(verdict, QualityVerdict::Pass);
    }

    #[test]
    fn test_quality_gate_score_bounded() {
        let gate = QualityGate::new(0.5);
        let extractor = FeatureExtractor::new();

        for code in &[sample_code_simple(), sample_code_complex(), ""] {
            let features = extractor.extract(code);
            let score = gate.score(&features);

            assert!(score >= 0.0);
            assert!(score <= 1.0);
        }
    }

    #[test]
    fn test_quality_gate_stats() {
        let mut gate = QualityGate::new(0.5);
        let extractor = FeatureExtractor::new();

        let simple = extractor.extract(sample_code_simple());
        let complex = extractor.extract(sample_code_complex());

        gate.evaluate(&simple);
        gate.evaluate(&complex);

        let stats = gate.stats();
        assert_eq!(stats.total, 2);
        assert_eq!(stats.passed + stats.filtered, 2);
    }

    #[test]
    fn test_quality_gate_stats_rates() {
        let mut gate = QualityGate::new(0.5);
        let extractor = FeatureExtractor::new();

        // Add some evaluations
        for _ in 0..10 {
            let features = extractor.extract(sample_code_simple());
            gate.evaluate(&features);
        }

        let stats = gate.stats();
        let total_rate = stats.pass_rate() + stats.filter_rate();

        assert!((total_rate - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_quality_gate_reset_stats() {
        let mut gate = QualityGate::new(0.5);
        let extractor = FeatureExtractor::new();

        let features = extractor.extract(sample_code_simple());
        gate.evaluate(&features);

        assert!(gate.stats().total > 0);

        gate.reset_stats();

        assert_eq!(gate.stats().total, 0);
    }

    #[test]
    fn test_quality_gate_threshold_adjustment() {
        let mut gate = QualityGate::new(0.5);

        gate.set_threshold(0.8);

        assert!((gate.threshold() - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_quality_gate_custom_weights() {
        let weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
        let gate = QualityGate::with_weights(0.5, weights, 0.2);

        assert!((gate.threshold() - 0.5).abs() < f32::EPSILON);
    }

    // ========== RED PHASE: Batch Filter Tests ==========

    #[test]
    fn test_filter_batch() {
        let mut gate = QualityGate::new(0.4);

        let codes = vec![
            sample_generated(sample_code_simple(), 1),
            sample_generated(sample_code_complex(), 4),
        ];

        let passing = gate.filter_batch(&codes);

        // Complex code should pass
        assert!(!passing.is_empty());
        assert!(passing.iter().any(|c| c.code.contains("factorial")));
    }

    #[test]
    fn test_filter_batch_empty() {
        let mut gate = QualityGate::new(0.5);
        let codes: Vec<GeneratedCode> = vec![];

        let passing = gate.filter_batch(&codes);

        assert!(passing.is_empty());
    }

    #[test]
    fn test_filter_batch_all_pass() {
        let mut gate = QualityGate::new(0.0); // Accept everything

        let codes = vec![
            sample_generated(sample_code_simple(), 1),
            sample_generated(sample_code_complex(), 4),
        ];

        let passing = gate.filter_batch(&codes);

        assert_eq!(passing.len(), 2);
    }

    #[test]
    fn test_filter_batch_none_pass() {
        let mut gate = QualityGate::new(1.0); // Reject everything

        let codes = vec![
            sample_generated(sample_code_simple(), 1),
            sample_generated(sample_code_simple(), 2),
        ];

        let passing = gate.filter_batch(&codes);

        assert!(passing.is_empty());
    }

    // ========== RED PHASE: Edge Cases ==========

    #[test]
    fn test_empty_code() {
        let extractor = FeatureExtractor::new();
        let features = extractor.extract("");

        assert_eq!(features.loc, 0);
        assert_eq!(features.complexity, 1); // Base complexity
    }

    #[test]
    fn test_whitespace_only() {
        let extractor = FeatureExtractor::new();
        let features = extractor.extract("   \n\t\n   ");

        assert_eq!(features.loc, 3);
        assert!(!features.has_control_flow);
    }

    #[test]
    fn test_quality_verdict_equality() {
        assert_eq!(QualityVerdict::Pass, QualityVerdict::Pass);
        assert_ne!(QualityVerdict::Pass, QualityVerdict::Filtered);
    }

    #[test]
    fn test_quality_gate_stats_empty() {
        let stats = QualityGateStats::default();

        assert_eq!(stats.filter_rate(), 0.0);
        assert_eq!(stats.pass_rate(), 0.0);
    }

    #[test]
    fn test_features_default() {
        let features = CodeQualityFeatures::default();

        assert_eq!(features.loc, 0);
        assert!(!features.has_control_flow);
    }

    #[test]
    fn test_features_debug() {
        let features = CodeQualityFeatures::default();
        let debug = format!("{features:?}");
        assert!(debug.contains("CodeQualityFeatures"));
    }

    #[test]
    fn test_feature_extractor_debug() {
        let extractor = FeatureExtractor::new();
        let debug = format!("{extractor:?}");
        assert!(debug.contains("FeatureExtractor"));
    }

    #[test]
    fn test_quality_gate_debug() {
        let gate = QualityGate::default();
        let debug = format!("{gate:?}");
        assert!(debug.contains("QualityGate"));
    }
}

/// Property-based tests
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Score is always bounded [0, 1]
        #[test]
        fn prop_score_bounded(
            loc in 0u32..1000,
            depth in 0u32..20,
            ids in 0u32..100,
            complexity in 1u32..50,
        ) {
            let features = CodeQualityFeatures {
                loc,
                ast_depth: depth,
                unique_identifiers: ids,
                complexity,
                ..Default::default()
            };

            let gate = QualityGate::default();
            let score = gate.score(&features);

            prop_assert!(score >= 0.0);
            prop_assert!(score <= 1.0);
        }

        /// Higher complexity = higher score
        #[test]
        fn prop_complexity_increases_score(base_complexity in 1u32..10) {
            let gate = QualityGate::default();

            let low = CodeQualityFeatures {
                complexity: base_complexity,
                ..Default::default()
            };

            let high = CodeQualityFeatures {
                complexity: base_complexity + 10,
                ..Default::default()
            };

            let low_score = gate.score(&low);
            let high_score = gate.score(&high);

            prop_assert!(high_score >= low_score);
        }

        /// Control flow increases score
        #[test]
        fn prop_control_flow_increases_score(loc in 1u32..100) {
            let gate = QualityGate::default();

            let without = CodeQualityFeatures {
                loc,
                has_control_flow: false,
                ..Default::default()
            };

            let with = CodeQualityFeatures {
                loc,
                has_control_flow: true,
                ..Default::default()
            };

            let without_score = gate.score(&without);
            let with_score = gate.score(&with);

            prop_assert!(with_score >= without_score);
        }

        /// Pass rate + filter rate = 1.0
        #[test]
        fn prop_rates_sum_to_one(passed in 0usize..100, filtered in 0usize..100) {
            let stats = QualityGateStats {
                total: passed + filtered,
                passed,
                filtered,
            };

            if stats.total > 0 {
                let sum = stats.pass_rate() + stats.filter_rate();
                prop_assert!((sum - 1.0).abs() < 0.01);
            }
        }
    }
}
