//! ML-based enhancements for TranspilerOracle
//!
//! Integrates aprender ML capabilities for smarter test prioritization
//! and bug prediction. See GH-2.

use crate::generator::GeneratedCode;

/// Features extracted from source code for ML prediction
#[derive(Debug, Clone, Default)]
pub struct CodeFeatures {
    /// AST depth
    pub ast_depth: usize,
    /// Number of AST nodes
    pub node_count: usize,
    /// Cyclomatic complexity estimate
    pub cyclomatic_complexity: usize,
    /// Number of unique identifiers
    pub identifier_count: usize,
    /// Number of function calls
    pub call_count: usize,
    /// Has loops
    pub has_loops: bool,
    /// Has conditionals
    pub has_conditionals: bool,
    /// Has exception handling
    pub has_exceptions: bool,
}

impl CodeFeatures {
    /// Extract features from a generated program
    #[must_use]
    pub fn from_program(program: &GeneratedCode) -> Self {
        Self {
            ast_depth: program.ast_depth,
            node_count: program.code.lines().count(),
            cyclomatic_complexity: estimate_complexity(&program.code),
            identifier_count: count_identifiers(&program.code),
            call_count: count_calls(&program.code),
            has_loops: program.code.contains("for") || program.code.contains("while"),
            has_conditionals: program.code.contains("if"),
            has_exceptions: program.code.contains("try") || program.code.contains("except"),
        }
    }

    /// Convert to feature vector for ML
    #[must_use]
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.ast_depth as f64,
            self.node_count as f64,
            self.cyclomatic_complexity as f64,
            self.identifier_count as f64,
            self.call_count as f64,
            if self.has_loops { 1.0 } else { 0.0 },
            if self.has_conditionals { 1.0 } else { 0.0 },
            if self.has_exceptions { 1.0 } else { 0.0 },
        ]
    }
}

/// Estimate cyclomatic complexity from code
fn estimate_complexity(code: &str) -> usize {
    let mut complexity = 1;
    for keyword in ["if", "elif", "for", "while", "and", "or", "except"] {
        complexity += code.matches(keyword).count();
    }
    complexity
}

/// Count unique identifiers in code
fn count_identifiers(code: &str) -> usize {
    code.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty() && s.chars().next().is_some_and(char::is_alphabetic))
        .collect::<std::collections::HashSet<_>>()
        .len()
}

/// Count function calls in code
fn count_calls(code: &str) -> usize {
    code.matches('(').count()
}

/// Bug predictor using ML model
pub trait BugPredictor: Send + Sync {
    /// Predict probability that a test case will expose a bug
    fn predict_bug_probability(&self, features: &CodeFeatures) -> f64;

    /// Batch prediction for efficiency
    fn predict_batch(&self, features: &[CodeFeatures]) -> Vec<f64> {
        features
            .iter()
            .map(|f| self.predict_bug_probability(f))
            .collect()
    }
}

/// Test prioritizer using similarity
pub trait TestPrioritizer: Send + Sync {
    /// Prioritize test cases by similarity to known failing tests
    fn prioritize(&self, tests: &[GeneratedCode], k: usize) -> Vec<usize>;

    /// Add a failing test to the index
    fn add_failing_test(&mut self, test: &GeneratedCode);

    /// Number of failing tests in index
    fn failing_count(&self) -> usize;
}

/// Simple baseline predictor (always returns 0.5)
#[derive(Debug, Clone, Default)]
pub struct BaselinePredictor;

impl BugPredictor for BaselinePredictor {
    fn predict_bug_probability(&self, _features: &CodeFeatures) -> f64 {
        0.5
    }
}

/// Complexity-based predictor (higher complexity = higher bug probability)
#[derive(Debug, Clone)]
pub struct ComplexityPredictor {
    /// Complexity threshold for high bug probability
    pub threshold: usize,
}

impl Default for ComplexityPredictor {
    fn default() -> Self {
        Self { threshold: 5 }
    }
}

impl BugPredictor for ComplexityPredictor {
    fn predict_bug_probability(&self, features: &CodeFeatures) -> f64 {
        let score = features.cyclomatic_complexity as f64 / self.threshold as f64;
        score.min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Language;

    fn sample_program() -> GeneratedCode {
        GeneratedCode {
            code: "def foo(x):\n    if x > 0:\n        return x\n    return 0".to_string(),
            language: Language::Python,
            ast_depth: 3,
            features: vec!["function".to_string(), "conditional".to_string()],
        }
    }

    #[test]
    fn test_code_features_from_program() {
        let program = sample_program();
        let features = CodeFeatures::from_program(&program);

        assert_eq!(features.ast_depth, 3);
        assert!(features.has_conditionals);
        assert!(!features.has_loops);
        assert!(features.node_count > 0);
    }

    #[test]
    fn test_code_features_to_vec() {
        let features = CodeFeatures {
            ast_depth: 3,
            node_count: 10,
            cyclomatic_complexity: 5,
            identifier_count: 8,
            call_count: 2,
            has_loops: true,
            has_conditionals: true,
            has_exceptions: false,
        };

        let vec = features.to_vec();
        assert_eq!(vec.len(), 8);
        assert_eq!(vec[0], 3.0); // ast_depth
        assert_eq!(vec[5], 1.0); // has_loops
        assert_eq!(vec[7], 0.0); // has_exceptions
    }

    #[test]
    fn test_estimate_complexity() {
        let simple = "x = 1";
        let complex = "if x and y or z:\n    for i in range(10):\n        pass";

        assert!(estimate_complexity(complex) > estimate_complexity(simple));
    }

    #[test]
    fn test_count_identifiers() {
        let code = "def foo(x, y):\n    return x + y";
        let count = count_identifiers(code);
        assert!(count >= 4); // def, foo, x, y, return
    }

    #[test]
    fn test_baseline_predictor() {
        let predictor = BaselinePredictor;
        let features = CodeFeatures::default();

        assert_eq!(predictor.predict_bug_probability(&features), 0.5);
    }

    #[test]
    fn test_complexity_predictor() {
        let predictor = ComplexityPredictor { threshold: 5 };

        let low = CodeFeatures {
            cyclomatic_complexity: 2,
            ..Default::default()
        };
        let high = CodeFeatures {
            cyclomatic_complexity: 10,
            ..Default::default()
        };

        assert!(predictor.predict_bug_probability(&low) < 0.5);
        assert_eq!(predictor.predict_bug_probability(&high), 1.0);
    }

    #[test]
    fn test_batch_prediction() {
        let predictor = BaselinePredictor;
        let features = vec![CodeFeatures::default(), CodeFeatures::default()];

        let predictions = predictor.predict_batch(&features);
        assert_eq!(predictions.len(), 2);
        assert!(predictions.iter().all(|&p| p == 0.5));
    }

    // RED PHASE: These tests should fail until we implement aprender integration

    #[test]
    #[ignore = "requires aprender ml feature"]
    fn test_random_forest_predictor() {
        // TODO: Implement RandomForestPredictor using aprender
        // let predictor = RandomForestPredictor::train(&training_data);
        // assert!(predictor.predict_bug_probability(&features) >= 0.0);
        // assert!(predictor.predict_bug_probability(&features) <= 1.0);
        unimplemented!("RandomForestPredictor not yet implemented")
    }

    #[test]
    #[ignore = "requires aprender ml feature"]
    fn test_hnsw_prioritizer() {
        // TODO: Implement HNSWPrioritizer using aprender
        // let mut prioritizer = HNSWPrioritizer::new();
        // prioritizer.add_failing_test(&failing_test);
        // let priorities = prioritizer.prioritize(&tests, 5);
        // assert_eq!(priorities.len(), 5);
        unimplemented!("HNSWPrioritizer not yet implemented")
    }

    #[test]
    #[ignore = "requires aprender ml feature"]
    fn test_incremental_learning() {
        // TODO: Implement incremental model updates
        // let mut predictor = IncrementalPredictor::new();
        // predictor.update(&new_examples);
        // assert!(predictor.version() > 0);
        unimplemented!("IncrementalPredictor not yet implemented")
    }
}
