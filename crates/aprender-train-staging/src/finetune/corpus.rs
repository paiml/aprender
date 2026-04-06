//! Test generation corpus loader
//!
//! Loads and manages training data for Rust test generation.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// A single training sample for test generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestGenSample {
    /// Source function code
    pub function: String,
    /// Generated unit tests
    pub unit_tests: String,
    /// Property-based tests (optional)
    #[serde(default)]
    pub property_tests: Option<String>,
    /// Metadata about the sample
    #[serde(default)]
    pub metadata: SampleMetadata,
}

/// Metadata about a training sample
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SampleMetadata {
    /// Source crate name
    #[serde(default)]
    pub crate_name: Option<String>,
    /// Cyclomatic complexity
    #[serde(default)]
    pub complexity: Option<u32>,
    /// Whether function uses generics
    #[serde(default)]
    pub has_generics: bool,
    /// Whether function uses lifetimes
    #[serde(default)]
    pub has_lifetimes: bool,
    /// Whether function is async
    #[serde(default)]
    pub is_async: bool,
}

/// Test generation corpus
#[derive(Debug, Clone)]
pub struct TestGenCorpus {
    /// Training samples
    pub train: Vec<TestGenSample>,
    /// Validation samples
    pub validation: Vec<TestGenSample>,
    /// Test samples (holdout)
    pub test: Vec<TestGenSample>,
}

/// Corpus statistics
#[derive(Debug, Clone)]
pub struct CorpusStats {
    /// Total number of samples
    pub total_samples: usize,
    /// Training samples
    pub train_samples: usize,
    /// Validation samples
    pub validation_samples: usize,
    /// Test samples
    pub test_samples: usize,
    /// Samples with property tests
    pub with_proptest: usize,
    /// Samples with generics
    pub with_generics: usize,
    /// Samples with lifetimes
    pub with_lifetimes: usize,
    /// Samples with async
    pub with_async: usize,
    /// Average function length (chars)
    pub avg_function_len: usize,
    /// Average test length (chars)
    pub avg_test_len: usize,
}

impl TestGenCorpus {
    /// Create empty corpus
    #[must_use]
    pub const fn new() -> Self {
        Self { train: Vec::new(), validation: Vec::new(), test: Vec::new() }
    }

    /// Load corpus from JSONL files
    ///
    /// # Errors
    ///
    /// Returns error if files cannot be read or parsed.
    pub fn load_jsonl(
        train_path: &Path,
        validation_path: &Path,
        test_path: &Path,
    ) -> Result<Self, CorpusError> {
        let train = Self::load_jsonl_file(train_path)?;
        let validation = Self::load_jsonl_file(validation_path)?;
        let test = Self::load_jsonl_file(test_path)?;

        Ok(Self { train, validation, test })
    }

    /// Load samples from a single JSONL file
    fn load_jsonl_file(path: &Path) -> Result<Vec<TestGenSample>, CorpusError> {
        let content =
            std::fs::read_to_string(path).map_err(|e| CorpusError::IoError(e.to_string()))?;

        let mut samples = Vec::new();
        for (line_num, line) in content.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let sample: TestGenSample = serde_json::from_str(line).map_err(|e| {
                CorpusError::ParseError { line: line_num + 1, message: e.to_string() }
            })?;
            samples.push(sample);
        }

        Ok(samples)
    }

    /// Create mock corpus for testing
    #[must_use]
    pub fn mock(train_size: usize, val_size: usize, test_size: usize) -> Self {
        let make_samples = |n: usize| -> Vec<TestGenSample> {
            (0..n)
                .map(|i| TestGenSample {
                    function: format!(
                        "/// Sample function {i}\npub fn sample_{i}(x: i32) -> i32 {{ x + {i} }}"
                    ),
                    unit_tests: format!(
                        "#[test]\nfn test_sample_{i}() {{ assert_eq!(sample_{i}(0), {i}); }}"
                    ),
                    property_tests: if i % 4 == 0 {
                        Some(format!(
                            "proptest! {{ #[test] fn prop_{i}(x in any::<i32>()) {{ prop_assert!(sample_{i}(x) >= x); }} }}"
                        ))
                    } else {
                        None
                    },
                    metadata: SampleMetadata {
                        crate_name: Some(format!("crate_{}", i % 10)),
                        complexity: Some((i % 15) as u32 + 1),
                        has_generics: i % 5 == 0,
                        has_lifetimes: i % 7 == 0,
                        is_async: i % 10 == 0,
                    },
                })
                .collect()
        };

        Self {
            train: make_samples(train_size),
            validation: make_samples(val_size),
            test: make_samples(test_size),
        }
    }

    /// Get corpus statistics
    #[must_use]
    pub fn stats(&self) -> CorpusStats {
        let all: Vec<&TestGenSample> =
            self.train.iter().chain(self.validation.iter()).chain(self.test.iter()).collect();

        let total = all.len();
        if total == 0 {
            return CorpusStats {
                total_samples: 0,
                train_samples: 0,
                validation_samples: 0,
                test_samples: 0,
                with_proptest: 0,
                with_generics: 0,
                with_lifetimes: 0,
                with_async: 0,
                avg_function_len: 0,
                avg_test_len: 0,
            };
        }

        let with_proptest = all.iter().filter(|s| s.property_tests.is_some()).count();
        let with_generics = all.iter().filter(|s| s.metadata.has_generics).count();
        let with_lifetimes = all.iter().filter(|s| s.metadata.has_lifetimes).count();
        let with_async = all.iter().filter(|s| s.metadata.is_async).count();

        let total_fn_len: usize = all.iter().map(|s| s.function.len()).sum();
        let total_test_len: usize = all.iter().map(|s| s.unit_tests.len()).sum();

        CorpusStats {
            total_samples: total,
            train_samples: self.train.len(),
            validation_samples: self.validation.len(),
            test_samples: self.test.len(),
            with_proptest,
            with_generics,
            with_lifetimes,
            with_async,
            avg_function_len: total_fn_len / total,
            avg_test_len: total_test_len / total,
        }
    }

    /// Total number of samples
    #[must_use]
    pub fn len(&self) -> usize {
        self.train.len() + self.validation.len() + self.test.len()
    }

    /// Check if corpus is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.train.is_empty() && self.validation.is_empty() && self.test.is_empty()
    }

    /// Shuffle training data with seed
    pub fn shuffle_train(&mut self, seed: u64) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Simple Fisher-Yates with deterministic pseudo-random
        let n = self.train.len();
        for i in (1..n).rev() {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            let j = (hasher.finish() as usize) % (i + 1);
            self.train.swap(i, j);
        }
    }

    /// Format sample as prompt for model
    #[must_use]
    pub fn format_prompt(sample: &TestGenSample) -> String {
        format!(
            "<|im_start|>system\n\
            You are a Rust testing expert. Generate comprehensive unit tests and property-based tests.\n\
            <|im_end|>\n\
            <|im_start|>user\n\
            Generate tests for this function:\n\n\
            ```rust\n{}\n```\n\
            <|im_end|>\n\
            <|im_start|>assistant\n",
            sample.function
        )
    }

    /// Format sample as target for training
    #[must_use]
    pub fn format_target(sample: &TestGenSample) -> String {
        let mut target = sample.unit_tests.clone();
        if let Some(ref prop) = sample.property_tests {
            target.push_str("\n\n");
            target.push_str(prop);
        }
        target.push_str("\n<|im_end|>");
        target
    }
}

impl Default for TestGenCorpus {
    fn default() -> Self {
        Self::new()
    }
}

/// Corpus loading error
#[derive(Debug, Clone)]
pub enum CorpusError {
    /// IO error
    IoError(String),
    /// JSON parse error
    ParseError { line: usize, message: String },
    /// Invalid format
    InvalidFormat(String),
}

impl std::fmt::Display for CorpusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(msg) => write!(f, "IO error: {msg}"),
            Self::ParseError { line, message } => {
                write!(f, "Parse error at line {line}: {message}")
            }
            Self::InvalidFormat(msg) => write!(f, "Invalid format: {msg}"),
        }
    }
}

impl std::error::Error for CorpusError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corpus_new() {
        let corpus = TestGenCorpus::new();
        assert!(corpus.is_empty());
        assert_eq!(corpus.len(), 0);
    }

    #[test]
    fn test_corpus_mock() {
        let corpus = TestGenCorpus::mock(100, 20, 20);
        assert_eq!(corpus.train.len(), 100);
        assert_eq!(corpus.validation.len(), 20);
        assert_eq!(corpus.test.len(), 20);
        assert_eq!(corpus.len(), 140);
        assert!(!corpus.is_empty());
    }

    #[test]
    fn test_corpus_stats() {
        let corpus = TestGenCorpus::mock(80, 10, 10);
        let stats = corpus.stats();

        assert_eq!(stats.total_samples, 100);
        assert_eq!(stats.train_samples, 80);
        assert_eq!(stats.validation_samples, 10);
        assert_eq!(stats.test_samples, 10);
        assert!(stats.with_proptest > 0);
        assert!(stats.avg_function_len > 0);
        assert!(stats.avg_test_len > 0);
    }

    #[test]
    fn test_corpus_stats_empty() {
        let corpus = TestGenCorpus::new();
        let stats = corpus.stats();
        assert_eq!(stats.total_samples, 0);
        assert_eq!(stats.avg_function_len, 0);
    }

    #[test]
    fn test_corpus_shuffle_deterministic() {
        let mut corpus1 = TestGenCorpus::mock(50, 0, 0);
        let mut corpus2 = TestGenCorpus::mock(50, 0, 0);

        corpus1.shuffle_train(42);
        corpus2.shuffle_train(42);

        // Same seed should produce same order
        for (a, b) in corpus1.train.iter().zip(corpus2.train.iter()) {
            assert_eq!(a.function, b.function);
        }
    }

    #[test]
    fn test_corpus_shuffle_different_seeds() {
        let mut corpus1 = TestGenCorpus::mock(50, 0, 0);
        let mut corpus2 = TestGenCorpus::mock(50, 0, 0);

        corpus1.shuffle_train(42);
        corpus2.shuffle_train(123);

        // Different seeds should produce different order
        let same_count = corpus1
            .train
            .iter()
            .zip(corpus2.train.iter())
            .filter(|(a, b)| a.function == b.function)
            .count();

        // Some might match by chance, but not all
        assert!(same_count < 50);
    }

    #[test]
    fn test_sample_serialization() {
        let sample = TestGenSample {
            function: "pub fn foo() {}".into(),
            unit_tests: "#[test] fn test_foo() {}".into(),
            property_tests: Some("proptest! {}".into()),
            metadata: SampleMetadata {
                crate_name: Some("test".into()),
                complexity: Some(5),
                has_generics: true,
                has_lifetimes: false,
                is_async: false,
            },
        };

        let json = serde_json::to_string(&sample).expect("JSON serialization should succeed");
        let restored: TestGenSample =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");

        assert_eq!(restored.function, sample.function);
        assert_eq!(restored.unit_tests, sample.unit_tests);
        assert_eq!(restored.property_tests, sample.property_tests);
        assert!(restored.metadata.has_generics);
    }

    #[test]
    fn test_format_prompt() {
        let sample = TestGenSample {
            function: "pub fn add(a: i32, b: i32) -> i32 { a + b }".into(),
            unit_tests: String::new(),
            property_tests: None,
            metadata: SampleMetadata::default(),
        };

        let prompt = TestGenCorpus::format_prompt(&sample);
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("pub fn add"));
        assert!(prompt.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_format_target() {
        let sample = TestGenSample {
            function: String::new(),
            unit_tests: "#[test] fn test() {}".into(),
            property_tests: Some("proptest! {}".into()),
            metadata: SampleMetadata::default(),
        };

        let target = TestGenCorpus::format_target(&sample);
        assert!(target.contains("#[test]"));
        assert!(target.contains("proptest!"));
        assert!(target.ends_with("<|im_end|>"));
    }

    #[test]
    fn test_corpus_error_display() {
        let io_err = CorpusError::IoError("file not found".into());
        assert!(io_err.to_string().contains("IO error"));

        let parse_err = CorpusError::ParseError { line: 5, message: "invalid json".into() };
        assert!(parse_err.to_string().contains("line 5"));
    }

    #[test]
    fn test_mock_metadata_distribution() {
        let corpus = TestGenCorpus::mock(100, 0, 0);
        let stats = corpus.stats();

        // ~20% should have generics (every 5th)
        assert!(stats.with_generics >= 15 && stats.with_generics <= 25);

        // ~25% should have proptest (every 4th)
        assert!(stats.with_proptest >= 20 && stats.with_proptest <= 30);

        // ~10% should be async (every 10th)
        assert!(stats.with_async >= 8 && stats.with_async <= 12);
    }

    #[test]
    fn test_corpus_error_invalid_format() {
        let err = CorpusError::InvalidFormat("bad format".into());
        assert!(err.to_string().contains("Invalid format"));
        assert!(err.to_string().contains("bad format"));
    }

    #[test]
    fn test_sample_metadata_default() {
        let meta = SampleMetadata::default();
        assert!(meta.crate_name.is_none());
        assert!(meta.complexity.is_none());
        assert!(!meta.has_generics);
        assert!(!meta.has_lifetimes);
        assert!(!meta.is_async);
    }

    #[test]
    fn test_corpus_default() {
        let corpus = TestGenCorpus::default();
        assert!(corpus.is_empty());
        assert_eq!(corpus.len(), 0);
    }

    #[test]
    fn test_format_target_without_proptest() {
        let sample = TestGenSample {
            function: String::new(),
            unit_tests: "#[test] fn test() { assert!(true); }".into(),
            property_tests: None,
            metadata: SampleMetadata::default(),
        };

        let target = TestGenCorpus::format_target(&sample);
        assert!(target.contains("#[test]"));
        assert!(!target.contains("proptest!"));
        assert!(target.ends_with("<|im_end|>"));
    }

    #[test]
    fn test_corpus_stats_with_lifetimes() {
        let corpus = TestGenCorpus::mock(7, 0, 0);
        let stats = corpus.stats();
        // Every 7th sample has lifetimes (i % 7 == 0)
        assert!(stats.with_lifetimes >= 1);
    }

    #[test]
    fn test_sample_with_all_metadata() {
        let sample = TestGenSample {
            function: "pub fn foo<T: Clone + 'a>(x: &'a T) -> T { x.clone() }".into(),
            unit_tests: "#[test] fn test() {}".into(),
            property_tests: Some("proptest! {}".into()),
            metadata: SampleMetadata {
                crate_name: Some("my_crate".into()),
                complexity: Some(15),
                has_generics: true,
                has_lifetimes: true,
                is_async: false,
            },
        };

        assert!(sample.metadata.has_generics);
        assert!(sample.metadata.has_lifetimes);
        assert_eq!(sample.metadata.complexity, Some(15));
    }

    #[test]
    fn test_corpus_load_jsonl_nonexistent() {
        let result = TestGenCorpus::load_jsonl(
            std::path::Path::new("/nonexistent/train.jsonl"),
            std::path::Path::new("/nonexistent/val.jsonl"),
            std::path::Path::new("/nonexistent/test.jsonl"),
        );
        assert!(result.is_err());
    }
}
