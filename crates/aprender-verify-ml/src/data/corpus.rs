//! Corpus management for verified tuples
//!
//! Provides loading, export, and integration with depyler-oracle training pipeline.
//! See VERIFICAR-003.

use crate::data::VerifiedTuple;
use crate::ml::CommitFeatures;
use crate::Language;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A training corpus containing verified tuples with associated features
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingCorpus {
    /// Verified tuples (source, target, correctness)
    pub tuples: Vec<VerifiedTuple>,
    /// Associated commit features (8-dim vectors)
    pub features: Vec<CommitFeatures>,
    /// Corpus metadata
    pub metadata: CorpusMetadata,
}

/// Metadata about the corpus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusMetadata {
    /// Corpus version
    pub version: String,
    /// Source language
    pub source_language: Language,
    /// Target language
    pub target_language: Language,
    /// Total number of examples
    pub count: usize,
    /// Number of correct translations
    pub correct_count: usize,
    /// Number of incorrect translations
    pub incorrect_count: usize,
    /// Generation timestamp (Unix epoch)
    pub timestamp: u64,
}

impl Default for CorpusMetadata {
    fn default() -> Self {
        Self {
            version: String::new(),
            source_language: Language::Python,
            target_language: Language::Rust,
            count: 0,
            correct_count: 0,
            incorrect_count: 0,
            timestamp: 0,
        }
    }
}

impl CorpusMetadata {
    /// Calculate accuracy ratio
    #[must_use]
    pub fn accuracy(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.correct_count as f64 / self.count as f64
        }
    }
}

/// Corpus format for export
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CorpusFormat {
    /// JSON format (single file)
    Json,
    /// JSON Lines format (one record per line)
    #[default]
    Jsonl,
    /// Parquet format (columnar, efficient)
    Parquet,
}

/// Corpus loader and exporter
#[derive(Debug, Default)]
pub struct CorpusManager {
    corpus: TrainingCorpus,
}

impl CorpusManager {
    /// Create a new empty corpus manager
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from existing corpus
    #[must_use]
    pub fn from_corpus(corpus: TrainingCorpus) -> Self {
        Self { corpus }
    }

    /// Get reference to corpus
    #[must_use]
    pub fn corpus(&self) -> &TrainingCorpus {
        &self.corpus
    }

    /// Add a verified tuple with features
    pub fn add(&mut self, tuple: VerifiedTuple, features: CommitFeatures) {
        if tuple.is_correct {
            self.corpus.metadata.correct_count += 1;
        } else {
            self.corpus.metadata.incorrect_count += 1;
        }
        self.corpus.metadata.count += 1;
        self.corpus.tuples.push(tuple);
        self.corpus.features.push(features);
    }

    /// Add multiple tuples without features (features will be default)
    pub fn add_tuples(&mut self, tuples: Vec<VerifiedTuple>) {
        for tuple in tuples {
            let features = CommitFeatures::default();
            self.add(tuple, features);
        }
    }

    /// Set corpus metadata
    pub fn set_metadata(&mut self, metadata: CorpusMetadata) {
        self.corpus.metadata = metadata;
    }

    /// Export corpus to file
    ///
    /// # Errors
    ///
    /// Returns error if export fails
    pub fn export(&self, path: &Path, format: CorpusFormat) -> std::io::Result<()> {
        match format {
            CorpusFormat::Json => self.export_json(path),
            CorpusFormat::Jsonl => self.export_jsonl(path),
            CorpusFormat::Parquet => {
                // Fallback to JSONL for now
                self.export_jsonl(path)
            }
        }
    }

    /// Load corpus from file
    ///
    /// # Errors
    ///
    /// Returns error if loading fails
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;

        // Try JSON first
        if let Ok(corpus) = serde_json::from_str::<TrainingCorpus>(&content) {
            return Ok(Self::from_corpus(corpus));
        }

        // Try JSONL
        let mut corpus = TrainingCorpus::default();
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(record) = serde_json::from_str::<CorpusRecord>(line) {
                let is_correct = record.tuple.is_correct;
                corpus.tuples.push(record.tuple);
                corpus.features.push(record.features);
                corpus.metadata.count += 1;
                if is_correct {
                    corpus.metadata.correct_count += 1;
                } else {
                    corpus.metadata.incorrect_count += 1;
                }
            }
        }

        Ok(Self::from_corpus(corpus))
    }

    fn export_json(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self.corpus)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    fn export_jsonl(&self, path: &Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        for (tuple, features) in self.corpus.tuples.iter().zip(self.corpus.features.iter()) {
            let record = CorpusRecord {
                tuple: tuple.clone(),
                features: features.clone(),
            };
            let line = serde_json::to_string(&record)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            writeln!(file, "{line}")?;
        }

        Ok(())
    }

    /// Convert to depyler-oracle compatible format
    ///
    /// Returns (feature_matrix, labels) for ML training
    #[must_use]
    pub fn to_training_data(&self) -> (Vec<[f32; 8]>, Vec<u8>) {
        let features: Vec<[f32; 8]> = self
            .corpus
            .features
            .iter()
            .map(CommitFeatures::to_array)
            .collect();

        let labels: Vec<u8> = self
            .corpus
            .tuples
            .iter()
            .map(|t| u8::from(t.is_correct))
            .collect();

        (features, labels)
    }

    /// Split corpus into train/test sets
    #[must_use]
    pub fn train_test_split(
        &self,
        train_ratio: f64,
        seed: u64,
    ) -> (TrainingCorpus, TrainingCorpus) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut train = TrainingCorpus::default();
        let mut test = TrainingCorpus::default();

        for (i, (tuple, features)) in self
            .corpus
            .tuples
            .iter()
            .zip(self.corpus.features.iter())
            .enumerate()
        {
            let mut hasher = DefaultHasher::new();
            (seed, i).hash(&mut hasher);
            let hash = hasher.finish();

            #[allow(clippy::cast_sign_loss)]
            let threshold = (train_ratio * u64::MAX as f64) as u64;

            let target = if hash < threshold {
                &mut train
            } else {
                &mut test
            };

            target.tuples.push(tuple.clone());
            target.features.push(features.clone());
            target.metadata.count += 1;
            if tuple.is_correct {
                target.metadata.correct_count += 1;
            } else {
                target.metadata.incorrect_count += 1;
            }
        }

        (train, test)
    }

    /// Filter corpus by correctness
    #[must_use]
    pub fn filter_correct(&self, correct: bool) -> TrainingCorpus {
        let mut filtered = TrainingCorpus::default();

        for (tuple, features) in self.corpus.tuples.iter().zip(self.corpus.features.iter()) {
            if tuple.is_correct == correct {
                filtered.tuples.push(tuple.clone());
                filtered.features.push(features.clone());
                filtered.metadata.count += 1;
                if correct {
                    filtered.metadata.correct_count += 1;
                } else {
                    filtered.metadata.incorrect_count += 1;
                }
            }
        }

        filtered
    }
}

/// Single record for JSONL format
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CorpusRecord {
    tuple: VerifiedTuple,
    features: CommitFeatures,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tuple(correct: bool) -> VerifiedTuple {
        VerifiedTuple {
            source_language: Language::Python,
            target_language: Language::Rust,
            source_code: "x = 1".to_string(),
            target_code: "let x = 1;".to_string(),
            is_correct: correct,
            execution_time_ms: 10,
        }
    }

    fn sample_features() -> CommitFeatures {
        CommitFeatures {
            lines_added: 5,
            lines_deleted: 2,
            files_changed: 1,
            churn_ratio: 0.7,
            has_test_changes: false,
            complexity_delta: 1.0,
            author_experience: 0.5,
            days_since_last_change: 7.0,
        }
    }

    // ========== RED PHASE: Test corpus creation ==========

    #[test]
    fn test_corpus_manager_new() {
        let manager = CorpusManager::new();
        assert_eq!(manager.corpus().tuples.len(), 0);
        assert_eq!(manager.corpus().metadata.count, 0);
    }

    #[test]
    fn test_corpus_add_tuple() {
        let mut manager = CorpusManager::new();
        manager.add(sample_tuple(true), sample_features());

        assert_eq!(manager.corpus().tuples.len(), 1);
        assert_eq!(manager.corpus().features.len(), 1);
        assert_eq!(manager.corpus().metadata.count, 1);
        assert_eq!(manager.corpus().metadata.correct_count, 1);
    }

    #[test]
    fn test_corpus_add_incorrect() {
        let mut manager = CorpusManager::new();
        manager.add(sample_tuple(false), sample_features());

        assert_eq!(manager.corpus().metadata.incorrect_count, 1);
        assert_eq!(manager.corpus().metadata.correct_count, 0);
    }

    #[test]
    fn test_corpus_add_tuples_batch() {
        let mut manager = CorpusManager::new();
        let tuples = vec![sample_tuple(true), sample_tuple(false), sample_tuple(true)];
        manager.add_tuples(tuples);

        assert_eq!(manager.corpus().tuples.len(), 3);
        assert_eq!(manager.corpus().metadata.correct_count, 2);
        assert_eq!(manager.corpus().metadata.incorrect_count, 1);
    }

    #[test]
    fn test_corpus_metadata_accuracy() {
        let metadata = CorpusMetadata {
            count: 100,
            correct_count: 80,
            incorrect_count: 20,
            ..Default::default()
        };

        assert!((metadata.accuracy() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_corpus_metadata_accuracy_empty() {
        let metadata = CorpusMetadata::default();
        assert!(metadata.accuracy().abs() < f64::EPSILON);
    }

    // ========== RED PHASE: Test training data conversion ==========

    #[test]
    fn test_to_training_data() {
        let mut manager = CorpusManager::new();
        manager.add(sample_tuple(true), sample_features());
        manager.add(sample_tuple(false), sample_features());

        let (features, labels) = manager.to_training_data();

        assert_eq!(features.len(), 2);
        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0], 1); // correct = 1
        assert_eq!(labels[1], 0); // incorrect = 0
    }

    #[test]
    fn test_to_training_data_feature_values() {
        let mut manager = CorpusManager::new();
        let features = sample_features();
        manager.add(sample_tuple(true), features.clone());

        let (feature_matrix, _) = manager.to_training_data();

        assert_eq!(feature_matrix[0][0], 5.0); // lines_added
        assert_eq!(feature_matrix[0][1], 2.0); // lines_deleted
        assert_eq!(feature_matrix[0][2], 1.0); // files_changed
    }

    // ========== RED PHASE: Test train/test split ==========

    #[test]
    fn test_train_test_split() {
        let mut manager = CorpusManager::new();
        for _ in 0..100 {
            manager.add(sample_tuple(true), sample_features());
        }

        let (train, test) = manager.train_test_split(0.8, 42);

        assert!(train.metadata.count > 0);
        assert!(test.metadata.count > 0);
        assert_eq!(train.metadata.count + test.metadata.count, 100);

        // Approximately 80/20 split
        let train_ratio = train.metadata.count as f64 / 100.0;
        assert!(train_ratio > 0.7 && train_ratio < 0.9);
    }

    #[test]
    fn test_train_test_split_deterministic() {
        let mut manager = CorpusManager::new();
        for _ in 0..50 {
            manager.add(sample_tuple(true), sample_features());
        }

        let (train1, _) = manager.train_test_split(0.8, 42);
        let (train2, _) = manager.train_test_split(0.8, 42);

        assert_eq!(train1.metadata.count, train2.metadata.count);
    }

    // ========== RED PHASE: Test filtering ==========

    #[test]
    fn test_filter_correct() {
        let mut manager = CorpusManager::new();
        manager.add(sample_tuple(true), sample_features());
        manager.add(sample_tuple(false), sample_features());
        manager.add(sample_tuple(true), sample_features());

        let correct_only = manager.filter_correct(true);

        assert_eq!(correct_only.metadata.count, 2);
        assert_eq!(correct_only.metadata.correct_count, 2);
        assert_eq!(correct_only.metadata.incorrect_count, 0);
    }

    #[test]
    fn test_filter_incorrect() {
        let mut manager = CorpusManager::new();
        manager.add(sample_tuple(true), sample_features());
        manager.add(sample_tuple(false), sample_features());

        let incorrect_only = manager.filter_correct(false);

        assert_eq!(incorrect_only.metadata.count, 1);
        assert_eq!(incorrect_only.metadata.incorrect_count, 1);
    }

    // ========== RED PHASE: Test export/load ==========

    #[test]
    fn test_export_load_jsonl() {
        let mut manager = CorpusManager::new();
        manager.add(sample_tuple(true), sample_features());
        manager.add(sample_tuple(false), sample_features());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corpus.jsonl");

        manager.export(&path, CorpusFormat::Jsonl).unwrap();
        assert!(path.exists());

        let loaded = CorpusManager::load(&path).unwrap();
        assert_eq!(loaded.corpus().tuples.len(), 2);
        assert_eq!(loaded.corpus().metadata.count, 2);
    }

    #[test]
    fn test_export_load_json() {
        let mut manager = CorpusManager::new();
        manager.add(sample_tuple(true), sample_features());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corpus.json");

        manager.export(&path, CorpusFormat::Json).unwrap();
        let loaded = CorpusManager::load(&path).unwrap();

        assert_eq!(loaded.corpus().tuples.len(), 1);
    }

    // ========== RED PHASE: Test metadata ==========

    #[test]
    fn test_set_metadata() {
        let mut manager = CorpusManager::new();
        let metadata = CorpusMetadata {
            version: "1.0.0".to_string(),
            source_language: Language::Python,
            target_language: Language::Rust,
            count: 0,
            correct_count: 0,
            incorrect_count: 0,
            timestamp: 1700000000,
        };

        manager.set_metadata(metadata);

        assert_eq!(manager.corpus().metadata.version, "1.0.0");
        assert_eq!(manager.corpus().metadata.timestamp, 1700000000);
    }

    #[test]
    fn test_corpus_format_default() {
        let format = CorpusFormat::default();
        assert_eq!(format, CorpusFormat::Jsonl);
    }

    #[test]
    fn test_training_corpus_default() {
        let corpus = TrainingCorpus::default();
        assert!(corpus.tuples.is_empty());
        assert!(corpus.features.is_empty());
        assert_eq!(corpus.metadata.count, 0);
    }

    #[test]
    fn test_corpus_manager_debug() {
        let manager = CorpusManager::new();
        let debug = format!("{manager:?}");
        assert!(debug.contains("CorpusManager"));
    }

    #[test]
    fn test_corpus_metadata_debug() {
        let metadata = CorpusMetadata::default();
        let debug = format!("{metadata:?}");
        assert!(debug.contains("CorpusMetadata"));
    }

    #[test]
    fn test_from_corpus() {
        let corpus = TrainingCorpus {
            tuples: vec![sample_tuple(true)],
            features: vec![sample_features()],
            metadata: CorpusMetadata {
                count: 1,
                correct_count: 1,
                ..Default::default()
            },
        };

        let manager = CorpusManager::from_corpus(corpus);
        assert_eq!(manager.corpus().tuples.len(), 1);
    }
}

/// Property-based tests for corpus operations
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_tuple() -> impl Strategy<Value = VerifiedTuple> {
        (any::<bool>(), 1u64..1000).prop_map(|(correct, time)| VerifiedTuple {
            source_language: Language::Python,
            target_language: Language::Rust,
            source_code: "x = 1".to_string(),
            target_code: "let x = 1;".to_string(),
            is_correct: correct,
            execution_time_ms: time,
        })
    }

    fn arb_features() -> impl Strategy<Value = CommitFeatures> {
        (0u32..100, 0u32..100, 1u32..10, any::<bool>()).prop_map(
            |(added, deleted, files, has_tests)| CommitFeatures {
                lines_added: added,
                lines_deleted: deleted,
                files_changed: files,
                churn_ratio: added as f32 / (added + deleted + 1) as f32,
                has_test_changes: has_tests,
                complexity_delta: 0.0,
                author_experience: 0.5,
                days_since_last_change: 7.0,
            },
        )
    }

    proptest! {
        /// Corpus count always matches tuples length
        #[test]
        fn prop_count_matches_tuples(n in 1usize..50) {
            let mut manager = CorpusManager::new();
            for _ in 0..n {
                let tuple = VerifiedTuple {
                    source_language: Language::Python,
                    target_language: Language::Rust,
                    source_code: "x = 1".to_string(),
                    target_code: "let x = 1;".to_string(),
                    is_correct: true,
                    execution_time_ms: 10,
                };
                manager.add(tuple, CommitFeatures::default());
            }

            prop_assert_eq!(manager.corpus().metadata.count, n);
            prop_assert_eq!(manager.corpus().tuples.len(), n);
            prop_assert_eq!(manager.corpus().features.len(), n);
        }

        /// Correct + incorrect = total count
        #[test]
        fn prop_correct_incorrect_sum(
            correct_count in 0usize..25,
            incorrect_count in 0usize..25,
        ) {
            let mut manager = CorpusManager::new();

            for _ in 0..correct_count {
                let tuple = VerifiedTuple {
                    source_language: Language::Python,
                    target_language: Language::Rust,
                    source_code: "x = 1".to_string(),
                    target_code: "let x = 1;".to_string(),
                    is_correct: true,
                    execution_time_ms: 10,
                };
                manager.add(tuple, CommitFeatures::default());
            }

            for _ in 0..incorrect_count {
                let tuple = VerifiedTuple {
                    source_language: Language::Python,
                    target_language: Language::Rust,
                    source_code: "x = 1".to_string(),
                    target_code: "let x = 1;".to_string(),
                    is_correct: false,
                    execution_time_ms: 10,
                };
                manager.add(tuple, CommitFeatures::default());
            }

            let meta = &manager.corpus().metadata;
            prop_assert_eq!(meta.correct_count + meta.incorrect_count, meta.count);
            prop_assert_eq!(meta.correct_count, correct_count);
            prop_assert_eq!(meta.incorrect_count, incorrect_count);
        }

        /// Train/test split preserves total count
        #[test]
        fn prop_split_preserves_count(n in 1usize..100, ratio in 0.1f64..0.9) {
            let mut manager = CorpusManager::new();
            for _ in 0..n {
                let tuple = VerifiedTuple {
                    source_language: Language::Python,
                    target_language: Language::Rust,
                    source_code: "x = 1".to_string(),
                    target_code: "let x = 1;".to_string(),
                    is_correct: true,
                    execution_time_ms: 10,
                };
                manager.add(tuple, CommitFeatures::default());
            }

            let (train, test) = manager.train_test_split(ratio, 42);

            prop_assert_eq!(train.metadata.count + test.metadata.count, n);
            prop_assert_eq!(train.tuples.len() + test.tuples.len(), n);
        }

        /// Training data conversion preserves length
        #[test]
        fn prop_training_data_length(n in 1usize..50) {
            let mut manager = CorpusManager::new();
            for _ in 0..n {
                let tuple = VerifiedTuple {
                    source_language: Language::Python,
                    target_language: Language::Rust,
                    source_code: "x = 1".to_string(),
                    target_code: "let x = 1;".to_string(),
                    is_correct: true,
                    execution_time_ms: 10,
                };
                manager.add(tuple, CommitFeatures::default());
            }

            let (features, labels) = manager.to_training_data();

            prop_assert_eq!(features.len(), n);
            prop_assert_eq!(labels.len(), n);
        }

        /// Labels match correctness
        #[test]
        fn prop_labels_match_correctness(correct in proptest::collection::vec(any::<bool>(), 1..20)) {
            let mut manager = CorpusManager::new();

            for &is_correct in &correct {
                let tuple = VerifiedTuple {
                    source_language: Language::Python,
                    target_language: Language::Rust,
                    source_code: "x = 1".to_string(),
                    target_code: "let x = 1;".to_string(),
                    is_correct,
                    execution_time_ms: 10,
                };
                manager.add(tuple, CommitFeatures::default());
            }

            let (_, labels) = manager.to_training_data();

            for (expected, &actual) in correct.iter().zip(labels.iter()) {
                let expected_label = if *expected { 1 } else { 0 };
                prop_assert_eq!(expected_label, actual);
            }
        }

        /// Filter correct preserves only correct
        #[test]
        fn prop_filter_correct_only(n in 1usize..20) {
            let mut manager = CorpusManager::new();

            for i in 0..n {
                let tuple = VerifiedTuple {
                    source_language: Language::Python,
                    target_language: Language::Rust,
                    source_code: "x = 1".to_string(),
                    target_code: "let x = 1;".to_string(),
                    is_correct: i % 2 == 0, // alternating
                    execution_time_ms: 10,
                };
                manager.add(tuple, CommitFeatures::default());
            }

            let filtered = manager.filter_correct(true);

            for tuple in &filtered.tuples {
                prop_assert!(tuple.is_correct);
            }
        }

        /// Accuracy is in [0, 1]
        #[test]
        fn prop_accuracy_bounded(correct in 0usize..100, total in 1usize..200) {
            let metadata = CorpusMetadata {
                count: total,
                correct_count: correct.min(total),
                incorrect_count: total.saturating_sub(correct.min(total)),
                ..Default::default()
            };

            let acc = metadata.accuracy();
            prop_assert!(acc >= 0.0);
            prop_assert!(acc <= 1.0);
        }
    }
}
