//! Jidoka Index Validator - Stop-on-Error Guarantees
//!
//! Implements Toyota Way Jidoka (自働化) principle for automatic defect detection.
//! Addresses common failure points in RAG engineering per Barnett et al. (2024).

use super::types::JidokaHalt;
use std::collections::HashMap;

/// Jidoka index validator for stop-on-error guarantees
///
/// Validates:
/// - Embedding dimensions match model
/// - No NaN/Inf in embeddings (Poka-Yoke)
/// - Document hashes match content (integrity)
#[derive(Debug)]
pub struct JidokaIndexValidator {
    /// Expected embedding dimensions
    expected_dims: usize,
    /// Model hash for verification
    model_hash: Option<[u8; 32]>,
    /// Validation statistics
    stats: ValidationStats,
}

/// Validation statistics
#[derive(Debug, Default, Clone)]
pub struct ValidationStats {
    /// Total validations performed
    pub total_validations: u64,
    /// Successful validations
    pub successful: u64,
    /// Failed validations
    pub failed: u64,
    /// Halts triggered
    pub halts: u64,
}

impl JidokaIndexValidator {
    /// Create a new validator with expected embedding dimensions
    pub fn new(expected_dims: usize) -> Self {
        Self { expected_dims, model_hash: None, stats: ValidationStats::default() }
    }

    /// Set expected model hash
    pub fn with_model_hash(mut self, hash: [u8; 32]) -> Self {
        self.model_hash = Some(hash);
        self
    }

    /// Run a validation check, updating stats based on the result.
    ///
    /// Increments `total_validations` unconditionally, then `failed`+`halts`
    /// on `Err` or `successful` on `Ok`.
    fn run_check(
        &mut self,
        check: impl FnOnce() -> Result<(), JidokaHalt>,
    ) -> Result<(), JidokaHalt> {
        self.stats.total_validations += 1;
        match check() {
            Ok(()) => {
                self.stats.successful += 1;
                Ok(())
            }
            Err(e) => {
                self.stats.failed += 1;
                self.stats.halts += 1;
                Err(e)
            }
        }
    }

    /// Validate an embedding vector
    pub fn validate_embedding(
        &mut self,
        doc_id: &str,
        embedding: &[f32],
    ) -> Result<(), JidokaHalt> {
        let expected_dims = self.expected_dims;
        self.run_check(|| {
            // Check dimensions
            if embedding.len() != expected_dims {
                return Err(JidokaHalt::DimensionMismatch {
                    expected: expected_dims,
                    actual: embedding.len(),
                });
            }

            // Check for NaN/Inf (Poka-Yoke)
            if embedding.iter().any(|v| v.is_nan() || v.is_infinite()) {
                return Err(JidokaHalt::CorruptedEmbedding { doc_id: doc_id.to_string() });
            }

            Ok(())
        })
    }

    /// Validate document content integrity
    pub fn validate_integrity(
        &mut self,
        doc_id: &str,
        content: &[u8],
        stored_hash: [u8; 32],
    ) -> Result<(), JidokaHalt> {
        self.run_check(|| {
            let computed_hash = compute_hash(content);
            if computed_hash != stored_hash {
                return Err(JidokaHalt::IntegrityViolation { doc_id: doc_id.to_string() });
            }
            Ok(())
        })
    }

    /// Validate model hash matches expected
    pub fn validate_model_hash(&mut self, actual_hash: [u8; 32]) -> Result<(), JidokaHalt> {
        let model_hash = self.model_hash;
        self.run_check(|| {
            if let Some(expected) = model_hash {
                if expected != actual_hash {
                    return Err(JidokaHalt::ModelMismatch {
                        expected: hex_encode(&expected),
                        actual: hex_encode(&actual_hash),
                    });
                }
            }
            Ok(())
        })
    }

    /// Validate a batch of embeddings
    pub fn validate_batch(
        &mut self,
        embeddings: &HashMap<String, Vec<f32>>,
    ) -> Result<(), JidokaHalt> {
        for (doc_id, embedding) in embeddings {
            self.validate_embedding(doc_id, embedding)?;
        }
        Ok(())
    }

    /// Get validation statistics
    pub fn stats(&self) -> &ValidationStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = ValidationStats::default();
    }

    /// Get expected dimensions
    pub fn expected_dims(&self) -> usize {
        self.expected_dims
    }
}

/// Fallback strategy when Jidoka halts occur
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FallbackStrategy {
    /// Serve from last validated index
    #[default]
    LastKnownGood,
    /// Serve from in-memory cache
    CacheOnly,
    /// Return "index unavailable" error
    Unavailable,
}

/// Jidoka halt handler
#[derive(Debug)]
pub struct JidokaHaltHandler {
    /// Fallback strategy
    strategy: FallbackStrategy,
    /// Halt history for debugging
    halt_history: Vec<HaltRecord>,
    /// Maximum history size
    max_history: usize,
}

/// Record of a Jidoka halt
#[derive(Debug, Clone)]
pub struct HaltRecord {
    /// Timestamp (Unix epoch ms)
    pub timestamp_ms: u64,
    /// Halt reason
    pub halt: JidokaHalt,
    /// Recovery action taken
    pub recovery_action: String,
}

impl JidokaHaltHandler {
    /// Create a new halt handler
    pub fn new(strategy: FallbackStrategy) -> Self {
        Self { strategy, halt_history: Vec::new(), max_history: 100 }
    }

    /// Handle a Jidoka halt
    pub fn handle_halt(&mut self, halt: JidokaHalt) -> FallbackStrategy {
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let recovery_action = match self.strategy {
            FallbackStrategy::LastKnownGood => "Rolling back to last validated index".to_string(),
            FallbackStrategy::CacheOnly => "Serving from in-memory cache".to_string(),
            FallbackStrategy::Unavailable => "Index marked unavailable".to_string(),
        };

        self.halt_history.push(HaltRecord { timestamp_ms, halt, recovery_action });

        // Trim history
        if self.halt_history.len() > self.max_history {
            self.halt_history.remove(0);
        }

        self.strategy
    }

    /// Get recent halts
    pub fn recent_halts(&self, count: usize) -> &[HaltRecord] {
        let start = self.halt_history.len().saturating_sub(count);
        &self.halt_history[start..]
    }

    /// Get halt count
    pub fn halt_count(&self) -> usize {
        self.halt_history.len()
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.halt_history.clear();
    }
}

impl Default for JidokaHaltHandler {
    fn default() -> Self {
        Self::new(FallbackStrategy::default())
    }
}

/// Compute hash for content (same algorithm as fingerprint)
fn compute_hash(data: &[u8]) -> [u8; 32] {
    let mut hash = [0u8; 32];
    let mut state: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in data {
        state ^= byte as u64;
        state = state.wrapping_mul(0x0100_0000_01b3);
    }
    for i in 0..4 {
        let chunk = state.wrapping_add(i as u64).to_le_bytes();
        hash[i * 8..(i + 1) * 8].copy_from_slice(&chunk);
    }
    hash
}

/// Encode hash as hex string
fn hex_encode(hash: &[u8; 32]) -> String {
    hash.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Standard test dimension used across unit tests.
    const TEST_DIM: usize = 4;

    /// A valid embedding of `TEST_DIM` elements.
    const VALID_EMBEDDING: [f32; TEST_DIM] = [0.1, 0.2, 0.3, 0.4];

    /// Create a fresh validator with `TEST_DIM` expected dimensions.
    fn test_validator() -> JidokaIndexValidator {
        JidokaIndexValidator::new(TEST_DIM)
    }

    /// Build a `TEST_DIM`-length embedding with a single poisoned value at position 1.
    fn poisoned_embedding(bad_value: f32) -> Vec<f32> {
        let mut v = VALID_EMBEDDING.to_vec();
        v[1] = bad_value;
        v
    }

    /// Assert that validating an embedding with a single non-finite value
    /// at position 1 yields `CorruptedEmbedding`.
    fn assert_corrupted_embedding(bad_value: f32) {
        let mut validator = test_validator();
        let embedding = poisoned_embedding(bad_value);
        let result = validator.validate_embedding("doc1", &embedding);
        assert!(matches!(result, Err(JidokaHalt::CorruptedEmbedding { .. })));
    }

    #[test]
    fn test_validator_creation() {
        let validator = JidokaIndexValidator::new(384);
        assert_eq!(validator.expected_dims(), 384);
    }

    #[test]
    fn test_validate_correct_embedding() {
        let mut validator = test_validator();
        let result = validator.validate_embedding("doc1", &VALID_EMBEDDING);
        assert!(result.is_ok());
        assert_eq!(validator.stats().successful, 1);
    }

    #[test]
    fn test_validate_wrong_dimensions() {
        let mut validator = test_validator();
        let embedding = vec![0.1, 0.2]; // Wrong size

        let result = validator.validate_embedding("doc1", &embedding);
        assert!(matches!(result, Err(JidokaHalt::DimensionMismatch { expected: 4, actual: 2 })));
        assert_eq!(validator.stats().halts, 1);
    }

    #[test]
    fn test_validate_nan_embedding() {
        assert_corrupted_embedding(f32::NAN);
    }

    #[test]
    fn test_validate_inf_embedding() {
        assert_corrupted_embedding(f32::INFINITY);
    }

    #[test]
    fn test_validate_neg_inf_embedding() {
        assert_corrupted_embedding(f32::NEG_INFINITY);
    }

    #[test]
    fn test_validate_integrity_correct() {
        let mut validator = test_validator();
        let content = b"test content";
        let hash = compute_hash(content);

        let result = validator.validate_integrity("doc1", content, hash);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_integrity_mismatch() {
        let mut validator = test_validator();
        let content = b"test content";
        let wrong_hash = [0u8; 32];

        let result = validator.validate_integrity("doc1", content, wrong_hash);
        assert!(matches!(result, Err(JidokaHalt::IntegrityViolation { .. })));
    }

    #[test]
    fn test_validate_model_hash() {
        let expected_hash = [1u8; 32];
        let mut validator = test_validator().with_model_hash(expected_hash);

        // Correct hash
        let result = validator.validate_model_hash(expected_hash);
        assert!(result.is_ok());

        // Wrong hash
        let result = validator.validate_model_hash([2u8; 32]);
        assert!(matches!(result, Err(JidokaHalt::ModelMismatch { .. })));
    }

    #[test]
    fn test_validate_batch() {
        let mut validator = test_validator();
        let mut embeddings = HashMap::new();
        embeddings.insert("doc1".to_string(), VALID_EMBEDDING.to_vec());
        embeddings.insert("doc2".to_string(), vec![0.5, 0.6, 0.7, 0.8]);

        let result = validator.validate_batch(&embeddings);
        assert!(result.is_ok());
        assert_eq!(validator.stats().successful, 2);
    }

    #[test]
    fn test_validate_batch_with_error() {
        let mut validator = test_validator();
        let mut embeddings = HashMap::new();
        embeddings.insert("doc1".to_string(), VALID_EMBEDDING.to_vec());
        embeddings.insert("doc2".to_string(), poisoned_embedding(f32::NAN));

        let result = validator.validate_batch(&embeddings);
        assert!(result.is_err());
    }

    #[test]
    fn test_halt_handler() {
        let mut handler = JidokaHaltHandler::new(FallbackStrategy::LastKnownGood);

        let halt = JidokaHalt::CorruptedEmbedding { doc_id: "doc1".to_string() };
        let strategy = handler.handle_halt(halt);

        assert_eq!(strategy, FallbackStrategy::LastKnownGood);
        assert_eq!(handler.halt_count(), 1);
    }

    #[test]
    fn test_halt_handler_history() {
        let mut handler = JidokaHaltHandler::new(FallbackStrategy::CacheOnly);

        for i in 0..5 {
            handler.handle_halt(JidokaHalt::CorruptedEmbedding { doc_id: format!("doc{}", i) });
        }

        let recent = handler.recent_halts(3);
        assert_eq!(recent.len(), 3);
    }

    #[test]
    fn test_fallback_strategy_default() {
        assert_eq!(FallbackStrategy::default(), FallbackStrategy::LastKnownGood);
    }

    #[test]
    fn test_reset_stats() {
        let mut validator = test_validator();
        validator.validate_embedding("doc1", &VALID_EMBEDDING).expect("unexpected failure");

        assert_eq!(validator.stats().successful, 1);

        validator.reset_stats();
        assert_eq!(validator.stats().successful, 0);
    }

    #[test]
    fn test_hex_encode() {
        let hash = [
            0x12, 0x34, 0xab, 0xcd, 0x00, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
        ];
        let hex = hex_encode(&hash);
        assert!(hex.starts_with("1234abcd00ff"));
    }

    // Property-based tests for Jidoka validator
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        /// Build a sequential embedding of the given dimension and inject a
        /// poison value at position `pos % dim`.
        fn embedding_with_poison(dim: usize, pos: usize, poison: f32) -> Vec<f32> {
            let mut v: Vec<f32> = (0..dim).map(|i| i as f32 / 100.0).collect();
            v[pos % dim] = poison;
            v
        }

        /// Assert that a poisoned embedding always fails validation.
        fn assert_poison_fails(
            dim: usize,
            pos: usize,
            poison: f32,
        ) -> Result<(), proptest::test_runner::TestCaseError> {
            let mut validator = JidokaIndexValidator::new(dim);
            let embedding = embedding_with_poison(dim, pos, poison);
            let result = validator.validate_embedding("test_doc", &embedding);
            prop_assert!(result.is_err());
            Ok(())
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(50))]

            /// Property: Valid embeddings always pass validation
            #[test]
            fn prop_valid_embeddings_pass(
                values in prop::collection::vec(-1.0f32..1.0, 4..64)
            ) {
                let mut validator = JidokaIndexValidator::new(values.len());
                let result = validator.validate_embedding("test_doc", &values);
                prop_assert!(result.is_ok());
            }

            /// Property: Wrong dimension always fails
            #[test]
            fn prop_wrong_dim_fails(
                expected_dim in 64usize..128,
                actual_dim in 1usize..32
            ) {
                let mut validator = JidokaIndexValidator::new(expected_dim);
                let embedding: Vec<f32> = (0..actual_dim).map(|i| i as f32 / 100.0).collect();
                let result = validator.validate_embedding("test_doc", &embedding);
                prop_assert!(result.is_err());
            }

            /// Property: NaN values always fail
            #[test]
            fn prop_nan_fails(dim in 4usize..64, nan_pos in 0usize..4) {
                assert_poison_fails(dim, nan_pos, f32::NAN)?;
            }

            /// Property: Infinite values always fail
            #[test]
            fn prop_inf_fails(dim in 4usize..64, inf_pos in 0usize..4) {
                assert_poison_fails(dim, inf_pos, f32::INFINITY)?;
            }

            /// Property: Stats correctly count validations
            #[test]
            fn prop_stats_count_validations(
                valid_count in 0u64..10,
                invalid_count in 0u64..10
            ) {
                let mut validator = JidokaIndexValidator::new(TEST_DIM);

                for i in 0..valid_count {
                    validator.validate_embedding(&format!("valid_{}", i), &VALID_EMBEDDING).ok();
                }
                for i in 0..invalid_count {
                    validator.validate_embedding(&format!("invalid_{}", i), &[0.1]).ok();
                }

                let stats = validator.stats();
                prop_assert_eq!(stats.total_validations, valid_count + invalid_count);
                prop_assert_eq!(stats.successful, valid_count);
                prop_assert_eq!(stats.failed, invalid_count);
            }

            /// Property: Reset clears all stats
            #[test]
            fn prop_reset_clears_stats(count in 1u64..20) {
                let mut validator = JidokaIndexValidator::new(TEST_DIM);

                for i in 0..count {
                    validator.validate_embedding(&format!("doc_{}", i), &VALID_EMBEDDING).ok();
                }

                validator.reset_stats();
                let stats = validator.stats();
                prop_assert_eq!(stats.total_validations, 0);
                prop_assert_eq!(stats.successful, 0);
                prop_assert_eq!(stats.failed, 0);
            }
        }
    }
}
