//! Embedding layer module
//!
//! This module provides token embedding layers for transformer models.

use crate::Tensor;
use std::collections::HashMap;

/// Embedding layer
pub struct Embedding {
    /// Embedding weight (vocab_size x hidden_size)
    pub weight: Tensor,
    /// Vocabulary size
    vocab_size: usize,
    /// Hidden dimension
    hidden_size: usize,
}

impl Embedding {
    /// Create new embedding layer with initialized weights
    pub fn new(vocab_size: usize, hidden_size: usize) -> Self {
        use super::init::{get_init_seed, rand_normal_seeded};
        // C-INIT-001: normal(0, 0.02) matching HuggingFace LLaMA
        Self {
            weight: Tensor::from_vec(
                rand_normal_seeded(vocab_size * hidden_size, get_init_seed(), "embed_tokens"),
                true,
            ),
            vocab_size,
            hidden_size,
        }
    }

    /// Create from parameters
    ///
    /// # Contract (PMAT-326)
    /// Validates weight.len() == vocab_size * hidden_size.
    /// Returns None if key is missing or shape is wrong.
    pub fn from_params(
        params: &HashMap<String, Tensor>,
        name: &str,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Option<Self> {
        let weight = params.get(name)?.clone();
        let expected = vocab_size * hidden_size;
        if weight.len() != expected {
            eprintln!(
                "[PMAT-326] Embedding '{name}': shape mismatch — got {} elements, expected {expected} ({vocab_size}x{hidden_size})",
                weight.len()
            );
            return None;
        }
        Some(Self { weight, vocab_size, hidden_size })
    }

    /// Forward pass - lookup embeddings for token IDs
    ///
    /// # Arguments
    /// * `token_ids` - Token IDs to look up
    ///
    /// # Returns
    /// Embedded vectors (seq_len * hidden_size, flattened)
    pub fn forward(&self, token_ids: &[u32]) -> Tensor {
        contract_pre_embedding_lookup!(token_ids);
        let mut output = Vec::with_capacity(token_ids.len() * self.hidden_size);

        for &token_id in token_ids {
            let idx = token_id as usize;
            if idx >= self.vocab_size {
                // N-09: OOB token → zeros. Contract: embedding-lookup-v1.yaml
                eprintln!(
                    "Warning: Embedding::forward token_id {} >= vocab_size {}. N-09 OOB escape.",
                    token_id, self.vocab_size
                );
                output.extend(std::iter::repeat_n(0.0, self.hidden_size));
            } else {
                let start = idx * self.hidden_size;
                let end = start + self.hidden_size;
                output.extend_from_slice(
                    &self.weight.data().as_slice().expect("embedding weight must be contiguous")
                        [start..end],
                );
            }
        }

        let result = Tensor::from_vec(output, true);
        contract_post_embedding_lookup!(result.data().as_slice().unwrap_or(&[]));
        result
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get hidden dimension
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

/// Learned absolute position embedding for encoder models (BERT, RoBERTa, CodeBERT).
///
/// Unlike RoPE (used by decoders), encoder models learn a position embedding table
/// of shape (max_position_embeddings × hidden_size) that is added to token embeddings.
///
/// # Contract (ENC-003)
/// - Output shape: seq_len × hidden_size (same as token embedding)
/// - Positions beyond max_position_embeddings are clamped to max-1
/// - Output is added element-wise to token embeddings
pub struct LearnedPositionEmbedding {
    /// Position embedding weight (max_positions × hidden_size)
    pub weight: Tensor,
    /// Maximum number of positions
    max_positions: usize,
    /// Hidden dimension
    hidden_size: usize,
}

impl LearnedPositionEmbedding {
    /// Create new learned position embedding with deterministic initialization
    pub fn new(max_positions: usize, hidden_size: usize) -> Self {
        let scale = (1.0 / hidden_size as f32).sqrt();
        Self {
            weight: Tensor::from_vec(
                (0..max_positions * hidden_size)
                    .map(|i| ((i as f32 * 0.0731).sin() * scale))
                    .collect(),
                true,
            ),
            max_positions,
            hidden_size,
        }
    }

    /// Create from pre-trained parameters
    pub fn from_params(
        params: &HashMap<String, Tensor>,
        name: &str,
        max_positions: usize,
        hidden_size: usize,
    ) -> Option<Self> {
        let weight = params.get(name)?.clone();
        let expected = max_positions * hidden_size;
        if weight.len() != expected {
            eprintln!(
                "[ENC-003] LearnedPositionEmbedding '{name}': shape mismatch — \
                 got {} elements, expected {expected} ({max_positions}×{hidden_size})",
                weight.len()
            );
            return None;
        }
        Some(Self { weight, max_positions, hidden_size })
    }

    /// Forward pass: return position embeddings for positions 0..seq_len
    ///
    /// Output is (seq_len × hidden_size, flattened) — add element-wise to token embeddings.
    pub fn forward(&self, seq_len: usize) -> Tensor {
        let clamped_len = seq_len.min(self.max_positions);
        let weight_slice = &self.weight.data().as_slice().expect("position weight contiguous")
            [..clamped_len * self.hidden_size];
        // For positions beyond max, repeat the last position embedding
        if seq_len <= self.max_positions {
            Tensor::from_vec(weight_slice.to_vec(), true)
        } else {
            let mut output = weight_slice.to_vec();
            let last_start = (self.max_positions - 1) * self.hidden_size;
            let last_end = last_start + self.hidden_size;
            let last_pos = &self.weight.data().as_slice().expect("position weight contiguous")
                [last_start..last_end];
            for _ in self.max_positions..seq_len {
                output.extend_from_slice(last_pos);
            }
            Tensor::from_vec(output, true)
        }
    }

    /// Get maximum positions
    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    /// Get hidden dimension
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_forward() {
        let embed = Embedding::new(100, 8);
        let tokens = vec![0, 5, 10];
        let output = embed.forward(&tokens);
        assert_eq!(output.len(), 3 * 8);
    }

    #[test]
    fn test_embedding_out_of_vocab() {
        let embed = Embedding::new(100, 8);
        let tokens = vec![0, 200]; // 200 is out of vocab
        let output = embed.forward(&tokens);
        assert_eq!(output.len(), 2 * 8);
        // Out of vocab should be zeros
        let data = output.data();
        for i in 8..16 {
            assert_eq!(data[i], 0.0);
        }
    }

    #[test]
    fn test_embedding_vocab_and_hidden_size() {
        let embed = Embedding::new(500, 16);
        assert_eq!(embed.vocab_size(), 500);
        assert_eq!(embed.hidden_size(), 16);
    }

    #[test]
    fn test_embedding_single_token() {
        let embed = Embedding::new(100, 8);
        let tokens = vec![42];
        let output = embed.forward(&tokens);
        assert_eq!(output.len(), 8);
        assert!(output.requires_grad());
    }

    #[test]
    fn test_embedding_requires_grad() {
        let embed = Embedding::new(100, 8);
        assert!(embed.weight.requires_grad());
    }

    #[test]
    fn test_embedding_from_params() {
        let mut params = HashMap::new();
        params.insert("embed.weight".to_string(), Tensor::from_vec(vec![0.1; 100 * 8], true));
        let embed = Embedding::from_params(&params, "embed.weight", 100, 8);
        assert!(embed.is_some());
        let embed = embed.expect("operation should succeed");
        assert_eq!(embed.vocab_size(), 100);
        assert_eq!(embed.hidden_size(), 8);
    }

    #[test]
    fn test_embedding_from_params_missing() {
        let params: HashMap<String, Tensor> = HashMap::new();
        let embed = Embedding::from_params(&params, "missing.weight", 100, 8);
        assert!(embed.is_none());
    }

    // =========================================================================
    // ENC-003: LearnedPositionEmbedding tests
    // =========================================================================

    #[test]
    fn enc_003_learned_position_embedding_shape() {
        let pos_embed = LearnedPositionEmbedding::new(514, 768);
        assert_eq!(pos_embed.max_positions(), 514);
        assert_eq!(pos_embed.hidden_size(), 768);
        let output = pos_embed.forward(10);
        assert_eq!(output.len(), 10 * 768);
    }

    #[test]
    fn enc_003_learned_position_embedding_deterministic() {
        let pe1 = LearnedPositionEmbedding::new(128, 32);
        let pe2 = LearnedPositionEmbedding::new(128, 32);
        let o1 = pe1.forward(10);
        let o2 = pe2.forward(10);
        assert_eq!(
            o1.data().as_slice().expect("contiguous"),
            o2.data().as_slice().expect("contiguous"),
        );
    }

    #[test]
    fn enc_003_learned_position_embedding_clamp_beyond_max() {
        let pe = LearnedPositionEmbedding::new(4, 8);
        let output = pe.forward(6); // 6 > max_positions=4
        assert_eq!(output.len(), 6 * 8);
        // Positions 4 and 5 should equal position 3 (clamped)
        let data = output.data();
        let slice = data.as_slice().expect("contiguous");
        let pos3 = &slice[3 * 8..4 * 8];
        let pos4 = &slice[4 * 8..5 * 8];
        let pos5 = &slice[5 * 8..6 * 8];
        assert_eq!(pos3, pos4);
        assert_eq!(pos3, pos5);
    }

    #[test]
    fn enc_003_learned_position_from_params() {
        let mut params = HashMap::new();
        params.insert("pos.weight".to_string(), Tensor::from_vec(vec![0.1; 128 * 32], true));
        let pe = LearnedPositionEmbedding::from_params(&params, "pos.weight", 128, 32);
        assert!(pe.is_some());
    }

    #[test]
    fn enc_003_learned_position_from_params_rejects_wrong_shape() {
        let mut params = HashMap::new();
        params.insert("pos.weight".to_string(), Tensor::from_vec(vec![0.1; 50], true));
        let pe = LearnedPositionEmbedding::from_params(&params, "pos.weight", 128, 32);
        assert!(pe.is_none());
    }

    // =========================================================================
    // FALSIFY-E7: Entrenar embedding contract gap analysis (Refs PMAT-326)
    //
    // Five-Whys: §2.1.1 "What Are Embeddings" falsification sweep
    //   Why 1: Trained model could have garbage embeddings
    //   Why 2: No data quality validation during training
    //   Why 3: Embedding uses raw Tensor, not ValidatedEmbedding
    //   Why 4: entrenar predates the ValidatedEmbedding contract
    //   Why 5: No cross-crate contract enforcement test existed
    //
    // Popper (1959): "These tests try to break the claim that
    // entrenar's embedding pipeline prevents degenerate models."
    // =========================================================================

    /// FALSIFY-E7a: Embedding initialization produces non-degenerate values
    ///
    /// The init formula `(i * 0.111).sin() * scale` MUST produce varied,
    /// finite values. If it doesn't, freshly-initialized models are DOA.
    #[test]
    fn falsify_e7a_init_produces_valid_embedding() {
        let embed = Embedding::new(100, 64);
        let data = embed.weight.data();
        let slice = data.as_slice().expect("data as slice");

        // No NaN
        let nan_count = slice.iter().filter(|v| v.is_nan()).count();
        assert_eq!(nan_count, 0, "FALSIFY-E7a: Init must not produce NaN");

        // No Inf
        let inf_count = slice.iter().filter(|v| v.is_infinite()).count();
        assert_eq!(inf_count, 0, "FALSIFY-E7a: Init must not produce Inf");

        // Not all zeros (<50% zeros per embedding contract)
        let zero_count = slice.iter().filter(|v| v.abs() < 1e-10).count();
        let zero_pct = 100.0 * zero_count as f64 / slice.len() as f64;
        assert!(zero_pct < 50.0,
            "FALSIFY-E7a: Init has {zero_pct:.1}% zeros — exceeds embedding contract threshold (50%)");

        // Values vary (not constant)
        let min = slice.iter().copied().fold(f32::INFINITY, f32::min);
        let max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (max - min).abs() > 1e-6,
            "FALSIFY-E7a: Init values are constant ({min}..{max}) — degenerate embedding"
        );
    }

    /// FALSIFY-E7b: Embedding shape matches vocab * hidden
    #[test]
    fn falsify_e7b_shape_matches_dimensions() {
        let vocab_size = 151;
        let hidden_size = 32;
        let embed = Embedding::new(vocab_size, hidden_size);
        assert_eq!(
            embed.weight.len(),
            vocab_size * hidden_size,
            "FALSIFY-E7b: Embedding length must be vocab_size * hidden_size"
        );
    }

    /// FALSIFY-E7c: from_params rejects wrong-shape tensor (PMAT-326 fix)
    ///
    /// from_params now validates weight.len() == vocab_size * hidden_size.
    /// A tensor of 50 elements is rejected when 100*8=800 is expected.
    #[test]
    fn falsify_e7c_from_params_rejects_wrong_shape() {
        let mut params = HashMap::new();
        // Intentionally wrong size: 50 elements for 100*8=800 expected
        params.insert("embed.weight".to_string(), Tensor::from_vec(vec![0.1; 50], true));
        let embed = Embedding::from_params(&params, "embed.weight", 100, 8);
        // FIXED (PMAT-326): now rejected
        assert!(
            embed.is_none(),
            "FALSIFY-E7c: PMAT-326 fix — from_params MUST reject wrong-shape embedding"
        );
    }

    /// FALSIFY-E7d: OOB token_id produces zeros (not panic)
    ///
    /// Contract divergence: aprender skips OOB tokens, realizar/entrenar zero-fill.
    /// This test documents entrenar's behavior.
    #[test]
    fn falsify_e7d_oob_token_produces_zeros_not_panic() {
        let embed = Embedding::new(100, 8);
        let tokens = vec![0, 999]; // 999 is way OOB
        let output = embed.forward(&tokens);
        assert_eq!(output.len(), 2 * 8);
        // Token 0 should have non-zero values
        let data = output.data();
        let token0_l2: f32 = (0..8).map(|i| data[i] * data[i]).sum::<f32>().sqrt();
        assert!(token0_l2 > 1e-6, "Token 0 should have non-zero embedding");
        // Token 999 should be all zeros
        let token999_l2: f32 = (8..16).map(|i| data[i] * data[i]).sum::<f32>().sqrt();
        assert!(token999_l2 < 1e-10, "OOB token should be zero-filled");
    }

    /// FALSIFY-E7e: Embedding init is deterministic (reproducible)
    #[test]
    fn falsify_e7e_init_deterministic() {
        let embed1 = Embedding::new(100, 64);
        let embed2 = Embedding::new(100, 64);
        let d1 = embed1.weight.data();
        let d2 = embed2.weight.data();
        assert_eq!(
            d1.as_slice().expect("operation should succeed"),
            d2.as_slice().expect("operation should succeed"),
            "FALSIFY-E7e: Same vocab+hidden must produce identical initialization"
        );
    }

    // =========================================================================
    // FALSIFY-EM-001..004: embedding-lookup-v1.yaml contract mapping
    //
    // Five-Whys (PMAT-354):
    //   Why 1: entrenar has E7a-e init tests but no forward-path EM-* tests
    //   Why 2: E7 tests validate initialization, not the lookup/forward contract
    //   Why 3: no mapping from embedding-lookup-v1.yaml to entrenar test names
    //   Why 4: entrenar predates the provable-contracts YAML
    //   Why 5: forward() was assumed correct because it's "just slicing"
    //
    // References:
    //   - provable-contracts/contracts/embedding-lookup-v1.yaml
    //   - src/transformer/embedding.rs::forward()
    // =========================================================================

    /// FALSIFY-EM-001: forward output shape = seq_len * hidden_size
    #[test]
    fn falsify_em_001_forward_output_shape() {
        let embed = Embedding::new(100, 32);

        for seq_len in [1, 3, 10, 50] {
            let tokens: Vec<u32> = (0..seq_len).collect();
            let output = embed.forward(&tokens);
            assert_eq!(
                output.len(),
                seq_len as usize * 32,
                "FALSIFIED EM-001: forward({seq_len} tokens) produced {} elements, expected {}",
                output.len(),
                seq_len as usize * 32
            );
        }
    }

    /// FALSIFY-EM-001b: empty input produces empty output
    #[test]
    fn falsify_em_001b_forward_empty_input() {
        let embed = Embedding::new(100, 32);
        let output = embed.forward(&[]);
        assert_eq!(output.len(), 0, "FALSIFIED EM-001b: empty input should produce 0 elements");
    }

    /// FALSIFY-EM-002: OOB token → zeros, no panic (N-09 escape)
    ///
    /// Contract: token_id >= vocab_size produces zero-filled output, not a panic.
    /// Valid tokens alongside OOB tokens must still produce correct results.
    #[test]
    fn falsify_em_002_oob_safety() {
        let vocab_size = 50;
        let hidden = 8;
        let embed = Embedding::new(vocab_size, hidden);

        // Pure OOB tokens
        let oob_output = embed.forward(&[999, 50, 100]);
        let oob_data = oob_output.data();
        for (i, &v) in oob_data.iter().enumerate() {
            assert!(v.abs() < 1e-10, "FALSIFIED EM-002: OOB output[{i}] = {v}, expected 0.0");
        }

        // Mixed valid + OOB: valid tokens must still be correct
        let mixed_output = embed.forward(&[0, 999, 49]);
        let mixed_data = mixed_output.data();
        let weight_data = embed.weight.data();

        // Token 0 (valid): should match weight row 0
        for d in 0..hidden {
            assert_eq!(
                mixed_data[d], weight_data[d],
                "FALSIFIED EM-002: valid token 0 corrupted at dim {d}"
            );
        }

        // Token 999 (OOB): should be zeros
        for d in 0..hidden {
            assert!(
                mixed_data[hidden + d].abs() < 1e-10,
                "FALSIFIED EM-002: OOB token 999 at dim {d} = {}, expected 0.0",
                mixed_data[hidden + d]
            );
        }

        // Token 49 (valid boundary): should match weight row 49
        for d in 0..hidden {
            assert_eq!(
                mixed_data[2 * hidden + d],
                weight_data[49 * hidden + d],
                "FALSIFIED EM-002: valid boundary token 49 corrupted at dim {d}"
            );
        }
    }

    /// FALSIFY-EM-003: forward determinism (same tokens → bit-identical output)
    #[test]
    fn falsify_em_003_forward_determinism() {
        let embed = Embedding::new(100, 64);
        let tokens = vec![5u32, 42, 0, 99, 17];

        let o1 = embed.forward(&tokens);
        let o2 = embed.forward(&tokens);

        assert_eq!(
            o1.data().as_slice().expect("operation should succeed"),
            o2.data().as_slice().expect("operation should succeed"),
            "FALSIFIED EM-003: forward() is non-deterministic"
        );
    }

    /// FALSIFY-EM-004: forward output is finite (no NaN, no Inf)
    #[test]
    fn falsify_em_004_forward_finite_output() {
        let embed = Embedding::new(200, 16);
        let tokens: Vec<u32> = (0..200).collect();
        let output = embed.forward(&tokens);
        let data = output.data();

        let nan_count = data.iter().filter(|v| v.is_nan()).count();
        let inf_count = data.iter().filter(|v| v.is_infinite()).count();

        assert_eq!(
            nan_count, 0,
            "FALSIFIED EM-004: forward output contains {nan_count} NaN values"
        );
        assert_eq!(
            inf_count, 0,
            "FALSIFIED EM-004: forward output contains {inf_count} Inf values"
        );
    }

    /// FALSIFY-EM-005: forward value correctness (extractive — output[i] = W[token_id])
    #[test]
    fn falsify_em_005_forward_value_correctness() {
        let embed = Embedding::new(50, 8);
        let tokens = vec![0u32, 10, 49];
        let output = embed.forward(&tokens);
        let out_data = output.data();
        let weight_data = embed.weight.data();

        // Token 0: output[0..8] == weight[0..8]
        for i in 0..8 {
            assert_eq!(
                out_data[i], weight_data[i],
                "FALSIFIED EM-005: output[{i}] != weight[{i}] for token 0"
            );
        }
        // Token 10: output[8..16] == weight[80..88]
        for i in 0..8 {
            assert_eq!(
                out_data[8 + i],
                weight_data[80 + i],
                "FALSIFIED EM-005: output[{}] != weight[{}] for token 10",
                8 + i,
                80 + i
            );
        }
    }

    // =========================================================================
    // FALSIFY-EMB-005: Non-zero embeddings (embedding-algebra-v1.yaml)
    //
    // Five-Whys (PMAT-354):
    //   Why 1: entrenar had E7a init tests but no FALSIFY-EMB-005 tagged test
    //   Why 2: E7a covers init validity, not the EMB "non-zero" algebra claim
    //   Why 3: no mapping from embedding-algebra-v1.yaml to entrenar test names
    //   Why 4: entrenar predates the provable-contracts YAML
    //   Why 5: forward output non-zero was assumed from init non-zero
    // =========================================================================

    // =========================================================================
    // FALSIFY-EMB-001: Lookup determinism (embedding-algebra-v1.yaml)
    //
    // Five-Whys (PMAT-354, Phase 8):
    //   Why 1: entrenar had EM-003 (determinism) but not EMB-001 (algebra contract)
    //   Why 2: EM-003 tests forward() determinism, EMB-001 tests per-token lookup identity
    //   Why 3: EMB-001 YAML says "proptest: embed(t) == embed(t) for random t"
    //   Why 4: no mapping from embedding-algebra-v1.yaml EMB-001 to entrenar tests
    //   Why 5: lookup determinism assumed from EM-003 but never isolated per-token
    // =========================================================================

    /// FALSIFY-EMB-001: same token always returns same vector
    #[test]
    fn falsify_emb_001_lookup_determinism() {
        let embed = Embedding::new(200, 48);
        for t in [0u32, 1, 42, 100, 199] {
            let v1 = embed.forward(&[t]);
            let v2 = embed.forward(&[t]);
            assert_eq!(
                v1.data(),
                v2.data(),
                "FALSIFIED EMB-001: embed({t}) != embed({t}) — non-deterministic lookup"
            );
        }
    }

    // =========================================================================
    // FALSIFY-EMB-002: Shape preservation (embedding-algebra-v1.yaml)
    //
    // Five-Whys (PMAT-354, Phase 8):
    //   Why 1: entrenar EM-001 tests output length but not EMB-002 per-token dimension
    //   Why 2: EMB-002 YAML says "embedding output is d_model-dimensional"
    //   Why 3: shape preservation for different hidden sizes never parametrically tested
    //   Why 4: entrenar only used hidden_size=64 in EM-001 tests
    //   Why 5: no systematic d_model variation in embedding tests
    // =========================================================================

    /// FALSIFY-EMB-002: embedding output dimension matches hidden_size
    #[test]
    fn falsify_emb_002_shape_preservation() {
        for (v, d) in [(100, 32), (200, 64), (500, 128), (50, 16)] {
            let embed = Embedding::new(v, d);
            let output = embed.forward(&[0, 1, 2]);
            assert_eq!(
                output.data().len(),
                3 * d,
                "FALSIFIED EMB-002: vocab={v}, d_model={d}, output len={} != 3*{d}",
                output.data().len()
            );
        }
    }

    // =========================================================================
    // FALSIFY-EMB-004: Vocabulary bounds (embedding-algebra-v1.yaml)
    //
    // Five-Whys (PMAT-354, Phase 8):
    //   Why 1: entrenar EM-002 tests OOB safety but not EMB-004 (algebra perspective)
    //   Why 2: EMB-004 YAML says "out-of-range IDs rejected"
    //   Why 3: entrenar silently zeros OOB (N-09) — need explicit boundary test
    //   Why 4: boundary between valid and OOB never tested at exact vocab_size edge
    //   Why 5: no EMB-004 tagged test existed in entrenar
    // =========================================================================

    /// FALSIFY-EMB-004: valid tokens non-zero, OOB tokens zero
    #[test]
    fn falsify_emb_004_vocabulary_bounds() {
        let vocab = 50;
        let d = 16;
        let embed = Embedding::new(vocab, d);

        // Last valid token must be non-zero
        let valid_output = embed.forward(&[vocab as u32 - 1]);
        let valid_norm: f32 = valid_output.data().iter().map(|v| v * v).sum();
        assert!(
            valid_norm > 0.0,
            "FALSIFIED EMB-004: valid token {} produced zero embedding",
            vocab - 1
        );

        // First OOB token must be zero (N-09 escape)
        let oob_output = embed.forward(&[vocab as u32]);
        let oob_norm: f32 = oob_output.data().iter().map(|v| v * v).sum();
        assert!(
            oob_norm == 0.0,
            "FALSIFIED EMB-004: OOB token {vocab} produced non-zero (norm={oob_norm})"
        );
    }

    /// FALSIFY-EMB-005: forward output is non-zero for valid tokens
    #[test]
    fn falsify_emb_005_forward_non_zero() {
        let embed = Embedding::new(100, 64);
        let tokens = vec![0u32, 42, 99];
        let output = embed.forward(&tokens);
        let data = output.data();

        let l2_norm: f32 = data.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(l2_norm > 1e-6, "FALSIFIED EMB-005: forward output is all-zero (L2={l2_norm})");
    }

    // =========================================================================
    // PROPTEST FALSIFY: Embedding property-based falsification
    //
    // Five-Whys (PMAT-354, Phase 9):
    //   Why 1: EM/EMB tests used fixed vocab=100, hidden=32/48/64
    //   Why 2: embedding forward() could have off-by-one at edge vocab sizes
    //   Why 3: proptest explores vocab/hidden/seq_len combos humans don't anticipate
    //   Why 4: determinism (EM-003, EMB-001) could break under certain init patterns
    //   Why 5: YAML contracts explicitly call for "proptest with random..."
    // =========================================================================

    mod em_proptest_falsify {
        use super::*;
        use proptest::prelude::*;

        // EM-001-prop: output shape for random seq_len and hidden_size
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]
            #[test]
            fn falsify_em_001_prop_output_shape(
                vocab_size in prop::sample::select(vec![50_usize, 100, 200, 500]),
                hidden_size in prop::sample::select(vec![16_usize, 32, 48, 64]),
                seq_len in 1_usize..32,
            ) {
                let embed = Embedding::new(vocab_size, hidden_size);
                let tokens: Vec<u32> = (0..seq_len).map(|i| (i % vocab_size) as u32).collect();
                let output = embed.forward(&tokens);
                prop_assert_eq!(
                    output.len(), seq_len * hidden_size,
                    "FALSIFIED EM-001-prop: len={} != {}*{}={} (v={})",
                    output.len(), seq_len, hidden_size, seq_len * hidden_size, vocab_size
                );
            }
        }

        // EM-003-prop: determinism for random tokens
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(50))]
            #[test]
            fn falsify_em_003_prop_determinism(
                vocab_size in prop::sample::select(vec![50_usize, 100, 200]),
                hidden_size in prop::sample::select(vec![16_usize, 32, 64]),
                token_ids in proptest::collection::vec(0_u32..49, 1..16),
            ) {
                let embed = Embedding::new(vocab_size, hidden_size);
                let out1 = embed.forward(&token_ids);
                let out2 = embed.forward(&token_ids);
                prop_assert_eq!(
                    out1.data(), out2.data(),
                    "FALSIFIED EM-003-prop: two calls differ (v={}, h={})",
                    vocab_size, hidden_size
                );
            }
        }

        // EM-004-prop: finite output for random tokens
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]
            #[test]
            fn falsify_em_004_prop_finite(
                vocab_size in prop::sample::select(vec![50_usize, 100, 200]),
                hidden_size in prop::sample::select(vec![16_usize, 32, 64]),
                token_ids in proptest::collection::vec(0_u32..49, 1..16),
            ) {
                let embed = Embedding::new(vocab_size, hidden_size);
                let output = embed.forward(&token_ids);
                for (i, v) in output.data().iter().enumerate() {
                    prop_assert!(
                        v.is_finite(),
                        "FALSIFIED EM-004-prop: output[{}]={} not finite (v={}, h={})",
                        i, v, vocab_size, hidden_size
                    );
                }
            }
        }
    }

    // =========================================================================
    // PROPTEST FALSIFY: EMB algebra property-based falsification
    //
    // Five-Whys (PMAT-354, Phase 9):
    //   Why 1: EMB-001/002/004/005 had zero proptest coverage in entrenar
    //   Why 2: Determinism (EMB-001) only tested 5 fixed token IDs
    //   Why 3: Shape preservation (EMB-002) only tested 4 (vocab, d) pairs
    //   Why 4: Vocabulary bounds (EMB-004) only tested vocab=50
    //   Why 5: proptest explores random token/dim combos at scale
    // =========================================================================

    mod emb_proptest_falsify {
        use super::*;
        use proptest::prelude::*;

        // EMB-001-prop: lookup determinism for random tokens
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]
            #[test]
            fn falsify_emb_001_prop_determinism(
                vocab_size in prop::sample::select(vec![50_usize, 100, 200]),
                hidden_size in prop::sample::select(vec![16_usize, 32, 64]),
                token_id in 0_u32..49,
            ) {
                let embed = Embedding::new(vocab_size, hidden_size);
                let v1 = embed.forward(&[token_id]);
                let v2 = embed.forward(&[token_id]);
                prop_assert_eq!(
                    v1.data(), v2.data(),
                    "FALSIFIED EMB-001-prop: embed({}) non-deterministic (v={}, h={})",
                    token_id, vocab_size, hidden_size
                );
            }
        }

        // EMB-002-prop: shape preservation for random dimensions
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]
            #[test]
            fn falsify_emb_002_prop_shape(
                vocab_size in prop::sample::select(vec![50_usize, 100, 200, 500]),
                hidden_size in prop::sample::select(vec![16_usize, 32, 48, 64, 128]),
                seq_len in 1_usize..16,
            ) {
                let embed = Embedding::new(vocab_size, hidden_size);
                let tokens: Vec<u32> = (0..seq_len).map(|i| (i % vocab_size) as u32).collect();
                let output = embed.forward(&tokens);
                prop_assert_eq!(
                    output.data().len(), seq_len * hidden_size,
                    "FALSIFIED EMB-002-prop: data len={} != {}*{}={} (v={})",
                    output.data().len(), seq_len, hidden_size, seq_len * hidden_size, vocab_size
                );
            }
        }

        // EMB-005-prop: non-zero output for random valid tokens
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]
            #[test]
            fn falsify_emb_005_prop_non_zero(
                vocab_size in prop::sample::select(vec![50_usize, 100, 200]),
                hidden_size in prop::sample::select(vec![16_usize, 32, 64]),
                token_ids in proptest::collection::vec(0_u32..49, 1..8),
            ) {
                let embed = Embedding::new(vocab_size, hidden_size);
                let output = embed.forward(&token_ids);
                let l2_norm: f32 = output.data().iter().map(|v| v * v).sum::<f32>().sqrt();
                prop_assert!(
                    l2_norm > 1e-6,
                    "FALSIFIED EMB-005-prop: output all-zero (L2={}, v={}, h={})",
                    l2_norm, vocab_size, hidden_size
                );
            }
        }
    }
}
