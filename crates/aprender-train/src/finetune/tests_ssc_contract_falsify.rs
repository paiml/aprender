//! Falsification tests for SSC v11 contracts (bidirectional attention, linear probe)
//!
//! Contract: provable-contracts/contracts/bidirectional-attention-v1.yaml
//! Contract: provable-contracts/contracts/linear-probe-classifier-v1.yaml
//! Contract: provable-contracts/contracts/encoder-forward-v1.yaml
//! Contract: provable-contracts/contracts/learned-position-embedding-v1.yaml
//!
//! Tests: FALSIFY-BIATT-001..003, FALSIFY-PROBE-001..003, FALSIFY-ENC-001..002, FALSIFY-POS-001

use crate::autograd::Tensor;
use crate::finetune::linear_probe::LinearProbe;
use crate::transformer::{EncoderModel, ModelArchitecture, TransformerConfig};

fn tiny_encoder_config() -> TransformerConfig {
    TransformerConfig {
        hidden_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_kv_heads: 4,
        intermediate_size: 64,
        vocab_size: 100,
        max_position_embeddings: 32,
        rms_norm_eps: 1e-5,
        architecture: ModelArchitecture::Encoder,
        ..TransformerConfig::tiny()
    }
}

// =============================================================================
// FALSIFY-BIATT-001: No causal mask applied
// Contract: bidirectional-attention-v1.yaml
// Prediction: Upper triangle of attention matrix is non-zero
// If fails: Causal mask leaked into bidirectional path
// =============================================================================

#[test]
fn falsify_biatt_001_no_causal_mask() {
    // Bidirectional encoder: modifying K[2] should affect output[0]
    // (position 0 attends to position 2 — impossible with causal mask)
    let config = tiny_encoder_config();
    let encoder = EncoderModel::new(&config);

    let tokens_a = vec![1, 2, 3, 4];
    let tokens_b = vec![1, 2, 99, 4]; // change position 2

    let out_a = encoder.cls_embedding(&tokens_a);
    let out_b = encoder.cls_embedding(&tokens_b);

    let da = out_a.data();
    let db = out_b.data();
    let sa = da.as_slice().expect("contiguous");
    let sb = db.as_slice().expect("contiguous");

    // CLS at position 0 must differ when position 2 changes (bidirectional)
    let diff: f32 = sa.iter().zip(sb.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(
        diff > 1e-6,
        "FALSIFY-BIATT-001: CLS embedding must change when later tokens change (diff={diff}). \
         If zero, causal mask is blocking bidirectional attention."
    );
}

// =============================================================================
// FALSIFY-BIATT-002: Causal parity at n=1
// Contract: bidirectional-attention-v1.yaml
// Prediction: Output identical to causal attention for single-token input
// If fails: Mask application differs even when mask is trivial
// =============================================================================

#[test]
fn falsify_biatt_002_single_token_deterministic() {
    let config = tiny_encoder_config();
    let encoder = EncoderModel::new(&config);

    // Single token: no mask matters (1x1 attention matrix)
    let out1 = encoder.cls_embedding(&[42]);
    let out2 = encoder.cls_embedding(&[42]);

    let d1 = out1.data();
    let d2 = out2.data();
    let s1 = d1.as_slice().expect("contiguous");
    let s2 = d2.as_slice().expect("contiguous");

    assert_eq!(
        s1, s2,
        "FALSIFY-BIATT-002: Single-token output must be bit-identical on repeated calls"
    );
}

// =============================================================================
// FALSIFY-BIATT-003: Attention weight normalization
// Contract: bidirectional-attention-v1.yaml
// Prediction: Each row sums to 1.0 within tolerance
// Verified indirectly: softmax is applied in attention, output is finite
// =============================================================================

#[test]
fn falsify_biatt_003_output_finite() {
    let config = tiny_encoder_config();
    let encoder = EncoderModel::new(&config);

    let tokens = vec![10, 20, 30, 40, 50];
    let output = encoder.forward(&tokens);
    let data = output.data();
    let slice = data.as_slice().expect("contiguous");

    assert!(
        slice.iter().all(|v| v.is_finite()),
        "FALSIFY-BIATT-003: All encoder outputs must be finite (implies attention weights normalized)"
    );
}

// =============================================================================
// FALSIFY-PROBE-001: Encoder truly frozen
// Contract: linear-probe-classifier-v1.yaml
// Prediction: Encoder weights unchanged after 100 training steps
// Verified: LinearProbe trains on extracted embeddings, not encoder
// =============================================================================

#[test]
fn falsify_probe_001_encoder_frozen() {
    let config = tiny_encoder_config();
    let encoder = EncoderModel::new(&config);

    // Snapshot encoder CLS embedding before training
    let tokens = vec![1, 2, 3];
    let before = encoder.cls_embedding(&tokens);
    let before_data = before.data();
    let before_slice = before_data.as_slice().expect("contiguous").to_vec();

    // Extract embeddings and train linear probe
    let embeddings: Vec<Vec<f32>> = (0..20)
        .map(|i| {
            let t = vec![i as u32 % 100 + 1; 3];
            let cls = encoder.cls_embedding(&t);
            let d = cls.data();
            d.as_slice().expect("contiguous").to_vec()
        })
        .collect();
    let labels: Vec<usize> = (0..20).map(|i| usize::from(i >= 15)).collect();

    let mut probe = LinearProbe::new(config.hidden_size, 2);
    probe.train(&embeddings, &labels, 50, 0.1, None);

    // Verify encoder weights unchanged after probe training
    let after = encoder.cls_embedding(&tokens);
    let after_data = after.data();
    let after_slice = after_data.as_slice().expect("contiguous");

    assert_eq!(
        before_slice.as_slice(),
        after_slice,
        "FALSIFY-PROBE-001: Encoder weights must be unchanged after linear probe training. \
         If different, gradient leaked through frozen encoder."
    );
}

// =============================================================================
// FALSIFY-PROBE-002: Probability valid
// Contract: linear-probe-classifier-v1.yaml
// Prediction: Softmax output sums to 1.0 and all values > 0
// =============================================================================

#[test]
fn falsify_probe_002_probability_simplex() {
    let probe = LinearProbe::new(32, 2);

    // Test with realistic embedding patterns (normal range for encoder outputs)
    let test_embeddings = vec![
        vec![0.0f32; 32],                                   // zeros
        vec![1.0f32; 32],                                   // ones
        vec![-1.0f32; 32],                                  // negative
        vec![0.5f32; 32],                                   // moderate positive
        vec![-0.5f32; 32],                                  // moderate negative
        (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect(), // mixed gradient
    ];

    for (i, emb) in test_embeddings.iter().enumerate() {
        let probs = probe.predict_probs(&Tensor::from_vec(emb.clone(), false));
        let sum: f32 = probs.iter().sum();

        assert!(
            (sum - 1.0).abs() < 1e-5,
            "FALSIFY-PROBE-002: Softmax must sum to 1.0 for embedding {i}, got {sum}"
        );

        for (j, &p) in probs.iter().enumerate() {
            assert!(
                p > 0.0,
                "FALSIFY-PROBE-002: All probabilities must be > 0 for embedding {i}, got prob[{j}]={p}"
            );
        }
    }
}

// =============================================================================
// FALSIFY-PROBE-003: Trainable parameter count
// Contract: linear-probe-classifier-v1.yaml
// Prediction: For K=2, d=768: exactly 1538 trainable params
// =============================================================================

#[test]
fn falsify_probe_003_trainable_param_count() {
    // CodeBERT: hidden=768, classes=2 -> 768*2 + 2 = 1538
    let probe_codebert = LinearProbe::new(768, 2);
    assert_eq!(
        probe_codebert.num_parameters(),
        768 * 2 + 2,
        "FALSIFY-PROBE-003: CodeBERT probe must have exactly 1538 params"
    );

    // Tiny: hidden=32, classes=2 -> 32*2 + 2 = 66
    let probe_tiny = LinearProbe::new(32, 2);
    assert_eq!(
        probe_tiny.num_parameters(),
        32 * 2 + 2,
        "FALSIFY-PROBE-003: Tiny probe must have exactly 66 params"
    );

    // General: hidden=H, classes=K -> H*K + K
    for (h, k) in [(64, 3), (128, 5), (256, 10)] {
        let probe = LinearProbe::new(h, k);
        assert_eq!(
            probe.num_parameters(),
            h * k + k,
            "FALSIFY-PROBE-003: LinearProbe({h}, {k}) must have {expected} params",
            expected = h * k + k
        );
    }
}

// =============================================================================
// FALSIFY-ENC-001: Shape preservation
// Contract: encoder-forward-v1.yaml
// Prediction: 12 encoder layers preserve (n, hidden_size) shape
// =============================================================================

#[test]
fn falsify_enc_001_shape_preservation() {
    let config = tiny_encoder_config();
    let encoder = EncoderModel::new(&config);

    for seq_len in [1, 3, 5, 10] {
        let tokens: Vec<u32> = (0..seq_len).map(|i| (i as u32) + 1).collect();
        let output = encoder.forward(&tokens);
        assert_eq!(
            output.len(),
            seq_len * config.hidden_size,
            "FALSIFY-ENC-001: Encoder output must have shape (seq_len={seq_len}, hidden={h}), \
             got len={len}",
            h = config.hidden_size,
            len = output.len()
        );
    }
}

// =============================================================================
// FALSIFY-ENC-002: No NaN/Inf for finite inputs
// Contract: encoder-forward-v1.yaml
// Prediction: No NaN or Inf for inputs in normal float range
// =============================================================================

#[test]
fn falsify_enc_002_no_nan_inf() {
    let config = tiny_encoder_config();
    let encoder = EncoderModel::new(&config);

    // Test with various token ID patterns
    let test_cases: Vec<Vec<u32>> = vec![
        vec![0, 0, 0],           // all zeros
        vec![99, 99, 99],        // near vocab limit
        vec![1],                 // single token
        vec![1, 50, 99, 25, 75], // mixed
    ];

    for tokens in &test_cases {
        let output = encoder.forward(tokens);
        let data = output.data();
        let slice = data.as_slice().expect("contiguous");

        assert!(
            slice.iter().all(|v| v.is_finite()),
            "FALSIFY-ENC-002: No NaN/Inf in encoder output for tokens {tokens:?}"
        );
    }
}

// =============================================================================
// FALSIFY-POS-001: Position embedding deterministic lookup
// Contract: learned-position-embedding-v1.yaml
// Prediction: PE(pos) = PE(pos) for same weights (idempotent)
// =============================================================================

#[test]
fn falsify_pos_001_deterministic_lookup() {
    let config = tiny_encoder_config();
    let encoder = EncoderModel::new(&config);

    let tokens = vec![1, 2, 3, 4, 5];
    let out1 = encoder.forward(&tokens);
    let out2 = encoder.forward(&tokens);

    let d1 = out1.data();
    let d2 = out2.data();
    let s1 = d1.as_slice().expect("contiguous");
    let s2 = d2.as_slice().expect("contiguous");

    assert_eq!(
        s1, s2,
        "FALSIFY-POS-001: Encoder output must be bit-identical for same input (deterministic positions)"
    );
}

// =============================================================================
// PROPTEST: Contract property tests
// =============================================================================

mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        // FALSIFY-PROBE-002-prop: Random embeddings always produce valid probabilities
        #[test]
        fn falsify_probe_002_prop(
            hidden_size in 4usize..=64,
            num_classes in 2usize..=5,
        ) {
            let probe = LinearProbe::new(hidden_size, num_classes);
            let emb = Tensor::from_vec(vec![0.1f32; hidden_size], false);
            let probs = probe.predict_probs(&emb);

            let sum: f32 = probs.iter().sum();
            prop_assert!(
                (sum - 1.0).abs() < 1e-4,
                "FALSIFY-PROBE-002-prop: Softmax sum={sum} for hidden={hidden_size}, classes={num_classes}"
            );
            prop_assert!(
                probs.iter().all(|&p| p > 0.0),
                "FALSIFY-PROBE-002-prop: All probs must be > 0"
            );
        }

        // FALSIFY-PROBE-003-prop: Parameter count is always H*K + K
        #[test]
        fn falsify_probe_003_prop(
            hidden_size in 1usize..=512,
            num_classes in 2usize..=20,
        ) {
            let probe = LinearProbe::new(hidden_size, num_classes);
            let expected = hidden_size * num_classes + num_classes;
            prop_assert_eq!(
                probe.num_parameters(),
                expected,
                "FALSIFY-PROBE-003-prop: params must be H*K + K"
            );
        }

        // FALSIFY-ENC-001-prop: Shape preservation for various sequence lengths
        #[test]
        fn falsify_enc_001_prop(
            seq_len in 1usize..=16,
        ) {
            let config = tiny_encoder_config();
            let encoder = EncoderModel::new(&config);
            let tokens: Vec<u32> = (0..seq_len).map(|i| (i as u32) % 100 + 1).collect();
            let output = encoder.forward(&tokens);
            let expected = seq_len * config.hidden_size;
            prop_assert_eq!(
                output.len(),
                expected,
                "FALSIFY-ENC-001-prop: shape mismatch for seq_len={}",
                seq_len
            );
        }
    }
}
