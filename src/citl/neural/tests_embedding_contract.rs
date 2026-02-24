// =========================================================================
// FALSIFY-EM: embedding-lookup-v1.yaml contract (citl neural Embedding)
//
// Five-Whys (PMAT-354):
//   Why 1: citl/neural had Embedding struct but zero FALSIFY-EM-* tests
//   Why 2: unit tests verify NeuralErrorEncoder end-to-end, not lookup invariants
//   Why 3: no mapping from embedding-lookup-v1.yaml to citl test names
//   Why 4: citl/neural predates the provable-contracts YAML convention
//   Why 5: embedding_lookup was "obviously correct" (array indexing + OOB guard)
//
// References:
//   - provable-contracts/contracts/embedding-lookup-v1.yaml
//   - Mikolov et al. (2013) "Efficient Estimation of Word Representations"
// =========================================================================

use super::*;

/// FALSIFY-EM-001: Output shape — embedding_lookup returns [batch, seq_len, embed_dim]
#[test]
fn falsify_em_001_output_shape() {
    let vocab_size = 50;
    let embed_dim = 16;
    let embed = Embedding::new(vocab_size, embed_dim);

    let test_cases = vec![
        (1, 5),  // batch=1, seq=5
        (2, 10), // batch=2, seq=10
        (1, 1),  // minimal
    ];

    for (batch, seq_len) in test_cases {
        let indices_data: Vec<f32> = (0..batch * seq_len)
            .map(|i| (i % vocab_size) as f32)
            .collect();
        let indices = Tensor::new(&indices_data, &[batch, seq_len]);
        let output = embed.forward(&indices);

        assert_eq!(
            output.shape(),
            &[batch, seq_len, embed_dim],
            "FALSIFIED EM-001: shape {:?} != [{batch}, {seq_len}, {embed_dim}]",
            output.shape()
        );
    }
}

/// FALSIFY-EM-002: OOB safety — out-of-range indices produce zeros, no panic
#[test]
fn falsify_em_002_oob_safety() {
    let vocab_size = 10;
    let embed_dim = 4;
    let embed = Embedding::new(vocab_size, embed_dim);

    // Token ID 999 is out of bounds
    let indices = Tensor::new(&[0.0, 999.0, 5.0], &[1, 3]);
    let output = embed.forward(&indices);

    // Shape must still be correct
    assert_eq!(
        output.shape(),
        &[1, 3, embed_dim],
        "FALSIFIED EM-002: OOB shape {:?} != [1, 3, {embed_dim}]",
        output.shape()
    );

    // OOB row (position 1) should be zeros
    let out_data = output.data();
    for d in 0..embed_dim {
        let val = out_data[1 * embed_dim + d];
        assert!(
            val.abs() < 1e-10,
            "FALSIFIED EM-002: OOB output[1][{d}] = {val}, expected 0.0"
        );
    }
}

/// FALSIFY-EM-003: Deterministic — same inputs → same outputs
#[test]
fn falsify_em_003_deterministic() {
    let embed = Embedding::new(20, 8);
    let indices = Tensor::new(&[0.0, 5.0, 19.0, 10.0], &[1, 4]);

    let out1 = embed.forward(&indices);
    let out2 = embed.forward(&indices);

    assert_eq!(
        out1.data(),
        out2.data(),
        "FALSIFIED EM-003: two calls with identical inputs differ"
    );
}

/// FALSIFY-EM-004: Finite output — all elements finite when weights are finite
#[test]
fn falsify_em_004_finite_output() {
    let embed = Embedding::new(100, 32);
    let indices_data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let indices = Tensor::new(&indices_data, &[1, 100]);
    let output = embed.forward(&indices);

    for (i, &val) in output.data().iter().enumerate() {
        assert!(
            val.is_finite(),
            "FALSIFIED EM-004: output[{i}] = {val} (not finite)"
        );
    }
}

/// FALSIFY-EM-005: Row lookup correctness — output[i] = W[token_ids[i]]
#[test]
fn falsify_em_005_row_lookup_correctness() {
    let vocab_size = 10;
    let embed_dim = 4;
    let embed = Embedding::new(vocab_size, embed_dim);

    let token_ids = [0.0_f32, 3.0, 7.0, 9.0];
    let indices = Tensor::new(&token_ids, &[1, 4]);
    let output = embed.forward(&indices);
    let out_data = output.data();
    let weight_data = embed.weight.data();

    for (seq_pos, &token_id) in token_ids.iter().enumerate() {
        let token_idx = token_id as usize;
        for d in 0..embed_dim {
            let expected = weight_data[token_idx * embed_dim + d];
            let actual = out_data[seq_pos * embed_dim + d];
            assert!(
                (actual - expected).abs() < 1e-10,
                "FALSIFIED EM-005: output[{seq_pos}][{d}] = {actual}, expected W[{token_idx}][{d}] = {expected}"
            );
        }
    }
}

mod citl_em_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-EM-001-prop: Output shape for random seq lengths
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        #[test]
        fn falsify_em_001_prop_output_shape(
            seq_len in 1..=20usize,
        ) {
            let vocab_size = 50;
            let embed_dim = 16;
            let embed = Embedding::new(vocab_size, embed_dim);

            let indices_data: Vec<f32> = (0..seq_len)
                .map(|i| (i % vocab_size) as f32)
                .collect();
            let indices = Tensor::new(&indices_data, &[1, seq_len]);
            let output = embed.forward(&indices);

            prop_assert_eq!(
                output.shape(),
                &[1, seq_len, embed_dim],
                "FALSIFIED EM-001-prop: shape {:?} != [1, {}, {}]",
                output.shape(), seq_len, embed_dim
            );
        }
    }

    /// FALSIFY-EM-004-prop: Finite output for random token IDs
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(15))]

        #[test]
        fn falsify_em_004_prop_finite_output(
            seed in 0..200u32,
        ) {
            let vocab_size = 30;
            let embed_dim = 8;
            let embed = Embedding::new(vocab_size, embed_dim);

            let seq_len = 10;
            let indices_data: Vec<f32> = (0..seq_len)
                .map(|i| ((i + seed as usize) % vocab_size) as f32)
                .collect();
            let indices = Tensor::new(&indices_data, &[1, seq_len]);
            let output = embed.forward(&indices);

            for (i, &val) in output.data().iter().enumerate() {
                prop_assert!(
                    val.is_finite(),
                    "FALSIFIED EM-004-prop: output[{}]={} not finite (seed={})",
                    i, val, seed
                );
            }
        }
    }
}
