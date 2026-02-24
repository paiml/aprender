// =========================================================================
// FALSIFY-EM: embedding-lookup-v1.yaml contract (aprender Embedding)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had Embedding tests but zero FALSIFY-EM-* tests
//   Why 2: unit tests verify forward() output shape, not lookup invariants
//   Why 3: no mapping from embedding-lookup-v1.yaml to aprender test names
//   Why 4: aprender predates the provable-contracts YAML convention
//   Why 5: embedding lookup was "obviously correct" (array indexing)
//
// References:
//   - provable-contracts/contracts/embedding-lookup-v1.yaml
//   - Mikolov et al. (2013) "Efficient Estimation of Word Representations"
// =========================================================================

use super::*;

/// FALSIFY-EM-001: Output shape — output.shape = (1, seq_len, d_model)
///
/// Embedding forward must return (batch=1, seq_len, hidden_size) for any valid seq_len.
#[test]
fn falsify_em_001_output_shape() {
    let vocab_size = 100;
    let hidden_size = 64;
    let embed = Embedding::new(vocab_size, hidden_size);

    let seq_lengths = [1, 5, 10, 50, 100];
    for &seq_len in &seq_lengths {
        let ids: Vec<u32> = (0..seq_len).map(|i| (i % vocab_size) as u32).collect();
        let output = embed.forward(&ids);
        assert_eq!(
            output.shape(),
            &[1, seq_len, hidden_size],
            "FALSIFIED EM-001: shape for seq_len={seq_len} is {:?}, expected [1, {seq_len}, {hidden_size}]",
            output.shape()
        );
    }
}

/// FALSIFY-EM-002: OOB panic freedom — out-of-range token IDs must not panic
///
/// Contract: embedding-lookup-v1.yaml requires token_ids[i] < vocab_size.
/// When violated, implementation must not panic — it emits zeros (N-09 escape).
#[test]
fn falsify_em_002_oob_panic_freedom() {
    let vocab_size = 50;
    let hidden_size = 16;
    let embed = Embedding::new(vocab_size, hidden_size);

    // All IDs are out of bounds
    let oob_ids = vec![50u32, 100, 999, u32::MAX];
    let output = embed.forward(&oob_ids);

    // Shape must still be correct
    assert_eq!(
        output.shape(),
        &[1, 4, hidden_size],
        "FALSIFIED EM-002: OOB shape {:?} != [1, 4, {hidden_size}]",
        output.shape()
    );

    // OOB rows must be zero-filled (N-09 escape)
    for (i, &val) in output.data().iter().enumerate() {
        assert!(
            val.abs() < 1e-10,
            "FALSIFIED EM-002: OOB output[{i}] = {val}, expected 0.0"
        );
    }
}

/// FALSIFY-EM-002b: Mixed valid + OOB token IDs
///
/// Valid IDs must still produce correct lookups even when OOB IDs are present.
#[test]
fn falsify_em_002b_mixed_valid_oob() {
    let vocab_size = 10;
    let hidden_size = 4;
    let embed = Embedding::new(vocab_size, hidden_size);

    // Mix valid (0, 5) and OOB (100) IDs
    let ids = vec![0u32, 100, 5];
    let output = embed.forward(&ids);
    let out_data = output.data();
    let weight_data = embed.weight().data();

    // Position 0 (token 0): must match weight row 0
    for d in 0..hidden_size {
        let expected = weight_data[0 * hidden_size + d];
        let actual = out_data[0 * hidden_size + d];
        assert!(
            (actual - expected).abs() < 1e-10,
            "FALSIFIED EM-002b: valid token at pos 0, dim {d}: {actual} != {expected}"
        );
    }

    // Position 1 (token 100): OOB → zeros
    for d in 0..hidden_size {
        let actual = out_data[1 * hidden_size + d];
        assert!(
            actual.abs() < 1e-10,
            "FALSIFIED EM-002b: OOB token at pos 1, dim {d}: {actual} != 0.0"
        );
    }

    // Position 2 (token 5): must match weight row 5
    for d in 0..hidden_size {
        let expected = weight_data[5 * hidden_size + d];
        let actual = out_data[2 * hidden_size + d];
        assert!(
            (actual - expected).abs() < 1e-10,
            "FALSIFIED EM-002b: valid token at pos 2, dim {d}: {actual} != {expected}"
        );
    }
}

/// FALSIFY-EM-003: Deterministic — two calls with same inputs produce identical output
#[test]
fn falsify_em_003_deterministic() {
    let embed = Embedding::new(50, 32);
    let ids = vec![0u32, 5, 10, 49, 1, 23];

    let out1 = embed.forward(&ids);
    let out2 = embed.forward(&ids);

    assert_eq!(
        out1.data(),
        out2.data(),
        "FALSIFIED EM-003: two calls with identical inputs differ"
    );
}

/// FALSIFY-EM-004: Finite output — all elements finite when weights are finite
#[test]
fn falsify_em_004_finite_output() {
    let embed = Embedding::new(200, 128);
    // Sample tokens across entire vocabulary
    let ids: Vec<u32> = (0..200).collect();
    let output = embed.forward(&ids);

    for (i, &val) in output.data().iter().enumerate() {
        assert!(
            val.is_finite(),
            "FALSIFIED EM-004: output[{i}] = {val} (not finite)"
        );
    }
}

/// FALSIFY-EM-005: Row lookup correctness — output[i] = W[token_ids[i]]
///
/// Each output row must exactly match the corresponding row in the weight matrix.
#[test]
fn falsify_em_005_row_lookup_correctness() {
    let vocab_size = 10;
    let hidden_size = 4;
    let embed = Embedding::new(vocab_size, hidden_size);

    let ids = vec![0u32, 3, 7, 9, 1];
    let output = embed.forward(&ids);
    let out_data = output.data();
    let weight_data = embed.weight().data();

    for (seq_pos, &token_id) in ids.iter().enumerate() {
        let token_idx = token_id as usize;
        for d in 0..hidden_size {
            let expected = weight_data[token_idx * hidden_size + d];
            // output shape is [1, seq_len, hidden], so offset is seq_pos * hidden + d
            let actual = out_data[seq_pos * hidden_size + d];
            assert!(
                (actual - expected).abs() < 1e-10,
                "FALSIFIED EM-005: output[{seq_pos}][{d}] = {actual}, expected W[{token_idx}][{d}] = {expected}"
            );
        }
    }
}
