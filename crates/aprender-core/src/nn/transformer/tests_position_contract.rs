// =========================================================================
// FALSIFY-AP: absolute-position-v1.yaml contract (aprender PositionalEncoding)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had PositionalEncoding tests but zero FALSIFY-AP-* tests
//   Why 2: unit tests verify shapes, not additive/identity invariants
//   Why 3: no mapping from absolute-position-v1.yaml to aprender test names
//   Why 4: aprender predates the provable-contracts YAML convention
//   Why 5: sinusoidal PE was "obviously correct" (textbook formula)
//
// References:
//   - provable-contracts/contracts/absolute-position-v1.yaml
//   - Vaswani et al. (2017) "Attention Is All You Need"
// =========================================================================

use super::*;

/// FALSIFY-AP-001: Shape preservation — output.shape = input.shape
#[test]
fn falsify_ap_001_shape_preservation() {
    let test_cases = vec![
        (1, 5, 32),   // batch=1, seq=5, d=32
        (2, 10, 64),  // batch=2, seq=10, d=64
        (1, 1, 16),   // minimal
        (4, 50, 128), // larger
    ];

    for (batch, seq_len, d_model) in test_cases {
        let mut pe = PositionalEncoding::new(d_model, 100).with_dropout(0.0);
        pe.eval(); // disable dropout

        let x = Tensor::ones(&[batch, seq_len, d_model]);
        let y = pe.forward(&x);

        assert_eq!(
            y.shape(),
            x.shape(),
            "FALSIFIED AP-001: output shape {:?} != input shape {:?} for (b={batch},s={seq_len},d={d_model})",
            y.shape(),
            x.shape()
        );
    }
}

/// FALSIFY-AP-002: Additive identity — zero position encoding → output = input
///
/// When all positional encoding values are zero (degenerate case), the output
/// should equal the input. Since PE uses sinusoidal encoding, we verify the
/// weaker property: output = input + PE, so (output - input) = PE which is
/// deterministic and independent of input values.
#[test]
fn falsify_ap_002_additive_property() {
    let d_model = 32;
    let mut pe = PositionalEncoding::new(d_model, 100).with_dropout(0.0);
    pe.eval();

    // Create two different inputs
    let x1 = Tensor::ones(&[1, 10, d_model]);
    let x2 = Tensor::new(
        &(0..10 * d_model)
            .map(|i| (i as f32 * 0.07).sin())
            .collect::<Vec<_>>(),
        &[1, 10, d_model],
    );

    let y1 = pe.forward(&x1);
    let y2 = pe.forward(&x2);

    // (y1 - x1) should equal (y2 - x2) because PE is additive and input-independent
    let pe_from_x1: Vec<f32> = y1
        .data()
        .iter()
        .zip(x1.data().iter())
        .map(|(&y, &x)| y - x)
        .collect();
    let pe_from_x2: Vec<f32> = y2
        .data()
        .iter()
        .zip(x2.data().iter())
        .map(|(&y, &x)| y - x)
        .collect();

    for (i, (&a, &b)) in pe_from_x1.iter().zip(pe_from_x2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "FALSIFIED AP-002: PE contribution differs at index {i}: {a} vs {b} \
             (PE must be additive and input-independent)"
        );
    }

    // Verify PE contribution is non-trivial (at least one non-zero)
    let pe_norm: f32 = pe_from_x1.iter().map(|v| v * v).sum();
    assert!(
        pe_norm > 1e-6,
        "FALSIFIED AP-002: PE contribution is all-zero (norm={pe_norm})"
    );
}

/// FALSIFY-AP-003: Finite output — all elements finite when input is finite
#[test]
fn falsify_ap_003_finite_output() {
    let mut pe = PositionalEncoding::new(64, 200).with_dropout(0.0);
    pe.eval();

    let x = Tensor::new(
        &(0..2 * 100 * 64)
            .map(|i| (i as f32 * 0.01).sin())
            .collect::<Vec<_>>(),
        &[2, 100, 64],
    );
    let y = pe.forward(&x);

    for (i, &val) in y.data().iter().enumerate() {
        assert!(
            val.is_finite(),
            "FALSIFIED AP-003: output[{i}] = {val} (not finite)"
        );
    }
}

/// FALSIFY-AP-004: Position-dependent — different positions produce different outputs
///
/// Two identical token embeddings at different positions must produce different outputs.
#[test]
fn falsify_ap_004_position_dependent() {
    let mut pe = PositionalEncoding::new(32, 100).with_dropout(0.0);
    pe.eval();

    // Apply PE to a sequence of 51 identical tokens

    // Apply PE to a sequence of 51 identical tokens
    let x_seq = Tensor::ones(&[1, 51, 32]);
    let y_seq = pe.forward(&x_seq);

    let d = 32;
    // Extract output for positions 0, 1, 5, 50
    let positions = [0, 1, 5, 50];
    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            let pos_a = positions[i];
            let pos_b = positions[j];
            let diff: f32 = (0..d)
                .map(|k| {
                    let a = y_seq.data()[pos_a * d + k];
                    let b = y_seq.data()[pos_b * d + k];
                    (a - b).abs()
                })
                .sum();
            assert!(
                diff > 1e-4,
                "FALSIFIED AP-004: positions {pos_a} and {pos_b} have identical output (diff={diff})"
            );
        }
    }
}

/// FALSIFY-AP-005: Deterministic — same input at same position always gives same output
#[test]
fn falsify_ap_005_deterministic() {
    let mut pe = PositionalEncoding::new(32, 50).with_dropout(0.0);
    pe.eval();

    let x = Tensor::new(
        &(0..1 * 10 * 32)
            .map(|i| (i as f32 * 0.03).cos())
            .collect::<Vec<_>>(),
        &[1, 10, 32],
    );

    let y1 = pe.forward(&x);
    let y2 = pe.forward(&x);

    assert_eq!(
        y1.data(),
        y2.data(),
        "FALSIFIED AP-005: two forward passes with identical input differ"
    );
}

// =========================================================================
// PROPTEST FALSIFY: Absolute position property-based falsification
//
// Five-Whys (PMAT-354, Phase 8):
//   Why 1: YAML AP-001/002/003/004 explicitly call for "proptest with random..."
//   Why 2: All 5 AP tests use fixed dimensions (32/64/128)
//   Why 3: Edge cases in unusual d_model values missed by hand-picked inputs
//   Why 4: proptest explores dimension space humans don't anticipate
//   Why 5: Position encoding math has d_model-dependent sin/cos alternation
//
// References:
//   - absolute-position-v1.yaml FALSIFY-AP-001: "proptest with random seq_len and d"
//   - absolute-position-v1.yaml FALSIFY-AP-002: "proptest with random token_embed"
// =========================================================================

mod proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    // AP-001-prop: Shape preservation for random dimensions
    // YAML: "proptest with random seq_len and d, verify output dimensions match"
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]
        #[test]
        fn falsify_ap_001_prop_shape(
            seq_len in 1_usize..64,
            d_model in prop::sample::select(vec![16_usize, 32, 48, 64, 128]),
        ) {
            let mut pe = PositionalEncoding::new(d_model, 200).with_dropout(0.0);
            pe.eval();
            let x = Tensor::ones(&[1, seq_len, d_model]);
            let y = pe.forward(&x);
            prop_assert_eq!(
                y.shape(), x.shape(),
                "FALSIFIED AP-001-prop: seq_len={}, d_model={}, output={:?}",
                seq_len, d_model, y.shape()
            );
        }
    }

    // AP-002-prop: Additive property — PE contribution is input-independent
    // YAML: "proptest with random token_embed and zero pos_embed"
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]
        #[test]
        fn falsify_ap_002_prop_additive(
            d_model in prop::sample::select(vec![16_usize, 32, 64]),
            seq_len in 1_usize..20,
        ) {
            let mut pe = PositionalEncoding::new(d_model, 100).with_dropout(0.0);
            pe.eval();

            let x1 = Tensor::ones(&[1, seq_len, d_model]);
            let x2_data: Vec<f32> = (0..seq_len * d_model)
                .map(|i| (i as f32 * 0.13).sin())
                .collect();
            let x2 = Tensor::new(&x2_data, &[1, seq_len, d_model]);

            let y1 = pe.forward(&x1);
            let y2 = pe.forward(&x2);

            // PE contribution = output - input, must be identical for both inputs
            for i in 0..seq_len * d_model {
                let pe1 = y1.data()[i] - x1.data()[i];
                let pe2 = y2.data()[i] - x2.data()[i];
                prop_assert!(
                    (pe1 - pe2).abs() < 1e-5,
                    "FALSIFIED AP-002-prop: PE[{i}] differs: {pe1} vs {pe2} (d={d_model}, s={seq_len})"
                );
            }
        }
    }

    // AP-003-prop: Finite output for random inputs
    // YAML: "proptest with random finite token_embed and pos_embed values"
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]
        #[test]
        fn falsify_ap_003_prop_finite(
            d_model in prop::sample::select(vec![16_usize, 32, 64]),
            seq_len in 1_usize..30,
        ) {
            let mut pe = PositionalEncoding::new(d_model, 200).with_dropout(0.0);
            pe.eval();

            let x_data: Vec<f32> = (0..seq_len * d_model)
                .map(|i| (i as f32 * 0.07).cos() * 10.0)
                .collect();
            let x = Tensor::new(&x_data, &[1, seq_len, d_model]);
            let y = pe.forward(&x);

            for (i, &v) in y.data().iter().enumerate() {
                prop_assert!(
                    v.is_finite(),
                    "FALSIFIED AP-003-prop: output[{i}]={v} (d={d_model}, s={seq_len})"
                );
            }
        }
    }
}
