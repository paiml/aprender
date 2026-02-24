// =========================================================================
// FALSIFY-BA: bias-add-v1.yaml contract (aprender functional::linear bias)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had zero FALSIFY-BA-* tests for broadcast bias addition
//   Why 2: bias add is inlined in linear(), tested only via integration tests
//   Why 3: no mapping from bias-add-v1.yaml to aprender test names
//   Why 4: aprender predates the provable-contracts YAML convention
//   Why 5: bias add was "obviously correct" (simple broadcast loop)
//
// References:
//   - provable-contracts/contracts/bias-add-v1.yaml
//   - Standard neural network practice — affine transformation bias term
// =========================================================================

use super::*;

/// FALSIFY-BA-001: Shape preservation — output shape = input shape
///
/// bias_add(x, bias) must return a tensor with the same shape as x.
#[test]
fn falsify_ba_001_shape_preservation() {
    let shapes: Vec<(usize, usize)> = vec![(1, 1), (1, 64), (4, 128), (16, 256), (32, 8)];

    for (rows, cols) in shapes {
        let x = Tensor::ones(&[rows, cols]);
        let weight = Tensor::ones(&[cols, cols]); // identity-ish
        let bias = Tensor::new(&vec![0.5; cols], &[cols]);

        let y = linear(&x, &weight, Some(&bias));
        assert_eq!(
            y.shape(),
            &[rows, cols],
            "FALSIFIED BA-001: linear({rows}, {cols}) output shape = {:?}, expected [{rows}, {cols}]",
            y.shape()
        );
    }
}

/// FALSIFY-BA-002: Zero-bias identity — y = x when bias = 0
///
/// Adding a zero bias vector must produce the same output as no bias.
#[test]
fn falsify_ba_002_zero_bias_identity() {
    let x = Tensor::new(
        &(0..4 * 8)
            .map(|i| (i as f32 * 0.1).sin())
            .collect::<Vec<_>>(),
        &[4, 8],
    );
    // Identity weight matrix
    let mut w_data = vec![0.0f32; 8 * 8];
    for i in 0..8 {
        w_data[i * 8 + i] = 1.0;
    }
    let weight = Tensor::new(&w_data, &[8, 8]);
    let zero_bias = Tensor::new(&vec![0.0; 8], &[8]);

    let y_no_bias = linear(&x, &weight, None);
    let y_zero_bias = linear(&x, &weight, Some(&zero_bias));

    for (i, (&a, &b)) in y_no_bias
        .data()
        .iter()
        .zip(y_zero_bias.data().iter())
        .enumerate()
    {
        assert!(
            (a - b).abs() < 1e-6,
            "FALSIFIED BA-002: element[{i}] no_bias={a}, zero_bias={b}"
        );
    }
}

/// FALSIFY-BA-003: Additivity — bias_add(bias_add(x, b1), b2) = bias_add(x, b1+b2)
///
/// Two sequential bias additions must equal a single addition of summed biases.
#[test]
fn falsify_ba_003_additivity() {
    let x = Tensor::new(
        &(0..3 * 4).map(|i| i as f32 * 0.5).collect::<Vec<_>>(),
        &[3, 4],
    );
    // Identity weight (so linear just does W^T @ x = x when W = I)
    let mut w_data = vec![0.0f32; 4 * 4];
    for i in 0..4 {
        w_data[i * 4 + i] = 1.0;
    }
    let weight = Tensor::new(&w_data, &[4, 4]);
    let b1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let b2 = Tensor::new(&[0.5, -0.5, 1.5, -1.0], &[4]);
    let b_sum = Tensor::new(&[1.5, 1.5, 4.5, 3.0], &[4]);

    // Double apply: linear(linear(x, I, b1), I, b2)
    let y_double = linear(&linear(&x, &weight, Some(&b1)), &weight, Some(&b2));
    // Single apply with sum: linear(x, I, b1+b2)
    let y_single = linear(&x, &weight, Some(&b_sum));

    for (i, (&a, &b)) in y_double
        .data()
        .iter()
        .zip(y_single.data().iter())
        .enumerate()
    {
        assert!(
            (a - b).abs() < 1e-5,
            "FALSIFIED BA-003: element[{i}] double={a}, single={b}"
        );
    }
}

/// FALSIFY-BA-004: Broadcast — same bias vector applied to every batch element
///
/// Every row of the output must have the same bias offset.
#[test]
fn falsify_ba_004_broadcast() {
    let cols = 4;
    let rows = 8;
    // Zero input so output = 0 + bias = bias for every row
    let x = Tensor::new(&vec![0.0; rows * cols], &[rows, cols]);
    let mut w_data = vec![0.0f32; cols * cols];
    for i in 0..cols {
        w_data[i * cols + i] = 1.0;
    }
    let weight = Tensor::new(&w_data, &[cols, cols]);
    let bias = Tensor::new(&[10.0, 20.0, 30.0, 40.0], &[cols]);

    let y = linear(&x, &weight, Some(&bias));
    let data = y.data();

    // Every row should equal the bias vector
    for row in 0..rows {
        for col in 0..cols {
            let val = data[row * cols + col];
            let expected = bias.data()[col];
            assert!(
                (val - expected).abs() < 1e-6,
                "FALSIFIED BA-004: row[{row}][{col}] = {val}, expected {expected}"
            );
        }
    }
}

mod bias_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-BA-001-prop: Shape preservation for random dimensions
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_ba_001_prop_shape_preservation(
            rows in 1..=16usize,
            cols in 1..=32usize,
        ) {
            let x = Tensor::ones(&[rows, cols]);
            let mut w_data = vec![0.0f32; cols * cols];
            for i in 0..cols {
                w_data[i * cols + i] = 1.0;
            }
            let weight = Tensor::new(&w_data, &[cols, cols]);
            let bias = Tensor::new(&vec![1.0; cols], &[cols]);
            let y = linear(&x, &weight, Some(&bias));

            prop_assert_eq!(
                y.shape(),
                &[rows, cols],
                "FALSIFIED BA-001-prop: shape {:?} != [{}, {}]",
                y.shape(), rows, cols
            );
        }
    }

    /// FALSIFY-BA-004-prop: Broadcast — zero input gives bias for all rows
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_ba_004_prop_broadcast(
            rows in 1..=8usize,
            cols in 1..=8usize,
            bias_val in -10.0f32..10.0,
        ) {
            let x = Tensor::new(&vec![0.0; rows * cols], &[rows, cols]);
            let mut w_data = vec![0.0f32; cols * cols];
            for i in 0..cols {
                w_data[i * cols + i] = 1.0;
            }
            let weight = Tensor::new(&w_data, &[cols, cols]);
            let bias = Tensor::new(&vec![bias_val; cols], &[cols]);
            let y = linear(&x, &weight, Some(&bias));

            for row in 0..rows {
                for col in 0..cols {
                    let val = y.data()[row * cols + col];
                    prop_assert!(
                        (val - bias_val).abs() < 1e-4,
                        "FALSIFIED BA-004-prop: row[{}][{}]={}, expected {}",
                        row, col, val, bias_val
                    );
                }
            }
        }
    }
}
