// =========================================================================
// FALSIFY-LF: loss-functions-v1.yaml contract (aprender MSELoss, L1Loss)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-LF-* tests for loss functions
//   Why 2: loss tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no mapping from loss-functions-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: MSE/L1 were "obviously correct" (textbook formulas)
//
// References:
//   - provable-contracts/contracts/loss-functions-v1.yaml
// =========================================================================

#[cfg(test)]
mod tests {
    use super::super::*;

    /// FALSIFY-LF-001: MSE is non-negative
    #[test]
    fn falsify_lf_001_mse_non_negative() {
        let pred = Tensor::new(&[1.0_f32, 2.0, 3.0], &[3]);
        let target = Tensor::new(&[1.5, 2.5, 3.5], &[3]);

        let criterion = MSELoss::new();
        let loss = criterion.forward(&pred, &target);
        assert!(
            loss.data()[0] >= 0.0,
            "FALSIFIED LF-001: MSE loss = {} < 0",
            loss.data()[0]
        );
    }

    /// FALSIFY-LF-002: MSE = 0 when pred == target
    #[test]
    fn falsify_lf_002_mse_zero_on_match() {
        let pred = Tensor::new(&[1.0_f32, 2.0, 3.0], &[3]);
        let target = Tensor::new(&[1.0, 2.0, 3.0], &[3]);

        let criterion = MSELoss::new();
        let loss = criterion.forward(&pred, &target);
        assert!(
            loss.data()[0].abs() < 1e-6,
            "FALSIFIED LF-002: MSE = {} for identical pred/target",
            loss.data()[0]
        );
    }

    /// FALSIFY-LF-003: L1 is non-negative
    #[test]
    fn falsify_lf_003_l1_non_negative() {
        let pred = Tensor::new(&[1.0_f32, 2.0, 3.0], &[3]);
        let target = Tensor::new(&[1.5, 2.5, 3.5], &[3]);

        let criterion = L1Loss::new();
        let loss = criterion.forward(&pred, &target);
        assert!(
            loss.data()[0] >= 0.0,
            "FALSIFIED LF-003: L1 loss = {} < 0",
            loss.data()[0]
        );
    }

    /// FALSIFY-LF-004: MSE(a, b) == MSE(b, a) (symmetric)
    #[test]
    fn falsify_lf_004_mse_symmetric() {
        let a = Tensor::new(&[1.0_f32, 3.0, 5.0], &[3]);
        let b = Tensor::new(&[2.0, 4.0, 6.0], &[3]);

        let criterion = MSELoss::new();
        let loss_ab = criterion.forward(&a, &b);
        let loss_ba = criterion.forward(&b, &a);
        assert!(
            (loss_ab.data()[0] - loss_ba.data()[0]).abs() < 1e-6,
            "FALSIFIED LF-004: MSE(a,b)={} != MSE(b,a)={}",
            loss_ab.data()[0],
            loss_ba.data()[0]
        );
    }

    /// FALSIFY-LF-005: L1(a, b) == L1(b, a) (symmetric)
    #[test]
    fn falsify_lf_005_l1_symmetric() {
        let a = Tensor::new(&[1.0_f32, 3.0, -2.0], &[3]);
        let b = Tensor::new(&[4.0, -1.0, 6.0], &[3]);

        let criterion = L1Loss::new();
        let loss_ab = criterion.forward(&a, &b);
        let loss_ba = criterion.forward(&b, &a);
        assert!(
            (loss_ab.data()[0] - loss_ba.data()[0]).abs() < 1e-6,
            "FALSIFIED LF-005: L1(a,b)={} != L1(b,a)={}",
            loss_ab.data()[0],
            loss_ba.data()[0]
        );
    }

    mod lf_proptest_falsify {
        use super::super::super::*;
        use proptest::prelude::*;

        /// FALSIFY-LF-001-prop: MSE is non-negative for random inputs
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]

            #[test]
            fn falsify_lf_001_prop_mse_non_negative(
                seed in 0..1000u32,
                n in 2..=16usize,
            ) {
                let pred_data: Vec<f32> = (0..n)
                    .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                    .collect();
                let target_data: Vec<f32> = (0..n)
                    .map(|i| ((i as f32 + seed as f32) * 0.73).cos() * 10.0)
                    .collect();

                let pred = Tensor::new(&pred_data, &[n]);
                let target = Tensor::new(&target_data, &[n]);

                let loss = MSELoss::new().forward(&pred, &target);
                prop_assert!(
                    loss.data()[0] >= 0.0,
                    "FALSIFIED LF-001-prop: MSE = {} < 0",
                    loss.data()[0]
                );
            }
        }

        /// FALSIFY-LF-005-prop: L1 symmetry for random inputs
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]

            #[test]
            fn falsify_lf_005_prop_l1_symmetric(
                seed in 0..1000u32,
                n in 2..=16usize,
            ) {
                let a_data: Vec<f32> = (0..n)
                    .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                    .collect();
                let b_data: Vec<f32> = (0..n)
                    .map(|i| ((i as f32 + seed as f32) * 0.73).cos() * 10.0)
                    .collect();

                let a = Tensor::new(&a_data, &[n]);
                let b = Tensor::new(&b_data, &[n]);

                let loss_ab = L1Loss::new().forward(&a, &b);
                let loss_ba = L1Loss::new().forward(&b, &a);
                prop_assert!(
                    (loss_ab.data()[0] - loss_ba.data()[0]).abs() < 1e-5,
                    "FALSIFIED LF-005-prop: L1(a,b)={} != L1(b,a)={}",
                    loss_ab.data()[0], loss_ba.data()[0]
                );
            }
        }

        /// FALSIFY-LF-002-prop: MSE = 0 when pred == target
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]

            #[test]
            fn falsify_lf_002_prop_mse_zero_on_match(
                seed in 0..1000u32,
                n in 2..=16usize,
            ) {
                let data: Vec<f32> = (0..n)
                    .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                    .collect();

                let pred = Tensor::new(&data, &[n]);
                let target = Tensor::new(&data, &[n]);

                let loss = MSELoss::new().forward(&pred, &target);
                prop_assert!(
                    loss.data()[0].abs() < 1e-5,
                    "FALSIFIED LF-002-prop: MSE(x,x) = {} != 0",
                    loss.data()[0]
                );
            }
        }
    }
}
