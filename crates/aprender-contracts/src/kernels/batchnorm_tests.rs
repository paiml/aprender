    use super::super::ulp::assert_ulp_eq;
    use super::*;
    use proptest::prelude::*;

    // ── Scalar known-answer tests ────────────────────────────────────────

    /// Verify constant input produces output equal to beta during training
    #[test]
    fn test_batchnorm_constant_input_training() {
        // All inputs constant for one channel -> output = beta (when gamma=1)
        // N=4, C=1, all values = 5.0
        let input = [5.0_f32, 5.0, 5.0, 5.0];
        let gamma = [1.0_f32];
        let beta = [3.0_f32];
        let mut running_mean = [0.0_f32];
        let mut running_var = [0.0_f32];
        let mut output = [0.0_f32; 4];

        batchnorm_scalar(
            &input,
            4,
            1,
            &gamma,
            &beta,
            1e-5,
            &mut running_mean,
            &mut running_var,
            &mut output,
            0.1,
            true,
        );

        // With constant input: mean=5.0, var=0.0, (x-mean)=0
        // output = gamma * 0 + beta = beta = 3.0
        for (i, &o) in output.iter().enumerate() {
            assert!((o - 3.0).abs() < 1e-3, "output[{i}] = {o}, expected ~3.0");
        }
    }

    /// Verify training mode updates running mean and variance with momentum
    #[test]
    fn test_batchnorm_training_updates_running_stats() {
        let input = [1.0_f32, 2.0, 3.0, 4.0]; // N=4, C=1
        let gamma = [1.0_f32];
        let beta = [0.0_f32];
        let mut running_mean = [0.0_f32];
        let mut running_var = [0.0_f32];
        let mut output = [0.0_f32; 4];

        batchnorm_scalar(
            &input,
            4,
            1,
            &gamma,
            &beta,
            1e-5,
            &mut running_mean,
            &mut running_var,
            &mut output,
            0.1,
            true,
        );

        // batch_mean = 2.5, batch_var = 1.25
        // running_mean = 0.9*0 + 0.1*2.5 = 0.25
        // running_var = 0.9*0 + 0.1*1.25 = 0.125
        assert!(
            (running_mean[0] - 0.25).abs() < 1e-5,
            "running_mean = {}, expected 0.25",
            running_mean[0]
        );
        assert!(
            (running_var[0] - 0.125).abs() < 1e-5,
            "running_var = {}, expected 0.125",
            running_var[0]
        );
    }

    /// Verify inference mode uses running stats and leaves them unchanged
    #[test]
    fn test_batchnorm_inference_uses_running_stats() {
        let input = [1.0_f32, 2.0, 3.0, 4.0]; // N=4, C=1
        let gamma = [1.0_f32];
        let beta = [0.0_f32];
        let mut running_mean = [10.0_f32]; // Intentionally different from batch stats
        let mut running_var = [4.0_f32];
        let mut output = [0.0_f32; 4];

        batchnorm_scalar(
            &input,
            4,
            1,
            &gamma,
            &beta,
            0.0,
            &mut running_mean,
            &mut running_var,
            &mut output,
            0.1,
            false,
        );

        // Inference uses running stats: inv_std = 1/sqrt(4) = 0.5
        // output[i] = (input[i] - 10) * 0.5
        let inv_std = 1.0 / 4.0_f32.sqrt();
        for (i, &x) in input.iter().enumerate() {
            let expected = (x - 10.0) * inv_std;
            assert!(
                (output[i] - expected).abs() < 1e-5,
                "output[{i}] = {}, expected {expected}",
                output[i]
            );
        }

        // Running stats should NOT change during inference
        assert!((running_mean[0] - 10.0).abs() < 1e-10);
        assert!((running_var[0] - 4.0).abs() < 1e-10);
    }

    /// Verify training and inference outputs differ when running stats diverge from batch stats
    #[test]
    fn test_batchnorm_inference_differs_from_training() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let gamma = [1.0_f32];
        let beta = [0.0_f32];

        // Training output
        let mut rm_train = [5.0_f32];
        let mut rv_train = [2.0_f32];
        let mut train_out = [0.0_f32; 4];
        batchnorm_scalar(
            &input,
            4,
            1,
            &gamma,
            &beta,
            1e-5,
            &mut rm_train,
            &mut rv_train,
            &mut train_out,
            0.1,
            true,
        );

        // Inference output (with different running stats)
        let mut rm_eval = [5.0_f32];
        let mut rv_eval = [2.0_f32];
        let mut eval_out = [0.0_f32; 4];
        batchnorm_scalar(
            &input,
            4,
            1,
            &gamma,
            &beta,
            1e-5,
            &mut rm_eval,
            &mut rv_eval,
            &mut eval_out,
            0.1,
            false,
        );

        // They should differ since batch stats != running stats
        let mut all_equal = true;
        for i in 0..4 {
            if (train_out[i] - eval_out[i]).abs() > 1e-6 {
                all_equal = false;
            }
        }
        assert!(
            !all_equal,
            "training and inference outputs should differ when running stats != batch stats"
        );
    }

    /// Verify batchnorm normalizes each channel independently with N=2, C=2
    #[test]
    fn test_batchnorm_multi_channel() {
        // N=2, C=2
        // Channel 0: [1.0, 3.0], mean=2.0, var=1.0
        // Channel 1: [2.0, 4.0], mean=3.0, var=1.0
        let input = [1.0_f32, 2.0, 3.0, 4.0]; // [sample0_ch0, sample0_ch1, sample1_ch0, sample1_ch1]
        let gamma = [1.0_f32, 1.0];
        let beta = [0.0_f32, 0.0];
        let mut running_mean = [0.0_f32, 0.0];
        let mut running_var = [0.0_f32, 0.0];
        let mut output = [0.0_f32; 4];

        batchnorm_scalar(
            &input,
            2,
            2,
            &gamma,
            &beta,
            1e-8,
            &mut running_mean,
            &mut running_var,
            &mut output,
            0.1,
            true,
        );

        // Channel 0: mean=2, var=1, inv_std=1/sqrt(1+eps)~1
        // (1-2)*1 = -1, (3-2)*1 = 1
        assert!((output[0] - (-1.0)).abs() < 1e-3);
        assert!((output[2] - 1.0).abs() < 1e-3);
        // Channel 1: mean=3, var=1
        // (2-3)*1 = -1, (4-3)*1 = 1
        assert!((output[1] - (-1.0)).abs() < 1e-3);
        assert!((output[3] - 1.0).abs() < 1e-3);
    }

    /// Verify batchnorm with batch size 1 produces output equal to beta
    #[test]
    fn test_batchnorm_single_sample() {
        // N=1 (batch size 1) -> var=0, output=beta when gamma=1
        let input = [7.0_f32, 3.0]; // N=1, C=2
        let gamma = [1.0_f32, 1.0];
        let beta = [5.0_f32, -5.0];
        let mut running_mean = [0.0_f32, 0.0];
        let mut running_var = [0.0_f32, 0.0];
        let mut output = [0.0_f32; 2];

        batchnorm_scalar(
            &input,
            1,
            2,
            &gamma,
            &beta,
            1e-5,
            &mut running_mean,
            &mut running_var,
            &mut output,
            0.1,
            true,
        );

        // With N=1: mean=input, var=0, (x-mean)=0, output=beta
        assert!((output[0] - 5.0).abs() < 1e-3, "output[0] = {}", output[0]);
        assert!(
            (output[1] - (-5.0)).abs() < 1e-3,
            "output[1] = {}",
            output[1]
        );
    }

    /// Verify batchnorm panics on input length mismatch
    #[test]
    #[should_panic(expected = "input length must be n * c")]
    fn test_batchnorm_input_length_mismatch() {
        let input = [1.0_f32, 2.0];
        let gamma = [1.0_f32];
        let beta = [0.0_f32];
        let mut rm = [0.0_f32];
        let mut rv = [0.0_f32];
        let mut output = [0.0_f32; 2];
        batchnorm_scalar(
            &input,
            3,
            1,
            &gamma,
            &beta,
            1e-5,
            &mut rm,
            &mut rv,
            &mut output,
            0.1,
            true,
        );
    }

    /// Verify batchnorm panics when batch size or channels is zero
    #[test]
    #[should_panic(expected = "batchnorm requires n > 0 and c > 0")]
    fn test_batchnorm_zero_batch() {
        let input: [f32; 0] = [];
        let gamma: [f32; 0] = [];
        let beta: [f32; 0] = [];
        let mut rm: [f32; 0] = [];
        let mut rv: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        batchnorm_scalar(
            &input,
            0,
            0,
            &gamma,
            &beta,
            1e-5,
            &mut rm,
            &mut rv,
            &mut output,
            0.1,
            true,
        );
    }

    // ── Property-based tests ─────────────────────────────────────────────

    proptest! {
        #[test]
        fn prop_batchnorm_training_finite(
            n in 2_usize..8,
            c in 1_usize..4,
        ) {
            let total = n * c;
            let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1 - 1.0).collect();
            let gamma = vec![1.0_f32; c];
            let beta = vec![0.0_f32; c];
            let mut running_mean = vec![0.0_f32; c];
            let mut running_var = vec![1.0_f32; c];
            let mut output = vec![0.0_f32; total];

            batchnorm_scalar(
                &input, n, c, &gamma, &beta, 1e-5,
                &mut running_mean, &mut running_var,
                &mut output, 0.1, true,
            );

            for (i, &o) in output.iter().enumerate() {
                prop_assert!(o.is_finite(), "output[{i}] = {o} is not finite");
            }
            for (i, &rv) in running_var.iter().enumerate() {
                prop_assert!(rv >= 0.0, "running_var[{i}] = {rv} is negative");
            }
        }

        #[test]
        fn prop_batchnorm_running_var_nonneg(
            n in 2_usize..8,
            c in 1_usize..4,
            iters in 1_usize..20,
        ) {
            let total = n * c;
            let mut running_mean = vec![0.0_f32; c];
            let mut running_var = vec![1.0_f32; c];

            for step in 0..iters {
                let input: Vec<f32> = (0..total)
                    .map(|i| ((i + step) as f32) * 0.3 - 2.0)
                    .collect();
                let gamma = vec![1.0_f32; c];
                let beta = vec![0.0_f32; c];
                let mut output = vec![0.0_f32; total];

                batchnorm_scalar(
                    &input, n, c, &gamma, &beta, 1e-5,
                    &mut running_mean, &mut running_var,
                    &mut output, 0.1, true,
                );
            }

            for (i, &rv) in running_var.iter().enumerate() {
                prop_assert!(
                    rv >= 0.0,
                    "running_var[{i}] = {rv} after {iters} iterations"
                );
            }
        }
    }

