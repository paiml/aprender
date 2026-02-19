
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_functional() {
        let x = Tensor::from_slice(&[-1.0, 0.0, 1.0]);
        let y = relu(&x);
        assert_eq!(y.data(), &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_leaky_relu_functional() {
        let x = Tensor::from_slice(&[-1.0, 0.0, 1.0]);
        let y = leaky_relu(&x, 0.1);
        assert_eq!(y.data(), &[-0.1, 0.0, 1.0]);
    }

    #[test]
    fn test_softmax_functional() {
        let x = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
        let y = softmax(&x, -1);

        // Should sum to 1
        let sum: f32 = y.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_log_softmax_functional() {
        let x = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
        let y = log_softmax(&x, -1);

        // exp(log_softmax) should sum to 1
        let sum: f32 = y.data().iter().map(|&v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_functional() {
        // Identity weight, no bias
        let x = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
        let weight = Tensor::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]);

        let y = linear(&x, &weight, None);
        assert_eq!(y.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_linear_functional_with_bias() {
        let x = Tensor::new(&[1.0, 2.0], &[1, 2]);
        let weight = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let bias = Tensor::new(&[10.0, 20.0], &[2]);

        let y = linear(&x, &weight, Some(&bias));
        assert_eq!(y.data(), &[11.0, 22.0]);
    }

    #[test]
    fn test_dropout_eval() {
        let x = Tensor::ones(&[100]);
        let y = dropout(&x, 0.5, false); // eval mode
        assert_eq!(y.data(), x.data());
    }

    // =========================================================================
    // Additional coverage tests for functional interface
    // =========================================================================

    #[test]
    fn test_sigmoid_functional() {
        let x = Tensor::from_slice(&[0.0, 1.0, -1.0]);
        let y = sigmoid(&x);
        // sigmoid(0) = 0.5
        assert!((y.data()[0] - 0.5).abs() < 1e-5);
        // sigmoid(x) is always in (0, 1)
        for &val in y.data() {
            assert!(val > 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_tanh_functional() {
        let x = Tensor::from_slice(&[0.0, 1.0, -1.0]);
        let y = tanh(&x);
        // tanh(0) = 0
        assert!((y.data()[0]).abs() < 1e-5);
        // tanh is in (-1, 1)
        for &val in y.data() {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_gelu_functional() {
        let x = Tensor::from_slice(&[0.0, 1.0, -1.0]);
        let y = gelu(&x);
        // GELU(0) = 0
        assert!((y.data()[0]).abs() < 1e-5);
        // GELU(1) ≈ 0.841
        assert!((y.data()[1] - 0.841).abs() < 0.01);
        // GELU(-1) ≈ -0.158
        assert!((y.data()[2] - (-0.158)).abs() < 0.02);
    }

    #[test]
    fn test_dropout_training_zeros_some() {
        // With training=true and p=0.5, some values should be zeroed
        let x = Tensor::ones(&[1000]);
        let y = dropout(&x, 0.5, true);

        let zeros = y.data().iter().filter(|&&v| v == 0.0).count();
        let scaled = y
            .data()
            .iter()
            .filter(|&&v| (v - 2.0).abs() < 0.001)
            .count();

        // With p=0.5, roughly half should be zero, half should be scaled by 2
        assert!(
            zeros > 300 && zeros < 700,
            "Expected ~500 zeros, got {zeros}"
        );
        assert!(
            scaled > 300 && scaled < 700,
            "Expected ~500 scaled, got {scaled}"
        );
    }

    #[test]
    fn test_dropout_zero_probability() {
        let x = Tensor::ones(&[100]);
        let y = dropout(&x, 0.0, true); // p=0 means no dropout
        assert_eq!(y.data(), x.data());
    }

    #[test]
    fn test_softmax_multi_batch() {
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let y = softmax(&x, -1);

        // Each row should sum to 1
        let sum1: f32 = y.data()[0..3].iter().sum();
        let sum2: f32 = y.data()[3..6].iter().sum();
        assert!((sum1 - 1.0).abs() < 1e-5);
        assert!((sum2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_log_softmax_multi_batch() {
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let y = log_softmax(&x, -1);

        // exp(log_softmax) should sum to 1 for each row
        let sum1: f32 = y.data()[0..3].iter().map(|v| v.exp()).sum();
        let sum2: f32 = y.data()[3..6].iter().map(|v| v.exp()).sum();
        assert!((sum1 - 1.0).abs() < 1e-5);
        assert!((sum2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_batch() {
        // Test linear with batch dimension
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let weight = Tensor::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);
        let bias = Tensor::new(&[10.0, 20.0], &[2]);

        let y = linear(&x, &weight, Some(&bias));
        // Row 1: [1,2,3] @ [[1,0],[0,1],[0,0]]^T + [10,20] = [1,2] + [10,20] = [11,22]
        assert!((y.data()[0] - 11.0).abs() < 1e-5);
        assert!((y.data()[1] - 22.0).abs() < 1e-5);
    }
}
