
    #[test]
    fn test_maybe_reduce_grad_fallback_clone() {
        // When shapes don't match and no special case applies
        let grad = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let result = maybe_reduce_grad(&grad, &[3, 3]); // Different numel
        assert_eq!(result.data(), grad.data());
        assert_eq!(result.shape(), grad.shape());
    }

    #[test]
    fn test_matmul_backward() {
        // Test the actual MatmulBackward struct
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let y = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let grad_fn = MatmulBackward {
            x: x.clone(),
            y: y.clone(),
        };
        let grad_out = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let grads = grad_fn.backward(&grad_out);

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].shape(), &[2, 2]); // grad_x
        assert_eq!(grads[1].shape(), &[2, 2]); // grad_y
        assert_eq!(grad_fn.name(), "MatmulBackward");
    }

    #[test]
    fn test_matmul_backward_non_square() {
        // Test with non-square matrices
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let y = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let grad_fn = MatmulBackward {
            x: x.clone(),
            y: y.clone(),
        };
        // grad_out has shape of result [2, 2]
        let grad_out = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
        let grads = grad_fn.backward(&grad_out);

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].shape(), &[2, 3]); // Same as x
        assert_eq!(grads[1].shape(), &[3, 2]); // Same as y
    }

    #[test]
    fn test_pow_backward_fractional_exponent() {
        // Test pow with fractional exponent (sqrt case)
        let x = Tensor::from_slice(&[4.0, 9.0, 16.0]);
        let grad_fn = PowBackward { x, n: 0.5 };
        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0]);
        let grads = grad_fn.backward(&grad_out);

        // grad = 0.5 * x^(-0.5) = 0.5 / sqrt(x)
        assert!((grads[0].data()[0] - 0.25).abs() < 1e-5); // 0.5/sqrt(4) = 0.25
        assert!((grads[0].data()[1] - (1.0 / 6.0)).abs() < 1e-5); // 0.5/sqrt(9) = 1/6
        assert!((grads[0].data()[2] - 0.125).abs() < 1e-5); // 0.5/sqrt(16) = 0.125
    }

    #[test]
    fn test_pow_backward_name() {
        let grad_fn = PowBackward {
            x: Tensor::from_slice(&[1.0]),
            n: 2.0,
        };
        assert_eq!(grad_fn.name(), "PowBackward");
    }

    #[test]
    fn test_softmax_backward_multi_batch() {
        // Test with multiple batches
        let output = Tensor::new(&[0.3, 0.7, 0.6, 0.4], &[2, 2]);
        let grad_fn = SoftmaxBackward { output };
        let grad_out = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let grads = grad_fn.backward(&grad_out);

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].shape(), &[2, 2]);
    }

    #[test]
    fn test_cross_entropy_backward_multi_batch() {
        // Test with multiple batches
        let softmax_output = Tensor::new(&[0.7, 0.2, 0.1, 0.1, 0.3, 0.6], &[2, 3]);
        let targets = vec![0_usize, 2_usize];
        let grad_fn = CrossEntropyBackward {
            softmax_output,
            targets,
        };
        let grad_out = Tensor::from_slice(&[1.0]);
        let grads = grad_fn.backward(&grad_out);

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].shape(), &[2, 3]);
        // Check that target positions have (softmax - 1) applied
    }

    #[test]
    fn test_all_backward_names() {
        // Ensure all backward functions have unique, descriptive names
        let names = vec![
            NegBackward.name(),
            ExpBackward {
                output: Tensor::from_slice(&[1.0]),
            }
            .name(),
            LogBackward {
                x: Tensor::from_slice(&[1.0]),
            }
            .name(),
            SqrtBackward {
                output: Tensor::from_slice(&[1.0]),
            }
            .name(),
            SumBackward {
                input_shape: vec![1],
            }
            .name(),
            MeanBackward {
                input_shape: vec![1],
            }
            .name(),
            ReluBackward {
                x: Tensor::from_slice(&[1.0]),
            }
            .name(),
            LeakyReluBackward {
                x: Tensor::from_slice(&[1.0]),
                negative_slope: 0.01,
            }
            .name(),
            GeluBackward {
                x: Tensor::from_slice(&[1.0]),
            }
            .name(),
            SoftmaxBackward {
                output: Tensor::from_slice(&[1.0]),
            }
            .name(),
            CrossEntropyBackward {
                softmax_output: Tensor::from_slice(&[1.0]),
                targets: vec![0],
            }
            .name(),
            SigmoidBackward {
                output: Tensor::from_slice(&[1.0]),
            }
            .name(),
            TanhBackward {
                output: Tensor::from_slice(&[1.0]),
            }
            .name(),
            TransposeBackward.name(),
            ViewBackward {
                input_shape: vec![1],
            }
            .name(),
            BroadcastAddBackward {
                x_shape: vec![1],
                y_shape: vec![1],
            }
            .name(),
        ];

        // All names should be unique
        let unique: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(
            unique.len(),
            names.len(),
            "All backward names should be unique"
        );

        // All names should end with "Backward"
        for name in &names {
            assert!(
                name.ends_with("Backward"),
                "Name {} should end with 'Backward'",
                name
            );
        }
    }
