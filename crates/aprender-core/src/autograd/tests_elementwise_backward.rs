
    #[test]
    fn test_add_backward() {
        let grad_fn = AddBackward {
            x_shape: vec![3],
            y_shape: vec![3],
        };
        let grad_out = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let grads = grad_fn.backward(&grad_out);

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].data(), &[1.0, 2.0, 3.0]);
        assert_eq!(grads[1].data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_mul_backward() {
        let x = Tensor::from_slice(&[2.0, 3.0]);
        let y = Tensor::from_slice(&[4.0, 5.0]);
        let grad_fn = MulBackward {
            x: x.clone(),
            y: y.clone(),
        };

        let grad_out = Tensor::from_slice(&[1.0, 1.0]);
        let grads = grad_fn.backward(&grad_out);

        // grad_x = grad_out * y = [1*4, 1*5] = [4, 5]
        assert_eq!(grads[0].data(), &[4.0, 5.0]);
        // grad_y = grad_out * x = [1*2, 1*3] = [2, 3]
        assert_eq!(grads[1].data(), &[2.0, 3.0]);
    }

    #[test]
    fn test_relu_backward() {
        let x = Tensor::from_slice(&[-1.0, 0.0, 1.0, 2.0]);
        let grad_fn = ReluBackward { x };

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0]);
        let grads = grad_fn.backward(&grad_out);

        // grad = grad_out where x > 0, else 0
        assert_eq!(grads[0].data(), &[0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sum_backward() {
        let grad_fn = SumBackward {
            input_shape: vec![3],
        };
        let grad_out = Tensor::new(&[2.0], &[1]);
        let grads = grad_fn.backward(&grad_out);

        // Gradient is broadcast to all elements
        assert_eq!(grads[0].data(), &[2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_mean_backward() {
        let grad_fn = MeanBackward {
            input_shape: vec![4],
        };
        let grad_out = Tensor::new(&[1.0], &[1]);
        let grads = grad_fn.backward(&grad_out);

        // Gradient is 1/n for each element
        assert_eq!(grads[0].data(), &[0.25, 0.25, 0.25, 0.25]);
    }

    #[test]
    fn test_transpose_2d() {
        let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let t_t = transpose_2d(&t);

        assert_eq!(t_t.shape(), &[3, 2]);
        assert_eq!(t_t.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_matmul_2d() {
        // [2, 3] @ [3, 2] = [2, 2]
        let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);

        let c = matmul_2d(&a, &b);

        assert_eq!(c.shape(), &[2, 2]);
        // Row 0: [1,2,3] @ [[1,2],[3,4],[5,6]] = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
        // Row 1: [4,5,6] @ [[1,2],[3,4],[5,6]] = [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
        assert_eq!(c.data(), &[22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_sub_backward() {
        let grad_fn = SubBackward {
            x_shape: vec![3],
            y_shape: vec![3],
        };
        let grad_out = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let grads = grad_fn.backward(&grad_out);

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].data(), &[1.0, 2.0, 3.0]);
        assert_eq!(grads[1].data(), &[-1.0, -2.0, -3.0]);
    }

    #[test]
    fn test_div_backward() {
        let x = Tensor::from_slice(&[6.0, 8.0]);
        let y = Tensor::from_slice(&[2.0, 4.0]);
        let grad_fn = DivBackward {
            x: x.clone(),
            y: y.clone(),
        };

        let grad_out = Tensor::from_slice(&[1.0, 1.0]);
        let grads = grad_fn.backward(&grad_out);

        // grad_x = grad_out / y = [1/2, 1/4] = [0.5, 0.25]
        assert_eq!(grads[0].data(), &[0.5, 0.25]);
        // grad_y = -grad_out * x / y^2 = [-1*6/4, -1*8/16] = [-1.5, -0.5]
        assert_eq!(grads[1].data(), &[-1.5, -0.5]);
    }

    #[test]
    fn test_pow_backward() {
        let x = Tensor::from_slice(&[2.0, 3.0]);
        let grad_fn = PowBackward {
            x: x.clone(),
            n: 2.0,
        };

        let grad_out = Tensor::from_slice(&[1.0, 1.0]);
        let grads = grad_fn.backward(&grad_out);

        // grad = n * x^(n-1) * grad_out = 2 * [2, 3] = [4, 6]
        assert_eq!(grads[0].data(), &[4.0, 6.0]);
    }

    #[test]
    fn test_exp_backward() {
        let output = Tensor::from_slice(&[2.718281828, 7.389056099]); // e^1, e^2
        let grad_fn = ExpBackward { output };

        let grad_out = Tensor::from_slice(&[1.0, 1.0]);
        let grads = grad_fn.backward(&grad_out);

        // grad = exp(x) * grad_out = output * grad_out
        assert!((grads[0].data()[0] - 2.718281828).abs() < 1e-5);
        assert!((grads[0].data()[1] - 7.389056099).abs() < 1e-5);
    }

    #[test]
    fn test_log_backward() {
        let x = Tensor::from_slice(&[1.0, 2.0, 4.0]);
        let grad_fn = LogBackward { x };

        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0]);
        let grads = grad_fn.backward(&grad_out);

        // grad = 1/x * grad_out
        assert_eq!(grads[0].data(), &[1.0, 0.5, 0.25]);
    }

    #[test]
    fn test_sigmoid_backward() {
        let output = Tensor::from_slice(&[0.5, 0.731]); // sigmoid values
        let grad_fn = SigmoidBackward { output };

        let grad_out = Tensor::from_slice(&[1.0, 1.0]);
        let grads = grad_fn.backward(&grad_out);

        // grad = sigmoid(x) * (1 - sigmoid(x)) * grad_out
        assert!((grads[0].data()[0] - 0.25).abs() < 1e-5); // 0.5 * 0.5
    }

    #[test]
    fn test_tanh_backward() {
        let output = Tensor::from_slice(&[0.0, 0.7616]); // tanh values
        let grad_fn = TanhBackward { output };

        let grad_out = Tensor::from_slice(&[1.0, 1.0]);
        let grads = grad_fn.backward(&grad_out);

        // grad = (1 - tanh(x)^2) * grad_out
        assert!((grads[0].data()[0] - 1.0).abs() < 1e-5); // 1 - 0^2
    }

    #[test]
    fn test_backward_names() {
        assert_eq!(
            AddBackward {
                x_shape: vec![],
                y_shape: vec![]
            }
            .name(),
            "AddBackward"
        );
        assert_eq!(
            SubBackward {
                x_shape: vec![],
                y_shape: vec![]
            }
            .name(),
            "SubBackward"
        );
        assert_eq!(
            MulBackward {
                x: Tensor::from_slice(&[1.0]),
                y: Tensor::from_slice(&[1.0])
            }
            .name(),
            "MulBackward"
        );
        assert_eq!(
            DivBackward {
                x: Tensor::from_slice(&[1.0]),
                y: Tensor::from_slice(&[1.0])
            }
            .name(),
            "DivBackward"
        );
    }

    #[test]
    fn test_neg_backward() {
        let grad_fn = NegBackward;
        let grad_out = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let grads = grad_fn.backward(&grad_out);

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].data(), &[-1.0, -2.0, -3.0]);
        assert_eq!(grad_fn.name(), "NegBackward");
    }

    #[test]
    fn test_sqrt_backward() {
        // sqrt(4) = 2, sqrt(9) = 3, sqrt(16) = 4
        let output = Tensor::from_slice(&[2.0, 3.0, 4.0]);
        let grad_fn = SqrtBackward { output };
        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0]);
        let grads = grad_fn.backward(&grad_out);

        // grad = grad_out / (2 * sqrt(x)) = 1 / (2 * output)
        assert_eq!(grads.len(), 1);
        assert!((grads[0].data()[0] - 0.25).abs() < 1e-5); // 1/(2*2)
        assert!((grads[0].data()[1] - (1.0 / 6.0)).abs() < 1e-5); // 1/(2*3)
        assert!((grads[0].data()[2] - 0.125).abs() < 1e-5); // 1/(2*4)
        assert_eq!(grad_fn.name(), "SqrtBackward");
    }

    #[test]
    fn test_leaky_relu_backward() {
        let x = Tensor::from_slice(&[1.0, -1.0, 0.0, 2.0]);
        let grad_fn = LeakyReluBackward {
            x,
            negative_slope: 0.01,
        };
        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0]);
        let grads = grad_fn.backward(&grad_out);

        assert_eq!(grads.len(), 1);
        // For x > 0: grad = 1.0, for x <= 0: grad = negative_slope
        assert!((grads[0].data()[0] - 1.0).abs() < 1e-5); // x = 1.0 > 0
        assert!((grads[0].data()[1] - 0.01).abs() < 1e-5); // x = -1.0 <= 0
        assert!((grads[0].data()[2] - 0.01).abs() < 1e-5); // x = 0.0 <= 0
        assert!((grads[0].data()[3] - 1.0).abs() < 1e-5); // x = 2.0 > 0
        assert_eq!(grad_fn.name(), "LeakyReluBackward");
    }

    #[test]
    fn test_gelu_backward() {
        let x = Tensor::from_slice(&[0.0, 1.0, -1.0]);
        let grad_fn = GeluBackward { x };
        let grad_out = Tensor::from_slice(&[1.0, 1.0, 1.0]);
        let grads = grad_fn.backward(&grad_out);

        assert_eq!(grads.len(), 1);
        // GELU'(0) ≈ 0.5
        assert!((grads[0].data()[0] - 0.5).abs() < 0.01);
        assert_eq!(grad_fn.name(), "GeluBackward");
    }

    #[test]
    fn test_softmax_backward() {
        // SoftmaxBackward expects 2D tensor (batch, features)
        let output = Tensor::new(&[0.5, 0.5], &[1, 2]);
        let grad_fn = SoftmaxBackward { output };
        let grad_out = Tensor::new(&[1.0, 0.0], &[1, 2]);
        let grads = grad_fn.backward(&grad_out);

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].shape(), &[1, 2]);
        assert_eq!(grad_fn.name(), "SoftmaxBackward");
    }

    #[test]
    fn test_cross_entropy_backward() {
        // CrossEntropyBackward expects 2D tensor (batch, num_classes)
        let softmax_output = Tensor::new(&[0.7, 0.2, 0.1], &[1, 3]);
        let targets = vec![0_usize]; // target class index (one per batch item)
        let grad_fn = CrossEntropyBackward {
            softmax_output,
            targets,
        };
        let grad_out = Tensor::from_slice(&[1.0]); // scalar after mean reduction
        let grads = grad_fn.backward(&grad_out);

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].shape(), &[1, 3]);
        assert_eq!(grad_fn.name(), "CrossEntropyBackward");
    }

    #[test]
    fn test_broadcast_add_backward() {
        let grad_fn = BroadcastAddBackward {
            x_shape: vec![2, 3],
            y_shape: vec![3],
        };
        let grad_out = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let grads = grad_fn.backward(&grad_out);

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].shape(), &[2, 3]); // x grad unchanged
        assert_eq!(grads[1].shape(), &[3]); // y grad reduced
        assert_eq!(grad_fn.name(), "BroadcastAddBackward");
    }

    #[test]
    fn test_view_backward() {
        let grad_fn = ViewBackward {
            input_shape: vec![2, 3],
        };
        let grad_out = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let grads = grad_fn.backward(&grad_out);

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].shape(), &[2, 3]); // Reshaped back
        assert_eq!(grad_fn.name(), "ViewBackward");
    }

    #[test]
    fn test_transpose_backward_fn() {
        let grad_fn = TransposeBackward;
        let grad_out = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let grads = grad_fn.backward(&grad_out);

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].shape(), &[2, 3]); // Transposed back
        assert_eq!(grad_fn.name(), "TransposeBackward");
    }

    // ========================================================================
    // Helper Function Tests
    // ========================================================================

    #[test]
    fn test_reduce_to_scalar() {
        // Test the scalar reduction helper
        let grad = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let result = reduce_to_scalar(&grad, &[1]);
        assert_eq!(result.data(), &[10.0]); // sum of all elements
        assert_eq!(result.shape(), &[1]);
    }

    #[test]
    fn test_reduce_to_scalar_empty_target() {
        let grad = Tensor::from_slice(&[2.0, 3.0, 5.0]);
        let result = reduce_to_scalar(&grad, &[]);
        assert_eq!(result.data(), &[10.0]);
        let empty_shape: &[usize] = &[];
        assert_eq!(result.shape(), empty_shape);
    }

    #[test]
    fn test_reduce_batch_to_features() {
        // Test 2D -> 1D reduction (summing over batch dimension)
        let grad = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let result = reduce_batch_to_features(&grad, &[3]);
        // Column sums: [1+4, 2+5, 3+6] = [5, 7, 9]
        assert_eq!(result.data(), &[5.0, 7.0, 9.0]);
        assert_eq!(result.shape(), &[3]);
    }

    #[test]
    fn test_needs_batch_reduction_true() {
        let grad = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert!(needs_batch_reduction(&grad, &[2]));
    }

    #[test]
    fn test_needs_batch_reduction_false_1d() {
        let grad = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        assert!(!needs_batch_reduction(&grad, &[3]));
    }

    #[test]
    fn test_needs_batch_reduction_false_shape_mismatch() {
        let grad = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert!(!needs_batch_reduction(&grad, &[4]));
    }

    #[test]
    fn test_maybe_reduce_grad_same_shape() {
        let grad = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
        let result = maybe_reduce_grad(&grad, &[3]);
        assert_eq!(result.data(), grad.data());
        assert_eq!(result.shape(), grad.shape());
    }

    #[test]
    fn test_maybe_reduce_grad_to_scalar() {
        let grad = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let result = maybe_reduce_grad(&grad, &[1]);
        assert_eq!(result.data(), &[10.0]);
        assert_eq!(result.shape(), &[1]);
    }

    #[test]
    fn test_maybe_reduce_grad_to_empty_scalar() {
        let grad = Tensor::from_slice(&[3.0, 7.0]);
        let result = maybe_reduce_grad(&grad, &[]);
        assert_eq!(result.data(), &[10.0]);
        let empty_shape: &[usize] = &[];
        assert_eq!(result.shape(), empty_shape);
    }

    #[test]
    fn test_maybe_reduce_grad_batch_to_features() {
        let grad = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let result = maybe_reduce_grad(&grad, &[2]);
        // Column sums: [1+3+5, 2+4+6] = [9, 12]
        assert_eq!(result.data(), &[9.0, 12.0]);
        assert_eq!(result.shape(), &[2]);
    }

    #[test]
    fn test_maybe_reduce_grad_reshape() {
        // When numel matches but shapes differ (and not 2D->1D case)
        let grad = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);
        let result = maybe_reduce_grad(&grad, &[2, 3]);
        assert_eq!(result.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(result.shape(), &[2, 3]);
    }
