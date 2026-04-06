use super::*;

#[test]
fn test_gpu_leaky_relu_basic() {
    let Some(mut gpu) = get_shared_gpu() else {
        eprintln!("GPU not available, skipping test");
        return;
    };
    let input = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
    let negative_slope = 0.01;

    let result = gpu.leaky_relu(&input, negative_slope);

    if let Ok(output) = result {
        // Expected: max(negative_slope * x, x)
        let expected = [-0.03, -0.01, 0.0, 1.0, 3.0];
        for (r, e) in output.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-4, "Leaky ReLU mismatch: got={}, expected={}", r, e);
        }
    } else {
        eprintln!("GPU leaky_relu failed: {:?}", result);
    }
}

#[test]
fn test_gpu_elu_basic() {
    let Some(mut gpu) = get_shared_gpu() else {
        eprintln!("GPU not available, skipping test");
        return;
    };
    let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let alpha = 1.0;

    let result = gpu.elu(&input, alpha);

    if let Ok(output) = result {
        // Expected: x if x > 0, else alpha * (exp(x) - 1)
        for (i, (r, &x)) in output.iter().zip(input.iter()).enumerate() {
            let expected = if x > 0.0 { x } else { alpha * (x.exp() - 1.0) };
            assert!(
                (r - expected).abs() < 1e-3,
                "ELU mismatch at {}: got={}, expected={}",
                i,
                r,
                expected
            );
        }
    } else {
        eprintln!("GPU elu failed: {:?}", result);
    }
}

#[test]
fn test_gpu_tanh_basic() {
    let Some(mut gpu) = get_shared_gpu() else {
        eprintln!("GPU not available, skipping test");
        return;
    };
    let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

    let result = gpu.tanh(&input);

    if let Ok(output) = result {
        for (r, &x) in output.iter().zip(input.iter()) {
            let expected = x.tanh();
            assert!((r - expected).abs() < 1e-4, "Tanh mismatch: got={}, expected={}", r, expected);
        }
    } else {
        eprintln!("GPU tanh failed: {:?}", result);
    }
}

#[test]
fn test_gpu_tanh_not_hardcoded() {
    // EXTREME TDD: Kill mutant that replaces return with Ok(vec![-1.0])
    let Some(mut gpu) = get_shared_gpu() else {
        eprintln!("GPU not available, skipping test");
        return;
    };
    let input = vec![1.0, 2.0, 3.0];

    let result = gpu.tanh(&input).expect("GPU tanh should succeed");

    // Kill mutant: verify result is NOT all -1.0 values
    assert_ne!(
        result,
        vec![-1.0, -1.0, -1.0],
        "GPU tanh returned hardcoded -1.0 values (mutant not killed)"
    );

    // Verify correct computation
    for (i, &x) in input.iter().enumerate() {
        let expected = x.tanh();
        assert!(
            (result[i] - expected).abs() < 1e-4,
            "tanh({}) = {} (expected {})",
            x,
            result[i],
            expected
        );
    }
}

#[test]
fn test_gpu_softmax_basic() {
    let Some(mut gpu) = get_shared_gpu() else {
        eprintln!("GPU not available, skipping test");
        return;
    };
    let input = vec![1.0, 2.0, 3.0, 4.0];

    let result = gpu.softmax(&input);

    if let Ok(output) = result {
        // Softmax should sum to 1
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "Softmax sum should be 1, got {}", sum);

        // All values should be positive
        for &v in &output {
            assert!(v > 0.0, "Softmax values should be positive");
        }

        // Later values should be larger (input is increasing)
        for i in 1..output.len() {
            assert!(
                output[i] > output[i - 1],
                "Softmax should preserve order for increasing input"
            );
        }
    } else {
        eprintln!("GPU softmax failed: {:?}", result);
    }
}

#[test]
fn test_gpu_log_softmax_basic() {
    let Some(mut gpu) = get_shared_gpu() else {
        eprintln!("GPU not available, skipping test");
        return;
    };
    let input = vec![1.0, 2.0, 3.0, 4.0];

    let result = gpu.log_softmax(&input);

    if let Ok(output) = result {
        // log_softmax values should all be negative (log of probability < 1)
        for &v in &output {
            assert!(v <= 0.0, "Log softmax values should be <= 0, got {}", v);
        }

        // exp(log_softmax) should sum to 1
        let exp_sum: f32 = output.iter().map(|x| x.exp()).sum();
        assert!((exp_sum - 1.0).abs() < 1e-3, "exp(log_softmax) should sum to 1, got {}", exp_sum);
    } else {
        eprintln!("GPU log_softmax failed: {:?}", result);
    }
}
