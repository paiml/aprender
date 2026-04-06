
// ==================== IMP-304: Trueno SIMD Layer Norm & RMS Norm ====================
// Per spec: 4x norm speedup for production inference
// Target: < 50µs for 4096 dim layer norm

/// IMP-304a: Test trueno layer_norm correctness
#[test]
fn test_imp_304a_trueno_layer_norm_correctness() {
    use trueno::Vector;

    // Test case 1: Simple normalization
    let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let vec = Vector::from_slice(&data);
    let gamma = Vector::from_slice(&vec![1.0; 5]);
    let beta = Vector::from_slice(&vec![0.0; 5]);

    let normed = vec.layer_norm(&gamma, &beta, 1e-5).expect("layer_norm");
    let normed_data = normed.as_slice().to_vec();

    // Verify: mean should be ~0, variance should be ~1
    let mean: f32 = normed_data.iter().sum::<f32>() / normed_data.len() as f32;
    let var: f32 =
        normed_data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / normed_data.len() as f32;

    assert!(
        mean.abs() < 1e-5,
        "IMP-304a: Mean should be ~0, got {}",
        mean
    );
    assert!(
        (var - 1.0).abs() < 0.1,
        "IMP-304a: Variance should be ~1, got {}",
        var
    );

    // Test case 2: With affine transform (gamma=2, beta=1)
    let gamma2 = Vector::from_slice(&vec![2.0; 5]);
    let beta2 = Vector::from_slice(&vec![1.0; 5]);
    let normed2 = vec
        .layer_norm(&gamma2, &beta2, 1e-5)
        .expect("layer_norm with affine");
    let normed2_data = normed2.as_slice().to_vec();

    // After gamma=2, beta=1: output = 2*normalized + 1
    // Mean should be ~1 (since normalized mean is 0)
    let mean2: f32 = normed2_data.iter().sum::<f32>() / normed2_data.len() as f32;
    assert!(
        (mean2 - 1.0).abs() < 0.1,
        "IMP-304a: Affine mean should be ~1, got {}",
        mean2
    );

    println!("\nIMP-304a: Trueno Layer Norm Correctness:");
    println!("  Simple: mean={:.6}, var={:.6}", mean, var);
    println!("  Affine (gamma=2, beta=1): mean={:.6}", mean2);
    println!("  Status: PASS");
}

/// IMP-304b: Test trueno layer_norm performance vs scalar
#[test]
fn test_imp_304b_trueno_layer_norm_perf_comparison() {
    use std::time::Instant;
    use trueno::Vector;

    let sizes = [768, 2048, 2560, 4096];
    let iterations = 1000;

    println!("\nIMP-304b: Layer Norm Performance (trueno SIMD vs scalar):");
    println!(
        "  {:>6} | {:>10} | {:>10} | {:>8}",
        "Dim", "Trueno µs", "Scalar µs", "Speedup"
    );
    println!("  -------|------------|------------|----------");

    for size in sizes {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let vec = Vector::from_slice(&data);

        // Trueno SIMD
        let start = Instant::now();
        for _ in 0..iterations {
            let _normed = vec.layer_norm_simple(1e-5).expect("layer_norm_simple");
        }
        let trueno_us = start.elapsed().as_micros() as f64 / iterations as f64;

        // Scalar baseline
        let start = Instant::now();
        for _ in 0..iterations {
            let mean: f32 = data.iter().sum::<f32>() / size as f32;
            let var: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / size as f32;
            let inv_std = (var + 1e-5).sqrt().recip();
            let _output: Vec<f32> = data.iter().map(|x| (x - mean) * inv_std).collect();
        }
        let scalar_us = start.elapsed().as_micros() as f64 / iterations as f64;

        let speedup = scalar_us / trueno_us;
        let status = if trueno_us < 50.0 { "PASS" } else { "FAIL" };

        println!(
            "  {:>6} | {:>10.2} | {:>10.2} | {:>7.2}x [{}]",
            size, trueno_us, scalar_us, speedup, status
        );
    }
}

/// IMP-304c: RMS Norm implementation (used by LLaMA, Mistral, etc.)
/// RMS Norm: x / sqrt(mean(x^2) + eps) * gamma
#[test]
fn test_imp_304c_rms_norm() {
    use std::time::Instant;
    use trueno::Vector;

    // RMS Norm helper function (trueno doesn't have native rms_norm yet)
    fn rms_norm_simd(input: &Vector<f32>, gamma: &[f32], eps: f32) -> Vec<f32> {
        let data = input.as_slice().to_vec();
        let n = data.len();

        // Compute RMS: sqrt(mean(x^2))
        let mean_sq: f32 = data.iter().map(|x| x * x).sum::<f32>() / n as f32;
        let rms = (mean_sq + eps).sqrt();
        let inv_rms = 1.0 / rms;

        // Apply normalization and scale
        data.iter()
            .zip(gamma.iter())
            .map(|(&x, &g)| x * inv_rms * g)
            .collect()
    }

    let sizes = [768, 2048, 2560, 4096];
    let iterations = 1000;

    println!("\nIMP-304c: RMS Norm Performance:");
    println!("  {:>6} | {:>10} | {:>8}", "Dim", "Latency µs", "Status");
    println!("  -------|------------|----------");

    for size in sizes {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01 + 0.1).collect();
        let vec = Vector::from_slice(&data);
        let gamma: Vec<f32> = vec![1.0; size];

        let start = Instant::now();
        for _ in 0..iterations {
            let _normed = rms_norm_simd(&vec, &gamma, 1e-5);
        }
        let avg_us = start.elapsed().as_micros() as f64 / iterations as f64;

        let status = if avg_us < 50.0 { "PASS" } else { "NEEDS OPT" };
        println!("  {:>6} | {:>10.2} | {}", size, avg_us, status);
    }

    // Verify correctness
    let test_data = vec![1.0_f32, 2.0, 3.0, 4.0];
    let test_vec = Vector::from_slice(&test_data);
    let test_gamma = vec![1.0; 4];
    let result = rms_norm_simd(&test_vec, &test_gamma, 1e-5);

    // Expected: RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.739
    // Output = [1/2.739, 2/2.739, 3/2.739, 4/2.739] ≈ [0.365, 0.730, 1.095, 1.461]
    let expected_rms = (30.0_f32 / 4.0).sqrt();
    let expected: Vec<f32> = test_data.iter().map(|x| x / expected_rms).collect();

    for (got, exp) in result.iter().zip(expected.iter()) {
        assert!(
            (got - exp).abs() < 1e-4,
            "IMP-304c: RMS norm mismatch: got {}, expected {}",
            got,
            exp
        );
    }
    println!("  Correctness: VERIFIED");
}

/// IMP-304d: Integration with realizar forward pass timing
#[test]
fn test_imp_304d_layer_norm_integration() {
    use std::time::Instant;
    use trueno::Vector;

    // Simulate phi-2 layer norm dimensions
    let hidden_dim = 2560;
    let num_layers = 32;
    let iterations = 100;

    let input: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.01).collect();
    let input_vec = Vector::from_slice(&input);

    // Time a full forward pass worth of layer norms (2 per layer: attn_norm + ffn_norm)
    let norms_per_forward = num_layers * 2;

    let start = Instant::now();
    for _ in 0..iterations {
        for _ in 0..norms_per_forward {
            let _normed = input_vec.layer_norm_simple(1e-5).expect("layer_norm");
        }
    }
    let total_us = start.elapsed().as_micros() as f64;
    let per_forward_us = total_us / iterations as f64;
    let per_norm_us = per_forward_us / norms_per_forward as f64;

    println!("\nIMP-304d: Layer Norm Integration (phi-2 scale):");
    println!("  Hidden dim: {}", hidden_dim);
    println!("  Layers: {} (× 2 norms each)", num_layers);
    println!("  Per norm: {:.2}µs", per_norm_us);
    println!(
        "  Per forward (all norms): {:.2}µs ({:.2}ms)",
        per_forward_us,
        per_forward_us / 1000.0
    );

    let target_ms = 5.0; // Target: all norms < 5ms per forward
    let status = if per_forward_us / 1000.0 < target_ms {
        "PASS"
    } else {
        "NEEDS WORK"
    };
    println!("  Status: {} (target: <{}ms)", status, target_ms);
}

// ==================== IMP-305: Trueno SIMD Softmax ====================
// Per spec: 4x softmax speedup with numerical stability
// Target: < 100µs for 32K vocab softmax

/// IMP-305a: Test trueno softmax correctness and numerical stability
#[test]
fn test_imp_305a_trueno_softmax_correctness() {
    use trueno::Vector;

    // Test case 1: Simple softmax
    let data = vec![1.0_f32, 2.0, 3.0, 4.0];
    let vec = Vector::from_slice(&data);
    let result = vec.softmax().expect("softmax");
    let result_data = result.as_slice().to_vec();

    // Softmax should sum to 1
    let sum: f32 = result_data.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "IMP-305a: Softmax should sum to 1, got {}",
        sum
    );

    // Higher inputs should have higher probabilities
    for i in 0..result_data.len() - 1 {
        assert!(
            result_data[i] < result_data[i + 1],
            "IMP-305a: Softmax should be monotonic"
        );
    }

    // Test case 2: Numerical stability with large values
    let large_data = vec![1000.0_f32, 1001.0, 1002.0, 1003.0];
    let large_vec = Vector::from_slice(&large_data);
    let large_result = large_vec.softmax().expect("softmax large");
    let large_result_data = large_result.as_slice();
    let large_sum: f32 = large_result_data.iter().sum();
    assert!(
        (large_sum - 1.0).abs() < 1e-4,
        "IMP-305a: Large value softmax should sum to 1, got {}",
        large_sum
    );
    assert!(
        large_result_data.iter().all(|&x| x.is_finite()),
        "IMP-305a: Large value softmax should be finite"
    );

    // Test case 3: Numerical stability with negative values
    let neg_data = vec![-1000.0_f32, -999.0, -998.0, -997.0];
    let neg_vec = Vector::from_slice(&neg_data);
    let neg_result = neg_vec.softmax().expect("softmax negative");
    let neg_sum: f32 = neg_result.as_slice().iter().sum();
    assert!(
        (neg_sum - 1.0).abs() < 1e-4,
        "IMP-305a: Negative value softmax should sum to 1, got {}",
        neg_sum
    );

    println!("\nIMP-305a: Trueno Softmax Correctness:");
    println!("  Simple: sum={:.6}, monotonic=true", sum);
    println!("  Large values (1000+): sum={:.6}, all finite", large_sum);
    println!("  Negative values: sum={:.6}", neg_sum);
    println!("  Status: PASS");
}

/// IMP-305b: Test trueno softmax performance
#[test]
fn test_imp_305b_trueno_softmax_perf() {
    use std::time::Instant;
    use trueno::Vector;

    // Test vocab sizes relevant to LLMs
    let sizes = [1024, 4096, 32000, 51200]; // Common vocab sizes
    let iterations = 1000;

    println!("\nIMP-305b: Softmax Performance:");
    println!(
        "  {:>6} | {:>10} | {:>8}",
        "VocabSz", "Latency µs", "Status"
    );
    println!("  -------|------------|----------");

    for size in sizes {
        let data: Vec<f32> = (0..size)
            .map(|i| (i as f32) * 0.001 - (size as f32 / 2000.0))
            .collect();
        let vec = Vector::from_slice(&data);

        let start = Instant::now();
        for _ in 0..iterations {
            let _result = vec.softmax().expect("softmax");
        }
        let avg_us = start.elapsed().as_micros() as f64 / iterations as f64;

        let target = if size <= 32000 { 100.0 } else { 200.0 };
        let status = if avg_us < target { "PASS" } else { "NEEDS OPT" };
        println!("  {:>6} | {:>10.2} | {}", size, avg_us, status);
    }
}

/// IMP-305c: Softmax integration with attention mechanism
#[test]
fn test_imp_305c_attention_softmax_integration() {
    use std::time::Instant;
    use trueno::Vector;

    // Simulate attention softmax: seq_len × seq_len scores
    let seq_lengths = [128, 256, 512, 1024];
    let num_heads = 32;
    let iterations = 100;

    println!("\nIMP-305c: Attention Softmax Integration:");
    println!(
        "  {:>8} | {:>12} | {:>12} | {:>8}",
        "SeqLen", "Per Head µs", "All Heads µs", "Status"
    );
    println!("  ---------|--------------|--------------|----------");

    for seq_len in seq_lengths {
        // Each head does seq_len softmax operations (one per query position)
        let scores: Vec<f32> = (0..seq_len).map(|i| (i as f32) * 0.1 - 5.0).collect();
        let scores_vec = Vector::from_slice(&scores);

        // Time softmax for all heads × all positions
        let start = Instant::now();
        for _ in 0..iterations {
            for _ in 0..num_heads {
                for _ in 0..seq_len {
                    let _probs = scores_vec.softmax().expect("softmax");
                }
            }
        }
        let total_us = start.elapsed().as_micros() as f64;
        let per_head_us = total_us / (iterations * num_heads) as f64;
        let all_heads_us = total_us / iterations as f64;

        let target_ms = 50.0; // Target: all attention softmax < 50ms
        let status = if all_heads_us / 1000.0 < target_ms {
            "PASS"
        } else {
            "SLOW"
        };

        println!(
            "  {:>8} | {:>12.2} | {:>12.2} | {}",
            seq_len, per_head_us, all_heads_us, status
        );
    }
}

/// IMP-305d: Combined norm + softmax timing (common pattern)
#[test]
fn test_imp_305d_norm_softmax_combined() {
    use std::time::Instant;
    use trueno::Vector;

    // Common inference pattern: layer_norm -> attention (with softmax) -> layer_norm
    let hidden_dim = 2560;
    let seq_len = 256;
    let iterations = 100;

    let hidden: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.01).collect();
    let hidden_vec = Vector::from_slice(&hidden);

    let scores: Vec<f32> = (0..seq_len).map(|i| (i as f32) * 0.1 - 12.8).collect();
    let scores_vec = Vector::from_slice(&scores);

    // Measure: 2× layer_norm + seq_len× softmax
    let start = Instant::now();
    for _ in 0..iterations {
        // Pre-attention norm
        let _normed1 = hidden_vec.layer_norm_simple(1e-5).expect("norm1");

        // Attention softmax (per position)
        for _ in 0..seq_len {
            let _probs = scores_vec.softmax().expect("softmax");
        }

        // Post-attention norm (before FFN)
        let _normed2 = hidden_vec.layer_norm_simple(1e-5).expect("norm2");
    }
    let total_us = start.elapsed().as_micros() as f64;
    let per_iter_us = total_us / iterations as f64;
    let per_iter_ms = per_iter_us / 1000.0;

    println!("\nIMP-305d: Combined Norm + Softmax (per layer):");
    println!("  Hidden dim: {}", hidden_dim);
    println!("  Seq len: {}", seq_len);
    println!("  Operations: 2× layer_norm + {}× softmax", seq_len);
    println!("  Total: {:.2}ms per layer", per_iter_ms);

    let target_ms = 100.0;
    let status = if per_iter_ms < target_ms {
        "PASS"
    } else {
        "NEEDS WORK"
    };
    println!("  Status: {} (target: <{}ms)", status, target_ms);
}
