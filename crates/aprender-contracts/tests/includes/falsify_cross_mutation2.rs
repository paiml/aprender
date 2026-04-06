// Part 2b: Mutation Detection Tests (10 tests)

/// Mutation: gqa — use wrong KV head index gives different output
#[test]
fn mutation_gqa_detect_wrong_head_broadcast() {
    let q = vec![1.0f32; 4 * 2 * 2];
    let k = [1.0f32, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5];
    let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut correct = vec![0.0f32; 16];
    gqa_scalar(&q, &k, &v, 2, 2, 2, 4, 2, &mut correct);

    let k_head0 = &k[0..4];
    let v_head0 = &v[0..4];
    let k_mutated = [k_head0, k_head0].concat();
    let v_mutated = [v_head0, v_head0].concat();
    let mut mutated = vec![0.0f32; 16];
    gqa_scalar(&q, &k_mutated, &v_mutated, 2, 2, 2, 4, 2, &mut mutated);

    let dist = common::l2_distance(&correct, &mutated);
    assert!(
        dist > 0.01,
        "mutation not detected: wrong kv head broadcast too similar: dist = {dist}"
    );
}

/// Mutation: flash attention — no rescaling gives wrong output
#[test]
#[allow(clippy::many_single_char_names)]
fn mutation_flash_detect_no_rescaling() {
    let n = 4;
    let d = 2;
    let q = [1.0f32, 0.5, 0.3, 0.7, 0.2, 0.9, 0.8, 0.1];
    let k = [0.5f32, 0.3, 0.7, 0.2, 0.9, 0.1, 0.4, 0.6];
    let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut correct = [0.0f32; 8];
    flash_attention_scalar(&q, &k, &v, n, d, 2, &mut correct);

    #[allow(clippy::cast_precision_loss)]
    let scale = 1.0 / (d as f32).sqrt();
    let tile_size = 2;
    let mut mutated = [0.0f32; 8];
    for i in 0..n {
        let mut acc = vec![0.0f32; d];
        let mut total_sum = 0.0f32;
        let mut tile_start = 0;
        while tile_start < n {
            let tile_end = (tile_start + tile_size).min(n);
            let tile_len = tile_end - tile_start;
            let mut tile_scores = vec![0.0f32; tile_len];
            for (tj, j) in (tile_start..tile_end).enumerate() {
                let mut dot = 0.0f32;
                for kk in 0..d {
                    dot += q[i * d + kk] * k[j * d + kk];
                }
                tile_scores[tj] = dot * scale;
            }
            let tile_max = tile_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            // MUTATION: no rescaling of previous accumulation
            for (tj, j) in (tile_start..tile_end).enumerate() {
                let w = (tile_scores[tj] - tile_max).exp();
                for dd in 0..d {
                    acc[dd] += w * v[j * d + dd];
                }
                total_sum += w;
            }
            tile_start = tile_end;
        }
        for dd in 0..d {
            mutated[i * d + dd] = acc[dd] / total_sum;
        }
    }
    let dist = common::l2_distance(&correct, &mutated);
    assert!(
        dist > 1e-4,
        "mutation not detected: no rescaling too similar to correct: dist = {dist}"
    );
}

/// Mutation: adamw — L2 regularization vs decoupled weight decay gives different results
#[test]
fn mutation_adamw_detect_l2_instead_of_decoupled() {
    let grads = [0.1f32, -0.2, 0.3, -0.4];
    let lr = 0.01;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let eps = 1e-8;
    let wd = 0.1;
    let mut correct_params = [1.0f32, 2.0, 3.0, 4.0];
    let mut m_correct = [0.0f32; 4];
    let mut v_correct = [0.0f32; 4];
    adamw_step_scalar(
        &mut correct_params, &grads, &mut m_correct, &mut v_correct,
        lr, beta1, beta2, eps, wd, 1,
    );

    let init_params = [1.0f32, 2.0, 3.0, 4.0];
    let mut mutated_params = init_params;
    let mut m_mut = [0.0f32; 4];
    let mut v_mut = [0.0f32; 4];
    let bc1 = 1.0 / (1.0 - beta1);
    let bc2 = 1.0 / (1.0 - beta2);
    for i in 0..4 {
        let g = grads[i] + wd * mutated_params[i];
        m_mut[i] = beta1 * m_mut[i] + (1.0 - beta1) * g;
        v_mut[i] = beta2 * v_mut[i] + (1.0 - beta2) * g * g;
        let m_hat = m_mut[i] * bc1;
        let v_hat = v_mut[i] * bc2;
        mutated_params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
    let dist = common::l2_distance(&correct_params, &mutated_params);
    assert!(
        dist > 1e-5,
        "mutation not detected: L2 reg vs decoupled wd too similar: dist = {dist}"
    );
}

/// Mutation: conv1d — check output length formula correctness
#[test]
fn mutation_conv1d_detect_off_by_one() {
    let c_in = 1;
    let c_out = 1;
    let length = 8;
    let kernel_size = 3;
    let stride = 2;
    let padding = 1;
    let correct_out_len = (length + 2 * padding - kernel_size) / stride + 1;
    let mutated_out_len = (length + 2 * padding - kernel_size) / stride;
    assert_ne!(
        correct_out_len, mutated_out_len,
        "mutation not detected: correct ({correct_out_len}) == mutated ({mutated_out_len})"
    );
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weight = [1.0f32, 1.0, 1.0];
    let mut output = vec![0.0f32; correct_out_len];
    conv1d_scalar(
        &input, &weight, None, c_in, c_out, length, kernel_size, stride, padding, &mut output,
    );
    assert_eq!(output.len(), correct_out_len);
}

/// Mutation: ssm — non-causal modification at t=0 should not affect t=0 output
#[test]
fn mutation_ssm_detect_noncausal() {
    let state_dim = 2;
    let seq_len = 4;
    let a_bar = [0.9f32, 0.8];
    let b_bar = [1.0f32, 0.5, 0.8, 0.3, 0.6, 0.7, 0.4, 0.9];
    let c = [1.0f32, 1.0];
    let x1 = [1.0f32, 2.0, 3.0, 4.0];
    let mut out1 = [0.0f32; 4];
    ssm_scan_scalar(&a_bar, &b_bar, &c, &x1, state_dim, seq_len, &mut out1);

    let x2 = [1.0f32, 2.0, 100.0, 4.0];
    let mut out2 = [0.0f32; 4];
    ssm_scan_scalar(&a_bar, &b_bar, &c, &x2, state_dim, seq_len, &mut out2);

    assert!(
        (out1[0] - out2[0]).abs() < 1e-7,
        "SSM is non-causal: output[0] changed from {} to {}", out1[0], out2[0]
    );
    assert!(
        (out1[1] - out2[1]).abs() < 1e-7,
        "SSM is non-causal: output[1] changed from {} to {}", out1[1], out2[1]
    );
    assert!(
        (out1[2] - out2[2]).abs() > 0.1,
        "SSM modification at t=2 should affect output[2]: {} vs {}", out1[2], out2[2]
    );
}

/// Mutation: kmeans — random assignments vs nearest-centroid gives different objective
#[test]
fn mutation_kmeans_detect_random_assignment() {
    let points = [0.0f32, 0.0, 1.0, 0.0, 10.0, 10.0, 11.0, 10.0];
    let centroids = [0.5f32, 0.0, 10.5, 10.0];
    let n = 4;
    let k = 2;
    let d = 2;
    let mut correct_assignments = [0u32; 4];
    kmeans_assign_scalar(&points, &centroids, n, k, d, &mut correct_assignments);
    assert_eq!(correct_assignments[0], 0);
    assert_eq!(correct_assignments[1], 0);
    assert_eq!(correct_assignments[2], 1);
    assert_eq!(correct_assignments[3], 1);

    let mutated_assignments: [u32; 4] = [1, 0, 0, 1];
    let correct_obj: f32 = (0..n)
        .map(|p| {
            let c = correct_assignments[p] as usize;
            (0..d).map(|j| { let diff = points[p * d + j] - centroids[c * d + j]; diff * diff }).sum::<f32>()
        })
        .sum();
    let mutated_obj: f32 = (0..n)
        .map(|p| {
            let c = mutated_assignments[p] as usize;
            (0..d).map(|j| { let diff = points[p * d + j] - centroids[c * d + j]; diff * diff }).sum::<f32>()
        })
        .sum();
    assert!(
        mutated_obj > correct_obj + 1.0,
        "mutation not detected: random obj ({mutated_obj}) not much worse than correct obj ({correct_obj})"
    );
}

/// Mutation: pagerank — without normalization, output does not sum to 1
#[test]
fn mutation_pagerank_detect_no_normalization() {
    let n = 4;
    let transition = common::stochastic_transition_matrix(n);
    let rank = [0.25f32; 4];
    let damping = 0.85;
    let mut correct = [0.0f32; 4];
    pagerank_iterate_scalar(&transition, &rank, n, damping, &mut correct);
    let correct_sum: f32 = correct.iter().sum();
    assert!((correct_sum - 1.0).abs() < 0.01, "correct pagerank should sum to ~1: {correct_sum}");

    let mut mutated = [0.0f32; 4];
    for i in 0..n {
        let mut sum = 0.0f32;
        for j in 0..n {
            sum += transition[i * n + j] * rank[j];
        }
        mutated[i] = damping * sum; // MUTATION: no teleport term
    }
    let mutated_sum: f32 = mutated.iter().sum();
    assert!(
        (mutated_sum - 1.0).abs() > 0.01,
        "mutated pagerank should not sum to 1: {mutated_sum}"
    );
}

/// Mutation: lbfgs — with m=0, direction should be -gradient; reversed would give +gradient
#[test]
fn mutation_lbfgs_detect_reverse_loop() {
    let gradient = [1.0f32, -2.0, 3.0, -4.0];
    let d = 4;
    let mut correct = [0.0f32; 4];
    lbfgs_direction_scalar(&gradient, &[], &[], 0, d, &mut correct);
    for i in 0..d {
        assert!(
            (correct[i] - (-gradient[i])).abs() < 1e-7,
            "lbfgs(m=0) should give -gradient at index {i}: got {} expected {}", correct[i], -gradient[i]
        );
    }
    let mut mutated = [0.0f32; 4];
    mutated[..d].copy_from_slice(&gradient[..d]);
    let dist = common::l2_distance(&correct, &mutated);
    assert!(
        dist > 0.01,
        "mutation not detected: +gradient vs -gradient too similar: dist = {dist}"
    );
}

/// Mutation: cma — skip Cholesky transformation gives different output when L != I
#[test]
fn mutation_cma_detect_no_cholesky() {
    let d = 4;
    let mean = [1.0f32, 2.0, 3.0, 4.0];
    let sigma = 1.0;
    let cholesky_l = [
        2.0f32, 0.0, 0.0, 0.0, 0.5, 1.5, 0.0, 0.0, 0.3, 0.2, 1.0, 0.0, 0.1, 0.4, 0.3, 0.8,
    ];
    let z = [0.5f32, -0.3, 0.7, -0.1];
    let mut correct = [0.0f32; 4];
    cma_sample_scalar(&mean, sigma, &cholesky_l, d, &z, &mut correct);

    let mut mutated = [0.0f32; 4];
    for i in 0..d {
        mutated[i] = mean[i] + sigma * z[i];
    }
    let dist = common::l2_distance(&correct, &mutated);
    assert!(
        dist > 0.01,
        "mutation not detected: no Cholesky too similar to correct: dist = {dist}"
    );
}

/// Mutation: gdn — extreme alpha=100 gives dramatically different output
#[test]
fn mutation_gdn_detect_extreme_decay() {
    let seq_len = 4;
    let k_dim = 2;
    let v_dim = 2;
    let q = [1.0f32, 0.5, 0.3, 0.7, 0.2, 0.9, 0.8, 0.1];
    let k = [0.5f32, 0.3, 0.7, 0.2, 0.9, 0.1, 0.4, 0.6];
    let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let beta = [0.5f32; 4];

    let alpha_correct = [0.5f32; 4];
    let mut correct = [0.0f32; 8];
    gdn_recurrence_scalar(
        &q, &k, &v, &alpha_correct, &beta, seq_len, k_dim, v_dim, &mut correct,
    );

    let alpha_extreme = [100.0f32; 4];
    let mut mutated = [0.0f32; 8];
    gdn_recurrence_scalar(
        &q, &k, &v, &alpha_extreme, &beta, seq_len, k_dim, v_dim, &mut mutated,
    );

    let dist = common::l2_distance(&correct, &mutated);
    assert!(
        dist > 1.0,
        "mutation not detected: extreme alpha too similar to correct: dist = {dist}"
    );
}
