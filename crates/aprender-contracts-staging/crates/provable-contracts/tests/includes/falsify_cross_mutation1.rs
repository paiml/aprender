// Part 2a: Mutation Detection Tests (11 tests)

/// Mutation: softmax — no max subtraction causes overflow on large inputs
#[test]
fn mutation_softmax_detect_no_max_subtraction() {
    let input = [1000.0f32, 1000.0, 1000.0, 1000.0];
    let mut correct = [0.0f32; 4];
    softmax_scalar(&input, &mut correct);
    common::assert_all_finite(&correct);
    common::assert_probability_distribution(&correct, 1e-5);

    let mut mutated = [0.0f32; 4];
    for (i, &x) in input.iter().enumerate() {
        mutated[i] = x.exp();
    }
    let sum: f32 = mutated.iter().sum();
    if sum.is_finite() && sum > 0.0 {
        for m in &mut mutated {
            *m /= sum;
        }
    }
    let mutated_has_nan_or_inf = mutated.iter().any(|x| !x.is_finite());
    assert!(
        mutated_has_nan_or_inf,
        "mutated softmax should overflow but got: {mutated:?}"
    );
}

/// Mutation: rmsnorm — eps=0 on zero input causes NaN
#[test]
fn mutation_rmsnorm_detect_zero_eps() {
    let input = [0.0f32, 0.0, 0.0, 0.0];
    let gamma = [1.0f32; 4];
    let mut correct = [0.0f32; 4];
    rmsnorm_scalar(&input, &gamma, 1e-5, &mut correct);
    common::assert_all_finite(&correct);

    let mut mutated = [0.0f32; 4];
    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    #[allow(clippy::cast_precision_loss)]
    let rms = (sum_sq / input.len() as f32 + 0.0).sqrt();
    let inv_rms = 1.0 / rms;
    for i in 0..4 {
        mutated[i] = input[i] * inv_rms * gamma[i];
    }
    let mutated_has_nan_or_inf = mutated.iter().any(|x| !x.is_finite());
    assert!(
        mutated_has_nan_or_inf,
        "mutated rmsnorm (eps=0, zero input) should produce NaN/inf but got: {mutated:?}"
    );
}

/// Mutation: layernorm — eps=0 on constant input causes NaN
#[test]
fn mutation_layernorm_detect_no_eps() {
    let input = [5.0f32, 5.0, 5.0, 5.0];
    let gamma = [1.0f32; 4];
    let beta = [0.0f32; 4];
    let mut correct = [0.0f32; 4];
    layernorm_scalar(&input, &gamma, &beta, 1e-5, &mut correct);
    common::assert_all_finite(&correct);

    #[allow(clippy::cast_precision_loss)]
    let n = input.len() as f32;
    let mean: f32 = input.iter().sum::<f32>() / n;
    let var: f32 = input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + 0.0).sqrt();
    let mut mutated = [0.0f32; 4];
    for i in 0..4 {
        mutated[i] = gamma[i] * (input[i] - mean) * inv_std + beta[i];
    }
    let mutated_has_nan_or_inf = mutated.iter().any(|x| !x.is_finite());
    assert!(
        mutated_has_nan_or_inf,
        "mutated layernorm (eps=0, constant input) should produce NaN/inf but got: {mutated:?}"
    );
}

/// Mutation: batchnorm — no running stat update leaves `running_mean` at 0
#[test]
fn mutation_batchnorm_detect_no_running_update() {
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let gamma = [1.0f32];
    let beta = [0.0f32];
    let mut running_mean = [0.0f32];
    let mut running_var = [0.0f32];
    let mut output = [0.0f32; 4];
    batchnorm_scalar(
        &input, 4, 1, &gamma, &beta, 1e-5,
        &mut running_mean, &mut running_var, &mut output, 0.1, true,
    );
    assert!(
        running_mean[0].abs() > 0.01,
        "running_mean should be updated after training: {}", running_mean[0]
    );
    let mutated_running_mean = 0.0f32;
    assert!(
        (running_mean[0] - mutated_running_mean).abs() > 0.01,
        "mutation not detected: running_mean ({}) same as no-update (0.0)", running_mean[0]
    );
}

/// Mutation: activation — replace gelu with relu gives different results for negative inputs
#[test]
fn mutation_activation_detect_relu_for_gelu() {
    let input = [-2.0f32, -1.0, -0.5, 0.5];
    let mut correct_gelu = [0.0f32; 4];
    gelu_scalar(&input, &mut correct_gelu);
    let mut mutated_relu = [0.0f32; 4];
    relu_scalar(&input, &mut mutated_relu);
    let dist = common::l2_distance(&correct_gelu, &mutated_relu);
    assert!(
        dist > 0.01,
        "mutation not detected: gelu and relu too similar: dist = {dist}"
    );
}

/// Mutation: silu — replace sigmoid with constant 0.5 gives different output
#[test]
fn mutation_silu_detect_constant_sigmoid() {
    let input = [2.0f32, -1.0, 3.0, -0.5];
    let mut correct = [0.0f32; 4];
    silu_standalone_scalar(&input, &mut correct);
    let mut mutated = [0.0f32; 4];
    for (i, &x) in input.iter().enumerate() {
        mutated[i] = 0.5 * x;
    }
    let dist = common::l2_distance(&correct, &mutated);
    assert!(
        dist > 0.01,
        "mutation not detected: silu and 0.5*x too similar: dist = {dist}"
    );
}

/// Mutation: swiglu — replace silu gate with relu gate gives different output
#[test]
fn mutation_swiglu_detect_relu_gate() {
    let gate = [-2.0f32, -1.0, 0.5, 2.0];
    let value = [1.0f32, 2.0, 3.0, 4.0];
    let mut correct = [0.0f32; 4];
    swiglu_scalar(&gate, &value, &mut correct);
    let mut mutated = [0.0f32; 4];
    for i in 0..4 {
        let relu_g = gate[i].max(0.0);
        mutated[i] = relu_g * value[i];
    }
    let dist = common::l2_distance(&correct, &mutated);
    assert!(
        dist > 0.01,
        "mutation not detected: swiglu(silu) vs swiglu(relu) too similar: dist = {dist}"
    );
}

/// Mutation: cross-entropy — no max subtraction in log-sum-exp causes overflow
#[test]
fn mutation_cross_entropy_detect_no_max_logsumexp() {
    let logits = [500.0f32, 500.0, 500.0, 500.0];
    let targets = [1.0f32, 0.0, 0.0, 0.0];
    let correct = cross_entropy_scalar(&targets, &logits);
    assert!(correct.is_finite());

    let sum_exp: f32 = logits.iter().map(|&x| x.exp()).sum();
    let lse = sum_exp.ln();
    let mut log_sm_mutated = [0.0f32; 4];
    for (i, &x) in logits.iter().enumerate() {
        log_sm_mutated[i] = x - lse;
    }
    let mutated_loss: f32 = -targets
        .iter()
        .zip(log_sm_mutated.iter())
        .map(|(&t, &ls)| t * ls)
        .sum::<f32>();
    let mutated_bad = !mutated_loss.is_finite();
    assert!(
        mutated_bad,
        "mutated cross-entropy should overflow but got: {mutated_loss}"
    );
}

/// Mutation: rope — swap sin and cos gives different output
#[test]
fn mutation_rope_detect_swap_sincos() {
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let position = 5u32;
    let dim = 4;
    let base = 10000.0f32;
    let mut correct = [0.0f32; 4];
    rope_scalar(&input, position, dim, base, &mut correct);

    let mut mutated = [0.0f32; 4];
    let half_dim = dim / 2;
    for k in 0..half_dim {
        #[allow(clippy::cast_precision_loss)]
        let freq = base.powf(-2.0 * k as f32 / dim as f32);
        #[allow(clippy::cast_precision_loss)]
        let theta = freq * position as f32;
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let x0 = input[2 * k];
        let x1 = input[2 * k + 1];
        mutated[2 * k] = x0 * sin_t - x1 * cos_t;
        mutated[2 * k + 1] = x0 * cos_t + x1 * sin_t;
    }
    let dist = common::l2_distance(&correct, &mutated);
    assert!(
        dist > 0.01,
        "mutation not detected: swapped sin/cos too similar to correct: dist = {dist}"
    );
}

/// Mutation: matmul — swap row/col indices gives wrong result
#[test]
#[allow(clippy::many_single_char_names)]
fn mutation_matmul_detect_swap_indices() {
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let m = 2;
    let p = 3;
    let n = 2;
    let mut correct = [0.0f32; 4];
    matmul_scalar(&a, &b, m, p, n, &mut correct);

    let mut mutated = [0.0f32; 4];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..p {
                sum += a[i * p + k] * b[k * n + j];
            }
            mutated[j * m + i] = sum;
        }
    }
    let dist = common::l2_distance(&correct, &mutated);
    assert!(
        dist > 0.01,
        "mutation not detected: swapped indices too similar: dist = {dist}"
    );
}

/// Mutation: attention — use `1/d_k` instead of `1/sqrt(d_k)` gives different output
#[test]
#[allow(clippy::many_single_char_names)]
fn mutation_attention_detect_wrong_scaling() {
    let q = [1.0f32, 0.5, 0.3, 0.7];
    let k = [0.5f32, 0.3, 0.7, 0.2];
    let v = [1.0f32, 2.0, 3.0, 4.0];
    let n = 2;
    let m = 2;
    let d_k = 2;
    let d_v = 2;
    let mut correct = [0.0f32; 4];
    attention_scalar(&q, &k, &v, n, m, d_k, d_v, &mut correct);

    #[allow(clippy::cast_precision_loss)]
    let wrong_scale = 1.0 / d_k as f32;
    let mut scores = vec![0.0f32; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut dot = 0.0f32;
            for kk in 0..d_k {
                dot += q[i * d_k + kk] * k[j * d_k + kk];
            }
            scores[i * m + j] = dot * wrong_scale;
        }
    }
    for i in 0..n {
        let row = &mut scores[i * m..(i + 1) * m];
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in row.iter_mut() {
            *s = (*s - max_val).exp();
            sum += *s;
        }
        for s in row.iter_mut() {
            *s /= sum;
        }
    }
    let mut mutated = [0.0f32; 4];
    for i in 0..n {
        for j in 0..d_v {
            let mut sum = 0.0f32;
            for jj in 0..m {
                sum += scores[i * m + jj] * v[jj * d_v + j];
            }
            mutated[i * d_v + j] = sum;
        }
    }
    let dist = common::l2_distance(&correct, &mutated);
    assert!(
        dist > 1e-4,
        "mutation not detected: wrong scaling too similar: dist = {dist}"
    );
}
