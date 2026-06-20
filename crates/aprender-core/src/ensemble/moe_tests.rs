pub(crate) use super::*;
pub(crate) use crate::ensemble::gating::SoftmaxGating;
pub(crate) use crate::linear_model::LinearRegression;

#[test]
fn test_moe_config_default() {
    let config = MoeConfig::default();
    assert_eq!(config.top_k, 1);
    assert!((config.capacity_factor - 1.0).abs() < 1e-6);
    assert!((config.expert_dropout - 0.0).abs() < 1e-6);
    assert!((config.load_balance_weight - 0.01).abs() < 1e-6);
}

#[test]
fn test_moe_config_builders() {
    let config = MoeConfig::default()
        .with_top_k(3)
        .with_capacity_factor(1.5)
        .with_expert_dropout(0.1)
        .with_load_balance_weight(0.05);

    assert_eq!(config.top_k, 3);
    assert!((config.capacity_factor - 1.5).abs() < 1e-6);
    assert!((config.expert_dropout - 0.1).abs() < 1e-6);
    assert!((config.load_balance_weight - 0.05).abs() < 1e-6);
}

#[test]
fn test_moe_config_clone() {
    let config = MoeConfig::default().with_top_k(2);
    let cloned = config.clone();
    assert_eq!(cloned.top_k, config.top_k);
}

#[test]
fn test_moe_config_debug() {
    let config = MoeConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("MoeConfig"));
}

#[test]
fn test_moe_builder_without_gating() {
    let expert = LinearRegression::new();
    let result = MixtureOfExperts::<LinearRegression, SoftmaxGating>::builder()
        .expert(expert)
        .build();
    assert!(result.is_err());
}

#[test]
fn test_moe_builder_basic() {
    let mut expert1 = LinearRegression::new();
    let mut expert2 = LinearRegression::new();

    // Use more samples than features for valid OLS
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("valid matrix");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    expert1.fit(&x, &y).expect("fit expert1");
    expert2.fit(&x, &y).expect("fit expert2");

    let gating = SoftmaxGating::new(1, 2);
    let moe = MixtureOfExperts::builder()
        .expert(expert1)
        .expert(expert2)
        .gating(gating)
        .build()
        .expect("build moe");

    assert_eq!(moe.n_experts(), 2);
}

#[test]
fn test_moe_predict() {
    let mut expert1 = LinearRegression::new();
    let mut expert2 = LinearRegression::new();

    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("valid matrix");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    expert1.fit(&x, &y).expect("fit expert1");
    expert2.fit(&x, &y).expect("fit expert2");

    let gating = SoftmaxGating::new(1, 2);
    let moe = MixtureOfExperts::builder()
        .expert(expert1)
        .expert(expert2)
        .gating(gating)
        .build()
        .expect("build moe");

    let input = vec![3.0];
    let prediction = moe.predict(&input);
    // Should produce some output
    assert!(prediction.is_finite());
}

#[test]
fn test_moe_predict_batch() {
    let mut expert = LinearRegression::new();
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("valid matrix");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    expert.fit(&x, &y).expect("fit expert");

    let gating = SoftmaxGating::new(1, 1);
    let moe = MixtureOfExperts::builder()
        .expert(expert)
        .gating(gating)
        .build()
        .expect("build moe");

    let inputs = Matrix::from_vec(2, 1, vec![1.0, 2.0]).expect("valid inputs");
    let predictions = moe.predict_batch(&inputs);
    assert_eq!(predictions.len(), 2);
}

#[test]
fn test_moe_config_getter() {
    let mut expert = LinearRegression::new();
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("valid matrix");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    expert.fit(&x, &y).expect("fit");

    let config = MoeConfig::default().with_top_k(2);
    let gating = SoftmaxGating::new(1, 1);
    let moe = MixtureOfExperts::builder()
        .expert(expert)
        .gating(gating)
        .config(config)
        .build()
        .expect("build moe");

    assert_eq!(moe.config().top_k, 2);
}

#[test]
fn test_moe_get_routing_weights() {
    let mut expert = LinearRegression::new();
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("valid matrix");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    expert.fit(&x, &y).expect("fit");

    let gating = SoftmaxGating::new(1, 1);
    let moe = MixtureOfExperts::builder()
        .expert(expert)
        .gating(gating)
        .build()
        .expect("build moe");

    let weights = moe.get_routing_weights(&[1.0]);
    assert_eq!(weights.len(), 1); // 1 expert
}

#[test]
fn test_moe_load_balance_loss_empty() {
    let mut expert = LinearRegression::new();
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("valid matrix");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    expert.fit(&x, &y).expect("fit");

    let gating = SoftmaxGating::new(1, 1);
    let moe = MixtureOfExperts::builder()
        .expert(expert)
        .gating(gating)
        .build()
        .expect("build moe");

    // Empty inputs
    let inputs = Matrix::from_vec(0, 1, vec![]).expect("empty matrix");
    let loss = moe.compute_load_balance_loss(&inputs);
    assert!((loss - 0.0).abs() < 1e-6);
}

#[test]
fn test_moe_expert_usage() {
    let mut expert = LinearRegression::new();
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("valid matrix");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    expert.fit(&x, &y).expect("fit");

    let gating = SoftmaxGating::new(1, 1);
    let moe = MixtureOfExperts::builder()
        .expert(expert)
        .gating(gating)
        .build()
        .expect("build moe");

    let usage = moe.expert_usage(&x);
    assert_eq!(usage.len(), 1); // 1 expert
}

#[test]
fn test_moe_expert_usage_empty() {
    let mut expert = LinearRegression::new();
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("valid matrix");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    expert.fit(&x, &y).expect("fit");

    let gating = SoftmaxGating::new(1, 1);
    let moe = MixtureOfExperts::builder()
        .expert(expert)
        .gating(gating)
        .build()
        .expect("build moe");

    // Empty inputs
    let empty = Matrix::from_vec(0, 1, vec![]).expect("empty matrix");
    let usage = moe.expert_usage(&empty);
    assert_eq!(usage.len(), 1);
    assert!((usage[0] - 0.0).abs() < 1e-6);
}

#[test]
fn test_moe_fit() {
    let mut expert = LinearRegression::new();
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("valid matrix");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    expert.fit(&x, &y).expect("fit");

    let gating = SoftmaxGating::new(1, 1);
    let mut moe = MixtureOfExperts::builder()
        .expert(expert)
        .gating(gating)
        .build()
        .expect("build moe");

    // Fit does nothing in simple implementation
    assert!(moe.fit(&x, &y).is_ok());
}

#[test]
fn test_moe_debug() {
    let mut expert = LinearRegression::new();
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("valid matrix");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    expert.fit(&x, &y).expect("fit");

    let gating = SoftmaxGating::new(1, 1);
    let moe = MixtureOfExperts::builder()
        .expert(expert)
        .gating(gating)
        .build()
        .expect("build moe");

    let debug_str = format!("{:?}", moe);
    assert!(debug_str.contains("MixtureOfExperts"));
}

#[test]
fn test_moe_builder_debug() {
    let builder = MixtureOfExperts::<LinearRegression, SoftmaxGating>::builder();
    let debug_str = format!("{:?}", builder);
    assert!(debug_str.contains("MoeBuilder"));
}

// ---------------------------------------------------------------------------
// PMAT-875: Switch Transformer load-balance aux loss must use the FULL router
// softmax for P_i (mean over ALL tokens, every expert), NOT the top-k-only
// gate probability. Reference: Fedus et al. 2021 Eq. 4-6; HF Mixtral
// `load_balancing_loss_func`.
// ---------------------------------------------------------------------------

/// Build a 4-expert / 3-feature MoE with non-trivial differentiated routing.
fn build_falsifier_moe(
    top_k: usize,
    alpha: f32,
) -> MixtureOfExperts<LinearRegression, SoftmaxGating> {
    let n_experts = 4usize;
    let n_features = 3usize;

    // Fit identical (trivial) experts; the load-balance loss only depends on
    // the router, not on expert outputs.
    let xtr = Matrix::from_vec(
        5,
        n_features,
        vec![
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
        ],
    )
    .expect("valid train matrix");
    let ytr = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);

    let mut builder = MixtureOfExperts::builder();
    for _ in 0..n_experts {
        let mut e = LinearRegression::new();
        e.fit(&xtr, &ytr).expect("fit expert");
        builder = builder.expert(e);
    }

    let config = MoeConfig::default()
        .with_top_k(top_k)
        .with_load_balance_weight(alpha);
    builder
        .gating(SoftmaxGating::new(n_features, n_experts))
        .config(config)
        .build()
        .expect("build moe")
}

/// Reference Switch-Transformer aux loss.
///
/// `full_softmax_p`: if true, accumulate P_i over the FULL softmax for every
/// expert on every token (CORRECT). If false, accumulate P_i only over the
/// top-k routed experts (the OLD BUGGY behaviour).
fn reference_aux_loss(
    moe: &MixtureOfExperts<LinearRegression, SoftmaxGating>,
    inputs: &Matrix<f32>,
    full_softmax_p: bool,
) -> f32 {
    let n_samples = inputs.n_rows();
    let n_experts = moe.n_experts();
    if n_samples == 0 || n_experts == 0 {
        return 0.0;
    }
    let top_k = moe.config().top_k.min(n_experts);

    let mut counts = vec![0.0f32; n_experts];
    let mut probs = vec![0.0f32; n_experts];

    for i in 0..n_samples {
        let row = inputs.row(i);
        let weights = moe.get_routing_weights(row.as_slice());

        let mut indexed: Vec<(usize, f32)> = weights.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // f_i: hard top-k dispatch.
        for (idx, _w) in indexed.iter().take(top_k) {
            counts[*idx] += 1.0;
        }

        // P_i: full softmax (correct) or top-k only (buggy).
        if full_softmax_p {
            for (e, &w) in weights.iter().enumerate() {
                probs[e] += w;
            }
        } else {
            for (idx, w) in indexed.iter().take(top_k) {
                probs[*idx] += w;
            }
        }
    }

    let n_tokens = (n_samples * top_k) as f32;
    let mut loss = 0.0f32;
    for e in 0..n_experts {
        let f_i = counts[e] / n_tokens.max(1.0);
        let p_i = probs[e] / n_samples as f32;
        loss += f_i * p_i;
    }
    loss * n_experts as f32 * moe.config().load_balance_weight
}

#[test]
fn test_moe_load_balance_uses_full_softmax_p_topk1() {
    let alpha = 0.01f32;
    let moe = build_falsifier_moe(1, alpha);

    // The gate routes by sign of sum(x): sum(x) > 0 picks the highest-index
    // expert, sum(x) < 0 the lowest. Mixing signs gives several experts a
    // nonzero dispatch fraction f_i, AND each picks up softmax mass on tokens
    // routed elsewhere — exactly the regime where top-k-only P_i and
    // full-softmax P_i diverge. Rows (by sum sign): +, -, +, -
    //   -> top-1 routes to experts 3, 0, 3, 0 respectively.
    let inputs = Matrix::from_vec(
        4,
        3,
        vec![
            3.0, 1.0, 1.0, -3.0, -1.0, -1.0, 2.0, 2.0, 2.0, -4.0, -1.0, -1.0,
        ],
    )
    .expect("valid inputs");

    let actual = moe.compute_load_balance_loss(&inputs);
    let expected_full = reference_aux_loss(&moe, &inputs, true);
    let expected_topk_only = reference_aux_loss(&moe, &inputs, false);

    // GREEN: matches the full-softmax Switch aux loss.
    assert!(
        (actual - expected_full).abs() < 1e-6,
        "load balance loss must equal full-softmax reference: actual={actual}, expected_full={expected_full}"
    );

    // The two references must genuinely differ, else the test cannot falsify
    // the bug.
    assert!(
        (expected_full - expected_topk_only).abs() > 1e-5,
        "test setup invalid: full-softmax and top-k-only references coincide \
         (full={expected_full}, topk_only={expected_topk_only})"
    );

    // RED guard: the (now fixed) code must NOT reproduce the top-k-only value.
    assert!(
        (actual - expected_topk_only).abs() > 1e-5,
        "load balance loss still equals the buggy top-k-only value: \
         actual={actual}, topk_only={expected_topk_only}"
    );
}

#[test]
fn test_moe_load_balance_uses_full_softmax_p_topk2() {
    let alpha = 0.02f32;
    let moe = build_falsifier_moe(2, alpha);

    // sum(x) > 0 -> top-2 = {expert 3, expert 2}; sum(x) < 0 -> {expert 0,
    // expert 1}. Mixed signs (rows: +, -, +, -, +) dispatch all four experts
    // and each carries softmax mass on tokens routed elsewhere.
    let inputs = Matrix::from_vec(
        5,
        3,
        vec![
            4.0, 1.0, 1.0, -4.0, -1.0, -1.0, 3.0, 2.0, 1.0, -3.0, -2.0, -1.0, 5.0, 1.0, 1.0,
        ],
    )
    .expect("valid inputs");

    let actual = moe.compute_load_balance_loss(&inputs);
    let expected_full = reference_aux_loss(&moe, &inputs, true);
    let expected_topk_only = reference_aux_loss(&moe, &inputs, false);

    assert!(
        (actual - expected_full).abs() < 1e-6,
        "top_k=2 load balance loss must equal full-softmax reference: \
         actual={actual}, expected_full={expected_full}"
    );
    assert!(
        (expected_full - expected_topk_only).abs() > 1e-5,
        "test setup invalid: references coincide for top_k=2"
    );
    assert!(
        (actual - expected_topk_only).abs() > 1e-5,
        "top_k=2 loss still equals the buggy top-k-only value"
    );
}

#[test]
fn test_moe_load_balance_minimized_under_uniform_routing() {
    // Under perfectly uniform routing (f_i = 1/N for all i, P_i = 1/N for all
    // i), the Switch aux loss equals alpha:
    //   loss = alpha * N * sum_i (1/N)(1/N) = alpha * N * N * (1/N^2) = alpha.
    // We approximate uniform routing with a near-zero input row, so every
    // logit is ~0 and the softmax is ~uniform. With top_k = N, every expert is
    // dispatched once per token, giving f_i = 1/N exactly.
    let alpha = 0.05f32;
    let n_experts = 4usize;
    let moe = build_falsifier_moe(n_experts, alpha); // top_k = n_experts

    // A single near-zero row -> uniform softmax, every expert in the top-k.
    let inputs = Matrix::from_vec(1, 3, vec![0.0, 0.0, 0.0]).expect("valid inputs");

    let loss = moe.compute_load_balance_loss(&inputs);

    // Loss is minimized at exactly alpha under uniform routing.
    assert!(
        (loss - alpha).abs() < 1e-5,
        "uniform-routing aux loss must equal alpha={alpha}, got {loss}"
    );
}
