// Part 1: Isolation Tests (21 tests)
/// Isolation: softmax through rmsnorm — softmax invariant (sum=1) violated
#[test]
fn isolation_softmax_through_rmsnorm() {
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let gamma = [1.0f32; 4];
    let mut softmax_out = [0.0f32; 4];
    softmax_scalar(&input, &mut softmax_out);
    let softmax_sum: f32 = softmax_out.iter().sum();
    assert!((softmax_sum - 1.0).abs() < 1e-5);

    let mut rms_out = [0.0f32; 4];
    rmsnorm_scalar(&input, &gamma, 1e-5, &mut rms_out);
    let rms_sum: f32 = rms_out.iter().sum();

    assert!(
        (rms_sum - 1.0).abs() > 0.01,
        "rmsnorm output unexpectedly sums to ~1.0: {rms_sum}"
    );
}

/// Isolation: rmsnorm through layernorm — rmsnorm output differs from layernorm output
#[test]
fn isolation_rmsnorm_through_layernorm() {
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let gamma = [1.0f32; 4];
    let beta = [0.0f32; 4];
    let mut rms_out = [0.0f32; 4];
    rmsnorm_scalar(&input, &gamma, 1e-5, &mut rms_out);
    let mut ln_out = [0.0f32; 4];
    layernorm_scalar(&input, &gamma, &beta, 1e-5, &mut ln_out);
    let rms_mean = common::mean(&rms_out);
    let ln_mean = common::mean(&ln_out);

    assert!(
        rms_mean.abs() > 0.1,
        "rmsnorm output unexpectedly has mean ~0: {rms_mean}"
    );
    assert!(
        ln_mean.abs() < 1e-4,
        "layernorm output should have mean ~0: {ln_mean}"
    );

    let dist = common::l2_distance(&rms_out, &ln_out);
    assert!(
        dist > 0.1,
        "rmsnorm and layernorm unexpectedly produce same output: dist = {dist}"
    );
}

/// Isolation: layernorm through softmax — layernorm invariant (mean=0) violated
#[test]
fn isolation_layernorm_through_softmax() {
    let input = [1.0f32, 2.0, 3.0, 4.0];

    let gamma = [1.0f32; 4];
    let beta = [0.0f32; 4];
    let mut ln_out = [0.0f32; 4];
    layernorm_scalar(&input, &gamma, &beta, 1e-5, &mut ln_out);
    let ln_mean = common::mean(&ln_out);
    assert!(ln_mean.abs() < 1e-4);

    let mut sm_out = [0.0f32; 4];
    softmax_scalar(&input, &mut sm_out);
    let sm_mean = common::mean(&sm_out);

    assert!(
        sm_mean.abs() > 0.01,
        "softmax output unexpectedly has mean ~0: {sm_mean}"
    );
}

/// Isolation: relu through silu — relu invariant (non-negative) violated
#[test]
fn isolation_relu_through_silu() {
    let input = [-2.0f32, -1.0, -0.5, -3.0];

    let mut relu_out = [0.0f32; 4];
    relu_scalar(&input, &mut relu_out);
    for &y in &relu_out {
        assert!(y >= 0.0);
    }

    let mut silu_out = [0.0f32; 4];
    silu_scalar(&input, &mut silu_out);
    let has_negative = silu_out.iter().any(|&y| y < 0.0);
    assert!(
        has_negative,
        "silu output unexpectedly has no negative values: {silu_out:?}"
    );
}

/// Isolation: softmax through matmul — softmax invariant (sum=1) violated
#[test]
fn isolation_softmax_through_attention() {
    let input = [1.0f32, 2.0, 3.0, 4.0];

    let mut sm_out = [0.0f32; 4];
    softmax_scalar(&input, &mut sm_out);
    let sm_sum: f32 = sm_out.iter().sum();
    assert!((sm_sum - 1.0).abs() < 1e-5);

    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [1.0f32, 0.0, 0.0, 1.0];
    let mut c = [0.0f32; 4];
    matmul_scalar(&a, &b, 2, 2, 2, &mut c);
    let matmul_sum: f32 = c.iter().sum();

    assert!(
        (matmul_sum - 1.0).abs() > 0.1,
        "matmul output unexpectedly sums to ~1.0: {matmul_sum}"
    );
}

/// Isolation: matmul through softmax — matmul result changes
#[test]
fn isolation_matmul_through_softmax() {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [2.0f32, 0.0, 1.0, 2.0];
    let mut ab = [0.0f32; 4];
    matmul_scalar(&a, &b, 2, 2, 2, &mut ab);

    let mut sm_out = [0.0f32; 4];
    softmax_scalar(&ab, &mut sm_out);

    let dist = common::l2_distance(&ab, &sm_out);
    assert!(
        dist > 1.0,
        "softmax of matmul result unexpectedly close to matmul result: dist = {dist}"
    );
}

/// Isolation: attention through softmax — attention output is not idempotent under softmax
#[test]
fn isolation_attention_through_softmax() {
    let q = [1.0f32, 0.0, 0.0, 1.0];
    let k = [1.0f32, 0.0, 0.0, 1.0];
    let v = [1.0f32, 2.0, 3.0, 4.0];
    let mut attn_out = [0.0f32; 4];
    attention_scalar(&q, &k, &v, 2, 2, 2, 2, &mut attn_out);

    let mut sm_out = [0.0f32; 4];
    softmax_scalar(&attn_out, &mut sm_out);

    let dist = common::l2_distance(&attn_out, &sm_out);
    assert!(
        dist > 0.01,
        "softmax(attention_output) unexpectedly equals attention_output: dist = {dist}"
    );
}

/// Isolation: gqa through matmul — different results
#[test]
fn isolation_gqa_through_matmul() {
    let q = [1.0f32, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5];
    let k = [1.0f32, 0.0, 0.0, 1.0];
    let v = [1.0f32, 2.0, 3.0, 4.0];
    let mut gqa_out = [0.0f32; 8];
    gqa_scalar(&q, &k, &v, 2, 2, 2, 2, 1, &mut gqa_out);

    let mut mm_out = [0.0f32; 8];
    matmul_scalar(&q, &k, 4, 2, 2, &mut mm_out);

    let dist = common::l2_distance(&gqa_out, &mm_out);
    assert!(
        dist > 0.01,
        "GQA output unexpectedly matches matmul output: dist = {dist}"
    );
}

/// Isolation: `flash_attention` vs attention — `tile_size` can affect numerical path
#[test]
fn isolation_flash_through_attention() {
    let q = [1.0f32, 0.5, 0.3, 0.7, 0.2, 0.9, 0.8, 0.1];
    let k = [0.5f32, 0.3, 0.7, 0.2, 0.9, 0.1, 0.4, 0.6];
    let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let mut flash_out = [0.0f32; 8];
    flash_attention_scalar(&q, &k, &v, 4, 2, 2, &mut flash_out);

    let mut attn_out = [0.0f32; 8];
    attention_scalar(&q, &k, &v, 4, 4, 2, 2, &mut attn_out);

    common::assert_all_finite(&flash_out);
    common::assert_all_finite(&attn_out);

    let mut attn_of_flash = [0.0f32; 8];
    attention_scalar(
        &flash_out,
        &flash_out,
        &flash_out,
        4,
        4,
        2,
        2,
        &mut attn_of_flash,
    );
    let dist = common::l2_distance(&flash_out, &attn_of_flash);
    assert!(
        dist > 0.01,
        "attention(flash_output) unexpectedly equals flash_output: dist = {dist}"
    );
}

/// Isolation: silu through relu — relu clips negative parts, changing result
#[test]
fn isolation_silu_through_relu() {
    let input = [-2.0f32, -1.0, 0.5, 2.0];

    let mut silu_out = [0.0f32; 4];
    silu_scalar(&input, &mut silu_out);

    let mut relu_of_silu = [0.0f32; 4];
    relu_scalar(&silu_out, &mut relu_of_silu);

    let dist = common::l2_distance(&silu_out, &relu_of_silu);
    assert!(
        dist > 0.01,
        "relu(silu(x)) unexpectedly equals silu(x): dist = {dist}"
    );
}

/// Isolation: swiglu vs just silu — gate matters
#[test]
fn isolation_swiglu_through_silu() {
    let gate = [1.0f32, -1.0, 2.0, -2.0];
    let value = [2.0f32, 3.0, 0.5, 1.0];

    let mut swiglu_out = [0.0f32; 4];
    swiglu_scalar(&gate, &value, &mut swiglu_out);

    let mut silu_out = [0.0f32; 4];
    silu_scalar(&gate, &mut silu_out);

    let dist = common::l2_distance(&swiglu_out, &silu_out);
    assert!(
        dist > 0.01,
        "swiglu output unexpectedly equals silu(gate): dist = {dist}"
    );
}

/// Isolation: softmax vs `log_softmax` — different functions
#[test]
fn isolation_cross_entropy_through_softmax() {
    let logits = [1.0f32, 2.0, 3.0, 4.0];

    let mut sm_out = [0.0f32; 4];
    softmax_scalar(&logits, &mut sm_out);

    let mut log_sm_out = [0.0f32; 4];
    log_softmax_scalar(&logits, &mut log_sm_out);

    let dist = common::l2_distance(&sm_out, &log_sm_out);
    assert!(
        dist > 0.1,
        "softmax and log_softmax unexpectedly produce same output: dist = {dist}"
    );
}

/// Isolation: rope through layernorm — layernorm changes the embedding
#[test]
fn isolation_rope_through_layernorm() {
    let input = [1.0f32, 2.0, 3.0, 4.0];

    let mut rope_out = [0.0f32; 4];
    rope_scalar(&input, 5, 4, 10000.0, &mut rope_out);

    let gamma = [1.0f32; 4];
    let beta = [0.0f32; 4];
    let mut ln_of_rope = [0.0f32; 4];
    layernorm_scalar(&rope_out, &gamma, &beta, 1e-5, &mut ln_of_rope);

    let dist = common::l2_distance(&rope_out, &ln_of_rope);
    assert!(
        dist > 0.01,
        "layernorm(rope(x)) unexpectedly equals rope(x): dist = {dist}"
    );
}

/// Isolation: adamw vs lbfgs — different optimization paths
#[test]
fn isolation_adamw_through_lbfgs() {
    let gradient = [1.0f32, -0.5, 0.3, -0.7];
    let d = 4;

    let mut params = [1.0f32, 2.0, 3.0, 4.0];
    let mut m_buf = [0.0f32; 4];
    let mut v_buf = [0.0f32; 4];
    adamw_step_scalar(
        &mut params,
        &gradient,
        &mut m_buf,
        &mut v_buf,
        0.001,
        0.9,
        0.999,
        1e-8,
        0.01,
        1,
    );
    let adamw_delta: Vec<f32> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .zip(params.iter())
        .map(|(a, b)| b - a)
        .collect();

    let mut direction = [0.0f32; 4];
    lbfgs_direction_scalar(&gradient, &[], &[], 0, d, &mut direction);

    let dist = common::l2_distance(&adamw_delta, &direction);
    assert!(
        dist > 0.01,
        "adamw update direction unexpectedly matches lbfgs direction: dist = {dist}"
    );
}

/// Isolation: conv1d vs matmul — different operations
#[test]
fn isolation_conv1d_through_matmul() {
    let input = [1.0f32, 2.0, 3.0, 4.0];
    let weight = [1.0f32, 1.0];
    let out_len = (4 - 2) + 1;
    let mut conv_out = vec![0.0f32; out_len];
    conv1d_scalar(&input, &weight, None, 1, 1, 4, 2, 1, 0, &mut conv_out);

    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [1.0f32, 1.0, 1.0, 1.0];
    let mut mm_out = [0.0f32; 4];
    matmul_scalar(&a, &b, 2, 2, 2, &mut mm_out);

    assert_ne!(conv_out.len(), mm_out.len());
}

/// Isolation: ssm vs conv1d — different operations on same data
#[test]
fn isolation_ssm_through_conv1d() {
    let seq_len = 4;
    let state_dim = 2;
    let a_bar = [0.9f32, 0.8];
    let b_bar = [1.0f32, 0.5, 0.8, 0.3, 0.6, 0.7, 0.4, 0.9];
    let c = [1.0f32, 1.0];
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let mut ssm_out = [0.0f32; 4];
    ssm_scan_scalar(&a_bar, &b_bar, &c, &x, state_dim, seq_len, &mut ssm_out);

    let weight = [1.0f32, 1.0];
    let conv_out_len = (4 - 2) + 1;
    let mut conv_out = vec![0.0f32; conv_out_len];
    conv1d_scalar(&x, &weight, None, 1, 1, 4, 2, 1, 0, &mut conv_out);

    assert_ne!(ssm_out.len(), conv_out.len());
    assert!(
        (ssm_out[0] - conv_out[0]).abs() > 0.01,
        "ssm and conv1d unexpectedly produce same first value"
    );
}

/// Isolation: kmeans assignments vs pagerank — different problem domains
#[test]
fn isolation_kmeans_through_pagerank() {
    let points = [0.0f32, 0.0, 1.0, 0.0, 10.0, 10.0, 11.0, 10.0];
    let centroids = [0.5f32, 0.0, 10.5, 10.0];
    let mut assignments = [0u32; 4];
    kmeans_assign_scalar(&points, &centroids, 4, 2, 2, &mut assignments);

    let transition = common::stochastic_transition_matrix(4);
    let rank = [0.25f32; 4];
    let mut pr_out = [0.0f32; 4];
    pagerank_iterate_scalar(&transition, &rank, 4, 0.85, &mut pr_out);

    assert_eq!(assignments[0], 0);
    assert_eq!(assignments[2], 1);
    assert!(pr_out.iter().all(|&x| x > 0.0 && x < 1.0));
}

/// Isolation: pagerank through softmax — softmax changes the distribution
#[test]
fn isolation_pagerank_through_softmax() {
    let n = 4;
    let rank = [0.4f32, 0.3, 0.2, 0.1];
    let transition = [
        0.1f32, 0.2, 0.3, 0.4, 0.4, 0.1, 0.2, 0.3, 0.3, 0.4, 0.1, 0.2, 0.2, 0.3, 0.4, 0.1,
    ];
    let mut pr_out = [0.0f32; 4];
    pagerank_iterate_scalar(&transition, &rank, n, 0.85, &mut pr_out);

    let mut sm_of_pr = [0.0f32; 4];
    softmax_scalar(&pr_out, &mut sm_of_pr);

    let dist = common::l2_distance(&pr_out, &sm_of_pr);
    assert!(
        dist > 0.001,
        "softmax(pagerank) unexpectedly equals pagerank: dist = {dist}"
    );
}

/// Isolation: lbfgs direction applied as adamw step — different result
#[test]
fn isolation_lbfgs_through_adamw() {
    let gradient = [1.0f32, -0.5, 0.3, -0.7];

    let mut direction = [0.0f32; 4];
    lbfgs_direction_scalar(&gradient, &[], &[], 0, 4, &mut direction);

    let mut params = [0.0f32; 4];
    let mut m_buf = [0.0f32; 4];
    let mut v_buf = [0.0f32; 4];
    adamw_step_scalar(
        &mut params,
        &direction,
        &mut m_buf,
        &mut v_buf,
        0.001,
        0.9,
        0.999,
        1e-8,
        0.0,
        1,
    );

    let dist = common::l2_distance(&params, &direction);
    assert!(
        dist > 0.01,
        "adamw(lbfgs_direction) unexpectedly equals lbfgs_direction: dist = {dist}"
    );
}

/// Isolation: cma sample through rmsnorm — changes the sample
#[test]
fn isolation_cma_through_rmsnorm() {
    let mean = [1.0f32, 2.0, 3.0, 4.0];
    let sigma = 0.5;
    let cholesky_l = [
        2.0f32, 0.0, 0.0, 0.0, 0.5, 1.5, 0.0, 0.0, 0.3, 0.2, 1.0, 0.0, 0.1, 0.4, 0.3, 0.8,
    ];
    let z = [0.5f32, -0.3, 0.7, -0.1];
    let mut cma_out = [0.0f32; 4];
    cma_sample_scalar(&mean, sigma, &cholesky_l, 4, &z, &mut cma_out);

    let gamma = [1.0f32; 4];
    let mut rms_of_cma = [0.0f32; 4];
    rmsnorm_scalar(&cma_out, &gamma, 1e-5, &mut rms_of_cma);

    let dist = common::l2_distance(&cma_out, &rms_of_cma);
    assert!(
        dist > 0.01,
        "rmsnorm(cma_sample) unexpectedly equals cma_sample: dist = {dist}"
    );
}

/// Isolation: gdn output vs ssm output — different recurrences
#[test]
fn isolation_gdn_through_ssm() {
    let seq_len = 4;
    let k_dim = 2;
    let v_dim = 2;

    let q = [1.0f32, 0.5, 0.3, 0.7, 0.2, 0.9, 0.8, 0.1];
    let k = [0.5f32, 0.3, 0.7, 0.2, 0.9, 0.1, 0.4, 0.6];
    let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let alpha = [0.9f32, 0.8, 0.7, 0.6];
    let beta = [0.1f32, 0.2, 0.3, 0.4];
    let mut gdn_out = [0.0f32; 8];
    gdn_recurrence_scalar(
        &q,
        &k,
        &v,
        &alpha,
        &beta,
        seq_len,
        k_dim,
        v_dim,
        &mut gdn_out,
    );

    let state_dim = 2;
    let a_bar = [0.9f32, 0.8];
    let b_bar = [0.5f32, 0.3, 0.7, 0.2, 0.9, 0.1, 0.4, 0.6];
    let c_ssm = [1.0f32, 1.0];
    let x_ssm = [1.0f32, 3.0, 5.0, 7.0];
    let mut ssm_out = [0.0f32; 4];
    ssm_scan_scalar(
        &a_bar,
        &b_bar,
        &c_ssm,
        &x_ssm,
        state_dim,
        seq_len,
        &mut ssm_out,
    );

    assert_ne!(gdn_out.len(), ssm_out.len());
    let dist = common::l2_distance(&gdn_out[..4], &ssm_out);
    assert!(
        dist > 0.01,
        "gdn and ssm first 4 outputs unexpectedly match: dist = {dist}"
    );
}
