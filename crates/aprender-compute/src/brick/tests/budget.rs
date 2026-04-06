use super::super::*;

// ========================================================================
// ByteBudget Tests (F224 falsification)
// ========================================================================

#[test]
fn test_byte_budget_from_throughput() {
    let budget = ByteBudget::from_throughput(25.0); // 25 GB/s
    assert!((budget.gb_per_sec - 25.0).abs() < 0.001);
    // 25 GB/s = 6.1M pages/sec = 0.164 µs/page
    assert!((budget.us_per_page - 0.164).abs() < 0.01);
    assert_eq!(budget.page_size, 4096);
}

#[test]
fn test_byte_budget_from_latency() {
    let budget = ByteBudget::from_latency(0.164); // 0.164 µs/page
    assert!((budget.us_per_page - 0.164).abs() < 0.001);
    // Should be ~25 GB/s
    assert!((budget.gb_per_sec - 25.0).abs() < 1.0);
}

#[test]
fn test_byte_budget_to_token_budget() {
    let byte_budget = ByteBudget::from_throughput(25.0);
    let token_budget = byte_budget.to_token_budget();

    // us_per_token should equal us_per_page
    assert!((token_budget.us_per_token - byte_budget.us_per_page).abs() < 0.001);
    // tokens_per_sec should equal pages_per_sec
    let pages_per_sec = 25.0 * 1e9 / 4096.0;
    assert!((token_budget.tokens_per_sec - pages_per_sec).abs() < 1000.0);
}

#[test]
fn test_byte_budget_is_met() {
    let budget = ByteBudget::from_throughput(25.0); // ~0.164 µs/page
    assert!(budget.is_met(0.10)); // Faster than budget
    assert!(budget.is_met(budget.us_per_page)); // Exactly at budget
    assert!(!budget.is_met(0.20)); // Slower than budget
}

#[test]
fn test_byte_budget_with_page_size() {
    let budget = ByteBudget::from_throughput(25.0).with_page_size(65536); // 64KB pages
    assert_eq!(budget.page_size, 65536);
    // 25 GB/s with 64KB pages = 381K pages/sec = 2.62 µs/page
    assert!((budget.us_per_page - 2.62).abs() < 0.1);
}

#[test]
fn test_byte_budget_throughput_from_latency() {
    // 0.164 µs/page with 4KB pages should be ~25 GB/s
    let throughput = ByteBudget::throughput_from_latency(0.164, 4096);
    assert!((throughput - 25.0).abs() < 1.0);
}

// ========================================================================
// Additional Coverage Tests
// ========================================================================

#[test]
fn test_token_result_map() {
    let result = TokenResult {
        output: 42,
        tokens_processed: 10,
        us_per_token: 5.0,
        tokens_per_sec: 200_000.0,
        budget_met: true,
        budget_utilization: 0.5,
    };

    let mapped = result.map(|x| x * 2);
    assert_eq!(mapped.output, 84);
    assert_eq!(mapped.tokens_processed, 10);
    assert!((mapped.us_per_token - 5.0).abs() < 0.001);
    assert!(mapped.budget_met);
}

#[test]
fn test_compute_assertion_equiv_with_tolerance() {
    let assertion = ComputeAssertion::equiv_with_tolerance(Backend::Scalar, 1e-3);
    match assertion {
        ComputeAssertion::Equivalence { baseline, tolerance } => {
            assert_eq!(baseline, Backend::Scalar);
            assert!((tolerance - 1e-3).abs() < 1e-10);
        }
        _ => panic!("Expected Equivalence assertion"),
    }
}

#[test]
fn test_brick_verification_failures() {
    let brick = ComputeBrick::new(DotOp::new(4));
    let verification = brick.verify();

    // Should have one failure (no assertions)
    let failures: Vec<_> = verification.failures().collect();
    assert_eq!(failures.len(), 1);
    assert!(!failures[0].passed);
}

#[test]
fn test_dot_op_size_mismatch() {
    let op = DotOp::new(4);
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0]; // Wrong size
    let result = op.execute((a, b), Backend::Scalar);
    assert!(result.is_err());
}

#[test]
fn test_add_op_size_mismatch() {
    let op = AddOp::new(4);
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0]; // Wrong size
    let result = op.execute((a, b), Backend::Scalar);
    assert!(result.is_err());
}

#[test]
fn test_matmul_op_size_mismatch_a() {
    let op = MatmulOp::new(2, 2, 2);
    let a = vec![1.0, 2.0, 3.0]; // Wrong size (should be 4)
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let result = op.execute((a, b), Backend::Scalar);
    assert!(result.is_err());
}

#[test]
fn test_matmul_op_size_mismatch_b() {
    let op = MatmulOp::new(2, 2, 2);
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0]; // Wrong size (should be 4)
    let result = op.execute((a, b), Backend::Scalar);
    assert!(result.is_err());
}

#[test]
fn test_softmax_op_empty() {
    let op = SoftmaxOp::new(0);
    let result = op.execute(vec![], Backend::Scalar).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_compute_brick_builder_methods() {
    let brick = ComputeBrick::new(DotOp::new(100))
        .assert_equiv_with_tolerance(Backend::Avx2, 1e-3)
        .budget_us_per_tok(100.0)
        .enforce_budget(true);

    assert_eq!(brick.name(), "dot");
    assert_eq!(brick.get_backend(), Backend::Auto);
    assert!((brick.get_budget().us_per_token - 100.0).abs() < 0.001);
    assert_eq!(brick.get_assertions().len(), 1);
}

#[test]
fn test_compute_brick_budget_method() {
    let budget = TokenBudget::from_throughput(100_000.0).with_batch_size(32);
    let brick = ComputeBrick::new(DotOp::new(100)).budget(budget);

    assert_eq!(brick.get_budget().batch_size, 32);
    assert!((brick.get_budget().tokens_per_sec - 100_000.0).abs() < 1.0);
}

#[test]
fn test_compute_brick_enforce_budget_fail() {
    let brick = ComputeBrick::new(DotOp::new(1000000)) // Very large to take time
        .budget_tok_per_sec(1e15) // Impossibly high target
        .backend(Backend::Scalar)
        .enforce_budget(true);

    let a: Vec<f32> = (0..1000000).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..1000000).map(|i| i as f32).collect();
    let result = brick.run((a, b));

    // Should fail due to budget exceeded
    assert!(result.is_err());
    if let Err(BrickError::BudgetExceeded { .. }) = result {
        // Expected
    } else {
        panic!("Expected BudgetExceeded error");
    }
}

#[test]
fn test_compute_brick_clone() {
    let brick = ComputeBrick::new(DotOp::new(4))
        .assert_finite()
        .budget_tok_per_sec(50_000.0)
        .backend(Backend::Scalar);

    let cloned = brick.clone();
    assert_eq!(cloned.name(), brick.name());
    assert_eq!(cloned.get_backend(), brick.get_backend());
    assert_eq!(cloned.get_assertions().len(), brick.get_assertions().len());
}

#[test]
fn test_compute_brick_debug() {
    let brick = ComputeBrick::new(DotOp::new(4)).assert_finite().backend(Backend::Avx2);

    let debug_str = format!("{:?}", brick);
    assert!(debug_str.contains("ComputeBrick"));
    assert!(debug_str.contains("dot"));
    assert!(debug_str.contains("Avx2")); // Debug uses variant name, not Display
}

#[test]
fn test_brick_layer_with_named() {
    let layer = BrickLayer::new().with_named("attention", 10_000.0).with_named("ffn", 5_000.0);

    assert_eq!(layer.bricks().len(), 2);
    assert!((layer.throughput_ceiling() - 5_000.0).abs() < 1.0);
    assert_eq!(layer.bottleneck(), Some("ffn"));
}

#[test]
fn test_brick_layer_empty() {
    let layer = BrickLayer::new();
    assert_eq!(layer.throughput_ceiling(), f64::INFINITY);
    assert_eq!(layer.bottleneck(), None);
}

#[test]
fn test_backend_all_variants_display() {
    assert_eq!(format!("{}", Backend::Sse2), "SSE2");
    assert_eq!(format!("{}", Backend::Avx512), "AVX-512");
    assert_eq!(format!("{}", Backend::Neon), "NEON");
    assert_eq!(format!("{}", Backend::Wasm), "WASM");
    assert_eq!(format!("{}", Backend::Wgpu), "wgpu");
    assert_eq!(format!("{}", Backend::Auto), "Auto");
}

#[test]
fn test_byte_budget_default() {
    let budget = ByteBudget::default();
    assert!((budget.gb_per_sec - 25.0).abs() < 0.001);
}

#[test]
fn test_byte_budget_utilization() {
    let budget = ByteBudget::from_throughput(25.0);
    let util = budget.utilization(budget.us_per_page / 2.0); // 50% of budget
    assert!((util - 0.5).abs() < 0.01);
}

#[test]
fn test_token_budget_with_batch_size() {
    let budget = TokenBudget::from_latency(50.0).with_batch_size(64);
    assert_eq!(budget.batch_size, 64);
}

#[test]
fn test_token_budget_with_batch_size_min() {
    let budget = TokenBudget::from_latency(50.0).with_batch_size(0);
    assert_eq!(budget.batch_size, 1); // Should clamp to 1
}

#[test]
fn test_compute_brick_run_zero_tokens() {
    let brick = ComputeBrick::new(SoftmaxOp::new(0)).backend(Backend::Scalar);

    let result = brick.run(vec![]).unwrap();
    assert!(result.output.is_empty());
    // Edge case: zero tokens should still work
}

#[test]
fn test_brick_verification_is_valid() {
    let brick = ComputeBrick::new(DotOp::new(4)).assert_finite().assert_bounds(-1000.0, 1000.0);

    let verification = brick.verify();
    assert!(verification.is_valid());
}

#[test]
fn test_compute_assertion_bounds() {
    let assertion = ComputeAssertion::bounds(-10.0, 10.0);
    match assertion {
        ComputeAssertion::Bounds { min, max } => {
            assert!((-10.0 - min).abs() < 0.001);
            assert!((10.0 - max).abs() < 0.001);
        }
        _ => panic!("Expected Bounds assertion"),
    }
}

#[test]
fn test_compute_assertion_finite() {
    let assertion = ComputeAssertion::finite();
    assert!(matches!(assertion, ComputeAssertion::Finite));
}

#[test]
fn test_backend_default() {
    let backend = Backend::default();
    assert_eq!(backend, Backend::Avx2);
}
