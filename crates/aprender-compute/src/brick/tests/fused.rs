use super::super::*;

// ========================================================================
// Fused LLM Operations Tests (PMAT-PERF-009)
// ========================================================================

#[test]
fn test_fused_qkv_op_new() {
    // Qwen 3B dimensions: hidden=3584, heads=28, kv_heads=4 (GQA)
    let op = FusedQKVOp::new(3584, 28, 4);
    assert_eq!(op.hidden_size, 3584);
    assert_eq!(op.num_heads, 28);
    assert_eq!(op.head_dim, 128); // 3584 / 28
    assert_eq!(op.kv_dim, 512); // 4 * 128
}

#[test]
fn test_fused_qkv_op_name() {
    let op = FusedQKVOp::new(1024, 8, 8);
    assert_eq!(op.name(), "fused_qkv");
}

#[test]
fn test_fused_qkv_op_execute_small() {
    let hidden_size = 4;
    let num_heads = 2;
    let num_kv_heads = 2;
    let head_dim = hidden_size / num_heads; // 2
    let kv_dim = num_kv_heads * head_dim; // 4

    let op = FusedQKVOp::new(hidden_size, num_heads, num_kv_heads);

    // Identity-like weights for testing
    let q_weight =
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let k_weight = q_weight.clone();
    let v_weight = q_weight.clone();

    let weights = FusedQKVWeights { q_weight, k_weight, v_weight };

    let x = vec![1.0, 2.0, 3.0, 4.0];
    let (q, k, v) = op.execute((x.clone(), weights), Backend::Scalar).unwrap();

    // With identity weights, output should equal input
    assert_eq!(q, x);
    assert_eq!(k.len(), kv_dim);
    assert_eq!(v.len(), kv_dim);
}

#[test]
fn test_fused_qkv_op_size_mismatch() {
    let op = FusedQKVOp::new(4, 2, 2);
    let weights = FusedQKVWeights {
        q_weight: vec![0.0; 16],
        k_weight: vec![0.0; 16],
        v_weight: vec![0.0; 16],
    };
    let x = vec![1.0, 2.0, 3.0]; // Wrong size (should be 4)

    let result = op.execute((x, weights), Backend::Scalar);
    assert!(result.is_err());
}

#[test]
fn test_fused_qkv_op_tokens() {
    // hidden=1024, kv_dim=256 (GQA with 4 heads, 2 kv_heads)
    let op = FusedQKVOp::new(1024, 4, 2);
    let weights = FusedQKVWeights { q_weight: vec![], k_weight: vec![], v_weight: vec![] };
    let tokens = op.tokens(&(vec![], weights));
    // Q (1024) + K (512) + V (512) = 2048
    assert_eq!(tokens, 1024 + 512 + 512);
}

#[test]
fn test_fused_gate_up_op_new() {
    // Qwen 3B dimensions
    let op = FusedGateUpOp::new(3584, 18944);
    assert_eq!(op.hidden_size, 3584);
    assert_eq!(op.intermediate_size, 18944);
}

#[test]
fn test_fused_gate_up_op_name() {
    let op = FusedGateUpOp::new(1024, 4096);
    assert_eq!(op.name(), "fused_gate_up");
}

#[test]
fn test_fused_gate_up_op_silu() {
    // SiLU(0) = 0 / (1 + 1) = 0
    assert!((FusedGateUpOp::silu(0.0)).abs() < 1e-6);
    // SiLU(x) for large x approaches x
    let large = FusedGateUpOp::silu(10.0);
    assert!((large - 10.0).abs() < 0.01);
}

#[test]
fn test_fused_gate_up_op_execute_small() {
    let hidden_size = 2;
    let intermediate_size = 3;

    let op = FusedGateUpOp::new(hidden_size, intermediate_size);

    // Simple weights
    let gate_weight = vec![
        1.0, 0.0, // intermediate[0] = x[0]
        0.0, 1.0, // intermediate[1] = x[1]
        1.0, 1.0, // intermediate[2] = x[0] + x[1]
    ];
    let up_weight = vec![
        1.0, 0.0, // up[0] = x[0]
        0.0, 1.0, // up[1] = x[1]
        0.5, 0.5, // up[2] = 0.5 * (x[0] + x[1])
    ];

    let weights = FusedGateUpWeights { gate_weight, up_weight };

    let x = vec![2.0, 3.0];
    let output = op.execute((x, weights), Backend::Scalar).unwrap();

    assert_eq!(output.len(), intermediate_size);
    // output[0] = SiLU(2.0) * 2.0
    // output[1] = SiLU(3.0) * 3.0
    // output[2] = SiLU(5.0) * 2.5
    assert!(output[0] > 0.0);
    assert!(output[1] > 0.0);
    assert!(output[2] > 0.0);
}

#[test]
fn test_fused_gate_up_op_size_mismatch() {
    let op = FusedGateUpOp::new(4, 8);
    let weights = FusedGateUpWeights { gate_weight: vec![0.0; 32], up_weight: vec![0.0; 32] };
    let x = vec![1.0, 2.0, 3.0]; // Wrong size (should be 4)

    let result = op.execute((x, weights), Backend::Scalar);
    assert!(result.is_err());
}

#[test]
fn test_fused_gate_up_op_tokens() {
    let op = FusedGateUpOp::new(1024, 4096);
    let weights = FusedGateUpWeights { gate_weight: vec![], up_weight: vec![] };
    let tokens = op.tokens(&(vec![], weights));
    assert_eq!(tokens, 4096);
}

#[test]
fn test_fused_qkv_compute_brick() {
    let op = FusedQKVOp::new(4, 2, 2);
    let brick = ComputeBrick::new(op)
        .assert_finite()
        .budget_tok_per_sec(1_000_000.0)
        .backend(Backend::Scalar);

    assert_eq!(brick.name(), "fused_qkv");
    let verification = brick.verify();
    assert!(verification.is_valid());
}

#[test]
fn test_fused_gate_up_compute_brick() {
    let op = FusedGateUpOp::new(4, 8);
    let brick = ComputeBrick::new(op)
        .assert_finite()
        .budget_tok_per_sec(1_000_000.0)
        .backend(Backend::Scalar);

    assert_eq!(brick.name(), "fused_gate_up");
    let verification = brick.verify();
    assert!(verification.is_valid());
}

#[test]
fn test_fused_ops_brick_layer() {
    // Build a transformer layer with fused ops
    let qkv_brick = ComputeBrick::new(FusedQKVOp::new(1024, 8, 8)).budget_tok_per_sec(100_000.0);
    let ffn_brick = ComputeBrick::new(FusedGateUpOp::new(1024, 4096)).budget_tok_per_sec(50_000.0); // FFN is typically slower

    let layer = BrickLayer::new().with_brick(&qkv_brick).with_brick(&ffn_brick);

    // Throughput ceiling should be the FFN (bottleneck)
    assert!((layer.throughput_ceiling() - 50_000.0).abs() < 1.0);
    assert_eq!(layer.bottleneck(), Some("fused_gate_up"));
}

#[test]
fn test_fused_qkv_weights_clone() {
    let weights = FusedQKVWeights {
        q_weight: vec![1.0, 2.0],
        k_weight: vec![3.0, 4.0],
        v_weight: vec![5.0, 6.0],
    };
    let cloned = weights.clone();
    assert_eq!(cloned.q_weight, weights.q_weight);
    assert_eq!(cloned.k_weight, weights.k_weight);
    assert_eq!(cloned.v_weight, weights.v_weight);
}

#[test]
fn test_fused_gate_up_weights_clone() {
    let weights = FusedGateUpWeights { gate_weight: vec![1.0, 2.0], up_weight: vec![3.0, 4.0] };
    let cloned = weights.clone();
    assert_eq!(cloned.gate_weight, weights.gate_weight);
    assert_eq!(cloned.up_weight, weights.up_weight);
}

#[test]
fn test_fused_qkv_op_clone() {
    let op = FusedQKVOp::new(1024, 8, 4);
    let cloned = op.clone();
    assert_eq!(cloned.hidden_size, op.hidden_size);
    assert_eq!(cloned.kv_dim, op.kv_dim);
    assert_eq!(cloned.num_heads, op.num_heads);
    assert_eq!(cloned.head_dim, op.head_dim);
}

#[test]
fn test_fused_gate_up_op_clone() {
    let op = FusedGateUpOp::new(1024, 4096);
    let cloned = op.clone();
    assert_eq!(cloned.hidden_size, op.hidden_size);
    assert_eq!(cloned.intermediate_size, op.intermediate_size);
}

#[test]
fn test_fused_qkv_weights_debug() {
    let weights = FusedQKVWeights { q_weight: vec![1.0], k_weight: vec![2.0], v_weight: vec![3.0] };
    let debug_str = format!("{:?}", weights);
    assert!(debug_str.contains("FusedQKVWeights"));
}

#[test]
fn test_fused_gate_up_weights_debug() {
    let weights = FusedGateUpWeights { gate_weight: vec![1.0], up_weight: vec![2.0] };
    let debug_str = format!("{:?}", weights);
    assert!(debug_str.contains("FusedGateUpWeights"));
}

#[test]
fn test_fused_qkv_op_debug() {
    let op = FusedQKVOp::new(1024, 8, 4);
    let debug_str = format!("{:?}", op);
    assert!(debug_str.contains("FusedQKVOp"));
    assert!(debug_str.contains("1024"));
}

#[test]
fn test_fused_gate_up_op_debug() {
    let op = FusedGateUpOp::new(1024, 4096);
    let debug_str = format!("{:?}", op);
    assert!(debug_str.contains("FusedGateUpOp"));
    assert!(debug_str.contains("1024"));
}

// ========================================================================
// BrickProfiler Tests (PAR-073)
// ========================================================================

#[test]
fn test_brick_profiler_disabled_by_default() {
    let profiler = BrickProfiler::new();
    assert!(!profiler.is_enabled());
}

#[test]
fn test_brick_profiler_enabled_constructor() {
    let profiler = BrickProfiler::enabled();
    assert!(profiler.is_enabled());
}

#[test]
fn test_brick_profiler_enable_disable() {
    let mut profiler = BrickProfiler::new();
    assert!(!profiler.is_enabled());
    profiler.enable();
    assert!(profiler.is_enabled());
    profiler.disable();
    assert!(!profiler.is_enabled());
}

#[test]
fn test_brick_profiler_timing() {
    let mut profiler = BrickProfiler::enabled();

    // Time a simple operation
    let timer = profiler.start("TestBrick");
    std::thread::sleep(std::time::Duration::from_micros(100));
    profiler.stop(timer, 1);

    // Verify stats were recorded
    let stats = profiler.stats("TestBrick").expect("stats should exist");
    assert_eq!(stats.count, 1);
    assert!(stats.avg_us() >= 50.0); // Should be at least 50µs (sleep + overhead)
    assert_eq!(stats.total_elements, 1);
}

#[test]
fn test_brick_profiler_multiple_samples() {
    let mut profiler = BrickProfiler::enabled();

    for _ in 0..10 {
        let timer = profiler.start("MultiBrick");
        // Small busy loop
        let mut sum = 0u64;
        for i in 0..1000 {
            sum = sum.wrapping_add(i);
        }
        let _ = sum; // Prevent optimization
        profiler.stop(timer, 1);
    }

    let stats = profiler.stats("MultiBrick").expect("stats should exist");
    assert_eq!(stats.count, 10);
    assert_eq!(stats.total_elements, 10);
}

#[test]
fn test_brick_profiler_multiple_bricks() {
    let mut profiler = BrickProfiler::enabled();

    let timer = profiler.start("BrickA");
    profiler.stop(timer, 1);

    let timer = profiler.start("BrickB");
    profiler.stop(timer, 2);

    assert!(profiler.stats("BrickA").is_some());
    assert!(profiler.stats("BrickB").is_some());
    assert_eq!(profiler.total_tokens(), 3);
}

#[test]
fn test_brick_profiler_disabled_no_record() {
    let mut profiler = BrickProfiler::new(); // Disabled by default

    let timer = profiler.start("DisabledBrick");
    profiler.stop(timer, 1);

    // Should not record anything when disabled
    assert!(profiler.stats("DisabledBrick").is_none());
    assert_eq!(profiler.total_tokens(), 0);
}

#[test]
fn test_brick_profiler_reset() {
    let mut profiler = BrickProfiler::enabled();

    let timer = profiler.start("ResetBrick");
    profiler.stop(timer, 5);

    assert_eq!(profiler.total_tokens(), 5);

    profiler.reset();

    assert_eq!(profiler.total_tokens(), 0);
    assert!(profiler.stats("ResetBrick").is_none());
}

#[test]
fn test_brick_profiler_summary() {
    let mut profiler = BrickProfiler::enabled();

    let timer = profiler.start("SummaryBrick");
    profiler.stop(timer, 10);

    let summary = profiler.summary();
    assert!(summary.contains("Brick Profiler Summary"));
    assert!(summary.contains("SummaryBrick"));
    assert!(summary.contains("10 tokens"));
}

#[test]
fn test_brick_stats_new() {
    let stats = BrickStats::new("TestStats");
    assert_eq!(stats.name, "TestStats");
    assert_eq!(stats.count, 0);
    assert_eq!(stats.total_ns, 0);
    assert_eq!(stats.min_ns, u64::MAX);
    assert_eq!(stats.max_ns, 0);
}

#[test]
fn test_brick_stats_add_sample() {
    let mut stats = BrickStats::new("Test");
    stats.add_sample(1000, 1); // 1µs
    stats.add_sample(2000, 1); // 2µs
    stats.add_sample(3000, 1); // 3µs

    assert_eq!(stats.count, 3);
    assert_eq!(stats.total_ns, 6000);
    assert_eq!(stats.min_ns, 1000);
    assert_eq!(stats.max_ns, 3000);
    assert_eq!(stats.total_elements, 3);
    assert!((stats.avg_us() - 2.0).abs() < 0.001);
}

#[test]
fn test_brick_stats_throughput() {
    let mut stats = BrickStats::new("Throughput");
    // 1000 elements in 1ms = 1,000,000 elements/sec
    stats.add_sample(1_000_000, 1000); // 1ms, 1000 elements

    let throughput = stats.throughput();
    assert!((throughput - 1_000_000.0).abs() < 1000.0);
}

#[test]
fn test_brick_timer_debug() {
    let profiler = BrickProfiler::new();
    let timer = profiler.start("DebugTimer");
    let debug_str = format!("{:?}", timer);
    assert!(debug_str.contains("BrickTimer"));
    assert!(debug_str.contains("DebugTimer"));
}

#[test]
fn test_brick_sample_clone() {
    let sample = BrickSample { brick_id: 42, elapsed_ns: 1000, elements: 5 };
    let cloned = sample;
    assert_eq!(cloned.brick_id, 42);
    assert_eq!(cloned.elapsed_ns, 1000);
    assert_eq!(cloned.elements, 5);
}
