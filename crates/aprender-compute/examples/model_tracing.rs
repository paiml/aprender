#![allow(clippy::disallowed_methods)]
//! Phase 13: Model-Level Inference Tracing Demo (E.11)
//!
//! This example demonstrates the five complementary tracing systems for
//! debugging transformer inference: activations, attention, logits,
//! quantization errors, and KV cache state.
//!
//! Run with: cargo run --example model_tracing

use trueno::brick::{
    AttentionTraceConfig, AttentionWeightTrace, BrickId, KvCacheSessionTrace, KvCacheStateTrace,
    LayerActivationTrace, LogitEvolutionTrace, ModelActivationTrace, ModelQuantizationError,
    ModelTracer, ModelTracerConfig, QuantType, QuantizationErrorTrace, TensorStats,
};

fn main() {
    println!("=== Phase 13: Model-Level Inference Tracing Demo ===\n");

    // Demo 1: TensorStats for anomaly detection
    demo_tensor_stats();

    // Demo 2: LayerActivationTrace for layer-by-layer monitoring
    demo_layer_activation_trace();

    // Demo 3: AttentionWeightTrace for debugging attention patterns
    demo_attention_trace();

    // Demo 4: LogitEvolutionTrace for understanding token selection
    demo_logit_evolution();

    // Demo 5: QuantizationErrorTrace for measuring quantization impact
    demo_quantization_error();

    // Demo 6: KvCacheStateTrace for cache monitoring
    demo_kv_cache_trace();

    // Demo 7: Unified ModelTracer workflow
    demo_model_tracer();

    println!("\n✓ Model tracing demo completed");
}

fn demo_tensor_stats() {
    println!("--- Demo 1: TensorStats (MLT-01) ---");

    // Normal tensor
    let normal_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let normal_stats = TensorStats::from_slice(&normal_data);
    println!(
        "Normal tensor: mean={:.2}, std={:.2}, L2={:.2}",
        normal_stats.mean, normal_stats.std, normal_stats.l2_norm
    );
    println!("  Anomaly: {}", if normal_stats.has_anomaly() { "YES" } else { "no" });

    // Tensor with NaN (bad!)
    let nan_data = vec![1.0, f32::NAN, 3.0, 4.0, 5.0];
    let nan_stats = TensorStats::from_slice(&nan_data);
    println!("NaN tensor: nan_count={}", nan_stats.nan_count);
    println!(
        "  Anomaly: {} - {}",
        if nan_stats.has_anomaly() { "YES" } else { "no" },
        nan_stats.anomaly_description().unwrap_or_default()
    );

    // Exploding tensor (bad!)
    let explode_data = vec![1.0, 2.0, 1e7, 4.0, 5.0];
    let explode_stats = TensorStats::from_slice(&explode_data);
    println!("Exploding tensor: max={:.2e}", explode_stats.max);
    println!(
        "  Anomaly: {} - {}",
        if explode_stats.has_anomaly() { "YES" } else { "no" },
        explode_stats.anomaly_description().unwrap_or_default()
    );

    println!();
}

fn demo_layer_activation_trace() {
    println!("--- Demo 2: LayerActivationTrace (MLT-01) ---");

    let mut model_trace = ModelActivationTrace::with_capacity(3);

    // Simulate 3 layers
    for layer_idx in 0..3 {
        let mut layer = LayerActivationTrace::new(layer_idx);

        // Simulate activation statistics at each stage
        let scale = 1.0 + layer_idx as f32 * 0.1;
        layer.input_stats = TensorStats::from_slice(&vec![scale; 100]);
        layer.post_norm_stats = TensorStats::from_slice(&vec![1.0; 100]);
        layer.post_attn_stats = TensorStats::from_slice(&vec![scale * 1.1; 100]);
        layer.post_ffn_stats = TensorStats::from_slice(&vec![scale * 1.2; 100]);
        layer.output_stats = TensorStats::from_slice(&vec![scale * 1.3; 100]);
        layer.residual_ratio = 0.85;

        println!(
            "Layer {}: input_mean={:.2}, output_mean={:.2}, residual={:.2}",
            layer_idx, layer.input_stats.mean, layer.output_stats.mean, layer.residual_ratio
        );

        model_trace.add_layer(layer);
    }

    model_trace.finalize();
    println!("Model anomaly: {}", if model_trace.has_anomaly { "YES" } else { "no" });
    println!();
}

fn demo_attention_trace() {
    println!("--- Demo 3: AttentionWeightTrace (MLT-02) ---");

    // Simulate attention weights for position 10 looking at positions 0-10
    let weights = vec![0.4, 0.05, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05, 0.1, 0.05, 0.05];

    let trace = AttentionWeightTrace::from_weights(
        0,  // layer 0
        0,  // head 0
        10, // query position
        &weights, 5, // top-k = 5
    );

    println!("Query position: {}", trace.query_pos);
    println!("Top-5 attended positions: {:?}", trace.top_k_positions);
    println!(
        "Top-5 weights: {:?}",
        trace.top_k_weights.iter().map(|w| format!("{:.2}", w)).collect::<Vec<_>>()
    );
    println!("Tail mass (outside top-5): {:.2}", trace.tail_mass);
    println!("Entropy: {:.3}", trace.entropy);

    // Check diagnostic patterns
    println!("Is attention sink (BOS)? {}", trace.is_attention_sink(0.3));
    println!("Has recency bias? {}", trace.has_recency_bias(3, 0.2));

    // Configure selective tracing
    let config = AttentionTraceConfig {
        top_k: 10,
        layers: Some(vec![0, 5, 10]), // Only trace layers 0, 5, 10
        heads: Some(vec![0]),         // Only trace head 0
        weight_threshold: 0.01,
    };
    println!("Trace layer 0? {}", config.should_trace_layer(0));
    println!("Trace layer 3? {}", config.should_trace_layer(3));

    println!();
}

fn demo_logit_evolution() {
    println!("--- Demo 4: LogitEvolutionTrace (MLT-03) ---");

    let mut trace = LogitEvolutionTrace::new(100, 0.7, 0.9);

    // Track token "hello" (id=42)
    let hello = trace.track_token(42, "hello".to_string());
    hello.record_layer(0.5, 500); // Layer 0: rank 500
    hello.record_layer(1.0, 200); // Layer 1: rank 200
    hello.record_layer(2.5, 50); // Layer 2: rank 50
    hello.record_layer(5.0, 10); // Layer 3: rank 10
    hello.record_layer(8.0, 1); // Layer 4: rank 1 (almost selected)
    hello.final_probability = 0.35;
    hello.final_rank = 1;

    // Track token "world" (id=99)
    let world = trace.track_token(99, "world".to_string());
    world.record_layer(1.0, 300);
    world.record_layer(2.0, 150);
    world.record_layer(3.0, 80);
    world.record_layer(4.0, 40);
    world.record_layer(9.0, 0); // Selected!
    world.final_probability = 0.40;
    world.final_rank = 0;

    println!(
        "Position: {}, Temperature: {}, Top-p: {}",
        trace.position, trace.temperature, trace.top_p
    );

    for token in &trace.tracked_tokens {
        println!(
            "Token '{}' (id={}): final_rank={}, final_prob={:.2}",
            token.token_str, token.token_id, token.final_rank, token.final_probability
        );
        println!("  Rank evolution: {:?}", token.per_layer_rank);
        if let Some(decisive) = token.decisive_layer() {
            println!("  Decisive layer: {}", decisive);
        }
    }

    // Demonstrate rank computation
    let logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
    println!("Rank of token 1 (logit=5.0): {}", LogitEvolutionTrace::compute_rank(&logits, 1));
    println!("Rank of token 0 (logit=1.0): {}", LogitEvolutionTrace::compute_rank(&logits, 0));

    println!();
}

fn demo_quantization_error() {
    println!("--- Demo 5: QuantizationErrorTrace (MLT-04) ---");

    // FP32 reference output
    let reference = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    // Q4_K quantized output (slight errors)
    let quantized = vec![1.02, 1.98, 3.05, 3.95, 5.1, 5.92, 7.08, 7.95];

    let trace = QuantizationErrorTrace::compute(
        BrickId::QkvProjection,
        5,
        &quantized,
        &reference,
        QuantType::Q4_K,
    );

    println!("Brick: {:?}, Layer: {}", trace.brick_id, trace.layer_idx);
    println!(
        "Quantization: {:?} ({:.1} bits/element, {:.1}x compression)",
        trace.quant_type,
        trace.quant_type.bits_per_element(),
        trace.quant_type.compression_ratio()
    );
    println!("MSE: {:.6}", trace.mse);
    println!("Max Abs Error: {:.4}", trace.max_abs_error);
    println!("Cosine Similarity: {:.6}", trace.cosine_similarity);
    println!("SNR: {:.1} dB", trace.snr_db);

    println!(
        "Status: {}",
        if trace.is_acceptable() {
            "ACCEPTABLE"
        } else if trace.is_warning() {
            "WARNING"
        } else {
            "CRITICAL"
        }
    );

    // Model-level aggregation
    let mut model_error = ModelQuantizationError::default();
    model_error.add_error(trace);

    // Add a critical error
    let bad_quantized = vec![5.0, 1.0, 4.0, 2.0, 3.0, 8.0, 6.0, 7.0]; // Wrong order
    let bad_trace = QuantizationErrorTrace::compute(
        BrickId::AttentionScore,
        3,
        &bad_quantized,
        &reference,
        QuantType::Q2_K,
    );
    model_error.add_error(bad_trace);

    println!("\nModel-level summary:");
    println!("  Total errors tracked: {}", model_error.brick_errors.len());
    println!("  Warnings: {}", model_error.warning_count());
    println!("  Criticals: {}", model_error.critical_count());
    if let Some(worst) = model_error.worst_brick() {
        println!(
            "  Worst brick: {:?} at layer {} (cosine={:.4})",
            worst.brick_id, worst.layer_idx, worst.cosine_similarity
        );
    }

    println!();
}

fn demo_kv_cache_trace() {
    println!("--- Demo 6: KvCacheStateTrace (MLT-05) ---");

    let mut session = KvCacheSessionTrace::default();

    // Simulate 50 generation steps
    for step in 0..50 {
        let mut trace = KvCacheStateTrace::new(step, 2048);
        trace.valid_positions = step + 1;
        trace.cache_size_bytes = (step + 1) * 4096; // 4KB per position
        trace.cache_hit_rate = if step == 0 { 0.0 } else { 0.95 };
        trace.oldest_position = 0;
        trace.evictions_this_step = if step > 40 { 1 } else { 0 };
        trace.fragmentation = (step as f32) * 0.01;
        trace.accessed_positions = (0..=step).collect();

        session.add_step(trace);
    }

    println!("Session summary:");
    println!("  Steps: {}", session.steps.len());
    println!("  Total evictions: {}", session.total_evictions);
    println!("  Peak memory: {} KB", session.peak_memory_bytes / 1024);
    println!("  Avg hit rate: {:.1}%", session.avg_hit_rate * 100.0);

    // Check last step
    if let Some(last) = session.steps.last() {
        println!("\nFinal step state:");
        println!("  Valid positions: {}/{}", last.valid_positions, last.max_positions);
        println!("  Utilization: {:.1}%", last.utilization() * 100.0);
        println!("  Window exhausted: {}", last.is_window_exhausted());
    }

    // Check for thrashing
    println!("\nThrashing detected: {}", session.has_thrashing(5, 0.5));

    println!();
}

fn demo_model_tracer() {
    println!("--- Demo 7: Unified ModelTracer ---");

    // Create tracer with full configuration
    let config = ModelTracerConfig::full();
    println!(
        "Config: activations={}, attention={}, logits={}, quant={}, kv_cache={}",
        config.trace_activations,
        config.trace_attention,
        config.trace_logits,
        config.trace_quant_error,
        config.trace_kv_cache
    );

    let mut tracer = ModelTracer::new(config);

    // Simulate 3 forward passes
    for pos in 0..3 {
        println!("\nForward pass {} ...", pos);
        tracer.begin_forward(pos);

        // Record layer activations (simulated)
        for layer_idx in 0..12 {
            let mut layer = LayerActivationTrace::new(layer_idx);

            // Inject anomaly in pass 2, layer 5
            if pos == 2 && layer_idx == 5 {
                layer.post_attn_stats = TensorStats::from_slice(&[f32::INFINITY, 1.0, 2.0]);
            } else {
                layer.input_stats = TensorStats::from_slice(&vec![1.0; 100]);
                layer.output_stats = TensorStats::from_slice(&vec![1.1; 100]);
            }

            tracer.record_layer_activation(layer);
        }

        // Record KV cache state
        let mut kv = KvCacheStateTrace::new(pos, 2048);
        kv.valid_positions = pos + 1;
        kv.cache_hit_rate = 0.95;
        tracer.record_kv_state(kv);

        // End forward and check for anomalies
        if let Some(anomaly) = tracer.end_forward() {
            println!("  ⚠ ANOMALY: {}", anomaly);
        } else {
            println!("  ✓ No anomalies");
        }
    }

    // Print summary
    let summary = tracer.summary();
    println!("\n{}", summary);

    // Demonstrate lightweight config (production use)
    println!("\nLightweight config (for production):");
    let lightweight = ModelTracerConfig::lightweight();
    println!(
        "  activations={}, attention={}, logits={}, quant={}, kv_cache={}",
        lightweight.trace_activations,
        lightweight.trace_attention,
        lightweight.trace_logits,
        lightweight.trace_quant_error,
        lightweight.trace_kv_cache
    );

    // Demonstrate disabled config (zero overhead)
    let disabled = ModelTracerConfig::default();
    println!("\nDisabled config (zero overhead): is_enabled={}", disabled.is_enabled());
}
