//! Realizar Observability Demo
//!
//! Demonstrates metrics, tracing, and A/B testing capabilities.
//!
//! Run with: `cargo run --example observability_demo --features server`

#[cfg(feature = "server")]
use realizar::observability::{ABTest, MetricPoint, ObservabilityConfig, Observer, Span};

#[cfg(feature = "server")]
fn main() {
    println!("=== Realizar Observability Demo ===\n");

    // 1. Configuration
    println!("--- Configuration ---");
    demo_configuration();

    // 2. Metrics
    println!("\n--- Metrics ---");
    demo_metrics();

    // 3. Tracing
    println!("\n--- Tracing ---");
    demo_tracing();

    // 4. A/B Testing
    println!("\n--- A/B Testing ---");
    demo_ab_testing();

    // 5. Full Observer
    println!("\n--- Full Observer ---");
    demo_observer();

    println!("\n=== Demo Complete ===");
}

#[cfg(not(feature = "server"))]
fn main() {
    println!("This example requires the 'server' feature.");
    println!("Run with: cargo run --example observability_demo --features server");
}

#[cfg(feature = "server")]
fn demo_configuration() {
    // Default configuration
    let default_config = ObservabilityConfig::default();
    println!("Default config:");
    println!("  Tracing enabled: {}", default_config.tracing_enabled);
    println!(
        "  Trace sample rate: {:.1}%",
        default_config.trace_sample_rate * 100.0
    );
    println!(
        "  A/B testing enabled: {}",
        default_config.ab_testing_enabled
    );

    // Custom configuration using builder pattern
    let custom_config = ObservabilityConfig::new()
        .with_trueno_db("trueno-db://localhost:8086")
        .with_tracing(true)
        .with_sample_rate(0.5); // 50% sampling

    println!("\nCustom config:");
    println!("  Trueno DB URI: {:?}", custom_config.trueno_db_uri);
    println!(
        "  Trace sample rate: {:.1}%",
        custom_config.trace_sample_rate * 100.0
    );
}

#[cfg(feature = "server")]
fn demo_metrics() {
    // Create metric points with value in constructor
    let latency = MetricPoint::new("inference_latency_ms", 45.5)
        .with_label("model", "llama3-8b")
        .with_label("quantization", "q4_k_m");

    println!("Metric: {}", latency.name);
    println!("  Value: {}", latency.value);
    println!("  Labels: {:?}", latency.labels);

    // Convert to Trueno-DB line protocol
    let line_protocol = latency.to_line_protocol();
    println!("  Line protocol: {}", line_protocol);

    // Multiple metrics
    let metrics = vec![
        MetricPoint::new("tokens_generated", 256.0).with_label("model", "llama3-8b"),
        MetricPoint::new("throughput_tps", 35.2).with_label("model", "llama3-8b"),
        MetricPoint::new("memory_mb", 4096.0).with_label("model", "llama3-8b"),
    ];

    println!("\nBatch metrics:");
    for metric in &metrics {
        println!("  {} = {}", metric.name, metric.value);
    }
}

#[cfg(feature = "server")]
fn demo_tracing() {
    // Create a root span with trace ID
    let trace_id = "trace-12345";
    let mut root_span = Span::new("inference_request", trace_id)
        .with_attribute("model", "llama3-8b")
        .with_attribute("prompt_tokens", "128");

    println!("Root span: {}", root_span.operation);
    println!("  Trace ID: {}", root_span.trace_id);
    println!("  Span ID: {}", root_span.span_id);

    // Create child spans
    let tokenize_span = root_span.child("tokenize");
    let mut tokenize_span = tokenize_span.with_attribute("vocab_size", "32000");
    tokenize_span.end_ok();
    println!(
        "  Child: {} ({:?})",
        tokenize_span.operation, tokenize_span.status
    );

    let generate_span = root_span.child("generate");
    let mut generate_span = generate_span
        .with_attribute("max_tokens", "256")
        .with_attribute("temperature", "0.7");
    generate_span.end_ok();
    println!(
        "  Child: {} ({:?})",
        generate_span.operation, generate_span.status
    );

    let mut decode_span = root_span.child("decode");
    decode_span.end_ok();
    println!(
        "  Child: {} ({:?})",
        decode_span.operation, decode_span.status
    );

    // End root span
    root_span.end_ok();
    println!("Root span ended: {:?}", root_span.status);

    // Error handling
    let mut error_span = Span::new("failed_request", "trace-error");
    error_span.end_error("Out of memory");
    println!("\nError span status: {:?}", error_span.status);
}

#[cfg(feature = "server")]
fn demo_ab_testing() {
    // Create an A/B test
    let test = ABTest::new("model_comparison")
        .with_variant("control", "llama3-8b-q4", 0.5)
        .with_variant("treatment", "llama3-8b-q8", 0.5);

    println!("A/B Test: {}", test.name);
    println!("Variants:");
    for variant in &test.variants {
        println!(
            "  - {} ({}): weight {:.0}%",
            variant.name,
            variant.model,
            variant.weight * 100.0
        );
    }
    println!("Valid config: {}", test.is_valid());

    // Deterministic variant selection
    println!("\nVariant selection (deterministic by user ID):");
    let users = ["user-001", "user-002", "user-003", "user-004", "user-005"];
    for user in users {
        if let Some(variant) = test.select(user) {
            println!("  {} -> {} ({})", user, variant.name, variant.model);
        }
    }

    // Same user always gets same variant
    println!("\nConsistency check:");
    let user = "consistent-user";
    let v1 = test.select(user).map(|v| v.name.clone());
    let v2 = test.select(user).map(|v| v.name.clone());
    println!("  First selection:  {:?}", v1);
    println!("  Second selection: {:?}", v2);
    println!("  Consistent: {}", v1 == v2);
}

#[cfg(feature = "server")]
fn demo_observer() {
    // Create observer with custom config
    let config = ObservabilityConfig::new()
        .with_tracing(true)
        .with_sample_rate(1.0); // 100% for demo

    let observer = Observer::new(config);

    // Generate request IDs
    println!("Request IDs:");
    for _ in 0..3 {
        println!("  {}", observer.next_request_id());
    }

    // Record inference metrics (model, tokens, latency_ms)
    println!("\nRecording inference metrics...");
    observer.record_inference("llama3-8b", 256, 45);
    observer.record_inference("llama3-8b", 312, 52);
    observer.record_inference("llama3-8b", 128, 30);

    // Record spans
    println!("Recording spans...");
    let span1 = Span::new("request_1", "trace-1").with_attribute("model", "llama3-8b");
    let span2 = Span::new("request_2", "trace-2").with_attribute("model", "codellama");
    observer.record_span(span1);
    observer.record_span(span2);

    // Record A/B test results
    println!("Recording A/B test results...");
    observer.record_ab_result("model_test", "control", true, 45, 256);
    observer.record_ab_result("model_test", "treatment", true, 52, 312);
    observer.record_ab_result("model_test", "control", false, 100, 0);

    // Get Prometheus metrics
    let prometheus = observer.prometheus_metrics();
    println!("\nPrometheus format (sample):");
    for line in prometheus.lines().take(10) {
        println!("  {}", line);
    }

    // A/B test results
    if let Some(results) = observer.get_ab_results("model_test") {
        println!("\nA/B test results for '{}':", results.test_name);
        for (name, result) in &results.variants {
            println!(
                "  {}: {} requests, {:.1}% success rate",
                name,
                result.requests,
                result.success_rate() * 100.0
            );
        }
    }
}
