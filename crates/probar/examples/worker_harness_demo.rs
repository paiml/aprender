//! Worker Harness Demo (PROBAR-SPEC-013)
//!
//! Demonstrates comprehensive Web Worker testing for WASM applications:
//! - Worker lifecycle state machine
//! - Lamport clock message ordering
//! - Ring buffer testing for audio streaming
//! - SharedArrayBuffer/Atomics testing
//!
//! Run with: cargo run --example worker_harness_demo -p jugar-probar

use jugar_probar::worker_harness::{
    RingBufferTestConfig, SharedMemoryTestConfig, WorkerLifecycleState, WorkerMetrics,
    WorkerTestConfig, WorkerTestHarness, WorkerTestResult,
};
use std::time::Duration;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Worker Harness Demo (PROBAR-SPEC-013)                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // =========================================================================
    // 1. Worker Test Configuration
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  1. Worker Test Configuration Presets");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let default_config = WorkerTestConfig::default();
    let minimal_config = WorkerTestConfig::minimal();
    let comprehensive_config = WorkerTestConfig::comprehensive();

    println!("  WorkerTestConfig::default():");
    print_config(&default_config);
    println!();

    println!("  WorkerTestConfig::minimal() (fast CI):");
    print_config(&minimal_config);
    println!();

    println!("  WorkerTestConfig::comprehensive() (thorough):");
    print_config(&comprehensive_config);
    println!();

    // =========================================================================
    // 2. Worker Lifecycle States
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  2. Worker Lifecycle State Machine");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let states = [
        WorkerLifecycleState::NotCreated,
        WorkerLifecycleState::Loading,
        WorkerLifecycleState::Initializing,
        WorkerLifecycleState::Ready,
        WorkerLifecycleState::Processing,
        WorkerLifecycleState::Error,
        WorkerLifecycleState::Terminated,
    ];

    println!("  Valid state transitions:");
    println!();
    println!("    NotCreated ──► Loading ──► Initializing ──► Ready");
    println!("                                                 │");
    println!("                                                 ▼");
    println!("                   Terminated ◄── Error ◄── Processing");
    println!();

    println!("  All states:");
    for state in &states {
        println!("    • {}", state);
    }
    println!();

    // =========================================================================
    // 3. Lifecycle Transition Testing
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  3. Lifecycle Transition Testing");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let harness = WorkerTestHarness::new();
    let failures = harness.test_lifecycle_transitions();

    if failures.is_empty() {
        println!("  ✓ All lifecycle transitions valid");
    } else {
        println!("  ✗ Found {} invalid transitions:", failures.len());
        for f in &failures {
            println!("    • {}", f);
        }
    }
    println!();

    // =========================================================================
    // 4. Lamport Clock Message Ordering
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  4. Lamport Clock Message Ordering");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Valid ordering
    let valid_timestamps: Vec<u64> = (0..10).map(|i| i * 10).collect();
    let valid_failures = harness.verify_message_ordering(&valid_timestamps);

    println!("  Testing valid sequence [0, 10, 20, 30, ...]:");
    if valid_failures.is_empty() {
        println!("    ✓ Message ordering verified");
    }
    println!();

    // Invalid ordering
    let invalid_timestamps = vec![1, 2, 5, 3, 6]; // 3 < 5 violates ordering
    let invalid_failures = harness.verify_message_ordering(&invalid_timestamps);

    println!("  Testing invalid sequence [1, 2, 5, 3, 6]:");
    if !invalid_failures.is_empty() {
        println!(
            "    ✗ Detected {} ordering violations:",
            invalid_failures.len()
        );
        for f in &invalid_failures {
            println!("      • {}", f);
        }
    }
    println!();

    // =========================================================================
    // 5. Ring Buffer Testing
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  5. Ring Buffer Testing (Audio Streaming)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let ring_config = RingBufferTestConfig::default();
    println!("  RingBufferTestConfig::default():");
    println!("    ├─ buffer_size: {} bytes", ring_config.buffer_size);
    println!("    ├─ sample_size: {} bytes", ring_config.sample_size);
    println!("    ├─ num_samples: {}", ring_config.num_samples);
    println!("    ├─ test_overflow: {}", ring_config.test_overflow);
    println!("    ├─ test_underrun: {}", ring_config.test_underrun);
    println!("    └─ test_concurrent: {}", ring_config.test_concurrent);
    println!();

    let ring_result = harness.test_ring_buffer(&ring_config);
    println!("  Ring buffer test results:");
    println!("    ├─ Writes succeeded: {}", ring_result.writes_succeeded);
    println!("    ├─ Reads succeeded: {}", ring_result.reads_succeeded);
    println!(
        "    ├─ Overflows detected: {}",
        ring_result.overflows_detected
    );
    println!(
        "    ├─ Underruns detected: {}",
        ring_result.underruns_detected
    );
    println!(
        "    └─ Passed: {}",
        if ring_result.passed { "✓" } else { "✗" }
    );
    println!();

    // =========================================================================
    // 6. Shared Memory Testing
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  6. Shared Memory Testing (Atomics)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let shared_config = SharedMemoryTestConfig::default();
    println!("  SharedMemoryTestConfig::default():");
    println!("    ├─ buffer_size: {} bytes", shared_config.buffer_size);
    println!("    ├─ num_atomic_ops: {}", shared_config.num_atomic_ops);
    println!(
        "    ├─ test_wait_notify: {}",
        shared_config.test_wait_notify
    );
    println!(
        "    ├─ test_concurrent_writes: {}",
        shared_config.test_concurrent_writes
    );
    println!("    └─ wait_timeout: {:?}", shared_config.wait_timeout);
    println!();

    let shared_result = harness.test_shared_memory(&shared_config);
    println!("  Shared memory test results:");
    println!(
        "    ├─ Atomics correct: {}",
        if shared_result.atomics_correct {
            "✓"
        } else {
            "✗"
        }
    );
    println!(
        "    ├─ Wait/notify works: {}",
        if shared_result.wait_notify_correct {
            "✓"
        } else {
            "✗"
        }
    );
    println!(
        "    ├─ Race conditions: {}",
        shared_result.race_conditions_detected
    );
    println!(
        "    └─ Passed: {}",
        if shared_result.is_passed() {
            "✓"
        } else {
            "✗"
        }
    );
    println!();

    // =========================================================================
    // 7. Worker Metrics
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  7. Worker Metrics & Memory Leak Detection");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let mut metrics = WorkerMetrics {
        initialization_time: Duration::from_millis(50),
        average_message_latency: Duration::from_micros(100),
        max_message_latency: Duration::from_millis(5),
        messages_processed: 10000,
        messages_dropped: 5,
        memory_start: 1024 * 1024,      // 1MB
        memory_end: 1024 * 1024 + 1024, // 1MB + 1KB
        error_recoveries: 2,
    };

    println!("  Sample metrics:");
    println!("    ├─ Init time: {:?}", metrics.initialization_time);
    println!("    ├─ Avg latency: {:?}", metrics.average_message_latency);
    println!("    ├─ Max latency: {:?}", metrics.max_message_latency);
    println!("    ├─ Messages processed: {}", metrics.messages_processed);
    println!("    ├─ Messages dropped: {}", metrics.messages_dropped);
    println!("    ├─ Memory growth: {} bytes", metrics.memory_growth());
    println!(
        "    └─ Memory leak: {}",
        if metrics.has_memory_leak() {
            "YES"
        } else {
            "NO"
        }
    );
    println!();

    // Simulate memory leak
    metrics.memory_end = 1024 * 1024 + 200 * 1024; // 200KB growth (20%)
    println!("  After 20% memory growth:");
    println!(
        "    └─ Memory leak detected: {}",
        if metrics.has_memory_leak() {
            "YES (>10%)"
        } else {
            "NO"
        }
    );
    println!();

    // =========================================================================
    // 8. Full Test Result
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  8. Full Test Result");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let result = WorkerTestResult {
        passed: true,
        lifecycle_passed: true,
        ordering_passed: true,
        shared_memory_passed: true,
        ring_buffer_passed: true,
        error_recovery_passed: true,
        memory_leak_passed: true,
        failures: vec![],
        metrics: WorkerMetrics::default(),
    };

    println!("{}", result);
    println!();

    // =========================================================================
    // 9. Generated JavaScript Snippets
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  9. Generated JavaScript for CDP Injection");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let lifecycle_js = WorkerTestHarness::lifecycle_test_js();
    println!("  lifecycle_test_js() preview:");
    println!(
        "    {}",
        &lifecycle_js[..lifecycle_js.len().min(100)].replace('\n', " ")
    );
    println!("    ... ({} chars total)", lifecycle_js.len());
    println!();

    let ring_buffer_js = WorkerTestHarness::ring_buffer_test_js(16384);
    println!("  ring_buffer_test_js(16384) preview:");
    println!(
        "    {}",
        &ring_buffer_js[..ring_buffer_js.len().min(100)].replace('\n', " ")
    );
    println!("    ... ({} chars total)", ring_buffer_js.len());
    println!();

    // =========================================================================
    // 10. Integration Example
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  10. Integration Example");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    println!("  Use in your tests:");
    println!();
    println!("    ```rust");
    println!("    use jugar_probar::worker_harness::*;");
    println!();
    println!("    #[test]");
    println!("    fn test_audio_worker() {{");
    println!("        let harness = WorkerTestHarness::with_config(");
    println!("            WorkerTestConfig::comprehensive()");
    println!("        );");
    println!();
    println!("        // Test lifecycle transitions");
    println!("        assert!(harness.test_lifecycle_transitions().is_empty());");
    println!();
    println!("        // Test ring buffer for audio streaming");
    println!("        let rb_result = harness.test_ring_buffer(&RingBufferTestConfig {{");
    println!("            buffer_size: 65536,  // 64KB audio buffer");
    println!("            sample_size: 512,");
    println!("            num_samples: 1000,");
    println!("            ..Default::default()");
    println!("        }});");
    println!("        assert!(rb_result.passed);");
    println!("    }}");
    println!("    ```");
    println!();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           Demo Complete - Worker Testing Ready!              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}

fn print_config(config: &WorkerTestConfig) {
    println!("    ├─ init_timeout: {:?}", config.init_timeout);
    println!("    ├─ command_timeout: {:?}", config.command_timeout);
    println!("    ├─ stress_iterations: {}", config.stress_iterations);
    println!(
        "    ├─ verify_lamport_ordering: {}",
        config.verify_lamport_ordering
    );
    println!("    ├─ test_shared_memory: {}", config.test_shared_memory);
    println!("    ├─ test_error_recovery: {}", config.test_error_recovery);
    println!("    └─ test_memory_leaks: {}", config.test_memory_leaks);
}
