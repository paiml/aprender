//! Integration tests for Sprint 42: Trace Context Propagation
//!
//! Tests W3C Trace Context + RENACER_LOGICAL_CLOCK propagation across process boundaries.

use renacer::trace_context::{LamportClock, TraceContext};
use serial_test::serial;
use std::env;

#[test]
#[serial]
fn test_trace_context_env_propagation() {
    // Create a trace context
    let trace_id = [
        0x0a, 0xf7, 0x65, 0x19, 0x16, 0xcd, 0x43, 0xdd, 0x84, 0x48, 0xeb, 0x21, 0x1c, 0x80, 0x31,
        0x9c,
    ];
    let parent_id = [0xb7, 0xad, 0x6b, 0x71, 0x69, 0x20, 0x33, 0x31];

    let ctx = TraceContext { version: 0, trace_id, parent_id, trace_flags: 1 };

    // Set environment variable
    ctx.set_env();

    // Verify it can be read back
    let retrieved = TraceContext::from_env().expect("test");
    assert_eq!(retrieved.trace_id, trace_id);
    assert_eq!(retrieved.parent_id, parent_id);
    assert_eq!(retrieved.trace_flags, 1);
    assert!(retrieved.is_sampled());

    // Cleanup
    env::remove_var("TRACEPARENT");
}

#[test]
#[serial]
fn test_logical_clock_env_propagation() {
    // Set logical clock
    let timestamp = 42u64;
    TraceContext::set_logical_clock_env(timestamp);

    // Verify it can be read back
    let retrieved = TraceContext::logical_clock_from_env().expect("test");
    assert_eq!(retrieved, timestamp);

    // Cleanup
    env::remove_var("RENACER_LOGICAL_CLOCK");
}

#[test]
#[serial]
fn test_logical_clock_env_missing() {
    env::remove_var("RENACER_LOGICAL_CLOCK");

    let result = TraceContext::logical_clock_from_env();
    assert_eq!(result, None);
}

#[test]
#[serial]
fn test_logical_clock_env_invalid() {
    env::set_var("RENACER_LOGICAL_CLOCK", "not_a_number");

    let result = TraceContext::logical_clock_from_env();
    assert_eq!(result, None);

    env::remove_var("RENACER_LOGICAL_CLOCK");
}

#[test]
#[serial]
fn test_logical_clock_env_large_value() {
    let timestamp = u64::MAX;
    TraceContext::set_logical_clock_env(timestamp);

    let retrieved = TraceContext::logical_clock_from_env().expect("test");
    assert_eq!(retrieved, timestamp);

    env::remove_var("RENACER_LOGICAL_CLOCK");
}

#[test]
#[serial]
fn test_combined_propagation() {
    // Scenario: Parent process sets both TRACEPARENT and RENACER_LOGICAL_CLOCK
    // Child process inherits both

    // Parent process
    let trace_id = [1u8; 16];
    let parent_id = [2u8; 8];
    let ctx = TraceContext { version: 0, trace_id, parent_id, trace_flags: 1 };
    ctx.set_env();

    let clock = LamportClock::new();
    let timestamp = clock.tick();
    TraceContext::set_logical_clock_env(timestamp);

    // Child process reads both
    let child_ctx = TraceContext::from_env().expect("test");
    let child_clock = TraceContext::logical_clock_from_env().expect("test");

    assert_eq!(child_ctx.trace_id, trace_id);
    assert_eq!(child_clock, timestamp);

    // Cleanup
    env::remove_var("TRACEPARENT");
    env::remove_var("RENACER_LOGICAL_CLOCK");
}

#[test]
#[serial]
fn test_fork_simulation() {
    // Simulate fork() scenario where child inherits environment

    // Parent sets up tracing context
    let parent_trace_id = [0xaa; 16];
    let parent_span_id = [0xbb; 8];
    let parent_ctx = TraceContext {
        version: 0,
        trace_id: parent_trace_id,
        parent_id: parent_span_id,
        trace_flags: 1,
    };
    parent_ctx.set_env();

    let parent_clock = LamportClock::new();
    parent_clock.tick();
    parent_clock.tick();
    let parent_timestamp = parent_clock.tick();
    TraceContext::set_logical_clock_env(parent_timestamp);

    // Child process (simulated)
    let child_ctx = TraceContext::from_env().expect("test");
    let child_start_clock = TraceContext::logical_clock_from_env().expect("test");

    // Child should have same trace_id (same trace)
    assert_eq!(child_ctx.trace_id, parent_trace_id);

    // Child should sync its clock with parent's timestamp
    let child_clock = LamportClock::new();
    let child_synced = child_clock.sync(child_start_clock);

    // Child's first event should be after parent's last event
    assert!(child_synced > parent_timestamp);

    // Cleanup
    env::remove_var("TRACEPARENT");
    env::remove_var("RENACER_LOGICAL_CLOCK");
}

#[test]
#[serial]
fn test_exec_simulation() {
    // Simulate exec() scenario where new process inherits environment

    // Original process
    let original_trace_id = [0x11; 16];
    let original_span_id = [0x22; 8];
    let original_ctx = TraceContext {
        version: 0,
        trace_id: original_trace_id,
        parent_id: original_span_id,
        trace_flags: 1,
    };
    original_ctx.set_env();
    TraceContext::set_logical_clock_env(100);

    // New process after exec() reads environment
    let new_ctx = TraceContext::from_env().expect("test");
    let new_clock_start = TraceContext::logical_clock_from_env().expect("test");

    // Should maintain trace continuity
    assert_eq!(new_ctx.trace_id, original_trace_id);
    assert_eq!(new_clock_start, 100);

    // New process creates child span
    let new_clock = LamportClock::with_initial_value(new_clock_start);
    let first_event = new_clock.tick();
    assert_eq!(first_event, 101);

    // Cleanup
    env::remove_var("TRACEPARENT");
    env::remove_var("RENACER_LOGICAL_CLOCK");
}

#[test]
#[serial]
fn test_multi_hop_propagation() {
    // Scenario: Process A → Process B → Process C
    // Each process increments logical clock and propagates

    // Process A
    let trace_id = [0xaa; 16];
    let span_a = [0x01; 8];
    let ctx_a = TraceContext { version: 0, trace_id, parent_id: span_a, trace_flags: 1 };
    ctx_a.set_env();

    let clock_a = LamportClock::new();
    let ts_a = clock_a.tick();
    TraceContext::set_logical_clock_env(ts_a);

    // Process B reads A's context
    let ctx_b_in = TraceContext::from_env().expect("test");
    let ts_b_in = TraceContext::logical_clock_from_env().expect("test");
    assert_eq!(ctx_b_in.trace_id, trace_id);
    assert_eq!(ts_b_in, ts_a);

    // Process B does work and propagates
    let clock_b = LamportClock::with_initial_value(ts_b_in);
    let ts_b = clock_b.tick();
    let span_b = [0x02; 8];
    let ctx_b = TraceContext { version: 0, trace_id, parent_id: span_b, trace_flags: 1 };
    ctx_b.set_env();
    TraceContext::set_logical_clock_env(ts_b);

    // Process C reads B's context
    let ctx_c_in = TraceContext::from_env().expect("test");
    let ts_c_in = TraceContext::logical_clock_from_env().expect("test");
    assert_eq!(ctx_c_in.trace_id, trace_id);
    assert_eq!(ts_c_in, ts_b);
    assert!(ts_c_in > ts_a); // C's timestamp > A's timestamp

    // Cleanup
    env::remove_var("TRACEPARENT");
    env::remove_var("RENACER_LOGICAL_CLOCK");
}

#[test]
#[serial]
fn test_trace_context_round_trip() {
    // Test that trace context can be serialized and deserialized via env vars

    let original = TraceContext {
        version: 0,
        trace_id: [
            0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xba, 0xbe, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc,
            0xde, 0xf0,
        ],
        parent_id: [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88],
        trace_flags: 1,
    };

    // Serialize
    original.set_env();

    // Deserialize
    let retrieved = TraceContext::from_env().expect("test");

    // Verify
    assert_eq!(retrieved, original);

    // Cleanup
    env::remove_var("TRACEPARENT");
}

#[test]
#[serial]
fn test_logical_clock_sync_across_processes() {
    // Scenario: Two concurrent processes sync their clocks

    // Process 1
    let clock1 = LamportClock::new();
    clock1.tick();
    clock1.tick();
    let ts1 = clock1.tick(); // ts1 = 3
    TraceContext::set_logical_clock_env(ts1);

    // Process 2 starts independently
    let clock2 = LamportClock::new();
    clock2.tick();
    clock2.tick();
    clock2.tick();
    clock2.tick();
    clock2.tick(); // ts2 = 5 (ahead of process 1)

    // Process 2 receives message from Process 1
    let ts1_received = TraceContext::logical_clock_from_env().expect("test");
    let ts2_synced = clock2.sync(ts1_received);

    // Process 2's clock should be max(5, 3) + 1 = 6
    assert_eq!(ts2_synced, 6);

    // Cleanup
    env::remove_var("RENACER_LOGICAL_CLOCK");
}

#[test]
#[serial]
fn test_trace_sampling_propagation() {
    // Test that sampling decision propagates

    // Sampled trace
    let sampled_ctx = TraceContext {
        version: 0,
        trace_id: [0x01; 16],
        parent_id: [0x02; 8],
        trace_flags: 0x01, // Sampled
    };
    sampled_ctx.set_env();

    let retrieved_sampled = TraceContext::from_env().expect("test");
    assert!(retrieved_sampled.is_sampled());

    // Unsampled trace
    let unsampled_ctx = TraceContext {
        version: 0,
        trace_id: [0x03; 16],
        parent_id: [0x04; 8],
        trace_flags: 0x00, // Not sampled
    };
    unsampled_ctx.set_env();

    let retrieved_unsampled = TraceContext::from_env().expect("test");
    assert!(!retrieved_unsampled.is_sampled());

    // Cleanup
    env::remove_var("TRACEPARENT");
}

#[test]
#[serial]
fn test_concurrent_env_var_access() {
    // Test thread safety of environment variable access

    use std::thread;

    let handles: Vec<_> = (0..10)
        .map(|i| {
            thread::spawn(move || {
                // Each thread sets its own logical clock
                TraceContext::set_logical_clock_env(i * 10);

                // Read it back (may race with other threads)
                let _ = TraceContext::logical_clock_from_env();
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("test");
    }

    // Cleanup
    env::remove_var("RENACER_LOGICAL_CLOCK");
}

#[test]
#[serial]
fn test_zero_logical_clock() {
    // Test edge case: logical clock = 0

    TraceContext::set_logical_clock_env(0);
    let retrieved = TraceContext::logical_clock_from_env().expect("test");
    assert_eq!(retrieved, 0);

    env::remove_var("RENACER_LOGICAL_CLOCK");
}

#[test]
#[serial]
fn test_batuta_integration_scenario() {
    // Scenario: Batuta transpiles Python → Rust, both should have same trace ID

    // Python execution
    let python_trace_id = [0xba; 16]; // Batuta trace
    let python_span_id = [0x01; 8];
    let python_ctx = TraceContext {
        version: 0,
        trace_id: python_trace_id,
        parent_id: python_span_id,
        trace_flags: 1,
    };
    python_ctx.set_env();

    let python_clock = LamportClock::new();
    let python_ts = python_clock.tick();
    TraceContext::set_logical_clock_env(python_ts);

    // Rust execution (transpiled) inherits context
    let rust_ctx = TraceContext::from_env().expect("test");
    let rust_clock_start = TraceContext::logical_clock_from_env().expect("test");

    // Both should have same trace ID (golden thread)
    assert_eq!(rust_ctx.trace_id, python_trace_id);

    // Rust's clock should sync with Python's
    let rust_clock = LamportClock::with_initial_value(rust_clock_start);
    let rust_ts = rust_clock.tick();
    assert!(rust_ts > python_ts); // Rust events happen after Python

    // Cleanup
    env::remove_var("TRACEPARENT");
    env::remove_var("RENACER_LOGICAL_CLOCK");
}
