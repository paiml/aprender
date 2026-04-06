//! Brick Tracing Demo
//!
//! This example demonstrates how to use renacer's ComputeBrick tracing
//! to detect performance anomalies and get syscall-level visibility.
//!
#![allow(clippy::unwrap_used)]
//! # Background
//!
//! The BrickTracer integrates with trueno's cbtop to provide escalation
//! from measurement to tracing when performance problems are detected:
//!
//! 1. cbtop measures ComputeBrick performance (CV%, efficiency)
//! 2. When CV > 15% or efficiency < 25%, escalate to renacer
//! 3. renacer captures syscall breakdown for root cause analysis
//! 4. Export spans to Jaeger/Tempo for visualization
//!
//! # Run
//!
//! ```bash
//! cargo run --example brick_trace_demo
//! ```

use renacer::brick_tracer::{BrickEscalationThresholds, BrickTracer, EscalationReason};
use std::time::Duration;

/// Simulates a ComputeBrick that performs some work
fn simulate_compute_brick(iterations: u64) -> u64 {
    let mut result = 0u64;
    for i in 0..iterations {
        result = result.wrapping_add(i);
    }
    result
}

/// Simulates an I/O-bound brick with thread sleeps
fn simulate_io_brick(sleep_us: u64) -> String {
    std::thread::sleep(Duration::from_micros(sleep_us));
    "io_complete".to_string()
}

fn main() {
    println!("=== Renacer Brick Tracing Demo ===\n");

    // Create a local tracer (no OTLP export)
    let tracer = BrickTracer::new_local();

    println!("1. Testing escalation thresholds\n");
    println!("   Default thresholds: CV > 15%, efficiency < 25%");
    println!("   Should trace (CV=20%, eff=80%): {}", tracer.should_trace(20.0, 80.0));
    println!("   Should trace (CV=5%, eff=10%): {}", tracer.should_trace(5.0, 10.0));
    println!("   Should trace (CV=5%, eff=80%): {}", tracer.should_trace(5.0, 80.0));
    println!();

    // Trace a compute-bound brick
    println!("2. Tracing compute-bound brick\n");
    let result = tracer.trace("MatMulBrick", 1000, || simulate_compute_brick(1_000_000));
    let meta = result.metadata.as_ref().unwrap();
    println!("   Result: {}", result.result);
    println!("   Duration: {} us", result.duration_us);
    println!("   Budget: {} us", meta.budget_us);
    println!("   Efficiency: {:.2}%", meta.efficiency * 100.0);
    println!("   Over budget: {}", meta.over_budget);
    println!("   Dominant syscall: {}", result.syscall_breakdown.dominant_syscall());
    println!();

    // Trace an I/O-bound brick
    println!("3. Tracing I/O-bound brick\n");
    let result = tracer.trace("DiskReadBrick", 500, || simulate_io_brick(1000));
    let meta = result.metadata.as_ref().unwrap();
    println!("   Result: {}", result.result);
    println!("   Duration: {} us", result.duration_us);
    println!("   Budget: {} us", meta.budget_us);
    println!("   Efficiency: {:.2}%", meta.efficiency * 100.0);
    println!("   Over budget: {}", meta.over_budget);
    println!();

    // Test escalation reasons
    println!("4. Escalation reason detection\n");
    let reason = tracer.escalation_reason(20.0, 10.0);
    println!("   CV=20%, eff=10% -> {}", reason);
    let reason = tracer.escalation_reason(5.0, 10.0);
    println!("   CV=5%, eff=10% -> {}", reason);
    let reason = tracer.escalation_reason(20.0, 80.0);
    println!("   CV=20%, eff=80% -> {}", reason);
    println!();

    // Custom thresholds
    println!("5. Custom thresholds\n");
    let thresholds = BrickEscalationThresholds::default()
        .with_cv(10.0) // Stricter CV threshold
        .with_efficiency(50.0) // Higher efficiency requirement
        .with_rate_limit(10); // Lower rate limit
    let tracer = BrickTracer::new_local().with_thresholds(thresholds);
    println!("   Custom: CV > 10%, efficiency < 50%");
    println!("   Should trace (CV=12%, eff=80%): {}", tracer.should_trace(12.0, 80.0));
    println!("   Should trace (CV=5%, eff=40%): {}", tracer.should_trace(5.0, 40.0));
    println!();

    // Trace with explicit reason
    println!("6. Trace with explicit escalation reason\n");
    let result = tracer.trace_with_reason("DebugBrick", 10000, EscalationReason::Manual, || {
        simulate_compute_brick(100)
    });
    println!("   Reason: {:?}", result.escalation_reason);
    println!();

    println!("=== Demo Complete ===\n");
    println!("For OTLP export to Jaeger/Tempo, use:");
    println!("  let tracer = BrickTracer::new(\"http://localhost:4317\")?;");
}
