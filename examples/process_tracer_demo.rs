//! Process Tracer Demo - ptop/presentar Integration Example
//!
//! This example demonstrates the process_tracer module for deep tracing
//! integration with ptop (process top) visualization.
//!
//! # Usage
//!
//! ```bash
//! # Run with sudo for ptrace access
//! sudo cargo run --example process_tracer_demo
//!
//! # Or trace a specific PID
//! sudo cargo run --example process_tracer_demo -- 12345
//! ```
//!
//! # Features Demonstrated
//!
//! - ProcessTraceConfig builder pattern
//! - SyscallBreakdown with category buckets
//! - Z-score anomaly detection
//! - OTLP span export format
//!
//! # Reference
//!
//! SPEC-057: ptop Deep Tracing Integration
//! docs/specifications/ptop-presentar-tracing-support.md

use renacer::process_tracer::{
    compute_baseline, is_available, syscall_name, zscore, ProcessTraceConfig, SyscallBreakdown,
    SyscallEvent, TraceResult,
};
use std::time::Duration;

fn main() {
    println!("=== Renacer Process Tracer Demo ===\n");

    // Check if ptrace is available
    println!("1. Checking ptrace availability...");
    if is_available() {
        println!("   ✓ ptrace is available (running as root or CAP_SYS_PTRACE)");
    } else {
        println!("   ✗ ptrace not available (run with sudo for full demo)");
        println!("   Continuing with synthetic data demo...\n");
    }

    // Demonstrate configuration builder
    println!("2. Building ProcessTraceConfig...");
    let config = ProcessTraceConfig::default()
        .with_timeout(Duration::from_secs(5))
        .with_max_syscalls(1000)
        .with_anomaly_threshold(3.0)
        .with_source_correlation(true)
        .with_rate_limit(100);

    println!("   Config: timeout=5s, max_syscalls=1000, threshold=3.0σ");
    println!("   Rate limit: {} traces/sec", 100);

    // Create synthetic syscall events for demonstration
    println!("\n3. Creating synthetic syscall events...");
    let events = vec![
        SyscallEvent::new("read".to_string(), 0, Duration::from_micros(50), 1024),
        SyscallEvent::new("read".to_string(), 0, Duration::from_micros(45), 512),
        SyscallEvent::new("read".to_string(), 0, Duration::from_micros(55), 2048),
        SyscallEvent::new("write".to_string(), 1, Duration::from_micros(100), 1024),
        SyscallEvent::new("write".to_string(), 1, Duration::from_micros(95), 512),
        SyscallEvent::new("mmap".to_string(), 9, Duration::from_micros(500), 0),
        SyscallEvent::new("mmap".to_string(), 9, Duration::from_micros(480), 0),
        SyscallEvent::new("futex".to_string(), 202, Duration::from_micros(1000), 0),
        SyscallEvent::new("futex".to_string(), 202, Duration::from_micros(950), 0),
        SyscallEvent::new("ioctl".to_string(), 16, Duration::from_micros(200), 0),
        // Anomalous syscall (very slow read)
        SyscallEvent::new("read".to_string(), 0, Duration::from_micros(5000), 1024),
    ];

    println!("   Created {} synthetic events", events.len());

    // Compute baseline from events
    println!("\n4. Computing syscall baseline...");
    let baseline = compute_baseline(&events);
    println!("   Baseline categories:");
    for (cat, mean) in baseline.mean_us.iter() {
        let std = baseline.std_us.get(cat).unwrap_or(&0.0);
        println!("     {}: mean={:.1}μs, std={:.1}μs", cat, mean, std);
    }

    // Build syscall breakdown
    println!("\n5. Building SyscallBreakdown...");
    let total_time_us = 10000; // 10ms total trace time
    let breakdown = SyscallBreakdown::from_events(&events, total_time_us);

    println!("   Category breakdown (total {}μs):", total_time_us);
    println!(
        "     mmap:    {}μs ({:.1}%)",
        breakdown.mmap_us,
        breakdown.mmap_us as f64 / total_time_us as f64 * 100.0
    );
    println!(
        "     futex:   {}μs ({:.1}%)",
        breakdown.futex_us,
        breakdown.futex_us as f64 / total_time_us as f64 * 100.0
    );
    println!(
        "     ioctl:   {}μs ({:.1}%)",
        breakdown.ioctl_us,
        breakdown.ioctl_us as f64 / total_time_us as f64 * 100.0
    );
    println!(
        "     read:    {}μs ({:.1}%)",
        breakdown.read_us,
        breakdown.read_us as f64 / total_time_us as f64 * 100.0
    );
    println!(
        "     write:   {}μs ({:.1}%)",
        breakdown.write_us,
        breakdown.write_us as f64 / total_time_us as f64 * 100.0
    );
    println!(
        "     other:   {}μs ({:.1}%)",
        breakdown.other_us,
        breakdown.other_us as f64 / total_time_us as f64 * 100.0
    );
    println!(
        "     compute: {}μs ({:.1}%)",
        breakdown.compute_us,
        breakdown.compute_us as f64 / total_time_us as f64 * 100.0
    );
    println!("   Syscall count: {}", breakdown.syscall_count);
    println!("   Efficiency: {:.1}%", breakdown.efficiency() * 100.0);

    // Detect anomalies using z-score
    println!("\n6. Detecting anomalies (threshold: {:.1}σ)...", config.anomaly_threshold);
    for event in &events {
        let z = zscore(event, &baseline);
        if z.abs() > config.anomaly_threshold {
            println!(
                "   ⚠️  ANOMALY: {} took {}μs (z-score: {:.2}σ)",
                event.syscall,
                event.duration.as_micros(),
                z
            );
        }
    }

    // Create TraceResult with anomaly detection
    println!("\n7. Building TraceResult with anomaly detection...");
    let result = TraceResult::new(std::process::id(), Duration::from_micros(total_time_us), events)
        .with_baseline(&baseline, config.anomaly_threshold);

    println!("   PID: {}", result.pid);
    println!("   Duration: {}μs", result.duration.as_micros());
    println!("   Events: {}", result.events.len());
    println!("   Anomalies: {}", result.anomalies.len());
    println!("   Max Z-score: {:.2}σ", result.max_zscore);

    // Show anomalies
    for anomaly in &result.anomalies {
        println!(
            "   → {} at {}μs (z={:.2}σ, expected={:.1}μs)",
            anomaly.syscall, anomaly.duration_us, anomaly.zscore, anomaly.expected_us
        );
    }

    // Demonstrate OTLP span export
    println!("\n8. OTLP span export format...");
    let span = result.to_otlp_span();
    println!("   Span name: {}", span.name);
    println!("   Trace ID: {:?}", span.trace_id);
    println!("   Span ID: {:?}", span.span_id);
    println!("   Attributes:");
    for attr in &span.attributes {
        println!("     {}: {:?}", attr.key, attr.value);
    }

    // Demonstrate syscall name lookup
    println!("\n9. Syscall number to name mapping...");
    let syscall_nums = [0, 1, 2, 3, 9, 16, 202, 60];
    for nr in syscall_nums {
        println!("   {} → {}", nr, syscall_name(nr));
    }

    println!("\n=== Demo Complete ===");
    println!("\nFor live tracing, run with sudo and a target PID:");
    println!("  sudo cargo run --example process_tracer_demo -- <PID>");
}
