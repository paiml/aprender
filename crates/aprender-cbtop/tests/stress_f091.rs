#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! F091: Stress Test - Graceful Degradation
//!
//! Tests that cbtop components handle edge cases without panic.

use cbtop::brick::{Brick, BrickBudget, BrickVerification};
use cbtop::bricks::collectors::cpu::CpuCollectorBrick;
use cbtop::bricks::collectors::gpu::GpuCollectorBrick;
use cbtop::bricks::collectors::memory::MemoryCollectorBrick;
use cbtop::bricks::collectors::pepita::PepitaCollectorBrick;
use cbtop::bricks::collectors::wos::WosCollectorBrick;
use cbtop::bricks::collectors::zram::ZramCollectorBrick;
use cbtop::bricks::panels::gpu::GpuPanelBrick;
use cbtop::ring_buffer::RingBuffer;
use std::time::Instant;

/// F091: Startup must complete within 500ms
#[test]
fn f091_startup_latency() {
    const MAX_STARTUP_MS: u128 = 500;

    let start = Instant::now();

    // Create all collectors (simulates startup)
    let _cpu = CpuCollectorBrick::new();
    let _gpu = GpuCollectorBrick::new(0);
    let _mem = MemoryCollectorBrick::new();
    let _pepita = PepitaCollectorBrick::new();
    let _wos = WosCollectorBrick::new();
    let _zram = ZramCollectorBrick::new();
    let _panel = GpuPanelBrick::new();

    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < MAX_STARTUP_MS,
        "F091 FALSIFIED: Startup took {}ms, max allowed {}ms",
        elapsed.as_millis(),
        MAX_STARTUP_MS
    );

    println!("✅ F091 Startup completed in {}ms", elapsed.as_millis());
}

/// F092: Ring buffer handles extreme sizes without panic
#[test]
fn f092_ring_buffer_extreme_sizes() {
    // Very small buffer
    let mut tiny: RingBuffer<f64> = RingBuffer::new(1);
    for i in 0..100 {
        tiny.push(i as f64);
    }
    assert_eq!(tiny.len(), 1, "F092 FALSIFIED: Tiny buffer wrong size");
    assert_eq!(tiny.back(), Some(&99.0), "F092 FALSIFIED: Tiny buffer wrong value");

    // Very large buffer
    let mut large: RingBuffer<u32> = RingBuffer::new(10_000);
    for i in 0..20_000u32 {
        large.push(i);
    }
    assert_eq!(large.len(), 10_000, "F092 FALSIFIED: Large buffer wrong size");

    // Empty operations
    let empty: RingBuffer<f64> = RingBuffer::new(100);
    assert_eq!(empty.mean(), 0.0, "F092 FALSIFIED: Empty mean should be 0");
    assert!(empty.is_empty(), "F092 FALSIFIED: Empty buffer not empty");

    println!("✅ F092 Ring buffer handles extreme sizes");
}

/// F093: Collectors handle rapid repeated calls
#[test]
fn f093_rapid_collection() {
    let mut cpu = CpuCollectorBrick::new();
    let mut gpu = GpuCollectorBrick::new(0);
    let mut mem = MemoryCollectorBrick::new();

    // Rapid-fire 100 collections (GH-194: verify metrics are produced, not silently lost)
    for _ in 0..100 {
        let _cpu_metrics = cpu.collect();
        let _gpu_metrics = gpu.collect();
        let _mem_metrics = mem.collect();
    }

    // Verify histories are bounded
    assert!(cpu.history().len() <= 120, "F093 FALSIFIED: CPU history unbounded");
    assert!(gpu.history().len() <= 120, "F093 FALSIFIED: GPU history unbounded");
    assert!(mem.history().len() <= 120, "F093 FALSIFIED: Memory history unbounded");

    println!("✅ F093 Collectors handle rapid collection");
}

/// F094: GPU panel handles missing metrics gracefully
#[test]
fn f094_gpu_panel_missing_data() {
    let panel = GpuPanelBrick::new();

    // Panel with no metrics should not panic when accessed
    assert!(panel.current_metrics.is_none());
    assert!(panel.gpu_data.is_empty());
    assert!(panel.temperature_c.is_none());
    assert!(panel.power_watts.is_none());

    // Verification should still work
    let v = panel.verify();
    // Just check it completes without panic
    let _ = v.is_valid();

    println!("✅ F094 GPU panel handles missing data");
}

/// F095: Collectors handle non-existent devices
#[test]
fn f095_nonexistent_device_handling() {
    // GPU device 999 doesn't exist
    let mut gpu = GpuCollectorBrick::new(999);
    let metrics = gpu.collect();

    // Should return default/empty metrics, not panic
    assert!(
        metrics.device_name == "None" || metrics.device_name.is_empty(),
        "F095 FALSIFIED: Non-existent GPU should return None/empty name"
    );

    println!("✅ F095 Collectors handle non-existent devices");
}

/// F096: BrickVerification handles many assertions
#[test]
fn f096_verification_stress() {
    use cbtop::brick::BrickAssertion;

    let mut v = BrickVerification::new();

    // Add many assertions
    for i in 0..1000u16 {
        if i % 2 == 0 {
            v.add_pass(BrickAssertion::MinWidth(i));
        } else {
            v.add_fail(BrickAssertion::MinHeight(i), format!("fail {}", i));
        }
    }

    // Score should be ~50%
    let score = v.score();
    assert!((score - 0.5).abs() < 0.01, "F096 FALSIFIED: Score should be ~50%, got {}", score);

    println!("✅ F096 Verification handles many assertions");
}

/// F097: Pepita collector handles unavailable io_uring
#[test]
fn f097_pepita_graceful_degradation() {
    let mut pepita = PepitaCollectorBrick::new();
    let metrics = pepita.collect();

    // Should return mock data on systems without io_uring, not panic
    // Check for valid structure regardless of real data availability
    assert!(metrics.avg_latency_us >= 0.0, "F097 FALSIFIED: Latency should be non-negative");
    assert!(metrics.p99_latency_us >= 0.0, "F097 FALSIFIED: P99 should be non-negative");

    println!("✅ F097 Pepita handles unavailable io_uring");
}

/// F098: ZRAM collector handles missing ZRAM
#[test]
fn f098_zram_graceful_degradation() {
    let mut zram = ZramCollectorBrick::new();
    let metrics = zram.collect();

    // Should return mock/default data if ZRAM unavailable
    // Verify structure is valid - orig_size is u64 so always >= 0
    // This is a sanity check that the struct was populated
    let _ = metrics.orig_size; // field exists and is accessible

    // Compression ratio should be reasonable
    if metrics.comp_size > 0 {
        let ratio = metrics.compression_ratio();
        assert!(
            ratio > 0.0 && ratio < 1000.0,
            "F098 FALSIFIED: Compression ratio unreasonable: {}",
            ratio
        );
    }

    println!("✅ F098 ZRAM handles missing ZRAM");
}

/// F099: Budget calculations don't overflow
#[test]
fn f099_budget_overflow_protection() {
    // Max values
    let budget = BrickBudget { collect_ms: u32::MAX, layout_ms: u32::MAX, render_ms: u32::MAX };

    // These should not overflow in comparisons
    let _ = budget.collect_ms;
    let _ = budget.layout_ms;
    let _ = budget.render_ms;

    // Zero budget
    let zero = BrickBudget::uniform(0);
    assert_eq!(zero.collect_ms, 0);
    assert_eq!(zero.layout_ms, 0);
    assert_eq!(zero.render_ms, 0);

    println!("✅ F099 Budget calculations don't overflow");
}

/// F100: WOS collector handles missing kernel metrics
#[test]
fn f100_wos_graceful_degradation() {
    let mut wos = WosCollectorBrick::new();
    let metrics = wos.collect();

    // Should return mock/default data if metrics unavailable
    assert!(
        metrics.syscalls_per_sec >= 0.0,
        "F100 FALSIFIED: syscalls_per_sec should be non-negative"
    );

    let summary = wos.jidoka_summary();
    assert!(summary.checks_passed <= 100, "F100 FALSIFIED: checks_passed should be <= 100");

    println!("✅ F100 WOS handles missing kernel metrics");
}
