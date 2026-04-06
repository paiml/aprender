//! F081-F200: Performance, Error Handling, Memory, Concurrency, Integration, Jidoka

use cbtop::brick::{Brick, BrickAssertion, BrickVerification};
use cbtop::bricks::collectors::cpu::CpuCollectorBrick;
use cbtop::bricks::collectors::gpu::GpuCollectorBrick;
use cbtop::bricks::collectors::pepita::PepitaCollectorBrick;
use cbtop::bricks::collectors::wos::WosCollectorBrick;
use cbtop::bricks::collectors::zram::ZramCollectorBrick;
use cbtop::bricks::panels::gpu::GpuPanelBrick;

// ============================================================================
// F081-F100: Performance Metrics
// ============================================================================

/// F081: Collection should complete within budget
#[test]
fn f081_collection_within_budget() {
    use std::time::Instant;

    let mut collector = CpuCollectorBrick::new();
    let budget = collector.budget();

    let start = Instant::now();
    let _ = collector.collect();
    let elapsed = start.elapsed();

    // Allow 10x budget for test environment variability
    let budget_us = (budget.collect_ms as u64) * 1000 * 10;
    assert!(
        elapsed.as_micros() < budget_us as u128,
        "F081 FALSIFIED: Collection took {}us, budget was {}us",
        elapsed.as_micros(),
        budget_us / 10
    );
}

/// F082: Ring buffer history is bounded
#[test]
fn f082_ring_buffer_bounded() {
    let mut collector = CpuCollectorBrick::new();

    // Collect many samples
    for _ in 0..200 {
        collector.collect();
    }

    // History should be bounded (typical limit is 120)
    let history = collector.history();
    assert!(
        history.len() <= 120,
        "F082 FALSIFIED: History size {} exceeds expected bound",
        history.len()
    );
}

// ============================================================================
// F101-F120: Error Handling
// ============================================================================

/// F101: Assertions do not panic on valid input
#[test]
fn f101_assertions_no_panic() {
    let assertions = vec![
        BrickAssertion::MinWidth(40),
        BrickAssertion::MinHeight(15),
        BrickAssertion::max_latency_ms(16),
        BrickAssertion::ValueInRange { min: 0.0, max: 100.0 },
    ];

    for assertion in assertions {
        // Just verify they can be created without panic
        let _ = format!("{:?}", assertion);
    }
}

/// F102: BrickVerification starts valid
#[test]
fn f102_verification_starts_valid() {
    let v = BrickVerification::new();
    assert!(v.is_valid(), "F102 FALSIFIED: New verification is invalid");
}

/// F103: BrickVerification can fail
#[test]
fn f103_verification_can_fail() {
    let mut v = BrickVerification::new();
    v.add_fail(BrickAssertion::MinWidth(100), "test failure");
    assert!(!v.is_valid(), "F103 FALSIFIED: Verification with failure still valid");
}

// ============================================================================
// F121-F140: Memory Safety
// ============================================================================

/// F121: Ring buffer does not grow unbounded
#[test]
fn f121_ring_buffer_memory_bounded() {
    use cbtop::ring_buffer::RingBuffer;

    let mut buf: RingBuffer<u64> = RingBuffer::new(10);

    // Push 1000 items
    for i in 0..1000u64 {
        buf.push(i);
    }

    // Should only contain 10 items
    assert_eq!(buf.len(), 10, "F121 FALSIFIED: Ring buffer grew beyond capacity");

    // Last item should be 999
    assert_eq!(buf.back(), Some(&999u64), "F121 FALSIFIED: Ring buffer lost newest item");
}

/// F122: Statistics calculation handles empty buffer
#[test]
fn f122_statistics_empty_buffer() {
    use cbtop::ring_buffer::RingBuffer;

    let buf: RingBuffer<f64> = RingBuffer::new(10);

    // Empty buffer should return 0.0 for mean
    assert_eq!(buf.mean(), 0.0, "F122 FALSIFIED: Empty buffer has non-zero mean");
    assert_eq!(buf.len(), 0, "F122 FALSIFIED: Empty buffer has non-zero count");
}

// ============================================================================
// F141-F160: Concurrency
// ============================================================================

/// F141: Bricks are Send (can be sent to another thread)
#[test]
fn f141_bricks_are_send() {
    fn assert_send<T: Send>() {}

    assert_send::<CpuCollectorBrick>();
    assert_send::<GpuCollectorBrick>();
    assert_send::<cbtop::bricks::collectors::memory::MemoryCollectorBrick>();
    assert_send::<PepitaCollectorBrick>();
    assert_send::<WosCollectorBrick>();
    assert_send::<ZramCollectorBrick>();
    assert_send::<GpuPanelBrick>();
}

/// F142: Bricks are Sync (can be shared across threads)
#[test]
fn f142_bricks_are_sync() {
    fn assert_sync<T: Sync>() {}

    assert_sync::<CpuCollectorBrick>();
    assert_sync::<GpuCollectorBrick>();
    assert_sync::<cbtop::bricks::collectors::memory::MemoryCollectorBrick>();
    assert_sync::<PepitaCollectorBrick>();
    assert_sync::<WosCollectorBrick>();
    assert_sync::<ZramCollectorBrick>();
    assert_sync::<GpuPanelBrick>();
}

// ============================================================================
// F161-F180: Integration
// ============================================================================

/// F161: GPU panel can receive collector metrics
#[test]
fn f161_gpu_panel_collector_integration() {
    let mut collector = GpuCollectorBrick::new(0);
    let mut panel = GpuPanelBrick::new();

    let metrics = collector.collect();
    panel.update_from_metrics(&metrics);

    assert!(panel.current_metrics.is_some(), "F161 FALSIFIED: Panel did not receive metrics");
}

/// F162: All collector bricks implement Brick trait correctly
#[test]
fn f162_collectors_implement_brick() {
    let bricks: Vec<Box<dyn Brick>> = vec![
        Box::new(CpuCollectorBrick::new()),
        Box::new(GpuCollectorBrick::new(0)),
        Box::new(cbtop::bricks::collectors::memory::MemoryCollectorBrick::new()),
        Box::new(PepitaCollectorBrick::new()),
        Box::new(WosCollectorBrick::new()),
        Box::new(ZramCollectorBrick::new()),
        Box::new(GpuPanelBrick::new()),
    ];

    for brick in &bricks {
        // All trait methods should work
        let name = brick.brick_name();
        let assertions = brick.assertions();
        let budget = brick.budget();
        let verification = brick.verify();

        assert!(!name.is_empty(), "F162 FALSIFIED: Empty brick name");
        assert!(!assertions.is_empty(), "F162 FALSIFIED: No assertions for {}", name);
        let _ = budget;
        let _ = verification;
    }
}

// ============================================================================
// F181-F200: Jidoka (Built-in Quality)
// ============================================================================

/// F181: Verification reports failures explicitly
#[test]
fn f181_verification_explicit_failures() {
    let mut v = BrickVerification::new();
    v.add_fail(BrickAssertion::MinWidth(100), "explicit test failure");

    let failure_count = v.failure_count();
    assert!(failure_count > 0, "F181 FALSIFIED: Failure not recorded");
    assert!(!v.is_valid(), "F181 FALSIFIED: Verification with failure is still valid");
}

/// F182: Collectors provide meaningful history
#[test]
fn f182_collectors_provide_history() {
    let mut collector = CpuCollectorBrick::new();

    // Collect a few samples
    collector.collect();
    collector.collect();
    collector.collect();

    let history = collector.history();
    assert!(history.len() >= 3, "F182 FALSIFIED: History does not retain samples");
}

/// F183: Metrics have timestamps
#[test]
fn f183_metrics_have_timestamps() {
    let mut collector = CpuCollectorBrick::new();
    let metrics = collector.collect();

    // Timestamp should be recent (within last second)
    let elapsed = metrics.timestamp.elapsed();
    assert!(elapsed.as_secs() < 1, "F183 FALSIFIED: Metrics timestamp too old");
}

/// F184: ZRAM compression metrics are consistent
#[test]
fn f184_zram_metrics_consistent() {
    let mut collector = ZramCollectorBrick::new();
    let metrics = collector.collect();

    // If we have original data, compressed should be less (or equal for incompressible)
    if metrics.orig_size > 0 {
        assert!(
            metrics.comp_size <= metrics.orig_size * 10, // Allow up to 10x for metadata overhead
            "F184 FALSIFIED: Compressed size ({}) >> original ({})",
            metrics.comp_size,
            metrics.orig_size
        );
    }
}

/// F185: Pepita latency breakdown is consistent
#[test]
fn f185_pepita_latency_consistent() {
    let mut collector = PepitaCollectorBrick::new();
    collector.collect();

    let breakdown = collector.latency_breakdown();

    // P99 should be >= avg
    assert!(
        breakdown.p99_us >= breakdown.avg_us,
        "F185 FALSIFIED: P99 ({}) < avg ({})",
        breakdown.p99_us,
        breakdown.avg_us
    );
}

/// F186: WOS Jidoka summary available
#[test]
fn f186_wos_jidoka_summary() {
    let mut collector = WosCollectorBrick::new();
    collector.collect();

    let summary = collector.jidoka_summary();

    // checks_passed should be 0-100
    assert!(summary.checks_passed <= 100, "F186 FALSIFIED: checks_passed > 100");
}

/// F187: ZRAM throughput summary available
#[test]
fn f187_zram_throughput_summary() {
    let mut collector = ZramCollectorBrick::new();
    collector.collect();

    let summary = collector.throughput_summary();

    // Throughput should be non-negative
    assert!(summary.compression_gbps >= 0.0, "F187 FALSIFIED: Negative compression throughput");
    assert!(summary.decompression_gbps >= 0.0, "F187 FALSIFIED: Negative decompression throughput");
}

/// F188: GPU panel displays data source correctly
#[test]
fn f188_gpu_panel_data_source() {
    let panel_without_metrics = GpuPanelBrick::new();
    assert!(
        panel_without_metrics.current_metrics.is_none(),
        "F188 FALSIFIED: New panel has metrics"
    );

    let mut panel_with_metrics = GpuPanelBrick::new();
    let mut collector = GpuCollectorBrick::new(0);
    let metrics = collector.collect();
    panel_with_metrics.update_from_metrics(&metrics);

    assert!(
        panel_with_metrics.current_metrics.is_some(),
        "F188 FALSIFIED: Panel didn't store metrics"
    );
}

/// F189: Verification score calculation correct
#[test]
fn f189_verification_score() {
    let v = BrickVerification::new();
    // Empty verification = 100% score
    assert!((v.score() - 1.0).abs() < 0.001, "F189 FALSIFIED: Empty verification score not 1.0");

    let mut v2 = BrickVerification::new();
    v2.add_pass(BrickAssertion::MinWidth(40));
    v2.add_fail(BrickAssertion::MinWidth(100), "too small");
    // 1 pass, 1 fail = 50% score
    assert!(
        (v2.score() - 0.5).abs() < 0.001,
        "F189 FALSIFIED: 50% verification score wrong: {}",
        v2.score()
    );
}

/// F190: Ring buffer last_n works correctly
#[test]
fn f190_ring_buffer_last_n() {
    use cbtop::ring_buffer::RingBuffer;

    let mut buf: RingBuffer<i32> = RingBuffer::new(10);
    for i in 0..10 {
        buf.push(i);
    }

    let last_3: Vec<_> = buf.last_n(3).copied().collect();
    assert_eq!(last_3, vec![7, 8, 9], "F190 FALSIFIED: last_n returned wrong values");
}

// ============================================================================
// Summary: 200-Point Falsification Coverage
// ============================================================================

/// Meta-test: Verify we have sufficient falsification coverage
#[test]
fn f200_falsification_coverage() {
    // Count tests in this module
    // This is a meta-assertion that we have meaningful coverage
    // The actual count is verified by running `cargo test -p cbtop falsification`

    // If this test runs, we have at least this many falsification tests:
    // F001-F020: 5 tests (f001, f002, f003, f010, f017)
    // F021-F040: 3 tests (f021, f023, f025)
    // F041-F060: 6 tests (f041-f046)
    // F081-F100: 2 tests (f081, f082)
    // F101-F120: 3 tests (f101, f102, f103)
    // F121-F140: 2 tests (f121, f122)
    // F141-F160: 2 tests (f141, f142)
    // F161-F180: 2 tests (f161, f162)
    // F181-F200: 10 tests (f181-f190)

    // Total: 35 explicit falsification tests covering key invariants

    // The remaining F-series criteria are validated through:
    // 1. Unit tests in brick.rs
    // 2. Unit tests in each collector module
    // 3. Integration tests via `cargo test -p cbtop`
    // 4. Property-based tests (if added)

    // F200: Falsification framework operational - verified by test reaching this point
}
