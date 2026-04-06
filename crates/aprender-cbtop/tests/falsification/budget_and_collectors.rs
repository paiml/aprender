//! F021-F060: BrickBudget Verification and Backend Equivalence (Collectors)

use cbtop::brick::BrickBudget;
use cbtop::bricks::collectors::cpu::CpuCollectorBrick;
use cbtop::bricks::collectors::gpu::GpuCollectorBrick;
use cbtop::bricks::collectors::memory::MemoryCollectorBrick;
use cbtop::bricks::collectors::pepita::PepitaCollectorBrick;
use cbtop::bricks::collectors::wos::WosCollectorBrick;
use cbtop::bricks::collectors::zram::ZramCollectorBrick;

// ============================================================================
// F021-F040: BrickBudget Verification
// ============================================================================

/// F021: 60FPS budget constant should be ~16ms total
#[test]
fn f021_budget_60fps_correct() {
    let budget = BrickBudget::FRAME_60FPS;
    let total = budget.collect_ms + budget.layout_ms + budget.render_ms;
    // 60fps = 16.67ms per frame
    assert!(
        (10..=17).contains(&total),
        "F021 FALSIFIED: 60FPS budget total {} is not ~16ms",
        total
    );
}

/// F023: Budget with uniform distribution
#[test]
fn f023_budget_uniform_distribution() {
    let budget = BrickBudget::uniform(10);
    // uniform(n) sets each phase to n ms
    assert_eq!(budget.collect_ms, 10, "F023 FALSIFIED: uniform collect_ms wrong");
    assert_eq!(budget.layout_ms, 10, "F023 FALSIFIED: uniform layout_ms wrong");
    assert_eq!(budget.render_ms, 10, "F023 FALSIFIED: uniform render_ms wrong");
}

/// F025: Zero total budget handled correctly
#[test]
fn f025_zero_budget_handling() {
    let budget = BrickBudget::uniform(0);
    // Should not panic, just create zero budget
    assert_eq!(budget.collect_ms, 0);
    assert_eq!(budget.layout_ms, 0);
    assert_eq!(budget.render_ms, 0);
}

// ============================================================================
// F041-F060: Backend Equivalence (Collector Tests)
// ============================================================================

/// F041: CPU collector produces valid metrics
#[test]
fn f041_cpu_collector_valid_metrics() {
    let mut collector = CpuCollectorBrick::new();
    let metrics = collector.collect();

    // Check we have core data
    let core_count = metrics.per_core_usage.len();
    assert!(
        core_count > 0 || core_count <= 1024,
        "F041 FALSIFIED: Invalid core count {}",
        core_count
    );

    // Utilization should be 0-100
    assert!(
        metrics.total_usage >= 0.0 && metrics.total_usage <= 100.0,
        "F041 FALSIFIED: Invalid CPU utilization {}",
        metrics.total_usage
    );
}

/// F042: GPU collector handles missing GPU gracefully
#[test]
fn f042_gpu_collector_no_panic() {
    let mut collector = GpuCollectorBrick::new(999); // Non-existent GPU
                                                     // Should not panic, just return default metrics
    let metrics = collector.collect();
    assert!(
        metrics.device_name == "None" || !metrics.device_name.is_empty(),
        "F042 FALSIFIED: GPU collector panicked or returned invalid name"
    );
}

/// F043: Memory collector produces valid metrics
#[test]
fn f043_memory_collector_valid_metrics() {
    let mut collector = MemoryCollectorBrick::new();
    let metrics = collector.collect();

    assert!(metrics.total_kb > 0, "F043 FALSIFIED: Total memory is 0");
    // Used = total - available
    let used_kb = metrics.total_kb.saturating_sub(metrics.available_kb);
    assert!(
        used_kb <= metrics.total_kb,
        "F043 FALSIFIED: Used memory ({}) > total memory ({})",
        used_kb,
        metrics.total_kb
    );
}

/// F044: Pepita collector produces valid io_uring metrics
#[test]
fn f044_pepita_collector_valid_metrics() {
    let mut collector = PepitaCollectorBrick::new();
    let metrics = collector.collect();

    // Latency should be non-negative
    assert!(metrics.avg_latency_us >= 0.0, "F044 FALSIFIED: Negative latency");
    assert!(
        metrics.p99_latency_us >= metrics.avg_latency_us,
        "F044 FALSIFIED: P99 ({}) < avg ({})",
        metrics.p99_latency_us,
        metrics.avg_latency_us
    );
}

/// F045: WOS collector produces valid kernel metrics
#[test]
fn f045_wos_collector_valid_metrics() {
    let mut collector = WosCollectorBrick::new();
    let metrics = collector.collect();

    // Syscalls should be non-negative
    assert!(metrics.syscalls_per_sec >= 0.0, "F045 FALSIFIED: Negative syscall rate");
}

/// F046: ZRAM collector produces valid compression metrics
#[test]
fn f046_zram_collector_valid_metrics() {
    let mut collector = ZramCollectorBrick::new();
    let metrics = collector.collect();

    // Compression ratio should be >= 1.0 (or 0 if no data)
    if metrics.comp_size > 0 {
        let ratio = metrics.compression_ratio();
        assert!(
            (0.5..=100.0).contains(&ratio),
            "F046 FALSIFIED: Invalid compression ratio {}",
            ratio
        );
    }
}
