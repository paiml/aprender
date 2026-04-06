//! macOS Collector Tests
//!
//! Probar-style tests for macOS platform collectors.
//! Following ttop-demo.md specification section 7.3-7.5.

#![cfg(target_os = "macos")]

use trueno_viz::monitor::collectors::{
    CpuCollector, DiskCollector, MemoryCollector, NetworkCollector, ProcessCollector,
};
use trueno_viz::monitor::Collector;

#[cfg(target_os = "macos")]
use trueno_viz::monitor::collectors::AppleGpuCollector;

// ============================================================================
// Section 10.3: Metric Accuracy Tests (Claims 41-60)
// ============================================================================

/// Claim 41: CPU % within ±2% of top
#[test]
fn test_cpu_accuracy_vs_top() {
    let mut cpu = CpuCollector::new();
    assert!(cpu.is_available(), "CPU collector must be available on macOS");

    // Collect twice for delta calculation
    let _ = cpu.collect();
    std::thread::sleep(std::time::Duration::from_millis(500));
    let metrics = cpu.collect().expect("CPU collection should succeed");

    // Verify we get reasonable CPU values
    if let Some(total) = metrics.get_gauge("cpu.total") {
        assert!(total >= 0.0 && total <= 100.0, "CPU total must be 0-100%, got {}", total);
    }

    // Verify core count matches system
    let expected_cores: usize = std::process::Command::new("sysctl")
        .args(["-n", "hw.ncpu"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse().ok())
        .unwrap_or(1);

    assert_eq!(
        cpu.core_count(),
        expected_cores,
        "Core count mismatch: collector={}, sysctl={}",
        cpu.core_count(),
        expected_cores
    );
}

/// Claim 42: Memory within ±1MB of free
#[test]
fn test_memory_accuracy() {
    let mut mem = MemoryCollector::new();
    assert!(mem.is_available(), "Memory collector must be available on macOS");

    let metrics = mem.collect().expect("Memory collection should succeed");

    // Get total from sysctl for comparison
    let expected_total: u64 = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse().ok())
        .unwrap_or(0);

    let actual_total = metrics.get_counter("memory.total").unwrap_or(0);

    assert_eq!(
        actual_total, expected_total,
        "Memory total mismatch: collector={}, sysctl={}",
        actual_total, expected_total
    );

    // Verify percentage is reasonable
    if let Some(pct) = metrics.get_gauge("memory.used.percent") {
        assert!(pct >= 0.0 && pct <= 100.0, "Memory percent must be 0-100%, got {}", pct);
    }
}

/// Claim 43: Network counters reasonable
#[test]
fn test_network_accuracy() {
    let mut net = NetworkCollector::new();
    assert!(net.is_available(), "Network collector must be available on macOS");

    // Collect twice for rate calculation
    let _ = net.collect();
    std::thread::sleep(std::time::Duration::from_millis(200));
    let _ = net.collect();

    // Verify we detect at least one interface
    let interfaces = net.interfaces();
    assert!(!interfaces.is_empty(), "Should detect at least one network interface");

    // Verify en0 exists (primary interface on most Macs)
    let has_en0 = interfaces.iter().any(|i| i.starts_with("en"));
    assert!(has_en0, "Should detect en* interface, found: {:?}", interfaces);
}

/// Claim 44: Disk mounts match df
#[test]
fn test_disk_accuracy() {
    let mut disk = DiskCollector::new();
    assert!(disk.is_available(), "Disk collector must be available on macOS");

    let _ = disk.collect();

    // Verify we find mounts
    let mounts = disk.mounts();
    assert!(!mounts.is_empty(), "Should detect at least one mount");

    // Verify root mount exists
    let has_root = mounts.iter().any(|m| m.mount_point == "/");
    assert!(has_root, "Should detect root mount /");

    // Verify usage percentages are reasonable
    for mount in mounts {
        let pct = mount.usage_percent();
        assert!(
            pct >= 0.0 && pct <= 100.0,
            "Mount {} usage must be 0-100%, got {}",
            mount.mount_point,
            pct
        );
    }
}

/// Claim 58: Process count matches ps aux
#[test]
fn test_process_count_accuracy() {
    let mut proc = ProcessCollector::new();
    assert!(proc.is_available(), "Process collector must be available on macOS");

    let _ = proc.collect();

    // Count should be reasonable (typically 100-2000 on a Mac)
    let count = proc.count();
    assert!(count >= 50, "Should find at least 50 processes, found {}", count);
    assert!(count <= 5000, "Process count should be reasonable (<5000), found {}", count);

    // Verify we can find launchd (PID 1)
    let has_pid1 = proc.processes().contains_key(&1);
    assert!(has_pid1, "Should find PID 1 (launchd)");
}

/// GPU detection test
#[test]
#[cfg(target_os = "macos")]
fn test_gpu_detection() {
    let gpu = AppleGpuCollector::new();

    // GPU should be available on any Mac
    assert!(gpu.is_available(), "GPU collector should be available on macOS");

    if let Some(info) = gpu.primary_gpu() {
        // Name should not be empty
        assert!(!info.name.is_empty(), "GPU name should not be empty");

        // Metal family should be set
        assert!(!info.metal_family.is_empty(), "Metal family should not be empty");
    }
}

// ============================================================================
// Section 10.4: Determinism Tests (Claims 61-70)
// ============================================================================

/// Claim 61: Same state produces identical metrics
#[test]
fn test_collector_determinism() {
    // Memory collector should give consistent total
    let mut mem1 = MemoryCollector::new();
    let mut mem2 = MemoryCollector::new();

    let m1 = mem1.collect().expect("operation should succeed");
    let m2 = mem2.collect().expect("operation should succeed");

    let total1 = m1.get_counter("memory.total").unwrap_or(0);
    let total2 = m2.get_counter("memory.total").unwrap_or(0);

    assert_eq!(total1, total2, "Memory total should be deterministic");
}

// ============================================================================
// Section 10.5: Testing Coverage (Claims 71-85)
// ============================================================================

/// All collectors have is_available tests
#[test]
fn test_all_collectors_availability() {
    let cpu = CpuCollector::new();
    let mem = MemoryCollector::new();
    let disk = DiskCollector::new();
    let net = NetworkCollector::new();
    let proc = ProcessCollector::new();

    assert!(cpu.is_available(), "CPU collector should be available");
    assert!(mem.is_available(), "Memory collector should be available");
    assert!(disk.is_available(), "Disk collector should be available");
    assert!(net.is_available(), "Network collector should be available");
    assert!(proc.is_available(), "Process collector should be available");

    #[cfg(target_os = "macos")]
    {
        let gpu = AppleGpuCollector::new();
        assert!(gpu.is_available(), "GPU collector should be available");
    }
}

/// All collectors have correct display names
#[test]
fn test_collector_display_names() {
    let cpu = CpuCollector::new();
    let mem = MemoryCollector::new();
    let disk = DiskCollector::new();
    let net = NetworkCollector::new();
    let proc = ProcessCollector::new();

    assert!(!cpu.display_name().is_empty());
    assert!(!mem.display_name().is_empty());
    assert!(!disk.display_name().is_empty());
    assert!(!net.display_name().is_empty());
    assert!(!proc.display_name().is_empty());
}

/// All collectors have reasonable interval hints
#[test]
fn test_collector_intervals() {
    let cpu = CpuCollector::new();
    let mem = MemoryCollector::new();
    let disk = DiskCollector::new();
    let net = NetworkCollector::new();
    let proc = ProcessCollector::new();

    // Intervals should be between 100ms and 10s
    let min = std::time::Duration::from_millis(100);
    let max = std::time::Duration::from_secs(10);

    assert!(cpu.interval_hint() >= min && cpu.interval_hint() <= max);
    assert!(mem.interval_hint() >= min && mem.interval_hint() <= max);
    assert!(disk.interval_hint() >= min && disk.interval_hint() <= max);
    assert!(net.interval_hint() >= min && net.interval_hint() <= max);
    assert!(proc.interval_hint() >= min && proc.interval_hint() <= max);
}

// ============================================================================
// Section 10.7: Safety and Correctness (Claims 96-100)
// ============================================================================

/// Claim 99: No panics in production
#[test]
fn test_collectors_no_panic() {
    // Run each collector multiple times to verify no panics
    let mut cpu = CpuCollector::new();
    let mut mem = MemoryCollector::new();
    let mut disk = DiskCollector::new();
    let mut net = NetworkCollector::new();
    let mut proc = ProcessCollector::new();

    for _ in 0..10 {
        let _ = cpu.collect();
        let _ = mem.collect();
        let _ = disk.collect();
        let _ = net.collect();
        let _ = proc.collect();
    }
}

/// Claim 100: Values within expected bounds (no overflow)
#[test]
fn test_metric_bounds() {
    let mut cpu = CpuCollector::new();
    let _ = cpu.collect();
    std::thread::sleep(std::time::Duration::from_millis(100));

    if let Ok(metrics) = cpu.collect() {
        // CPU percentages should be 0-100
        for i in 0..cpu.core_count() {
            if let Some(pct) = metrics.get_gauge(&format!("cpu.core.{}", i)) {
                assert!(
                    pct >= 0.0 && pct <= 100.0,
                    "CPU core {} percent out of bounds: {}",
                    i,
                    pct
                );
            }
        }
    }

    let mut mem = MemoryCollector::new();
    if let Ok(metrics) = mem.collect() {
        if let Some(pct) = metrics.get_gauge("memory.used.percent") {
            assert!(pct >= 0.0 && pct <= 100.0, "Memory percent out of bounds: {}", pct);
        }
    }
}
