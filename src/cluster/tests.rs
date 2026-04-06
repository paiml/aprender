// Comprehensive tests for TOML-based syscall clustering
//
// Toyota Way Principle: Jidoka (Quality at the Source)
// - Unit tests validate configuration-driven clustering
// - Tests ensure Open-Closed Principle compliance
// - Edge cases prevent production defects

use super::*;
use anyhow::Result;
use std::io::Write;
use tempfile::NamedTempFile;

/// Helper: classify a syscall and return the cluster name (if any).
fn classify_name(registry: &ClusterRegistry, syscall: &str) -> Option<String> {
    let fds = FdTable::new();
    registry.classify(syscall, &[], &fds).map(|c| c.name.clone())
}

/// Test that default transpiler clusters load correctly from embedded TOML
#[test]
fn test_default_transpiler_clusters() {
    let registry =
        ClusterRegistry::default_transpiler_clusters().expect("Failed to load default clusters");

    // Verify standard clusters exist
    assert!(registry.get_cluster("MemoryAllocation").is_some());
    assert!(registry.get_cluster("FileIO").is_some());
    assert!(registry.get_cluster("ProcessControl").is_some());
    assert!(registry.get_cluster("Synchronization").is_some());
    assert!(registry.get_cluster("Randomness").is_some());
    assert!(registry.get_cluster("Networking").is_some());

    // Verify critical severity for networking
    let networking = registry.get_cluster("Networking").expect("test");
    assert_eq!(networking.severity, Severity::Critical);
    assert_eq!(networking.anomaly_threshold, 0.0);
}

/// Test classification of standard syscalls
#[test]
fn test_classify_standard_syscalls() {
    let registry = ClusterRegistry::default_transpiler_clusters().expect("test");

    // Memory allocation
    assert_eq!(classify_name(&registry, "mmap").as_deref(), Some("MemoryAllocation"));
    assert_eq!(classify_name(&registry, "brk").as_deref(), Some("MemoryAllocation"));

    // File I/O
    assert_eq!(classify_name(&registry, "read").as_deref(), Some("FileIO"));
    assert_eq!(classify_name(&registry, "write").as_deref(), Some("FileIO"));

    // Process control
    assert_eq!(classify_name(&registry, "fork").as_deref(), Some("ProcessControl"));

    // Synchronization (RED FLAG for single-threaded transpilers)
    assert_eq!(classify_name(&registry, "futex").as_deref(), Some("Synchronization"));

    // Networking (CRITICAL - telemetry leaks)
    assert_eq!(classify_name(&registry, "socket").as_deref(), Some("Networking"));
}

/// Test future-proof syscall support (mmap3, clone3)
#[test]
fn test_future_proof_syscalls() {
    let registry = ClusterRegistry::default_transpiler_clusters().expect("test");

    // New kernel syscalls should be pre-configured
    assert_eq!(classify_name(&registry, "mmap3").as_deref(), Some("MemoryAllocation"));
    assert_eq!(classify_name(&registry, "clone3").as_deref(), Some("ProcessControl"));
}

/// Test GPU cluster with args_filter (context-aware classification)
#[test]
fn test_gpu_cluster_with_filter() {
    let registry = ClusterRegistry::default_transpiler_clusters().expect("test");

    // ioctl without GPU fd → no GPU cluster match
    let fds = FdTable::new();
    assert!(registry.classify("ioctl", &["3".to_string()], &fds).is_none());

    // ioctl with GPU fd → GPU cluster match
    let mut fds_gpu = FdTable::new();
    fds_gpu.insert(3, "/dev/nvidia0".to_string());
    let cluster = registry.classify("ioctl", &["3".to_string()], &fds_gpu);
    assert!(cluster.is_some());
    assert_eq!(cluster.expect("test").name, "GPU");
}

/// Test anomaly detection thresholds
#[test]
fn test_anomaly_detection() {
    let registry = ClusterRegistry::default_transpiler_clusters().expect("test");

    // MemoryAllocation: 50% threshold
    let mem_cluster = registry.get_cluster("MemoryAllocation").expect("test");
    assert!(!mem_cluster.is_anomalous(100, 140)); // +40% = acceptable
    assert!(mem_cluster.is_anomalous(100, 160)); // +60% = anomaly

    // Networking: 0% threshold (ANY networking is anomaly)
    let net_cluster = registry.get_cluster("Networking").expect("test");
    assert!(net_cluster.is_anomalous(0, 1)); // New networking = CRITICAL
    assert!(!net_cluster.is_anomalous(0, 0)); // No networking = OK
}

/// Test custom TOML loading with user-defined clusters
#[test]
fn test_custom_toml_clusters() -> Result<()> {
    let mut file = NamedTempFile::new()?;
    writeln!(
        file,
        r#"
[[cluster]]
name = "TensorFlow"
description = "TensorFlow C API calls"
syscalls = ["dlopen", "dlsym"]
expected_for_transpiler = false
anomaly_threshold = 0.0
severity = "medium"

[cluster.args_filter]
arg_contains = "libtensorflow"
"#
    )?;
    file.flush()?;

    let registry = ClusterRegistry::from_toml(file.path())?;

    // dlopen without TensorFlow → no match
    let cluster = registry.classify("dlopen", &["/usr/lib/libm.so".to_string()], &FdTable::new());
    assert!(cluster.is_none());

    // dlopen with libtensorflow → TensorFlow cluster
    let cluster =
        registry.classify("dlopen", &["/usr/lib/libtensorflow.so".to_string()], &FdTable::new());
    assert!(cluster.is_some());
    assert_eq!(cluster.expect("test").name, "TensorFlow");

    Ok(())
}

/// Test Poka-Yoke: duplicate syscall detection
#[test]
fn test_duplicate_syscall_error() {
    let mut file = NamedTempFile::new().expect("test");
    writeln!(
        file,
        r#"
[[cluster]]
name = "ClusterA"
description = "First cluster"
syscalls = ["mmap", "read"]
expected_for_transpiler = true
anomaly_threshold = 0.5
severity = "medium"

[[cluster]]
name = "ClusterB"
description = "Second cluster"
syscalls = ["write", "mmap"]
expected_for_transpiler = true
anomaly_threshold = 0.5
severity = "medium"
"#
    )
    .expect("test");
    file.flush().expect("test");

    let result = ClusterRegistry::from_toml(file.path());
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Duplicate syscall 'mmap'"));
    assert!(err_msg.contains("ClusterA"));
    assert!(err_msg.contains("ClusterB"));
}

/// Test expected_for_transpiler flag
#[test]
fn test_expected_for_transpiler() {
    let registry = ClusterRegistry::default_transpiler_clusters().expect("test");

    // Expected clusters
    assert!(registry.get_cluster("MemoryAllocation").expect("test").expected_for_transpiler);
    assert!(registry.get_cluster("FileIO").expect("test").expected_for_transpiler);

    // Unexpected clusters (should trigger warnings)
    assert!(!registry.get_cluster("Networking").expect("test").expected_for_transpiler);
    assert!(!registry.get_cluster("Synchronization").expect("test").expected_for_transpiler);
    assert!(!registry.get_cluster("GPU").expect("test").expected_for_transpiler);
}

/// Test severity ordering for prioritization
#[test]
fn test_severity_prioritization() {
    let registry = ClusterRegistry::default_transpiler_clusters().expect("test");

    let networking = registry.get_cluster("Networking").expect("test");
    let synchronization = registry.get_cluster("Synchronization").expect("test");
    let process_control = registry.get_cluster("ProcessControl").expect("test");
    let memory = registry.get_cluster("MemoryAllocation").expect("test");

    // Critical > High > Medium
    assert!(networking.severity == Severity::Critical);
    assert!(synchronization.severity == Severity::Critical);
    assert!(process_control.severity > memory.severity);
}
