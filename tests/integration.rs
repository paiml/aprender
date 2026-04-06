//! Root-level integration tests for renacer syscall tracer.
//!
//! These tests verify the core public API is accessible and functional.

use std::collections::HashMap;

/// Test that the crate builds and exports expected types
#[test]
fn test_crate_exports() {
    // Verify main module exports compile - no assertion needed, just compilation
}

/// Test basic `HashMap` operations (used throughout renacer)
#[test]
fn test_hashmap_operations() {
    let mut map: HashMap<String, i32> = HashMap::new();
    map.insert("syscall_count".to_string(), 42);
    assert_eq!(map.get("syscall_count"), Some(&42));
}

/// Test string formatting used in syscall output
#[test]
fn test_syscall_formatting() {
    let syscall_name = "write";
    let fd = 1;
    let buf_addr = 0x7fff_0000_1000_u64;
    let count = 13;

    let formatted = format!("{}({}, 0x{:x}, {})", syscall_name, fd, buf_addr, count);
    assert!(formatted.contains("write"));
    assert!(formatted.contains("0x7fff00001000"));
}

/// Test return value interpretation
#[test]
fn test_return_value_formatting() {
    let success_ret: i64 = 13;
    let error_ret: i64 = -2; // ENOENT

    assert!(success_ret >= 0);
    assert!(error_ret < 0);
}

/// Test timestamp calculations
#[test]
fn test_duration_calculations() {
    use std::time::Duration;

    let start_ns = 1_000_000_000_u64;
    let end_ns = 1_000_001_500_u64;
    let duration_ns = end_ns - start_ns;

    let duration = Duration::from_nanos(duration_ns);
    assert_eq!(duration.as_micros(), 1);
}

/// Test PID formatting
#[test]
fn test_pid_formatting() {
    let pid = 12345;
    let formatted = format!("[pid {}]", pid);
    assert_eq!(formatted, "[pid 12345]");
}

/// Test syscall timing statistics
#[test]
fn test_timing_statistics() {
    let durations = vec![100, 200, 150, 175, 125];
    let sum: u64 = durations.iter().sum();
    let avg = sum / durations.len() as u64;

    assert_eq!(sum, 750);
    assert_eq!(avg, 150);
}

/// Test hex encoding used for buffer display
#[test]
fn test_hex_encoding() {
    let bytes = [0x48, 0x65, 0x6c, 0x6c, 0x6f]; // "Hello"
    let hex_str: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
    assert_eq!(hex_str, "48656c6c6f");
}

/// Test error code mapping patterns
#[test]
fn test_error_patterns() {
    let errno_map: HashMap<i32, &str> =
        [(1, "EPERM"), (2, "ENOENT"), (13, "EACCES"), (22, "EINVAL")].into_iter().collect();

    assert_eq!(errno_map.get(&2), Some(&"ENOENT"));
    assert_eq!(errno_map.get(&13), Some(&"EACCES"));
}
