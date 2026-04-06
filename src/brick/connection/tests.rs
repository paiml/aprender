use super::*;

#[test]
fn test_managed_connection_new() {
    let conn =
        ManagedConnection::new("test_connection", Duration::from_secs(60), Duration::from_secs(10));

    assert_eq!(conn.inner(), &"test_connection");
    assert!(conn.is_valid());
    assert!(!conn.is_expired());
    assert!(!conn.is_idle());
}

#[test]
fn test_managed_connection_inner_mut() {
    let mut conn =
        ManagedConnection::new(vec![1, 2, 3], Duration::from_secs(60), Duration::from_secs(10));

    conn.inner_mut().push(4);
    assert_eq!(conn.inner(), &vec![1, 2, 3, 4]);
}

#[test]
fn test_managed_connection_into_inner() {
    let conn = ManagedConnection::new(42u32, Duration::from_secs(60), Duration::from_secs(10));

    let value = conn.into_inner();
    assert_eq!(value, 42);
}

#[test]
fn test_managed_connection_touch() {
    let mut conn =
        ManagedConnection::new("test", Duration::from_secs(60), Duration::from_millis(10));

    std::thread::sleep(Duration::from_millis(5));
    let idle_before = conn.idle_time();

    conn.touch();
    let idle_after = conn.idle_time();

    assert!(idle_after < idle_before);
}

#[test]
fn test_managed_connection_health_failures() {
    let mut conn = ManagedConnection::new("test", Duration::from_secs(60), Duration::from_secs(10));

    assert!(conn.is_valid());

    conn.record_health_failure();
    conn.record_health_failure();
    assert!(conn.is_valid()); // 2 failures, still valid

    conn.record_health_failure();
    assert!(!conn.is_valid()); // 3 failures, now invalid

    conn.reset_health();
    assert!(conn.is_valid()); // Reset, valid again
}

#[test]
fn test_managed_connection_age() {
    let conn = ManagedConnection::new("test", Duration::from_secs(60), Duration::from_secs(10));

    let age = conn.age();
    assert!(age < Duration::from_millis(100));
}

#[test]
fn test_managed_connection_ttl_expiry() {
    let conn = ManagedConnection::new("test", Duration::from_millis(10), Duration::from_secs(60));

    assert!(!conn.is_expired());

    std::thread::sleep(Duration::from_millis(15));

    assert!(conn.is_expired());
    assert!(!conn.is_valid());
}

#[test]
fn test_managed_connection_idle_expiry() {
    let conn = ManagedConnection::new("test", Duration::from_secs(60), Duration::from_millis(10));

    assert!(!conn.is_idle());

    std::thread::sleep(Duration::from_millis(15));

    assert!(conn.is_idle());
    assert!(!conn.is_valid());
}

/// FALSIFICATION TEST: Verify all three conditions must pass for validity
///
/// A connection is valid only if: not_expired AND not_idle AND healthy.
/// If any one fails, the connection should be invalid.
#[test]
fn test_falsify_validity_requires_all_conditions() {
    // Test 1: Healthy and not idle, but expired
    let expired = ManagedConnection::new("test", Duration::from_millis(5), Duration::from_secs(60));
    std::thread::sleep(Duration::from_millis(10));
    assert!(!expired.is_valid(), "FALSIFICATION FAILED: Expired connection should be invalid");

    // Test 2: Healthy and not expired, but idle
    let idle = ManagedConnection::new("test", Duration::from_secs(60), Duration::from_millis(5));
    std::thread::sleep(Duration::from_millis(10));
    assert!(!idle.is_valid(), "FALSIFICATION FAILED: Idle connection should be invalid");

    // Test 3: Not expired and not idle, but unhealthy
    let mut unhealthy =
        ManagedConnection::new("test", Duration::from_secs(60), Duration::from_secs(60));
    unhealthy.record_health_failure();
    unhealthy.record_health_failure();
    unhealthy.record_health_failure();
    assert!(!unhealthy.is_valid(), "FALSIFICATION FAILED: Unhealthy connection should be invalid");

    // Test 4: All conditions pass
    let valid = ManagedConnection::new("test", Duration::from_secs(60), Duration::from_secs(60));
    assert!(valid.is_valid(), "FALSIFICATION FAILED: Fresh connection should be valid");
}

/// FALSIFICATION TEST: Touch must reset idle timer
///
/// Uses 200ms timeout with 100ms sleeps to provide 100ms margin against timing jitter.
/// Previous 20ms/15ms (5ms margin) caused flaky failures on Mac.
#[test]
fn test_falsify_touch_resets_idle() {
    let mut conn =
        ManagedConnection::new("test", Duration::from_secs(60), Duration::from_millis(200));

    // Wait until almost idle (100ms of 200ms timeout = 50% margin)
    std::thread::sleep(Duration::from_millis(100));
    assert!(!conn.is_idle(), "Should not be idle yet at 100ms with 200ms timeout");

    // Touch to reset
    conn.touch();

    // Wait another 100ms (would be 200ms total without touch, but only 100ms since touch)
    std::thread::sleep(Duration::from_millis(100));
    assert!(!conn.is_idle(), "FALSIFICATION FAILED: Touch should have reset idle timer");

    // Now wait until actually idle (another 150ms = 250ms since touch > 200ms timeout)
    std::thread::sleep(Duration::from_millis(150));
    assert!(conn.is_idle(), "Should be idle now (250ms since touch > 200ms timeout)");
}

// =========================================================================
// KeepAliveConfig Tests
// =========================================================================

#[test]
fn test_keep_alive_config_new() {
    let config = KeepAliveConfig::new();
    assert!(config.enabled);
    assert_eq!(config.timeout_secs, 60);
    assert_eq!(config.max_requests, 100);
}

#[test]
fn test_keep_alive_config_disabled() {
    let config = KeepAliveConfig::disabled();
    assert!(!config.enabled);
    assert_eq!(config.timeout_secs, 0);
    assert_eq!(config.max_requests, 0);
}

#[test]
fn test_keep_alive_config_from_header() {
    let config = KeepAliveConfig::from_header("timeout=30, max=50");
    assert!(config.enabled);
    assert_eq!(config.timeout_secs, 30);
    assert_eq!(config.max_requests, 50);
}

#[test]
fn test_keep_alive_config_from_header_partial() {
    // Only timeout specified
    let config = KeepAliveConfig::from_header("timeout=15");
    assert_eq!(config.timeout_secs, 15);
    assert_eq!(config.max_requests, 100); // Default

    // Only max specified
    let config = KeepAliveConfig::from_header("max=25");
    assert_eq!(config.timeout_secs, 60); // Default
    assert_eq!(config.max_requests, 25);
}

#[test]
fn test_keep_alive_config_from_header_invalid() {
    // Invalid values are ignored
    let config = KeepAliveConfig::from_header("timeout=abc, max=xyz");
    assert_eq!(config.timeout_secs, 60); // Default
    assert_eq!(config.max_requests, 100); // Default
}

#[test]
fn test_keep_alive_config_should_keep_alive() {
    let config = KeepAliveConfig::new(); // max_requests = 100
    assert!(config.should_keep_alive(0));
    assert!(config.should_keep_alive(50));
    assert!(config.should_keep_alive(99));
    assert!(!config.should_keep_alive(100));
    assert!(!config.should_keep_alive(200));
}

#[test]
fn test_keep_alive_config_should_keep_alive_disabled() {
    let config = KeepAliveConfig::disabled();
    assert!(!config.should_keep_alive(0));
}

#[test]
fn test_keep_alive_config_default() {
    let config = KeepAliveConfig::default();
    assert_eq!(config, KeepAliveConfig::new());
}

// =========================================================================
// ConnectionState Tests
// =========================================================================

#[test]
fn test_connection_state_new() {
    let state = ConnectionState::new();
    assert_eq!(state.bits(), 0);
    assert!(!state.is_set(ConnectionState::OPEN));
}

#[test]
fn test_connection_state_open_connection() {
    let state = ConnectionState::open_connection();
    assert!(state.is_set(ConnectionState::OPEN));
    assert!(state.is_set(ConnectionState::WRITABLE));
    assert!(!state.is_set(ConnectionState::READABLE));
}

#[test]
fn test_connection_state_set_clear() {
    let mut state = ConnectionState::new();

    state.set(ConnectionState::OPEN);
    assert!(state.is_set(ConnectionState::OPEN));

    state.set(ConnectionState::READABLE);
    assert!(state.is_set(ConnectionState::OPEN));
    assert!(state.is_set(ConnectionState::READABLE));

    state.clear(ConnectionState::OPEN);
    assert!(!state.is_set(ConnectionState::OPEN));
    assert!(state.is_set(ConnectionState::READABLE));
}

#[test]
fn test_connection_state_is_healthy() {
    let mut state = ConnectionState::open_connection();
    assert!(state.is_healthy());

    state.set(ConnectionState::ERROR);
    assert!(!state.is_healthy());

    state.clear(ConnectionState::ERROR);
    state.set(ConnectionState::CLOSING);
    assert!(!state.is_healthy());
}

#[test]
fn test_connection_state_can_read() {
    let mut state = ConnectionState::open_connection();
    assert!(!state.can_read()); // Not readable yet

    state.set(ConnectionState::READABLE);
    assert!(state.can_read());

    state.clear(ConnectionState::OPEN);
    assert!(!state.can_read()); // Not open
}

#[test]
fn test_connection_state_can_write() {
    let mut state = ConnectionState::open_connection();
    assert!(state.can_write());

    state.set(ConnectionState::CLOSING);
    assert!(!state.can_write()); // Can't write when closing

    state.clear(ConnectionState::CLOSING);
    state.clear(ConnectionState::WRITABLE);
    assert!(!state.can_write()); // Not writable
}

#[test]
fn test_connection_state_bits() {
    let mut state = ConnectionState::new();
    state.set(ConnectionState::OPEN);
    state.set(ConnectionState::WRITABLE);
    assert_eq!(state.bits(), 0b0000_0101);
}

#[test]
fn test_connection_state_default() {
    let state = ConnectionState::default();
    assert_eq!(state.bits(), 0);
}

/// FALSIFICATION TEST: ConnectionState bitflags must be non-overlapping
#[test]
fn test_falsify_connection_state_flags_unique() {
    let flags = [
        ConnectionState::OPEN,
        ConnectionState::READABLE,
        ConnectionState::WRITABLE,
        ConnectionState::HAS_PENDING,
        ConnectionState::KEEP_ALIVE,
        ConnectionState::UPGRADE,
        ConnectionState::CLOSING,
        ConnectionState::ERROR,
    ];

    // Each flag must be a power of 2 (single bit set)
    for flag in &flags {
        assert!(
            flag.is_power_of_two(),
            "FALSIFICATION FAILED: Flag {:08b} is not a power of 2",
            flag
        );
    }

    // No two flags should overlap
    for i in 0..flags.len() {
        for j in (i + 1)..flags.len() {
            assert_eq!(
                flags[i] & flags[j],
                0,
                "FALSIFICATION FAILED: Flags {:08b} and {:08b} overlap",
                flags[i],
                flags[j]
            );
        }
    }
}

/// FALSIFICATION TEST: KeepAliveConfig should_keep_alive boundary
#[test]
fn test_falsify_keep_alive_boundary() {
    let config = KeepAliveConfig { enabled: true, timeout_secs: 60, max_requests: 100 };

    // At exactly max_requests, should NOT keep alive
    assert!(
        !config.should_keep_alive(100),
        "FALSIFICATION FAILED: should_keep_alive should return false at max_requests"
    );

    // One below max should still be true
    assert!(
        config.should_keep_alive(99),
        "FALSIFICATION FAILED: should_keep_alive should return true below max_requests"
    );
}
