// Sprint 33: W3C Trace Context Propagation
//
// Implements W3C Trace Context standard for distributed tracing.
// Reference: https://www.w3.org/TR/trace-context/
//
// Format: version-trace_id-parent_id-trace_flags
// Example: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01

use std::fmt;

/// W3C Trace Context (traceparent header)
///
/// Represents distributed trace context passed across service boundaries.
/// Enables Renacer to create syscall spans as children of application spans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceContext {
    pub version: u8,
    pub trace_id: [u8; 16], // 128-bit trace ID
    pub parent_id: [u8; 8], // 64-bit parent span ID
    pub trace_flags: u8,    // 8-bit flags (01 = sampled)
}

impl TraceContext {
    /// Parse W3C traceparent string
    ///
    /// Format: "00-{trace_id}-{parent_id}-{flags}"
    /// Example: "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    pub fn parse(traceparent: &str) -> Result<Self, TraceContextError> {
        // Split into 4 parts: version-trace_id-parent_id-flags
        let parts: Vec<&str> = traceparent.split('-').collect();
        if parts.len() != 4 {
            return Err(TraceContextError::InvalidFormat);
        }

        // Parse version (must be "00")
        let version =
            u8::from_str_radix(parts[0], 16).map_err(|_| TraceContextError::InvalidVersion)?;
        if version != 0 {
            return Err(TraceContextError::InvalidVersion);
        }

        // Parse trace_id (32 hex chars = 16 bytes)
        if parts[1].len() != 32 {
            return Err(TraceContextError::InvalidTraceId);
        }
        let trace_id = hex_to_bytes_16(parts[1]).ok_or(TraceContextError::InvalidTraceId)?;

        // Validate trace_id is not all zeros
        if trace_id.iter().all(|&b| b == 0) {
            return Err(TraceContextError::AllZeroTraceId);
        }

        // Parse parent_id (16 hex chars = 8 bytes)
        if parts[2].len() != 16 {
            return Err(TraceContextError::InvalidParentId);
        }
        let parent_id = hex_to_bytes_8(parts[2]).ok_or(TraceContextError::InvalidParentId)?;

        // Validate parent_id is not all zeros
        if parent_id.iter().all(|&b| b == 0) {
            return Err(TraceContextError::AllZeroParentId);
        }

        // Parse trace_flags (2 hex chars = 1 byte)
        if parts[3].len() != 2 {
            return Err(TraceContextError::InvalidTraceFlags);
        }
        let trace_flags =
            u8::from_str_radix(parts[3], 16).map_err(|_| TraceContextError::InvalidTraceFlags)?;

        Ok(TraceContext { version, trace_id, parent_id, trace_flags })
    }

    /// Extract trace context from environment variables
    ///
    /// Checks TRACEPARENT and `OTEL_TRACEPARENT` (in that order)
    pub fn from_env() -> Option<Self> {
        std::env::var("TRACEPARENT")
            .or_else(|_| std::env::var("OTEL_TRACEPARENT"))
            .ok()
            .and_then(|s| Self::parse(&s).ok())
    }

    /// Set trace context as environment variable (Sprint 42: Batuta Integration)
    ///
    /// Sets TRACEPARENT for child processes to inherit trace context.
    /// This enables distributed tracing across `exec()` boundaries.
    pub fn set_env(&self) {
        std::env::set_var("TRACEPARENT", self.to_string());
    }

    /// Extract logical clock from environment variable (Sprint 42: Batuta Integration)
    ///
    /// Checks `RENACER_LOGICAL_CLOCK` environment variable.
    /// Returns None if not set or invalid format.
    pub fn logical_clock_from_env() -> Option<u64> {
        std::env::var("RENACER_LOGICAL_CLOCK").ok().and_then(|s| s.parse::<u64>().ok())
    }

    /// Set logical clock as environment variable (Sprint 42: Batuta Integration)
    ///
    /// Sets `RENACER_LOGICAL_CLOCK` for child processes to inherit.
    /// This propagates Lamport logical clocks across `exec()` boundaries.
    pub fn set_logical_clock_env(timestamp: u64) {
        std::env::set_var("RENACER_LOGICAL_CLOCK", timestamp.to_string());
    }

    /// Check if trace is sampled (`trace_flags` & 0x01)
    pub fn is_sampled(&self) -> bool {
        self.trace_flags & 0x01 != 0
    }

    /// Convert `trace_id` to OpenTelemetry `TraceId`
    #[cfg(feature = "otlp")]
    pub fn otel_trace_id(&self) -> opentelemetry::trace::TraceId {
        opentelemetry::trace::TraceId::from_bytes(self.trace_id)
    }

    /// Convert `parent_id` to OpenTelemetry `SpanId`
    #[cfg(feature = "otlp")]
    pub fn otel_parent_id(&self) -> opentelemetry::trace::SpanId {
        opentelemetry::trace::SpanId::from_bytes(self.parent_id)
    }
}

impl fmt::Display for TraceContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:02x}-{}-{}-{:02x}",
            self.version,
            bytes_to_hex_16(&self.trace_id),
            bytes_to_hex_8(&self.parent_id),
            self.trace_flags
        )
    }
}

/// Trace Context parsing errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceContextError {
    /// Invalid format (must be version-trace_id-parent_id-flags)
    InvalidFormat,
    /// Invalid version (must be 00)
    InvalidVersion,
    /// Invalid `trace_id` (must be 32 hex chars)
    InvalidTraceId,
    /// Invalid `parent_id` (must be 16 hex chars)
    InvalidParentId,
    /// Invalid `trace_flags` (must be 2 hex chars)
    InvalidTraceFlags,
    /// Trace ID is all zeros (forbidden by W3C spec)
    AllZeroTraceId,
    /// Parent ID is all zeros (forbidden by W3C spec)
    AllZeroParentId,
}

impl fmt::Display for TraceContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFormat => {
                write!(f, "Invalid traceparent format (expected: version-trace_id-parent_id-flags)")
            }
            Self::InvalidVersion => write!(f, "Invalid version (must be 00)"),
            Self::InvalidTraceId => write!(f, "Invalid trace_id (must be 32 hex characters)"),
            Self::InvalidParentId => write!(f, "Invalid parent_id (must be 16 hex characters)"),
            Self::InvalidTraceFlags => write!(f, "Invalid trace_flags (must be 2 hex characters)"),
            Self::AllZeroTraceId => write!(f, "Trace ID cannot be all zeros"),
            Self::AllZeroParentId => write!(f, "Parent ID cannot be all zeros"),
        }
    }
}

impl std::error::Error for TraceContextError {}

// Helper functions for hex conversion

fn hex_to_bytes_16(hex: &str) -> Option<[u8; 16]> {
    let mut bytes = [0u8; 16];
    for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
        if i >= 16 {
            return None;
        }
        let hex_str = std::str::from_utf8(chunk).ok()?;
        bytes[i] = u8::from_str_radix(hex_str, 16).ok()?;
    }
    Some(bytes)
}

fn hex_to_bytes_8(hex: &str) -> Option<[u8; 8]> {
    let mut bytes = [0u8; 8];
    for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
        if i >= 8 {
            return None;
        }
        let hex_str = std::str::from_utf8(chunk).ok()?;
        bytes[i] = u8::from_str_radix(hex_str, 16).ok()?;
    }
    Some(bytes)
}

fn bytes_to_hex_16(bytes: &[u8; 16]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn bytes_to_hex_8(bytes: &[u8; 8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// Compile-time thread-safety verification (Sprint 59)
static_assertions::assert_impl_all!(TraceContext: Send, Sync);
static_assertions::assert_impl_all!(TraceContextError: Send, Sync);

// ============================================================================
// UNIT TESTS (EXTREME TDD - RED Phase)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1: Parse valid traceparent
    #[test]
    fn test_parse_valid_traceparent() {
        let traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
        let ctx = TraceContext::parse(traceparent).expect("test");

        assert_eq!(ctx.version, 0);
        assert_eq!(ctx.trace_flags, 1);
        assert!(ctx.is_sampled());

        // Verify trace_id
        let expected_trace_id = [
            0x0a, 0xf7, 0x65, 0x19, 0x16, 0xcd, 0x43, 0xdd, 0x84, 0x48, 0xeb, 0x21, 0x1c, 0x80,
            0x31, 0x9c,
        ];
        assert_eq!(ctx.trace_id, expected_trace_id);

        // Verify parent_id
        let expected_parent_id = [0xb7, 0xad, 0x6b, 0x71, 0x69, 0x20, 0x33, 0x31];
        assert_eq!(ctx.parent_id, expected_parent_id);
    }

    // Test 2: Parse another valid traceparent (not sampled)
    #[test]
    fn test_parse_valid_traceparent_not_sampled() {
        let traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00";
        let ctx = TraceContext::parse(traceparent).expect("test");

        assert_eq!(ctx.version, 0);
        assert_eq!(ctx.trace_flags, 0);
        assert!(!ctx.is_sampled());
    }

    // Test 3: Invalid format (missing parts)
    #[test]
    fn test_parse_invalid_format_missing_parts() {
        let traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331";
        let result = TraceContext::parse(traceparent);

        assert_eq!(result, Err(TraceContextError::InvalidFormat));
    }

    // Test 4: Invalid format (too many parts)
    #[test]
    fn test_parse_invalid_format_too_many_parts() {
        let traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01-extra";
        let result = TraceContext::parse(traceparent);

        assert_eq!(result, Err(TraceContextError::InvalidFormat));
    }

    // Test 5: Invalid format (empty string)
    #[test]
    fn test_parse_invalid_format_empty() {
        let result = TraceContext::parse("");
        assert_eq!(result, Err(TraceContextError::InvalidFormat));
    }

    // Test 6: All-zero trace_id
    #[test]
    fn test_parse_all_zero_trace_id() {
        let traceparent = "00-00000000000000000000000000000000-b7ad6b7169203331-01";
        let result = TraceContext::parse(traceparent);

        assert_eq!(result, Err(TraceContextError::AllZeroTraceId));
    }

    // Test 7: All-zero parent_id
    #[test]
    fn test_parse_all_zero_parent_id() {
        let traceparent = "00-0af7651916cd43dd8448eb211c80319c-0000000000000000-01";
        let result = TraceContext::parse(traceparent);

        assert_eq!(result, Err(TraceContextError::AllZeroParentId));
    }

    // Test 8: Invalid version
    #[test]
    fn test_parse_invalid_version() {
        let traceparent = "99-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
        let result = TraceContext::parse(traceparent);

        assert_eq!(result, Err(TraceContextError::InvalidVersion));
    }

    // Test 9: Invalid version (non-hex)
    #[test]
    fn test_parse_invalid_version_non_hex() {
        let traceparent = "XX-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
        let result = TraceContext::parse(traceparent);

        assert_eq!(result, Err(TraceContextError::InvalidVersion));
    }

    // Test 10: Invalid trace_id (wrong length)
    #[test]
    fn test_parse_invalid_trace_id_wrong_length() {
        let traceparent = "00-0af7651916cd43dd-b7ad6b7169203331-01";
        let result = TraceContext::parse(traceparent);

        assert_eq!(result, Err(TraceContextError::InvalidTraceId));
    }

    // Test 11: Invalid trace_id (non-hex)
    #[test]
    fn test_parse_invalid_trace_id_non_hex() {
        let traceparent = "00-ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ-b7ad6b7169203331-01";
        let result = TraceContext::parse(traceparent);

        assert_eq!(result, Err(TraceContextError::InvalidTraceId));
    }

    // Test 12: Invalid parent_id (wrong length)
    #[test]
    fn test_parse_invalid_parent_id_wrong_length() {
        let traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad-01";
        let result = TraceContext::parse(traceparent);

        assert_eq!(result, Err(TraceContextError::InvalidParentId));
    }

    // Test 13: Invalid parent_id (non-hex)
    #[test]
    fn test_parse_invalid_parent_id_non_hex() {
        let traceparent = "00-0af7651916cd43dd8448eb211c80319c-XXXXXXXXXXXXXXXX-01";
        let result = TraceContext::parse(traceparent);

        assert_eq!(result, Err(TraceContextError::InvalidParentId));
    }

    // Test 14: Invalid trace_flags (wrong length)
    #[test]
    fn test_parse_invalid_trace_flags_wrong_length() {
        let traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-1";
        let result = TraceContext::parse(traceparent);

        assert_eq!(result, Err(TraceContextError::InvalidTraceFlags));
    }

    // Test 15: Invalid trace_flags (non-hex)
    #[test]
    fn test_parse_invalid_trace_flags_non_hex() {
        let traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-XX";
        let result = TraceContext::parse(traceparent);

        assert_eq!(result, Err(TraceContextError::InvalidTraceFlags));
    }

    // Test 16: is_sampled() with flag = 01
    #[test]
    fn test_is_sampled_flag_set() {
        let traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
        let ctx = TraceContext::parse(traceparent).expect("test");

        assert!(ctx.is_sampled());
    }

    // Test 17: is_sampled() with flag = 00
    #[test]
    fn test_is_sampled_flag_unset() {
        let traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-00";
        let ctx = TraceContext::parse(traceparent).expect("test");

        assert!(!ctx.is_sampled());
    }

    // Test 18: Display formatting
    #[test]
    fn test_display_formatting() {
        let traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
        let ctx = TraceContext::parse(traceparent).expect("test");

        assert_eq!(ctx.to_string(), traceparent);
    }

    // Test 19: from_env() with TRACEPARENT set
    #[test]
    fn test_from_env_traceparent() {
        std::env::set_var("TRACEPARENT", "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01");

        let ctx = TraceContext::from_env();
        assert!(ctx.is_some());

        let ctx = ctx.expect("test");
        assert_eq!(ctx.version, 0);
        assert!(ctx.is_sampled());

        std::env::remove_var("TRACEPARENT");
    }

    // Test 20: from_env() with OTEL_TRACEPARENT set
    #[test]
    fn test_from_env_otel_traceparent() {
        std::env::remove_var("TRACEPARENT");
        std::env::set_var(
            "OTEL_TRACEPARENT",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00",
        );

        let ctx = TraceContext::from_env();
        assert!(ctx.is_some());

        let ctx = ctx.expect("test");
        assert_eq!(ctx.version, 0);
        assert!(!ctx.is_sampled());

        std::env::remove_var("OTEL_TRACEPARENT");
    }

    // Test 21: from_env() with no env var set
    #[test]
    fn test_from_env_missing() {
        std::env::remove_var("TRACEPARENT");
        std::env::remove_var("OTEL_TRACEPARENT");

        let ctx = TraceContext::from_env();
        assert!(ctx.is_none());
    }

    // Test 22: from_env() with invalid format
    #[test]
    fn test_from_env_invalid_format() {
        std::env::set_var("TRACEPARENT", "INVALID");

        let ctx = TraceContext::from_env();
        assert!(ctx.is_none());

        std::env::remove_var("TRACEPARENT");
    }

    // Test 23: Clone trait
    #[test]
    fn test_trace_context_clone() {
        let traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
        let ctx1 = TraceContext::parse(traceparent).expect("test");
        let ctx2 = ctx1.clone();

        assert_eq!(ctx1, ctx2);
    }

    // Test 24: Debug trait
    #[test]
    fn test_trace_context_debug() {
        let traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
        let ctx = TraceContext::parse(traceparent).expect("test");

        let debug_str = format!("{:?}", ctx);
        assert!(debug_str.contains("TraceContext"));
    }

    // Test 25: Error Display trait
    #[test]
    fn test_error_display() {
        let err = TraceContextError::InvalidFormat;
        assert!(err.to_string().contains("Invalid traceparent format"));

        let err = TraceContextError::AllZeroTraceId;
        assert!(err.to_string().contains("all zeros"));
    }
}

// ============================================================================
// LAMPORT CLOCK IMPLEMENTATION (Specification Section 6.2)
// ============================================================================

use std::sync::atomic::{AtomicU64, Ordering};

/// Lamport Clock for happens-before ordering
///
/// Implements Lamport's logical clocks for establishing causal relationships
/// between events in a distributed tracing system.
///
/// Reference: "Time, Clocks, and the Ordering of Events in a Distributed System"
/// Lamport, L. Communications of the ACM, 21(7), 558-565 (1978)
///
/// # Properties
///
/// 1. **Transitivity**: a → b ∧ b → c ⇒ a → c
/// 2. **Irreflexivity**: ¬(a → a)
/// 3. **Timestamp consistency**: a → b ⇒ timestamp(a) < timestamp(b)
#[derive(Debug)]
pub struct LamportClock {
    counter: AtomicU64,
}

impl LamportClock {
    /// Create a new Lamport clock starting at 0
    pub fn new() -> Self {
        LamportClock { counter: AtomicU64::new(0) }
    }

    /// Create a Lamport clock with a specific starting value
    pub fn with_initial_value(initial: u64) -> Self {
        LamportClock { counter: AtomicU64::new(initial) }
    }

    /// Increment clock on local event
    ///
    /// Returns the new timestamp after increment.
    /// Uses `SeqCst` ordering to ensure happens-before consistency.
    pub fn tick(&self) -> u64 {
        self.counter.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Synchronize clock on message receive
    ///
    /// Updates local clock to max(local, remote) + 1.
    /// Ensures received messages have higher timestamps than their causally preceding events.
    pub fn sync(&self, remote_timestamp: u64) -> u64 {
        loop {
            let current = self.counter.load(Ordering::SeqCst);
            let new_value = current.max(remote_timestamp) + 1;

            match self.counter.compare_exchange(
                current,
                new_value,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return new_value,
                Err(_) => continue, // Retry on concurrent modification
            }
        }
    }

    /// Get current clock value without modifying it
    pub fn now(&self) -> u64 {
        self.counter.load(Ordering::SeqCst)
    }

    /// Check if timestamp a happens-before timestamp b
    ///
    /// Returns true if a < b (strict temporal ordering).
    /// Note: This is a simple timestamp comparison. Full happens-before
    /// semantics require checking causal chains (see `UnifiedTrace`).
    pub fn happens_before(a: u64, b: u64) -> bool {
        a < b
    }
}

impl Default for LamportClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for LamportClock {
    fn clone(&self) -> Self {
        LamportClock { counter: AtomicU64::new(self.counter.load(Ordering::SeqCst)) }
    }
}

// Compile-time thread-safety verification for LamportClock
static_assertions::assert_impl_all!(LamportClock: Send, Sync);

// ============================================================================
// LAMPORT CLOCK TESTS (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod lamport_tests {
    use super::*;

    // Test 1: Clock starts at 0
    #[test]
    fn test_clock_starts_at_zero() {
        let clock = LamportClock::new();
        assert_eq!(clock.now(), 0);
    }

    // Test 2: Clock starts at custom value
    #[test]
    fn test_clock_with_initial_value() {
        let clock = LamportClock::with_initial_value(100);
        assert_eq!(clock.now(), 100);
    }

    // Test 3: Tick increments by 1
    #[test]
    fn test_tick_increments() {
        let clock = LamportClock::new();
        assert_eq!(clock.tick(), 1);
        assert_eq!(clock.tick(), 2);
        assert_eq!(clock.tick(), 3);
    }

    // Test 4: Tick returns incremented value
    #[test]
    fn test_tick_return_value() {
        let clock = LamportClock::new();
        let ts1 = clock.tick();
        let ts2 = clock.tick();
        assert!(ts2 > ts1);
        assert_eq!(ts2, ts1 + 1);
    }

    // Test 5: Sync with lower remote timestamp
    #[test]
    fn test_sync_lower_remote() {
        let clock = LamportClock::new();
        clock.tick(); // local = 1
        clock.tick(); // local = 2
        clock.tick(); // local = 3

        let new_ts = clock.sync(1); // remote = 1 < local = 3
        assert_eq!(new_ts, 4); // max(3, 1) + 1 = 4
    }

    // Test 6: Sync with higher remote timestamp
    #[test]
    fn test_sync_higher_remote() {
        let clock = LamportClock::new();
        clock.tick(); // local = 1

        let new_ts = clock.sync(10); // remote = 10 > local = 1
        assert_eq!(new_ts, 11); // max(1, 10) + 1 = 11
    }

    // Test 7: Sync with equal remote timestamp
    #[test]
    fn test_sync_equal_remote() {
        let clock = LamportClock::new();
        clock.tick(); // local = 1
        clock.tick(); // local = 2
        clock.tick(); // local = 3

        let new_ts = clock.sync(3); // remote = 3 == local = 3
        assert_eq!(new_ts, 4); // max(3, 3) + 1 = 4
    }

    // Test 8: Multiple syncs in sequence
    #[test]
    fn test_multiple_syncs() {
        let clock = LamportClock::new();

        let ts1 = clock.sync(5);
        assert_eq!(ts1, 6); // max(0, 5) + 1

        let ts2 = clock.sync(10);
        assert_eq!(ts2, 11); // max(6, 10) + 1

        let ts3 = clock.sync(8);
        assert_eq!(ts3, 12); // max(11, 8) + 1
    }

    // Test 9: Interleaved ticks and syncs
    #[test]
    fn test_interleaved_operations() {
        let clock = LamportClock::new();

        assert_eq!(clock.tick(), 1);
        assert_eq!(clock.sync(5), 6);
        assert_eq!(clock.tick(), 7);
        assert_eq!(clock.tick(), 8);
        assert_eq!(clock.sync(10), 11);
    }

    // Test 10: now() doesn't modify clock
    #[test]
    fn test_now_readonly() {
        let clock = LamportClock::new();
        clock.tick(); // local = 1

        assert_eq!(clock.now(), 1);
        assert_eq!(clock.now(), 1); // Still 1, not incremented
        assert_eq!(clock.now(), 1);
    }

    // Test 11: happens_before with a < b
    #[test]
    fn test_happens_before_true() {
        assert!(LamportClock::happens_before(1, 2));
        assert!(LamportClock::happens_before(10, 20));
        assert!(LamportClock::happens_before(0, 1));
    }

    // Test 12: happens_before with a >= b
    #[test]
    fn test_happens_before_false() {
        assert!(!LamportClock::happens_before(2, 1));
        assert!(!LamportClock::happens_before(5, 5));
        assert!(!LamportClock::happens_before(10, 5));
    }

    // Test 13: Clone preserves value
    #[test]
    fn test_clone_preserves_value() {
        let clock1 = LamportClock::new();
        clock1.tick();
        clock1.tick();
        clock1.tick(); // clock1 = 3

        let clock2 = clock1.clone();
        assert_eq!(clock2.now(), 3);
    }

    // Test 14: Cloned clocks are independent
    #[test]
    fn test_cloned_clocks_independent() {
        let clock1 = LamportClock::new();
        clock1.tick(); // clock1 = 1

        let clock2 = clock1.clone();

        clock1.tick(); // clock1 = 2
        clock2.tick(); // clock2 = 2

        assert_eq!(clock1.now(), 2);
        assert_eq!(clock2.now(), 2);

        clock1.tick(); // clock1 = 3
        assert_eq!(clock1.now(), 3);
        assert_eq!(clock2.now(), 2); // clock2 unchanged
    }

    // Test 15: Default trait
    #[test]
    fn test_default_trait() {
        let clock: LamportClock = Default::default();
        assert_eq!(clock.now(), 0);
    }

    // Test 16: Large timestamp values
    #[test]
    fn test_large_timestamps() {
        let clock = LamportClock::with_initial_value(u64::MAX - 10);
        assert_eq!(clock.now(), u64::MAX - 10);

        // Note: This will overflow, which is acceptable for Lamport clocks
        // In production, we'd handle overflow gracefully
        let _ts = clock.tick();
    }

    // Test 17: Sync updates clock correctly
    #[test]
    fn test_sync_updates_clock() {
        let clock = LamportClock::new();
        clock.sync(100);

        // After sync, local clock should be 101
        assert_eq!(clock.now(), 101);
    }

    // Test 18: Transitivity property
    #[test]
    fn test_transitivity_property() {
        let a = 1u64;
        let b = 5u64;
        let c = 10u64;

        // If a → b and b → c, then a → c
        assert!(LamportClock::happens_before(a, b));
        assert!(LamportClock::happens_before(b, c));
        assert!(LamportClock::happens_before(a, c));
    }

    // Test 19: Irreflexivity property
    #[test]
    fn test_irreflexivity_property() {
        let a = 5u64;

        // ¬(a → a): An event cannot happen before itself
        assert!(!LamportClock::happens_before(a, a));
    }

    // Test 20: Timestamp consistency
    #[test]
    fn test_timestamp_consistency() {
        let clock = LamportClock::new();

        let ts1 = clock.tick();
        let ts2 = clock.tick();

        // If event1 happens before event2, then ts1 < ts2
        assert!(LamportClock::happens_before(ts1, ts2));
        assert!(ts1 < ts2);
    }

    // Test 21: Concurrent operations (simulated)
    #[test]
    fn test_concurrent_ticks() {
        use std::sync::Arc;
        use std::thread;

        let clock = Arc::new(LamportClock::new());
        let mut handles = vec![];

        // Spawn 10 threads, each doing 10 ticks
        for _ in 0..10 {
            let clock_clone = Arc::clone(&clock);
            let handle = thread::spawn(move || {
                for _ in 0..10 {
                    clock_clone.tick();
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("test");
        }

        // After 10 threads × 10 ticks = 100 total ticks
        assert_eq!(clock.now(), 100);
    }

    // Test 22: Concurrent syncs (simulated)
    #[test]
    fn test_concurrent_syncs() {
        use std::sync::Arc;
        use std::thread;

        let clock = Arc::new(LamportClock::new());
        let mut handles = vec![];

        // Spawn 5 threads, each syncing with increasing remote timestamps
        for i in 0..5 {
            let clock_clone = Arc::clone(&clock);
            let remote_ts = (i as u64 + 1) * 10; // 10, 20, 30, 40, 50
            let handle = thread::spawn(move || {
                clock_clone.sync(remote_ts);
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("test");
        }

        // Final value should be at least 51 (max remote + 1)
        assert!(clock.now() >= 51);
    }

    // Test 23: Debug trait
    #[test]
    fn test_debug_trait() {
        let clock = LamportClock::new();
        let debug_str = format!("{:?}", clock);
        assert!(debug_str.contains("LamportClock"));
    }

    // Test 24: Sync with zero timestamp
    #[test]
    fn test_sync_with_zero() {
        let clock = LamportClock::new();
        let ts = clock.sync(0);
        assert_eq!(ts, 1); // max(0, 0) + 1 = 1
    }

    // Test 25: Multiple ticks preserve order
    #[test]
    fn test_multiple_ticks_ordering() {
        let clock = LamportClock::new();

        let timestamps: Vec<u64> = (0..100).map(|_| clock.tick()).collect();

        // Verify all timestamps are strictly increasing
        for i in 1..timestamps.len() {
            assert!(timestamps[i] > timestamps[i - 1]);
        }
    }
}

// Kani formal verification proofs
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Prove tick() always returns a strictly increasing value
    #[kani::proof]
    fn proof_tick_monotonicity() {
        let clock = LamportClock::new();
        let ts1 = clock.tick();
        let ts2 = clock.tick();
        kani::assert(ts2 > ts1, "tick must be strictly monotonic");
    }

    /// Prove sync() result is always greater than both local and remote timestamps
    #[kani::proof]
    fn proof_sync_dominates() {
        let remote: u64 = kani::any();
        kani::assume(remote < u64::MAX - 1);
        let clock = LamportClock::new();
        let result = clock.sync(remote);
        kani::assert(result > remote, "sync result must exceed remote timestamp");
    }

    /// Prove happens_before is irreflexive: not (a -> a)
    #[kani::proof]
    fn proof_happens_before_irreflexive() {
        let a: u64 = kani::any();
        kani::assert(!LamportClock::happens_before(a, a), "happens-before must be irreflexive");
    }

    /// Prove happens_before is transitive: (a -> b) && (b -> c) => (a -> c)
    #[kani::proof]
    fn proof_happens_before_transitive() {
        let a: u64 = kani::any();
        let b: u64 = kani::any();
        let c: u64 = kani::any();
        kani::assume(LamportClock::happens_before(a, b));
        kani::assume(LamportClock::happens_before(b, c));
        kani::assert(LamportClock::happens_before(a, c), "happens-before must be transitive");
    }

    /// Prove is_sampled only depends on bit 0 of trace_flags
    #[kani::proof]
    fn proof_is_sampled_bit0() {
        let flags: u8 = kani::any();
        let ctx =
            TraceContext { version: 0, trace_id: [1; 16], parent_id: [1; 8], trace_flags: flags };
        kani::assert(ctx.is_sampled() == (flags & 0x01 != 0), "is_sampled must check bit 0");
    }
}
