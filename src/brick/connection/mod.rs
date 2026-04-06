//! Connection Management
//!
//! AWP-06: Managed connections with TTL, idle timeout, and health tracking.
//! AWP-10: Keep-Alive normalization for HTTP connections.
//! AWP-12: Bitflags connection state for efficient state tracking.

use std::time::{Duration, Instant};

// ----------------------------------------------------------------------------
// AWP-06: Connection TTL + Health Check
// ----------------------------------------------------------------------------

/// Connection with TTL and health tracking.
#[derive(Debug)]
pub struct ManagedConnection<T> {
    /// The underlying connection
    inner: T,
    /// When the connection was created
    created_at: Instant,
    /// When the connection was last used
    last_used: Instant,
    /// Maximum lifetime (TTL)
    max_lifetime: Duration,
    /// Maximum idle time
    max_idle: Duration,
    /// Health check failures
    health_failures: usize,
}

impl<T> ManagedConnection<T> {
    /// Create a new managed connection.
    pub fn new(inner: T, max_lifetime: Duration, max_idle: Duration) -> Self {
        let now = Instant::now();
        Self { inner, created_at: now, last_used: now, max_lifetime, max_idle, health_failures: 0 }
    }

    /// Check if the connection is still valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let now = Instant::now();
        let not_expired = now.duration_since(self.created_at) < self.max_lifetime;
        let not_idle = now.duration_since(self.last_used) < self.max_idle;
        let healthy = self.health_failures < 3;
        not_expired && not_idle && healthy
    }

    /// Check if the connection has expired (TTL exceeded).
    #[must_use]
    pub fn is_expired(&self) -> bool {
        self.created_at.elapsed() >= self.max_lifetime
    }

    /// Check if the connection is idle.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.last_used.elapsed() >= self.max_idle
    }

    /// Mark the connection as used.
    pub fn touch(&mut self) {
        self.last_used = Instant::now();
    }

    /// Record a health check failure.
    pub fn record_health_failure(&mut self) {
        self.health_failures += 1;
    }

    /// Reset health failure count.
    pub fn reset_health(&mut self) {
        self.health_failures = 0;
    }

    /// Get the underlying connection.
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Get mutable access to the underlying connection.
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    /// Consume and return the underlying connection.
    pub fn into_inner(self) -> T {
        self.inner
    }

    /// Get connection age.
    #[must_use]
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get idle time.
    #[must_use]
    pub fn idle_time(&self) -> Duration {
        self.last_used.elapsed()
    }

    /// Get health failure count (for testing/diagnostics).
    #[must_use]
    pub fn health_failures(&self) -> usize {
        self.health_failures
    }
}

// ----------------------------------------------------------------------------
// AWP-10: Keep-Alive Normalization
// ----------------------------------------------------------------------------

/// Normalized keep-alive configuration.
///
/// Canonicalizes various keep-alive settings into a standard form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeepAliveConfig {
    /// Whether keep-alive is enabled
    pub enabled: bool,
    /// Timeout duration in seconds
    pub timeout_secs: u32,
    /// Maximum number of requests per connection
    pub max_requests: u32,
}

impl KeepAliveConfig {
    /// Create with default values.
    pub fn new() -> Self {
        Self { enabled: true, timeout_secs: 60, max_requests: 100 }
    }

    /// Disabled keep-alive.
    pub fn disabled() -> Self {
        Self { enabled: false, timeout_secs: 0, max_requests: 0 }
    }

    /// Parse from HTTP header value (e.g., "timeout=5, max=100").
    pub fn from_header(header: &str) -> Self {
        let mut config = Self::new();

        for part in header.split(',') {
            let part = part.trim();
            if let Some((key, val)) = part.split_once('=') {
                let key = key.trim().to_lowercase();
                let val = val.trim();

                match key.as_str() {
                    "timeout" => {
                        if let Ok(t) = val.parse() {
                            config.timeout_secs = t;
                        }
                    }
                    "max" => {
                        if let Ok(m) = val.parse() {
                            config.max_requests = m;
                        }
                    }
                    _ => {} // Ignore unknown header parameters
                }
            }
        }

        config
    }

    /// Check if connection should be kept alive after n requests.
    #[must_use]
    pub fn should_keep_alive(&self, request_count: u32) -> bool {
        self.enabled && request_count < self.max_requests
    }
}

impl Default for KeepAliveConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ----------------------------------------------------------------------------
// AWP-12: Bitflags Connection State
// ----------------------------------------------------------------------------

/// Compact connection state using bitflags.
///
/// Efficiently represents multiple boolean states in a single byte.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConnectionState(u8);

impl ConnectionState {
    /// Connection is open
    pub const OPEN: u8 = 0b0000_0001;
    /// Connection is readable
    pub const READABLE: u8 = 0b0000_0010;
    /// Connection is writable
    pub const WRITABLE: u8 = 0b0000_0100;
    /// Connection has pending data
    pub const HAS_PENDING: u8 = 0b0000_1000;
    /// Connection is in keep-alive mode
    pub const KEEP_ALIVE: u8 = 0b0001_0000;
    /// Connection upgrade requested (e.g., WebSocket)
    pub const UPGRADE: u8 = 0b0010_0000;
    /// Connection is closing
    pub const CLOSING: u8 = 0b0100_0000;
    /// Connection has error
    pub const ERROR: u8 = 0b1000_0000;

    /// Create new state with no flags set.
    #[must_use]
    pub fn new() -> Self {
        Self(0)
    }

    /// Create state with initial open + writable.
    #[must_use]
    pub fn open_connection() -> Self {
        Self(Self::OPEN | Self::WRITABLE)
    }

    /// Set a flag.
    pub fn set(&mut self, flag: u8) {
        self.0 |= flag;
    }

    /// Clear a flag.
    pub fn clear(&mut self, flag: u8) {
        self.0 &= !flag;
    }

    /// Check if flag is set.
    #[must_use]
    pub fn is_set(&self, flag: u8) -> bool {
        self.0 & flag != 0
    }

    /// Check if connection is open and healthy.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.is_set(Self::OPEN) && !self.is_set(Self::ERROR) && !self.is_set(Self::CLOSING)
    }

    /// Check if connection can read.
    #[must_use]
    pub fn can_read(&self) -> bool {
        self.is_set(Self::OPEN) && self.is_set(Self::READABLE)
    }

    /// Check if connection can write.
    #[must_use]
    pub fn can_write(&self) -> bool {
        self.is_set(Self::OPEN) && self.is_set(Self::WRITABLE) && !self.is_set(Self::CLOSING)
    }

    /// Get raw bits.
    #[must_use]
    pub fn bits(&self) -> u8 {
        self.0
    }
}

#[cfg(test)]
mod tests;
