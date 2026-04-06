#![allow(missing_docs)]
//! AWP-04: HTTP/2 Stream Capacity

/// HTTP/2 flow control window state.
///
/// Tracks send and receive window sizes for stream-level flow control.
#[derive(Debug, Clone)]
pub struct StreamCapacity {
    /// Connection-level send window
    connection_send: i32,
    /// Stream-level send window
    stream_send: i32,
    /// Receive window (how much we can receive)
    receive_window: i32,
    /// Initial window size
    initial_window: i32,
    /// Whether stream is blocked on flow control
    is_blocked: bool,
}

impl StreamCapacity {
    /// Default window size (HTTP/2 spec: 65535).
    pub const DEFAULT_WINDOW: i32 = 65535;

    /// Create with default windows.
    pub fn new() -> Self {
        Self {
            connection_send: Self::DEFAULT_WINDOW,
            stream_send: Self::DEFAULT_WINDOW,
            receive_window: Self::DEFAULT_WINDOW,
            initial_window: Self::DEFAULT_WINDOW,
            is_blocked: false,
        }
    }

    /// Create with custom initial window.
    pub fn with_initial_window(initial: i32) -> Self {
        Self {
            connection_send: initial,
            stream_send: initial,
            receive_window: initial,
            initial_window: initial,
            is_blocked: false,
        }
    }

    /// Reserve capacity for sending.
    pub fn reserve_send(&mut self, bytes: i32) -> Result<(), FlowControlError> {
        if bytes < 0 {
            return Err(FlowControlError::NegativeReservation);
        }

        let available = self.available_send();
        if bytes > available {
            self.is_blocked = true;
            return Err(FlowControlError::InsufficientCapacity { requested: bytes, available });
        }

        self.stream_send -= bytes;
        self.connection_send -= bytes;
        self.is_blocked = false;
        Ok(())
    }

    /// Release send capacity (after WINDOW_UPDATE).
    pub fn release_send(&mut self, bytes: i32) {
        self.stream_send += bytes;
        self.connection_send += bytes;
        if self.available_send() > 0 {
            self.is_blocked = false;
        }
    }

    /// Consume receive window (data received).
    pub fn consume_receive(&mut self, bytes: i32) {
        self.receive_window -= bytes;
    }

    /// Replenish receive window (sending WINDOW_UPDATE).
    pub fn replenish_receive(&mut self, bytes: i32) {
        self.receive_window += bytes;
    }

    /// Get available send capacity.
    #[must_use]
    pub fn available_send(&self) -> i32 {
        self.stream_send.min(self.connection_send).max(0)
    }

    /// Get available receive capacity.
    #[must_use]
    pub fn available_receive(&self) -> i32 {
        self.receive_window.max(0)
    }

    /// Check if stream is blocked on flow control.
    #[must_use]
    pub fn is_blocked(&self) -> bool {
        self.is_blocked
    }

    /// Check if receive window needs replenishment.
    #[must_use]
    pub fn needs_window_update(&self) -> bool {
        self.receive_window < self.initial_window / 2
    }
}

impl Default for StreamCapacity {
    fn default() -> Self {
        Self::new()
    }
}

/// Flow control errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlowControlError {
    /// Tried to reserve negative bytes
    NegativeReservation,
    /// Not enough capacity
    InsufficientCapacity { requested: i32, available: i32 },
}
