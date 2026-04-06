//! AWP-09: Smart Payload Wake Skip

/// Wake skip optimization state.
///
/// Tracks whether a wakeup is actually needed or can be skipped
/// to avoid unnecessary context switches.
#[derive(Debug, Default)]
pub struct WakeSkipState {
    /// Number of items pending
    pending_items: usize,
    /// Whether there's a registered waker
    has_waker: bool,
    /// Last poll had work to do
    last_poll_had_work: bool,
    /// Consecutive empty polls
    empty_poll_count: u32,
    /// Threshold for skipping wakes
    skip_threshold: u32,
}

impl WakeSkipState {
    /// Create with skip threshold.
    pub fn new(skip_threshold: u32) -> Self {
        Self {
            pending_items: 0,
            has_waker: false,
            last_poll_had_work: false,
            empty_poll_count: 0,
            skip_threshold,
        }
    }

    /// Register that a waker exists.
    pub fn register_waker(&mut self) {
        self.has_waker = true;
    }

    /// Clear waker registration.
    pub fn clear_waker(&mut self) {
        self.has_waker = false;
    }

    /// Add pending items.
    pub fn add_pending(&mut self, count: usize) {
        self.pending_items += count;
    }

    /// Remove pending items.
    pub fn remove_pending(&mut self, count: usize) {
        self.pending_items = self.pending_items.saturating_sub(count);
    }

    /// Record poll result.
    pub fn record_poll(&mut self, had_work: bool) {
        self.last_poll_had_work = had_work;
        if had_work {
            self.empty_poll_count = 0;
        } else {
            self.empty_poll_count += 1;
        }
    }

    /// Determine if wake should be skipped.
    #[must_use]
    pub fn should_skip_wake(&self) -> bool {
        // Skip if:
        // 1. No waker registered
        // 2. Already has pending items (will be polled anyway)
        // 3. Had recent empty polls (probably will be empty again)
        if !self.has_waker {
            return true;
        }
        if self.pending_items > 0 && self.last_poll_had_work {
            return true; // Already has work queued
        }
        if self.empty_poll_count >= self.skip_threshold {
            return true; // Likely to be empty again
        }
        false
    }

    /// Check if wake is needed.
    #[must_use]
    pub fn needs_wake(&self) -> bool {
        !self.should_skip_wake() && self.pending_items > 0
    }

    /// Get pending count.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.pending_items
    }

    /// Reset empty poll tracking (after successful wake).
    pub fn reset_tracking(&mut self) {
        self.empty_poll_count = 0;
    }
}
