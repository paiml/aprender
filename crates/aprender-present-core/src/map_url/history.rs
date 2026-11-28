//! Headless history policy for the map runtime (MAP-6 slice 1).
//!
//! Selections and lens switches each push one history entry. Camera moves add
//! no entries: the last move is written with `replace_state` once the debounce
//! has elapsed. Time is an explicit tick, so there are no timers and no web_sys.
//!
//! RED stub: the policy methods below do nothing yet. The GREEN commit fills them in.

use super::state::{Camera, MapUrlError, MapUrlState};

/// Where the policy writes history. The browser adapter implements this;
/// tests use `FakeHistory`.
pub trait HistorySink {
    /// Adds one entry with `hash` as its URL.
    fn push_state(&mut self, hash: &str);
    /// Replaces the current entry's URL with `hash`, adding no entry.
    fn replace_state(&mut self, hash: &str);
}

/// Fake history: an entry count and the current hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FakeHistory {
    entries: usize,
    current: String,
}

impl FakeHistory {
    /// A history with one entry whose URL is `initial_hash`.
    #[must_use]
    pub fn new(initial_hash: &str) -> Self {
        Self {
            entries: 1,
            current: initial_hash.to_owned(),
        }
    }

    /// Number of history entries.
    #[must_use]
    pub fn entries(&self) -> usize {
        self.entries
    }

    /// URL of the current entry.
    #[must_use]
    pub fn current_hash(&self) -> &str {
        &self.current
    }
}

impl HistorySink for FakeHistory {
    fn push_state(&mut self, _hash: &str) {}

    fn replace_state(&mut self, _hash: &str) {}
}

/// Owns the URL state and decides when each change touches history.
#[derive(Debug, Clone)]
pub struct HistoryPolicy {
    state: MapUrlState,
    debounce_ticks: u64,
    camera_moved_at: Option<u64>,
}

impl HistoryPolicy {
    /// A policy over `state`. A camera replace fires `debounce_ticks` after the last move.
    #[must_use]
    pub fn new(state: MapUrlState, debounce_ticks: u64) -> Self {
        Self {
            state,
            debounce_ticks,
            camera_moved_at: None,
        }
    }

    /// The current URL state.
    #[must_use]
    pub fn state(&self) -> &MapUrlState {
        &self.state
    }

    /// True while a camera move is waiting for its debounced replace.
    #[must_use]
    pub fn camera_pending(&self) -> bool {
        self.camera_moved_at.is_some()
    }

    /// Records a camera move at tick `now`. Writes no history entry.
    ///
    /// # Errors
    /// Returns `NonFinite` for a non-finite camera.
    pub fn camera_move<H: HistorySink>(
        &mut self,
        _now: u64,
        _camera: Camera,
        _history: &mut H,
    ) -> Result<(), MapUrlError> {
        Ok(())
    }

    /// Writes the debounced camera replace if `now` is at least `debounce_ticks`
    /// past the last move. Returns true if it wrote.
    ///
    /// # Errors
    /// Returns an error if the state cannot be encoded.
    pub fn tick<H: HistorySink>(
        &mut self,
        _now: u64,
        _history: &mut H,
    ) -> Result<bool, MapUrlError> {
        Ok(false)
    }

    /// Selects a city and pushes one history entry.
    ///
    /// # Errors
    /// Returns an error if the state cannot be encoded.
    pub fn select<H: HistorySink>(
        &mut self,
        _city: Option<&str>,
        _history: &mut H,
    ) -> Result<(), MapUrlError> {
        Ok(())
    }

    /// Switches the lens and pushes one history entry.
    ///
    /// # Errors
    /// Returns an error if the state cannot be encoded.
    pub fn switch_lens<H: HistorySink>(
        &mut self,
        _lens: &str,
        _history: &mut H,
    ) -> Result<(), MapUrlError> {
        Ok(())
    }
}
