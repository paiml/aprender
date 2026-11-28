//! Headless history policy for the map runtime (MAP-6 slice 1).
//!
//! Selections and lens switches each push one history entry. Camera moves add
//! no entries: the last move is written with `replace_state` once the debounce
//! has elapsed. Time is an explicit tick, so there are no timers and no web_sys.

use super::codec::encode;
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
    fn push_state(&mut self, hash: &str) {
        self.entries += 1;
        self.current = hash.to_owned();
    }

    fn replace_state(&mut self, hash: &str) {
        self.current = hash.to_owned();
    }
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

    /// Records a camera move at tick `now`. Writes no history entry; the
    /// history sink is taken so the no-write rule is part of the signature.
    ///
    /// # Errors
    /// Returns `NonFinite` for a non-finite camera.
    pub fn camera_move<H: HistorySink>(
        &mut self,
        now: u64,
        camera: Camera,
        _history: &mut H,
    ) -> Result<(), MapUrlError> {
        if !camera.is_finite() {
            return Err(MapUrlError::NonFinite);
        }
        self.state.camera = Some(camera);
        self.camera_moved_at = Some(now);
        Ok(())
    }

    /// Writes the debounced camera replace if `now` is at least `debounce_ticks`
    /// past the last move. Returns true if it wrote.
    ///
    /// # Errors
    /// Returns an error if the state cannot be encoded.
    pub fn tick<H: HistorySink>(&mut self, now: u64, history: &mut H) -> Result<bool, MapUrlError> {
        match self.camera_moved_at {
            Some(moved) if now.saturating_sub(moved) >= self.debounce_ticks => {
                let hash = encode(&self.state)?;
                history.replace_state(&hash);
                self.camera_moved_at = None;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Selects a city and pushes one history entry.
    ///
    /// # Errors
    /// Returns an error if the state cannot be encoded.
    pub fn select<H: HistorySink>(
        &mut self,
        city: Option<&str>,
        history: &mut H,
    ) -> Result<(), MapUrlError> {
        self.state.city = city.map(str::to_owned);
        self.push(history)
    }

    /// Switches the lens and pushes one history entry.
    ///
    /// # Errors
    /// Returns an error if the state cannot be encoded.
    pub fn switch_lens<H: HistorySink>(
        &mut self,
        lens: &str,
        history: &mut H,
    ) -> Result<(), MapUrlError> {
        self.state.lens = Some(lens.to_owned());
        self.push(history)
    }

    /// Pushes the current state as one entry. The pushed hash carries the
    /// latest camera, so any pending camera replace is absorbed.
    fn push<H: HistorySink>(&mut self, history: &mut H) -> Result<(), MapUrlError> {
        let hash = encode(&self.state)?;
        history.push_state(&hash);
        self.camera_moved_at = None;
        Ok(())
    }
}
