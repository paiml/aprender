//! Headless interaction core for the map runtime (APR-MAP-001 MAP-6 slice 2).
//!
//! Pointer and keyboard input become [`Layers`] events and [`HistoryPolicy`] updates. No browser
//! and no clock: the caller passes `now_ms`. Every method returns a [`Report`] with the renders
//! the layer scheduler performed and the history write the host must make, if any.

use super::{
    camera::Camera2D,
    hex::HexGrid,
    layers::{EntityId, LensId, Renders},
    urlstate::{Nav, UrlState},
    MapError,
};

/// Pan step of an arrow key, in screen pixels.
pub const PAN_STEP_PX: f64 = 50.0;
/// Zoom factor of the `+` and `-` keys, anchored at the viewport centre.
pub const ZOOM_FACTOR: f64 = 2.0;

/// Keys with a binding. Names are the DOM `KeyboardEvent.key` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    ArrowLeft,
    ArrowRight,
    ArrowUp,
    ArrowDown,
    ZoomIn,
    ZoomOut,
    Escape,
}

impl Key {
    /// The binding for a DOM key name, or `None` when the key is not bound.
    pub fn from_name(name: &str) -> Option<Self> {
        unimplemented!("MAP-6 slice 2 RED: Key::from_name {name}")
    }
}

/// What one input did: renders per layer, whether a frame is requested, and the history write.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    pub renders: Renders,
    pub frame_requested: bool,
    pub nav: Nav,
}

/// Interaction state: the grid, the layer scheduler and the URL history policy.
#[derive(Debug, Clone, PartialEq)]
pub struct Interaction {
    grid: HexGrid,
}

impl Interaction {
    /// Starts the interaction at `camera` with `active_lens`. Paints every layer once.
    pub fn new(grid: HexGrid, camera: Camera2D, active_lens: Option<LensId>) -> (Self, Report) {
        unimplemented!("MAP-6 slice 2 RED: Interaction::new {camera:?} {active_lens:?} {grid:?}")
    }

    /// Current camera.
    pub fn camera(&self) -> &Camera2D {
        unimplemented!("MAP-6 slice 2 RED: camera")
    }

    /// Entity under the pointer, if any.
    pub fn hovered(&self) -> Option<EntityId> {
        unimplemented!("MAP-6 slice 2 RED: hovered")
    }

    /// Selected entity, if any.
    pub fn selected(&self) -> Option<EntityId> {
        unimplemented!("MAP-6 slice 2 RED: selected")
    }

    /// Overlay flag.
    pub fn overlay_on(&self) -> bool {
        unimplemented!("MAP-6 slice 2 RED: overlay_on")
    }

    /// The URL state as the history policy holds it.
    pub fn url_state(&self) -> &UrlState {
        unimplemented!("MAP-6 slice 2 RED: url_state")
    }

    /// Pointer over a screen pixel: sets the hovered cell. Writes no history.
    pub fn hover(&mut self, screen: (f64, f64)) -> Result<Report, MapError> {
        unimplemented!("MAP-6 slice 2 RED: hover {screen:?}")
    }

    /// Pointer click on a screen pixel: selects the cell under it, or clears the selection off
    /// the grid.
    pub fn click(&mut self, screen: (f64, f64), now_ms: u64) -> Result<Report, MapError> {
        unimplemented!("MAP-6 slice 2 RED: click {screen:?} {now_ms}")
    }

    /// The layer toggle: flips the overlay flag and repaints the overlay once.
    pub fn toggle_overlay(&mut self) -> Result<Report, MapError> {
        unimplemented!("MAP-6 slice 2 RED: toggle_overlay")
    }

    /// A key press by DOM name. Unbound names are a no-op.
    pub fn key(&mut self, name: &str, now_ms: u64) -> Result<Report, MapError> {
        unimplemented!("MAP-6 slice 2 RED: key {name} {now_ms}")
    }

    /// Writes the debounced camera replace once the camera has been still long enough.
    pub fn poll_history(&mut self, now_ms: u64) -> Result<Nav, MapError> {
        unimplemented!("MAP-6 slice 2 RED: poll_history {now_ms}")
    }
}

#[cfg(test)]
#[path = "interact_tests.rs"]
mod tests;
