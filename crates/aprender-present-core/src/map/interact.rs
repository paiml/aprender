//! Headless interaction core for the map runtime (APR-MAP-001 MAP-6 slice 2).
//!
//! Pointer and keyboard input become [`Layers`] events and [`HistoryPolicy`] updates. No browser
//! and no clock: the caller passes `now_ms`. Every method returns a [`Report`] with the renders
//! the layer scheduler performed and the history write the host must make, if any.

use super::{
    camera::Camera2D,
    hex::HexGrid,
    layers::{EntityId, Event, Layers, LensId, Outcome, OverlayChange, Renders},
    pick::pick,
    urlstate::{HistoryPolicy, Nav, UrlEvent, UrlState},
    MapError, MAX_SCALE, MIN_SCALE,
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
        Some(match name {
            "ArrowLeft" => Self::ArrowLeft,
            "ArrowRight" => Self::ArrowRight,
            "ArrowUp" => Self::ArrowUp,
            "ArrowDown" => Self::ArrowDown,
            "+" => Self::ZoomIn,
            "-" => Self::ZoomOut,
            "Escape" => Self::Escape,
            _ => return None,
        })
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
    layers: Layers,
    history: HistoryPolicy,
}

/// Report for an input that changed nothing.
fn quiet() -> Report {
    Report {
        renders: Renders::default(),
        frame_requested: false,
        nav: Nav::Nothing,
    }
}

fn report(out: Outcome, nav: Nav) -> Report {
    Report {
        renders: out.renders,
        frame_requested: out.frame_requested,
        nav,
    }
}

fn add(a: Renders, b: Renders) -> Renders {
    Renders {
        base: a.base.saturating_add(b.base),
        lens: a.lens.saturating_add(b.lens),
        overlay: a.overlay.saturating_add(b.overlay),
    }
}

impl Interaction {
    /// Starts the interaction at `camera` with `active_lens`. Paints every layer once.
    pub fn new(grid: HexGrid, camera: Camera2D, active_lens: Option<LensId>) -> (Self, Report) {
        let (layers, first) = Layers::new(camera, active_lens);
        let history = HistoryPolicy::new(UrlState::new(Some(camera), None, active_lens));
        (
            Self {
                grid,
                layers,
                history,
            },
            report(first, Nav::Nothing),
        )
    }

    /// Current camera.
    pub fn camera(&self) -> &Camera2D {
        self.layers.camera()
    }

    /// Entity under the pointer, if any.
    pub fn hovered(&self) -> Option<EntityId> {
        self.layers.hovered()
    }

    /// Selected entity, if any.
    pub fn selected(&self) -> Option<EntityId> {
        self.layers.selected()
    }

    /// Overlay flag.
    pub fn overlay_on(&self) -> bool {
        self.layers.overlay_on()
    }

    /// The URL state as the history policy holds it.
    pub fn url_state(&self) -> &UrlState {
        self.history.state()
    }

    /// Pointer over a screen pixel: sets the hovered cell. Writes no history.
    pub fn hover(&mut self, screen: (f64, f64)) -> Result<Report, MapError> {
        let cell = pick(&self.grid, self.layers.camera(), screen)?;
        let out = self
            .layers
            .handle(Event::Overlay(OverlayChange::Hover(cell.map(u64::from))))?;
        Ok(report(out, Nav::Nothing))
    }

    /// Pointer click on a screen pixel: selects the cell under it, or clears the selection off
    /// the grid.
    pub fn click(&mut self, screen: (f64, f64), now_ms: u64) -> Result<Report, MapError> {
        let cell = pick(&self.grid, self.layers.camera(), screen)?;
        self.select(cell.map(u64::from), now_ms)
    }

    /// The layer toggle: flips the overlay flag and repaints the overlay once.
    pub fn toggle_overlay(&mut self) -> Result<Report, MapError> {
        let out = self
            .layers
            .handle(Event::Overlay(OverlayChange::LayerToggle))?;
        Ok(report(out, Nav::Nothing))
    }

    /// A key press by DOM name. Unbound names are a no-op.
    pub fn key(&mut self, name: &str, now_ms: u64) -> Result<Report, MapError> {
        let Some(key) = Key::from_name(name) else {
            return Ok(quiet());
        };
        match key {
            Key::ArrowLeft => self.pan(PAN_STEP_PX, 0.0, now_ms),
            Key::ArrowRight => self.pan(-PAN_STEP_PX, 0.0, now_ms),
            Key::ArrowUp => self.pan(0.0, PAN_STEP_PX, now_ms),
            Key::ArrowDown => self.pan(0.0, -PAN_STEP_PX, now_ms),
            Key::ZoomIn => self.zoom(ZOOM_FACTOR, now_ms),
            Key::ZoomOut => self.zoom(1.0 / ZOOM_FACTOR, now_ms),
            Key::Escape => self.select(None, now_ms),
        }
    }

    /// Writes the debounced camera replace once the camera has been still long enough.
    pub fn poll_history(&mut self, now_ms: u64) -> Result<Nav, MapError> {
        self.history.poll(now_ms)
    }

    /// Sets the selection. A change adds one history entry; the overlay repaints either way.
    fn select(&mut self, sel: Option<EntityId>, now_ms: u64) -> Result<Report, MapError> {
        let nav = if self.layers.selected() == sel {
            Nav::Nothing
        } else {
            self.history.apply(UrlEvent::Select(sel), now_ms)?
        };
        let out = self
            .layers
            .handle(Event::Overlay(OverlayChange::Select(sel)))?;
        Ok(report(out, nav))
    }

    /// Drags the map by a pixel offset and records the camera as a debounced replace.
    fn pan(&mut self, dx_px: f64, dy_px: f64, now_ms: u64) -> Result<Report, MapError> {
        let out = self.layers.handle(Event::Pan { dx_px, dy_px })?;
        let nav = self
            .history
            .apply(UrlEvent::Camera(*self.layers.camera()), now_ms)?;
        Ok(report(out, nav))
    }

    /// Scales by `factor` about the viewport centre. A scale already at its limit does nothing.
    fn zoom(&mut self, factor: f64, now_ms: u64) -> Result<Report, MapError> {
        let current = self.layers.camera().scale();
        let target = (current * factor).clamp(MIN_SCALE, MAX_SCALE);
        if target == current {
            return Ok(quiet());
        }
        let anchor = self.layers.camera().screen_centre();
        let motion = self.layers.handle(Event::ZoomMotion {
            anchor,
            scale: target,
        })?;
        let settle = self.layers.handle(Event::ZoomSettle)?;
        let nav = self
            .history
            .apply(UrlEvent::Camera(*self.layers.camera()), now_ms)?;
        let out = Outcome {
            renders: add(motion.renders, settle.renders),
            frame_requested: motion.frame_requested || settle.frame_requested,
        };
        Ok(report(out, nav))
    }
}

#[cfg(test)]
#[path = "interact_tests.rs"]
mod tests;
