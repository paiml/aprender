//! Headless layer-cache scheduler for the map runtime (MAP-5, contract
//! `aprender-viz-map-layer-cache-v1`, `render_table`).
//!
//! Three layers (base, lens, overlay) each keep rasters for the cameras they were drawn at. A
//! layer renders only when no cached raster covers the current camera (same scale, centre within
//! [`CACHE_MARGIN_PX`] screen pixels) or when an event invalidates it. Zoom motion renders
//! nothing; the uncovered layers render once when the zoom settles. The scheduler reports which
//! layers rendered and whether a frame callback is requested. It never draws.

use std::collections::{BTreeMap, BTreeSet};

use super::{camera::Camera2D, MapError};

/// Screen-pixel half-width of the cached raster beyond the viewport.
pub const CACHE_MARGIN_PX: f64 = 160.0;

/// Identifier of a choropleth lens.
pub type LensId = u32;

/// Identifier of an overlay entity (hovered or selected feature).
pub type EntityId = u64;

/// Number of renders of each layer caused by one event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Renders {
    pub base: u32,
    pub lens: u32,
    pub overlay: u32,
}

impl Renders {
    /// Renders across all three layers.
    pub fn total(&self) -> u32 {
        self.base
            .saturating_add(self.lens)
            .saturating_add(self.overlay)
    }
}

/// Result of handling one event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Outcome {
    pub renders: Renders,
    /// True when the event changed what is on screen, so a frame callback is requested.
    pub frame_requested: bool,
}

/// Change to the overlay layer: hover, selection, or the layer toggle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayChange {
    Hover(Option<EntityId>),
    Select(Option<EntityId>),
    /// Flips the overlay on or off and repaints it once. The flag is reported by
    /// [`Layers::overlay_on`]; it does not suppress the repaint.
    LayerToggle,
}

/// Input to the scheduler.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Event {
    /// An input-free interval. Requests no frame.
    Idle,
    /// Drag by a pixel offset (content follows the pointer).
    Pan { dx_px: f64, dy_px: f64 },
    /// One step of a zoom in progress: moves the camera, renders nothing.
    ZoomMotion { anchor: (f64, f64), scale: f64 },
    /// The zoom has stopped: renders every layer that is not covered at the settled camera.
    ZoomSettle,
    /// Idle work: pre-paints the base layer for the camera a zoom to `next_scale` would reach.
    IdlePrepaint { anchor: (f64, f64), next_scale: f64 },
    /// Makes a lens active. The lens layer is rebuilt once.
    LensSwitch(LensId),
    /// One delta message for a lens. Active lens: the lens layer is rebuilt once. Inactive lens:
    /// the delta is stored and nothing renders.
    LensDelta(LensId),
    /// A change to the overlay.
    Overlay(OverlayChange),
}

/// Rounding slack for the margin test, so a pan of exactly the margin stays inside it.
const MARGIN_TOLERANCE_PX: f64 = 1e-6;

/// A raster drawn for a camera.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Raster {
    centre: (f64, f64),
    scale: f64,
}

impl Raster {
    fn of(camera: &Camera2D) -> Self {
        Self {
            centre: camera.centre(),
            scale: camera.scale(),
        }
    }

    /// Covers `camera` when the scale is identical and the centre is within the cache margin.
    fn covers(&self, camera: &Camera2D) -> bool {
        if self.scale != camera.scale() {
            return false;
        }
        let (cx, cy) = camera.centre();
        let dx = (cx - self.centre.0) * self.scale;
        let dy = (cy - self.centre.1) * self.scale;
        let limit = CACHE_MARGIN_PX + MARGIN_TOLERANCE_PX;
        dx.abs() <= limit && dy.abs() <= limit
    }
}

/// Per-layer cache: the raster for the current camera and one pre-painted raster for the next.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct LayerCache {
    primary: Option<Raster>,
    prefetch: Option<Raster>,
}

impl LayerCache {
    fn covers(&self, camera: &Camera2D) -> bool {
        [self.primary, self.prefetch]
            .iter()
            .flatten()
            .any(|r| r.covers(camera))
    }

    /// Paints the primary raster unless one covers `camera`. Returns the render count.
    fn paint_if_uncovered(&mut self, camera: &Camera2D) -> u32 {
        if self.primary.is_some_and(|r| r.covers(camera)) {
            return 0;
        }
        if self.prefetch.is_some_and(|r| r.covers(camera)) {
            // The pre-painted raster is now the current view: keep it out of the prefetch slot,
            // which the next idle pre-paint overwrites.
            std::mem::swap(&mut self.primary, &mut self.prefetch);
            return 0;
        }
        self.primary = Some(Raster::of(camera));
        1
    }

    /// Pre-paints the prefetch raster unless one covers `camera`. Returns the render count.
    fn prefetch_if_uncovered(&mut self, camera: &Camera2D) -> u32 {
        if self.covers(camera) {
            0
        } else {
            self.prefetch = Some(Raster::of(camera));
            1
        }
    }

    fn invalidate(&mut self) {
        self.primary = None;
        self.prefetch = None;
    }
}

/// The layer-cache state machine: camera, three layer caches, lens and overlay state.
#[derive(Debug, Clone, PartialEq)]
pub struct Layers {
    camera: Camera2D,
    base: LayerCache,
    lens: LayerCache,
    overlay: LayerCache,
    active_lens: Option<LensId>,
    pending: BTreeMap<LensId, u32>,
    hovered: Option<EntityId>,
    selected: Option<EntityId>,
    brushed: BTreeSet<EntityId>,
    overlay_on: bool,
    moving: bool,
    frame_pending: bool,
}

fn quiet() -> Outcome {
    outcome(Renders::default(), false)
}

fn outcome(renders: Renders, moved: bool) -> Outcome {
    Outcome {
        renders,
        frame_requested: moved || renders.total() > 0,
    }
}

impl Layers {
    /// Creates the scheduler and paints every layer once for `camera`. Without an active lens the
    /// lens layer is not drawn.
    pub fn new(camera: Camera2D, active_lens: Option<LensId>) -> (Self, Outcome) {
        let mut layers = Self {
            camera,
            base: LayerCache::default(),
            lens: LayerCache::default(),
            overlay: LayerCache::default(),
            active_lens,
            pending: BTreeMap::new(),
            hovered: None,
            selected: None,
            brushed: BTreeSet::new(),
            overlay_on: true,
            moving: false,
            frame_pending: false,
        };
        let renders = layers.refresh();
        let first = outcome(renders, false);
        layers.frame_pending = first.frame_requested;
        (layers, first)
    }

    /// Consumes the pending frame request. The host calls this when it schedules a frame; `false`
    /// means it must not request one.
    pub fn take_frame(&mut self) -> bool {
        std::mem::take(&mut self.frame_pending)
    }

    /// Applies one event and reports the renders and frame request it causes.
    pub fn handle(&mut self, event: Event) -> Result<Outcome, MapError> {
        let out = match event {
            Event::Idle => Ok(quiet()),
            Event::Pan { dx_px, dy_px } => self.pan(dx_px, dy_px),
            Event::ZoomMotion { anchor, scale } => self.zoom_motion(anchor, scale),
            Event::ZoomSettle => Ok(self.settle()),
            Event::IdlePrepaint { anchor, next_scale } => self.prepaint(anchor, next_scale),
            Event::LensSwitch(id) => Ok(self.lens_switch(id)),
            Event::LensDelta(id) => Ok(self.lens_delta(id)),
            Event::Overlay(change) => Ok(self.overlay_change(change)),
        }?;
        self.frame_pending |= out.frame_requested;
        Ok(out)
    }

    /// Current camera.
    pub fn camera(&self) -> &Camera2D {
        &self.camera
    }

    /// True between a zoom motion and the zoom settle.
    pub fn is_moving(&self) -> bool {
        self.moving
    }

    /// The active lens, if any.
    pub fn active_lens(&self) -> Option<LensId> {
        self.active_lens
    }

    /// Deltas stored for `lens` while it was inactive and not yet applied.
    pub fn pending_deltas(&self, lens: LensId) -> u32 {
        self.pending.get(&lens).copied().unwrap_or(0)
    }

    /// Hovered entity.
    pub fn hovered(&self) -> Option<EntityId> {
        self.hovered
    }

    /// Selected entity.
    pub fn selected(&self) -> Option<EntityId> {
        self.selected
    }

    /// Overlay flag flipped by [`OverlayChange::LayerToggle`].
    pub fn overlay_on(&self) -> bool {
        self.overlay_on
    }

    /// Paints the uncovered layers, or nothing while the camera is moving.
    fn refresh(&mut self) -> Renders {
        if self.moving {
            return Renders::default();
        }
        let cam = self.camera;
        let lens = if self.active_lens.is_some() {
            self.lens.paint_if_uncovered(&cam)
        } else {
            0
        };
        Renders {
            base: self.base.paint_if_uncovered(&cam),
            lens,
            overlay: self.overlay.paint_if_uncovered(&cam),
        }
    }

    fn pan(&mut self, dx_px: f64, dy_px: f64) -> Result<Outcome, MapError> {
        self.camera.pan(dx_px, dy_px)?;
        Ok(outcome(self.refresh(), dx_px != 0.0 || dy_px != 0.0))
    }

    fn zoom_motion(&mut self, anchor: (f64, f64), scale: f64) -> Result<Outcome, MapError> {
        let before = self.camera.scale();
        self.camera.zoom_about(anchor, scale)?;
        self.moving = true;
        Ok(outcome(Renders::default(), self.camera.scale() != before))
    }

    fn settle(&mut self) -> Outcome {
        self.moving = false;
        outcome(self.refresh(), false)
    }

    fn prepaint(&mut self, anchor: (f64, f64), next_scale: f64) -> Result<Outcome, MapError> {
        if self.moving {
            return Ok(quiet());
        }
        let mut next = self.camera;
        next.zoom_about(anchor, next_scale)?;
        let base = self.base.prefetch_if_uncovered(&next);
        Ok(Outcome {
            renders: Renders {
                base,
                ..Renders::default()
            },
            frame_requested: false,
        })
    }

    fn lens_switch(&mut self, id: LensId) -> Outcome {
        if self.active_lens == Some(id) {
            return quiet();
        }
        self.active_lens = Some(id);
        self.pending.remove(&id);
        self.lens.invalidate();
        outcome(self.refresh(), false)
    }

    fn lens_delta(&mut self, id: LensId) -> Outcome {
        if self.active_lens != Some(id) {
            let count = self.pending.entry(id).or_insert(0);
            *count = count.saturating_add(1);
            return quiet();
        }
        self.lens.invalidate();
        outcome(self.refresh(), false)
    }

    fn overlay_change(&mut self, change: OverlayChange) -> Outcome {
        match change {
            OverlayChange::Hover(h) => self.hovered = h,
            OverlayChange::Select(s) => self.selected = s,
            OverlayChange::LayerToggle => self.overlay_on = !self.overlay_on,
        }
        self.overlay.invalidate();
        outcome(self.refresh(), false)
    }

    /// Replaces the brushed selection set and repaints the overlay once.
    pub fn set_brush(&mut self, cells: BTreeSet<EntityId>) -> Outcome {
        self.brushed = cells;
        self.overlay.invalidate();
        let out = outcome(self.refresh(), false);
        self.frame_pending |= out.frame_requested;
        out
    }

    /// Cells in the brushed selection set.
    pub fn brushed(&self) -> &BTreeSet<EntityId> {
        &self.brushed
    }
}

#[cfg(test)]
#[path = "layers_tests.rs"]
mod tests;
