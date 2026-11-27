//! Camera for the map runtime (MAP-5, contract `aprender-viz-map-camera-v1`).
//!
//! A screen point `a` maps to the world point `w(a) = c + (a - screen_centre) / s`, where `s` is
//! screen pixels per world unit and `c` is the camera centre in world units.

use super::{check_finite, check_non_negative, check_point, check_scale, MapError};

/// A 2D camera: world centre, screen pixels per world unit, and viewport size in pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera2D {
    centre: (f64, f64),
    scale: f64,
    viewport: (f64, f64),
}

impl Camera2D {
    /// Builds a camera. Rejects non-finite values, a non-positive scale and a negative viewport.
    pub fn new(centre: (f64, f64), scale: f64, viewport: (f64, f64)) -> Result<Self, MapError> {
        check_point(centre, "centre")?;
        let scale = check_scale(scale, "scale")?;
        check_point(viewport, "viewport")?;
        if viewport.0 < 0.0 || viewport.1 < 0.0 {
            return Err(MapError::Negative("viewport"));
        }
        Ok(Self {
            centre,
            scale,
            viewport,
        })
    }

    /// World point at the centre of the camera.
    pub fn centre(&self) -> (f64, f64) {
        self.centre
    }

    /// Screen pixels per world unit.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Viewport size in pixels.
    pub fn viewport(&self) -> (f64, f64) {
        self.viewport
    }

    /// Screen pixel at the middle of the viewport.
    pub fn screen_centre(&self) -> (f64, f64) {
        (self.viewport.0 / 2.0, self.viewport.1 / 2.0)
    }

    /// World point under a screen pixel.
    pub fn world_at(&self, screen: (f64, f64)) -> (f64, f64) {
        let sc = self.screen_centre();
        (
            self.centre.0 + (screen.0 - sc.0) / self.scale,
            self.centre.1 + (screen.1 - sc.1) / self.scale,
        )
    }

    /// Screen pixel of a world point.
    pub fn screen_at(&self, world: (f64, f64)) -> (f64, f64) {
        let sc = self.screen_centre();
        (
            sc.0 + (world.0 - self.centre.0) * self.scale,
            sc.1 + (world.1 - self.centre.1) * self.scale,
        )
    }

    /// Drags the map by `(dx_px, dy_px)`: the content follows the pointer, so the centre moves
    /// by the opposite amount in world units. Leaves the camera unchanged on error.
    pub fn pan(&mut self, dx_px: f64, dy_px: f64) -> Result<(), MapError> {
        check_finite(dx_px, "pan")?;
        check_finite(dy_px, "pan")?;
        let next = (
            self.centre.0 - dx_px / self.scale,
            self.centre.1 - dy_px / self.scale,
        );
        self.set_centre(next)
    }

    /// Changes the scale to `new_scale` while keeping the world point under `anchor` fixed.
    /// Leaves the camera unchanged on error.
    pub fn zoom_about(&mut self, anchor: (f64, f64), new_scale: f64) -> Result<(), MapError> {
        check_point(anchor, "anchor")?;
        let new_scale = check_scale(new_scale, "scale")?;
        let w = self.world_at(anchor);
        let sc = self.screen_centre();
        let centre = (
            w.0 - (anchor.0 - sc.0) / new_scale,
            w.1 - (anchor.1 - sc.1) / new_scale,
        );
        check_point(centre, "centre")?;
        self.centre = centre;
        self.scale = new_scale;
        Ok(())
    }

    fn set_centre(&mut self, centre: (f64, f64)) -> Result<(), MapError> {
        check_point(centre, "centre")?;
        self.centre = centre;
        Ok(())
    }
}

/// Eased zoom about a fixed screen anchor. Each step is an anchored zoom, so the world point
/// under the anchor stays where it was; the last step sets the target scale exactly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZoomAnim {
    anchor: (f64, f64),
    from: f64,
    to: f64,
    duration_s: f64,
    elapsed_s: f64,
    finished: bool,
}

impl ZoomAnim {
    /// Starts an animation from the camera's current scale to `to_scale` over `duration_s`.
    pub fn start(
        camera: &Camera2D,
        anchor: (f64, f64),
        to_scale: f64,
        duration_s: f64,
    ) -> Result<Self, MapError> {
        check_point(anchor, "anchor")?;
        let to = check_scale(to_scale, "scale")?;
        let duration_s = check_scale(duration_s, "duration")?;
        Ok(Self {
            anchor,
            from: camera.scale(),
            to,
            duration_s,
            elapsed_s: 0.0,
            finished: false,
        })
    }

    /// True once the target scale has been applied.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Scale at normalised time `u` in `[0, 1]`, interpolated geometrically with ease-out.
    pub fn scale_at(&self, u: f64) -> f64 {
        let e = ease_out_cubic(u);
        self.from * (self.to / self.from).powf(e)
    }

    /// Advances by `dt_s` seconds. Returns `Ok(true)` while still moving and `Ok(false)` on the
    /// step that lands on the target and on every call after it.
    pub fn step(&mut self, camera: &mut Camera2D, dt_s: f64) -> Result<bool, MapError> {
        if self.finished {
            return Ok(false);
        }
        check_non_negative(dt_s, "dt")?;
        self.elapsed_s += dt_s;
        let u = self.elapsed_s / self.duration_s;
        if u >= 1.0 {
            camera.zoom_about(self.anchor, self.to)?;
            self.finished = true;
            return Ok(false);
        }
        camera.zoom_about(self.anchor, self.scale_at(u))?;
        Ok(true)
    }
}

fn ease_out_cubic(u: f64) -> f64 {
    let v = 1.0 - u.clamp(0.0, 1.0);
    1.0 - v * v * v
}

/// Path between two camera states. `at(0)` is the start camera and `at(1)` is the target.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlyTo {
    start: Camera2D,
    target: Camera2D,
}

impl FlyTo {
    /// Pairs a start camera with a target. Both must share a viewport size.
    pub fn new(start: Camera2D, target: Camera2D) -> Result<Self, MapError> {
        if start.viewport() != target.viewport() {
            return Err(MapError::ViewportMismatch);
        }
        Ok(Self { start, target })
    }

    /// The camera the path starts from.
    pub fn start(&self) -> Camera2D {
        self.start
    }

    /// The camera the path ends at.
    pub fn target(&self) -> Camera2D {
        self.target
    }

    /// Camera at parameter `t`, clamped to `[0, 1]`. Endpoints are returned exactly. The centre
    /// moves linearly and the scale geometrically.
    pub fn at(&self, t: f64) -> Result<Camera2D, MapError> {
        check_finite(t, "t")?;
        let t = t.clamp(0.0, 1.0);
        if t == 0.0 {
            return Ok(self.start);
        }
        if t == 1.0 {
            return Ok(self.target);
        }
        let (c0, c1) = (self.start.centre(), self.target.centre());
        let centre = (c0.0 + t * (c1.0 - c0.0), c0.1 + t * (c1.1 - c0.1));
        let s0 = self.start.scale();
        let scale = s0 * (self.target.scale() / s0).powf(t);
        Camera2D::new(centre, scale, self.start.viewport())
    }
}

#[cfg(test)]
#[path = "camera_tests.rs"]
mod tests;
