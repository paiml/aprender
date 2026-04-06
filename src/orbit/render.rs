//! Platform-agnostic render commands for orbital visualization.
//!
//! Implements the command pattern for rendering, allowing the same
//! simulation to be displayed on TUI (ratatui) or WASM (Canvas/WebGL).
//!
//! # References
//!
//! [19] Gamma et al., "Design Patterns," 1994.

use crate::orbit::heijunka::HeijunkaStatus;
use crate::orbit::jidoka::JidokaStatus;
use crate::orbit::physics::NBodyState;
use serde::{Deserialize, Serialize};

/// RGBA color representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    /// Create new color.
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Create opaque color.
    #[must_use]
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self::new(r, g, b, 255)
    }

    // Common colors
    pub const WHITE: Self = Self::rgb(255, 255, 255);
    pub const BLACK: Self = Self::rgb(0, 0, 0);
    pub const RED: Self = Self::rgb(255, 0, 0);
    pub const GREEN: Self = Self::rgb(0, 255, 0);
    pub const BLUE: Self = Self::rgb(0, 0, 255);
    pub const YELLOW: Self = Self::rgb(255, 255, 0);
    pub const CYAN: Self = Self::rgb(0, 255, 255);
    pub const ORANGE: Self = Self::rgb(255, 165, 0);

    // Celestial body colors
    pub const SUN: Self = Self::rgb(255, 204, 0);
    pub const MERCURY: Self = Self::rgb(169, 169, 169);
    pub const VENUS: Self = Self::rgb(255, 198, 73);
    pub const EARTH: Self = Self::rgb(100, 149, 237);
    pub const MARS: Self = Self::rgb(193, 68, 14);
}

/// Platform-agnostic render command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RenderCommand {
    /// Clear the screen.
    Clear { color: Color },

    /// Draw a circle (body).
    DrawCircle {
        x: f64,
        y: f64,
        radius: f64,
        color: Color,
        filled: bool,
    },

    /// Draw a line.
    DrawLine {
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        color: Color,
    },

    /// Draw orbit path (series of points).
    DrawOrbitPath {
        points: Vec<(f64, f64)>,
        color: Color,
    },

    /// Draw text label.
    DrawText {
        x: f64,
        y: f64,
        text: String,
        color: Color,
    },

    /// Draw velocity vector.
    DrawVelocity {
        x: f64,
        y: f64,
        vx: f64,
        vy: f64,
        scale: f64,
        color: Color,
    },

    /// Highlight a body (Jidoka warning).
    HighlightBody {
        x: f64,
        y: f64,
        radius: f64,
        color: Color,
    },

    /// Set camera view.
    SetCamera {
        center_x: f64,
        center_y: f64,
        zoom: f64,
    },
}

/// Camera/view configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Camera {
    pub center_x: f64,
    pub center_y: f64,
    pub zoom: f64,
    pub width: f64,
    pub height: f64,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            center_x: 0.0,
            center_y: 0.0,
            zoom: 1.0,
            width: 800.0,
            height: 600.0,
        }
    }
}

impl Camera {
    /// Convert world coordinates to screen coordinates.
    #[must_use]
    pub fn world_to_screen(&self, x: f64, y: f64) -> (f64, f64) {
        let sx = (x - self.center_x) * self.zoom + self.width / 2.0;
        let sy = (y - self.center_y) * self.zoom + self.height / 2.0;
        (sx, sy)
    }

    /// Convert screen coordinates to world coordinates.
    #[must_use]
    pub fn screen_to_world(&self, sx: f64, sy: f64) -> (f64, f64) {
        let x = (sx - self.width / 2.0) / self.zoom + self.center_x;
        let y = (sy - self.height / 2.0) / self.zoom + self.center_y;
        (x, y)
    }

    /// Adjust zoom to fit given bounds.
    pub fn fit_bounds(&mut self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) {
        self.center_x = (min_x + max_x) / 2.0;
        self.center_y = (min_y + max_y) / 2.0;

        let width_span = max_x - min_x;
        let height_span = max_y - min_y;

        let zoom_x = if width_span > 0.0 {
            self.width / width_span * 0.9
        } else {
            1.0
        };
        let zoom_y = if height_span > 0.0 {
            self.height / height_span * 0.9
        } else {
            1.0
        };

        self.zoom = zoom_x.min(zoom_y);
    }
}

/// Body appearance configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyAppearance {
    pub name: String,
    pub color: Color,
    pub radius: f64,
    pub show_velocity: bool,
    pub show_orbit_trail: bool,
}

impl Default for BodyAppearance {
    fn default() -> Self {
        Self {
            name: "Body".to_string(),
            color: Color::WHITE,
            radius: 5.0,
            show_velocity: false,
            show_orbit_trail: true,
        }
    }
}

/// Renderer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderConfig {
    pub camera: Camera,
    pub bodies: Vec<BodyAppearance>,
    pub show_jidoka_status: bool,
    pub show_heijunka_status: bool,
    pub velocity_scale: f64,
    pub orbit_trail_length: usize,
    pub scale_factor: f64, // meters per pixel
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            camera: Camera::default(),
            bodies: vec![
                BodyAppearance {
                    name: "Sun".to_string(),
                    color: Color::SUN,
                    radius: 10.0,
                    show_velocity: false,
                    show_orbit_trail: false,
                },
                BodyAppearance {
                    name: "Earth".to_string(),
                    color: Color::EARTH,
                    radius: 5.0,
                    show_velocity: true,
                    show_orbit_trail: true,
                },
            ],
            show_jidoka_status: true,
            show_heijunka_status: true,
            velocity_scale: 1e-4,
            orbit_trail_length: 1000,
            scale_factor: 1e9, // 1 pixel = 1e9 meters
        }
    }
}

/// Orbit trail for visualizing past positions.
#[derive(Debug, Clone, Default)]
pub struct OrbitTrail {
    points: Vec<(f64, f64)>,
    max_length: usize,
}

impl OrbitTrail {
    /// Create new orbit trail.
    #[must_use]
    pub fn new(max_length: usize) -> Self {
        Self {
            points: Vec::with_capacity(max_length),
            max_length,
        }
    }

    /// Add a point to the trail.
    pub fn push(&mut self, x: f64, y: f64) {
        // Don't add points if max_length is 0
        if self.max_length == 0 {
            return;
        }
        if self.points.len() >= self.max_length {
            self.points.remove(0);
        }
        self.points.push((x, y));
    }

    /// Get trail points.
    #[must_use]
    pub fn points(&self) -> &[(f64, f64)] {
        &self.points
    }

    /// Clear the trail.
    pub fn clear(&mut self) {
        self.points.clear();
    }
}

/// Generate render commands from simulation state.
#[must_use]
pub fn render_state(
    state: &NBodyState,
    config: &RenderConfig,
    trails: &[OrbitTrail],
    jidoka: Option<&JidokaStatus>,
    heijunka: Option<&HeijunkaStatus>,
) -> Vec<RenderCommand> {
    let mut commands = Vec::new();

    commands.push(RenderCommand::Clear {
        color: Color::BLACK,
    });
    commands.push(RenderCommand::SetCamera {
        center_x: config.camera.center_x,
        center_y: config.camera.center_y,
        zoom: config.camera.zoom,
    });

    render_orbit_trails(&mut commands, trails, config);
    render_bodies(&mut commands, state, config);
    render_jidoka_status(&mut commands, jidoka, config.show_jidoka_status);
    render_heijunka_status(&mut commands, heijunka, config.show_heijunka_status);

    commands
}

fn render_orbit_trails(
    commands: &mut Vec<RenderCommand>,
    trails: &[OrbitTrail],
    config: &RenderConfig,
) {
    for (i, trail) in trails.iter().enumerate() {
        let Some(body_config) = config.bodies.get(i) else {
            continue;
        };
        if !body_config.show_orbit_trail {
            continue;
        }

        let scaled_points: Vec<(f64, f64)> = trail
            .points()
            .iter()
            .map(|(x, y)| (x / config.scale_factor, y / config.scale_factor))
            .collect();

        if !scaled_points.is_empty() {
            commands.push(RenderCommand::DrawOrbitPath {
                points: scaled_points,
                color: body_config.color,
            });
        }
    }
}

fn render_bodies(commands: &mut Vec<RenderCommand>, state: &NBodyState, config: &RenderConfig) {
    for (i, body) in state.bodies.iter().enumerate() {
        let (x, y, _) = body.position.as_meters();
        let sx = x / config.scale_factor;
        let sy = y / config.scale_factor;
        let appearance = config.bodies.get(i).cloned().unwrap_or_default();

        commands.push(RenderCommand::DrawCircle {
            x: sx,
            y: sy,
            radius: appearance.radius,
            color: appearance.color,
            filled: true,
        });

        if appearance.show_velocity {
            let (vx, vy, _) = body.velocity.as_mps();
            commands.push(RenderCommand::DrawVelocity {
                x: sx,
                y: sy,
                vx: vx * config.velocity_scale,
                vy: vy * config.velocity_scale,
                scale: 1.0,
                color: Color::GREEN,
            });
        }

        commands.push(RenderCommand::DrawText {
            x: sx + appearance.radius + 2.0,
            y: sy,
            text: appearance.name.clone(),
            color: Color::WHITE,
        });
    }
}

fn render_jidoka_status(
    commands: &mut Vec<RenderCommand>,
    jidoka: Option<&JidokaStatus>,
    show: bool,
) {
    let Some(status) = jidoka.filter(|_| show) else {
        return;
    };

    let status_color = jidoka_status_color(status);
    let suffix = if status.close_encounter_warning {
        "⚠ Close"
    } else {
        "OK"
    };
    commands.push(RenderCommand::DrawText {
        x: 10.0,
        y: 10.0,
        text: format!(
            "Jidoka: E={:.2e} L={:.2e} {suffix}",
            status.energy_error, status.angular_momentum_error
        ),
        color: status_color,
    });
}

fn jidoka_status_color(status: &JidokaStatus) -> Color {
    if status.energy_ok && status.angular_momentum_ok && status.finite_ok {
        Color::GREEN
    } else if status.warning_count > 0 {
        Color::ORANGE
    } else {
        Color::RED
    }
}

fn render_heijunka_status(
    commands: &mut Vec<RenderCommand>,
    heijunka: Option<&HeijunkaStatus>,
    show: bool,
) {
    let Some(status) = heijunka.filter(|_| show) else {
        return;
    };

    let budget_color = if status.utilization <= 1.0 {
        Color::GREEN
    } else {
        Color::RED
    };
    commands.push(RenderCommand::DrawText {
        x: 10.0,
        y: 25.0,
        text: format!(
            "Heijunka: {:.1}ms/{:.1}ms ({:.0}%) Q={:?}",
            status.used_ms,
            status.budget_ms,
            status.utilization * 100.0,
            status.quality,
        ),
        color: budget_color,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbit::physics::OrbitBody;
    use crate::orbit::units::{OrbitMass, Position3D, Velocity3D, AU, EARTH_MASS, G, SOLAR_MASS};

    fn create_test_state() -> NBodyState {
        let v_circular = (G * SOLAR_MASS / AU).sqrt();
        let bodies = vec![
            OrbitBody::new(
                OrbitMass::from_kg(SOLAR_MASS),
                Position3D::zero(),
                Velocity3D::zero(),
            ),
            OrbitBody::new(
                OrbitMass::from_kg(EARTH_MASS),
                Position3D::from_au(1.0, 0.0, 0.0),
                Velocity3D::from_mps(0.0, v_circular, 0.0),
            ),
        ];
        NBodyState::new(bodies, 0.0)
    }

    #[test]
    fn test_color_rgb() {
        let c = Color::rgb(255, 128, 0);
        assert_eq!(c.r, 255);
        assert_eq!(c.g, 128);
        assert_eq!(c.b, 0);
        assert_eq!(c.a, 255);
    }

    #[test]
    fn test_color_constants() {
        assert_eq!(Color::WHITE.r, 255);
        assert_eq!(Color::BLACK.r, 0);
        assert_eq!(Color::SUN.r, 255);
    }

    #[test]
    fn test_camera_default() {
        let cam = Camera::default();
        assert!((cam.center_x - 0.0).abs() < 1e-10);
        assert!((cam.zoom - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_camera_world_to_screen() {
        let mut cam = Camera::default();
        cam.width = 800.0;
        cam.height = 600.0;

        let (sx, sy) = cam.world_to_screen(0.0, 0.0);
        assert!((sx - 400.0).abs() < 1e-10);
        assert!((sy - 300.0).abs() < 1e-10);
    }

    #[test]
    fn test_camera_screen_to_world() {
        let mut cam = Camera::default();
        cam.width = 800.0;
        cam.height = 600.0;

        let (x, y) = cam.screen_to_world(400.0, 300.0);
        assert!((x - 0.0).abs() < 1e-10);
        assert!((y - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_camera_fit_bounds() {
        let mut cam = Camera::default();
        cam.width = 800.0;
        cam.height = 600.0;

        cam.fit_bounds(-100.0, 100.0, -100.0, 100.0);
        assert!((cam.center_x - 0.0).abs() < 1e-10);
        assert!((cam.center_y - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_orbit_trail_new() {
        let trail = OrbitTrail::new(100);
        assert_eq!(trail.points().len(), 0);
    }

    #[test]
    fn test_orbit_trail_push() {
        let mut trail = OrbitTrail::new(3);
        trail.push(1.0, 1.0);
        trail.push(2.0, 2.0);
        trail.push(3.0, 3.0);
        assert_eq!(trail.points().len(), 3);

        trail.push(4.0, 4.0);
        assert_eq!(trail.points().len(), 3);
        assert!((trail.points()[0].0 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_orbit_trail_clear() {
        let mut trail = OrbitTrail::new(100);
        trail.push(1.0, 1.0);
        trail.clear();
        assert_eq!(trail.points().len(), 0);
    }

    #[test]
    fn test_render_config_default() {
        let config = RenderConfig::default();
        assert!(config.show_jidoka_status);
        assert!(config.show_heijunka_status);
        assert_eq!(config.bodies.len(), 2);
    }

    #[test]
    fn test_render_state_generates_commands() {
        let state = create_test_state();
        let config = RenderConfig::default();
        let trails = vec![OrbitTrail::new(100), OrbitTrail::new(100)];

        let commands = render_state(&state, &config, &trails, None, None);

        assert!(!commands.is_empty());
        // Should have Clear, SetCamera, and at least 2 DrawCircle (for bodies)
        assert!(commands.len() >= 4);
    }

    #[test]
    fn test_render_state_with_jidoka_status() {
        let state = create_test_state();
        let config = RenderConfig::default();
        let trails = vec![OrbitTrail::new(100), OrbitTrail::new(100)];

        let jidoka = JidokaStatus {
            energy_ok: true,
            angular_momentum_ok: true,
            finite_ok: true,
            energy_error: 1e-9,
            angular_momentum_error: 1e-12,
            min_separation: AU,
            close_encounter_warning: false,
            warning_count: 0,
        };

        let commands = render_state(&state, &config, &trails, Some(&jidoka), None);

        // Should include Jidoka status text
        let has_jidoka_text = commands.iter().any(
            |cmd| matches!(cmd, RenderCommand::DrawText { text, .. } if text.contains("Jidoka")),
        );
        assert!(has_jidoka_text);
    }

    #[test]
    fn test_body_appearance_default() {
        let appearance = BodyAppearance::default();
        assert_eq!(appearance.name, "Body");
        assert!(appearance.show_orbit_trail);
    }

    #[test]
    fn test_orbit_trail_zero_max_length() {
        // This should not panic - edge case fix
        let mut trail = OrbitTrail::new(0);
        trail.push(1.0, 1.0);
        trail.push(2.0, 2.0);
        // With max_length=0, no points should be added
        assert_eq!(trail.points().len(), 0);
    }
}
