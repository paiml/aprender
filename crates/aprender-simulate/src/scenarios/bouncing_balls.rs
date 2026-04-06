//! Bouncing balls physics simulation.
//!
//! 2D elastic collisions with gravity, wall boundaries, and velocity-dependent
//! color mapping. Produces `RenderCommand` output for SVG/keyframe export.

use crate::orbit::render::{Color, RenderCommand};

/// A single ball in the simulation.
#[derive(Debug, Clone)]
pub struct Ball {
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub radius: f64,
    pub mass: f64,
    /// Trail of recent positions.
    pub trail: Vec<(f64, f64)>,
}

/// Simulation configuration.
#[derive(Debug, Clone)]
pub struct BouncingBallsConfig {
    /// Canvas width in pixels.
    pub width: f64,
    /// Canvas height in pixels.
    pub height: f64,
    /// Gravitational acceleration (pixels/s^2, positive = downward).
    pub gravity: f64,
    /// Coefficient of restitution (1.0 = perfectly elastic).
    pub restitution: f64,
    /// Maximum trail length per ball.
    pub trail_length: usize,
    /// Number of balls.
    pub ball_count: usize,
}

impl Default for BouncingBallsConfig {
    fn default() -> Self {
        Self {
            width: 1920.0,
            height: 1080.0,
            gravity: 400.0,
            restitution: 0.92,
            trail_length: 30,
            ball_count: 8,
        }
    }
}

/// Complete simulation state.
#[derive(Debug, Clone)]
pub struct BouncingBallsState {
    pub balls: Vec<Ball>,
    pub config: BouncingBallsConfig,
}

impl BouncingBallsState {
    /// Create initial state from a seed.
    #[must_use]
    pub fn new(config: BouncingBallsConfig, seed: u64) -> Self {
        let mut balls = Vec::with_capacity(config.ball_count);
        // Simple deterministic PRNG (xorshift64)
        let mut rng = seed.wrapping_add(1);

        for _ in 0..config.ball_count {
            rng = xorshift64(rng);
            let radius = 20.0 + (rng % 30) as f64;
            rng = xorshift64(rng);
            let x = radius + (rng % (config.width as u64 - 2 * radius as u64)) as f64;
            rng = xorshift64(rng);
            let y = radius + (rng % ((config.height as u64) / 2)) as f64;
            rng = xorshift64(rng);
            let vx = (rng % 400) as f64 - 200.0;
            rng = xorshift64(rng);
            let vy = (rng % 300) as f64 - 150.0;

            balls.push(Ball {
                x,
                y,
                vx,
                vy,
                radius,
                mass: radius * radius, // mass proportional to area
                trail: Vec::with_capacity(config.trail_length),
            });
        }

        Self { balls, config }
    }

    /// Advance physics by `dt` seconds.
    pub fn step(&mut self, dt: f64) {
        let gravity = self.config.gravity;
        let restitution = self.config.restitution;
        let w = self.config.width;
        let h = self.config.height;
        let trail_len = self.config.trail_length;

        // Update velocities and positions
        for ball in &mut self.balls {
            ball.vy += gravity * dt;
            ball.x += ball.vx * dt;
            ball.y += ball.vy * dt;

            // Wall collisions
            if ball.x - ball.radius < 0.0 {
                ball.x = ball.radius;
                ball.vx = -ball.vx * restitution;
            }
            if ball.x + ball.radius > w {
                ball.x = w - ball.radius;
                ball.vx = -ball.vx * restitution;
            }
            if ball.y - ball.radius < 0.0 {
                ball.y = ball.radius;
                ball.vy = -ball.vy * restitution;
            }
            if ball.y + ball.radius > h {
                ball.y = h - ball.radius;
                ball.vy = -ball.vy * restitution;
            }

            // Record trail
            ball.trail.push((ball.x, ball.y));
            if ball.trail.len() > trail_len {
                ball.trail.remove(0);
            }
        }

        // Ball-ball elastic collisions
        let n = self.balls.len();
        for i in 0..n {
            for j in (i + 1)..n {
                resolve_collision(&mut self.balls, i, j, restitution);
            }
        }
    }

    /// Generate render commands for the current state.
    #[must_use]
    pub fn render(&self) -> Vec<RenderCommand> {
        let mut commands = Vec::with_capacity(self.balls.len() * 3 + 2);

        // Identity camera: screen coordinates pass through unchanged
        commands.push(RenderCommand::SetCamera {
            center_x: self.config.width / 2.0,
            center_y: self.config.height / 2.0,
            zoom: 1.0,
        });

        // Dark background
        commands.push(RenderCommand::Clear {
            color: Color::rgb(18, 18, 24),
        });

        // Floor line
        commands.push(RenderCommand::DrawLine {
            x1: 0.0,
            y1: self.config.height - 2.0,
            x2: self.config.width,
            y2: self.config.height - 2.0,
            color: Color::rgb(60, 60, 80),
        });

        // Trails
        for ball in &self.balls {
            if ball.trail.len() >= 2 {
                commands.push(RenderCommand::DrawOrbitPath {
                    points: ball.trail.clone(),
                    color: Color::new(255, 255, 255, 40),
                });
            }
        }

        // Balls with velocity-mapped color
        for ball in &self.balls {
            let speed = (ball.vx * ball.vx + ball.vy * ball.vy).sqrt();
            let color = velocity_to_color(speed);

            commands.push(RenderCommand::DrawCircle {
                x: ball.x,
                y: ball.y,
                radius: ball.radius,
                color,
                filled: true,
            });
        }

        // Speed labels
        for ball in &self.balls {
            let speed = (ball.vx * ball.vx + ball.vy * ball.vy).sqrt();
            commands.push(RenderCommand::DrawText {
                x: ball.x,
                y: ball.y - ball.radius - 8.0,
                text: format!("{speed:.0}"),
                color: Color::rgb(180, 180, 200),
            });
        }

        commands
    }
}

/// Xorshift64 PRNG — deterministic, no dependencies.
fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

/// Map speed (0-600 px/s) to a blue→cyan→green→yellow→red gradient.
fn velocity_to_color(speed: f64) -> Color {
    let t = (speed / 600.0).clamp(0.0, 1.0);
    if t < 0.25 {
        // Blue → Cyan
        let s = t / 0.25;
        Color::rgb(30, (100.0 + 155.0 * s) as u8, 255)
    } else if t < 0.5 {
        // Cyan → Green
        let s = (t - 0.25) / 0.25;
        Color::rgb(30, 255, (255.0 - 155.0 * s) as u8)
    } else if t < 0.75 {
        // Green → Yellow
        let s = (t - 0.5) / 0.25;
        Color::rgb((30.0 + 225.0 * s) as u8, 255, (100.0 - 70.0 * s) as u8)
    } else {
        // Yellow → Red
        let s = (t - 0.75) / 0.25;
        Color::rgb(255, (255.0 - 155.0 * s) as u8, 30)
    }
}

/// Resolve elastic collision between two balls.
fn resolve_collision(balls: &mut [Ball], i: usize, j: usize, restitution: f64) {
    let dx = balls[j].x - balls[i].x;
    let dy = balls[j].y - balls[i].y;
    let dist_sq = dx * dx + dy * dy;
    let min_dist = balls[i].radius + balls[j].radius;

    if dist_sq >= min_dist * min_dist || dist_sq < 1e-10 {
        return;
    }

    let dist = dist_sq.sqrt();
    let nx = dx / dist;
    let ny = dy / dist;

    // Relative velocity along collision normal
    let dvx = balls[i].vx - balls[j].vx;
    let dvy = balls[i].vy - balls[j].vy;
    let dvn = dvx * nx + dvy * ny;

    // Don't resolve if separating
    if dvn < 0.0 {
        return;
    }

    let m1 = balls[i].mass;
    let m2 = balls[j].mass;
    let impulse = (1.0 + restitution) * dvn / (m1 + m2);

    balls[i].vx -= impulse * m2 * nx;
    balls[i].vy -= impulse * m2 * ny;
    balls[j].vx += impulse * m1 * nx;
    balls[j].vy += impulse * m1 * ny;

    // Separate overlapping balls
    let overlap = min_dist - dist;
    let sep = overlap / 2.0 + 0.5;
    balls[i].x -= sep * nx;
    balls[i].y -= sep * ny;
    balls[j].x += sep * nx;
    balls[j].y += sep * ny;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = BouncingBallsConfig::default();
        assert_eq!(cfg.width, 1920.0);
        assert_eq!(cfg.height, 1080.0);
        assert_eq!(cfg.ball_count, 8);
    }

    #[test]
    fn test_new_state_creates_correct_ball_count() {
        let cfg = BouncingBallsConfig {
            ball_count: 5,
            ..Default::default()
        };
        let state = BouncingBallsState::new(cfg, 42);
        assert_eq!(state.balls.len(), 5);
    }

    #[test]
    fn test_balls_within_bounds() {
        let cfg = BouncingBallsConfig::default();
        let state = BouncingBallsState::new(cfg, 42);
        for ball in &state.balls {
            assert!(ball.x >= ball.radius);
            assert!(ball.x <= 1920.0 - ball.radius);
            assert!(ball.y >= ball.radius);
        }
    }

    #[test]
    fn test_deterministic_seed() {
        let cfg = BouncingBallsConfig::default();
        let s1 = BouncingBallsState::new(cfg.clone(), 42);
        let s2 = BouncingBallsState::new(cfg, 42);
        for (a, b) in s1.balls.iter().zip(s2.balls.iter()) {
            assert_eq!(a.x, b.x);
            assert_eq!(a.y, b.y);
            assert_eq!(a.vx, b.vx);
            assert_eq!(a.vy, b.vy);
        }
    }

    #[test]
    fn test_different_seeds_differ() {
        let cfg = BouncingBallsConfig::default();
        let s1 = BouncingBallsState::new(cfg.clone(), 42);
        let s2 = BouncingBallsState::new(cfg, 99);
        let differs = s1
            .balls
            .iter()
            .zip(s2.balls.iter())
            .any(|(a, b)| a.x != b.x);
        assert!(differs);
    }

    #[test]
    fn test_step_applies_gravity() {
        let cfg = BouncingBallsConfig {
            ball_count: 1,
            gravity: 100.0,
            ..Default::default()
        };
        let mut state = BouncingBallsState::new(cfg, 42);
        let vy_before = state.balls[0].vy;
        state.step(0.1);
        // Gravity should increase downward velocity
        assert!(state.balls[0].vy > vy_before);
    }

    #[test]
    fn test_wall_collision_keeps_in_bounds() {
        let cfg = BouncingBallsConfig {
            ball_count: 1,
            ..Default::default()
        };
        let mut state = BouncingBallsState::new(cfg, 42);
        // Run many steps
        for _ in 0..1000 {
            state.step(1.0 / 60.0);
        }
        let ball = &state.balls[0];
        assert!(ball.x >= ball.radius);
        assert!(ball.x <= 1920.0);
        assert!(ball.y >= ball.radius);
        assert!(ball.y <= 1080.0);
    }

    #[test]
    fn test_trail_grows() {
        let cfg = BouncingBallsConfig {
            ball_count: 1,
            trail_length: 10,
            ..Default::default()
        };
        let mut state = BouncingBallsState::new(cfg, 42);
        assert!(state.balls[0].trail.is_empty());
        for _ in 0..5 {
            state.step(1.0 / 60.0);
        }
        assert_eq!(state.balls[0].trail.len(), 5);
    }

    #[test]
    fn test_trail_caps_at_max_length() {
        let cfg = BouncingBallsConfig {
            ball_count: 1,
            trail_length: 5,
            ..Default::default()
        };
        let mut state = BouncingBallsState::new(cfg, 42);
        for _ in 0..20 {
            state.step(1.0 / 60.0);
        }
        assert_eq!(state.balls[0].trail.len(), 5);
    }

    #[test]
    fn test_render_produces_commands() {
        let cfg = BouncingBallsConfig {
            ball_count: 3,
            ..Default::default()
        };
        let state = BouncingBallsState::new(cfg, 42);
        let commands = state.render();
        // Clear + floor line + 3 balls + 3 labels = 8 minimum
        assert!(commands.len() >= 8);
    }

    #[test]
    fn test_render_starts_with_camera_then_clear() {
        let cfg = BouncingBallsConfig {
            ball_count: 1,
            ..Default::default()
        };
        let state = BouncingBallsState::new(cfg, 42);
        let commands = state.render();
        assert!(matches!(commands[0], RenderCommand::SetCamera { .. }));
        assert!(matches!(commands[1], RenderCommand::Clear { .. }));
    }

    #[test]
    fn test_velocity_color_gradient() {
        let slow = velocity_to_color(0.0);
        let fast = velocity_to_color(600.0);
        // Slow = blue-ish, fast = red-ish
        assert!(slow.b > slow.r);
        assert!(fast.r > fast.b);
    }

    #[test]
    fn test_velocity_color_clamps() {
        let over = velocity_to_color(9999.0);
        let under = velocity_to_color(-10.0);
        // Should not panic, should clamp
        assert_eq!(over.r, 255);
        assert!(under.b == 255);
    }

    #[test]
    fn test_elastic_collision_conserves_momentum() {
        let cfg = BouncingBallsConfig {
            ball_count: 2,
            gravity: 0.0,
            restitution: 1.0,
            ..Default::default()
        };
        let mut state = BouncingBallsState::new(cfg, 42);
        // Set up head-on collision
        state.balls[0].x = 500.0;
        state.balls[0].y = 540.0;
        state.balls[0].vx = 100.0;
        state.balls[0].vy = 0.0;
        state.balls[1].x = 560.0;
        state.balls[1].y = 540.0;
        state.balls[1].vx = -100.0;
        state.balls[1].vy = 0.0;

        let m1 = state.balls[0].mass;
        let m2 = state.balls[1].mass;
        let px_before = m1 * state.balls[0].vx + m2 * state.balls[1].vx;
        let py_before = m1 * state.balls[0].vy + m2 * state.balls[1].vy;

        state.step(1.0 / 60.0);

        let px_after = m1 * state.balls[0].vx + m2 * state.balls[1].vx;
        let py_after = m1 * state.balls[0].vy + m2 * state.balls[1].vy;

        assert!((px_before - px_after).abs() < 1.0);
        assert!((py_before - py_after).abs() < 1.0);
    }

    #[test]
    fn test_xorshift_deterministic() {
        assert_eq!(xorshift64(42), xorshift64(42));
        assert_ne!(xorshift64(42), xorshift64(43));
    }

    #[test]
    fn test_render_after_many_steps() {
        let cfg = BouncingBallsConfig::default();
        let mut state = BouncingBallsState::new(cfg, 42);
        for _ in 0..600 {
            state.step(1.0 / 60.0);
        }
        let commands = state.render();
        // Should still produce valid commands after 10s of simulation
        assert!(!commands.is_empty());
        // All balls should still be in bounds
        for ball in &state.balls {
            assert!(ball.x >= 0.0 && ball.x <= 1920.0);
            assert!(ball.y >= 0.0 && ball.y <= 1080.0);
        }
    }
}
