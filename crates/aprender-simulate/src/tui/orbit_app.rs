//! Orbit TUI application state and logic.
//!
//! This module contains the testable state and logic for the orbit TUI demo.
//! Terminal I/O is handled by the binary, but all state management lives here.
//!
//! # `ComputeBlocks` Integration (SIMULAR-CB-001)
//!
//! Uses SIMD-optimized `SparklineBlock` and `LoadTrendBlock` from presentar-terminal
//! for energy/momentum conservation visualization and frame budget trends.

use crate::demos::{DemoEngine, OrbitalEngine};
use crate::orbit::physics::YoshidaIntegrator;
use crate::orbit::prelude::*;
use crate::orbit::render::OrbitTrail;
use crate::tui::compute_blocks::SimulationMetrics;
use crossterm::event::KeyCode;

/// Embedded default Earth-Sun YAML configuration.
const DEFAULT_ORBIT_YAML: &str = include_str!("../../examples/experiments/orbit_earth_sun.yaml");

/// Application state for the orbit TUI demo.
pub struct OrbitApp {
    /// N-body simulation state.
    pub state: NBodyState,
    /// Jidoka guard for physics validation.
    pub jidoka: OrbitJidokaGuard,
    /// Heijunka scheduler for frame budgeting.
    pub heijunka: HeijunkaScheduler,
    /// Yoshida integrator (stored for reference).
    _integrator: YoshidaIntegrator,
    /// Orbit trails for visualization.
    pub trails: Vec<OrbitTrail>,
    /// Configuration for the orbit.
    pub config: KeplerConfig,
    /// Whether the simulation is paused.
    pub paused: bool,
    /// Time scale multiplier.
    pub time_scale: f64,
    /// Simulated time in days.
    pub sim_time_days: f64,
    /// Frame counter.
    pub frame_count: u64,
    /// Whether the app should quit.
    pub should_quit: bool,
    /// SIMD-optimized simulation metrics (`ComputeBlocks` from presentar-terminal)
    pub metrics: SimulationMetrics,
}

impl OrbitApp {
    /// Create a new orbit application with default Earth-Sun configuration.
    ///
    /// Uses the embedded `orbit_earth_sun.yaml` configuration.
    #[must_use]
    pub fn new() -> Self {
        // Load from embedded YAML for YAML-first architecture
        Self::from_yaml(DEFAULT_ORBIT_YAML)
            .unwrap_or_else(|_| Self::from_config(KeplerConfig::default()))
    }

    /// Create from YAML configuration string.
    ///
    /// # Errors
    ///
    /// Returns error if YAML parsing fails.
    pub fn from_yaml(yaml: &str) -> Result<Self, crate::demos::DemoError> {
        let engine = OrbitalEngine::from_yaml(yaml)?;
        let config = engine.kepler_config();
        Ok(Self::from_config(config))
    }

    /// Create from a specific `KeplerConfig`.
    #[must_use]
    pub fn from_config(config: KeplerConfig) -> Self {
        let state = config.build(1e6);

        let mut jidoka = OrbitJidokaGuard::new(OrbitJidokaConfig::default());
        jidoka.initialize(&state);

        let heijunka_config = HeijunkaConfig {
            frame_budget_ms: 33.0, // 30 FPS target
            physics_budget_fraction: 0.5,
            base_dt: 3600.0, // 1 hour per physics step
            max_substeps: 24,
            ..HeijunkaConfig::default()
        };
        let heijunka = HeijunkaScheduler::new(heijunka_config);

        let trails = vec![
            OrbitTrail::new(0),   // Sun doesn't need trail
            OrbitTrail::new(500), // Earth trail
        ];

        Self {
            state,
            jidoka,
            heijunka,
            _integrator: YoshidaIntegrator::new(),
            trails,
            config,
            paused: false,
            time_scale: 1.0,
            sim_time_days: 0.0,
            frame_count: 0,
            should_quit: false,
            metrics: SimulationMetrics::new(),
        }
    }

    /// Reset the simulation to initial state.
    pub fn reset(&mut self) {
        self.state = self.config.build(1e6);
        self.jidoka = OrbitJidokaGuard::new(OrbitJidokaConfig::default());
        self.jidoka.initialize(&self.state);
        self.sim_time_days = 0.0;
        self.frame_count = 0;
        self.metrics.reset();

        for trail in &mut self.trails {
            trail.clear();
        }
    }

    /// Update the simulation for one frame.
    pub fn update(&mut self) {
        if self.paused {
            return;
        }

        // Execute physics with Heijunka time budget
        if let Ok(result) = self.heijunka.execute_frame(&mut self.state) {
            self.sim_time_days += result.sim_time_advanced / 86400.0;

            // Update trails
            for (i, body) in self.state.bodies.iter().enumerate() {
                if i < self.trails.len() {
                    let (x, y, _) = body.position.as_meters();
                    self.trails[i].push(x, y);
                }
            }
        }

        // Check Jidoka guards
        let response = self.jidoka.check(&self.state);
        if response.should_pause() || response.should_halt() {
            self.paused = true;
        }

        // Update ComputeBlock metrics (SIMD-optimized sparklines)
        let energy = self.state.total_energy();
        let momentum = self.state.angular_momentum_magnitude();
        let heijunka_status = self.heijunka.status();
        self.metrics
            .update(energy, momentum, heijunka_status.utilization);

        self.frame_count += 1;
    }

    /// Handle a key press.
    pub fn handle_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            KeyCode::Char(' ') => self.paused = !self.paused,
            KeyCode::Char('r') => self.reset(),
            KeyCode::Char('+' | '=') => {
                self.time_scale = (self.time_scale * 2.0).min(1000.0);
            }
            KeyCode::Char('-') => {
                self.time_scale = (self.time_scale / 2.0).max(0.1);
            }
            _ => {}
        }
    }

    /// Get the Jidoka status for display.
    #[must_use]
    pub fn jidoka_status(&self) -> &JidokaStatus {
        self.jidoka.status()
    }

    /// Get the Heijunka status for display.
    #[must_use]
    pub fn heijunka_status(&self) -> HeijunkaStatus {
        self.heijunka.status().clone()
    }

    /// Get the total energy of the system.
    #[must_use]
    pub fn total_energy(&self) -> f64 {
        self.state.total_energy()
    }
}

impl Default for OrbitApp {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_app() {
        let app = OrbitApp::new();
        assert!(!app.paused);
        assert!(!app.should_quit);
        assert_eq!(app.frame_count, 0);
        assert!((app.time_scale - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reset() {
        let mut app = OrbitApp::new();
        app.frame_count = 100;
        app.sim_time_days = 50.0;
        app.reset();
        assert_eq!(app.frame_count, 0);
        assert!((app.sim_time_days).abs() < f64::EPSILON);
    }

    #[test]
    fn test_handle_key_quit() {
        let mut app = OrbitApp::new();
        assert!(!app.should_quit);
        app.handle_key(KeyCode::Char('q'));
        assert!(app.should_quit);
    }

    #[test]
    fn test_handle_key_esc() {
        let mut app = OrbitApp::new();
        assert!(!app.should_quit);
        app.handle_key(KeyCode::Esc);
        assert!(app.should_quit);
    }

    #[test]
    fn test_handle_key_pause() {
        let mut app = OrbitApp::new();
        assert!(!app.paused);
        app.handle_key(KeyCode::Char(' '));
        assert!(app.paused);
        app.handle_key(KeyCode::Char(' '));
        assert!(!app.paused);
    }

    #[test]
    fn test_handle_key_reset() {
        let mut app = OrbitApp::new();
        app.frame_count = 50;
        app.handle_key(KeyCode::Char('r'));
        assert_eq!(app.frame_count, 0);
    }

    #[test]
    fn test_handle_key_time_scale_increase() {
        let mut app = OrbitApp::new();
        let initial = app.time_scale;
        app.handle_key(KeyCode::Char('+'));
        assert!((app.time_scale - initial * 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_handle_key_time_scale_increase_equals() {
        let mut app = OrbitApp::new();
        let initial = app.time_scale;
        app.handle_key(KeyCode::Char('='));
        assert!((app.time_scale - initial * 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_handle_key_time_scale_decrease() {
        let mut app = OrbitApp::new();
        app.time_scale = 4.0;
        app.handle_key(KeyCode::Char('-'));
        assert!((app.time_scale - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_time_scale_max_limit() {
        let mut app = OrbitApp::new();
        app.time_scale = 900.0;
        app.handle_key(KeyCode::Char('+'));
        assert!((app.time_scale - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_time_scale_min_limit() {
        let mut app = OrbitApp::new();
        app.time_scale = 0.15;
        app.handle_key(KeyCode::Char('-'));
        assert!((app.time_scale - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_update_when_paused() {
        let mut app = OrbitApp::new();
        app.paused = true;
        let initial_frame = app.frame_count;
        app.update();
        assert_eq!(app.frame_count, initial_frame);
    }

    #[test]
    fn test_update_increments_frame() {
        let mut app = OrbitApp::new();
        assert_eq!(app.frame_count, 0);
        app.update();
        assert_eq!(app.frame_count, 1);
    }

    #[test]
    fn test_jidoka_status() {
        let mut app = OrbitApp::new();
        // Run at least one update to initialize jidoka properly
        app.update();
        let status = app.jidoka_status();
        // After one update, checks should pass
        assert!(status.finite_ok);
    }

    #[test]
    fn test_heijunka_status() {
        let mut app = OrbitApp::new();
        // Run at least one update to get valid heijunka stats
        app.update();
        let status = app.heijunka_status();
        // Budget should be set from config
        assert!(status.budget_ms >= 0.0);
    }

    #[test]
    fn test_total_energy() {
        let app = OrbitApp::new();
        let energy = app.total_energy();
        // Energy should be negative for bound orbits
        assert!(energy < 0.0);
    }

    #[test]
    fn test_default() {
        let app = OrbitApp::default();
        assert!(!app.should_quit);
    }

    #[test]
    fn test_unknown_key_ignored() {
        let mut app = OrbitApp::new();
        let paused_before = app.paused;
        let quit_before = app.should_quit;
        app.handle_key(KeyCode::Char('x')); // Unknown key
        assert_eq!(app.paused, paused_before);
        assert_eq!(app.should_quit, quit_before);
    }

    #[test]
    fn test_trails_initialized() {
        let app = OrbitApp::new();
        assert_eq!(app.trails.len(), 2);
    }

    #[test]
    fn test_update_advances_time() {
        let mut app = OrbitApp::new();
        let initial_time = app.sim_time_days;
        app.update();
        assert!(app.sim_time_days > initial_time || app.paused);
    }
}
