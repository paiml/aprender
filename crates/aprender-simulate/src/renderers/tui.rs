//! Generic TUI Renderer for `DemoEngine` Implementations
//!
//! Per specification SIMULAR-DEMO-002: This renderer is engine-agnostic.
//! Any `DemoEngine` implementation can be rendered through this interface.
//!
//! # Usage
//!
//! ```ignore
//! use simular::demos::{DemoEngine, OrbitalEngine};
//! use simular::renderers::DemoRenderer;
//!
//! let yaml = std::fs::read_to_string("config.yaml")?;
//! let engine = OrbitalEngine::from_yaml(&yaml)?;
//! let renderer = DemoRenderer::new(engine);
//! renderer.run()?;
//! ```

use crate::demos::{CriterionResult, DemoEngine, DemoMeta};
use serde::Serialize;

/// Trait for renderable demo data.
///
/// Engines that want TUI rendering implement this to provide
/// display-friendly data without coupling to ratatui.
pub trait RenderableDemo {
    /// Get demo title for display.
    fn title(&self) -> String;

    /// Get current status line.
    fn status_line(&self) -> String;

    /// Get key metrics as (label, value) pairs.
    fn metrics(&self) -> Vec<(String, String)>;

    /// Get current step count.
    fn current_step(&self) -> u64;

    /// Check if demo is paused/complete.
    fn is_running(&self) -> bool;
}

/// Generic demo renderer that works with any `DemoEngine`.
///
/// This is the unified renderer that replaces separate `OrbitApp`, `TspApp`, etc.
#[derive(Debug)]
pub struct DemoRenderer<E: DemoEngine> {
    engine: E,
    running: bool,
    paused: bool,
    step_count: u64,
}

impl<E: DemoEngine> DemoRenderer<E> {
    /// Create a new renderer for the given engine.
    #[must_use]
    pub fn new(engine: E) -> Self {
        Self {
            engine,
            running: true,
            paused: false,
            step_count: 0,
        }
    }

    /// Get reference to the engine.
    #[must_use]
    pub fn engine(&self) -> &E {
        &self.engine
    }

    /// Get mutable reference to the engine.
    pub fn engine_mut(&mut self) -> &mut E {
        &mut self.engine
    }

    /// Check if renderer is running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.running && !self.engine.is_complete()
    }

    /// Check if renderer is paused.
    #[must_use]
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Toggle pause state.
    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    /// Stop the renderer.
    pub fn stop(&mut self) {
        self.running = false;
    }

    /// Reset the engine.
    pub fn reset(&mut self) {
        self.engine.reset();
        self.step_count = 0;
        self.paused = false;
    }

    /// Advance the simulation by one step.
    pub fn step(&mut self) -> E::StepResult {
        let result = self.engine.step();
        self.step_count += 1;
        result
    }

    /// Get demo metadata.
    #[must_use]
    pub fn meta(&self) -> &DemoMeta {
        self.engine.meta()
    }

    /// Get current state.
    #[must_use]
    pub fn state(&self) -> E::State {
        self.engine.state()
    }

    /// Evaluate falsification criteria.
    #[must_use]
    pub fn evaluate_criteria(&self) -> Vec<CriterionResult> {
        self.engine.evaluate_criteria()
    }

    /// Get step count.
    #[must_use]
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Get seed for reproducibility.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.engine.seed()
    }
}

/// Render data for TUI display.
///
/// This struct contains all data needed to render a frame,
/// decoupled from the actual rendering implementation.
#[derive(Debug, Clone, Serialize)]
pub struct RenderFrame {
    /// Demo title.
    pub title: String,
    /// Demo type (e.g., "orbit", "tsp").
    pub demo_type: String,
    /// Current step number.
    pub step: u64,
    /// Seed for reproducibility.
    pub seed: u64,
    /// Whether demo is paused.
    pub paused: bool,
    /// Whether demo is complete.
    pub complete: bool,
    /// Key-value metrics for display.
    pub metrics: Vec<(String, String)>,
    /// Falsification criteria results.
    pub criteria: Vec<CriterionResult>,
}

impl<E: DemoEngine> DemoRenderer<E>
where
    E::State: Serialize,
{
    /// Generate a render frame from current state.
    #[must_use]
    pub fn render_frame(&self) -> RenderFrame {
        let meta = self.engine.meta();

        RenderFrame {
            title: format!("{} ({})", meta.id, meta.version),
            demo_type: meta.demo_type.clone(),
            step: self.step_count,
            seed: self.engine.seed(),
            paused: self.paused,
            complete: self.engine.is_complete(),
            metrics: Vec::new(), // Engines can provide custom metrics
            criteria: self.engine.evaluate_criteria(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demos::OrbitalEngine;

    const TEST_YAML: &str = r#"
simulation:
  type: orbit
  name: "Test Orbit"

meta:
  id: "TEST-001"
  version: "1.0.0"
  demo_type: orbit

reproducibility:
  seed: 42

scenario:
  type: kepler
  central_body:
    name: "Sun"
    mass_kg: 1.989e30
    position: [0.0, 0.0, 0.0]
  orbiter:
    name: "Earth"
    mass_kg: 5.972e24
    semi_major_axis_m: 1.496e11
    eccentricity: 0.0167

integrator:
  type: stormer_verlet
  dt_seconds: 3600.0
"#;

    #[test]
    fn test_renderer_creation() {
        let engine = OrbitalEngine::from_yaml(TEST_YAML).unwrap();
        let renderer = DemoRenderer::new(engine);

        assert!(renderer.is_running());
        assert!(!renderer.is_paused());
        assert_eq!(renderer.step_count(), 0);
    }

    #[test]
    fn test_renderer_step() {
        let engine = OrbitalEngine::from_yaml(TEST_YAML).unwrap();
        let mut renderer = DemoRenderer::new(engine);

        renderer.step();
        assert_eq!(renderer.step_count(), 1);

        renderer.step();
        assert_eq!(renderer.step_count(), 2);
    }

    #[test]
    fn test_renderer_pause() {
        let engine = OrbitalEngine::from_yaml(TEST_YAML).unwrap();
        let mut renderer = DemoRenderer::new(engine);

        assert!(!renderer.is_paused());
        renderer.toggle_pause();
        assert!(renderer.is_paused());
        renderer.toggle_pause();
        assert!(!renderer.is_paused());
    }

    #[test]
    fn test_renderer_reset() {
        let engine = OrbitalEngine::from_yaml(TEST_YAML).unwrap();
        let mut renderer = DemoRenderer::new(engine);

        renderer.step();
        renderer.step();
        assert_eq!(renderer.step_count(), 2);

        renderer.reset();
        assert_eq!(renderer.step_count(), 0);
    }

    #[test]
    fn test_renderer_meta() {
        let engine = OrbitalEngine::from_yaml(TEST_YAML).unwrap();
        let renderer = DemoRenderer::new(engine);

        let meta = renderer.meta();
        assert_eq!(meta.id, "TEST-001");
        assert_eq!(meta.demo_type, "orbit");
    }

    #[test]
    fn test_render_frame() {
        let engine = OrbitalEngine::from_yaml(TEST_YAML).unwrap();
        let mut renderer = DemoRenderer::new(engine);

        renderer.step();
        let frame = renderer.render_frame();

        assert!(frame.title.contains("TEST-001"));
        assert_eq!(frame.demo_type, "orbit");
        assert_eq!(frame.step, 1);
        assert_eq!(frame.seed, 42);
        assert!(!frame.paused);
    }

    #[test]
    fn test_renderer_stop() {
        let engine = OrbitalEngine::from_yaml(TEST_YAML).unwrap();
        let mut renderer = DemoRenderer::new(engine);

        assert!(renderer.is_running());
        renderer.stop();
        assert!(!renderer.is_running());
    }

    #[test]
    fn test_renderer_criteria() {
        let engine = OrbitalEngine::from_yaml(TEST_YAML).unwrap();
        let renderer = DemoRenderer::new(engine);

        let criteria = renderer.evaluate_criteria();
        assert!(!criteria.is_empty());
    }
}
