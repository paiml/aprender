//! TSP GRASP TUI application state and logic.
//!
//! This module contains the testable state and logic for the TSP GRASP TUI demo.
//! Terminal I/O is handled by the binary, but all state management lives here.
//!
//! # YAML-First Architecture
//!
//! The TUI supports loading TSP instances from YAML files:
//!
//! ```bash
//! # Load Bay Area TSP instance
//! cargo run --bin tsp_tui -- examples/experiments/bay_area_tsp.yaml
//! ```
//!
//! Or use the 'L' key to trigger file loading (handled by binary).

use crate::demos::tsp_grasp::{ConstructionMethod, TspGraspDemo};
use crate::demos::tsp_instance::{TspInstanceError, TspInstanceYaml};
use crate::demos::EddDemo;
use crossterm::event::KeyCode;
use std::path::Path;

/// Application state for the TSP GRASP TUI demo.
pub struct TspApp {
    /// The underlying GRASP demo.
    pub demo: TspGraspDemo,
    /// Whether the simulation is paused.
    pub paused: bool,
    /// Whether auto-run is enabled.
    pub auto_run: bool,
    /// Frame counter.
    pub frame_count: u64,
    /// Whether the app should quit.
    pub should_quit: bool,
    /// Convergence history (scaled tour lengths).
    pub convergence_history: Vec<u64>,
    /// Maximum history length.
    pub max_history: usize,
    /// Loaded YAML instance (if any).
    pub loaded_instance: Option<TspInstanceYaml>,
    /// Path to loaded file (if any).
    pub loaded_path: Option<String>,
}

impl TspApp {
    /// Create a new TSP application.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of cities
    /// * `seed` - Random seed for reproducibility
    #[must_use]
    pub fn new(n: usize, seed: u64) -> Self {
        let mut demo = TspGraspDemo::new(seed, n);
        // Run initial GRASP iteration so we have something to display
        demo.grasp_iteration();

        let mut convergence_history = Vec::new();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let tour_len = (demo.best_tour_length * 1000.0).max(0.0) as u64;
        convergence_history.push(tour_len);

        Self {
            demo,
            paused: false,
            auto_run: true, // Start running automatically
            frame_count: 0,
            should_quit: false,
            convergence_history,
            max_history: 50,
            loaded_instance: None,
            loaded_path: None,
        }
    }

    /// Create application from YAML string.
    ///
    /// # Errors
    ///
    /// Returns error if YAML parsing or validation fails.
    ///
    /// # Example
    ///
    /// ```
    /// use simular::tui::tsp_app::TspApp;
    ///
    /// let yaml = include_str!("../../examples/experiments/bay_area_tsp.yaml");
    /// let app = TspApp::from_yaml(yaml).expect("YAML should parse");
    /// assert_eq!(app.demo.n, 20); // 20-city California instance
    /// ```
    pub fn from_yaml(yaml: &str) -> Result<Self, TspInstanceError> {
        let instance = TspInstanceYaml::from_yaml(yaml)?;
        instance.validate()?;

        let mut demo = TspGraspDemo::from_instance(&instance);
        demo.grasp_iteration();

        let mut convergence_history = Vec::new();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let tour_len = (demo.best_tour_length * 1000.0).max(0.0) as u64;
        convergence_history.push(tour_len);

        Ok(Self {
            demo,
            paused: false,
            auto_run: true,
            frame_count: 0,
            should_quit: false,
            convergence_history,
            max_history: 50,
            loaded_instance: Some(instance),
            loaded_path: None,
        })
    }

    /// Create application from YAML file.
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or YAML is invalid.
    ///
    /// # Example
    ///
    /// ```
    /// use simular::tui::tsp_app::TspApp;
    ///
    /// let app = TspApp::from_yaml_file("examples/experiments/bay_area_tsp.yaml")
    ///     .expect("File should load");
    /// assert_eq!(app.demo.n, 20); // 20-city California instance
    /// assert!(app.loaded_path.is_some());
    /// ```
    pub fn from_yaml_file<P: AsRef<Path>>(path: P) -> Result<Self, TspInstanceError> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let instance = TspInstanceYaml::from_yaml_file(&path)?;
        instance.validate()?;

        let mut demo = TspGraspDemo::from_instance(&instance);
        demo.grasp_iteration();

        let mut convergence_history = Vec::new();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let tour_len = (demo.best_tour_length * 1000.0).max(0.0) as u64;
        convergence_history.push(tour_len);

        Ok(Self {
            demo,
            paused: false,
            auto_run: true,
            frame_count: 0,
            should_quit: false,
            convergence_history,
            max_history: 50,
            loaded_instance: Some(instance),
            loaded_path: Some(path_str),
        })
    }

    /// Get the instance ID if loaded from YAML.
    #[must_use]
    pub fn instance_id(&self) -> Option<&str> {
        self.loaded_instance.as_ref().map(|i| i.meta.id.as_str())
    }

    /// Get the optimal known value if loaded from YAML.
    #[must_use]
    pub fn optimal_known(&self) -> Option<u32> {
        self.loaded_instance
            .as_ref()
            .and_then(|i| i.meta.optimal_known)
    }

    /// Reset the simulation.
    pub fn reset(&mut self) {
        let seed = self.demo.seed;
        let n = self.demo.n;
        self.demo = TspGraspDemo::new(seed, n);
        self.convergence_history.clear();
        self.frame_count = 0;
    }

    /// Execute one step of the simulation.
    pub fn step(&mut self) {
        contract_pre_iterator!();
        contract_post_configuration!(&"ok");
        if !self.paused || self.auto_run {
            self.demo.grasp_iteration();

            // Track convergence history
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let tour_len = (self.demo.best_tour_length * 1000.0).max(0.0) as u64;
            self.convergence_history.push(tour_len);
            if self.convergence_history.len() > self.max_history {
                self.convergence_history.remove(0);
            }
        }
        self.frame_count += 1;
    }

    /// Handle a key press.
    pub fn handle_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            KeyCode::Char(' ') => {
                self.auto_run = !self.auto_run;
                self.paused = !self.auto_run;
            }
            KeyCode::Char('g') => {
                // Single GRASP iteration
                self.paused = false;
                self.step();
                self.paused = true;
                self.auto_run = false;
            }
            KeyCode::Char('r') => self.reset(),
            KeyCode::Char('+' | '=') => {
                let new_size = (self.demo.rcl_size + 1).min(10);
                self.demo.set_rcl_size(new_size);
            }
            KeyCode::Char('-') => {
                let new_size = self.demo.rcl_size.saturating_sub(1).max(1);
                self.demo.set_rcl_size(new_size);
            }
            KeyCode::Char('m') => {
                // Cycle construction method
                let new_method = match self.demo.construction_method {
                    ConstructionMethod::RandomizedGreedy => ConstructionMethod::NearestNeighbor,
                    ConstructionMethod::NearestNeighbor => ConstructionMethod::Random,
                    ConstructionMethod::Random => ConstructionMethod::RandomizedGreedy,
                };
                self.demo.set_construction_method(new_method);
            }
            _ => {}
        }
    }

    /// Check if the app should quit.
    #[must_use]
    pub const fn should_quit(&self) -> bool {
        self.should_quit
    }

    /// Get the optimality gap.
    #[must_use]
    pub fn optimality_gap(&self) -> f64 {
        self.demo.optimality_gap()
    }

    /// Get the construction method name.
    #[must_use]
    pub fn construction_method_name(&self) -> &'static str {
        match self.demo.construction_method {
            ConstructionMethod::RandomizedGreedy => "Randomized Greedy",
            ConstructionMethod::NearestNeighbor => "Nearest Neighbor",
            ConstructionMethod::Random => "Random",
        }
    }

    /// Get the falsification status.
    #[must_use]
    pub fn falsification_status(&self) -> crate::demos::FalsificationStatus {
        self.demo.get_falsification_status()
    }

    /// Verify the governing equation.
    #[must_use]
    pub fn verify_equation(&self) -> bool {
        self.demo.verify_equation()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_app() {
        let app = TspApp::new(10, 42);
        assert!(!app.paused);
        assert!(app.auto_run);
        assert!(!app.should_quit);
        assert_eq!(app.frame_count, 0);
        assert_eq!(app.demo.n, 10);
    }

    #[test]
    fn test_reset() {
        let mut app = TspApp::new(10, 42);
        app.frame_count = 100;
        app.convergence_history.push(500);
        app.reset();
        assert_eq!(app.frame_count, 0);
        assert!(app.convergence_history.is_empty());
    }

    #[test]
    fn test_handle_key_quit() {
        let mut app = TspApp::new(10, 42);
        assert!(!app.should_quit());
        app.handle_key(KeyCode::Char('q'));
        assert!(app.should_quit());
    }

    #[test]
    fn test_handle_key_esc() {
        let mut app = TspApp::new(10, 42);
        app.handle_key(KeyCode::Esc);
        assert!(app.should_quit());
    }

    #[test]
    fn test_handle_key_pause_toggle() {
        let mut app = TspApp::new(10, 42);
        assert!(app.auto_run);
        assert!(!app.paused);

        app.handle_key(KeyCode::Char(' '));
        assert!(!app.auto_run);
        assert!(app.paused);

        app.handle_key(KeyCode::Char(' '));
        assert!(app.auto_run);
        assert!(!app.paused);
    }

    #[test]
    fn test_handle_key_single_step() {
        let mut app = TspApp::new(10, 42);
        let initial_restarts = app.demo.restarts;
        app.handle_key(KeyCode::Char('g'));
        assert!(app.paused);
        assert!(!app.auto_run);
        assert!(app.demo.restarts > initial_restarts);
    }

    #[test]
    fn test_handle_key_reset() {
        let mut app = TspApp::new(10, 42);
        app.frame_count = 50;
        app.handle_key(KeyCode::Char('r'));
        assert_eq!(app.frame_count, 0);
    }

    #[test]
    fn test_handle_key_rcl_increase() {
        let mut app = TspApp::new(10, 42);
        let initial_size = app.demo.rcl_size;
        app.handle_key(KeyCode::Char('+'));
        assert_eq!(app.demo.rcl_size, initial_size + 1);
    }

    #[test]
    fn test_handle_key_rcl_increase_equals() {
        let mut app = TspApp::new(10, 42);
        let initial_size = app.demo.rcl_size;
        app.handle_key(KeyCode::Char('='));
        assert_eq!(app.demo.rcl_size, initial_size + 1);
    }

    #[test]
    fn test_handle_key_rcl_decrease() {
        let mut app = TspApp::new(10, 42);
        app.demo.set_rcl_size(5);
        app.handle_key(KeyCode::Char('-'));
        assert_eq!(app.demo.rcl_size, 4);
    }

    #[test]
    fn test_rcl_size_max_limit() {
        let mut app = TspApp::new(10, 42);
        app.demo.set_rcl_size(10);
        app.handle_key(KeyCode::Char('+'));
        assert_eq!(app.demo.rcl_size, 10);
    }

    #[test]
    fn test_rcl_size_min_limit() {
        let mut app = TspApp::new(10, 42);
        app.demo.set_rcl_size(1);
        app.handle_key(KeyCode::Char('-'));
        assert_eq!(app.demo.rcl_size, 1);
    }

    #[test]
    fn test_handle_key_cycle_method() {
        let mut app = TspApp::new(10, 42);
        assert!(matches!(
            app.demo.construction_method,
            ConstructionMethod::RandomizedGreedy
        ));

        app.handle_key(KeyCode::Char('m'));
        assert!(matches!(
            app.demo.construction_method,
            ConstructionMethod::NearestNeighbor
        ));

        app.handle_key(KeyCode::Char('m'));
        assert!(matches!(
            app.demo.construction_method,
            ConstructionMethod::Random
        ));

        app.handle_key(KeyCode::Char('m'));
        assert!(matches!(
            app.demo.construction_method,
            ConstructionMethod::RandomizedGreedy
        ));
    }

    #[test]
    fn test_step_increments_frame() {
        let mut app = TspApp::new(10, 42);
        assert_eq!(app.frame_count, 0);
        app.step();
        assert_eq!(app.frame_count, 1);
    }

    #[test]
    fn test_step_tracks_convergence() {
        let mut app = TspApp::new(10, 42);
        let initial_len = app.convergence_history.len();
        app.step();
        assert_eq!(app.convergence_history.len(), initial_len + 1);
    }

    #[test]
    fn test_convergence_history_max_limit() {
        let mut app = TspApp::new(10, 42);
        app.max_history = 5;
        for _ in 0..10 {
            app.step();
        }
        assert!(app.convergence_history.len() <= 5);
    }

    #[test]
    fn test_optimality_gap() {
        let app = TspApp::new(10, 42);
        let gap = app.optimality_gap();
        assert!(gap >= 0.0);
    }

    #[test]
    fn test_construction_method_name() {
        let mut app = TspApp::new(10, 42);
        assert_eq!(app.construction_method_name(), "Randomized Greedy");

        app.demo
            .set_construction_method(ConstructionMethod::NearestNeighbor);
        assert_eq!(app.construction_method_name(), "Nearest Neighbor");

        app.demo.set_construction_method(ConstructionMethod::Random);
        assert_eq!(app.construction_method_name(), "Random");
    }

    #[test]
    fn test_falsification_status() {
        let app = TspApp::new(10, 42);
        let status = app.falsification_status();
        // Just verify it returns something
        assert!(!status.message.is_empty());
    }

    #[test]
    fn test_verify_equation() {
        let app = TspApp::new(10, 42);
        // Should verify equation (tour length matches computed length)
        let verified = app.verify_equation();
        assert!(verified);
    }

    #[test]
    fn test_unknown_key_ignored() {
        let mut app = TspApp::new(10, 42);
        let paused_before = app.paused;
        let quit_before = app.should_quit;
        app.handle_key(KeyCode::Char('x')); // Unknown key
        assert_eq!(app.paused, paused_before);
        assert_eq!(app.should_quit, quit_before);
    }

    #[test]
    fn test_step_when_paused() {
        let mut app = TspApp::new(10, 42);
        app.paused = true;
        app.auto_run = false;
        let restarts_before = app.demo.restarts;
        app.step();
        // Should still increment frame count
        assert_eq!(app.frame_count, 1);
        // But should not run GRASP iteration
        assert_eq!(app.demo.restarts, restarts_before);
    }

    #[test]
    fn test_initial_convergence_history() {
        let app = TspApp::new(10, 42);
        // Should have initial entry from constructor
        assert!(!app.convergence_history.is_empty());
    }

    // =========================================================================
    // YAML Loading Tests (OR-001-08)
    // =========================================================================

    // Small 6-city instance for unit tests (inline)
    const SMALL_6_CITY_YAML: &str = r#"
meta:
  id: "TSP-TEST-006"
  version: "1.0.0"
  description: "6-city test"
  units: "miles"
  optimal_known: 115
cities:
  - { id: 0, name: "SF", alias: "SF", coords: { lat: 37.77, lon: -122.41 } }
  - { id: 1, name: "OAK", alias: "OAK", coords: { lat: 37.80, lon: -122.27 } }
  - { id: 2, name: "SJ", alias: "SJ", coords: { lat: 37.33, lon: -121.88 } }
  - { id: 3, name: "PA", alias: "PA", coords: { lat: 37.44, lon: -122.14 } }
  - { id: 4, name: "BRK", alias: "BRK", coords: { lat: 37.87, lon: -122.27 } }
  - { id: 5, name: "FRE", alias: "FRE", coords: { lat: 37.54, lon: -121.98 } }
matrix:
  - [0,12,48,35,14,42]
  - [12,0,42,30,4,30]
  - [48,42,0,15,46,17]
  - [35,30,15,0,32,18]
  - [14,4,46,32,0,32]
  - [42,30,17,18,32,0]
"#;

    // Full 20-city California instance from file
    const CALIFORNIA_20_YAML: &str = include_str!("../../examples/experiments/bay_area_tsp.yaml");

    #[test]
    fn test_from_yaml_small_6_city() {
        let app = TspApp::from_yaml(SMALL_6_CITY_YAML).expect("YAML should parse");
        assert_eq!(app.demo.n, 6);
        assert!(app.loaded_instance.is_some());
        assert!(app.loaded_path.is_none()); // Not from file
    }

    #[test]
    fn test_from_yaml_california_20_city() {
        let app = TspApp::from_yaml(CALIFORNIA_20_YAML).expect("YAML should parse");
        assert_eq!(app.demo.n, 20);
        assert_eq!(app.instance_id(), Some("TSP-CA-020"));
    }

    #[test]
    fn test_from_yaml_instance_id() {
        let app = TspApp::from_yaml(SMALL_6_CITY_YAML).expect("YAML should parse");
        assert_eq!(app.instance_id(), Some("TSP-TEST-006"));
    }

    #[test]
    fn test_from_yaml_optimal_known() {
        let app = TspApp::from_yaml(SMALL_6_CITY_YAML).expect("YAML should parse");
        assert_eq!(app.optimal_known(), Some(115));
    }

    #[test]
    fn test_from_yaml_no_optimal_known() {
        let yaml = r#"
meta:
  id: "TEST"
  description: "Test"
cities:
  - id: 0
    name: "A"
    alias: "A"
    coords: { lat: 0.0, lon: 0.0 }
  - id: 1
    name: "B"
    alias: "B"
    coords: { lat: 1.0, lon: 1.0 }
matrix:
  - [0, 10]
  - [10, 0]
"#;
        let app = TspApp::from_yaml(yaml).expect("YAML should parse");
        assert_eq!(app.optimal_known(), None);
    }

    #[test]
    fn test_from_yaml_file_success() {
        let app = TspApp::from_yaml_file("examples/experiments/bay_area_tsp.yaml")
            .expect("File should load");
        assert_eq!(app.demo.n, 20); // 20-city California instance
        assert!(app.loaded_path.is_some());
        assert!(app.loaded_path.as_ref().unwrap().contains("bay_area"));
    }

    #[test]
    fn test_from_yaml_file_not_found() {
        let result = TspApp::from_yaml_file("/nonexistent/path.yaml");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_yaml_invalid() {
        let result = TspApp::from_yaml("invalid yaml: [[[");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_yaml_has_convergence_history() {
        let app = TspApp::from_yaml(SMALL_6_CITY_YAML).expect("YAML should parse");
        assert!(!app.convergence_history.is_empty());
    }

    #[test]
    fn test_from_yaml_can_step() {
        let mut app = TspApp::from_yaml(SMALL_6_CITY_YAML).expect("YAML should parse");
        let initial_restarts = app.demo.restarts;
        app.step();
        assert!(app.demo.restarts > initial_restarts);
    }

    #[test]
    fn test_instance_id_none_when_not_loaded() {
        let app = TspApp::new(10, 42);
        assert!(app.instance_id().is_none());
    }

    #[test]
    fn test_optimal_known_none_when_not_loaded() {
        let app = TspApp::new(10, 42);
        assert!(app.optimal_known().is_none());
    }
}

// =============================================================================
// Property-Based Tests for TUI Visualization (OR-001-14)
// =============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Property: Convergence history never exceeds max_history.
        #[test]
        fn prop_convergence_history_bounded(seed in 0u64..5000, steps in 0usize..100) {
            let mut app = TspApp::new(10, seed);
            for _ in 0..steps {
                app.step();
            }
            prop_assert!(
                app.convergence_history.len() <= app.max_history,
                "History {} > max {}",
                app.convergence_history.len(),
                app.max_history
            );
        }

        /// Property: Frame count increases monotonically.
        #[test]
        fn prop_frame_count_monotonic(seed in 0u64..5000, steps in 1usize..50) {
            let mut app = TspApp::new(10, seed);
            let mut last_frame = app.frame_count;
            for _ in 0..steps {
                app.step();
                prop_assert!(
                    app.frame_count >= last_frame,
                    "Frame count decreased from {} to {}",
                    last_frame,
                    app.frame_count
                );
                last_frame = app.frame_count;
            }
        }

        /// Property: Best tour length is always positive after step.
        #[test]
        fn prop_tour_length_positive(seed in 0u64..5000, n in 4usize..20) {
            let mut app = TspApp::new(n, seed);
            app.step();
            prop_assert!(
                app.demo.best_tour_length > 0.0,
                "Tour length {} should be positive",
                app.demo.best_tour_length
            );
        }

        /// Property: Best tour visits all cities exactly once.
        #[test]
        fn prop_tour_visits_all_cities(seed in 0u64..5000, n in 4usize..15) {
            let mut app = TspApp::new(n, seed);
            app.step();

            let mut visited = app.demo.best_tour.clone();
            visited.sort();
            let expected: Vec<usize> = (0..n).collect();
            prop_assert_eq!(
                visited, expected,
                "Tour must visit all {} cities exactly once",
                n
            );
        }

        /// Property: RCL size stays within bounds after key presses.
        #[test]
        fn prop_rcl_bounded_after_keys(seed in 0u64..1000, increments in 0usize..20, decrements in 0usize..20) {
            let mut app = TspApp::new(10, seed);
            for _ in 0..increments {
                app.handle_key(KeyCode::Char('+'));
            }
            for _ in 0..decrements {
                app.handle_key(KeyCode::Char('-'));
            }
            prop_assert!(
                app.demo.rcl_size >= 1 && app.demo.rcl_size <= 10,
                "RCL size {} out of bounds [1, 10]",
                app.demo.rcl_size
            );
        }

        /// Property: Method cycling returns to original after 3 presses.
        #[test]
        fn prop_method_cycle_returns(seed in 0u64..5000) {
            let mut app = TspApp::new(10, seed);
            let original = app.construction_method_name();
            for _ in 0..3 {
                app.handle_key(KeyCode::Char('m'));
            }
            prop_assert_eq!(
                app.construction_method_name(),
                original,
                "Method should cycle back after 3 presses"
            );
        }

        /// Property: Convergence history entries are always positive.
        #[test]
        fn prop_convergence_history_positive(seed in 0u64..5000, steps in 0usize..30) {
            let mut app = TspApp::new(10, seed);
            for _ in 0..steps {
                app.step();
            }
            for (i, &val) in app.convergence_history.iter().enumerate() {
                prop_assert!(val > 0, "History[{}] = {} should be positive", i, val);
            }
        }

        /// Property: Reset clears convergence history.
        #[test]
        fn prop_reset_clears_history(seed in 0u64..5000, steps in 1usize..20) {
            let mut app = TspApp::new(10, seed);
            for _ in 0..steps {
                app.step();
            }
            app.reset();
            prop_assert!(
                app.convergence_history.is_empty(),
                "History should be empty after reset, got {}",
                app.convergence_history.len()
            );
        }

        /// Property: Paused state prevents iteration but not frame count.
        #[test]
        fn prop_paused_prevents_iteration(seed in 0u64..5000, steps in 1usize..10) {
            let mut app = TspApp::new(10, seed);
            app.paused = true;
            app.auto_run = false;
            let restarts_before = app.demo.restarts;

            for _ in 0..steps {
                app.step();
            }

            prop_assert_eq!(
                app.demo.restarts,
                restarts_before,
                "Paused should prevent GRASP iterations"
            );
            prop_assert!(
                app.frame_count > 0,
                "Frame count should still increment when paused"
            );
        }
    }
}

// =============================================================================
// Mutation-Sensitive Tests for TUI (OR-001-14)
// =============================================================================

#[cfg(test)]
mod mutation_tests {
    use super::*;

    /// Mutation test: Step actually runs GRASP iteration.
    #[test]
    fn mutation_step_runs_grasp() {
        let mut app = TspApp::new(10, 42);
        let restarts_before = app.demo.restarts;
        app.step();
        assert!(
            app.demo.restarts > restarts_before,
            "Step must run GRASP iteration (restarts should increase)"
        );
    }

    /// Mutation test: Quit key actually sets should_quit.
    #[test]
    fn mutation_quit_sets_flag() {
        let mut app = TspApp::new(10, 42);
        assert!(!app.should_quit);
        app.handle_key(KeyCode::Char('q'));
        assert!(app.should_quit, "Quit key must set should_quit flag");
    }

    /// Mutation test: Space actually toggles pause.
    #[test]
    fn mutation_space_toggles_pause() {
        let mut app = TspApp::new(10, 42);
        let auto_before = app.auto_run;
        app.handle_key(KeyCode::Char(' '));
        assert_ne!(app.auto_run, auto_before, "Space must toggle auto_run");
    }

    /// Mutation test: Plus key actually increases RCL.
    #[test]
    fn mutation_plus_increases_rcl() {
        let mut app = TspApp::new(10, 42);
        app.demo.set_rcl_size(3);
        app.handle_key(KeyCode::Char('+'));
        assert_eq!(app.demo.rcl_size, 4, "Plus must increase RCL size");
    }

    /// Mutation test: Minus key actually decreases RCL.
    #[test]
    fn mutation_minus_decreases_rcl() {
        let mut app = TspApp::new(10, 42);
        app.demo.set_rcl_size(5);
        app.handle_key(KeyCode::Char('-'));
        assert_eq!(app.demo.rcl_size, 4, "Minus must decrease RCL size");
    }

    /// Mutation test: Reset actually clears state.
    #[test]
    fn mutation_reset_clears_state() {
        let mut app = TspApp::new(10, 42);
        app.step();
        app.step();
        app.reset();
        assert_eq!(app.frame_count, 0, "Reset must clear frame count");
        assert!(
            app.convergence_history.is_empty(),
            "Reset must clear convergence history"
        );
    }

    /// Mutation test: Convergence history actually tracks tour length.
    #[test]
    fn mutation_history_tracks_tour() {
        let mut app = TspApp::new(10, 42);
        app.step();
        let last = *app.convergence_history.last().unwrap();
        // Value should be tour length * 1000 (as u64)
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let expected = (app.demo.best_tour_length * 1000.0).max(0.0) as u64;
        assert_eq!(last, expected, "History must track tour length * 1000");
    }
}
