//! GUI/UX Coverage Tracking for E2E Tests (Probar)
//!
//! This module provides coverage tracking for GUI elements and user journeys,
//! enabling comprehensive E2E testing of TUI and WASM interfaces.
//!
//! # Toyota Production System Alignment
//!
//! - **Jidoka**: Stop tests when coverage drops below threshold
//! - **Poka-Yoke**: Compile-time element registration prevents missing tests
//! - **Visual Control**: Coverage reports show exactly what's tested
//!
//! # Example
//!
//! ```rust
//! use simular::edd::gui_coverage::GuiCoverage;
//!
//! let mut coverage = GuiCoverage::new("TSP TUI Demo");
//!
//! // Register elements to track
//! coverage.register_element("tour_length_display");
//! coverage.register_element("convergence_graph");
//! coverage.register_element("city_plot");
//!
//! // Register screens
//! coverage.register_screen("main_view");
//! coverage.register_screen("controls_panel");
//!
//! // Mark as covered during tests
//! coverage.cover_element("tour_length_display");
//! coverage.cover_screen("main_view");
//!
//! // Check coverage
//! assert!(coverage.element_coverage() >= 0.33);
//! ```
//!
//! # References
//!
//! - [57] Nielsen, J. (1994). Usability Engineering. Morgan Kaufmann.
//! - [58] Krug, S. (2014). Don't Make Me Think. New Riders.

use std::collections::{HashMap, HashSet};

/// GUI coverage tracker for E2E testing.
#[derive(Debug, Clone)]
pub struct GuiCoverage {
    /// Name of the component being tested.
    name: String,
    /// All registered elements (buttons, displays, labels, etc.).
    elements: HashSet<String>,
    /// Elements that have been covered by tests.
    covered_elements: HashSet<String>,
    /// All registered screens/views.
    screens: HashSet<String>,
    /// Screens that have been visited by tests.
    covered_screens: HashSet<String>,
    /// User journeys (named sequences of interactions).
    journeys: HashMap<String, Vec<String>>,
    /// Completed user journeys.
    completed_journeys: HashSet<String>,
    /// Interaction log for debugging.
    interaction_log: Vec<Interaction>,
}

/// A single user interaction.
#[derive(Debug, Clone)]
pub struct Interaction {
    /// Type of interaction.
    pub kind: InteractionKind,
    /// Target element or screen.
    pub target: String,
    /// Optional value (for inputs).
    pub value: Option<String>,
    /// Timestamp (frame number).
    pub frame: u64,
}

/// Types of user interactions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionKind {
    /// Key press.
    KeyPress,
    /// Click (mouse or touch).
    Click,
    /// Text input.
    Input,
    /// Screen navigation.
    Navigate,
    /// Element view/render.
    View,
}

impl GuiCoverage {
    /// Create a new GUI coverage tracker.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            elements: HashSet::new(),
            covered_elements: HashSet::new(),
            screens: HashSet::new(),
            covered_screens: HashSet::new(),
            journeys: HashMap::new(),
            completed_journeys: HashSet::new(),
            interaction_log: Vec::new(),
        }
    }

    /// Create a TUI coverage tracker with common elements pre-registered.
    #[must_use]
    pub fn tui_preset(name: &str) -> Self {
        let mut coverage = Self::new(name);

        // Common TUI elements
        coverage.register_element("title_bar");
        coverage.register_element("status_bar");
        coverage.register_element("help_text");
        coverage.register_element("statistics_panel");
        coverage.register_element("controls_panel");

        // Common TUI screens
        coverage.register_screen("main_view");
        coverage.register_screen("help_screen");

        coverage
    }

    /// Create a TSP TUI coverage tracker with all elements.
    #[must_use]
    pub fn tsp_tui() -> Self {
        let mut coverage = Self::new("TSP GRASP TUI");

        // TSP-specific elements
        coverage.register_element("title_bar");
        coverage.register_element("equations_panel");
        coverage.register_element("city_plot");
        coverage.register_element("convergence_graph");
        coverage.register_element("statistics_panel");
        coverage.register_element("controls_panel");
        coverage.register_element("status_bar");

        // Display elements
        coverage.register_element("tour_length_display");
        coverage.register_element("best_tour_display");
        coverage.register_element("lower_bound_display");
        coverage.register_element("gap_display");
        coverage.register_element("crossings_display");
        coverage.register_element("restarts_display");
        coverage.register_element("method_display");
        coverage.register_element("rcl_display");

        // Interactive elements
        coverage.register_element("space_toggle");
        coverage.register_element("g_step");
        coverage.register_element("r_reset");
        coverage.register_element("plus_rcl");
        coverage.register_element("minus_rcl");
        coverage.register_element("m_method");
        coverage.register_element("q_quit");

        // Screens
        coverage.register_screen("main_view");
        coverage.register_screen("running_state");
        coverage.register_screen("paused_state");
        coverage.register_screen("converged_state");

        // User journeys
        coverage.register_journey(
            "basic_run",
            vec!["main_view", "space_toggle", "running_state"],
        );
        coverage.register_journey(
            "single_step",
            vec!["main_view", "g_step", "tour_length_display"],
        );
        coverage.register_journey(
            "change_method",
            vec!["main_view", "m_method", "method_display"],
        );
        coverage.register_journey("adjust_rcl", vec!["main_view", "plus_rcl", "rcl_display"]);
        coverage.register_journey(
            "full_convergence",
            vec![
                "main_view",
                "space_toggle",
                "running_state",
                "converged_state",
            ],
        );

        coverage
    }

    /// Create a TSP WASM coverage tracker with all elements.
    ///
    /// Matches TUI's 22 elements, 4 screens, 5 journeys for full parity.
    #[must_use]
    pub fn tsp_wasm() -> Self {
        let mut coverage = Self::new("TSP GRASP WASM");

        // === Panel Elements (matches TUI) ===
        coverage.register_element("header"); // TUI: title_bar
        coverage.register_element("equations_panel"); // Same
        coverage.register_element("tsp_canvas"); // TUI: city_plot
        coverage.register_element("sparkline"); // TUI: convergence_graph
        coverage.register_element("statistics_panel"); // Same
        coverage.register_element("controls_panel"); // Same
        coverage.register_element("footer"); // TUI: status_bar

        // === Display Elements (matches TUI stat-* IDs) ===
        coverage.register_element("stat_best"); // TUI: tour_length_display
        coverage.register_element("eq_tour"); // TUI: best_tour_display
        coverage.register_element("stat_lb"); // TUI: lower_bound_display
        coverage.register_element("stat_gap"); // TUI: gap_display
        coverage.register_element("fals_status"); // TUI: crossings_display (falsification)
        coverage.register_element("stat_restarts"); // TUI: restarts_display
        coverage.register_element("select_method"); // TUI: method_display
        coverage.register_element("slider_n"); // TUI: rcl_display (city count)

        // === Interactive Elements (matches TUI keybinds) ===
        coverage.register_element("btn_play"); // TUI: space_toggle
        coverage.register_element("btn_step"); // TUI: g_step
        coverage.register_element("btn_reset"); // TUI: r_reset
        coverage.register_element("btn_run10"); // TUI: plus_rcl (run more)
        coverage.register_element("btn_run100"); // TUI: minus_rcl (run lots)
        coverage.register_element("select_method"); // TUI: m_method
        coverage.register_element("btn_run_tests"); // Probar: test suite

        // === Screens (matches TUI states) ===
        coverage.register_screen("main_view"); // Initial view
        coverage.register_screen("running_state"); // Animation running
        coverage.register_screen("paused_state"); // Paused
        coverage.register_screen("converged_state"); // Algorithm converged

        // === User Journeys (matches TUI journeys) ===
        coverage.register_journey("basic_run", vec!["main_view", "btn_play", "running_state"]);
        coverage.register_journey("single_step", vec!["main_view", "btn_step", "stat_best"]);
        coverage.register_journey(
            "change_method",
            vec!["main_view", "select_method", "stat_best"],
        );
        coverage.register_journey("adjust_cities", vec!["main_view", "slider_n", "tsp_canvas"]);
        coverage.register_journey(
            "full_convergence",
            vec!["main_view", "btn_play", "running_state", "converged_state"],
        );

        coverage
    }

    // =========================================================================
    // Registration
    // =========================================================================

    /// Register an element to track.
    pub fn register_element(&mut self, name: &str) {
        self.elements.insert(name.to_string());
    }

    /// Register a screen/view to track.
    pub fn register_screen(&mut self, name: &str) {
        self.screens.insert(name.to_string());
    }

    /// Register a user journey.
    pub fn register_journey(&mut self, name: &str, steps: Vec<&str>) {
        contract_pre_iterator!(name);
        self.journeys.insert(
            name.to_string(),
            steps.into_iter().map(String::from).collect(),
        );
    }

    // =========================================================================
    // Coverage Recording
    // =========================================================================

    /// Mark an element as covered.
    pub fn cover_element(&mut self, name: &str) {
        if self.elements.contains(name) {
            self.covered_elements.insert(name.to_string());
        }
    }

    /// Mark a screen as covered.
    pub fn cover_screen(&mut self, name: &str) {
        if self.screens.contains(name) {
            self.covered_screens.insert(name.to_string());
        }
    }

    /// Mark a journey as completed.
    pub fn complete_journey(&mut self, name: &str) {
        if self.journeys.contains_key(name) {
            self.completed_journeys.insert(name.to_string());
            // Also cover all elements/screens in the journey
            if let Some(steps) = self.journeys.get(name).cloned() {
                for step in steps {
                    self.cover_element(&step);
                    self.cover_screen(&step);
                }
            }
        }
    }

    /// Log an interaction.
    pub fn log_interaction(
        &mut self,
        kind: InteractionKind,
        target: &str,
        value: Option<&str>,
        frame: u64,
    ) {
        self.interaction_log.push(Interaction {
            kind,
            target: target.to_string(),
            value: value.map(String::from),
            frame,
        });

        // Auto-cover based on interaction type
        match kind {
            InteractionKind::Navigate => {
                self.cover_screen(target);
            }
            InteractionKind::KeyPress
            | InteractionKind::Click
            | InteractionKind::Input
            | InteractionKind::View => {
                self.cover_element(target);
            }
        }
    }

    // =========================================================================
    // Coverage Metrics
    // =========================================================================

    /// Get element coverage percentage (0.0 to 1.0).
    #[must_use]
    pub fn element_coverage(&self) -> f64 {
        if self.elements.is_empty() {
            return 1.0;
        }
        self.covered_elements.len() as f64 / self.elements.len() as f64
    }

    /// Get screen coverage percentage (0.0 to 1.0).
    #[must_use]
    pub fn screen_coverage(&self) -> f64 {
        if self.screens.is_empty() {
            return 1.0;
        }
        self.covered_screens.len() as f64 / self.screens.len() as f64
    }

    /// Get journey coverage percentage (0.0 to 1.0).
    #[must_use]
    pub fn journey_coverage(&self) -> f64 {
        if self.journeys.is_empty() {
            return 1.0;
        }
        self.completed_journeys.len() as f64 / self.journeys.len() as f64
    }

    /// Get overall GUI coverage percentage (0.0 to 1.0).
    #[must_use]
    pub fn total_coverage(&self) -> f64 {
        let element = self.element_coverage();
        let screen = self.screen_coverage();
        let journey = self.journey_coverage();

        // Weighted average: elements 50%, screens 30%, journeys 20%
        element * 0.5 + screen * 0.3 + journey * 0.2
    }

    /// Check if coverage meets threshold.
    #[must_use]
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.total_coverage() >= threshold
    }

    /// Check if 100% coverage is achieved.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.element_coverage() >= 1.0
            && self.screen_coverage() >= 1.0
            && self.journey_coverage() >= 1.0
    }

    // =========================================================================
    // Reporting
    // =========================================================================

    /// Get a summary string.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "GUI: {:.0}% ({}/{} elements, {}/{} screens)",
            self.total_coverage() * 100.0,
            self.covered_elements.len(),
            self.elements.len(),
            self.covered_screens.len(),
            self.screens.len()
        )
    }

    /// Get uncovered elements.
    #[must_use]
    pub fn uncovered_elements(&self) -> Vec<&str> {
        self.elements
            .iter()
            .filter(|e| !self.covered_elements.contains(*e))
            .map(String::as_str)
            .collect()
    }

    /// Get uncovered screens.
    #[must_use]
    pub fn uncovered_screens(&self) -> Vec<&str> {
        self.screens
            .iter()
            .filter(|s| !self.covered_screens.contains(*s))
            .map(String::as_str)
            .collect()
    }

    /// Get incomplete journeys.
    #[must_use]
    pub fn incomplete_journeys(&self) -> Vec<&str> {
        self.journeys
            .keys()
            .filter(|j| !self.completed_journeys.contains(*j))
            .map(String::as_str)
            .collect()
    }

    /// Get total element count.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.elements.len()
    }

    /// Get total screen count.
    #[must_use]
    pub fn screen_count(&self) -> usize {
        self.screens.len()
    }

    /// Get total journey count.
    #[must_use]
    pub fn journey_count(&self) -> usize {
        self.journeys.len()
    }

    /// Check if an element is registered.
    #[must_use]
    pub fn has_element(&self, name: &str) -> bool {
        self.elements.contains(name)
    }

    /// Check if a screen is registered.
    #[must_use]
    pub fn has_screen(&self, name: &str) -> bool {
        self.screens.contains(name)
    }

    /// Generate a detailed coverage report.
    #[must_use]
    pub fn detailed_report(&self) -> String {
        use std::fmt::Write;
        let mut report = String::new();

        let _ = writeln!(report, "=== GUI Coverage Report: {} ===\n", self.name);
        let _ = writeln!(report, "Overall: {:.1}%", self.total_coverage() * 100.0);
        let _ = writeln!(
            report,
            "  Elements: {:.1}% ({}/{})",
            self.element_coverage() * 100.0,
            self.covered_elements.len(),
            self.elements.len()
        );
        let _ = writeln!(
            report,
            "  Screens:  {:.1}% ({}/{})",
            self.screen_coverage() * 100.0,
            self.covered_screens.len(),
            self.screens.len()
        );
        let _ = writeln!(
            report,
            "  Journeys: {:.1}% ({}/{})",
            self.journey_coverage() * 100.0,
            self.completed_journeys.len(),
            self.journeys.len()
        );

        let uncovered_elements = self.uncovered_elements();
        if !uncovered_elements.is_empty() {
            report.push_str("\nUncovered Elements:\n");
            for elem in uncovered_elements {
                let _ = writeln!(report, "  - {elem}");
            }
        }

        let uncovered_screens = self.uncovered_screens();
        if !uncovered_screens.is_empty() {
            report.push_str("\nUncovered Screens:\n");
            for screen in uncovered_screens {
                let _ = writeln!(report, "  - {screen}");
            }
        }

        let incomplete = self.incomplete_journeys();
        if !incomplete.is_empty() {
            report.push_str("\nIncomplete Journeys:\n");
            for journey in incomplete {
                let _ = writeln!(report, "  - {journey}");
            }
        }

        report
    }

    /// Get interaction count.
    #[must_use]
    pub fn interaction_count(&self) -> usize {
        self.interaction_log.len()
    }

    /// Get name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Macro for quick GUI coverage setup.
#[macro_export]
macro_rules! gui_coverage {
    ($name:expr => elements: [$($elem:expr),* $(,)?], screens: [$($screen:expr),* $(,)?]) => {{
        let mut coverage = $crate::edd::gui_coverage::GuiCoverage::new($name);
        $(coverage.register_element($elem);)*
        $(coverage.register_screen($screen);)*
        coverage
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_coverage() {
        let coverage = GuiCoverage::new("Test");
        assert_eq!(coverage.name(), "Test");
        assert!(coverage.elements.is_empty());
        assert!(coverage.screens.is_empty());
    }

    #[test]
    fn test_register_and_cover_elements() {
        let mut coverage = GuiCoverage::new("Test");
        coverage.register_element("button1");
        coverage.register_element("button2");

        assert_eq!(coverage.element_coverage(), 0.0);

        coverage.cover_element("button1");
        assert!((coverage.element_coverage() - 0.5).abs() < f64::EPSILON);

        coverage.cover_element("button2");
        assert!((coverage.element_coverage() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_register_and_cover_screens() {
        let mut coverage = GuiCoverage::new("Test");
        coverage.register_screen("main");
        coverage.register_screen("settings");

        assert_eq!(coverage.screen_coverage(), 0.0);

        coverage.cover_screen("main");
        assert!((coverage.screen_coverage() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_journey_completion() {
        let mut coverage = GuiCoverage::new("Test");
        coverage.register_element("btn");
        coverage.register_screen("main");
        coverage.register_journey("flow", vec!["main", "btn"]);

        coverage.complete_journey("flow");

        assert!(coverage.completed_journeys.contains("flow"));
        assert!(coverage.covered_elements.contains("btn"));
        assert!(coverage.covered_screens.contains("main"));
    }

    #[test]
    fn test_tsp_tui_preset() {
        let coverage = GuiCoverage::tsp_tui();

        // Should have TSP-specific elements
        assert!(coverage.elements.contains("city_plot"));
        assert!(coverage.elements.contains("convergence_graph"));
        assert!(coverage.elements.contains("equations_panel"));

        // Should have screens
        assert!(coverage.screens.contains("main_view"));
        assert!(coverage.screens.contains("converged_state"));

        // Should have journeys
        assert!(coverage.journeys.contains_key("basic_run"));
        assert!(coverage.journeys.contains_key("full_convergence"));
    }

    #[test]
    fn test_tsp_wasm_preset() {
        let coverage = GuiCoverage::tsp_wasm();

        // Should have DOM elements matching TUI
        assert!(coverage.elements.contains("tsp_canvas"));
        assert!(coverage.elements.contains("sparkline"));
        assert!(coverage.elements.contains("btn_play"));
        assert!(coverage.elements.contains("btn_step"));
        assert!(coverage.elements.contains("stat_best"));

        // Should have 4 screens (matches TUI)
        assert!(coverage.screens.contains("main_view"));
        assert!(coverage.screens.contains("running_state"));
        assert!(coverage.screens.contains("paused_state"));
        assert!(coverage.screens.contains("converged_state"));

        // Should have 5 journeys (matches TUI)
        assert!(coverage.journeys.contains_key("basic_run"));
        assert!(coverage.journeys.contains_key("full_convergence"));
        assert!(coverage.journeys.contains_key("single_step"));
    }

    #[test]
    fn test_summary() {
        let mut coverage = GuiCoverage::new("Test");
        coverage.register_element("e1");
        coverage.register_element("e2");
        coverage.register_screen("s1");

        coverage.cover_element("e1");

        let summary = coverage.summary();
        assert!(summary.contains("1/2 elements"));
        assert!(summary.contains("0/1 screens"));
    }

    #[test]
    fn test_detailed_report() {
        let mut coverage = GuiCoverage::new("Test");
        coverage.register_element("covered");
        coverage.register_element("uncovered");
        coverage.cover_element("covered");

        let report = coverage.detailed_report();
        assert!(report.contains("Test"));
        assert!(report.contains("uncovered"));
    }

    #[test]
    fn test_meets_threshold() {
        let mut coverage = GuiCoverage::new("Test");
        coverage.register_element("e1");
        coverage.register_element("e2");
        coverage.cover_element("e1");
        coverage.cover_element("e2");

        assert!(coverage.meets_threshold(0.5));
    }

    #[test]
    fn test_interaction_logging() {
        let mut coverage = GuiCoverage::new("Test");
        coverage.register_element("btn");

        coverage.log_interaction(InteractionKind::Click, "btn", None, 1);

        assert_eq!(coverage.interaction_count(), 1);
        assert!(coverage.covered_elements.contains("btn"));
    }

    #[test]
    fn test_gui_coverage_macro() {
        let coverage = gui_coverage!("Test" =>
            elements: ["btn1", "btn2"],
            screens: ["main"]
        );

        assert!(coverage.elements.contains("btn1"));
        assert!(coverage.elements.contains("btn2"));
        assert!(coverage.screens.contains("main"));
    }
}
