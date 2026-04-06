//! WASM dashboard for canvas rendering.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use super::collector::WasmMetricsCollector;
use super::options::WasmDashboardOptions;
use super::utils::{generate_sparkline, normalize_values};

/// WASM dashboard for canvas rendering.
///
/// Renders training metrics to a canvas element in the browser.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
#[derive(Debug)]
pub struct WasmDashboard {
    options: WasmDashboardOptions,
    loss_history: Vec<f64>,
    accuracy_history: Vec<f64>,
    max_history: usize,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl WasmDashboard {
    /// Create a new dashboard with default options.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        Self {
            options: WasmDashboardOptions::new(),
            loss_history: Vec::new(),
            accuracy_history: Vec::new(),
            max_history: 100,
        }
    }

    /// Create a dashboard with custom options.
    pub fn with_options(options: WasmDashboardOptions) -> Self {
        Self { options, loss_history: Vec::new(), accuracy_history: Vec::new(), max_history: 100 }
    }

    /// Set maximum history length.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn max_history(mut self, max: usize) -> Self {
        self.max_history = max;
        self
    }

    /// Update dashboard with new metrics from collector.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn update(&mut self, collector: &WasmMetricsCollector) {
        // Get latest values
        let loss = collector.loss_mean();
        let accuracy = collector.accuracy_mean();

        // Add to history if valid
        if !loss.is_nan() {
            self.loss_history.push(loss);
            if self.loss_history.len() > self.max_history {
                self.loss_history.remove(0);
            }
        }

        if !accuracy.is_nan() {
            self.accuracy_history.push(accuracy);
            if self.accuracy_history.len() > self.max_history {
                self.accuracy_history.remove(0);
            }
        }
    }

    /// Add a loss value directly to history.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn add_loss(&mut self, value: f64) {
        if !value.is_nan() && !value.is_infinite() {
            self.loss_history.push(value);
            if self.loss_history.len() > self.max_history {
                self.loss_history.remove(0);
            }
        }
    }

    /// Add an accuracy value directly to history.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn add_accuracy(&mut self, value: f64) {
        if !value.is_nan() && !value.is_infinite() {
            self.accuracy_history.push(value);
            if self.accuracy_history.len() > self.max_history {
                self.accuracy_history.remove(0);
            }
        }
    }

    /// Get loss history length.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn loss_history_len(&self) -> usize {
        self.loss_history.len()
    }

    /// Get accuracy history length.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn accuracy_history_len(&self) -> usize {
        self.accuracy_history.len()
    }

    /// Clear all history.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn clear(&mut self) {
        self.loss_history.clear();
        self.accuracy_history.clear();
    }

    /// Get canvas width.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn width(&self) -> u32 {
        self.options.width
    }

    /// Get canvas height.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn height(&self) -> u32 {
        self.options.height
    }

    /// Get loss history as normalized Y coordinates (0.0-1.0).
    /// Returns empty vec if no history.
    pub fn loss_normalized(&self) -> Vec<f64> {
        normalize_values(&self.loss_history)
    }

    /// Get accuracy history as normalized Y coordinates (0.0-1.0).
    /// Accuracy is already 0-1 typically, but this handles edge cases.
    pub fn accuracy_normalized(&self) -> Vec<f64> {
        normalize_values(&self.accuracy_history)
    }

    /// Get X coordinates for plotting (normalized 0.0-1.0).
    pub fn x_coordinates(&self, len: usize) -> Vec<f64> {
        if len <= 1 {
            return vec![0.5];
        }
        (0..len).map(|i| i as f64 / (len - 1) as f64).collect()
    }

    /// Generate sparkline characters for terminal display.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn loss_sparkline(&self) -> String {
        generate_sparkline(&self.loss_history, 20)
    }

    /// Generate sparkline characters for accuracy.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn accuracy_sparkline(&self) -> String {
        generate_sparkline(&self.accuracy_history, 20)
    }

    /// Get dashboard state as JSON.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn state_json(&self) -> String {
        let state = DashboardState {
            width: self.options.width,
            height: self.options.height,
            loss_history: self.loss_history.clone(),
            accuracy_history: self.accuracy_history.clone(),
            loss_color: self.options.loss_color.clone(),
            accuracy_color: self.options.accuracy_color.clone(),
            background_color: self.options.background_color.clone(),
        };
        serde_json::to_string(&state).unwrap_or_else(|_err| "{}".to_string())
    }
}

impl Default for WasmDashboard {
    fn default() -> Self {
        Self::new()
    }
}

/// Dashboard state for JSON serialization.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct DashboardState {
    width: u32,
    height: u32,
    loss_history: Vec<f64>,
    accuracy_history: Vec<f64>,
    loss_color: String,
    accuracy_color: String,
    background_color: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_dashboard_new() {
        let dashboard = WasmDashboard::new();
        assert_eq!(dashboard.loss_history_len(), 0);
        assert_eq!(dashboard.accuracy_history_len(), 0);
        assert_eq!(dashboard.width(), 800);
        assert_eq!(dashboard.height(), 400);
    }

    #[test]
    fn test_wasm_dashboard_add_loss() {
        let mut dashboard = WasmDashboard::new();
        dashboard.add_loss(0.5);
        dashboard.add_loss(0.3);
        assert_eq!(dashboard.loss_history_len(), 2);
    }

    #[test]
    fn test_wasm_dashboard_add_accuracy() {
        let mut dashboard = WasmDashboard::new();
        dashboard.add_accuracy(0.8);
        dashboard.add_accuracy(0.9);
        assert_eq!(dashboard.accuracy_history_len(), 2);
    }

    #[test]
    fn test_wasm_dashboard_ignores_nan() {
        let mut dashboard = WasmDashboard::new();
        dashboard.add_loss(0.5);
        dashboard.add_loss(f64::NAN);
        dashboard.add_loss(0.3);
        assert_eq!(dashboard.loss_history_len(), 2);
    }

    #[test]
    fn test_wasm_dashboard_ignores_inf() {
        let mut dashboard = WasmDashboard::new();
        dashboard.add_accuracy(0.8);
        dashboard.add_accuracy(f64::INFINITY);
        assert_eq!(dashboard.accuracy_history_len(), 1);
    }

    #[test]
    fn test_wasm_dashboard_max_history() {
        let mut dashboard = WasmDashboard::new().max_history(5);
        for i in 0..10 {
            dashboard.add_loss(f64::from(i));
        }
        assert_eq!(dashboard.loss_history_len(), 5);
    }

    #[test]
    fn test_wasm_dashboard_clear() {
        let mut dashboard = WasmDashboard::new();
        dashboard.add_loss(0.5);
        dashboard.add_accuracy(0.8);
        dashboard.clear();
        assert_eq!(dashboard.loss_history_len(), 0);
        assert_eq!(dashboard.accuracy_history_len(), 0);
    }

    #[test]
    fn test_wasm_dashboard_update() {
        let mut collector = WasmMetricsCollector::new();
        collector.record_loss(0.5);
        collector.record_accuracy(0.8);

        let mut dashboard = WasmDashboard::new();
        dashboard.update(&collector);

        assert_eq!(dashboard.loss_history_len(), 1);
        assert_eq!(dashboard.accuracy_history_len(), 1);
    }

    #[test]
    fn test_wasm_dashboard_sparkline() {
        let mut dashboard = WasmDashboard::new();
        for i in 0..10 {
            dashboard.add_loss(f64::from(i) / 10.0);
        }

        let sparkline = dashboard.loss_sparkline();
        assert!(!sparkline.is_empty());
        assert!(sparkline.chars().count() <= 20);
    }

    #[test]
    fn test_wasm_dashboard_state_json() {
        let mut dashboard = WasmDashboard::new();
        dashboard.add_loss(0.5);
        dashboard.add_accuracy(0.8);

        let json = dashboard.state_json();
        assert!(json.contains("width"));
        assert!(json.contains("loss_history"));
        assert!(json.contains("accuracy_history"));
    }

    #[test]
    fn test_wasm_dashboard_x_coordinates() {
        let dashboard = WasmDashboard::new();

        let coords = dashboard.x_coordinates(5);
        assert_eq!(coords.len(), 5);
        assert!((coords[0] - 0.0).abs() < 1e-6);
        assert!((coords[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_wasm_dashboard_x_coordinates_single() {
        let dashboard = WasmDashboard::new();
        let coords = dashboard.x_coordinates(1);
        assert_eq!(coords.len(), 1);
        assert!((coords[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_wasm_dashboard_loss_normalized() {
        let mut dashboard = WasmDashboard::new();
        dashboard.add_loss(0.0);
        dashboard.add_loss(5.0);
        dashboard.add_loss(10.0);

        let normalized = dashboard.loss_normalized();
        assert_eq!(normalized.len(), 3);
        assert!((normalized[0] - 0.0).abs() < 1e-6);
        assert!((normalized[1] - 0.5).abs() < 1e-6);
        assert!((normalized[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_wasm_dashboard_with_options() {
        let opts = WasmDashboardOptions::new().width(1024).height(768);
        let dashboard = WasmDashboard::with_options(opts);
        assert_eq!(dashboard.width(), 1024);
        assert_eq!(dashboard.height(), 768);
    }

    #[test]
    fn test_wasm_dashboard_default() {
        let dashboard = WasmDashboard::default();
        assert_eq!(dashboard.width(), 800);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Property: Dashboard respects max_history
        #[test]
        fn prop_dashboard_max_history(
            max in 5usize..50,
            count in 10usize..200
        ) {
            let mut dashboard = WasmDashboard::new().max_history(max);

            for i in 0..count {
                dashboard.add_loss(i as f64);
            }

            prop_assert!(dashboard.loss_history_len() <= max);
        }

        /// Property: X coordinates span [0, 1]
        #[test]
        fn prop_x_coords_span(len in 2usize..100) {
            let dashboard = WasmDashboard::new();
            let coords = dashboard.x_coordinates(len);

            prop_assert_eq!(coords.len(), len);
            prop_assert!((coords[0] - 0.0).abs() < 1e-10);
            prop_assert!((coords[len - 1] - 1.0).abs() < 1e-10);
        }

        /// Property: Dashboard ignores NaN values
        #[test]
        fn prop_dashboard_ignores_nan(
            valid_count in 1usize..20,
            nan_count in 1usize..10
        ) {
            let mut dashboard = WasmDashboard::new();

            for i in 0..valid_count {
                dashboard.add_loss(i as f64);
            }
            for _ in 0..nan_count {
                dashboard.add_loss(f64::NAN);
            }

            prop_assert_eq!(dashboard.loss_history_len(), valid_count);
        }

        /// Property: JSON state is valid
        #[test]
        fn prop_json_state_valid(values in prop::collection::vec(0.0f64..100.0, 0..50)) {
            let mut dashboard = WasmDashboard::new();
            for v in &values {
                dashboard.add_loss(*v);
            }

            let json = dashboard.state_json();
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(&json);
            prop_assert!(parsed.is_ok());
        }
    }
}
