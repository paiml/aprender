//! Dashboard rendering options.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// Default canvas width in pixels for the WASM dashboard
const DEFAULT_DASHBOARD_WIDTH: u32 = 800;
/// Default canvas height in pixels for the WASM dashboard
const DEFAULT_DASHBOARD_HEIGHT: u32 = 400;

/// Dashboard rendering options.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
#[derive(Debug, Clone)]
pub struct WasmDashboardOptions {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) background_color: String,
    pub(crate) loss_color: String,
    pub(crate) accuracy_color: String,
    pub(crate) show_sparklines: bool,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl WasmDashboardOptions {
    /// Create default dashboard options.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        Self {
            width: DEFAULT_DASHBOARD_WIDTH,
            height: DEFAULT_DASHBOARD_HEIGHT,
            background_color: "#1a1a2e".to_string(),
            loss_color: "#ff6b6b".to_string(),
            accuracy_color: "#4ecdc4".to_string(),
            show_sparklines: true,
        }
    }

    /// Set width in pixels.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn width(mut self, width: u32) -> Self {
        self.width = width;
        self
    }

    /// Set height in pixels.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn height(mut self, height: u32) -> Self {
        self.height = height;
        self
    }

    /// Set background color (hex format).
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn background_color(mut self, color: &str) -> Self {
        self.background_color = color.to_string();
        self
    }

    /// Set loss color (hex format).
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn loss_color(mut self, color: &str) -> Self {
        self.loss_color = color.to_string();
        self
    }

    /// Set accuracy color (hex format).
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn accuracy_color(mut self, color: &str) -> Self {
        self.accuracy_color = color.to_string();
        self
    }

    /// Enable/disable sparklines.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn show_sparklines(mut self, show: bool) -> Self {
        self.show_sparklines = show;
        self
    }

    /// Get width.
    pub fn get_width(&self) -> u32 {
        self.width
    }

    /// Get height.
    pub fn get_height(&self) -> u32 {
        self.height
    }
}

impl Default for WasmDashboardOptions {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_dashboard_options_default() {
        let opts = WasmDashboardOptions::new();
        assert_eq!(opts.width, 800);
        assert_eq!(opts.height, 400);
        assert_eq!(opts.background_color, "#1a1a2e");
    }

    #[test]
    fn test_wasm_dashboard_options_builder() {
        let opts = WasmDashboardOptions::new()
            .width(1024)
            .height(768)
            .background_color("#ffffff")
            .loss_color("#ff0000")
            .accuracy_color("#00ff00")
            .show_sparklines(false);

        assert_eq!(opts.width, 1024);
        assert_eq!(opts.height, 768);
        assert_eq!(opts.background_color, "#ffffff");
        assert_eq!(opts.loss_color, "#ff0000");
        assert_eq!(opts.accuracy_color, "#00ff00");
        assert!(!opts.show_sparklines);
    }

    #[test]
    fn test_wasm_dashboard_options_default_trait() {
        let opts = WasmDashboardOptions::default();
        assert_eq!(opts.get_width(), 800);
        assert_eq!(opts.get_height(), 400);
    }
}
