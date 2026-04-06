//! WebAssembly bindings for trueno-viz.
//!
//! Provides JavaScript-accessible functions for creating visualizations
//! in the browser without any JavaScript charting libraries.
//!
//! # Usage (JavaScript)
//!
//! ```javascript
//! import init, { scatter_plot, line_chart, histogram } from 'trueno-viz';
//!
//! await init();
//!
//! // Create a scatter plot and get PNG data
//! const pngData = scatter_plot(
//!     new Float32Array([1, 2, 3, 4, 5]),
//!     new Float32Array([2, 4, 3, 5, 4]),
//!     { width: 800, height: 600, color: '#4285F4' }
//! );
//!
//! // Display in an image element
//! const blob = new Blob([pngData], { type: 'image/png' });
//! document.getElementById('chart').src = URL.createObjectURL(blob);
//! ```

use batuta_common::display::WithDimensions;
use wasm_bindgen::prelude::*;

use crate::color::Rgba;
use crate::grammar::{Aes, Coord, GGPlot, Geom, Theme};
use crate::output::PngEncoder;
use crate::plots::{BinStrategy, Heatmap, Histogram, LineChart, LineSeries, ScatterPlot};
use crate::prompt;

// ============================================================================
// Initialization
// ============================================================================

/// Initialize the WASM module.
///
/// Call this before using any other functions.
#[wasm_bindgen(start)]
pub fn init() {
    // WASM module initialized
}

// ============================================================================
// Plot Options
// ============================================================================

/// Options for plot rendering.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct PlotOptions {
    width: u32,
    height: u32,
    color: String,
    background: String,
    title: Option<String>,
    point_size: f32,
    line_width: f32,
}

#[wasm_bindgen]
impl PlotOptions {
    /// Create default plot options.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            width: 800,
            height: 600,
            color: "#4285F4".to_string(),
            background: "#FFFFFF".to_string(),
            title: None,
            point_size: 5.0,
            line_width: 2.0,
        }
    }

    /// Set width in pixels.
    #[wasm_bindgen]
    pub fn width(mut self, width: u32) -> Self {
        self.width = width;
        self
    }

    /// Set height in pixels.
    #[wasm_bindgen]
    pub fn height(mut self, height: u32) -> Self {
        self.height = height;
        self
    }

    /// Set the primary color (hex format: #RRGGBB).
    #[wasm_bindgen]
    pub fn color(mut self, color: &str) -> Self {
        self.color = color.to_string();
        self
    }

    /// Set the background color (hex format: #RRGGBB).
    #[wasm_bindgen]
    pub fn background(mut self, bg: &str) -> Self {
        self.background = bg.to_string();
        self
    }

    /// Set the title.
    #[wasm_bindgen]
    pub fn title(mut self, title: &str) -> Self {
        self.title = Some(title.to_string());
        self
    }

    /// Set point size for scatter plots.
    #[wasm_bindgen]
    pub fn point_size(mut self, size: f32) -> Self {
        self.point_size = size;
        self
    }

    /// Set line width for line charts.
    #[wasm_bindgen]
    pub fn line_width(mut self, width: f32) -> Self {
        self.line_width = width;
        self
    }
}

impl Default for PlotOptions {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn parse_hex_color(hex: &str) -> Rgba {
    let hex = hex.trim_start_matches('#');
    if hex.len() == 6 {
        let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(0);
        let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(0);
        let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(0);
        Rgba::new(r, g, b, 255)
    } else {
        Rgba::new(66, 133, 244, 255) // Default blue
    }
}

// ============================================================================
// Plot Functions
// ============================================================================

/// Create a scatter plot and return PNG data.
///
/// # Arguments
///
/// * `x` - X coordinates as Float32Array
/// * `y` - Y coordinates as Float32Array
/// * `options` - Plot options (optional)
///
/// # Returns
///
/// PNG image data as Uint8Array
#[wasm_bindgen]
pub fn scatter_plot(
    x: &[f32],
    y: &[f32],
    options: Option<PlotOptions>,
) -> Result<Vec<u8>, JsValue> {
    let opts = options.unwrap_or_default();
    let color = parse_hex_color(&opts.color);

    let plot = ScatterPlot::new()
        .x(x)
        .y(y)
        .color(color)
        .size(opts.point_size)
        .dimensions(opts.width, opts.height)
        .build()
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let fb = plot.to_framebuffer().map_err(|e| JsValue::from_str(&e.to_string()))?;

    PngEncoder::to_bytes(&fb).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Create a line chart and return PNG data.
///
/// # Arguments
///
/// * `x` - X coordinates as Float32Array
/// * `y` - Y coordinates as Float32Array
/// * `options` - Plot options (optional)
///
/// # Returns
///
/// PNG image data as Uint8Array
#[wasm_bindgen]
pub fn line_chart(x: &[f32], y: &[f32], options: Option<PlotOptions>) -> Result<Vec<u8>, JsValue> {
    let opts = options.unwrap_or_default();
    let color = parse_hex_color(&opts.color);

    let plot = LineChart::new()
        .add_series(LineSeries::new("data").data(x, y).color(color).thickness(opts.line_width))
        .dimensions(opts.width, opts.height)
        .build()
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let fb = plot.to_framebuffer().map_err(|e| JsValue::from_str(&e.to_string()))?;

    PngEncoder::to_bytes(&fb).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Create a histogram and return PNG data.
///
/// # Arguments
///
/// * `data` - Data values as Float32Array
/// * `bins` - Number of bins (optional, defaults to 10)
/// * `options` - Plot options (optional)
///
/// # Returns
///
/// PNG image data as Uint8Array
#[wasm_bindgen]
pub fn histogram(
    data: &[f32],
    bins: Option<usize>,
    options: Option<PlotOptions>,
) -> Result<Vec<u8>, JsValue> {
    let opts = options.unwrap_or_default();
    let color = parse_hex_color(&opts.color);
    let num_bins = bins.unwrap_or(10);

    let plot = Histogram::new()
        .data(data)
        .bins(BinStrategy::Fixed(num_bins))
        .color(color)
        .dimensions(opts.width, opts.height)
        .build()
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let fb = plot.to_framebuffer().map_err(|e| JsValue::from_str(&e.to_string()))?;

    PngEncoder::to_bytes(&fb).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Create a heatmap and return PNG data.
///
/// # Arguments
///
/// * `data` - Flattened 2D data as Float32Array (row-major)
/// * `rows` - Number of rows
/// * `cols` - Number of columns
/// * `options` - Plot options (optional)
///
/// # Returns
///
/// PNG image data as Uint8Array
#[wasm_bindgen]
pub fn heatmap(
    data: &[f32],
    rows: usize,
    cols: usize,
    options: Option<PlotOptions>,
) -> Result<Vec<u8>, JsValue> {
    let opts = options.unwrap_or_default();

    let plot = Heatmap::new()
        .data(data, rows, cols)
        .dimensions(opts.width, opts.height)
        .build()
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let fb = plot.to_framebuffer().map_err(|e| JsValue::from_str(&e.to_string()))?;

    PngEncoder::to_bytes(&fb).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Create a plot from a text prompt and return PNG data.
///
/// # Arguments
///
/// * `prompt` - Text prompt (e.g., "scatter x=[1,2,3] y=[4,5,6] color=blue")
///
/// # Returns
///
/// PNG image data as Uint8Array
///
/// # Example
///
/// ```javascript
/// const png = from_prompt("scatter x=[1,2,3,4,5] y=[2,4,3,5,4] color=red size=8");
/// ```
#[wasm_bindgen]
pub fn from_prompt(prompt_str: &str) -> Result<Vec<u8>, JsValue> {
    let fb = prompt::from_prompt(prompt_str).map_err(|e| JsValue::from_str(&e.to_string()))?;

    PngEncoder::to_bytes(&fb).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Create a Grammar of Graphics plot and return PNG data.
///
/// # Arguments
///
/// * `x` - X coordinates
/// * `y` - Y coordinates
/// * `geom` - Geometry type: "point", "line", "bar", "area"
/// * `theme` - Theme: "grey", "minimal", "dark", "classic"
/// * `options` - Plot options
///
/// # Returns
///
/// PNG image data as Uint8Array
#[wasm_bindgen]
pub fn ggplot(
    x: &[f32],
    y: &[f32],
    geom: &str,
    theme: &str,
    options: Option<PlotOptions>,
) -> Result<Vec<u8>, JsValue> {
    let opts = options.unwrap_or_default();
    let color = parse_hex_color(&opts.color);

    let geom_layer = match geom {
        "point" => Geom::point().aes(Aes::new().color_value(color).size_value(opts.point_size)),
        "line" => Geom::line(),
        "bar" => Geom::bar().aes(Aes::new().color_value(color)),
        "area" => Geom::area().aes(Aes::new().color_value(color)),
        _ => Geom::point().aes(Aes::new().color_value(color)),
    };

    let theme_obj = match theme {
        "minimal" => Theme::minimal(),
        "dark" => Theme::dark(),
        "classic" => Theme::classic(),
        "bw" => Theme::bw(),
        "void" => Theme::void(),
        _ => Theme::grey(),
    };

    let plot = GGPlot::new()
        .data_xy(x, y)
        .geom(geom_layer)
        .aes(Aes::new().color_value(color))
        .theme(theme_obj)
        .coord(Coord::cartesian())
        .dimensions(opts.width, opts.height)
        .build()
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let fb = plot.to_framebuffer().map_err(|e| JsValue::from_str(&e.to_string()))?;

    PngEncoder::to_bytes(&fb).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Get the library version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hex_color() {
        let color = parse_hex_color("#FF0000");
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 0);
        assert_eq!(color.b, 0);
    }

    #[test]
    fn test_parse_hex_color_no_hash() {
        let color = parse_hex_color("00FF00");
        assert_eq!(color.r, 0);
        assert_eq!(color.g, 255);
        assert_eq!(color.b, 0);
    }

    #[test]
    fn test_plot_options_default() {
        let opts = PlotOptions::new();
        assert_eq!(opts.width, 800);
        assert_eq!(opts.height, 600);
    }

    #[test]
    fn test_plot_options_builder() {
        let opts = PlotOptions::new().width(1024).height(768).color("#FF5500").point_size(10.0);

        assert_eq!(opts.width, 1024);
        assert_eq!(opts.height, 768);
        assert_eq!(opts.color, "#FF5500");
        assert!((opts.point_size - 10.0).abs() < 0.01);
    }
}
