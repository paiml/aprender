//! trueno-viz WASM Demo Package
//!
//! GPU-first visualization with tiered compute fallback:
//! - Tier 1: WebGPU compute shaders (preferred, requires `webgpu` feature)
//! - Tier 2: WASM SIMD128 (128-bit SIMD)
//! - Tier 3: Scalar fallback (always available)

#![allow(clippy::unwrap_used)] // WASM demo code

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use web_sys::{console, window};

#[cfg(feature = "webgpu")]
use wasm_bindgen_futures::JsFuture;

// ============================================================================
// Initialization
// ============================================================================

/// Initialize panic hook for better error messages in browser console
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    console::log_1(&"trueno-viz WASM initialized".into());
}

/// Get library version
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// ============================================================================
// Compute Tier Detection
// ============================================================================

/// Detected compute tier
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeTier {
    /// WebGPU compute shaders available
    WebGPU,
    /// WASM SIMD128 (128-bit vectors)
    Simd128,
    /// Scalar fallback
    Scalar,
}

/// Compute capabilities detected at runtime
#[wasm_bindgen]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComputeCapabilities {
    tier: ComputeTier,
    webgpu_available: bool,
    simd128_available: bool,
    adapter_name: Option<String>,
}

#[wasm_bindgen]
impl ComputeCapabilities {
    #[wasm_bindgen(getter)]
    pub fn tier(&self) -> ComputeTier {
        self.tier
    }

    #[wasm_bindgen(getter)]
    pub fn webgpu_available(&self) -> bool {
        self.webgpu_available
    }

    #[wasm_bindgen(getter)]
    pub fn simd128_available(&self) -> bool {
        self.simd128_available
    }

    #[wasm_bindgen(getter)]
    pub fn adapter_name(&self) -> Option<String> {
        self.adapter_name.clone()
    }

    #[wasm_bindgen]
    pub fn tier_name(&self) -> String {
        match self.tier {
            ComputeTier::WebGPU => "WebGPU".to_string(),
            ComputeTier::Simd128 => "SIMD128".to_string(),
            ComputeTier::Scalar => "Scalar".to_string(),
        }
    }

    #[wasm_bindgen]
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

/// Detect available compute capabilities
#[wasm_bindgen]
#[cfg(feature = "webgpu")]
pub async fn detect_compute_tier() -> ComputeCapabilities {
    let webgpu_result = detect_webgpu().await;
    let simd128_available = detect_simd128();

    let (webgpu_available, adapter_name) = match webgpu_result {
        Ok(name) => (true, Some(name)),
        Err(_) => (false, None),
    };

    let tier = if webgpu_available {
        ComputeTier::WebGPU
    } else if simd128_available {
        ComputeTier::Simd128
    } else {
        ComputeTier::Scalar
    };

    ComputeCapabilities {
        tier,
        webgpu_available,
        simd128_available,
        adapter_name,
    }
}

/// Detect available compute capabilities (non-WebGPU build)
#[wasm_bindgen]
#[cfg(not(feature = "webgpu"))]
pub fn detect_compute_tier() -> ComputeCapabilities {
    let simd128_available = detect_simd128();

    let tier = if simd128_available {
        ComputeTier::Simd128
    } else {
        ComputeTier::Scalar
    };

    ComputeCapabilities {
        tier,
        webgpu_available: false,
        simd128_available,
        adapter_name: None,
    }
}

/// Check if WebGPU is available and return adapter name
#[cfg(feature = "webgpu")]
async fn detect_webgpu() -> Result<String, JsValue> {
    let window = window().ok_or_else(|| JsValue::from_str("No window"))?;
    let navigator = window.navigator();

    // Get GPU object
    let gpu = navigator.gpu();

    // Request adapter
    let adapter_promise = gpu.request_adapter();
    let adapter_result = JsFuture::from(adapter_promise).await?;

    if adapter_result.is_null() || adapter_result.is_undefined() {
        return Err(JsValue::from_str("No adapter"));
    }

    let _adapter: web_sys::GpuAdapter = adapter_result.dyn_into()?;

    // WebGPU adapter available - detailed info is obtained via JS-side detection
    Ok("WebGPU Available".to_string())
}

/// Check if WASM SIMD128 is supported
fn detect_simd128() -> bool {
    // SIMD128 detection is compile-time, but we can check at runtime
    // if the binary was compiled with SIMD support
    #[cfg(target_feature = "simd128")]
    {
        true
    }
    #[cfg(not(target_feature = "simd128"))]
    {
        false
    }
}

// ============================================================================
// Visualization Data Structures
// ============================================================================

/// Point in 2D space
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Point2D {
    pub x: f32,
    pub y: f32,
}

#[wasm_bindgen]
impl Point2D {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// RGBA color
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

#[wasm_bindgen]
impl Color {
    #[wasm_bindgen(constructor)]
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    #[wasm_bindgen]
    pub fn to_css(&self) -> String {
        if self.a == 255 {
            format!("rgb({},{},{})", self.r, self.g, self.b)
        } else {
            format!(
                "rgba({},{},{},{})",
                self.r,
                self.g,
                self.b,
                self.a as f32 / 255.0
            )
        }
    }
}

// Color constants (outside wasm_bindgen impl)
impl Color {
    pub const RED: Color = Color { r: 255, g: 0, b: 0, a: 255 };
    pub const GREEN: Color = Color { r: 0, g: 255, b: 0, a: 255 };
    pub const BLUE: Color = Color { r: 0, g: 0, b: 255, a: 255 };
    pub const WHITE: Color = Color { r: 255, g: 255, b: 255, a: 255 };
    pub const BLACK: Color = Color { r: 0, g: 0, b: 0, a: 255 };
}

// ============================================================================
// Scatter Plot Visualization
// ============================================================================

/// Scatter plot configuration
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct ScatterConfig {
    width: u32,
    height: u32,
    point_size: f32,
    color: Color,
}

#[wasm_bindgen]
impl ScatterConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            width: 800,
            height: 600,
            point_size: 4.0,
            color: Color::BLUE,
        }
    }

    #[wasm_bindgen]
    pub fn dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    #[wasm_bindgen]
    pub fn point_size(mut self, size: f32) -> Self {
        self.point_size = size;
        self
    }

    #[wasm_bindgen]
    pub fn color(mut self, r: u8, g: u8, b: u8) -> Self {
        self.color = Color::new(r, g, b, 255);
        self
    }
}

impl Default for ScatterConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Scatter plot renderer with tiered compute
#[wasm_bindgen]
pub struct ScatterPlot {
    config: ScatterConfig,
    x_data: Vec<f32>,
    y_data: Vec<f32>,
}

#[wasm_bindgen]
impl ScatterPlot {
    #[wasm_bindgen(constructor)]
    pub fn new(config: ScatterConfig) -> Self {
        Self {
            config,
            x_data: Vec::new(),
            y_data: Vec::new(),
        }
    }

    /// Set data from JavaScript arrays
    #[wasm_bindgen]
    pub fn set_data(&mut self, x: Vec<f32>, y: Vec<f32>) {
        self.x_data = x;
        self.y_data = y;
    }

    /// Get point count
    #[wasm_bindgen]
    pub fn point_count(&self) -> usize {
        self.x_data.len()
    }

    /// Compute min/max bounds (uses appropriate compute tier)
    #[wasm_bindgen]
    pub fn compute_bounds(&self) -> JsValue {
        let (x_min, x_max, y_min, y_max) = self.compute_bounds_scalar();

        let bounds = serde_json::json!({
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        });

        serde_wasm_bindgen::to_value(&bounds).unwrap_or(JsValue::NULL)
    }

    /// Scalar implementation of bounds computation
    fn compute_bounds_scalar(&self) -> (f32, f32, f32, f32) {
        if self.x_data.is_empty() {
            return (0.0, 1.0, 0.0, 1.0);
        }

        let mut x_min = f32::MAX;
        let mut x_max = f32::MIN;
        let mut y_min = f32::MAX;
        let mut y_max = f32::MIN;

        for (&x, &y) in self.x_data.iter().zip(self.y_data.iter()) {
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            y_min = y_min.min(y);
            y_max = y_max.max(y);
        }

        (x_min, x_max, y_min, y_max)
    }

    /// Transform points to screen coordinates
    #[wasm_bindgen]
    pub fn transform_to_screen(&self) -> Vec<f32> {
        let (x_min, x_max, y_min, y_max) = self.compute_bounds_scalar();
        let x_range = x_max - x_min;
        let y_range = y_max - y_min;

        let margin = 40.0;
        let plot_width = self.config.width as f32 - 2.0 * margin;
        let plot_height = self.config.height as f32 - 2.0 * margin;

        let mut screen_coords = Vec::with_capacity(self.x_data.len() * 2);

        for (&x, &y) in self.x_data.iter().zip(self.y_data.iter()) {
            let sx = margin + (x - x_min) / x_range * plot_width;
            let sy = margin + (1.0 - (y - y_min) / y_range) * plot_height;
            screen_coords.push(sx);
            screen_coords.push(sy);
        }

        screen_coords
    }

    /// Generate SVG representation
    #[wasm_bindgen]
    pub fn to_svg(&self) -> String {
        let screen_coords = self.transform_to_screen();
        let color = self.config.color.to_css();
        let r = self.config.point_size / 2.0;

        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
            self.config.width, self.config.height, self.config.width, self.config.height
        );

        // Background
        svg.push_str(r#"<rect width="100%" height="100%" fill="white"/>"#);

        // Points
        for chunk in screen_coords.chunks(2) {
            if chunk.len() == 2 {
                svg.push_str(&format!(
                    r#"<circle cx="{:.1}" cy="{:.1}" r="{}" fill="{}"/>"#,
                    chunk[0], chunk[1], r, color
                ));
            }
        }

        svg.push_str("</svg>");
        svg
    }
}

// ============================================================================
// Histogram Visualization
// ============================================================================

/// Histogram configuration
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct HistogramConfig {
    width: u32,
    height: u32,
    bins: usize,
    color: Color,
}

#[wasm_bindgen]
impl HistogramConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            width: 800,
            height: 600,
            bins: 20,
            color: Color::BLUE,
        }
    }

    #[wasm_bindgen]
    pub fn dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    #[wasm_bindgen]
    pub fn bins(mut self, bins: usize) -> Self {
        self.bins = bins;
        self
    }
}

impl Default for HistogramConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Histogram renderer
#[wasm_bindgen]
pub struct Histogram {
    config: HistogramConfig,
    data: Vec<f32>,
    bin_counts: Vec<u32>,
}

#[wasm_bindgen]
impl Histogram {
    #[wasm_bindgen(constructor)]
    pub fn new(config: HistogramConfig) -> Self {
        Self {
            config,
            data: Vec::new(),
            bin_counts: Vec::new(),
        }
    }

    /// Set data
    #[wasm_bindgen]
    pub fn set_data(&mut self, data: Vec<f32>) {
        self.data = data;
        self.compute_bins();
    }

    /// Compute histogram bins
    fn compute_bins(&mut self) {
        if self.data.is_empty() {
            self.bin_counts = vec![0; self.config.bins];
            return;
        }

        let min = self.data.iter().cloned().fold(f32::MAX, f32::min);
        let max = self.data.iter().cloned().fold(f32::MIN, f32::max);
        let bin_width = (max - min) / self.config.bins as f32;

        self.bin_counts = vec![0u32; self.config.bins];

        for &value in &self.data {
            let bin_idx = ((value - min) / bin_width) as usize;
            let bin_idx = bin_idx.min(self.config.bins - 1);
            self.bin_counts[bin_idx] += 1;
        }
    }

    /// Get bin counts
    #[wasm_bindgen]
    pub fn get_bin_counts(&self) -> Vec<u32> {
        self.bin_counts.clone()
    }

    /// Generate SVG representation
    #[wasm_bindgen]
    pub fn to_svg(&self) -> String {
        let margin = 40.0;
        let plot_width = self.config.width as f32 - 2.0 * margin;
        let plot_height = self.config.height as f32 - 2.0 * margin;

        let max_count = *self.bin_counts.iter().max().unwrap_or(&1) as f32;
        let bar_width = plot_width / self.config.bins as f32;
        let color = self.config.color.to_css();

        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
            self.config.width, self.config.height, self.config.width, self.config.height
        );

        svg.push_str(r#"<rect width="100%" height="100%" fill="white"/>"#);

        for (i, &count) in self.bin_counts.iter().enumerate() {
            let bar_height = (count as f32 / max_count) * plot_height;
            let x = margin + i as f32 * bar_width;
            let y = margin + plot_height - bar_height;

            svg.push_str(&format!(
                r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="{}" stroke="white" stroke-width="1"/>"#,
                x, y, bar_width, bar_height, color
            ));
        }

        svg.push_str("</svg>");
        svg
    }
}

// ============================================================================
// Performance Benchmarking
// ============================================================================

/// Benchmark result
#[wasm_bindgen]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkResult {
    operation: String,
    data_size: usize,
    time_ms: f64,
    throughput: f64,
    tier: String,
}

#[wasm_bindgen]
impl BenchmarkResult {
    #[wasm_bindgen(getter)]
    pub fn operation(&self) -> String {
        self.operation.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn data_size(&self) -> usize {
        self.data_size
    }

    #[wasm_bindgen(getter)]
    pub fn time_ms(&self) -> f64 {
        self.time_ms
    }

    #[wasm_bindgen(getter)]
    pub fn throughput(&self) -> f64 {
        self.throughput
    }

    #[wasm_bindgen]
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

/// Run a scatter plot benchmark
#[wasm_bindgen]
pub fn benchmark_scatter(point_count: usize) -> BenchmarkResult {
    let window = window().expect("window");
    let performance = window.performance().expect("performance");

    // Generate test data
    let x_data: Vec<f32> = (0..point_count).map(|i| i as f32).collect();
    let y_data: Vec<f32> = (0..point_count)
        .map(|i| (i as f32 * 0.01).sin() * 100.0)
        .collect();

    let mut scatter = ScatterPlot::new(ScatterConfig::new());
    scatter.set_data(x_data, y_data);

    // Benchmark transform
    let start = performance.now();
    let _coords = scatter.transform_to_screen();
    let end = performance.now();

    let time_ms = end - start;
    let throughput = point_count as f64 / (time_ms / 1000.0);

    BenchmarkResult {
        operation: "scatter_transform".to_string(),
        data_size: point_count,
        time_ms,
        throughput,
        tier: "Scalar".to_string(), // TODO: detect actual tier
    }
}

/// Run a histogram benchmark
#[wasm_bindgen]
pub fn benchmark_histogram(data_size: usize, bins: usize) -> BenchmarkResult {
    let window = window().expect("window");
    let performance = window.performance().expect("performance");

    // Generate test data
    let data: Vec<f32> = (0..data_size)
        .map(|i| {
            let x = i as f32 / data_size as f32;
            (x * 6.28).sin() * 50.0 + 50.0
        })
        .collect();

    let mut histogram = Histogram::new(HistogramConfig::new().bins(bins));

    let start = performance.now();
    histogram.set_data(data);
    let end = performance.now();

    let time_ms = end - start;
    let throughput = data_size as f64 / (time_ms / 1000.0);

    BenchmarkResult {
        operation: "histogram_binning".to_string(),
        data_size,
        time_ms,
        throughput,
        tier: "Scalar".to_string(),
    }
}

// ============================================================================
// ASCII Visualization (Terminal-style)
// ============================================================================

/// Generate ASCII bar chart from histogram data
#[wasm_bindgen]
pub fn histogram_to_ascii(counts: Vec<u32>, _width: usize, height: usize) -> String {
    if counts.is_empty() {
        return String::new();
    }

    let max_count = *counts.iter().max().unwrap_or(&1) as f32;
    // Unicode block elements: U+2581 to U+2588
    let bar_chars = [' ', '\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}'];

    let mut output = String::new();

    for row in (0..height).rev() {
        let threshold = (row as f32 + 0.5) / height as f32;

        for &count in &counts {
            let normalized = count as f32 / max_count;
            let char_idx = if normalized > threshold {
                let frac = (normalized - threshold) * height as f32;
                (frac * 8.0).min(7.0) as usize
            } else {
                0
            };
            output.push(bar_chars[char_idx.min(7)]);
        }
        output.push('\n');
    }

    output
}
