//! Theme system with CIELAB perceptual gradients.
//!
//! Provides color themes and smooth gradient interpolation for terminal UIs.
//! Based on trueno-viz theme system for visual consistency.

use presentar_core::Color;

/// A color gradient with 2-3 stops for smooth interpolation.
#[derive(Debug, Clone)]
pub struct Gradient {
    /// Gradient color stops (RGB hex strings like "#FF0000").
    stops: Vec<Color>,
}

impl Gradient {
    /// Create a two-color gradient.
    #[must_use]
    pub fn two(start: Color, end: Color) -> Self {
        Self {
            stops: vec![start, end],
        }
    }

    /// Create a three-color gradient.
    #[must_use]
    pub fn three(start: Color, mid: Color, end: Color) -> Self {
        Self {
            stops: vec![start, mid, end],
        }
    }

    /// Create gradient from hex strings.
    #[must_use]
    pub fn from_hex(stops: &[&str]) -> Self {
        Self {
            stops: stops.iter().map(|s| parse_hex(s)).collect(),
        }
    }

    /// Sample the gradient at position t (0.0 - 1.0).
    #[must_use]
    pub fn sample(&self, t: f64) -> Color {
        // Handle NaN/Inf by treating as 0.0
        let t = if t.is_finite() {
            t.clamp(0.0, 1.0)
        } else {
            0.0
        };

        if self.stops.is_empty() {
            return Color::WHITE;
        }

        if self.stops.len() == 1 {
            return self.stops[0];
        }

        // Find the segment
        let segment_count = self.stops.len() - 1;
        let segment_size = 1.0 / segment_count as f64;
        let segment = ((t / segment_size) as usize).min(segment_count - 1);
        let local_t = (t - segment as f64 * segment_size) / segment_size;

        let start = self.stops[segment];
        let end = self.stops[segment + 1];

        interpolate_lab(start, end, local_t)
    }

    /// Get color for a percentage value (0-100).
    #[must_use]
    pub fn for_percent(&self, percent: f64) -> Color {
        self.sample(percent / 100.0)
    }
}

impl Default for Gradient {
    fn default() -> Self {
        // Green -> Yellow -> Red (classic usage gradient)
        Self::from_hex(&["#00FF00", "#FFFF00", "#FF0000"])
    }
}

/// Theme configuration for terminal UI.
#[derive(Debug, Clone)]
pub struct Theme {
    /// Theme name.
    pub name: String,
    /// Background color.
    pub background: Color,
    /// Foreground (text) color.
    pub foreground: Color,
    /// Border color.
    pub border: Color,
    /// Dim/inactive color.
    pub dim: Color,
    /// CPU usage gradient.
    pub cpu: Gradient,
    /// Memory usage gradient.
    pub memory: Gradient,
    /// GPU usage gradient.
    pub gpu: Gradient,
    /// Temperature gradient.
    pub temperature: Gradient,
    /// Network gradient.
    pub network: Gradient,
}

impl Default for Theme {
    fn default() -> Self {
        Self::tokyo_night()
    }
}

impl Theme {
    /// Create a new default theme.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Tokyo Night theme (dark, modern).
    #[must_use]
    pub fn tokyo_night() -> Self {
        Self {
            name: "tokyo_night".to_string(),
            background: parse_hex("#1a1b26"),
            foreground: parse_hex("#c0caf5"),
            border: parse_hex("#414868"),
            dim: parse_hex("#565f89"),
            cpu: Gradient::from_hex(&["#7aa2f7", "#e0af68", "#f7768e"]),
            memory: Gradient::from_hex(&["#9ece6a", "#e0af68", "#f7768e"]),
            gpu: Gradient::from_hex(&["#bb9af7", "#7dcfff", "#f7768e"]),
            temperature: Gradient::from_hex(&["#7dcfff", "#e0af68", "#f7768e"]),
            network: Gradient::from_hex(&["#7dcfff", "#9ece6a"]),
        }
    }

    /// Dracula theme (dark, purple).
    #[must_use]
    pub fn dracula() -> Self {
        Self {
            name: "dracula".to_string(),
            background: parse_hex("#282a36"),
            foreground: parse_hex("#f8f8f2"),
            border: parse_hex("#6272a4"),
            dim: parse_hex("#44475a"),
            cpu: Gradient::from_hex(&["#50fa7b", "#f1fa8c", "#ff5555"]),
            memory: Gradient::from_hex(&["#8be9fd", "#f1fa8c", "#ff5555"]),
            gpu: Gradient::from_hex(&["#bd93f9", "#ff79c6", "#ff5555"]),
            temperature: Gradient::from_hex(&["#8be9fd", "#ffb86c", "#ff5555"]),
            network: Gradient::from_hex(&["#8be9fd", "#50fa7b"]),
        }
    }

    /// Nord theme (cool, arctic).
    #[must_use]
    pub fn nord() -> Self {
        Self {
            name: "nord".to_string(),
            background: parse_hex("#2e3440"),
            foreground: parse_hex("#eceff4"),
            border: parse_hex("#4c566a"),
            dim: parse_hex("#3b4252"),
            cpu: Gradient::from_hex(&["#a3be8c", "#ebcb8b", "#bf616a"]),
            memory: Gradient::from_hex(&["#88c0d0", "#ebcb8b", "#bf616a"]),
            gpu: Gradient::from_hex(&["#b48ead", "#81a1c1", "#bf616a"]),
            temperature: Gradient::from_hex(&["#88c0d0", "#ebcb8b", "#bf616a"]),
            network: Gradient::from_hex(&["#88c0d0", "#a3be8c"]),
        }
    }

    /// Monokai theme (classic).
    #[must_use]
    pub fn monokai() -> Self {
        Self {
            name: "monokai".to_string(),
            background: parse_hex("#272822"),
            foreground: parse_hex("#f8f8f2"),
            border: parse_hex("#49483e"),
            dim: parse_hex("#75715e"),
            cpu: Gradient::from_hex(&["#a6e22e", "#e6db74", "#f92672"]),
            memory: Gradient::from_hex(&["#66d9ef", "#e6db74", "#f92672"]),
            gpu: Gradient::from_hex(&["#ae81ff", "#fd971f", "#f92672"]),
            temperature: Gradient::from_hex(&["#66d9ef", "#fd971f", "#f92672"]),
            network: Gradient::from_hex(&["#66d9ef", "#a6e22e"]),
        }
    }

    /// Get color for CPU usage percentage.
    #[must_use]
    pub fn cpu_color(&self, percent: f64) -> Color {
        self.cpu.for_percent(percent)
    }

    /// Get color for memory usage percentage.
    #[must_use]
    pub fn memory_color(&self, percent: f64) -> Color {
        self.memory.for_percent(percent)
    }

    /// Get color for GPU usage percentage.
    #[must_use]
    pub fn gpu_color(&self, percent: f64) -> Color {
        self.gpu.for_percent(percent)
    }

    /// Get color for temperature (0-100 mapped to cold-hot).
    #[must_use]
    pub fn temp_color(&self, temp_c: f64, max_temp: f64) -> Color {
        let percent = (temp_c / max_temp * 100.0).clamp(0.0, 100.0);
        self.temperature.for_percent(percent)
    }
}

/// Parse hex color string to Color.
fn parse_hex(hex: &str) -> Color {
    let hex = hex.trim_start_matches('#');
    if hex.len() != 6 {
        return Color::WHITE;
    }

    let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(255);
    let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(255);
    let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(255);

    // Color uses 0.0-1.0 range
    Color::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0)
}

/// Interpolate between two colors using CIELAB for perceptual uniformity.
#[allow(clippy::many_single_char_names)]
fn interpolate_lab(start: Color, end: Color, t: f64) -> Color {
    // Convert to LAB
    let lab1 = rgb_to_lab(start);
    let lab2 = rgb_to_lab(end);

    // Interpolate in LAB space
    let l = lab1.0 + t * (lab2.0 - lab1.0);
    let a = lab1.1 + t * (lab2.1 - lab1.1);
    let b = lab1.2 + t * (lab2.2 - lab1.2);

    // Convert back to RGB
    lab_to_rgb(l, a, b)
}

/// Convert RGB to CIELAB.
#[allow(clippy::many_single_char_names, clippy::unreadable_literal)]
fn rgb_to_lab(c: Color) -> (f64, f64, f64) {
    // Color is already in 0-1 range
    let r = c.r as f64;
    let g = c.g as f64;
    let b = c.b as f64;

    // sRGB to linear
    let r = if r > 0.04045 {
        ((r + 0.055) / 1.055).powf(2.4)
    } else {
        r / 12.92
    };
    let g = if g > 0.04045 {
        ((g + 0.055) / 1.055).powf(2.4)
    } else {
        g / 12.92
    };
    let b = if b > 0.04045 {
        ((b + 0.055) / 1.055).powf(2.4)
    } else {
        b / 12.92
    };

    // Linear RGB to XYZ
    let x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    let y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    let z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;

    // XYZ to LAB (D65 white point)
    let x = x / 0.95047;
    let z = z / 1.08883;

    let fx = if x > 0.008856 {
        x.cbrt()
    } else {
        (7.787 * x) + (16.0 / 116.0)
    };
    let fy = if y > 0.008856 {
        y.cbrt()
    } else {
        (7.787 * y) + (16.0 / 116.0)
    };
    let fz = if z > 0.008856 {
        z.cbrt()
    } else {
        (7.787 * z) + (16.0 / 116.0)
    };

    let l = (116.0 * fy) - 16.0;
    let a = 500.0 * (fx - fy);
    let b_val = 200.0 * (fy - fz);

    (l, a, b_val)
}

/// Convert CIELAB to RGB.
#[allow(clippy::many_single_char_names, clippy::unreadable_literal)]
fn lab_to_rgb(l: f64, a: f64, b: f64) -> Color {
    // LAB to XYZ
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;

    let x = if fx.powi(3) > 0.008856 {
        fx.powi(3)
    } else {
        (fx - 16.0 / 116.0) / 7.787
    };
    let y = if l > 7.9996 { fy.powi(3) } else { l / 903.3 };
    let z = if fz.powi(3) > 0.008856 {
        fz.powi(3)
    } else {
        (fz - 16.0 / 116.0) / 7.787
    };

    // D65 white point
    let x = x * 0.95047;
    let z = z * 1.08883;

    // XYZ to linear RGB
    let r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314;
    let g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560;
    let b_val = x * 0.0556434 + y * -0.2040259 + z * 1.0572252;

    // Linear to sRGB
    let r = if r > 0.0031308 {
        1.055 * r.powf(1.0 / 2.4) - 0.055
    } else {
        12.92 * r
    };
    let g = if g > 0.0031308 {
        1.055 * g.powf(1.0 / 2.4) - 0.055
    } else {
        12.92 * g
    };
    let b_val = if b_val > 0.0031308 {
        1.055 * b_val.powf(1.0 / 2.4) - 0.055
    } else {
        12.92 * b_val
    };

    // Clamp to 0-1 range
    let r = r.clamp(0.0, 1.0) as f32;
    let g = g.clamp(0.0, 1.0) as f32;
    let b_val = b_val.clamp(0.0, 1.0) as f32;

    Color::new(r, g, b_val, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hex() {
        let c = parse_hex("#FF0000");
        assert!((c.r - 1.0).abs() < 0.01); // 0-1 range
        assert!((c.g - 0.0).abs() < 0.01);
        assert!((c.b - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_hex_green() {
        let c = parse_hex("#00FF00");
        assert!((c.r - 0.0).abs() < 0.01);
        assert!((c.g - 1.0).abs() < 0.01); // 0-1 range
        assert!((c.b - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_gradient_two() {
        let g = Gradient::two(Color::RED, Color::BLUE);
        let start = g.sample(0.0);
        let end = g.sample(1.0);
        assert!((start.r - 1.0).abs() < 0.01); // 0-1 range
        assert!((end.b - 1.0).abs() < 0.01); // 0-1 range
    }

    #[test]
    fn test_gradient_three() {
        let g = Gradient::three(Color::RED, Color::GREEN, Color::BLUE);
        let mid = g.sample(0.5);
        // At 0.5, should be close to green
        assert!(mid.g > mid.r);
        assert!(mid.g > mid.b);
    }

    #[test]
    fn test_gradient_from_hex() {
        let g = Gradient::from_hex(&["#FF0000", "#00FF00"]);
        let start = g.sample(0.0);
        assert!((start.r - 1.0).abs() < 0.01); // 0-1 range
    }

    #[test]
    fn test_gradient_for_percent() {
        let g = Gradient::default();
        let _ = g.for_percent(50.0);
        let _ = g.for_percent(0.0);
        let _ = g.for_percent(100.0);
    }

    #[test]
    fn test_theme_tokyo_night() {
        let t = Theme::tokyo_night();
        assert_eq!(t.name, "tokyo_night");
    }

    #[test]
    fn test_theme_dracula() {
        let t = Theme::dracula();
        assert_eq!(t.name, "dracula");
    }

    #[test]
    fn test_theme_nord() {
        let t = Theme::nord();
        assert_eq!(t.name, "nord");
    }

    #[test]
    fn test_theme_monokai() {
        let t = Theme::monokai();
        assert_eq!(t.name, "monokai");
    }

    #[test]
    fn test_theme_cpu_color() {
        let t = Theme::default();
        let _ = t.cpu_color(0.0);
        let _ = t.cpu_color(50.0);
        let _ = t.cpu_color(100.0);
    }

    #[test]
    fn test_theme_memory_color() {
        let t = Theme::default();
        let _ = t.memory_color(75.0);
    }

    #[test]
    fn test_theme_gpu_color() {
        let t = Theme::default();
        let _ = t.gpu_color(25.0);
    }

    #[test]
    fn test_theme_temp_color() {
        let t = Theme::default();
        let _ = t.temp_color(65.0, 100.0);
    }

    #[test]
    fn test_gradient_empty() {
        let g = Gradient { stops: vec![] };
        let c = g.sample(0.5);
        assert_eq!(c, Color::WHITE);
    }

    #[test]
    fn test_gradient_single() {
        let g = Gradient {
            stops: vec![Color::RED],
        };
        let c = g.sample(0.5);
        assert!((c.r - 1.0).abs() < 0.01); // 0-1 range
    }

    #[test]
    fn test_gradient_clamp() {
        let g = Gradient::default();
        let _ = g.sample(-1.0); // Should clamp to 0
        let _ = g.sample(2.0); // Should clamp to 1
    }

    #[test]
    fn test_lab_roundtrip() {
        // Use 0-1 range values
        let original = Color::new(0.5, 0.25, 0.75, 1.0);
        let lab = rgb_to_lab(original);
        let back = lab_to_rgb(lab.0, lab.1, lab.2);
        assert!((original.r - back.r).abs() < 0.02);
        assert!((original.g - back.g).abs() < 0.02);
        assert!((original.b - back.b).abs() < 0.02);
    }

    #[test]
    fn test_interpolate_lab_endpoints() {
        let start = Color::RED;
        let end = Color::BLUE;

        let at_start = interpolate_lab(start, end, 0.0);
        let at_end = interpolate_lab(start, end, 1.0);

        assert!((at_start.r - 1.0).abs() < 0.02); // 0-1 range
        assert!((at_end.b - 1.0).abs() < 0.02); // 0-1 range
    }

    #[test]
    fn test_theme_default() {
        let t = Theme::default();
        assert_eq!(t.name, "tokyo_night");
    }

    #[test]
    fn test_parse_hex_invalid() {
        let c = parse_hex("invalid");
        assert_eq!(c, Color::WHITE);
    }

    #[test]
    fn test_parse_hex_no_hash() {
        let c = parse_hex("FF0000");
        assert!((c.r - 1.0).abs() < 0.01); // 0-1 range
    }

    #[test]
    fn test_parse_hex_blue() {
        let c = parse_hex("#0000FF");
        assert!((c.r - 0.0).abs() < 0.01);
        assert!((c.g - 0.0).abs() < 0.01);
        assert!((c.b - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_hex_mixed() {
        let c = parse_hex("#808080");
        assert!((c.r - 0.5).abs() < 0.01);
        assert!((c.g - 0.5).abs() < 0.01);
        assert!((c.b - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_gradient_default() {
        let g = Gradient::default();
        // Green -> Yellow -> Red
        let start = g.sample(0.0);
        assert!(start.g > 0.9); // Green
        let end = g.sample(1.0);
        assert!(end.r > 0.9); // Red
    }

    #[test]
    fn test_theme_new() {
        let t = Theme::new();
        assert_eq!(t.name, "tokyo_night");
    }

    #[test]
    fn test_theme_colors_non_panic() {
        let t = Theme::default();
        // Test all color functions don't panic
        for pct in [0.0, 25.0, 50.0, 75.0, 100.0] {
            let _ = t.cpu_color(pct);
            let _ = t.memory_color(pct);
            let _ = t.gpu_color(pct);
        }
        for temp in [0.0, 25.0, 50.0, 75.0, 100.0] {
            let _ = t.temp_color(temp, 100.0);
        }
    }

    #[test]
    fn test_temp_color_cold() {
        let t = Theme::default();
        let cold = t.temp_color(0.0, 100.0);
        let hot = t.temp_color(100.0, 100.0);
        // Cold and hot should be different
        assert!((cold.r - hot.r).abs() > 0.1 || (cold.g - hot.g).abs() > 0.1);
    }

    #[test]
    fn test_temp_color_clamped() {
        let t = Theme::default();
        // Over max should clamp to 100%
        let over = t.temp_color(150.0, 100.0);
        let at_max = t.temp_color(100.0, 100.0);
        assert!((over.r - at_max.r).abs() < 0.01);
    }

    #[test]
    fn test_gradient_four_stops() {
        // Test gradient with more stops
        let g = Gradient {
            stops: vec![Color::RED, Color::GREEN, Color::BLUE, Color::WHITE],
        };
        let _ = g.sample(0.0);
        let _ = g.sample(0.33);
        let _ = g.sample(0.66);
        let _ = g.sample(1.0);
    }

    #[test]
    fn test_theme_background_foreground() {
        let t = Theme::tokyo_night();
        // Background should be dark
        assert!(t.background.r < 0.2);
        assert!(t.background.g < 0.2);
        // Foreground should be light
        assert!(t.foreground.r > 0.5);
        assert!(t.foreground.g > 0.5);
    }

    #[test]
    fn test_theme_border_dim() {
        let t = Theme::dracula();
        // Border and dim should be mid-range
        assert!(t.border.r > 0.2 && t.border.r < 0.8);
        assert!(t.dim.r > 0.1 && t.dim.r < 0.5);
    }

    #[test]
    fn test_interpolate_lab_midpoint() {
        let start = Color::new(0.0, 0.0, 0.0, 1.0); // Black
        let end = Color::new(1.0, 1.0, 1.0, 1.0); // White
        let mid = interpolate_lab(start, end, 0.5);
        // Middle gray should be around 0.5
        assert!(mid.r > 0.3 && mid.r < 0.7);
    }

    #[test]
    fn test_rgb_lab_roundtrip_black() {
        let black = Color::new(0.0, 0.0, 0.0, 1.0);
        let lab = rgb_to_lab(black);
        let back = lab_to_rgb(lab.0, lab.1, lab.2);
        assert!((back.r - 0.0).abs() < 0.02);
    }

    #[test]
    fn test_rgb_lab_roundtrip_white() {
        let white = Color::new(1.0, 1.0, 1.0, 1.0);
        let lab = rgb_to_lab(white);
        let back = lab_to_rgb(lab.0, lab.1, lab.2);
        assert!((back.r - 1.0).abs() < 0.02);
    }

    #[test]
    fn test_gradient_segment_boundary() {
        let g = Gradient::three(Color::RED, Color::GREEN, Color::BLUE);
        // At exactly 0.5, should be at/near green
        let at_half = g.sample(0.5);
        assert!(at_half.g > at_half.r);
        assert!(at_half.g > at_half.b);
    }

    #[test]
    fn test_all_themes_valid() {
        let themes = [
            Theme::tokyo_night(),
            Theme::dracula(),
            Theme::nord(),
            Theme::monokai(),
        ];
        for t in themes {
            assert!(!t.name.is_empty());
            // All gradients should work
            let _ = t.cpu.sample(0.5);
            let _ = t.memory.sample(0.5);
            let _ = t.gpu.sample(0.5);
            let _ = t.temperature.sample(0.5);
            let _ = t.network.sample(0.5);
        }
    }

    #[test]
    fn test_parse_hex_short() {
        // Too short should return white
        let c = parse_hex("#FFF");
        assert_eq!(c, Color::WHITE);
    }

    #[test]
    fn test_parse_hex_long() {
        // Too long should return white
        let c = parse_hex("#FFFFFFFF");
        assert_eq!(c, Color::WHITE);
    }
}
