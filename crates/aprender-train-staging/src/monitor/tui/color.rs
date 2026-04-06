//! Terminal Color Support (ENT-122)
//!
//! Provides ANSI color output with automatic terminal capability detection.
//! Based on presentar's color system with semantic colors for training metrics.

use super::state::{LossTrend, TrainingStatus};
use std::fmt;

/// Safely convert an f32 to u8 with bounds clamping.
/// Clamps to [0.0, 255.0] then converts through u16 with `try_from` for safety.
#[inline]
fn clamped_f32_to_u8(value: f32) -> u8 {
    let clamped = value.clamp(0.0, 255.0);
    // After clamp, value is in [0.0, 255.0]. Cast to u16 is safe for this range.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let wide = clamped as u16;
    // u16 in [0, 255] always fits in u8; unwrap_or provides defense-in-depth
    u8::try_from(wide).unwrap_or(u8::MAX)
}

/// Safely convert a non-negative f32 to usize with bounds clamping.
#[inline]
fn clamped_f32_to_usize(value: f32) -> usize {
    let clamped = value.max(0.0);
    // Value is non-negative after max(0.0); cast is safe for practical display sizes
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let result = clamped as usize;
    result
}

/// Terminal color capability mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorMode {
    /// True color (24-bit RGB)
    TrueColor,
    /// 256 color palette
    Color256,
    /// 16 color palette
    Color16,
    /// No color (monochrome)
    #[default]
    Mono,
}

impl ColorMode {
    /// Detect terminal color capability from environment
    pub fn detect() -> Self {
        Self::detect_with_env(
            std::env::var("COLORTERM").ok().as_deref(),
            std::env::var("TERM").ok().as_deref(),
            std::env::var("NO_COLOR").ok().as_deref(),
        )
    }

    /// Detect with explicit environment values (for testing)
    pub fn detect_with_env(
        colorterm: Option<&str>,
        term: Option<&str>,
        no_color: Option<&str>,
    ) -> Self {
        // NO_COLOR takes precedence
        if no_color.is_some() {
            return Self::Mono;
        }

        // Check COLORTERM for truecolor support
        if let Some(ct) = colorterm {
            if ct.contains("truecolor") || ct.contains("24bit") {
                return Self::TrueColor;
            }
        }

        // Check TERM for capability hints
        if let Some(term) = term {
            if term.contains("256color") || term.contains("kitty") || term.contains("alacritty") {
                return Self::Color256;
            }
            if term.contains("xterm") || term.contains("screen") || term.contains("tmux") {
                return Self::Color16;
            }
            if term == "dumb" || term.is_empty() {
                return Self::Mono;
            }
        }

        // Default to 16 colors for unknown terminals
        Self::Color16
    }
}

/// RGB color
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }
}

impl From<(u8, u8, u8)> for Rgb {
    fn from((r, g, b): (u8, u8, u8)) -> Self {
        Self { r, g, b }
    }
}

impl Rgb {
    /// Convert to ANSI 256-color index (approximate)
    pub fn to_256(self) -> u8 {
        // Use the 216 color cube (indices 16-231)
        // Each channel has 6 levels: 0, 95, 135, 175, 215, 255
        let r6 = u8::try_from(u16::from(self.r) * 5 / 255).unwrap_or(u8::MAX);
        let g6 = u8::try_from(u16::from(self.g) * 5 / 255).unwrap_or(u8::MAX);
        let b6 = u8::try_from(u16::from(self.b) * 5 / 255).unwrap_or(u8::MAX);
        16 + 36 * r6 + 6 * g6 + b6
    }

    /// Convert to ANSI 16-color index (approximate)
    pub fn to_16(self) -> u8 {
        // Use max channel for brightness detection (saturated colors should be bright)
        let max_channel = self.r.max(self.g).max(self.b);
        let is_bright = max_channel > 180;

        // Determine dominant color
        let r_dom = self.r >= self.g && self.r >= self.b;
        let g_dom = self.g >= self.r && self.g >= self.b;
        let b_dom = self.b >= self.r && self.b >= self.g;

        // Mix detection
        let r_present = self.r > 85;
        let g_present = self.g > 85;
        let b_present = self.b > 85;

        let base = match (r_present, g_present, b_present) {
            (true, true, true) => 7,   // white
            (true, true, false) => 3,  // yellow
            (true, false, true) => 5,  // magenta
            (false, true, true) => 6,  // cyan
            (true, false, false) => 1, // red
            (false, true, false) => 2, // green
            (false, false, true) => 4, // blue
            (false, false, false) => {
                // Near black - check if any color is dominant
                if r_dom && self.r > 40 {
                    1
                } else if g_dom && self.g > 40 {
                    2
                } else if b_dom && self.b > 40 {
                    4
                } else {
                    0
                }
            }
        };

        if is_bright {
            base + 8
        } else {
            base
        }
    }
}

/// Styled text with foreground color
pub struct Styled<'a> {
    text: &'a str,
    fg: Option<Rgb>,
    bold: bool,
    mode: ColorMode,
}

impl<'a> Styled<'a> {
    pub fn new(text: &'a str, mode: ColorMode) -> Self {
        Self { text, fg: None, bold: false, mode }
    }

    pub fn fg(mut self, color: impl Into<Rgb>) -> Self {
        self.fg = Some(color.into());
        self
    }

    pub fn bold(mut self) -> Self {
        self.bold = true;
        self
    }
}

impl fmt::Display for Styled<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.mode == ColorMode::Mono {
            return write!(f, "{}", self.text);
        }

        let mut has_style = false;

        // Bold
        if self.bold {
            write!(f, "\x1b[1m")?;
            has_style = true;
        }

        // Foreground color
        if let Some(rgb) = self.fg {
            match self.mode {
                ColorMode::TrueColor => {
                    write!(f, "\x1b[38;2;{};{};{}m", rgb.r, rgb.g, rgb.b)?;
                }
                ColorMode::Color256 => {
                    write!(f, "\x1b[38;5;{}m", rgb.to_256())?;
                }
                ColorMode::Color16 => {
                    let code = rgb.to_16();
                    if code >= 8 {
                        write!(f, "\x1b[9{}m", code - 8)?;
                    } else {
                        write!(f, "\x1b[3{code}m")?;
                    }
                }
                ColorMode::Mono => {}
            }
            has_style = true;
        }

        write!(f, "{}", self.text)?;

        if has_style {
            write!(f, "\x1b[0m")?;
        }

        Ok(())
    }
}

/// Semantic color palette for training metrics
#[derive(Debug, Clone)]
pub struct TrainingPalette {
    pub mode: ColorMode,
}

impl Default for TrainingPalette {
    fn default() -> Self {
        Self { mode: ColorMode::detect() }
    }
}

impl TrainingPalette {
    pub fn new(mode: ColorMode) -> Self {
        Self { mode }
    }

    /// Style text with this palette's color mode
    pub fn style<'a>(&self, text: &'a str) -> Styled<'a> {
        Styled::new(text, self.mode)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Semantic Colors
    // ─────────────────────────────────────────────────────────────────────────

    /// Success/good state (green)
    pub const SUCCESS: Rgb = Rgb::new(80, 200, 120);

    /// Warning state (yellow/orange)
    pub const WARNING: Rgb = Rgb::new(255, 193, 7);

    /// Error/danger state (red)
    pub const ERROR: Rgb = Rgb::new(244, 67, 54);

    /// Info/neutral (blue)
    pub const INFO: Rgb = Rgb::new(33, 150, 243);

    /// Muted/secondary text (gray)
    pub const MUTED: Rgb = Rgb::new(158, 158, 158);

    /// Primary accent (cyan)
    pub const PRIMARY: Rgb = Rgb::new(0, 188, 212);

    // ─────────────────────────────────────────────────────────────────────────
    // Threshold Lookup
    // ─────────────────────────────────────────────────────────────────────────

    /// Return the color for the first threshold in `thresholds` that `value` does not exceed,
    /// or `fallback` if `value` exceeds all thresholds.
    ///
    /// `thresholds` is a slice of `(upper_bound_exclusive, color)` pairs in ascending order.
    fn threshold_color(value: f32, thresholds: &[(f32, Rgb)], fallback: Rgb) -> Rgb {
        for &(bound, color) in thresholds {
            if value <= bound {
                return color;
            }
        }
        fallback
    }

    // ─────────────────────────────────────────────────────────────────────────
    // GPU Metrics Colors
    // ─────────────────────────────────────────────────────────────────────────

    /// Color for GPU utilization based on percentage
    pub fn gpu_util_color(percent: f32) -> Rgb {
        let p = percent.clamp(0.0, 100.0);
        Self::threshold_color(
            p,
            &[(30.0, Self::MUTED), (70.0, Self::SUCCESS), (90.0, Self::INFO)],
            Self::PRIMARY,
        )
    }

    /// Color for VRAM usage based on percentage
    pub fn vram_color(percent: f32) -> Rgb {
        let p = percent.clamp(0.0, 100.0);
        Self::threshold_color(
            p,
            &[(50.0, Self::SUCCESS), (75.0, Self::INFO), (90.0, Self::WARNING)],
            Self::ERROR,
        )
    }

    /// Color for temperature in Celsius
    pub fn temp_color(celsius: f32) -> Rgb {
        let t = celsius.clamp(0.0, 200.0);
        Self::threshold_color(
            t,
            &[(50.0, Self::SUCCESS), (70.0, Self::INFO), (80.0, Self::WARNING)],
            Self::ERROR,
        )
    }

    /// Color for power usage based on percentage of limit
    pub fn power_color(percent: f32) -> Rgb {
        let p = percent.clamp(0.0, 100.0);
        Self::threshold_color(
            p,
            &[(60.0, Self::SUCCESS), (80.0, Self::INFO), (95.0, Self::WARNING)],
            Self::ERROR,
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Training Metrics Colors
    // ─────────────────────────────────────────────────────────────────────────

    /// Color for gradient norm (explosion warning)
    pub fn grad_norm_color(norm: f32) -> Rgb {
        Self::threshold_color(
            norm,
            &[(1.0, Self::SUCCESS), (5.0, Self::INFO), (10.0, Self::WARNING)],
            Self::ERROR,
        )
    }

    /// Color for loss value (lower is better)
    /// Returns a gradient from red (high loss) to green (low loss)
    pub fn loss_color(loss: f32, min_loss: f32, max_loss: f32) -> Rgb {
        if max_loss <= min_loss {
            return Self::INFO;
        }

        let normalized = ((loss - min_loss) / (max_loss - min_loss)).clamp(0.0, 1.0);

        // Gradient from green (0.0) to yellow (0.5) to red (1.0)
        let (r, g, b) = if normalized < 0.5 {
            // Green to yellow
            let t = normalized * 2.0;
            (
                clamped_f32_to_u8(80.0 + t * 175.0),
                clamped_f32_to_u8(200.0 - t * 7.0),
                clamped_f32_to_u8(120.0 - t * 113.0),
            )
        } else {
            // Yellow to red
            let t = (normalized - 0.5) * 2.0;
            (
                clamped_f32_to_u8(255.0 - t * 11.0),
                clamped_f32_to_u8(193.0 - t * 126.0),
                clamped_f32_to_u8(7.0 + t * 47.0),
            )
        };

        Rgb::new(r, g, b)
    }

    /// Color for training status
    pub fn status_color(status: &TrainingStatus) -> Rgb {
        match status {
            TrainingStatus::Running => Self::SUCCESS,
            TrainingStatus::Completed => Self::PRIMARY,
            TrainingStatus::Paused => Self::WARNING,
            TrainingStatus::Failed(_) => Self::ERROR,
            TrainingStatus::Initializing => Self::INFO,
        }
    }

    /// Color for loss trend indicator
    pub fn loss_trend_color(trend: &LossTrend) -> Rgb {
        match trend {
            LossTrend::Decreasing => Self::SUCCESS, // Good - loss is going down
            LossTrend::Stable => Self::INFO,        // Neutral - plateauing
            LossTrend::Increasing => Self::ERROR,   // Bad - loss is going up
            LossTrend::Unknown => Self::MUTED,      // Not enough data
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Progress Bar Colors
    // ─────────────────────────────────────────────────────────────────────────

    /// Color for progress bar fill based on completion percentage
    pub fn progress_color(percent: f32) -> Rgb {
        let p = percent.clamp(0.0, 100.0);
        if p <= 75.0 {
            Self::INFO // In progress (blue)
        } else if p < 100.0 {
            Self::SUCCESS // Almost done (green)
        } else {
            Self::PRIMARY // Complete (cyan)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Colored Progress Bar
// ─────────────────────────────────────────────────────────────────────────────

/// Render a colored progress bar
pub fn colored_bar(value: f32, max: f32, width: usize, color: Rgb, mode: ColorMode) -> String {
    let percent = if max > 0.0 { value / max } else { 0.0 };
    let percent = percent.clamp(0.0, 1.0);
    // Safe: width is a display column count, clamped to u16 range for lossless f32 conversion
    let width_clamped = u16::try_from(width).unwrap_or(u16::MAX);
    let filled_f32 = f32::from(width_clamped) * percent;
    let filled = clamped_f32_to_usize(filled_f32.clamp(0.0, f32::from(width_clamped))).min(width);
    let empty = width.saturating_sub(filled);

    let filled_str: String = std::iter::repeat_n('█', filled).collect();
    let empty_str: String = std::iter::repeat_n('░', empty).collect();

    if mode == ColorMode::Mono {
        format!("{filled_str}{empty_str}")
    } else {
        format!(
            "{}{}",
            Styled::new(&filled_str, mode).fg(color),
            Styled::new(&empty_str, mode).fg(TrainingPalette::MUTED)
        )
    }
}

/// Render a colored value with semantic coloring
pub fn colored_value<T: fmt::Display>(value: T, color: Rgb, mode: ColorMode) -> String {
    let text = value.to_string();
    Styled::new(&text, mode).fg(color).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_mode_detection() {
        // NO_COLOR takes precedence
        assert_eq!(
            ColorMode::detect_with_env(Some("truecolor"), Some("xterm-256color"), Some("1")),
            ColorMode::Mono
        );

        // COLORTERM truecolor
        assert_eq!(ColorMode::detect_with_env(Some("truecolor"), None, None), ColorMode::TrueColor);

        // TERM 256color
        assert_eq!(
            ColorMode::detect_with_env(None, Some("xterm-256color"), None),
            ColorMode::Color256
        );

        // TERM xterm
        assert_eq!(ColorMode::detect_with_env(None, Some("xterm"), None), ColorMode::Color16);

        // TERM dumb
        assert_eq!(ColorMode::detect_with_env(None, Some("dumb"), None), ColorMode::Mono);
    }

    #[test]
    fn test_rgb_to_256() {
        // Black
        assert_eq!(Rgb::new(0, 0, 0).to_256(), 16);
        // White
        assert_eq!(Rgb::new(255, 255, 255).to_256(), 231);
        // Red
        assert_eq!(Rgb::new(255, 0, 0).to_256(), 196);
        // Green
        assert_eq!(Rgb::new(0, 255, 0).to_256(), 46);
        // Blue
        assert_eq!(Rgb::new(0, 0, 255).to_256(), 21);
    }

    #[test]
    fn test_rgb_to_16() {
        // Bright red
        assert_eq!(Rgb::new(255, 50, 50).to_16(), 9); // bright red
                                                      // Bright green
        assert_eq!(Rgb::new(50, 255, 50).to_16(), 10); // bright green
                                                       // Dark blue
        assert_eq!(Rgb::new(0, 0, 100).to_16(), 4); // blue
    }

    #[test]
    fn test_rgb_to_16_all_boolean_combos() {
        // (r>85, g>85, b>85) = (true, true, true) → white (7), bright → 15
        assert_eq!(Rgb::new(200, 200, 200).to_16(), 15); // bright white

        // (true, true, false) → yellow (3), bright → 11
        assert_eq!(Rgb::new(200, 200, 50).to_16(), 11); // bright yellow

        // (true, false, true) → magenta (5), bright → 13
        assert_eq!(Rgb::new(200, 50, 200).to_16(), 13); // bright magenta

        // (false, true, true) → cyan (6), bright → 14
        assert_eq!(Rgb::new(50, 200, 200).to_16(), 14); // bright cyan

        // (true, false, false) → red (1), not bright → 1
        assert_eq!(Rgb::new(100, 50, 50).to_16(), 1); // dark red

        // (false, true, false) → green (2), not bright → 2
        assert_eq!(Rgb::new(50, 100, 50).to_16(), 2); // dark green

        // (false, false, true) → blue (4), not bright → 4
        assert_eq!(Rgb::new(50, 50, 100).to_16(), 4); // dark blue

        // (false, false, false) with dominant channels
        assert_eq!(Rgb::new(60, 20, 20).to_16(), 1); // near-black, r dominant
        assert_eq!(Rgb::new(20, 60, 20).to_16(), 2); // near-black, g dominant
        assert_eq!(Rgb::new(20, 20, 60).to_16(), 4); // near-black, b dominant
        assert_eq!(Rgb::new(20, 20, 20).to_16(), 0); // true black
    }

    #[test]
    fn test_styled_display_truecolor() {
        let styled = Styled::new("test", ColorMode::TrueColor).fg(Rgb::new(255, 0, 0));
        let output = styled.to_string();
        assert!(output.contains("\x1b[38;2;255;0;0m"));
        assert!(output.contains("test"));
        assert!(output.ends_with("\x1b[0m"));
    }

    #[test]
    fn test_styled_display_mono() {
        let styled = Styled::new("test", ColorMode::Mono).fg(Rgb::new(255, 0, 0));
        let output = styled.to_string();
        assert_eq!(output, "test");
    }

    #[test]
    fn test_gpu_util_color() {
        assert_eq!(TrainingPalette::gpu_util_color(20.0), TrainingPalette::MUTED);
        assert_eq!(TrainingPalette::gpu_util_color(50.0), TrainingPalette::SUCCESS);
        assert_eq!(TrainingPalette::gpu_util_color(80.0), TrainingPalette::INFO);
        assert_eq!(TrainingPalette::gpu_util_color(95.0), TrainingPalette::PRIMARY);
    }

    #[test]
    fn test_temp_color() {
        assert_eq!(TrainingPalette::temp_color(40.0), TrainingPalette::SUCCESS);
        assert_eq!(TrainingPalette::temp_color(65.0), TrainingPalette::INFO);
        assert_eq!(TrainingPalette::temp_color(75.0), TrainingPalette::WARNING);
        assert_eq!(TrainingPalette::temp_color(85.0), TrainingPalette::ERROR);
    }

    #[test]
    fn test_grad_norm_color() {
        assert_eq!(TrainingPalette::grad_norm_color(0.5), TrainingPalette::SUCCESS);
        assert_eq!(TrainingPalette::grad_norm_color(3.0), TrainingPalette::INFO);
        assert_eq!(TrainingPalette::grad_norm_color(8.0), TrainingPalette::WARNING);
        assert_eq!(TrainingPalette::grad_norm_color(20.0), TrainingPalette::ERROR);
    }

    #[test]
    fn test_loss_color_gradient() {
        let min = 0.0;
        let max = 1.0;

        // Low loss should be greenish
        let low = TrainingPalette::loss_color(0.1, min, max);
        assert!(low.g > low.r); // More green than red

        // High loss should be reddish
        let high = TrainingPalette::loss_color(0.9, min, max);
        assert!(high.r > high.g); // More red than green
    }

    #[test]
    fn test_status_color_all_variants() {
        // Exercise every match arm in TrainingPalette::status_color
        let running = TrainingPalette::status_color(&TrainingStatus::Running);
        assert_eq!(running, TrainingPalette::SUCCESS);

        let completed = TrainingPalette::status_color(&TrainingStatus::Completed);
        assert_eq!(completed, TrainingPalette::PRIMARY);

        let paused = TrainingPalette::status_color(&TrainingStatus::Paused);
        assert_eq!(paused, TrainingPalette::WARNING);

        let failed = TrainingPalette::status_color(&TrainingStatus::Failed("error".to_string()));
        assert_eq!(failed, TrainingPalette::ERROR);

        let initializing = TrainingPalette::status_color(&TrainingStatus::Initializing);
        assert_eq!(initializing, TrainingPalette::INFO);

        // Verify exhaustive match on all TrainingStatus variants
        for status in &[
            TrainingStatus::Running,
            TrainingStatus::Completed,
            TrainingStatus::Paused,
            TrainingStatus::Failed("test".to_string()),
            TrainingStatus::Initializing,
        ] {
            match status {
                TrainingStatus::Running => {
                    assert_eq!(TrainingPalette::status_color(status), TrainingPalette::SUCCESS);
                }
                TrainingStatus::Completed => {
                    assert_eq!(TrainingPalette::status_color(status), TrainingPalette::PRIMARY);
                }
                TrainingStatus::Paused => {
                    assert_eq!(TrainingPalette::status_color(status), TrainingPalette::WARNING);
                }
                TrainingStatus::Failed(_) => {
                    assert_eq!(TrainingPalette::status_color(status), TrainingPalette::ERROR);
                }
                TrainingStatus::Initializing => {
                    assert_eq!(TrainingPalette::status_color(status), TrainingPalette::INFO);
                }
            }
        }
    }

    #[test]
    fn test_colored_bar() {
        let bar = colored_bar(50.0, 100.0, 10, TrainingPalette::SUCCESS, ColorMode::Mono);
        assert!(bar.contains('█'));
        assert!(bar.contains('░'));
        assert_eq!(bar.chars().filter(|&c| c == '█').count(), 5);
        assert_eq!(bar.chars().filter(|&c| c == '░').count(), 5);
    }

    // ── Additional coverage tests ──

    #[test]
    fn test_clamped_f32_to_u8_boundaries() {
        assert_eq!(clamped_f32_to_u8(0.0), 0);
        assert_eq!(clamped_f32_to_u8(255.0), 255);
        assert_eq!(clamped_f32_to_u8(-10.0), 0);
        assert_eq!(clamped_f32_to_u8(300.0), 255);
        assert_eq!(clamped_f32_to_u8(127.5), 127);
    }

    #[test]
    fn test_clamped_f32_to_usize_boundaries() {
        assert_eq!(clamped_f32_to_usize(0.0), 0);
        assert_eq!(clamped_f32_to_usize(-5.0), 0);
        assert_eq!(clamped_f32_to_usize(10.0), 10);
        assert_eq!(clamped_f32_to_usize(100.5), 100);
    }

    #[test]
    fn test_color_mode_default() {
        assert_eq!(ColorMode::default(), ColorMode::Mono);
    }

    #[test]
    fn test_color_mode_detect_24bit() {
        assert_eq!(ColorMode::detect_with_env(Some("24bit"), None, None), ColorMode::TrueColor);
    }

    #[test]
    fn test_color_mode_detect_kitty() {
        assert_eq!(ColorMode::detect_with_env(None, Some("kitty"), None), ColorMode::Color256);
    }

    #[test]
    fn test_color_mode_detect_alacritty() {
        assert_eq!(ColorMode::detect_with_env(None, Some("alacritty"), None), ColorMode::Color256);
    }

    #[test]
    fn test_color_mode_detect_screen() {
        assert_eq!(ColorMode::detect_with_env(None, Some("screen"), None), ColorMode::Color16);
    }

    #[test]
    fn test_color_mode_detect_tmux() {
        assert_eq!(ColorMode::detect_with_env(None, Some("tmux"), None), ColorMode::Color16);
    }

    #[test]
    fn test_color_mode_detect_empty_term() {
        assert_eq!(ColorMode::detect_with_env(None, Some(""), None), ColorMode::Mono);
    }

    #[test]
    fn test_color_mode_detect_unknown_term() {
        assert_eq!(
            ColorMode::detect_with_env(None, Some("something-unknown"), None),
            ColorMode::Color16
        );
    }

    #[test]
    fn test_color_mode_detect_no_env() {
        assert_eq!(ColorMode::detect_with_env(None, None, None), ColorMode::Color16);
    }

    #[test]
    fn test_rgb_new() {
        let c = Rgb::new(10, 20, 30);
        assert_eq!(c.r, 10);
        assert_eq!(c.g, 20);
        assert_eq!(c.b, 30);
    }

    #[test]
    fn test_rgb_from_tuple() {
        let c: Rgb = (100, 200, 50).into();
        assert_eq!(c.r, 100);
        assert_eq!(c.g, 200);
        assert_eq!(c.b, 50);
    }

    #[test]
    fn test_rgb_to_256_midrange() {
        let c = Rgb::new(128, 128, 128);
        let idx = c.to_256();
        // Should be in the 216-color cube range (16-231)
        assert!((16..=231).contains(&idx));
    }

    #[test]
    fn test_rgb_to_16_near_black_no_dominant() {
        // All channels below 40 -> black (0)
        assert_eq!(Rgb::new(10, 10, 10).to_16(), 0);
    }

    #[test]
    fn test_styled_display_256color() {
        let styled = Styled::new("hello", ColorMode::Color256).fg(Rgb::new(255, 0, 0));
        let output = styled.to_string();
        assert!(output.contains("\x1b[38;5;"));
        assert!(output.contains("hello"));
        assert!(output.ends_with("\x1b[0m"));
    }

    #[test]
    fn test_styled_display_16color_bright() {
        let styled = Styled::new("bright", ColorMode::Color16).fg(Rgb::new(255, 50, 50));
        let output = styled.to_string();
        // Bright red -> code 9 -> \x1b[91m
        assert!(output.contains("\x1b[9"));
        assert!(output.contains("bright"));
    }

    #[test]
    fn test_styled_display_16color_dark() {
        let styled = Styled::new("dark", ColorMode::Color16).fg(Rgb::new(0, 0, 100));
        let output = styled.to_string();
        // Dark blue -> code 4 -> \x1b[34m
        assert!(output.contains("\x1b[3"));
        assert!(output.contains("dark"));
    }

    #[test]
    fn test_styled_bold() {
        let styled = Styled::new("bold", ColorMode::TrueColor).bold();
        let output = styled.to_string();
        assert!(output.contains("\x1b[1m"));
        assert!(output.contains("bold"));
        assert!(output.ends_with("\x1b[0m"));
    }

    #[test]
    fn test_styled_bold_and_fg() {
        let styled = Styled::new("both", ColorMode::TrueColor).fg(Rgb::new(0, 255, 0)).bold();
        let output = styled.to_string();
        assert!(output.contains("\x1b[1m"));
        assert!(output.contains("\x1b[38;2;0;255;0m"));
    }

    #[test]
    fn test_styled_no_color_no_bold() {
        let styled = Styled::new("plain", ColorMode::TrueColor);
        let output = styled.to_string();
        // No styles applied, no escape codes
        assert_eq!(output, "plain");
    }

    #[test]
    fn test_training_palette_new() {
        let palette = TrainingPalette::new(ColorMode::TrueColor);
        assert_eq!(palette.mode, ColorMode::TrueColor);
    }

    #[test]
    fn test_training_palette_style() {
        let palette = TrainingPalette::new(ColorMode::Mono);
        let styled = palette.style("text").fg(TrainingPalette::SUCCESS);
        let output = styled.to_string();
        assert_eq!(output, "text"); // Mono mode, no escape codes
    }

    #[test]
    fn test_vram_color_thresholds() {
        assert_eq!(TrainingPalette::vram_color(30.0), TrainingPalette::SUCCESS);
        assert_eq!(TrainingPalette::vram_color(60.0), TrainingPalette::INFO);
        assert_eq!(TrainingPalette::vram_color(80.0), TrainingPalette::WARNING);
        assert_eq!(TrainingPalette::vram_color(95.0), TrainingPalette::ERROR);
    }

    #[test]
    fn test_power_color_thresholds() {
        assert_eq!(TrainingPalette::power_color(40.0), TrainingPalette::SUCCESS);
        assert_eq!(TrainingPalette::power_color(70.0), TrainingPalette::INFO);
        assert_eq!(TrainingPalette::power_color(90.0), TrainingPalette::WARNING);
        assert_eq!(TrainingPalette::power_color(99.0), TrainingPalette::ERROR);
    }

    #[test]
    fn test_loss_color_equal_min_max() {
        // When max <= min, should return INFO
        let color = TrainingPalette::loss_color(0.5, 1.0, 1.0);
        assert_eq!(color, TrainingPalette::INFO);
    }

    #[test]
    fn test_loss_color_at_min() {
        let color = TrainingPalette::loss_color(0.0, 0.0, 10.0);
        // Should be greenish (low loss)
        assert!(color.g > color.r);
    }

    #[test]
    fn test_loss_color_at_max() {
        let color = TrainingPalette::loss_color(10.0, 0.0, 10.0);
        // Should be reddish (high loss)
        assert!(color.r > color.g);
    }

    #[test]
    fn test_loss_color_at_midpoint() {
        let color = TrainingPalette::loss_color(5.0, 0.0, 10.0);
        // At 0.5 normalized -> yellow range, high R and G
        assert!(color.r > 200);
        assert!(color.g > 150);
    }

    #[test]
    fn test_loss_trend_color_all_variants() {
        assert_eq!(
            TrainingPalette::loss_trend_color(&LossTrend::Decreasing),
            TrainingPalette::SUCCESS
        );
        assert_eq!(TrainingPalette::loss_trend_color(&LossTrend::Stable), TrainingPalette::INFO);
        assert_eq!(
            TrainingPalette::loss_trend_color(&LossTrend::Increasing),
            TrainingPalette::ERROR
        );
        assert_eq!(TrainingPalette::loss_trend_color(&LossTrend::Unknown), TrainingPalette::MUTED);
    }

    #[test]
    fn test_progress_color_thresholds() {
        assert_eq!(TrainingPalette::progress_color(50.0), TrainingPalette::INFO);
        assert_eq!(TrainingPalette::progress_color(75.0), TrainingPalette::INFO);
        assert_eq!(TrainingPalette::progress_color(90.0), TrainingPalette::SUCCESS);
        assert_eq!(TrainingPalette::progress_color(100.0), TrainingPalette::PRIMARY);
    }

    #[test]
    fn test_progress_color_boundary_at_75() {
        assert_eq!(TrainingPalette::progress_color(75.0), TrainingPalette::INFO);
        assert_eq!(TrainingPalette::progress_color(75.01), TrainingPalette::SUCCESS);
    }

    #[test]
    fn test_colored_bar_zero_value() {
        let bar = colored_bar(0.0, 100.0, 10, TrainingPalette::SUCCESS, ColorMode::Mono);
        assert_eq!(bar.chars().filter(|&c| c == '░').count(), 10);
        assert_eq!(bar.chars().filter(|&c| c == '█').count(), 0);
    }

    #[test]
    fn test_colored_bar_full_value() {
        let bar = colored_bar(100.0, 100.0, 10, TrainingPalette::SUCCESS, ColorMode::Mono);
        assert_eq!(bar.chars().filter(|&c| c == '█').count(), 10);
        assert_eq!(bar.chars().filter(|&c| c == '░').count(), 0);
    }

    #[test]
    fn test_colored_bar_zero_max() {
        let bar = colored_bar(50.0, 0.0, 10, TrainingPalette::SUCCESS, ColorMode::Mono);
        // Zero max -> 0% fill
        assert_eq!(bar.chars().filter(|&c| c == '░').count(), 10);
    }

    #[test]
    fn test_colored_bar_truecolor() {
        let bar = colored_bar(50.0, 100.0, 10, TrainingPalette::SUCCESS, ColorMode::TrueColor);
        // Should contain ANSI escape sequences
        assert!(bar.contains("\x1b["));
    }

    #[test]
    fn test_colored_value_mono() {
        let s = colored_value(42, Rgb::new(255, 0, 0), ColorMode::Mono);
        assert_eq!(s, "42");
    }

    #[test]
    fn test_colored_value_truecolor() {
        let s = colored_value(3.14, Rgb::new(0, 255, 0), ColorMode::TrueColor);
        assert!(s.contains("3.14"));
        assert!(s.contains("\x1b[38;2;0;255;0m"));
    }

    #[test]
    fn test_threshold_color_below_first() {
        let color = TrainingPalette::threshold_color(
            5.0,
            &[(10.0, Rgb::new(1, 2, 3)), (20.0, Rgb::new(4, 5, 6))],
            Rgb::new(7, 8, 9),
        );
        assert_eq!(color, Rgb::new(1, 2, 3));
    }

    #[test]
    fn test_threshold_color_between_thresholds() {
        let color = TrainingPalette::threshold_color(
            15.0,
            &[(10.0, Rgb::new(1, 2, 3)), (20.0, Rgb::new(4, 5, 6))],
            Rgb::new(7, 8, 9),
        );
        assert_eq!(color, Rgb::new(4, 5, 6));
    }

    #[test]
    fn test_threshold_color_above_all() {
        let color = TrainingPalette::threshold_color(
            25.0,
            &[(10.0, Rgb::new(1, 2, 3)), (20.0, Rgb::new(4, 5, 6))],
            Rgb::new(7, 8, 9),
        );
        assert_eq!(color, Rgb::new(7, 8, 9));
    }

    #[test]
    fn test_threshold_color_empty_thresholds() {
        let color = TrainingPalette::threshold_color(5.0, &[], Rgb::new(7, 8, 9));
        assert_eq!(color, Rgb::new(7, 8, 9));
    }

    #[test]
    fn test_gpu_util_color_clamping() {
        // Below 0 should clamp
        assert_eq!(TrainingPalette::gpu_util_color(-10.0), TrainingPalette::MUTED);
        // Above 100 should clamp
        assert_eq!(TrainingPalette::gpu_util_color(150.0), TrainingPalette::PRIMARY);
    }

    #[test]
    fn test_temp_color_clamping() {
        assert_eq!(TrainingPalette::temp_color(-20.0), TrainingPalette::SUCCESS);
        assert_eq!(TrainingPalette::temp_color(250.0), TrainingPalette::ERROR);
    }

    #[test]
    fn test_colored_bar_overflow_value() {
        // Value > max should clamp
        let bar = colored_bar(200.0, 100.0, 10, TrainingPalette::SUCCESS, ColorMode::Mono);
        assert_eq!(bar.chars().filter(|&c| c == '█').count(), 10);
    }

    #[test]
    fn test_styled_display_mono_with_bold_and_fg() {
        // In Mono mode, bold and fg are ignored
        let styled = Styled::new("mono", ColorMode::Mono).fg(Rgb::new(255, 0, 0)).bold();
        let output = styled.to_string();
        assert_eq!(output, "mono"); // No escape codes
    }

    #[test]
    fn test_semantic_color_constants_distinct() {
        // Verify all semantic colors are distinct
        let colors = [
            TrainingPalette::SUCCESS,
            TrainingPalette::WARNING,
            TrainingPalette::ERROR,
            TrainingPalette::INFO,
            TrainingPalette::MUTED,
            TrainingPalette::PRIMARY,
        ];
        for i in 0..colors.len() {
            for j in (i + 1)..colors.len() {
                assert_ne!(colors[i], colors[j], "colors at {i} and {j} should differ");
            }
        }
    }
}
