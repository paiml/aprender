//! Terminal output encoder (ASCII/Unicode/ANSI).
//!
//! Renders framebuffers to terminal-compatible text output.
//! Supports multiple rendering modes:
//! - ASCII: Uses characters like ` .:-=+*#%@` for grayscale
//! - Unicode: Uses block characters (▄ ▀ █) for higher resolution
//! - ANSI: Adds 24-bit color codes for full color output

use crate::framebuffer::Framebuffer;
use std::fmt::Write as FmtWrite;

/// Terminal rendering mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TerminalMode {
    /// ASCII grayscale characters (widest compatibility)
    Ascii,
    /// Unicode half-block characters (2x vertical resolution)
    #[default]
    UnicodeHalfBlock,
    /// Unicode full blocks with ANSI 24-bit color
    AnsiTrueColor,
}

/// Terminal encoder configuration.
#[derive(Debug, Clone)]
pub struct TerminalEncoder {
    mode: TerminalMode,
    width: Option<u32>,
    height: Option<u32>,
    invert: bool,
}

impl Default for TerminalEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl TerminalEncoder {
    /// ASCII grayscale ramp from dark to light (10 levels).
    const ASCII_RAMP: &'static [char] = &[' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];

    /// Create a new terminal encoder with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self { mode: TerminalMode::default(), width: None, height: None, invert: false }
    }

    /// Set the rendering mode.
    #[must_use]
    pub fn mode(mut self, mode: TerminalMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the target width in characters.
    /// If not set, uses framebuffer width (scaled appropriately for mode).
    #[must_use]
    pub fn width(mut self, width: u32) -> Self {
        self.width = Some(width);
        self
    }

    /// Set the target height in characters/lines.
    /// If not set, calculates from width to preserve aspect ratio.
    #[must_use]
    pub fn height(mut self, height: u32) -> Self {
        self.height = Some(height);
        self
    }

    /// Invert the output (light on dark vs dark on light).
    #[must_use]
    pub fn invert(mut self, invert: bool) -> Self {
        self.invert = invert;
        self
    }

    /// Render a framebuffer to a string.
    #[must_use]
    #[allow(unused_variables)] // generated contract macro binds `path` without assertion
    pub fn render(&self, fb: &Framebuffer) -> String {
        contract_pre_configuration!(fb.pixels());
        let result = match self.mode {
            TerminalMode::Ascii => self.render_ascii(fb),
            TerminalMode::UnicodeHalfBlock => self.render_unicode_half_block(fb),
            TerminalMode::AnsiTrueColor => self.render_ansi_true_color(fb),
        };
        contract_post_configuration!(&result);
        result
    }

    /// Render using ASCII grayscale characters.
    fn render_ascii(&self, fb: &Framebuffer) -> String {
        let (target_w, target_h) = self.compute_dimensions(fb, 2.0);
        let mut output = String::with_capacity((target_w + 1) as usize * target_h as usize);

        let scale_x = fb.width() as f32 / target_w as f32;
        let scale_y = fb.height() as f32 / target_h as f32;

        for y in 0..target_h {
            for x in 0..target_w {
                let luma = self.sample_luma(fb, x, y, scale_x, scale_y);
                let idx = Self::luma_to_index(luma);
                output.push(Self::ASCII_RAMP[idx]);
            }
            output.push('\n');
        }

        output
    }

    /// Render using Unicode half-block characters.
    /// Each character represents 2 vertical pixels using ▀ (upper half) or ▄ (lower half).
    fn render_unicode_half_block(&self, fb: &Framebuffer) -> String {
        let (target_w, target_h) = self.compute_dimensions(fb, 1.0);
        // Round up to even height for half-blocks
        let target_h = (target_h + 1) & !1;

        let mut output =
            String::with_capacity((target_w * 4 + 1) as usize * (target_h / 2) as usize);

        let scale_x = fb.width() as f32 / target_w as f32;
        let scale_y = fb.height() as f32 / target_h as f32;

        for y in (0..target_h).step_by(2) {
            for x in 0..target_w {
                let top = self.sample_color(fb, x, y, scale_x, scale_y);
                let bottom = self.sample_color(fb, x, y + 1, scale_x, scale_y);

                // Use ANSI escape for foreground (top) and background (bottom)
                // ▀ U+2580 = upper half block
                let _ = write!(
                    output,
                    "\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m▀",
                    top.0, top.1, top.2, bottom.0, bottom.1, bottom.2
                );
            }
            output.push_str("\x1b[0m\n");
        }

        output
    }

    /// Render using full blocks with ANSI 24-bit true color.
    fn render_ansi_true_color(&self, fb: &Framebuffer) -> String {
        let (target_w, target_h) = self.compute_dimensions(fb, 2.0);
        let mut output = String::with_capacity((target_w * 20 + 1) as usize * target_h as usize);

        let scale_x = fb.width() as f32 / target_w as f32;
        let scale_y = fb.height() as f32 / target_h as f32;

        for y in 0..target_h {
            for x in 0..target_w {
                let (r, g, b) = self.sample_color(fb, x, y, scale_x, scale_y);
                // Full block with background color + space
                let _ = write!(output, "\x1b[48;2;{r};{g};{b}m ");
            }
            output.push_str("\x1b[0m\n");
        }

        output
    }

    /// Compute target dimensions preserving aspect ratio.
    /// `char_aspect` is the approximate width/height ratio of a character (typically 2.0 for monospace).
    fn compute_dimensions(&self, fb: &Framebuffer, char_aspect: f32) -> (u32, u32) {
        let fb_aspect = fb.width() as f32 / fb.height() as f32;

        match (self.width, self.height) {
            (Some(w), Some(h)) => (w, h),
            (Some(w), None) => {
                let h = (w as f32 / fb_aspect / char_aspect).round() as u32;
                (w, h.max(1))
            }
            (None, Some(h)) => {
                let w = (h as f32 * fb_aspect * char_aspect).round() as u32;
                (w.max(1), h)
            }
            (None, None) => {
                // Default to 80 characters wide
                let w = 80u32.min(fb.width());
                let h = (w as f32 / fb_aspect / char_aspect).round() as u32;
                (w, h.max(1))
            }
        }
    }

    /// Sample and compute luminance at a scaled position.
    fn sample_luma(&self, fb: &Framebuffer, x: u32, y: u32, scale_x: f32, scale_y: f32) -> f32 {
        let fx = (x as f32 * scale_x).min((fb.width() - 1) as f32);
        let fy = (y as f32 * scale_y).min((fb.height() - 1) as f32);

        if let Some(pixel) = fb.get_pixel(fx as u32, fy as u32) {
            // Rec. 709 luminance coefficients
            let luma = 0.2126 * (f32::from(pixel.r) / 255.0)
                + 0.7152 * (f32::from(pixel.g) / 255.0)
                + 0.0722 * (f32::from(pixel.b) / 255.0);

            if self.invert {
                1.0 - luma
            } else {
                luma
            }
        } else {
            0.0
        }
    }

    /// Sample color at a scaled position.
    fn sample_color(
        &self,
        fb: &Framebuffer,
        x: u32,
        y: u32,
        scale_x: f32,
        scale_y: f32,
    ) -> (u8, u8, u8) {
        let fx = (x as f32 * scale_x).min((fb.width() - 1) as f32);
        let fy = (y as f32 * scale_y).min((fb.height() - 1) as f32);

        if let Some(pixel) = fb.get_pixel(fx as u32, fy as u32) {
            if self.invert {
                (255 - pixel.r, 255 - pixel.g, 255 - pixel.b)
            } else {
                (pixel.r, pixel.g, pixel.b)
            }
        } else {
            (0, 0, 0)
        }
    }

    /// Convert luminance (0.0-1.0) to ASCII ramp index.
    fn luma_to_index(luma: f32) -> usize {
        let idx = (luma * (Self::ASCII_RAMP.len() - 1) as f32).round() as usize;
        idx.min(Self::ASCII_RAMP.len() - 1)
    }

    /// Write output directly to stdout.
    pub fn print(&self, fb: &Framebuffer) {
        print!("{}", self.render(fb));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Rgba;

    #[test]
    fn test_ascii_render_basic() {
        let mut fb = Framebuffer::new(10, 10).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);

        let encoder = TerminalEncoder::new().mode(TerminalMode::Ascii).width(5);

        let output = encoder.render(&fb);

        // Should be filled with '@' (brightest ASCII char)
        assert!(output.contains('@'));
        assert!(!output.contains(' ')); // No dark pixels
    }

    #[test]
    fn test_ascii_render_black() {
        let mut fb = Framebuffer::new(10, 10).expect("framebuffer creation should succeed");
        fb.clear(Rgba::BLACK);

        let encoder = TerminalEncoder::new().mode(TerminalMode::Ascii).width(5);

        let output = encoder.render(&fb);

        // Should be filled with ' ' (darkest ASCII char)
        for ch in output.chars() {
            if ch != '\n' {
                assert_eq!(ch, ' ');
            }
        }
    }

    #[test]
    fn test_unicode_half_block_contains_ansi() {
        let mut fb = Framebuffer::new(10, 10).expect("framebuffer creation should succeed");
        fb.clear(Rgba::RED);

        let encoder = TerminalEncoder::new().mode(TerminalMode::UnicodeHalfBlock).width(5);

        let output = encoder.render(&fb);

        // Should contain ANSI escape codes
        assert!(output.contains("\x1b[38;2;"));
        // Should contain half-block character
        assert!(output.contains('▀'));
        // Should contain reset code
        assert!(output.contains("\x1b[0m"));
    }

    #[test]
    fn test_ansi_true_color_contains_escapes() {
        let mut fb = Framebuffer::new(10, 10).expect("framebuffer creation should succeed");
        fb.clear(Rgba::BLUE);

        let encoder = TerminalEncoder::new().mode(TerminalMode::AnsiTrueColor).width(5);

        let output = encoder.render(&fb);

        // Should contain background color escape
        assert!(output.contains("\x1b[48;2;"));
        // Should contain blue color (0, 0, 255)
        assert!(output.contains("48;2;0;0;255"));
    }

    #[test]
    fn test_invert_mode() {
        let mut fb = Framebuffer::new(10, 10).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);

        let encoder = TerminalEncoder::new().mode(TerminalMode::Ascii).width(5).invert(true);

        let output = encoder.render(&fb);

        // White inverted should be dark (space)
        for ch in output.chars() {
            if ch != '\n' {
                assert_eq!(ch, ' ');
            }
        }
    }

    #[test]
    fn test_aspect_ratio_preservation() {
        let fb = Framebuffer::new(200, 100).expect("framebuffer creation should succeed");

        let encoder = TerminalEncoder::new().mode(TerminalMode::Ascii).width(40);

        let output = encoder.render(&fb);
        let lines: Vec<&str> = output.lines().collect();

        // 200:100 = 2:1 aspect ratio
        // With char_aspect of 2.0, 40 width should give ~10 height
        assert!(lines.len() <= 12); // Allow some rounding
        assert!(lines.len() >= 8);
    }

    #[test]
    fn test_gradient_produces_varied_output() {
        let mut fb = Framebuffer::new(100, 10).expect("framebuffer creation should succeed");

        // Create horizontal gradient
        for x in 0..100 {
            let gray = (x as f32 / 99.0 * 255.0) as u8;
            let color = Rgba::new(gray, gray, gray, 255);
            for y in 0..10 {
                fb.set_pixel(x, y, color);
            }
        }

        let encoder = TerminalEncoder::new().mode(TerminalMode::Ascii).width(50);

        let output = encoder.render(&fb);
        let first_line: String =
            output.lines().next().expect("iterator should have next element").chars().collect();

        // Should have varied characters
        let unique_chars: std::collections::HashSet<char> = first_line.chars().collect();
        assert!(unique_chars.len() >= 5, "Gradient should produce varied ASCII");
    }

    #[test]
    fn test_custom_dimensions() {
        let fb = Framebuffer::new(100, 100).expect("framebuffer creation should succeed");

        let encoder = TerminalEncoder::new().mode(TerminalMode::Ascii).width(20).height(10);

        let output = encoder.render(&fb);
        let lines: Vec<&str> = output.lines().collect();

        assert_eq!(lines.len(), 10);
        assert_eq!(lines[0].len(), 20);
    }

    #[test]
    fn test_default_width_capped_at_80() {
        let fb = Framebuffer::new(1000, 100).expect("framebuffer creation should succeed");

        let encoder = TerminalEncoder::new().mode(TerminalMode::Ascii);
        let output = encoder.render(&fb);
        let first_line = output.lines().next().expect("iterator should have next element");

        assert!(first_line.len() <= 80);
    }
}
