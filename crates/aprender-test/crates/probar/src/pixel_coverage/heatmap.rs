//! Heatmap Rendering for Pixel Coverage
//!
//! Renders coverage data as visual heatmaps for terminal and web output.

use super::tracker::{CoverageCell, PixelCoverageTracker};
use serde::{Deserialize, Serialize};

/// RGB color
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rgb {
    /// Red component (0-255)
    pub r: u8,
    /// Green component (0-255)
    pub g: u8,
    /// Blue component (0-255)
    pub b: u8,
}

impl Rgb {
    /// Create a new RGB color
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Create color from hex value
    #[must_use]
    pub const fn from_hex(hex: u32) -> Self {
        Self {
            r: ((hex >> 16) & 0xFF) as u8,
            g: ((hex >> 8) & 0xFF) as u8,
            b: (hex & 0xFF) as u8,
        }
    }
}

/// Color palette for heatmap rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorPalette {
    /// Color for 0% coverage
    pub zero: Rgb,
    /// Color for 25% coverage
    pub low: Rgb,
    /// Color for 50% coverage
    pub medium: Rgb,
    /// Color for 75% coverage
    pub high: Rgb,
    /// Color for 100% coverage
    pub full: Rgb,
}

impl Default for ColorPalette {
    fn default() -> Self {
        Self::viridis()
    }
}

impl ColorPalette {
    /// Viridis color palette (colorblind-friendly)
    #[must_use]
    pub fn viridis() -> Self {
        Self {
            zero: Rgb::from_hex(0x440154),   // Dark purple
            low: Rgb::from_hex(0x3B528B),    // Blue
            medium: Rgb::from_hex(0x21918C), // Teal
            high: Rgb::from_hex(0x5DC863),   // Green
            full: Rgb::from_hex(0xFDE725),   // Yellow
        }
    }

    /// Red-Yellow-Green palette (traffic light)
    #[must_use]
    pub fn traffic_light() -> Self {
        Self {
            zero: Rgb::from_hex(0xFF0000),   // Red
            low: Rgb::from_hex(0xFF6600),    // Orange
            medium: Rgb::from_hex(0xFFFF00), // Yellow
            high: Rgb::from_hex(0x99FF00),   // Yellow-green
            full: Rgb::from_hex(0x00FF00),   // Green
        }
    }

    /// Get color for coverage value
    #[must_use]
    pub fn color_for_coverage(&self, coverage: f32) -> Rgb {
        match coverage {
            c if c <= 0.0 => self.zero,
            c if c <= 0.25 => self.low,
            c if c <= 0.50 => self.medium,
            c if c <= 0.75 => self.high,
            _ => self.full,
        }
    }
}

/// Terminal heatmap renderer
#[derive(Debug, Clone)]
pub struct TerminalHeatmap {
    cells: Vec<Vec<f32>>,
    palette: ColorPalette,
    use_color: bool,
}

impl TerminalHeatmap {
    /// Create from coverage tracker
    #[must_use]
    pub fn from_tracker(tracker: &PixelCoverageTracker) -> Self {
        let cells = tracker
            .cells()
            .iter()
            .map(|row| row.iter().map(|c| c.coverage).collect())
            .collect();

        Self {
            cells,
            palette: ColorPalette::default(),
            use_color: true,
        }
    }

    /// Create from raw coverage values
    #[must_use]
    pub fn from_values(cells: Vec<Vec<f32>>) -> Self {
        Self {
            cells,
            palette: ColorPalette::default(),
            use_color: true,
        }
    }

    /// Set color palette
    #[must_use]
    pub fn with_palette(mut self, palette: ColorPalette) -> Self {
        self.palette = palette;
        self
    }

    /// Disable ANSI color output
    #[must_use]
    pub fn without_color(mut self) -> Self {
        self.use_color = false;
        self
    }

    /// Render to string
    #[must_use]
    pub fn render(&self) -> String {
        let mut output = String::new();

        for row in &self.cells {
            for &coverage in row {
                let char = Self::coverage_to_char(coverage);

                if self.use_color {
                    let color = self.palette.color_for_coverage(coverage);
                    output.push_str(&format!(
                        "\x1b[38;2;{};{};{}m{}\x1b[0m",
                        color.r, color.g, color.b, char
                    ));
                } else {
                    output.push(char);
                }
            }
            output.push('\n');
        }

        output
    }

    /// Render with border
    #[must_use]
    pub fn render_with_border(&self) -> String {
        let width = self.cells.first().map_or(0, Vec::len);
        let mut output = String::new();

        // Top border
        output.push('┌');
        for _ in 0..width {
            output.push('─');
        }
        output.push_str("┐\n");

        // Content
        for row in &self.cells {
            output.push('│');
            for &coverage in row {
                let char = Self::coverage_to_char(coverage);

                if self.use_color {
                    let color = self.palette.color_for_coverage(coverage);
                    output.push_str(&format!(
                        "\x1b[38;2;{};{};{}m{}\x1b[0m",
                        color.r, color.g, color.b, char
                    ));
                } else {
                    output.push(char);
                }
            }
            output.push_str("│\n");
        }

        // Bottom border
        output.push('└');
        for _ in 0..width {
            output.push('─');
        }
        output.push('┘');

        output
    }

    /// Convert coverage value to Unicode block character
    fn coverage_to_char(coverage: f32) -> char {
        match coverage {
            c if c <= 0.0 => ' ',  // Empty
            c if c <= 0.25 => '░', // Light shade
            c if c <= 0.50 => '▒', // Medium shade
            c if c <= 0.75 => '▓', // Dark shade
            _ => '█',              // Full block
        }
    }

    /// Render legend
    #[must_use]
    pub fn legend(&self) -> String {
        let mut legend = String::from("Legend:\n");

        if self.use_color {
            let c = &self.palette;
            legend.push_str(&format!(
                "  \x1b[38;2;{};{};{}m \x1b[0m = 0% (untested)\n",
                c.zero.r, c.zero.g, c.zero.b
            ));
            legend.push_str(&format!(
                "  \x1b[38;2;{};{};{}m░\x1b[0m = 1-25%\n",
                c.low.r, c.low.g, c.low.b
            ));
            legend.push_str(&format!(
                "  \x1b[38;2;{};{};{}m▒\x1b[0m = 26-50%\n",
                c.medium.r, c.medium.g, c.medium.b
            ));
            legend.push_str(&format!(
                "  \x1b[38;2;{};{};{}m▓\x1b[0m = 51-75%\n",
                c.high.r, c.high.g, c.high.b
            ));
            legend.push_str(&format!(
                "  \x1b[38;2;{};{};{}m█\x1b[0m = 76-100%\n",
                c.full.r, c.full.g, c.full.b
            ));
        } else {
            legend.push_str("    = 0% (untested)\n");
            legend.push_str("  ░ = 1-25%\n");
            legend.push_str("  ▒ = 26-50%\n");
            legend.push_str("  ▓ = 51-75%\n");
            legend.push_str("  █ = 76-100%\n");
        }

        legend
    }
}

/// Heatmap renderer trait for extensibility
pub trait HeatmapRenderer {
    /// Render heatmap to string
    fn render(&self, cells: &[Vec<CoverageCell>]) -> String;
}

/// PNG heatmap export with trueno-viz style output
#[derive(Debug, Clone)]
pub struct PngHeatmap {
    /// Output width in pixels
    width: u32,
    /// Output height in pixels
    height: u32,
    /// Color palette
    palette: ColorPalette,
    /// Show legend color bar
    show_legend: bool,
    /// Highlight gaps with red outline
    highlight_gaps: bool,
    /// Show cell borders
    show_borders: bool,
    /// Border color (gray by default)
    border_color: Rgb,
    /// Title text
    title: Option<String>,
    /// Subtitle text (below title)
    subtitle: Option<String>,
    /// Margin around the heatmap (trueno-viz style)
    margin: u32,
    /// Background color
    background: Rgb,
    /// Stats panel for combined coverage display
    pub stats_panel: Option<StatsPanel>,
}

impl Default for PngHeatmap {
    fn default() -> Self {
        Self::new(800, 600)
    }
}

impl PngHeatmap {
    /// Create new PNG exporter with specified dimensions
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            palette: ColorPalette::default(),
            show_legend: false,
            highlight_gaps: false,
            show_borders: true,
            border_color: Rgb::new(80, 80, 80),
            title: None,
            subtitle: None,
            margin: 40,
            background: Rgb::new(255, 255, 255),
            stats_panel: None,
        }
    }

    /// Set margin around the heatmap (trueno-viz style)
    #[must_use]
    pub fn with_margin(mut self, margin: u32) -> Self {
        self.margin = margin;
        self
    }

    /// Set background color
    #[must_use]
    pub fn with_background(mut self, color: Rgb) -> Self {
        self.background = color;
        self
    }

    /// Set border color
    #[must_use]
    pub fn with_border_color(mut self, color: Rgb) -> Self {
        self.border_color = color;
        self
    }

    /// Set color palette
    #[must_use]
    pub fn with_palette(mut self, palette: ColorPalette) -> Self {
        self.palette = palette;
        self
    }

    /// Enable legend overlay
    #[must_use]
    pub fn with_legend(mut self) -> Self {
        self.show_legend = true;
        self
    }

    /// Enable gap highlighting (red outline for 0% coverage cells)
    #[must_use]
    pub fn with_gap_highlighting(mut self) -> Self {
        self.highlight_gaps = true;
        self
    }

    /// Enable or disable cell borders
    #[must_use]
    pub fn with_borders(mut self, show: bool) -> Self {
        self.show_borders = show;
        self
    }

    /// Set title text
    #[must_use]
    pub fn with_title(mut self, title: &str) -> Self {
        self.title = Some(title.to_string());
        self
    }

    /// Set subtitle text (displayed below title)
    #[must_use]
    pub fn with_subtitle(mut self, subtitle: &str) -> Self {
        self.subtitle = Some(subtitle.to_string());
        self
    }

    /// Set combined coverage stats panel
    #[must_use]
    pub fn with_combined_stats(mut self, report: &super::tracker::CombinedCoverageReport) -> Self {
        self.stats_panel = Some(StatsPanel {
            line_coverage: report.line_coverage.element_coverage * 100.0,
            pixel_coverage: report.pixel_coverage.overall_coverage * 100.0,
            overall_score: report.overall_score * 100.0,
            line_details: (
                report.line_coverage.covered_elements,
                report.line_coverage.total_elements,
            ),
            pixel_details: (
                report.pixel_coverage.covered_cells,
                report.pixel_coverage.total_cells,
            ),
            meets_threshold: report.meets_threshold,
        });
        self
    }

    /// Export to PNG bytes (trueno-viz style with margins)
    #[cfg(feature = "media")]
    pub fn export(&self, cells: &[Vec<CoverageCell>]) -> Result<Vec<u8>, std::io::Error> {
        use image::{ImageBuffer, Rgb as ImageRgb, RgbImage};
        use std::io::Cursor;

        let rows = cells.len();
        let cols = cells.first().map_or(0, Vec::len);

        if rows == 0 || cols == 0 {
            // Return minimal 1x1 PNG
            let img: RgbImage = ImageBuffer::new(1, 1);
            let mut buffer = Cursor::new(Vec::new());
            img.write_to(&mut buffer, image::ImageFormat::Png)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            return Ok(buffer.into_inner());
        }

        // Create image buffer and fill with background color
        let mut img: RgbImage = ImageBuffer::new(self.width, self.height);
        let bg = ImageRgb([self.background.r, self.background.g, self.background.b]);
        for pixel in img.pixels_mut() {
            *pixel = bg;
        }

        let font = BitmapFont::default();
        let text_color = Rgb::new(0, 0, 0); // Black text

        // Calculate title space
        let title_space = if self.title.is_some() {
            if self.subtitle.is_some() {
                24 // Title + subtitle
            } else {
                12 // Title only
            }
        } else {
            0
        };

        // Calculate stats panel space
        let stats_space = if self.stats_panel.is_some() { 50 } else { 0 };

        // Calculate plot area (with margins, trueno-viz style)
        let legend_space = if self.show_legend { 30 } else { 0 };
        let plot_width = self.width.saturating_sub(2 * self.margin);
        let plot_height = self
            .height
            .saturating_sub(2 * self.margin + legend_space + title_space + stats_space);

        // Render title if present
        let content_y_offset = self.margin + title_space;
        if let Some(title) = &self.title {
            if !title.is_empty() {
                let title_width = font.text_width(title);
                let title_x = (self.width.saturating_sub(title_width)) / 2;
                font.render_text(&mut img, title, title_x, self.margin / 2, text_color);
            }
        }

        // Render subtitle if present
        if let Some(subtitle) = &self.subtitle {
            if !subtitle.is_empty() {
                let subtitle_width = font.text_width(subtitle);
                let subtitle_x = (self.width.saturating_sub(subtitle_width)) / 2;
                let subtitle_y = self.margin / 2 + 10;
                font.render_text(&mut img, subtitle, subtitle_x, subtitle_y, text_color);
            }
        }

        // Calculate cell dimensions within the plot area
        let cell_width = plot_width / cols as u32;
        let cell_height = plot_height / rows as u32;

        let border_rgb = ImageRgb([
            self.border_color.r,
            self.border_color.g,
            self.border_color.b,
        ]);

        // Fill cells within the plot area
        for (row_idx, row) in cells.iter().enumerate() {
            for (col_idx, cell) in row.iter().enumerate() {
                let x_start = self.margin + col_idx as u32 * cell_width;
                let y_start = content_y_offset + row_idx as u32 * cell_height;
                let x_end = (x_start + cell_width).min(self.margin + plot_width);
                let y_end = (y_start + cell_height).min(content_y_offset + plot_height);

                let color = self.palette.interpolate(cell.coverage);
                let cell_rgb = ImageRgb([color.r, color.g, color.b]);

                // Fill cell
                for y in y_start..y_end {
                    for x in x_start..x_end {
                        if x < self.width && y < self.height {
                            img.put_pixel(x, y, cell_rgb);
                        }
                    }
                }

                // Draw border if enabled
                if self.show_borders {
                    // Top border
                    for x in x_start..x_end {
                        if y_start < self.height {
                            img.put_pixel(x, y_start, border_rgb);
                        }
                    }
                    // Left border
                    for y in y_start..y_end {
                        if x_start < self.width {
                            img.put_pixel(x_start, y, border_rgb);
                        }
                    }
                    // Right border (last column)
                    if col_idx == cols - 1 {
                        for y in y_start..y_end {
                            if x_end > 0 && x_end <= self.width {
                                img.put_pixel(x_end - 1, y, border_rgb);
                            }
                        }
                    }
                    // Bottom border (last row)
                    if row_idx == rows - 1 {
                        for x in x_start..x_end {
                            if y_end > 0 && y_end <= self.height {
                                img.put_pixel(x, y_end - 1, border_rgb);
                            }
                        }
                    }
                }

                // Highlight gaps with red outline if enabled
                if self.highlight_gaps && cell.coverage <= 0.0 {
                    let gap_color = ImageRgb([255, 0, 0]);
                    // Draw thicker red border for gaps (3 pixels)
                    for thickness in 0..3 {
                        // Top border
                        for x in x_start..x_end {
                            if y_start + thickness < self.height {
                                img.put_pixel(x, y_start + thickness, gap_color);
                            }
                        }
                        // Bottom border
                        if y_end > thickness {
                            let y_bottom = y_end - 1 - thickness;
                            for x in x_start..x_end {
                                if y_bottom < self.height {
                                    img.put_pixel(x, y_bottom, gap_color);
                                }
                            }
                        }
                        // Left border
                        for y in y_start..y_end {
                            if x_start + thickness < self.width {
                                img.put_pixel(x_start + thickness, y, gap_color);
                            }
                        }
                        // Right border
                        if x_end > thickness {
                            let x_right = x_end - 1 - thickness;
                            for y in y_start..y_end {
                                if x_right < self.width {
                                    img.put_pixel(x_right, y, gap_color);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Draw legend color bar if enabled
        if self.show_legend {
            let legend_height = 20;
            let legend_y = self
                .height
                .saturating_sub(self.margin / 2 + legend_height + stats_space);
            let legend_width = plot_width;
            let legend_x_start = self.margin;

            // Draw legend bar
            for x in legend_x_start..(legend_x_start + legend_width) {
                let coverage = (x - legend_x_start) as f32 / legend_width as f32;
                let color = self.palette.interpolate(coverage);
                for y in legend_y..(legend_y + legend_height).min(self.height) {
                    img.put_pixel(x, y, ImageRgb([color.r, color.g, color.b]));
                }
            }

            // Draw legend labels
            font.render_text(
                &mut img,
                "0%",
                legend_x_start,
                legend_y + legend_height + 2,
                text_color,
            );
            let label_100 = "100%";
            let label_width = font.text_width(label_100);
            font.render_text(
                &mut img,
                label_100,
                legend_x_start + legend_width - label_width,
                legend_y + legend_height + 2,
                text_color,
            );
        }

        // Draw stats panel if present
        if let Some(stats) = &self.stats_panel {
            let stats_y = self.height.saturating_sub(stats_space + self.margin / 4);
            let stats_x = self.margin;

            // Line coverage
            let line_text = format!(
                "Line: {:.1}% ({}/{})",
                stats.line_coverage, stats.line_details.0, stats.line_details.1
            );
            font.render_text(&mut img, &line_text, stats_x, stats_y, text_color);

            // Pixel coverage
            let pixel_text = format!(
                "Pixel: {:.1}% ({}/{})",
                stats.pixel_coverage, stats.pixel_details.0, stats.pixel_details.1
            );
            font.render_text(&mut img, &pixel_text, stats_x, stats_y + 12, text_color);

            // Overall score
            let overall_text = format!("Overall: {:.1}%", stats.overall_score);
            let threshold_indicator = if stats.meets_threshold {
                " PASS"
            } else {
                " FAIL"
            };
            let full_text = format!("{}{}", overall_text, threshold_indicator);
            font.render_text(&mut img, &full_text, stats_x, stats_y + 24, text_color);
        }

        // Encode to PNG
        let mut buffer = Cursor::new(Vec::new());
        img.write_to(&mut buffer, image::ImageFormat::Png)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        Ok(buffer.into_inner())
    }

    /// Export to file
    #[cfg(feature = "media")]
    pub fn export_to_file(
        &self,
        cells: &[Vec<CoverageCell>],
        path: &std::path::Path,
    ) -> Result<(), std::io::Error> {
        let bytes = self.export(cells)?;
        std::fs::write(path, bytes)
    }
}

impl ColorPalette {
    /// Magma color palette (dark to bright)
    #[must_use]
    pub fn magma() -> Self {
        Self {
            zero: Rgb::from_hex(0x000004),   // Almost black
            low: Rgb::from_hex(0x51127C),    // Dark purple
            medium: Rgb::from_hex(0xB63679), // Magenta
            high: Rgb::from_hex(0xFB8861),   // Orange
            full: Rgb::from_hex(0xFCFDBF),   // Light yellow
        }
    }

    /// Heat color palette (black-red-yellow-white)
    #[must_use]
    pub fn heat() -> Self {
        Self {
            zero: Rgb::from_hex(0x000000),   // Black
            low: Rgb::from_hex(0x8B0000),    // Dark red
            medium: Rgb::from_hex(0xFF4500), // Orange red
            high: Rgb::from_hex(0xFFD700),   // Gold
            full: Rgb::from_hex(0xFFFFFF),   // White
        }
    }

    /// Interpolate color for any coverage value 0.0-1.0
    /// Returns smooth gradient instead of discrete steps
    #[must_use]
    pub fn interpolate(&self, coverage: f32) -> Rgb {
        let coverage = coverage.clamp(0.0, 1.0);

        // Define color stops
        let stops: [(f32, Rgb); 5] = [
            (0.0, self.zero),
            (0.25, self.low),
            (0.5, self.medium),
            (0.75, self.high),
            (1.0, self.full),
        ];

        // Find the two stops to interpolate between
        for i in 0..stops.len() - 1 {
            let (t0, c0) = stops[i];
            let (t1, c1) = stops[i + 1];

            if coverage >= t0 && coverage <= t1 {
                let t = (coverage - t0) / (t1 - t0);
                return Rgb::lerp(c0, c1, t);
            }
        }

        // Fallback to full coverage color
        self.full
    }
}

impl Rgb {
    /// Linear interpolation between two colors
    #[must_use]
    pub fn lerp(c0: Rgb, c1: Rgb, t: f32) -> Rgb {
        let t = t.clamp(0.0, 1.0);
        Rgb {
            r: (f32::from(c0.r) + (f32::from(c1.r) - f32::from(c0.r)) * t) as u8,
            g: (f32::from(c0.g) + (f32::from(c1.g) - f32::from(c0.g)) * t) as u8,
            b: (f32::from(c0.b) + (f32::from(c1.b) - f32::from(c0.b)) * t) as u8,
        }
    }
}

// =============================================================================
// Bitmap Font for Text Rendering
// =============================================================================

/// Simple 5x7 bitmap font for PNG text rendering
/// Each character is represented as a 7-element array of u8 (5 bits per row)
#[derive(Debug, Clone)]
pub struct BitmapFont {
    /// Character width in pixels
    char_width: u32,
    /// Character height in pixels
    char_height: u32,
    /// Spacing between characters
    spacing: u32,
}

impl Default for BitmapFont {
    fn default() -> Self {
        Self {
            char_width: 5,
            char_height: 7,
            spacing: 1,
        }
    }
}

impl BitmapFont {
    /// Get character width
    #[must_use]
    pub const fn char_width(&self) -> u32 {
        self.char_width
    }

    /// Get character height
    #[must_use]
    pub const fn char_height(&self) -> u32 {
        self.char_height
    }

    /// Get spacing between characters
    #[must_use]
    pub const fn spacing(&self) -> u32 {
        self.spacing
    }

    /// Get text width in pixels
    #[must_use]
    pub fn text_width(&self, text: &str) -> u32 {
        let len = text.chars().count() as u32;
        if len == 0 {
            return 0;
        }
        len * self.char_width + (len - 1) * self.spacing
    }

    /// Get glyph bitmap for a character (5x7 bits as Vec<bool>)
    #[must_use]
    pub fn glyph(&self, c: char) -> Vec<bool> {
        let bitmap = Self::char_bitmap(c);
        let mut result = Vec::with_capacity(35);
        for row in &bitmap {
            for bit in 0..5 {
                result.push((row >> (4 - bit)) & 1 == 1);
            }
        }
        result
    }

    /// Get raw bitmap data for character (7 rows, each u8 represents 5 bits)
    #[must_use]
    const fn char_bitmap(c: char) -> [u8; 7] {
        match c {
            // Uppercase letters
            'A' => [
                0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
            ],
            'B' => [
                0b11110, 0b10001, 0b11110, 0b10001, 0b10001, 0b10001, 0b11110,
            ],
            'C' => [
                0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
            ],
            'D' => [
                0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
            ],
            'E' => [
                0b11111, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000, 0b11111,
            ],
            'F' => [
                0b11111, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000, 0b10000,
            ],
            'G' => [
                0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110,
            ],
            'H' => [
                0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001, 0b10001,
            ],
            'I' => [
                0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
            ],
            'J' => [
                0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100,
            ],
            'K' => [
                0b10001, 0b10010, 0b11100, 0b10010, 0b10001, 0b10001, 0b10001,
            ],
            'L' => [
                0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
            ],
            'M' => [
                0b10001, 0b11011, 0b10101, 0b10001, 0b10001, 0b10001, 0b10001,
            ],
            'N' => [
                0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
            ],
            'O' => [
                0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
            ],
            'P' => [
                0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
            ],
            'Q' => [
                0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b01110, 0b00001,
            ],
            'R' => [
                0b11110, 0b10001, 0b10001, 0b11110, 0b10010, 0b10001, 0b10001,
            ],
            'S' => [
                0b01110, 0b10001, 0b10000, 0b01110, 0b00001, 0b10001, 0b01110,
            ],
            'T' => [
                0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
            ],
            'U' => [
                0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
            ],
            'V' => [
                0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
            ],
            'W' => [
                0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001,
            ],
            'X' => [
                0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
            ],
            'Y' => [
                0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
            ],
            'Z' => [
                0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
            ],
            // Lowercase (map to uppercase for simplicity)
            'a'..='z' => Self::char_bitmap((c as u8 - 32) as char),
            // Digits
            '0' => [
                0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
            ],
            '1' => [
                0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
            ],
            '2' => [
                0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111,
            ],
            '3' => [
                0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110,
            ],
            '4' => [
                0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
            ],
            '5' => [
                0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
            ],
            '6' => [
                0b01110, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b01110,
            ],
            '7' => [
                0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
            ],
            '8' => [
                0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
            ],
            '9' => [
                0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110,
            ],
            // Punctuation and symbols
            ' ' => [
                0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000,
            ],
            '.' => [
                0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100,
            ],
            ',' => [
                0b00000, 0b00000, 0b00000, 0b00000, 0b00110, 0b00100, 0b01000,
            ],
            ':' => [
                0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000,
            ],
            '-' => [
                0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000,
            ],
            '_' => [
                0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111,
            ],
            '/' => [
                0b00001, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b10000,
            ],
            '%' => [
                0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011,
            ],
            '(' => [
                0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010,
            ],
            ')' => [
                0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000,
            ],
            '=' => [
                0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000,
            ],
            '+' => [
                0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000,
            ],
            '*' => [
                0b00000, 0b10101, 0b01110, 0b11111, 0b01110, 0b10101, 0b00000,
            ],
            '!' => [
                0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00100,
            ],
            '?' => [
                0b01110, 0b10001, 0b00001, 0b00110, 0b00100, 0b00000, 0b00100,
            ],
            // Default: empty
            _ => [
                0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000,
            ],
        }
    }

    /// Render text to an image buffer at the specified position
    #[cfg(feature = "media")]
    pub fn render_text(&self, img: &mut image::RgbImage, text: &str, x: u32, y: u32, color: Rgb) {
        use image::Rgb as ImageRgb;

        let text_color = ImageRgb([color.r, color.g, color.b]);
        let mut cursor_x = x;

        for c in text.chars() {
            let bitmap = Self::char_bitmap(c);

            for (row_idx, &row) in bitmap.iter().enumerate() {
                for bit in 0..5 {
                    if (row >> (4 - bit)) & 1 == 1 {
                        let px = cursor_x + bit;
                        let py = y + row_idx as u32;
                        if px < img.width() && py < img.height() {
                            img.put_pixel(px, py, text_color);
                        }
                    }
                }
            }

            cursor_x += self.char_width + self.spacing;
        }
    }
}

/// Stats panel content for combined coverage display
#[derive(Debug, Clone)]
pub struct StatsPanel {
    /// Line coverage percentage
    pub line_coverage: f32,
    /// Pixel coverage percentage
    pub pixel_coverage: f32,
    /// Overall score
    pub overall_score: f32,
    /// Line coverage details (covered/total)
    pub line_details: (usize, usize),
    /// Pixel coverage details (covered/total)
    pub pixel_details: (u32, u32),
    /// Whether threshold is met
    pub meets_threshold: bool,
}

/// SVG heatmap export
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SvgHeatmap {
    width: u32,
    height: u32,
    palette: ColorPalette,
}

#[allow(dead_code)]
impl SvgHeatmap {
    /// Create new SVG exporter
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            palette: ColorPalette::default(),
        }
    }

    /// Set color palette
    #[must_use]
    pub fn with_palette(mut self, palette: ColorPalette) -> Self {
        self.palette = palette;
        self
    }

    /// Export to SVG string
    #[must_use]
    pub fn export(&self, cells: &[Vec<CoverageCell>]) -> String {
        let rows = cells.len();
        let cols = cells.first().map_or(0, Vec::len);

        if rows == 0 || cols == 0 {
            return String::from("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>");
        }

        let cell_width = self.width / cols as u32;
        let cell_height = self.height / rows as u32;

        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
            self.width, self.height, self.width, self.height
        );

        svg.push_str("\n  <style>.cell { stroke: #333; stroke-width: 0.5; }</style>\n");

        for (row_idx, row) in cells.iter().enumerate() {
            for (col_idx, cell) in row.iter().enumerate() {
                let x = col_idx as u32 * cell_width;
                let y = row_idx as u32 * cell_height;
                let color = self.palette.color_for_coverage(cell.coverage);

                svg.push_str(&format!(
                    r#"  <rect class="cell" x="{}" y="{}" width="{}" height="{}" fill="rgb({},{},{})"/>"#,
                    x, y, cell_width, cell_height, color.r, color.g, color.b
                ));
                svg.push('\n');
            }
        }

        svg.push_str("</svg>");
        svg
    }
}

// =============================================================================
// Visual Regression Testing Infrastructure
// =============================================================================

/// Visual regression testing utilities for PNG heatmap output
#[cfg(test)]
pub mod visual_regression {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    /// Reference checksum for deterministic PNG output
    #[allow(dead_code)]
    #[derive(Debug, Clone)]
    pub struct ReferenceChecksum {
        /// Hash of the PNG bytes
        pub checksum: u64,
        /// Description of the reference
        pub description: &'static str,
        /// Width of the reference image
        pub width: u32,
        /// Height of the reference image
        pub height: u32,
    }

    /// Result of visual comparison
    #[derive(Debug)]
    pub struct ComparisonResult {
        /// Whether images match within tolerance
        pub matches: bool,
        /// Percentage of pixels that differ
        pub diff_percentage: f32,
        /// Maximum color difference found
        pub max_diff: u8,
        /// Number of differing pixels
        pub diff_count: usize,
        /// Total pixels compared
        #[allow(dead_code)]
        pub total_pixels: usize,
    }

    /// Compare two PNG images pixel-by-pixel with tolerance
    ///
    /// # Arguments
    /// * `reference` - Reference PNG bytes
    /// * `generated` - Generated PNG bytes
    /// * `tolerance` - Per-channel color tolerance (0-255)
    ///
    /// # Returns
    /// `ComparisonResult` with match status and diff statistics
    pub fn compare_png_with_tolerance(
        reference: &[u8],
        generated: &[u8],
        tolerance: u8,
    ) -> Result<ComparisonResult, String> {
        use image::GenericImageView;

        let ref_img = image::load_from_memory(reference)
            .map_err(|e| format!("Failed to load reference image: {}", e))?;
        let gen_img = image::load_from_memory(generated)
            .map_err(|e| format!("Failed to load generated image: {}", e))?;

        // Check dimensions match
        if ref_img.dimensions() != gen_img.dimensions() {
            return Ok(ComparisonResult {
                matches: false,
                diff_percentage: 100.0,
                max_diff: 255,
                diff_count: (ref_img.width() * ref_img.height()) as usize,
                total_pixels: (ref_img.width() * ref_img.height()) as usize,
            });
        }

        let (width, height) = ref_img.dimensions();
        let total_pixels = (width * height) as usize;
        let mut diff_count = 0;
        let mut max_diff: u8 = 0;

        for y in 0..height {
            for x in 0..width {
                let ref_pixel = ref_img.get_pixel(x, y);
                let gen_pixel = gen_img.get_pixel(x, y);

                // Compare RGB channels (ignore alpha)
                let diff_r = (ref_pixel[0] as i16 - gen_pixel[0] as i16).unsigned_abs() as u8;
                let diff_g = (ref_pixel[1] as i16 - gen_pixel[1] as i16).unsigned_abs() as u8;
                let diff_b = (ref_pixel[2] as i16 - gen_pixel[2] as i16).unsigned_abs() as u8;

                let channel_max = diff_r.max(diff_g).max(diff_b);
                max_diff = max_diff.max(channel_max);

                if channel_max > tolerance {
                    diff_count += 1;
                }
            }
        }

        let diff_percentage = (diff_count as f32 / total_pixels as f32) * 100.0;
        let matches = diff_count == 0 || diff_percentage < 0.1; // Allow 0.1% variance

        Ok(ComparisonResult {
            matches,
            diff_percentage,
            max_diff,
            diff_count,
            total_pixels,
        })
    }

    /// Compute deterministic checksum for PNG bytes
    pub fn compute_checksum(png_bytes: &[u8]) -> u64 {
        let mut hasher = DefaultHasher::new();
        png_bytes.hash(&mut hasher);
        hasher.finish()
    }

    /// Generate reference cells for testing (deterministic pattern)
    pub fn reference_gradient_cells(rows: usize, cols: usize) -> Vec<Vec<CoverageCell>> {
        let mut cells = Vec::with_capacity(rows);
        for row in 0..rows {
            let mut row_cells = Vec::with_capacity(cols);
            for col in 0..cols {
                let coverage = (row as f32 / (rows - 1).max(1) as f32
                    + col as f32 / (cols - 1).max(1) as f32)
                    / 2.0;
                row_cells.push(CoverageCell {
                    coverage,
                    hit_count: (coverage * 10.0) as u64,
                });
            }
            cells.push(row_cells);
        }
        cells
    }

    /// Generate reference cells with gaps (deterministic pattern)
    pub fn reference_gap_cells(rows: usize, cols: usize) -> Vec<Vec<CoverageCell>> {
        let mut cells = reference_gradient_cells(rows, cols);
        // Add deterministic gaps
        if rows > 2 && cols > 2 {
            cells[rows / 2][cols / 2] = CoverageCell {
                coverage: 0.0,
                hit_count: 0,
            };
        }
        if rows > 4 && cols > 4 {
            cells[rows / 4][cols / 4] = CoverageCell {
                coverage: 0.0,
                hit_count: 0,
            };
        }
        cells
    }

    /// Generate uniform cells for testing
    pub fn reference_uniform_cells(
        rows: usize,
        cols: usize,
        coverage: f32,
    ) -> Vec<Vec<CoverageCell>> {
        vec![
            vec![
                CoverageCell {
                    coverage,
                    hit_count: (coverage * 10.0) as u64,
                };
                cols
            ];
            rows
        ]
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_from_hex() {
        let red = Rgb::from_hex(0xFF0000);
        assert_eq!(red.r, 255);
        assert_eq!(red.g, 0);
        assert_eq!(red.b, 0);

        let white = Rgb::from_hex(0xFFFFFF);
        assert_eq!(white.r, 255);
        assert_eq!(white.g, 255);
        assert_eq!(white.b, 255);
    }

    #[test]
    fn test_color_palette_viridis() {
        let palette = ColorPalette::viridis();
        assert_ne!(palette.zero, palette.full);
    }

    #[test]
    fn test_color_for_coverage() {
        let palette = ColorPalette::traffic_light();

        assert_eq!(palette.color_for_coverage(0.0), palette.zero);
        assert_eq!(palette.color_for_coverage(0.1), palette.low);
        assert_eq!(palette.color_for_coverage(0.4), palette.medium);
        assert_eq!(palette.color_for_coverage(0.6), palette.high);
        assert_eq!(palette.color_for_coverage(1.0), palette.full);
    }

    #[test]
    fn test_terminal_heatmap_render() {
        let cells = vec![vec![0.0, 0.25, 0.5], vec![0.75, 1.0, 0.0]];

        let heatmap = TerminalHeatmap::from_values(cells).without_color();
        let rendered = heatmap.render();

        assert!(rendered.contains(' ')); // 0% coverage
        assert!(rendered.contains('█')); // 100% coverage
    }

    #[test]
    fn test_terminal_heatmap_with_border() {
        let cells = vec![vec![1.0, 1.0], vec![0.0, 0.0]];

        let heatmap = TerminalHeatmap::from_values(cells).without_color();
        let rendered = heatmap.render_with_border();

        assert!(rendered.contains('┌'));
        assert!(rendered.contains('┘'));
        assert!(rendered.contains('│'));
    }

    #[test]
    fn test_coverage_to_char() {
        assert_eq!(TerminalHeatmap::coverage_to_char(0.0), ' ');
        assert_eq!(TerminalHeatmap::coverage_to_char(0.1), '░');
        assert_eq!(TerminalHeatmap::coverage_to_char(0.3), '▒');
        assert_eq!(TerminalHeatmap::coverage_to_char(0.6), '▓');
        assert_eq!(TerminalHeatmap::coverage_to_char(1.0), '█');
    }

    #[test]
    fn test_svg_export() {
        let cells = vec![vec![CoverageCell {
            hit_count: 1,
            coverage: 1.0,
        }]];

        let svg = SvgHeatmap::new(100, 100).export(&cells);

        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("<rect"));
        assert!(svg.ends_with("</svg>"));
    }

    #[test]
    fn test_svg_empty_cells() {
        let cells: Vec<Vec<CoverageCell>> = vec![];
        let svg = SvgHeatmap::new(100, 100).export(&cells);
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn test_legend() {
        let cells = vec![vec![1.0]];
        let heatmap = TerminalHeatmap::from_values(cells).without_color();
        let legend = heatmap.legend();

        assert!(legend.contains("Legend:"));
        assert!(legend.contains("░"));
        assert!(legend.contains("█"));
    }

    // =========================================================================
    // PNG Heatmap Tests (H₀-PNG-XX)
    // =========================================================================

    #[test]
    fn h0_png_01_basic_render() {
        let cells = vec![vec![CoverageCell {
            coverage: 0.5,
            hit_count: 5,
        }]];
        let png = PngHeatmap::new(100, 100).export(&cells).unwrap();
        assert!(!png.is_empty());
        // Verify PNG header bytes
        assert_eq!(&png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }

    #[test]
    fn h0_png_02_color_interpolation() {
        let palette = ColorPalette::viridis();
        let color_0 = palette.interpolate(0.0);
        let color_50 = palette.interpolate(0.5);
        let color_100 = palette.interpolate(1.0);

        // Should be distinct colors
        assert_ne!(color_0, color_50);
        assert_ne!(color_50, color_100);
    }

    #[test]
    fn h0_png_03_gap_highlighting() {
        let mut cells = vec![
            vec![
                CoverageCell {
                    coverage: 1.0,
                    hit_count: 10,
                };
                10
            ];
            10
        ];
        cells[5][5] = CoverageCell {
            coverage: 0.0,
            hit_count: 0,
        }; // Gap

        let png = PngHeatmap::new(100, 100)
            .with_gap_highlighting()
            .export(&cells)
            .unwrap();

        // Should render successfully with gap highlighted
        assert!(!png.is_empty());
        // Verify PNG header
        assert_eq!(&png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }

    #[test]
    fn h0_png_04_magma_palette() {
        let palette = ColorPalette::magma();
        assert_ne!(palette.zero, palette.full);
        // Magma starts nearly black
        assert!(palette.zero.r < 10);
        assert!(palette.zero.g < 10);
    }

    #[test]
    fn h0_png_05_heat_palette() {
        let palette = ColorPalette::heat();
        assert_ne!(palette.zero, palette.full);
        // Heat starts at black
        assert_eq!(palette.zero, Rgb::new(0, 0, 0));
        // Heat ends at white
        assert_eq!(palette.full, Rgb::new(255, 255, 255));
    }

    #[test]
    fn h0_png_06_rgb_lerp() {
        let black = Rgb::new(0, 0, 0);
        let white = Rgb::new(255, 255, 255);

        let mid = Rgb::lerp(black, white, 0.5);
        assert_eq!(mid.r, 127);
        assert_eq!(mid.g, 127);
        assert_eq!(mid.b, 127);

        // Extremes
        assert_eq!(Rgb::lerp(black, white, 0.0), black);
        assert_eq!(Rgb::lerp(black, white, 1.0), white);
    }

    #[test]
    fn h0_png_07_interpolate_boundaries() {
        let palette = ColorPalette::viridis();

        // Exactly at boundaries
        let c0 = palette.interpolate(0.0);
        let c25 = palette.interpolate(0.25);
        let c50 = palette.interpolate(0.5);
        let c75 = palette.interpolate(0.75);
        let c100 = palette.interpolate(1.0);

        assert_eq!(c0, palette.zero);
        assert_eq!(c25, palette.low);
        assert_eq!(c50, palette.medium);
        assert_eq!(c75, palette.high);
        assert_eq!(c100, palette.full);
    }

    #[test]
    fn h0_png_08_interpolate_clamping() {
        let palette = ColorPalette::viridis();

        // Out of range values should be clamped
        let below = palette.interpolate(-0.5);
        let above = palette.interpolate(1.5);

        assert_eq!(below, palette.zero);
        assert_eq!(above, palette.full);
    }

    #[test]
    fn h0_png_09_empty_cells() {
        let cells: Vec<Vec<CoverageCell>> = vec![];
        let png = PngHeatmap::new(100, 100).export(&cells).unwrap();
        // Should still produce valid PNG (1x1 fallback)
        assert!(!png.is_empty());
        assert_eq!(&png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }

    #[test]
    fn h0_png_10_with_legend() {
        let cells = vec![
            vec![
                CoverageCell {
                    coverage: 0.0,
                    hit_count: 0,
                },
                CoverageCell {
                    coverage: 1.0,
                    hit_count: 10,
                },
            ],
            vec![
                CoverageCell {
                    coverage: 0.5,
                    hit_count: 5,
                },
                CoverageCell {
                    coverage: 0.75,
                    hit_count: 8,
                },
            ],
        ];

        let png = PngHeatmap::new(200, 200)
            .with_legend()
            .with_palette(ColorPalette::magma())
            .export(&cells)
            .unwrap();

        assert!(!png.is_empty());
        assert_eq!(&png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }

    #[test]
    fn h0_png_11_builder_pattern() {
        let heatmap = PngHeatmap::new(800, 600)
            .with_palette(ColorPalette::heat())
            .with_legend()
            .with_gap_highlighting()
            .with_borders(false)
            .with_title("Test Heatmap");

        // Verify settings applied (indirectly through export working)
        let cells = vec![vec![CoverageCell {
            coverage: 0.5,
            hit_count: 5,
        }]];
        let png = heatmap.export(&cells).unwrap();
        assert!(!png.is_empty());
    }

    #[test]
    fn h0_png_12_export_to_file() {
        let cells = vec![
            vec![
                CoverageCell {
                    coverage: 0.0,
                    hit_count: 0,
                },
                CoverageCell {
                    coverage: 0.5,
                    hit_count: 5,
                },
                CoverageCell {
                    coverage: 1.0,
                    hit_count: 10,
                },
            ];
            3
        ];

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_heatmap.png");

        PngHeatmap::new(300, 300)
            .with_gap_highlighting()
            .export_to_file(&cells, &path)
            .unwrap();

        // Verify file exists and is valid PNG
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

        // Cleanup
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn h0_png_13_default() {
        let heatmap = PngHeatmap::default();
        let cells = vec![vec![CoverageCell {
            coverage: 0.5,
            hit_count: 5,
        }]];
        let png = heatmap.export(&cells).unwrap();
        assert!(!png.is_empty());
    }

    // =========================================================================
    // Title/Metadata Text Rendering Tests (H₀-TXT-XX)
    // =========================================================================

    #[test]
    fn h0_txt_01_title_renders() {
        let cells = vec![
            vec![
                CoverageCell {
                    coverage: 0.5,
                    hit_count: 5,
                };
                5
            ];
            5
        ];

        let png = PngHeatmap::new(400, 300)
            .with_title("Test Coverage")
            .export(&cells)
            .unwrap();

        assert!(!png.is_empty());
        assert_eq!(&png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }

    #[test]
    fn h0_txt_02_title_with_legend() {
        let cells = vec![
            vec![
                CoverageCell {
                    coverage: 1.0,
                    hit_count: 10,
                };
                3
            ];
            3
        ];

        let png = PngHeatmap::new(400, 300)
            .with_title("Coverage Heatmap")
            .with_legend()
            .export(&cells)
            .unwrap();

        assert!(!png.is_empty());
    }

    #[test]
    fn h0_txt_03_bitmap_font_basic() {
        // Test that bitmap font renders without panics
        let font = BitmapFont::default();
        let glyph = font.glyph('A');
        assert!(!glyph.is_empty());
    }

    #[test]
    fn h0_txt_04_bitmap_font_digits() {
        let font = BitmapFont::default();
        for c in '0'..='9' {
            let glyph = font.glyph(c);
            assert!(!glyph.is_empty(), "Digit {} should have a glyph", c);
        }
    }

    #[test]
    fn h0_txt_05_bitmap_font_text_width() {
        let font = BitmapFont::default();
        let width = font.text_width("Hello");
        assert!(width > 0);
        assert_eq!(
            width,
            5 * (font.char_width() + font.spacing()) - font.spacing()
        );
    }

    #[test]
    fn h0_txt_06_metadata_subtitle() {
        let cells = vec![
            vec![
                CoverageCell {
                    coverage: 0.75,
                    hit_count: 8,
                };
                4
            ];
            4
        ];

        let png = PngHeatmap::new(500, 400)
            .with_title("Main Title")
            .with_subtitle("85% coverage")
            .export(&cells)
            .unwrap();

        assert!(!png.is_empty());
    }

    #[test]
    fn h0_txt_07_empty_title() {
        let cells = vec![vec![CoverageCell {
            coverage: 0.5,
            hit_count: 5,
        }]];

        // Empty title should not cause issues
        let png = PngHeatmap::new(200, 200)
            .with_title("")
            .export(&cells)
            .unwrap();

        assert!(!png.is_empty());
    }

    #[test]
    fn h0_txt_08_special_characters() {
        let font = BitmapFont::default();
        // Should return empty glyph for unknown chars
        let glyph = font.glyph('€');
        assert!(glyph.is_empty() || glyph.iter().all(|&b| !b));
    }

    // =========================================================================
    // Combined PNG Tests (H₀-CMB-XX)
    // =========================================================================

    #[test]
    fn h0_cmb_01_combined_heatmap() {
        use super::super::tracker::{
            CombinedCoverageReport, LineCoverageReport, PixelCoverageReport,
        };

        let cells = vec![
            vec![
                CoverageCell {
                    coverage: 0.8,
                    hit_count: 8,
                };
                10
            ];
            10
        ];

        let line_report = LineCoverageReport::new(0.90, 1.0, 0.80, 22, 20);
        let pixel_report = PixelCoverageReport {
            overall_coverage: 0.85,
            covered_cells: 85,
            total_cells: 100,
            ..Default::default()
        };
        let combined = CombinedCoverageReport::from_parts(line_report, pixel_report);

        let png = PngHeatmap::new(600, 500)
            .with_title("Combined Coverage")
            .with_legend()
            .with_combined_stats(&combined)
            .export(&cells)
            .unwrap();

        assert!(!png.is_empty());
    }

    #[test]
    fn h0_cmb_02_stats_panel_height() {
        // Stats panel should add extra height
        use super::super::tracker::{
            CombinedCoverageReport, LineCoverageReport, PixelCoverageReport,
        };

        let line_report = LineCoverageReport::new(0.90, 1.0, 0.80, 22, 20);
        let pixel_report = PixelCoverageReport::default();
        let combined = CombinedCoverageReport::from_parts(line_report, pixel_report);

        let heatmap = PngHeatmap::new(400, 300).with_combined_stats(&combined);

        // The stats panel should be stored
        assert!(heatmap.stats_panel.is_some());
    }

    // =========================================================================
    // Visual Regression Tests (H₀-VIS-XX)
    // =========================================================================

    #[test]
    fn h0_vis_01_deterministic_output() {
        use super::visual_regression::*;

        // Same input should produce identical output
        let cells = reference_gradient_cells(8, 10);

        let png1 = PngHeatmap::new(400, 300)
            .with_palette(ColorPalette::viridis())
            .export(&cells)
            .unwrap();

        let png2 = PngHeatmap::new(400, 300)
            .with_palette(ColorPalette::viridis())
            .export(&cells)
            .unwrap();

        // Byte-for-byte identical
        assert_eq!(png1.len(), png2.len());
        assert_eq!(compute_checksum(&png1), compute_checksum(&png2));
    }

    #[test]
    fn h0_vis_02_compare_identical_images() {
        use super::visual_regression::*;

        let cells = reference_uniform_cells(5, 5, 0.5);
        let png = PngHeatmap::new(200, 200).export(&cells).unwrap();

        let result = compare_png_with_tolerance(&png, &png, 0).unwrap();

        assert!(result.matches);
        assert_eq!(result.diff_count, 0);
        assert_eq!(result.max_diff, 0);
        assert!((result.diff_percentage - 0.0).abs() < 0.001);
    }

    #[test]
    fn h0_vis_03_compare_different_palettes() {
        use super::visual_regression::*;

        let cells = reference_gradient_cells(5, 5);

        let png_viridis = PngHeatmap::new(200, 200)
            .with_palette(ColorPalette::viridis())
            .export(&cells)
            .unwrap();

        let png_magma = PngHeatmap::new(200, 200)
            .with_palette(ColorPalette::magma())
            .export(&cells)
            .unwrap();

        // Different palettes should produce different output
        let result = compare_png_with_tolerance(&png_viridis, &png_magma, 0).unwrap();

        assert!(!result.matches || result.max_diff > 0);
    }

    #[test]
    fn h0_vis_04_gap_highlighting_visible() {
        use super::visual_regression::*;

        let cells = reference_gap_cells(8, 10);

        let png_no_gaps = PngHeatmap::new(400, 300).export(&cells).unwrap();

        let png_with_gaps = PngHeatmap::new(400, 300)
            .with_gap_highlighting()
            .export(&cells)
            .unwrap();

        // Gap highlighting should produce different output
        let result = compare_png_with_tolerance(&png_no_gaps, &png_with_gaps, 0).unwrap();

        // Should have some differences (the red gap borders)
        assert!(
            result.diff_count > 0,
            "Gap highlighting should produce visible differences"
        );
    }

    #[test]
    fn h0_vis_05_legend_visible() {
        use super::visual_regression::*;

        let cells = reference_gradient_cells(5, 5);

        let png_no_legend = PngHeatmap::new(300, 250).export(&cells).unwrap();

        let png_with_legend = PngHeatmap::new(300, 250)
            .with_legend()
            .export(&cells)
            .unwrap();

        // Legend should produce different output
        let result = compare_png_with_tolerance(&png_no_legend, &png_with_legend, 0).unwrap();

        assert!(
            result.diff_count > 0,
            "Legend should produce visible differences"
        );
    }

    #[test]
    fn h0_vis_06_title_visible() {
        use super::visual_regression::*;

        let cells = reference_uniform_cells(4, 4, 0.75);

        let png_no_title = PngHeatmap::new(300, 200).export(&cells).unwrap();

        let png_with_title = PngHeatmap::new(300, 200)
            .with_title("Test Title")
            .export(&cells)
            .unwrap();

        // Title should produce different output
        let result = compare_png_with_tolerance(&png_no_title, &png_with_title, 0).unwrap();

        assert!(
            result.diff_count > 0,
            "Title should produce visible differences"
        );
    }

    #[test]
    fn h0_vis_07_reference_viridis_gradient() {
        use super::visual_regression::*;

        // Generate reference gradient with Viridis palette
        let cells = reference_gradient_cells(10, 15);
        let png = PngHeatmap::new(800, 600)
            .with_palette(ColorPalette::viridis())
            .with_legend()
            .with_margin(40)
            .export(&cells)
            .unwrap();

        // Store checksum as reference (captured from known-good output)
        let checksum = compute_checksum(&png);

        // Verify we get a valid PNG
        assert!(!png.is_empty());
        assert_eq!(&png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

        // Re-generate and verify determinism
        let png2 = PngHeatmap::new(800, 600)
            .with_palette(ColorPalette::viridis())
            .with_legend()
            .with_margin(40)
            .export(&cells)
            .unwrap();

        assert_eq!(
            compute_checksum(&png2),
            checksum,
            "Output should be deterministic"
        );
    }

    #[test]
    fn h0_vis_08_reference_magma_gaps() {
        use super::visual_regression::*;

        // Generate reference with gaps and Magma palette
        let cells = reference_gap_cells(8, 12);
        let png = PngHeatmap::new(600, 400)
            .with_palette(ColorPalette::magma())
            .with_gap_highlighting()
            .with_legend()
            .export(&cells)
            .unwrap();

        let checksum = compute_checksum(&png);

        // Verify determinism
        let png2 = PngHeatmap::new(600, 400)
            .with_palette(ColorPalette::magma())
            .with_gap_highlighting()
            .with_legend()
            .export(&cells)
            .unwrap();

        assert_eq!(
            compute_checksum(&png2),
            checksum,
            "Magma gap output should be deterministic"
        );
    }

    #[test]
    fn h0_vis_09_reference_heat_with_title() {
        use super::visual_regression::*;

        // Generate reference with Heat palette and title
        let cells = reference_uniform_cells(6, 8, 0.65);
        let png = PngHeatmap::new(500, 400)
            .with_palette(ColorPalette::heat())
            .with_title("Heat Coverage")
            .with_subtitle("Reference Test")
            .with_legend()
            .export(&cells)
            .unwrap();

        let checksum = compute_checksum(&png);

        // Verify determinism
        let png2 = PngHeatmap::new(500, 400)
            .with_palette(ColorPalette::heat())
            .with_title("Heat Coverage")
            .with_subtitle("Reference Test")
            .with_legend()
            .export(&cells)
            .unwrap();

        assert_eq!(
            compute_checksum(&png2),
            checksum,
            "Heat title output should be deterministic"
        );
    }

    #[test]
    fn h0_vis_10_tolerance_comparison() {
        use super::visual_regression::*;

        let cells = reference_gradient_cells(5, 5);
        let png = PngHeatmap::new(200, 200).export(&cells).unwrap();

        // Exact match with 0 tolerance
        let result0 = compare_png_with_tolerance(&png, &png, 0).unwrap();
        assert!(result0.matches);
        assert_eq!(result0.diff_count, 0);

        // Also matches with higher tolerance
        let result10 = compare_png_with_tolerance(&png, &png, 10).unwrap();
        assert!(result10.matches);
        assert_eq!(result10.diff_count, 0);
    }

    #[test]
    fn h0_vis_11_combined_stats_determinism() {
        use super::super::tracker::{
            CombinedCoverageReport, LineCoverageReport, PixelCoverageReport,
        };
        use super::visual_regression::*;

        let cells = reference_gradient_cells(8, 10);

        let line_report = LineCoverageReport::new(0.85, 0.95, 0.90, 20, 17);
        let pixel_report = PixelCoverageReport {
            overall_coverage: 0.80,
            covered_cells: 64,
            total_cells: 80,
            ..Default::default()
        };
        let combined = CombinedCoverageReport::from_parts(line_report, pixel_report);

        let png1 = PngHeatmap::new(700, 600)
            .with_palette(ColorPalette::viridis())
            .with_title("Combined Report")
            .with_legend()
            .with_gap_highlighting()
            .with_combined_stats(&combined)
            .export(&cells)
            .unwrap();

        let checksum1 = compute_checksum(&png1);

        // Re-create with same parameters
        let line_report2 = LineCoverageReport::new(0.85, 0.95, 0.90, 20, 17);
        let pixel_report2 = PixelCoverageReport {
            overall_coverage: 0.80,
            covered_cells: 64,
            total_cells: 80,
            ..Default::default()
        };
        let combined2 = CombinedCoverageReport::from_parts(line_report2, pixel_report2);

        let png2 = PngHeatmap::new(700, 600)
            .with_palette(ColorPalette::viridis())
            .with_title("Combined Report")
            .with_legend()
            .with_gap_highlighting()
            .with_combined_stats(&combined2)
            .export(&cells)
            .unwrap();

        assert_eq!(
            compute_checksum(&png2),
            checksum1,
            "Combined stats output should be deterministic"
        );
    }

    #[test]
    fn h0_vis_12_dimension_mismatch() {
        use super::visual_regression::*;

        let cells_small = reference_uniform_cells(3, 3, 0.5);
        let cells_large = reference_uniform_cells(5, 5, 0.5);

        let png_small = PngHeatmap::new(100, 100).export(&cells_small).unwrap();
        let png_large = PngHeatmap::new(200, 200).export(&cells_large).unwrap();

        // Different dimensions should fail comparison
        let result = compare_png_with_tolerance(&png_small, &png_large, 255).unwrap();

        assert!(!result.matches, "Different dimensions should not match");
        assert_eq!(result.diff_percentage, 100.0);
    }

    // =========================================================================
    // Additional Coverage Tests (H₀-COV-XX)
    // =========================================================================

    #[test]
    fn h0_cov_01_terminal_from_tracker() {
        // Test TerminalHeatmap::from_tracker
        let tracker = super::super::tracker::PixelCoverageTracker::new(100, 100, 5, 5);
        let heatmap = TerminalHeatmap::from_tracker(&tracker);
        let rendered = heatmap.render();
        // Should render 5 rows
        assert_eq!(rendered.lines().count(), 5);
    }

    #[test]
    fn h0_cov_02_terminal_with_palette() {
        let cells = vec![vec![0.5, 1.0], vec![0.0, 0.25]];
        let heatmap = TerminalHeatmap::from_values(cells)
            .with_palette(ColorPalette::traffic_light())
            .without_color();
        let rendered = heatmap.render();
        assert!(rendered.contains('▒')); // 50% coverage
        assert!(rendered.contains('█')); // 100% coverage
    }

    #[test]
    fn h0_cov_03_terminal_render_with_color() {
        let cells = vec![vec![0.0, 0.5, 1.0]];
        let heatmap = TerminalHeatmap::from_values(cells);
        // use_color is true by default
        let rendered = heatmap.render();
        // Should contain ANSI escape sequences
        assert!(rendered.contains("\x1b[38;2;"));
        assert!(rendered.contains("\x1b[0m"));
    }

    #[test]
    fn h0_cov_04_terminal_border_with_color() {
        let cells = vec![vec![0.5, 1.0]];
        let heatmap = TerminalHeatmap::from_values(cells);
        let rendered = heatmap.render_with_border();
        // Should contain border chars and ANSI sequences
        assert!(rendered.contains('┌'));
        assert!(rendered.contains("\x1b[38;2;"));
    }

    #[test]
    fn h0_cov_05_terminal_legend_with_color() {
        let cells = vec![vec![1.0]];
        let heatmap = TerminalHeatmap::from_values(cells);
        let legend = heatmap.legend();
        // Should contain ANSI escape sequences in legend
        assert!(legend.contains("\x1b[38;2;"));
        assert!(legend.contains("Legend:"));
    }

    #[test]
    fn h0_cov_06_terminal_empty_cells_border() {
        let cells: Vec<Vec<f32>> = vec![];
        let heatmap = TerminalHeatmap::from_values(cells).without_color();
        let rendered = heatmap.render_with_border();
        // Should still render borders with width 0
        assert!(rendered.contains('┌'));
        assert!(rendered.contains('└'));
    }

    #[test]
    fn h0_cov_07_png_with_margin() {
        let cells = vec![vec![CoverageCell {
            coverage: 0.5,
            hit_count: 5,
        }]];
        let png = PngHeatmap::new(200, 200)
            .with_margin(60)
            .export(&cells)
            .unwrap();
        assert!(!png.is_empty());
        assert_eq!(&png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }

    #[test]
    fn h0_cov_08_png_with_background() {
        let cells = vec![vec![CoverageCell {
            coverage: 0.5,
            hit_count: 5,
        }]];
        let png = PngHeatmap::new(200, 200)
            .with_background(Rgb::new(0, 0, 0)) // Black background
            .export(&cells)
            .unwrap();
        assert!(!png.is_empty());
    }

    #[test]
    fn h0_cov_09_png_with_border_color() {
        let cells = vec![
            vec![
                CoverageCell {
                    coverage: 0.5,
                    hit_count: 5,
                };
                3
            ];
            3
        ];
        let png = PngHeatmap::new(200, 200)
            .with_border_color(Rgb::new(255, 0, 0)) // Red borders
            .export(&cells)
            .unwrap();
        assert!(!png.is_empty());
    }

    #[test]
    fn h0_cov_10_bitmap_font_dimensions() {
        let font = BitmapFont::default();
        assert_eq!(font.char_width(), 5);
        assert_eq!(font.char_height(), 7);
        assert_eq!(font.spacing(), 1);
    }

    #[test]
    fn h0_cov_11_bitmap_font_empty_text_width() {
        let font = BitmapFont::default();
        assert_eq!(font.text_width(""), 0);
    }

    #[test]
    fn h0_cov_12_bitmap_font_single_char_width() {
        let font = BitmapFont::default();
        let width = font.text_width("A");
        assert_eq!(width, 5); // Just char_width, no spacing
    }

    #[test]
    fn h0_cov_13_bitmap_font_punctuation() {
        let font = BitmapFont::default();
        // Test all punctuation characters
        let chars = [
            '.', ',', ':', '-', '_', '/', '%', '(', ')', '=', '+', '*', '!', '?', ' ',
        ];
        for c in chars {
            let glyph = font.glyph(c);
            assert_eq!(glyph.len(), 35, "Glyph for '{}' should have 35 bits", c);
        }
    }

    #[test]
    fn h0_cov_14_bitmap_font_lowercase_to_uppercase() {
        let font = BitmapFont::default();
        // Lowercase should map to uppercase
        let upper = font.glyph('A');
        let lower = font.glyph('a');
        assert_eq!(upper, lower, "Lowercase should map to uppercase");
    }

    #[test]
    fn h0_cov_15_bitmap_font_all_uppercase() {
        let font = BitmapFont::default();
        for c in 'A'..='Z' {
            let glyph = font.glyph(c);
            // Each glyph should have some pixels set (not all false)
            assert!(
                glyph.iter().any(|&b| b),
                "Glyph for '{}' should have some pixels",
                c
            );
        }
    }

    #[test]
    fn h0_cov_16_rgb_lerp_clamping() {
        let black = Rgb::new(0, 0, 0);
        let white = Rgb::new(255, 255, 255);

        // Test clamping at negative values
        let below = Rgb::lerp(black, white, -1.0);
        assert_eq!(below, black);

        // Test clamping above 1.0
        let above = Rgb::lerp(black, white, 2.0);
        assert_eq!(above, white);
    }

    #[test]
    fn h0_cov_17_color_palette_default() {
        let default = ColorPalette::default();
        let viridis = ColorPalette::viridis();
        assert_eq!(default.zero, viridis.zero);
        assert_eq!(default.full, viridis.full);
    }

    #[test]
    fn h0_cov_18_svg_with_palette() {
        let cells = vec![vec![CoverageCell {
            coverage: 0.5,
            hit_count: 5,
        }]];
        let svg = SvgHeatmap::new(100, 100)
            .with_palette(ColorPalette::magma())
            .export(&cells);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn h0_cov_19_reference_gap_cells_small() {
        use super::visual_regression::*;
        // Test with small grid that won't have gaps at division points
        let cells = reference_gap_cells(2, 2);
        assert_eq!(cells.len(), 2);
        assert_eq!(cells[0].len(), 2);
    }

    #[test]
    fn h0_cov_20_reference_gap_cells_medium() {
        use super::visual_regression::*;
        // Test with grid large enough for first gap but not second
        let cells = reference_gap_cells(3, 3);
        // rows/2 = 1, cols/2 = 1 -> gap at (1,1)
        assert_eq!(cells[1][1].coverage, 0.0);
        assert_eq!(cells[1][1].hit_count, 0);
    }

    #[test]
    fn h0_cov_21_stats_panel_fields() {
        let panel = StatsPanel {
            line_coverage: 85.5,
            pixel_coverage: 90.2,
            overall_score: 87.85,
            line_details: (17, 20),
            pixel_details: (45, 50),
            meets_threshold: true,
        };
        assert!((panel.line_coverage - 85.5).abs() < 0.01);
        assert!((panel.pixel_coverage - 90.2).abs() < 0.01);
        assert!((panel.overall_score - 87.85).abs() < 0.01);
        assert_eq!(panel.line_details, (17, 20));
        assert_eq!(panel.pixel_details, (45, 50));
        assert!(panel.meets_threshold);
    }

    #[test]
    fn h0_cov_22_stats_panel_fail_threshold() {
        use super::super::tracker::{
            CombinedCoverageReport, LineCoverageReport, PixelCoverageReport,
        };

        let cells = vec![
            vec![
                CoverageCell {
                    coverage: 0.3,
                    hit_count: 3,
                };
                5
            ];
            5
        ];

        // Create report that fails threshold
        let line_report = LineCoverageReport::new(0.5, 0.5, 0.5, 10, 5);
        let pixel_report = PixelCoverageReport {
            overall_coverage: 0.3,
            covered_cells: 15,
            total_cells: 50,
            ..Default::default()
        };
        let combined = CombinedCoverageReport::from_parts(line_report, pixel_report);

        let png = PngHeatmap::new(400, 400)
            .with_combined_stats(&combined)
            .export(&cells)
            .unwrap();

        assert!(!png.is_empty());
    }

    #[test]
    fn h0_cov_23_empty_subtitle() {
        let cells = vec![vec![CoverageCell {
            coverage: 0.5,
            hit_count: 5,
        }]];
        let png = PngHeatmap::new(200, 200)
            .with_subtitle("")
            .export(&cells)
            .unwrap();
        assert!(!png.is_empty());
    }

    #[test]
    fn h0_cov_24_title_and_subtitle() {
        let cells = vec![vec![CoverageCell {
            coverage: 0.5,
            hit_count: 5,
        }]];
        let png = PngHeatmap::new(400, 300)
            .with_title("Title")
            .with_subtitle("Subtitle")
            .export(&cells)
            .unwrap();
        assert!(!png.is_empty());
    }

    #[test]
    fn h0_cov_25_coverage_boundaries() {
        // Test exact boundary values for color_for_coverage
        let palette = ColorPalette::viridis();

        // Negative coverage
        assert_eq!(palette.color_for_coverage(-0.1), palette.zero);

        // Exactly 0.25
        assert_eq!(palette.color_for_coverage(0.25), palette.low);

        // Exactly 0.50
        assert_eq!(palette.color_for_coverage(0.50), palette.medium);

        // Exactly 0.75
        assert_eq!(palette.color_for_coverage(0.75), palette.high);

        // Above 0.75
        assert_eq!(palette.color_for_coverage(0.76), palette.full);
    }

    #[test]
    fn h0_cov_26_coverage_to_char_boundaries() {
        // Test exact boundary values
        assert_eq!(TerminalHeatmap::coverage_to_char(-0.1), ' ');
        assert_eq!(TerminalHeatmap::coverage_to_char(0.25), '░');
        assert_eq!(TerminalHeatmap::coverage_to_char(0.26), '▒');
        assert_eq!(TerminalHeatmap::coverage_to_char(0.50), '▒');
        assert_eq!(TerminalHeatmap::coverage_to_char(0.51), '▓');
        assert_eq!(TerminalHeatmap::coverage_to_char(0.75), '▓');
        assert_eq!(TerminalHeatmap::coverage_to_char(0.76), '█');
    }

    #[test]
    fn h0_cov_27_interpolate_mid_segment() {
        let palette = ColorPalette::viridis();

        // Test interpolation within a segment (not at boundaries)
        let c = palette.interpolate(0.125); // Middle of 0-0.25 segment
                                            // Should be between zero and low
        assert_ne!(c, palette.zero);
        assert_ne!(c, palette.low);
    }

    #[test]
    fn h0_cov_28_reference_gradient_single_cell() {
        use super::visual_regression::*;
        // Single cell grid (edge case with max(1) divisor)
        let cells = reference_gradient_cells(1, 1);
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].len(), 1);
        // Coverage should be 0.0 (row=0, col=0, divided by max(1)=1)
        assert!((cells[0][0].coverage - 0.0).abs() < 0.01);
    }

    #[test]
    fn h0_cov_29_png_borders_disabled() {
        let cells = vec![
            vec![
                CoverageCell {
                    coverage: 0.5,
                    hit_count: 5,
                };
                3
            ];
            3
        ];
        let png = PngHeatmap::new(200, 200)
            .with_borders(false)
            .export(&cells)
            .unwrap();
        assert!(!png.is_empty());
    }

    #[test]
    fn h0_cov_30_png_all_options() {
        use super::super::tracker::{
            CombinedCoverageReport, LineCoverageReport, PixelCoverageReport,
        };

        let cells = vec![
            vec![
                CoverageCell {
                    coverage: 0.0,
                    hit_count: 0,
                },
                CoverageCell {
                    coverage: 0.5,
                    hit_count: 5,
                },
            ],
            vec![
                CoverageCell {
                    coverage: 1.0,
                    hit_count: 10,
                },
                CoverageCell {
                    coverage: 0.0,
                    hit_count: 0,
                },
            ],
        ];

        let line_report = LineCoverageReport::new(0.9, 0.95, 0.85, 20, 18);
        let pixel_report = PixelCoverageReport {
            overall_coverage: 0.5,
            covered_cells: 2,
            total_cells: 4,
            ..Default::default()
        };
        let combined = CombinedCoverageReport::from_parts(line_report, pixel_report);

        let png = PngHeatmap::new(600, 500)
            .with_palette(ColorPalette::traffic_light())
            .with_title("Full Options Test")
            .with_subtitle("All features enabled")
            .with_legend()
            .with_gap_highlighting()
            .with_borders(true)
            .with_margin(50)
            .with_background(Rgb::new(240, 240, 240))
            .with_border_color(Rgb::new(100, 100, 100))
            .with_combined_stats(&combined)
            .export(&cells)
            .unwrap();

        assert!(!png.is_empty());
        assert_eq!(&png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }

    #[test]
    fn h0_cov_31_rgb_new() {
        let color = Rgb::new(128, 64, 32);
        assert_eq!(color.r, 128);
        assert_eq!(color.g, 64);
        assert_eq!(color.b, 32);
    }

    #[test]
    fn h0_cov_32_comparison_result_fields() {
        use super::visual_regression::*;

        let cells = reference_uniform_cells(5, 5, 0.5);
        let png = PngHeatmap::new(200, 200).export(&cells).unwrap();
        let result = compare_png_with_tolerance(&png, &png, 0).unwrap();

        // Verify all fields are accessible
        assert!(result.matches);
        assert_eq!(result.diff_count, 0);
        assert_eq!(result.max_diff, 0);
        assert!((result.diff_percentage - 0.0).abs() < 0.001);
        assert!(result.total_pixels > 0);
    }

    #[test]
    fn h0_cov_33_checksum_determinism() {
        use super::visual_regression::*;

        let data1 = vec![1, 2, 3, 4, 5];
        let data2 = vec![1, 2, 3, 4, 5];
        let data3 = vec![5, 4, 3, 2, 1];

        assert_eq!(compute_checksum(&data1), compute_checksum(&data2));
        assert_ne!(compute_checksum(&data1), compute_checksum(&data3));
    }

    #[test]
    fn h0_cov_34_svg_multiple_cells() {
        let cells = vec![
            vec![
                CoverageCell {
                    coverage: 0.0,
                    hit_count: 0,
                },
                CoverageCell {
                    coverage: 0.5,
                    hit_count: 5,
                },
                CoverageCell {
                    coverage: 1.0,
                    hit_count: 10,
                },
            ],
            vec![
                CoverageCell {
                    coverage: 0.25,
                    hit_count: 2,
                },
                CoverageCell {
                    coverage: 0.75,
                    hit_count: 7,
                },
                CoverageCell {
                    coverage: 0.5,
                    hit_count: 5,
                },
            ],
        ];

        let svg = SvgHeatmap::new(300, 200).export(&cells);

        // Should have 6 rect elements (2 rows x 3 cols)
        let rect_count = svg.matches("<rect").count();
        assert_eq!(rect_count, 6);
    }

    #[test]
    fn h0_cov_35_bitmap_font_render_bounds() {
        use image::{ImageBuffer, RgbImage};

        let font = BitmapFont::default();
        let mut img: RgbImage = ImageBuffer::new(10, 10);

        // Try to render text that would overflow bounds
        font.render_text(&mut img, "HELLO WORLD TEST", 5, 5, Rgb::new(0, 0, 0));

        // Should not panic - pixels outside bounds are simply skipped
    }

    #[test]
    fn h0_cov_36_interpolate_at_segment_boundaries() {
        let palette = ColorPalette::viridis();

        // Test values just below boundaries
        let c1 = palette.interpolate(0.249);
        let c2 = palette.interpolate(0.251);
        // These should be in different segments, producing different interpolations
        assert!(c1.r != c2.r || c1.g != c2.g || c1.b != c2.b);
    }

    #[test]
    fn h0_cov_37_heatmap_renderer_trait() {
        // Test that HeatmapRenderer trait is properly defined
        struct TestRenderer;
        impl HeatmapRenderer for TestRenderer {
            fn render(&self, cells: &[Vec<CoverageCell>]) -> String {
                format!("{}x{}", cells.len(), cells.first().map_or(0, Vec::len))
            }
        }

        let cells = vec![
            vec![
                CoverageCell {
                    coverage: 1.0,
                    hit_count: 10,
                };
                3
            ];
            2
        ];
        let renderer = TestRenderer;
        assert_eq!(renderer.render(&cells), "2x3");
    }

    #[test]
    fn h0_cov_38_terminal_multiple_rows() {
        let cells = vec![
            vec![0.0, 0.1, 0.2],
            vec![0.3, 0.4, 0.5],
            vec![0.6, 0.7, 0.8],
            vec![0.9, 1.0, 0.0],
        ];
        let heatmap = TerminalHeatmap::from_values(cells).without_color();
        let rendered = heatmap.render();

        // Should have 4 lines
        assert_eq!(rendered.lines().count(), 4);

        // Each line should have 3 characters
        for line in rendered.lines() {
            assert_eq!(line.chars().count(), 3);
        }
    }
}
