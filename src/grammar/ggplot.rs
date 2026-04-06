//! Main GGPlot builder and renderer.
//!
//! Combines all Grammar of Graphics components into a complete visualization.

use crate::color::Rgba;
use crate::error::{Error, Result};
use crate::framebuffer::Framebuffer;
use crate::render::{draw_circle, draw_line_aa, draw_rect, draw_rect_outline, i32_px};
use crate::scale::{LinearScale, Scale};

use super::aes::Aes;
use super::coord::Coord;
use super::data::DataFrame;
use super::facet::Facet;
use super::geom::{Geom, GeomType, PointShape};
use super::theme::Theme;

/// A layer in the plot.
#[derive(Debug, Clone)]
pub struct Layer {
    /// The geometry.
    pub geom: Geom,
    /// Layer-specific data (if different from plot data).
    pub data: Option<DataFrame>,
    /// Layer-specific aesthetics.
    pub aes: Aes,
}

impl Layer {
    /// Create a new layer from a geometry.
    #[must_use]
    pub fn new(geom: Geom) -> Self {
        Self { aes: geom.aes.clone().unwrap_or_default(), geom, data: None }
    }

    /// Set layer-specific data.
    #[must_use]
    pub fn data(mut self, data: DataFrame) -> Self {
        self.data = Some(data);
        self
    }

    /// Set layer aesthetics.
    #[must_use]
    pub fn aes(mut self, aes: Aes) -> Self {
        self.aes = aes;
        self
    }
}

/// Grammar of Graphics plot builder.
#[derive(Debug, Clone)]
pub struct GGPlot {
    /// Plot data.
    data: DataFrame,
    /// Global aesthetic mappings.
    aes: Aes,
    /// Layers.
    layers: Vec<Layer>,
    /// Coordinate system.
    coord: Coord,
    /// Faceting.
    facet: Facet,
    /// Theme.
    theme: Theme,
    /// Width in pixels.
    width: u32,
    /// Height in pixels.
    height: u32,
    /// Title.
    title: Option<String>,
    /// X-axis label.
    xlab: Option<String>,
    /// Y-axis label.
    ylab: Option<String>,
}

impl Default for GGPlot {
    fn default() -> Self {
        Self::new()
    }
}

impl GGPlot {
    /// Create a new plot builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: DataFrame::new(),
            aes: Aes::new(),
            layers: Vec::new(),
            coord: Coord::cartesian(),
            facet: Facet::None,
            theme: Theme::grey(),
            width: 800,
            height: 600,
            title: None,
            xlab: None,
            ylab: None,
        }
    }

    /// Set the data.
    #[must_use]
    pub fn data(mut self, data: DataFrame) -> Self {
        self.data = data;
        self
    }

    /// Convenience: set x and y data directly.
    #[must_use]
    pub fn data_xy(mut self, x: &[f32], y: &[f32]) -> Self {
        self.data = DataFrame::from_xy(x, y);
        self.aes = self.aes.x("x").y("y");
        self
    }

    /// Set global aesthetics.
    #[must_use]
    pub fn aes(mut self, aes: Aes) -> Self {
        self.aes = aes;
        self
    }

    /// Add a geometry layer.
    #[must_use]
    pub fn geom(mut self, geom: Geom) -> Self {
        self.layers.push(Layer::new(geom));
        self
    }

    /// Add a layer.
    #[must_use]
    pub fn layer(mut self, layer: Layer) -> Self {
        self.layers.push(layer);
        self
    }

    /// Set coordinate system.
    #[must_use]
    pub fn coord(mut self, coord: Coord) -> Self {
        self.coord = coord;
        self
    }

    /// Set faceting.
    #[must_use]
    pub fn facet(mut self, facet: Facet) -> Self {
        self.facet = facet;
        self
    }

    /// Set theme.
    #[must_use]
    pub fn theme(mut self, theme: Theme) -> Self {
        self.theme = theme;
        self
    }

    /// Set dimensions.
    #[must_use]
    pub fn dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set title.
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set x-axis label.
    #[must_use]
    pub fn xlab(mut self, label: impl Into<String>) -> Self {
        self.xlab = Some(label.into());
        self
    }

    /// Set y-axis label.
    #[must_use]
    pub fn ylab(mut self, label: impl Into<String>) -> Self {
        self.ylab = Some(label.into());
        self
    }

    /// Build the plot.
    ///
    /// # Errors
    ///
    /// Returns an error if the plot cannot be built.
    pub fn build(self) -> Result<BuiltGGPlot> {
        // Validate we have at least one layer
        if self.layers.is_empty() {
            return Err(Error::Rendering("No geometry layers specified".into()));
        }

        Ok(BuiltGGPlot {
            data: self.data,
            aes: self.aes,
            layers: self.layers,
            coord: self.coord,
            theme: self.theme,
            width: self.width,
            height: self.height,
            title: self.title,
        })
    }
}

/// A built GGPlot ready for rendering.
#[derive(Debug)]
pub struct BuiltGGPlot {
    data: DataFrame,
    aes: Aes,
    layers: Vec<Layer>,
    coord: Coord,
    theme: Theme,
    width: u32,
    height: u32,
    #[allow(dead_code)]
    title: Option<String>,
}

impl BuiltGGPlot {
    /// Render to framebuffer.
    ///
    /// # Errors
    ///
    /// Returns an error if rendering fails.
    pub fn to_framebuffer(&self) -> Result<Framebuffer> {
        let mut fb = Framebuffer::new(self.width, self.height)?;

        // Fill background
        fb.clear(self.theme.background);

        let margin = self.theme.margin;
        let plot_x = margin;
        let plot_y = margin;
        let plot_w = self.width.saturating_sub(2 * margin);
        let plot_h = self.height.saturating_sub(2 * margin);

        // Draw panel background
        draw_rect(
            &mut fb,
            i32_px(plot_x),
            i32_px(plot_y),
            plot_w,
            plot_h,
            self.theme.panel_background,
        );

        // Compute data ranges for scales
        let (x_min, x_max, y_min, y_max) = self.compute_data_ranges();

        // Apply coordinate limits if set
        let (x_min, x_max, y_min, y_max) = match &self.coord {
            Coord::Cartesian { xlim, ylim, .. } => {
                let (xmin, xmax) = xlim.unwrap_or((x_min, x_max));
                let (ymin, ymax) = ylim.unwrap_or((y_min, y_max));
                (xmin, xmax, ymin, ymax)
            }
            _ => (x_min, x_max, y_min, y_max),
        };

        // Create scales
        let x_scale = LinearScale::new((x_min, x_max), (plot_x as f32, (plot_x + plot_w) as f32))?;
        let y_scale = LinearScale::new((y_min, y_max), ((plot_y + plot_h) as f32, plot_y as f32))?; // Inverted for screen coords

        // Draw grid
        if self.theme.show_grid {
            self.draw_grid(&mut fb, &x_scale, &y_scale, plot_x, plot_y, plot_w, plot_h);
        }

        // Draw each layer
        for layer in &self.layers {
            self.render_layer(&mut fb, layer, &x_scale, &y_scale);
        }

        // Draw axes
        if self.theme.show_axis {
            self.draw_axes(&mut fb, plot_x, plot_y, plot_w, plot_h);
        }

        // Draw panel border
        if self.theme.show_panel_border {
            draw_rect_outline(
                &mut fb,
                i32_px(plot_x),
                i32_px(plot_y),
                plot_w,
                plot_h,
                self.theme.axis_color,
                1, // thickness
            );
        }

        Ok(fb)
    }

    /// Compute data ranges across all layers.
    fn compute_data_ranges(&self) -> (f32, f32, f32, f32) {
        let mut x_min = f32::MAX;
        let mut x_max = f32::MIN;
        let mut y_min = f32::MAX;
        let mut y_max = f32::MIN;

        for layer in &self.layers {
            let data = layer.data.as_ref().unwrap_or(&self.data);
            let layer_aes = self.aes.merge(&layer.aes);

            // Get x column
            if let Some(x_col) = &layer_aes.x {
                if let Some(x_data) = data.get_f32(x_col) {
                    for &v in &x_data {
                        if v.is_finite() {
                            x_min = x_min.min(v);
                            x_max = x_max.max(v);
                        }
                    }
                }
            }

            // Get y column
            if let Some(y_col) = &layer_aes.y {
                if let Some(y_data) = data.get_f32(y_col) {
                    for &v in &y_data {
                        if v.is_finite() {
                            y_min = y_min.min(v);
                            y_max = y_max.max(v);
                        }
                    }
                }
            }
        }

        // Handle empty data or single point
        if x_min >= x_max {
            x_min -= 1.0;
            x_max += 1.0;
        }
        if y_min >= y_max {
            y_min -= 1.0;
            y_max += 1.0;
        }

        // Add small padding
        let x_pad = (x_max - x_min) * 0.05;
        let y_pad = (y_max - y_min) * 0.05;

        (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)
    }

    /// Draw grid lines.
    #[allow(clippy::too_many_arguments)]
    fn draw_grid(
        &self,
        fb: &mut Framebuffer,
        x_scale: &LinearScale,
        y_scale: &LinearScale,
        plot_x: u32,
        plot_y: u32,
        plot_w: u32,
        plot_h: u32,
    ) {
        let color = self.theme.grid_color;

        // Draw horizontal grid lines (5 lines)
        for i in 0..=5 {
            let t = i as f32 / 5.0;
            let y_val = y_scale.domain().0 + t * (y_scale.domain().1 - y_scale.domain().0);
            let y_px = y_scale.scale(y_val);

            draw_line_aa(fb, plot_x as f32, y_px, (plot_x + plot_w) as f32, y_px, color);
        }

        // Draw vertical grid lines (5 lines)
        for i in 0..=5 {
            let t = i as f32 / 5.0;
            let x_val = x_scale.domain().0 + t * (x_scale.domain().1 - x_scale.domain().0);
            let x_px = x_scale.scale(x_val);

            draw_line_aa(fb, x_px, plot_y as f32, x_px, (plot_y + plot_h) as f32, color);
        }
    }

    /// Draw axes.
    fn draw_axes(&self, fb: &mut Framebuffer, plot_x: u32, plot_y: u32, plot_w: u32, plot_h: u32) {
        let color = self.theme.axis_color;

        // X axis (bottom)
        draw_line_aa(
            fb,
            plot_x as f32,
            (plot_y + plot_h) as f32,
            (plot_x + plot_w) as f32,
            (plot_y + plot_h) as f32,
            color,
        );

        // Y axis (left)
        draw_line_aa(
            fb,
            plot_x as f32,
            plot_y as f32,
            plot_x as f32,
            (plot_y + plot_h) as f32,
            color,
        );
    }

    /// Render a single layer.
    fn render_layer(
        &self,
        fb: &mut Framebuffer,
        layer: &Layer,
        x_scale: &LinearScale,
        y_scale: &LinearScale,
    ) {
        let data = layer.data.as_ref().unwrap_or(&self.data);
        let aes = self.aes.merge(&layer.aes);

        // Get data
        let x_col = aes.x.as_deref().unwrap_or("x");
        let y_col = aes.y.as_deref().unwrap_or("y");

        let x_data = data.get_f32(x_col).unwrap_or_default();
        let y_data = data.get_f32(y_col).unwrap_or_default();

        let n = x_data.len().min(y_data.len());
        if n == 0 {
            return;
        }

        // Get style from aesthetics
        let color = aes.color_value.unwrap_or(Rgba::new(66, 133, 244, 255));
        let size = aes.size_value.unwrap_or(5.0);

        match &layer.geom.geom_type {
            GeomType::Point { shape } => {
                Self::render_points(fb, &x_data, &y_data, x_scale, y_scale, color, size, *shape);
            }
            GeomType::Line { width } => {
                Self::render_line(fb, &x_data, &y_data, x_scale, y_scale, color, *width);
            }
            GeomType::Bar { width: bar_width } => {
                Self::render_bars(fb, &x_data, &y_data, x_scale, y_scale, color, *bar_width);
            }
            GeomType::Area { alpha } => {
                let area_color = Rgba::new(color.r, color.g, color.b, (255.0 * alpha) as u8);
                Self::render_area(fb, &x_data, &y_data, x_scale, y_scale, area_color);
            }
            GeomType::Hline { yintercept } => {
                let y_px = y_scale.scale(*yintercept);
                draw_line_aa(fb, x_scale.range().0, y_px, x_scale.range().1, y_px, color);
            }
            GeomType::Vline { xintercept } => {
                let x_px = x_scale.scale(*xintercept);
                draw_line_aa(fb, x_px, y_scale.range().0, x_px, y_scale.range().1, color);
            }
            _ => {} // Other geoms not fully implemented yet
        }
    }

    /// Render point geometry.
    #[allow(clippy::too_many_arguments)]
    fn render_points(
        fb: &mut Framebuffer,
        x_data: &[f32],
        y_data: &[f32],
        x_scale: &LinearScale,
        y_scale: &LinearScale,
        color: Rgba,
        size: f32,
        shape: PointShape,
    ) {
        for i in 0..x_data.len().min(y_data.len()) {
            let x = x_scale.scale(x_data[i]);
            let y = y_scale.scale(y_data[i]);
            let r = (size / 2.0) as i32;

            match shape {
                PointShape::Circle => {
                    draw_circle(fb, x as i32, y as i32, r, color);
                }
                PointShape::Square => {
                    draw_rect(
                        fb,
                        x as i32 - r,
                        y as i32 - r,
                        (r * 2) as u32,
                        (r * 2) as u32,
                        color,
                    );
                }
                _ => {
                    // Fallback to circle for other shapes
                    draw_circle(fb, x as i32, y as i32, r, color);
                }
            }
        }
    }

    /// Render line geometry.
    #[allow(clippy::too_many_arguments)]
    fn render_line(
        fb: &mut Framebuffer,
        x_data: &[f32],
        y_data: &[f32],
        x_scale: &LinearScale,
        y_scale: &LinearScale,
        color: Rgba,
        _width: f32,
    ) {
        let n = x_data.len().min(y_data.len());
        if n < 2 {
            return;
        }

        for i in 0..(n - 1) {
            let x0 = x_scale.scale(x_data[i]);
            let y0 = y_scale.scale(y_data[i]);
            let x1 = x_scale.scale(x_data[i + 1]);
            let y1 = y_scale.scale(y_data[i + 1]);

            draw_line_aa(fb, x0, y0, x1, y1, color);
        }
    }

    /// Render bar geometry.
    #[allow(clippy::too_many_arguments)]
    fn render_bars(
        fb: &mut Framebuffer,
        x_data: &[f32],
        y_data: &[f32],
        x_scale: &LinearScale,
        y_scale: &LinearScale,
        color: Rgba,
        bar_width: f32,
    ) {
        let n = x_data.len().min(y_data.len());
        if n == 0 {
            return;
        }

        // Calculate bar width in pixels
        let x_range = x_scale.range().1 - x_scale.range().0;
        let bar_px_width = (x_range / n as f32 * bar_width).max(1.0) as u32;
        let baseline = y_scale.scale(0.0);

        for i in 0..n {
            let x = x_scale.scale(x_data[i]);
            let y = y_scale.scale(y_data[i]);

            let left = (x - bar_px_width as f32 / 2.0) as i32;
            let top = y.min(baseline) as i32;
            let height = (y - baseline).abs() as u32;

            draw_rect(fb, left, top, bar_px_width, height.max(1), color);
        }
    }

    /// Render area geometry.
    #[allow(clippy::too_many_arguments)]
    fn render_area(
        fb: &mut Framebuffer,
        x_data: &[f32],
        y_data: &[f32],
        x_scale: &LinearScale,
        y_scale: &LinearScale,
        color: Rgba,
    ) {
        let n = x_data.len().min(y_data.len());
        if n < 2 {
            return;
        }

        let baseline = y_scale.scale(0.0);

        // Draw vertical slices for area fill
        for i in 0..n {
            let x = x_scale.scale(x_data[i]) as i32;
            let y = y_scale.scale(y_data[i]);
            let y_top = y.min(baseline) as i32;
            let y_bot = y.max(baseline) as i32;

            for py in y_top..=y_bot {
                if x >= 0 && (x as u32) < fb.width() && py >= 0 && (py as u32) < fb.height() {
                    fb.blend_pixel(x as u32, py as u32, color);
                }
            }
        }

        // Draw line on top
        Self::render_line(
            fb,
            x_data,
            y_data,
            x_scale,
            y_scale,
            Rgba::new(color.r, color.g, color.b, 255),
            1.0,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggplot_basic() {
        let plot = GGPlot::new()
            .data_xy(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0])
            .geom(Geom::point())
            .build()
            .expect("operation should succeed");

        let fb = plot.to_framebuffer().expect("operation should succeed");
        assert_eq!(fb.width(), 800);
        assert_eq!(fb.height(), 600);
    }

    #[test]
    fn test_ggplot_with_theme() {
        let plot = GGPlot::new()
            .data_xy(&[1.0, 2.0], &[3.0, 4.0])
            .geom(Geom::point())
            .theme(Theme::dark())
            .dimensions(400, 300)
            .build()
            .expect("operation should succeed");

        let fb = plot.to_framebuffer().expect("operation should succeed");
        assert_eq!(fb.width(), 400);
        assert_eq!(fb.height(), 300);
    }

    #[test]
    fn test_ggplot_multiple_layers() {
        let plot = GGPlot::new()
            .data_xy(&[1.0, 2.0, 3.0, 4.0], &[1.0, 4.0, 2.0, 5.0])
            .geom(Geom::line())
            .geom(Geom::point().aes(Aes::new().color_value(Rgba::RED)))
            .build()
            .expect("operation should succeed");

        let fb = plot.to_framebuffer().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_ggplot_error_no_layers() {
        let result = GGPlot::new().data_xy(&[1.0], &[2.0]).build();

        assert!(result.is_err());
    }

    #[test]
    fn test_ggplot_coord_limits() {
        let plot = GGPlot::new()
            .data_xy(&[1.0, 2.0], &[3.0, 4.0])
            .geom(Geom::point())
            .coord(Coord::cartesian().xlim(0.0, 5.0).ylim(0.0, 10.0))
            .build()
            .expect("operation should succeed");

        let fb = plot.to_framebuffer().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_layer_with_aes() {
        let plot = GGPlot::new()
            .data_xy(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0])
            .aes(Aes::new().color_value(Rgba::BLUE))
            .geom(Geom::point().aes(Aes::new().size_value(10.0)))
            .build()
            .expect("operation should succeed");

        let fb = plot.to_framebuffer().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_layer_data() {
        let layer_data = DataFrame::from_xy(&[5.0, 6.0], &[7.0, 8.0]);
        let layer = Layer::new(Geom::point()).data(layer_data);
        assert!(layer.data.is_some());
    }

    #[test]
    fn test_layer_aes() {
        let layer = Layer::new(Geom::point()).aes(Aes::new().color_value(Rgba::GREEN));
        assert_eq!(layer.aes.color_value, Some(Rgba::GREEN));
    }

    #[test]
    fn test_ggplot_data() {
        let df = DataFrame::from_xy(&[1.0, 2.0], &[3.0, 4.0]);
        let plot = GGPlot::new()
            .data(df)
            .aes(Aes::new().x("x").y("y"))
            .geom(Geom::point())
            .build()
            .expect("operation should succeed");

        let fb = plot.to_framebuffer().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_ggplot_layer() {
        let layer = Layer::new(Geom::line());
        let plot = GGPlot::new()
            .data_xy(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0])
            .layer(layer)
            .build()
            .expect("builder should produce valid result");

        let fb = plot.to_framebuffer().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_ggplot_facet() {
        let plot = GGPlot::new()
            .data_xy(&[1.0, 2.0], &[3.0, 4.0])
            .geom(Geom::point())
            .facet(Facet::wrap("category", 2))
            .build()
            .expect("operation should succeed");

        let fb = plot.to_framebuffer().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_ggplot_title_labels() {
        let plot = GGPlot::new()
            .data_xy(&[1.0, 2.0], &[3.0, 4.0])
            .geom(Geom::point())
            .title("My Plot")
            .xlab("X Axis")
            .ylab("Y Axis")
            .build()
            .expect("operation should succeed");

        let fb = plot.to_framebuffer().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_ggplot_bar() {
        let plot = GGPlot::new()
            .data_xy(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0])
            .geom(Geom::bar())
            .build()
            .expect("operation should succeed");

        let fb = plot.to_framebuffer().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_ggplot_area() {
        let plot = GGPlot::new()
            .data_xy(&[1.0, 2.0, 3.0, 4.0], &[1.0, 3.0, 2.0, 4.0])
            .geom(Geom::area())
            .build()
            .expect("operation should succeed");

        let fb = plot.to_framebuffer().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_ggplot_hline() {
        let plot = GGPlot::new()
            .data_xy(&[1.0, 2.0, 3.0], &[1.0, 4.0, 2.0])
            .geom(Geom::point())
            .geom(Geom::hline(2.5))
            .build()
            .expect("operation should succeed");

        let fb = plot.to_framebuffer().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_ggplot_vline() {
        let plot = GGPlot::new()
            .data_xy(&[1.0, 2.0, 3.0], &[1.0, 4.0, 2.0])
            .geom(Geom::point())
            .geom(Geom::vline(1.5))
            .build()
            .expect("operation should succeed");

        let fb = plot.to_framebuffer().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_ggplot_square_points() {
        let plot = GGPlot::new()
            .data_xy(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0])
            .geom(Geom::point().shape(PointShape::Square))
            .build()
            .expect("operation should succeed");

        let fb = plot.to_framebuffer().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_ggplot_triangle_points() {
        // Other shapes fallback to circle
        let plot = GGPlot::new()
            .data_xy(&[1.0, 2.0], &[3.0, 4.0])
            .geom(Geom::point().shape(PointShape::Triangle))
            .build()
            .expect("operation should succeed");

        let fb = plot.to_framebuffer().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_ggplot_single_point() {
        // Edge case: single point triggers range adjustment
        let plot = GGPlot::new()
            .data_xy(&[5.0], &[5.0])
            .geom(Geom::point())
            .build()
            .expect("builder should produce valid result");

        let fb = plot.to_framebuffer().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_ggplot_theme_bw() {
        // Theme with panel border
        let plot = GGPlot::new()
            .data_xy(&[1.0, 2.0], &[3.0, 4.0])
            .geom(Geom::point())
            .theme(Theme::bw())
            .build()
            .expect("operation should succeed");

        let fb = plot.to_framebuffer().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_ggplot_theme_void() {
        // Theme with no grid/axes
        let plot = GGPlot::new()
            .data_xy(&[1.0, 2.0], &[3.0, 4.0])
            .geom(Geom::point())
            .theme(Theme::void())
            .build()
            .expect("operation should succeed");

        let fb = plot.to_framebuffer().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_ggplot_layer_specific_data() {
        let layer =
            Layer::new(Geom::point()).data(DataFrame::from_xy(&[10.0, 20.0], &[30.0, 40.0]));

        let plot = GGPlot::new()
            .data_xy(&[1.0, 2.0], &[3.0, 4.0])
            .layer(layer)
            .build()
            .expect("builder should produce valid result");

        let fb = plot.to_framebuffer().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_ggplot_default() {
        let plot = GGPlot::default();
        assert!(plot.layers.is_empty());
    }

    #[test]
    fn test_layer_debug_clone() {
        let layer = Layer::new(Geom::point());
        let layer2 = layer.clone();
        let _ = format!("{layer2:?}");
    }

    #[test]
    fn test_ggplot_debug_clone() {
        let plot = GGPlot::new().data_xy(&[1.0], &[2.0]);
        let plot2 = plot.clone();
        let _ = format!("{plot2:?}");
    }

    #[test]
    fn test_built_ggplot_debug() {
        let built = GGPlot::new()
            .data_xy(&[1.0], &[2.0])
            .geom(Geom::point())
            .build()
            .expect("builder should produce valid result");
        let _ = format!("{built:?}");
    }

    #[test]
    fn test_ggplot_coord_polar() {
        // Non-cartesian coord doesn't apply limits
        let plot = GGPlot::new()
            .data_xy(&[1.0, 2.0], &[3.0, 4.0])
            .geom(Geom::point())
            .coord(Coord::polar())
            .build()
            .expect("operation should succeed");

        let fb = plot.to_framebuffer().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_ggplot_negative_values_bar() {
        // Bars with negative y values
        let plot = GGPlot::new()
            .data_xy(&[1.0, 2.0, 3.0], &[-2.0, 3.0, -1.0])
            .geom(Geom::bar())
            .build()
            .expect("operation should succeed");

        let fb = plot.to_framebuffer().expect("operation should succeed");
        assert!(fb.width() > 0);
    }
}
