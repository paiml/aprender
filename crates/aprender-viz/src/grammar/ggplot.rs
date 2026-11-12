//! Main GGPlot builder and renderer.
//!
//! Combines all Grammar of Graphics components into a complete visualization.

use crate::color::Rgba;
use crate::error::{Error, Result};
use crate::framebuffer::Framebuffer;
use crate::output::{SvgEncoder, TextAnchor};
use crate::render::{draw_circle, draw_line_aa, draw_rect, draw_rect_outline, i32_px};
use crate::scale::{LinearScale, Scale};

use super::aes::Aes;
use super::build::{
    point_defaults, point_diameter_px, BuiltLayer, BuiltLayout, PointAes, PositionScale,
};
use super::coord::{apply, apply_limits, Coord};
use super::data::{DataFrame, DataValue};
use super::facet::{panels, Facet, Panel};
use super::geom::{Geom, GeomType, PointShape};
use super::path;
use super::scale_spec::{require_scale_scope, PositionScaleSpec, ResolvedAxis};
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
    x_scale: Option<PositionScaleSpec>,
    y_scale: Option<PositionScaleSpec>,
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
            x_scale: None,
            y_scale: None,
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

    /// `scale_x_*()`: shape the x position scale (continuous, log10 or discrete).
    #[must_use]
    pub fn scale_x(mut self, spec: impl Into<PositionScaleSpec>) -> Self {
        self.x_scale = Some(spec.into());
        self
    }

    /// `scale_y_*()`: shape the y position scale (continuous, log10 or discrete).
    #[must_use]
    pub fn scale_y(mut self, spec: impl Into<PositionScaleSpec>) -> Self {
        self.y_scale = Some(spec.into());
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
            facet: self.facet,
            theme: self.theme,
            width: self.width,
            height: self.height,
            title: self.title,
            xlab: self.xlab,
            ylab: self.ylab,
            x_scale: self.x_scale,
            y_scale: self.y_scale,
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
    facet: Facet,
    theme: Theme,
    width: u32,
    height: u32,
    title: Option<String>,
    xlab: Option<String>,
    ylab: Option<String>,
    x_scale: Option<PositionScaleSpec>,
    y_scale: Option<PositionScaleSpec>,
}

/// Where each panel sits in the figure.
///
/// Panels share one scale domain (ggplot2's `scales = "fixed"`), so marks stay comparable between
/// panels; only the pixel rectangle differs.
struct PanelGrid {
    cell_w: u32,
    cell_h: u32,
    margin: u32,
}

impl PanelGrid {
    fn new(panels: &[Panel], width: u32, height: u32, margin: u32) -> Self {
        let ncol = panels.iter().map(|p| p.col + 1).max().unwrap_or(1);
        let nrow = panels.iter().map(|p| p.row + 1).max().unwrap_or(1);
        Self {
            cell_w: width / u32::try_from(ncol).unwrap_or(1).max(1),
            cell_h: height / u32::try_from(nrow).unwrap_or(1).max(1),
            margin,
        }
    }

    /// The drawable rectangle of one panel: `(x, y, w, h)`.
    fn rect(&self, p: &Panel) -> (u32, u32, u32, u32) {
        let cx = u32::try_from(p.col).unwrap_or(0).saturating_mul(self.cell_w);
        let cy = u32::try_from(p.row).unwrap_or(0).saturating_mul(self.cell_h);
        (
            cx.saturating_add(self.margin),
            cy.saturating_add(self.margin),
            self.cell_w.saturating_sub(self.margin.saturating_mul(2)),
            self.cell_h.saturating_sub(self.margin.saturating_mul(2)),
        )
    }
}

/// One panel as drawn: its pixel rectangle, its pixel scales, and the in-domain breaks of the
/// bottom (`x_ticks`) and left (`y_ticks`) axes with their labels.
struct PanelFrame {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
    x_scale: LinearScale,
    y_scale: LinearScale,
    x_ticks: Vec<(f64, String)>,
    y_ticks: Vec<(f64, String)>,
}

impl BuiltGGPlot {
    /// The plot title, if one was set.
    #[must_use]
    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    /// The x-axis label, if one was set.
    #[must_use]
    pub fn xlab(&self) -> Option<&str> {
        self.xlab.as_deref()
    }

    /// The y-axis label, if one was set.
    #[must_use]
    pub fn ylab(&self) -> Option<&str> {
        self.ylab.as_deref()
    }

    /// The coordinate system.
    #[must_use]
    pub fn coord(&self) -> &Coord {
        &self.coord
    }

    /// The faceting.
    #[must_use]
    pub fn facet(&self) -> &Facet {
        &self.facet
    }

    /// Render to framebuffer, one sub-plot per facet panel.
    ///
    /// # Errors
    ///
    /// Propagates [`Error::UnknownColumn`] if the facet names a column the data lacks, and
    /// [`Error::UnsupportedCoord`] for a `Polar` or `Fixed` coordinate system, and
    /// [`Error::Rendering`] whenever [`Self::layout`] resolves position scales (`scale_x`/`scale_y`
    /// was set, or a position column holds text): the renderer draws from its own linear scales and
    /// honours neither a position-scale spec (log10, limits, discrete levels) nor a discrete axis, so
    /// it refuses instead of drawing a plot that disagrees with `layout()`. Use `layout()` for the
    /// scaled view.
    pub fn to_framebuffer(&self) -> Result<Framebuffer> {
        if self.uses_position_scales() {
            return Err(Error::Rendering(
                "scale_x/scale_y and text (discrete) positions are not drawn by the renderer yet; \
                 use layout()"
                    .into(),
            ));
        }
        let mut fb = Framebuffer::new(self.width, self.height)?;

        // Fill background
        fb.clear(self.theme.background);

        let ps = panels(&self.facet, &self.data)?;
        let grid = PanelGrid::new(&ps, self.width, self.height, self.theme.margin);
        for p in &ps {
            self.render_panel(&mut fb, p, &grid)?;
        }

        Ok(fb)
    }

    /// Render to SVG: the framebuffer as an embedded image, with the title, x label and y label
    /// drawn as `<text>`. The framebuffer has no font, so [`Self::to_framebuffer`] carries none of
    /// the three; this is the render that does (#5334, #3590).
    ///
    /// # Errors
    ///
    /// Everything [`Self::to_framebuffer`] returns, plus PNG-encoding failures.
    pub fn to_svg(&self) -> Result<String> {
        let fb = self.to_framebuffer()?;
        let mut svg = SvgEncoder::from_framebuffer(&fb)?;
        let (w, h) = (self.width as f32, self.height as f32);
        let m = self.theme.margin as f32;
        let ink = self.theme.axis_color;
        if let Some(t) = &self.title {
            svg = svg.text_anchored(w / 2.0, (m / 2.0).max(14.0), t, 16.0, ink, TextAnchor::Middle);
        }
        if let Some(x) = &self.xlab {
            svg = svg.text_anchored(
                w / 2.0,
                h - (m / 4.0).max(4.0),
                x,
                12.0,
                ink,
                TextAnchor::Middle,
            );
        }
        if let Some(y) = &self.ylab {
            svg = svg.text_anchored(m, (m * 0.9).max(12.0), y, 12.0, ink, TextAnchor::Start);
        }
        if self.theme.show_axis {
            svg = self.svg_axis_text(svg)?;
        }
        Ok(svg.render())
    }

    /// Tick marks and break labels of every panel, at the breaks of the layout's position
    /// scales (ggplot2's `axis.ticks` + `axis.text`; PLOT-GG-001, #6909).
    fn svg_axis_text(&self, mut svg: SvgEncoder) -> Result<SvgEncoder> {
        const TICK_PX: f32 = 4.0;
        const AXIS_TEXT_PX: f32 = 10.0;
        let ink = self.theme.axis_color;
        let ps = panels(&self.facet, &self.data)?;
        let grid = PanelGrid::new(&ps, self.width, self.height, self.theme.margin);
        for p in &ps {
            let f = self.panel_frame(p, &grid)?;
            let bottom = (f.y + f.h) as f32;
            let left = f.x as f32;
            // A tick whose pixel is not finite (a domain wider than f32) is not written: SVG
            // attributes never carry NaN or inf.
            for (v, label) in &f.x_ticks {
                let Some(px) = Self::pixel(&f.x_scale, *v as f32) else { continue };
                svg = svg.line(px, bottom, px, bottom + TICK_PX, ink, 1.0).text_anchored(
                    px,
                    bottom + TICK_PX + AXIS_TEXT_PX,
                    label,
                    AXIS_TEXT_PX,
                    ink,
                    TextAnchor::Middle,
                );
            }
            for (v, label) in &f.y_ticks {
                let Some(py) = Self::pixel(&f.y_scale, *v as f32) else { continue };
                svg = svg.line(left - TICK_PX, py, left, py, ink, 1.0).text_anchored(
                    left - TICK_PX - 2.0,
                    py + AXIS_TEXT_PX * 0.35,
                    label,
                    AXIS_TEXT_PX,
                    ink,
                    TextAnchor::End,
                );
            }
        }
        Ok(svg)
    }

    /// One panel's pixel rectangle, its pixel scales, and the breaks (with labels) that fall
    /// inside its domain. `coord_flip()` swaps the scales, so the bottom axis carries y's breaks.
    fn panel_frame(&self, p: &Panel, grid: &PanelGrid) -> Result<PanelFrame> {
        let (x, y, w, h) = grid.rect(p);

        // Panels share one domain, so marks stay comparable between them.
        let (xs, ys) = self.position_scales();
        let ((x_min, x_max), (y_min, y_max)) =
            apply_limits(&self.coord, xs.continuous_range, ys.continuous_range)?;

        // The scale pipeline below is still f32; the coordinate surface above is f64 (APEX-2c).
        // A domain that is not finite as f32 (no data, a NaN or infinite limit, or one past
        // f32::MAX) maps no value to a finite pixel: [`Self::pixel`] then yields `None` and every
        // mark and tick is skipped, so nothing non-finite is drawn or written.
        let (right, bottom) = (x as f32 + w as f32, y as f32 + h as f32);
        let x_scale = LinearScale::new((x_min as f32, x_max as f32), (x as f32, right))?;
        let y_scale = LinearScale::new(
            (y_min as f32, y_max as f32),
            (bottom, y as f32), // Inverted for screen coords
        )?;

        let flip = matches!(self.coord, Coord::Cartesian { flip: true, .. });
        let (bx, by) = if flip { (&ys, &xs) } else { (&xs, &ys) };
        let inside = |s: &PositionScale, (lo, hi): (f64, f64)| -> Vec<(f64, String)> {
            s.breaks
                .iter()
                .zip(&s.labels)
                .filter(|(b, _)| **b >= lo.min(hi) && **b <= lo.max(hi))
                .map(|(b, l)| (*b, l.clone()))
                .collect()
        };
        Ok(PanelFrame {
            x,
            y,
            w,
            h,
            x_ticks: inside(bx, (x_min, x_max)),
            y_ticks: inside(by, (y_min, y_max)),
            x_scale,
            y_scale,
        })
    }

    /// Draw one panel: background, scales, grid, layers, axes, border.
    fn render_panel(&self, fb: &mut Framebuffer, p: &Panel, grid: &PanelGrid) -> Result<()> {
        let f = self.panel_frame(p, grid)?;
        let (plot_x, plot_y, plot_w, plot_h) = (f.x, f.y, f.w, f.h);
        let (x_scale, y_scale) = (&f.x_scale, &f.y_scale);

        draw_rect(fb, i32_px(plot_x), i32_px(plot_y), plot_w, plot_h, self.theme.panel_background);

        if self.theme.show_grid {
            self.draw_grid(fb, &f);
        }

        for layer in &self.layers {
            self.render_layer(fb, layer, p, x_scale, y_scale)?;
        }

        if self.theme.show_axis {
            self.draw_axes(fb, plot_x, plot_y, plot_w, plot_h);
        }

        if self.theme.show_panel_border {
            draw_rect_outline(
                fb,
                i32_px(plot_x),
                i32_px(plot_y),
                plot_w,
                plot_h,
                self.theme.axis_color,
                1, // thickness
            );
        }

        Ok(())
    }

    /// The rows of `data` that belong to panel `p`.
    ///
    /// A layer carrying its own frame is faceted by the same variables as the plot, so it is
    /// partitioned in its own right and matched to the panel by level — a layer is not replicated
    /// whole into every panel.
    fn panel_rows(&self, data: &DataFrame, p: &Panel) -> Result<Vec<usize>> {
        if std::ptr::eq(data, &self.data) {
            return Ok(p.rows.clone());
        }
        let own = panels(&self.facet, data)?;
        Ok(own.into_iter().find(|q| q.levels == p.levels).map_or_else(Vec::new, |q| q.rows))
    }

    /// The panel's points, in coordinate space.
    ///
    /// Reads both columns row-aligned, keeps only rows where *both* are finite numbers (a NaN or
    /// infinite row is dropped, for every geom), and pushes each
    /// surviving pair through [`apply`] — so the coordinate transform happens before any geom sees
    /// a point, and `x[i]`/`y[i]` are guaranteed to come from the same record.
    ///
    /// The finite filter is load-bearing, not a second guard: [`BuiltGGPlot::pixel`] would skip a
    /// non-finite mark, but a bar's width is `range / n`, a polyline joins its neighbours and an
    /// area needs two points, all counted on the rows kept here. Removing it changes the bars
    /// (test `dropped_rows_change_nothing_for_every_geom`).
    fn panel_points(
        &self,
        data: &DataFrame,
        aes: &Aes,
        rows: &[usize],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let xs = data.get_f32(aes.x.as_deref().unwrap_or("x")).unwrap_or_default();
        let ys = data.get_f32(aes.y.as_deref().unwrap_or("y")).unwrap_or_default();

        let mut xo = Vec::with_capacity(rows.len());
        let mut yo = Vec::with_capacity(rows.len());
        for &i in rows {
            let (Some(Some(x)), Some(Some(y))) = (xs.get(i).copied(), ys.get(i).copied()) else {
                continue;
            };
            // A NaN or infinite coordinate has no pixel: the row is dropped for every geom
            // (ggplot2 drops it too, "Removed n rows containing non-finite values").
            if !(x.is_finite() && y.is_finite()) {
                continue;
            }
            let (tx, ty) = apply(&self.coord, f64::from(x), f64::from(y))?;
            xo.push(tx as f32);
            yo.push(ty as f32);
        }
        Ok((xo, yo))
    }

    /// Widen `acc` to cover the finite values of one column.
    ///
    /// A column that is unmapped or absent leaves the accumulator untouched, which is how a layer
    /// that sets only `x` contributes nothing to the `y` range. Alignment is irrelevant here — only
    /// the extremes matter — so this reads the compacting accessor deliberately.
    fn extend_range(acc: (f32, f32), data: &DataFrame, col: Option<&String>) -> (f32, f32) {
        let Some(col) = col else { return acc };
        let Some(values) = data.get_f32_present(col) else { return acc };
        values.iter().filter(|v| v.is_finite()).fold(acc, |(lo, hi), &v| (lo.min(v), hi.max(v)))
    }

    /// The x and y position scales, trained on every layer's data (ggplot2's scale training).
    fn position_scales(&self) -> (PositionScale, PositionScale) {
        let (x, y) = self.trained_domains();
        (PositionScale::continuous(x), PositionScale::continuous(y))
    }

    /// The finite x and y domains every layer trains (`lo > hi` on an axis nothing trained).
    fn trained_domains(&self) -> ((f64, f64), (f64, f64)) {
        let mut x = (f32::MAX, f32::MIN);
        let mut y = (f32::MAX, f32::MIN);

        for layer in &self.layers {
            // Reference lines do not inherit x/y; hline trains y and vline trains x with the
            // intercept, abline trains neither (ggplot2: `inherit.aes = FALSE`).
            match &layer.geom.geom_type {
                GeomType::Hline { yintercept } if yintercept.is_finite() => {
                    y = (y.0.min(*yintercept), y.1.max(*yintercept));
                    continue;
                }
                GeomType::Vline { xintercept } if xintercept.is_finite() => {
                    x = (x.0.min(*xintercept), x.1.max(*xintercept));
                    continue;
                }
                GeomType::Hline { .. } | GeomType::Vline { .. } | GeomType::Abline { .. } => {
                    continue
                }
                _ => {}
            }
            let data = layer.data.as_ref().unwrap_or(&self.data);
            let layer_aes = self.aes.merge(&layer.aes);
            x = Self::extend_range(x, data, layer_aes.x.as_ref());
            y = Self::extend_range(y, data, layer_aes.y.as_ref());
        }

        let wide = |(lo, hi): (f32, f32)| (f64::from(lo), f64::from(hi));
        (wide(x), wide(y))
    }

    /// The columns that feed one axis, one per layer (`x_axis` selects x or y).
    fn axis_columns(&self, x_axis: bool) -> Vec<&[DataValue]> {
        self.layers
            .iter()
            .filter_map(|layer| {
                let data = layer.data.as_ref().unwrap_or(&self.data);
                let aes = self.aes.merge(&layer.aes);
                let name = if x_axis { aes.x } else { aes.y };
                data.get(name.as_deref().unwrap_or(if x_axis { "x" } else { "y" }))
            })
            .collect()
    }

    /// The resolved x and y axes, trained on every layer's data and shaped by `scale_x`/`scale_y`.
    fn resolved_axes(&self) -> Result<(ResolvedAxis, ResolvedAxis)> {
        Ok((
            ResolvedAxis::resolve(self.x_scale.as_ref(), &self.axis_columns(true))?,
            ResolvedAxis::resolve(self.y_scale.as_ref(), &self.axis_columns(false))?,
        ))
    }

    /// True when [`Self::layout`] resolves position scales (PLOT-GG-001 rows 2-4): `scale_x` or
    /// `scale_y` was set, or a position column of a `geom_point` layer holds text (ggplot2's
    /// default discrete scale). Otherwise `layout()` is row 1's linear path, where a text position
    /// in a line, path, step or reference layer stays that path's typed error.
    pub(super) fn uses_position_scales(&self) -> bool {
        self.x_scale.is_some()
            || self.y_scale.is_some()
            || self.layers.iter().any(|layer| {
                if !matches!(layer.geom.geom_type, GeomType::Point { .. }) {
                    return false;
                }
                let data = layer.data.as_ref().unwrap_or(&self.data);
                let aes = self.aes.merge(&layer.aes);
                let holds_text = |name: Option<&str>, default: &str| {
                    data.get(name.unwrap_or(default))
                        .is_some_and(|c| c.iter().any(|v| v.as_str().is_some()))
                };
                holds_text(aes.x.as_deref(), "x") || holds_text(aes.y.as_deref(), "y")
            })
    }

    /// The resolved plot, as ggplot2's `ggplot_build()` computes it: each layer's positions and
    /// aesthetics, and the trained, expanded x/y scales with breaks and labels.
    ///
    /// With a position scale (`scale_x`/`scale_y`, or a text position column), layer positions are
    /// in scale space (`log10` values on a log axis, level positions on a discrete one) and every
    /// row is kept (a position the scale cannot place is `NaN`, ggplot2's `NA`); the renderer does
    /// not draw such a plot (see [`Self::to_framebuffer`]). Without one, rows with a non-finite
    /// position are dropped and the renderer draws from these same scales.
    ///
    /// # Errors
    ///
    /// Returns an error for a path, step or reference layer (`geom_line`, `geom_path`, `geom_step`, `geom_hline`,
    /// `geom_vline`, `geom_abline`) whose input is outside what `grammar/path.rs` documents:
    /// text or infinite positions, a missing column, a data-mapped colour, an unsupported facet, or
    /// a degenerate axis range; and [`Error::UnsupportedCoord`] for such a layer under any
    /// coordinate system but the default `coord_cartesian()` (`layout()` applies none). While such
    /// a layer is present, any layer with a stat other than identity is an
    /// [`Error::UnsupportedStat`] (`layout()` computes none), and any other geom than these six
    /// and `geom_point` is an [`Error::UnsupportedLayer`].
    ///
    /// With a position scale, the plot must be `geom_point` layers with the identity stat under
    /// the default `coord_cartesian()` and no facet ([`Error::UnsupportedLayer`],
    /// [`Error::UnsupportedStat`], [`Error::UnsupportedCoord`] otherwise), and
    /// [`Error::ScaleDomain`] is returned for text in a continuous scale, or a continuous axis with
    /// no finite value to train on (ggplot2 would return an infinite range there).
    pub fn layout(&self) -> Result<BuiltLayout> {
        path::require_default_coord(&self.layers, &self.coord)?;
        path::require_identity_stats(&self.layers)?;
        if self.uses_position_scales() {
            return self.scaled_layout();
        }
        let layers = self.layers.iter().map(|l| self.build_layer(l)).collect::<Result<Vec<_>>>()?;
        let (x, y) = self.trained_domains();
        path::require_trained(&self.layers, x, y)?;
        Ok(BuiltLayout { layers, x: PositionScale::continuous(x), y: PositionScale::continuous(y) })
    }

    /// [`Self::layout`] through the resolved position scales (rows 2-4).
    fn scaled_layout(&self) -> Result<BuiltLayout> {
        require_scale_scope(&self.layers, &self.coord, &self.facet)?;
        let (x_axis, y_axis) = self.resolved_axes()?;
        let layers = self
            .layers
            .iter()
            .map(|l| self.scaled_layer(l, &x_axis, &y_axis))
            .collect::<Result<_>>()?;
        Ok(BuiltLayout { layers, x: x_axis.scale, y: y_axis.scale })
    }

    /// One `geom_point` layer through the resolved scales: one entry per row of its data (a cell
    /// past the end of a short column is missing), a position the scale cannot place is `NaN`
    /// (ggplot2's `NA`).
    ///
    /// # Errors
    ///
    /// [`Error::UnknownColumn`] when a position column is absent (R: "object not found"), and the
    /// axis's own error for a value it cannot hold ([`ResolvedAxis::position`]).
    fn scaled_layer(
        &self,
        layer: &Layer,
        x_axis: &ResolvedAxis,
        y_axis: &ResolvedAxis,
    ) -> Result<BuiltLayer> {
        let data = layer.data.as_ref().unwrap_or(&self.data);
        let aes = self.aes.merge(&layer.aes);
        let column = |name: Option<&str>, default: &str| {
            let name = name.unwrap_or(default);
            data.get(name)
                .ok_or_else(|| Error::UnknownColumn(format!("no position column {name:?}")))
        };
        let (xs, ys) = (column(aes.x.as_deref(), "x")?, column(aes.y.as_deref(), "y")?);
        const MISSING: DataValue = DataValue::Null;
        let mut x = Vec::with_capacity(data.nrow());
        let mut y = Vec::with_capacity(data.nrow());
        for i in 0..data.nrow() {
            x.push(x_axis.position(xs.get(i).unwrap_or(&MISSING))?);
            y.push(y_axis.position(ys.get(i).unwrap_or(&MISSING))?);
        }
        let points = Self::point_aes(layer, &aes, x.len());
        Ok(BuiltLayer { x, y, points, path: None })
    }

    /// One layer's built data: rows where both positions are finite numbers, plus point aesthetics.
    fn build_layer(&self, layer: &Layer) -> Result<BuiltLayer> {
        if path::is_path_family(&layer.geom.geom_type) {
            return path::build_layer(layer, &self.aes, &self.data, &self.facet);
        }
        let data = layer.data.as_ref().unwrap_or(&self.data);
        let aes = self.aes.merge(&layer.aes);
        let xs = data.get_f32(aes.x.as_deref().unwrap_or("x")).unwrap_or_default();
        let ys = data.get_f32(aes.y.as_deref().unwrap_or("y")).unwrap_or_default();
        let (x, y): (Vec<f64>, Vec<f64>) = xs
            .iter()
            .zip(&ys)
            .filter_map(|(a, b)| Some((f64::from((*a)?), f64::from((*b)?))))
            .filter(|(a, b)| a.is_finite() && b.is_finite())
            .unzip();
        let points = Self::point_aes(layer, &aes, x.len());
        Ok(BuiltLayer { x, y, points, path: None })
    }

    /// `n` copies of a point layer's aesthetics (`None` for any other geom).
    fn point_aes(layer: &Layer, aes: &Aes, n: usize) -> Option<Vec<PointAes>> {
        match layer.geom.geom_type {
            GeomType::Point { shape } => {
                let p = PointAes {
                    shape: shape.pch(),
                    colour: aes.color_value.unwrap_or(point_defaults::COLOUR),
                    fill: None,
                    size: aes.size_value.map_or(point_defaults::SIZE_MM, f64::from),
                    alpha: aes.alpha_value.map(f64::from),
                    stroke: point_defaults::STROKE_MM,
                };
                Some(vec![p; n])
            }
            _ => None,
        }
    }

    /// Draw the major grid lines, one per break of each position scale (ggplot2's
    /// `panel.grid.major`), so the grid lines up with the break labels [`Self::to_svg`] draws.
    fn draw_grid(&self, fb: &mut Framebuffer, f: &PanelFrame) {
        let color = self.theme.grid_color;
        let (left, right) = (f.x as f32, (f.x + f.w) as f32);
        let (top, bottom) = (f.y as f32, (f.y + f.h) as f32);

        for (v, _) in &f.y_ticks {
            let y_px = f.y_scale.scale(*v as f32);
            draw_line_aa(fb, left, y_px, right, y_px, color);
        }
        for (v, _) in &f.x_ticks {
            let x_px = f.x_scale.scale(*v as f32);
            draw_line_aa(fb, x_px, top, x_px, bottom, color);
        }
    }

    /// Draw axes.
    fn draw_axes(&self, fb: &mut Framebuffer, plot_x: u32, plot_y: u32, plot_w: u32, plot_h: u32) {
        let color = self.theme.axis_color;

        // X axis (bottom)
        draw_line_aa(
            fb,
            plot_x as f32,
            plot_y as f32 + plot_h as f32,
            plot_x as f32 + plot_w as f32,
            plot_y as f32 + plot_h as f32,
            color,
        );

        // Y axis (left)
        draw_line_aa(
            fb,
            plot_x as f32,
            plot_y as f32,
            plot_x as f32,
            plot_y as f32 + plot_h as f32,
            color,
        );
    }

    /// Render a single layer.
    fn render_layer(
        &self,
        fb: &mut Framebuffer,
        layer: &Layer,
        panel: &Panel,
        x_scale: &LinearScale,
        y_scale: &LinearScale,
    ) -> Result<()> {
        let data = layer.data.as_ref().unwrap_or(&self.data);
        let aes = self.aes.merge(&layer.aes);

        let rows = self.panel_rows(data, panel)?;
        let (x_data, y_data) = self.panel_points(data, &aes, &rows)?;

        let n = x_data.len().min(y_data.len());
        if n == 0 {
            return Ok(());
        }

        // Get style from aesthetics
        let color = aes.color_value.unwrap_or(Rgba::new(66, 133, 244, 255));

        match &layer.geom.geom_type {
            GeomType::Point { shape } => {
                // ggplot2's GeomPoint defaults: black, size 1.5 mm, stroke 0.5 mm.
                let colour = aes.color_value.unwrap_or(point_defaults::COLOUR);
                let size_mm = aes.size_value.map_or(point_defaults::SIZE_MM, f64::from);
                let diameter = point_diameter_px(size_mm, point_defaults::STROKE_MM) as f32;
                Self::render_points(
                    fb, &x_data, &y_data, x_scale, y_scale, colour, diameter, *shape,
                );
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

        Ok(())
    }

    /// The pixel `v` maps to, or `None` when the scale gives no finite pixel (the domain is
    /// wider than `f32`, or `v` overflows it). A mark with no pixel is skipped, never drawn at an
    /// arbitrary one.
    fn pixel(scale: &LinearScale, v: f32) -> Option<f32> {
        let p = scale.scale(v);
        p.is_finite().then_some(p)
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
            let (Some(x), Some(y)) =
                (Self::pixel(x_scale, x_data[i]), Self::pixel(y_scale, y_data[i]))
            else {
                continue;
            };
            let r = (size / 2.0) as i32;

            match shape {
                PointShape::Circle => {
                    draw_circle(fb, x as i32, y as i32, r, color);
                }
                PointShape::Square => {
                    // Saturating: `x as i32 - r` and `2 * r` overflow for an extreme size.
                    let side = (r.max(0) as u32).saturating_mul(2);
                    draw_rect(
                        fb,
                        (x as i32).saturating_sub(r),
                        (y as i32).saturating_sub(r),
                        side,
                        side,
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
            let (Some(x0), Some(y0), Some(x1), Some(y1)) = (
                Self::pixel(x_scale, x_data[i]),
                Self::pixel(y_scale, y_data[i]),
                Self::pixel(x_scale, x_data[i + 1]),
                Self::pixel(y_scale, y_data[i + 1]),
            ) else {
                continue;
            };

            // `draw_line_aa` clips the segment to the canvas before it walks a pixel.
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
        let Some(baseline) = Self::pixel(y_scale, 0.0) else {
            return;
        };

        for i in 0..n {
            let (Some(x), Some(y)) =
                (Self::pixel(x_scale, x_data[i]), Self::pixel(y_scale, y_data[i]))
            else {
                continue;
            };

            let left = (x - bar_px_width as f32 / 2.0) as i32;
            let top = y.min(baseline) as i32;
            let height = (y - baseline).abs() as u32;

            // `draw_rect` clips in i64, so an extreme `left`/`top`/`height` cannot overflow.
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

        let Some(baseline) = Self::pixel(y_scale, 0.0) else {
            return;
        };
        let last_row = i32_px(fb.height()) - 1;

        // Draw vertical slices for area fill
        for i in 0..n {
            let (Some(xp), Some(y)) =
                (Self::pixel(x_scale, x_data[i]), Self::pixel(y_scale, y_data[i]))
            else {
                continue;
            };
            let x = xp as i32;
            if x < 0 || (x as u32) >= fb.width() {
                continue;
            }
            // Only the rows on the canvas are walked: the slice's own extent can be 2^31 rows.
            let y_top = (y.min(baseline) as i32).max(0);
            let y_bot = (y.max(baseline) as i32).min(last_row);

            for py in y_top..=y_bot {
                fb.blend_pixel(x as u32, py as u32, color);
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
    fn test_ggplot_facet_unknown_column_refuses() {
        // This is the ORIGINAL test_ggplot_facet, which faceted by "category" on a frame built
        // from `data_xy` — a frame whose only columns are "x" and "y". It asserted `width > 0`
        // and passed, because the facet was dropped on the floor by `build()`. Now the missing
        // column is an error, and that is the whole point of the row.
        let plot = GGPlot::new()
            .data_xy(&[1.0, 2.0], &[3.0, 4.0])
            .geom(Geom::point())
            .facet(Facet::wrap("category", 2))
            .build()
            .expect("operation should succeed");

        assert!(matches!(plot.to_framebuffer(), Err(Error::UnknownColumn(_))));
    }

    #[test]
    fn test_ggplot_facet_renders_one_panel_per_level() {
        let mut df = DataFrame::new();
        df.add_column_f32("x", &[1.0, 2.0, 3.0, 4.0]);
        df.add_column_f32("y", &[1.0, 2.0, 3.0, 4.0]);
        df.add_column_str("category", &["b", "a", "b", "c"]);

        let plot = GGPlot::new()
            .data(df)
            .aes(Aes::new().x("x").y("y"))
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

    /// FALSIFY-VIZ-LABELS-001: a label the caller sets is in the rendered SVG, and a plot
    /// without one has none (so the test cannot pass on a constant).
    #[test]
    fn test_ggplot_svg_draws_title_and_axis_labels() {
        let build = |labelled: bool| {
            let mut g = GGPlot::new().data_xy(&[1.0, 2.0], &[3.0, 4.0]).geom(Geom::point());
            if labelled {
                g = g.title("My Plot").xlab("X Axis").ylab("Y Axis");
            }
            g.build().expect("operation should succeed").to_svg().expect("operation should succeed")
        };
        let svg = build(true);
        for label in ["My Plot", "X Axis", "Y Axis"] {
            assert!(svg.contains(&format!(">{label}</text>")), "{label} not drawn");
        }
        let bare = build(false);
        for label in ["My Plot", "X Axis", "Y Axis"] {
            assert!(!bare.contains(&format!(">{label}</text>")), "an unlabelled plot drew {label}");
        }
    }

    /// PLOT-GG-001: the SVG carries one tick label per in-range break of each position scale,
    /// the same strings as the layout (and so as ggplot2's oracle for row 1), and a theme
    /// without axes carries none.
    #[test]
    fn test_ggplot_svg_draws_break_labels_from_the_layout() {
        let plot = |theme: Theme| {
            GGPlot::new()
                .data_xy(&[1.5, 13.75], &[-2.25, 7.5])
                .geom(Geom::point())
                .theme(theme)
                .build()
                .expect("plot builds")
        };
        let built = plot(Theme::grey());
        let layout = built.layout().expect("layout");
        let svg = built.to_svg().expect("svg renders");
        let texts = svg.matches("</text>").count();
        assert_eq!(texts, layout.x.labels.len() + layout.y.labels.len(), "{svg}");
        for label in layout.x.labels.iter().chain(&layout.y.labels) {
            assert!(svg.contains(&format!(">{label}</text>")), "break label {label} missing");
        }
        assert_eq!(plot(Theme::void()).to_svg().expect("svg").matches("</text>").count(), 0);
    }

    /// `coord_flip()` puts y's breaks on the bottom axis.
    #[test]
    fn test_ggplot_svg_flip_swaps_break_labels() {
        let built = GGPlot::new()
            .data_xy(&[1.5, 13.75], &[-2.25, 7.5])
            .geom(Geom::point())
            .coord(Coord::cartesian().flip())
            .build()
            .expect("plot builds");
        let svg = built.to_svg().expect("svg renders");
        assert!(svg.contains(">-2.5</text>") && svg.contains(">10</text>"), "{svg}");
    }

    /// A zero-width zoom on either axis has no pixel scale: `panel_frame` returns the scale's
    /// error for x and for y instead of dividing by zero.
    #[test]
    fn test_ggplot_degenerate_limits_are_scale_errors() {
        for coord in [Coord::cartesian().xlim(3.0, 3.0), Coord::cartesian().ylim(3.0, 3.0)] {
            let built = GGPlot::new()
                .data_xy(&[1.5, 13.75], &[-2.25, 7.5])
                .geom(Geom::point())
                .coord(coord.clone())
                .build()
                .expect("plot builds");
            assert!(matches!(built.to_svg(), Err(Error::ScaleDomain(_))), "{coord:?}");
        }
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

    /// Bars drawn through `render_bars` on a 400x300 canvas, with the picture each must make
    /// computed from the scale alone (paiml/aprender#6949, F1).
    ///
    /// The x scale maps `x = 1 ..= n` onto pixels 40..371 and the y scale maps `0 ..= 4` onto
    /// pixels 280..20 (65 px per unit), so every pixel below is exactly representable in `f32`.
    /// A bar is its slot (`331 / n` px) times the bar width, cut to whole pixels, centred on the
    /// pixel of its x; it runs from the baseline (`y = 0`, pixel 280) to the pixel of its y.
    /// A bar that starts left of the canvas is clipped there: its far edge stays where the scale
    /// puts it, and it is not shifted right onto the canvas.
    fn check_bars(ys: &[f32], width: f32) {
        let n = ys.len();
        let xs: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let x_scale = LinearScale::new((1.0, n as f32), (40.0, 371.0)).expect("scale");
        let y_scale = LinearScale::new((0.0, 4.0), (280.0, 20.0)).expect("scale");
        let mut fb = Framebuffer::new(400, 300).expect("fb");
        fb.clear(Rgba::WHITE);
        let ink = Rgba::new(10, 20, 30, 255);
        BuiltGGPlot::render_bars(&mut fb, &xs, ys, &x_scale, &y_scale, ink, width);

        let slot = 331.0_f64 / n as f64;
        let bar_w = (slot * f64::from(width)).max(1.0).floor() as i64;
        let mut rects = Vec::new();
        for (i, &y) in ys.iter().enumerate() {
            let centre = 40.0 + 331.0 * i as f64 / (n - 1) as f64;
            let left = (centre - bar_w as f64 / 2.0).trunc() as i64;
            let (top_px, bottom_px) = (280.0 - 65.0 * f64::from(y), 280.0);
            let top = top_px.min(bottom_px).trunc() as i64;
            let height = ((top_px - bottom_px).abs().trunc() as i64).max(1);
            rects.push((left, left + bar_w, top, top + height));
        }
        // The case is built to start left of the canvas, which is what the bug was about.
        assert!(rects[0].0 < 0, "case does not put the first bar's left edge off canvas");

        for y in 0..300_i64 {
            for x in 0..400_i64 {
                let want =
                    rects.iter().any(|&(l, r, t, b)| (l..r).contains(&x) && (t..b).contains(&y));
                let got = fb.get_pixel(x as u32, y as u32) == Some(ink);
                assert_eq!(got, want, "n={n} width={width}: pixel ({x}, {y}); bars {rects:?}");
            }
        }
    }

    #[test]
    fn two_bars_keep_their_far_edge_when_the_left_edge_is_off_canvas() {
        // 331 / 2 * 0.9 = 148 px wide, centred on 40: columns -34..114 clip to 0..114.
        check_bars(&[4.0, 3.0], 0.9);
    }

    #[test]
    fn three_bars_keep_their_far_edge_when_the_left_edge_is_off_canvas() {
        // 331 / 3 * 0.9 = 99 px wide, centred on 40: columns -9..90 clip to 0..90.
        check_bars(&[4.0, 2.0, 3.0], 0.9);
    }

    #[test]
    fn wide_bars_keep_their_far_edge_when_the_left_edge_is_off_canvas() {
        // Width 2.0: 331 px wide, centred on 40: columns -125..206 clip to 0..206.
        check_bars(&[4.0, 3.0], 2.0);
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
        // A polar coord has no transform yet, so rendering REFUSES. It used to fall through a
        // catch-all arm and render Cartesian output, which is indistinguishable from a polar plot
        // that works.
        let plot = GGPlot::new()
            .data_xy(&[1.0, 2.0], &[3.0, 4.0])
            .geom(Geom::point())
            .coord(Coord::polar())
            .build()
            .expect("operation should succeed");

        assert!(matches!(plot.to_framebuffer(), Err(Error::UnsupportedCoord(_))));
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
