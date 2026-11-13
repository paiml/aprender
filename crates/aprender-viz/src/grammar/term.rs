//! Terminal render target for a built plot (PLOT-GG-001, paiml/aprender#6909).
//!
//! Draws a [`BuiltGGPlot`] as a grid of character cells: title, y-axis label, y break labels and
//! ticks, the panel with one glyph per point, the x axis with ticks, x break labels, x-axis
//! label. Every position, break and label comes from [`BuiltGGPlot::layout`], the same numbers the
//! ggplot2 oracle checks, so the terminal shows the breaks ggplot2 would label.
//!
//! The output is a pure function of the plot and [`TerminalPlot`]: the same input and size give
//! byte-identical text. Without colour the text carries no escape sequences at all.
//!
//! Supported today: point, line, path, step and reference layers, `coord_cartesian()` (with limits, without flip), no facets, no
//! `scale_x`/`scale_y` and no text position.
//! Anything else is refused with an error naming what is missing; nothing falls back silently.

use crate::error::{Error, Result};

use super::build::{BuiltLayer, BuiltLayout};
use super::coord::{apply_limits, Coord};
use super::facet::Facet;
use super::ggplot::BuiltGGPlot;
use super::path::BuiltPath;
use super::stroke;

/// Smallest canvas, in character cells, the terminal target draws on.
pub const MIN_TERMINAL_WIDTH: u32 = 20;
/// Smallest canvas height, in lines.
pub const MIN_TERMINAL_HEIGHT: u32 = 8;
/// Largest canvas, in character cells. The canvas holds one cell per column and line, so the
/// size is bounded before anything is allocated (100000x100000 was an 80 GB allocation).
pub const MAX_TERMINAL_WIDTH: u32 = 1000;
/// Largest canvas height, in lines.
pub const MAX_TERMINAL_HEIGHT: u32 = 1000;
/// The glyph a point is drawn with (ggplot2's default `shape = 19`, a solid circle).
pub const POINT_GLYPH: char = '●';
/// The glyph a line, path, step or reference line is drawn with, one per cell it crosses.
pub const LINE_GLYPH: char = '·';
/// Fewest columns the panel may have; a narrower one is refused.
pub const MIN_PANEL_WIDTH: usize = 8;

/// Size and colour of a terminal render.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerminalPlot {
    /// Width in character cells.
    pub width: u32,
    /// Height in lines; the output has exactly this many lines.
    pub height: u32,
    /// Emit ANSI SGR colour sequences.
    pub color: bool,
}

/// What a cell is part of, which decides its colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ink {
    Plain,
    Title,
    Axis,
    /// A point of this RGB colour.
    Point(u8, u8, u8),
}

impl Ink {
    /// The SGR sequence that starts this ink (empty for plain text).
    fn sgr(self) -> String {
        match self {
            Ink::Plain => String::new(),
            Ink::Title => "\x1b[1m".into(),
            Ink::Axis => "\x1b[90m".into(),
            // Black (ggplot2's default point colour) would vanish on a dark terminal: draw it in
            // the terminal's own foreground, bold.
            Ink::Point(0, 0, 0) => "\x1b[1m".into(),
            Ink::Point(r, g, b) => format!("\x1b[38;2;{r};{g};{b}m"),
        }
    }
}

/// A fixed-size grid of cells. Writes outside the grid are dropped, which is how long text is
/// clipped at the edge.
struct Canvas {
    w: usize,
    h: usize,
    cells: Vec<(char, Ink)>,
}

impl Canvas {
    fn new(w: usize, h: usize) -> Self {
        Self { w, h, cells: vec![(' ', Ink::Plain); w * h] }
    }

    fn put(&mut self, col: usize, row: usize, ch: char, ink: Ink) {
        if col < self.w && row < self.h {
            self.cells[row * self.w + col] = (ch, ink);
        }
    }

    fn text(&mut self, col: usize, row: usize, s: &str, ink: Ink) {
        for (i, ch) in s.chars().enumerate() {
            self.put(col + i, row, ch, ink);
        }
    }

    /// Every line, trailing blanks trimmed, each ended by `\n`; SGR runs when `color`.
    fn render(&self, color: bool) -> String {
        let mut out = String::new();
        for row in self.cells.chunks(self.w) {
            let used = row.iter().rposition(|(c, _)| *c != ' ').map_or(0, |i| i + 1);
            let mut ink = Ink::Plain;
            for &(ch, cell_ink) in &row[..used] {
                if color && cell_ink != ink && ch != ' ' {
                    if ink != Ink::Plain {
                        out.push_str("\x1b[0m");
                    }
                    out.push_str(&cell_ink.sgr());
                    ink = cell_ink;
                }
                out.push(ch);
            }
            if ink != Ink::Plain {
                out.push_str("\x1b[0m");
            }
            out.push('\n');
        }
        out
    }
}

/// Map `v` in `[lo, hi]` to a cell index in `0..n` (`n >= 1`).
fn cell(v: f64, (lo, hi): (f64, f64), n: usize) -> usize {
    let t = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
    (t * (n - 1) as f64).round() as usize
}

fn in_domain(v: f64, (lo, hi): (f64, f64)) -> bool {
    v >= lo && v <= hi
}

/// Refuse a canvas outside [`MIN_TERMINAL_WIDTH`] x [`MIN_TERMINAL_HEIGHT`] ..
/// [`MAX_TERMINAL_WIDTH`] x [`MAX_TERMINAL_HEIGHT`], before anything is allocated.
fn check_canvas(opts: TerminalPlot) -> Result<()> {
    if opts.width < MIN_TERMINAL_WIDTH || opts.height < MIN_TERMINAL_HEIGHT {
        return Err(Error::Rendering(format!(
            "terminal plot needs at least {MIN_TERMINAL_WIDTH}x{MIN_TERMINAL_HEIGHT} cells, got {}x{}",
            opts.width, opts.height
        )));
    }
    if opts.width > MAX_TERMINAL_WIDTH || opts.height > MAX_TERMINAL_HEIGHT {
        return Err(Error::Rendering(format!(
            "terminal plot is at most {MAX_TERMINAL_WIDTH}x{MAX_TERMINAL_HEIGHT} cells, got {}x{}",
            opts.width, opts.height
        )));
    }
    Ok(())
}

/// Where the panel sits on the canvas and the data domains it maps onto.
struct Frame {
    /// First panel row.
    top: usize,
    /// Panel height in rows.
    plot_h: usize,
    /// Column of the y axis line.
    axis_col: usize,
    /// First panel column.
    plot_left: usize,
    /// Panel width in columns.
    plot_w: usize,
    /// Width of the y break labels.
    gutter: usize,
    /// Visible x domain.
    xd: (f64, f64),
    /// Visible y domain.
    yd: (f64, f64),
}

impl Frame {
    /// Size the panel under `top` header rows and `below` footer rows of a `w` x `h` canvas.
    fn new(
        (w, h): (usize, usize),
        (top, below): (usize, usize),
        gutter: usize,
        (xd, yd): ((f64, f64), (f64, f64)),
    ) -> Result<Self> {
        let axis_col = gutter + 1;
        let plot_left = axis_col + 1;
        if plot_left + MIN_PANEL_WIDTH > w {
            return Err(Error::Rendering(format!(
                "terminal plot is {w} cells wide; its y labels need {gutter}, leaving no room for the panel"
            )));
        }
        Ok(Self {
            top,
            plot_h: h - top - below,
            axis_col,
            plot_left,
            plot_w: w - plot_left,
            gutter,
            xd,
            yd,
        })
    }

    fn row_of(&self, y: f64) -> usize {
        self.top + self.plot_h - 1 - cell(y, self.yd, self.plot_h)
    }

    fn col_of(&self, x: f64) -> usize {
        self.plot_left + cell(x, self.xd, self.plot_w)
    }
}

/// The y breaks and labels inside the visible domain `yd`.
fn y_ticks(layout: &BuiltLayout, yd: (f64, f64)) -> Vec<(f64, &str)> {
    layout
        .y
        .breaks
        .iter()
        .zip(&layout.y.labels)
        .filter(|(b, _)| in_domain(**b, yd))
        .map(|(b, l)| (*b, l.as_str()))
        .collect()
}

/// Width of the widest tick label.
fn gutter_of(ticks: &[(f64, &str)]) -> usize {
    ticks.iter().map(|(_, l)| l.chars().count()).max().unwrap_or(0)
}

/// Draw one x break label centred under `col`, unless it would touch the previous one;
/// `free` is the first column not yet used.
fn put_x_label(c: &mut Canvas, (col, row): (usize, usize), label: &str, free: &mut usize) {
    let len = label.chars().count();
    let start = col.saturating_sub(len / 2).min(c.w.saturating_sub(len));
    if start >= *free {
        c.text(start, row, label, Ink::Axis);
        *free = start + len + 1;
    }
}

impl BuiltGGPlot {
    /// Render to text for a terminal of `opts.width` x `opts.height` cells.
    ///
    /// # Errors
    ///
    /// [`Error::Rendering`] for a canvas below [`MIN_TERMINAL_WIDTH`] x [`MIN_TERMINAL_HEIGHT`]
    /// or above [`MAX_TERMINAL_WIDTH`] x [`MAX_TERMINAL_HEIGHT`] (or too narrow for the y
    /// labels), a facet, `coord_flip()`, or a layer that is not a point
    /// layer; [`Error::UnsupportedCoord`] for polar or fixed coordinates.
    pub fn to_terminal(&self, opts: TerminalPlot) -> Result<String> {
        check_canvas(opts)?;
        self.check_drawable()?;
        let layout = self.layout()?;
        let (xd, yd) =
            apply_limits(self.coord(), layout.x.continuous_range, layout.y.continuous_range)?;
        if layout.layers.iter().any(|l| l.points.is_none() && l.path.is_none()) {
            return Err(Error::Rendering(
                "the terminal target draws point, line, path, step and reference layers only"
                    .into(),
            ));
        }

        let (w, h) = (opts.width as usize, opts.height as usize);
        let mut c = Canvas::new(w, h);
        let top = self.draw_header(&mut c);
        let below = 2 + usize::from(self.xlab().is_some());
        let ticks = y_ticks(&layout, yd);
        let frame = Frame::new((w, h), (top, below), gutter_of(&ticks), (xd, yd))?;

        draw_y_axis(&mut c, &frame, &ticks);
        self.draw_x_axis(&mut c, &frame, &layout);
        draw_layers(&mut c, &frame, &layout);
        Ok(c.render(opts.color))
    }

    /// Refuse what the terminal target does not draw: facets, `coord_flip()`, and position scales
    /// (`scale_x`/`scale_y` or a text position), whose positions and labels it does not map.
    fn check_drawable(&self) -> Result<()> {
        if self.uses_position_scales() {
            return Err(Error::Rendering(
                "the terminal target does not draw scale_x/scale_y or text (discrete) positions yet"
                    .into(),
            ));
        }
        if !matches!(self.facet(), Facet::None) {
            return Err(Error::Rendering("the terminal target does not draw facets yet".into()));
        }
        if matches!(self.coord(), Coord::Cartesian { flip: true, .. }) {
            return Err(Error::Rendering(
                "the terminal target does not draw coord_flip() yet".into(),
            ));
        }
        Ok(())
    }

    /// Title (centred) and y label (left) from the top; returns the first panel row.
    fn draw_header(&self, c: &mut Canvas) -> usize {
        let mut top = 0;
        if let Some(t) = self.title() {
            c.text(c.w.saturating_sub(t.chars().count()) / 2, top, t, Ink::Title);
            top += 1;
        }
        if let Some(t) = self.ylab() {
            c.text(0, top, t, Ink::Plain);
            top += 1;
        }
        top
    }

    /// x axis, ticks and centred break labels, then the x label. A label that would touch the
    /// previous one is skipped rather than overprinted.
    fn draw_x_axis(&self, c: &mut Canvas, f: &Frame, layout: &BuiltLayout) {
        let axis_row = f.top + f.plot_h;
        c.put(f.axis_col, axis_row, '└', Ink::Axis);
        for col in f.plot_left..c.w {
            c.put(col, axis_row, '─', Ink::Axis);
        }
        let mut free = 0;
        let breaks = layout.x.breaks.iter().zip(&layout.x.labels);
        for (b, label) in breaks.filter(|(b, _)| in_domain(**b, f.xd)) {
            let col = f.col_of(*b);
            c.put(col, axis_row, '┬', Ink::Axis);
            put_x_label(c, (col, axis_row + 1), label, &mut free);
        }
        if let Some(t) = self.xlab() {
            c.text(
                f.plot_left + f.plot_w.saturating_sub(t.chars().count()) / 2,
                axis_row + 2,
                t,
                Ink::Plain,
            );
        }
    }
}

/// y axis line, ticks and right-aligned break labels.
fn draw_y_axis(c: &mut Canvas, f: &Frame, ticks: &[(f64, &str)]) {
    for r in f.top..f.top + f.plot_h {
        c.put(f.axis_col, r, '│', Ink::Axis);
    }
    for (b, label) in ticks {
        let r = f.row_of(*b);
        c.put(f.axis_col, r, '┤', Ink::Axis);
        c.text(f.gutter - label.chars().count(), r, label, Ink::Axis);
    }
}

/// Every layer in order: a point layer one glyph per point, a stroke layer one glyph per cell
/// its clipped segments cross.
fn draw_layers(c: &mut Canvas, f: &Frame, layout: &BuiltLayout) {
    for layer in &layout.layers {
        match &layer.path {
            Some(path) => draw_stroke_layer(c, f, layer, path),
            None => draw_point_layer(c, f, layer),
        }
    }
}

/// One stroke layer: a glyph in every cell its clipped segments cross.
fn draw_stroke_layer(c: &mut Canvas, f: &Frame, layer: &BuiltLayer, path: &BuiltPath) {
    let ink = Ink::Point(path.aes.colour.r, path.aes.colour.g, path.aes.colour.b);
    for line in stroke::polylines(layer, 0, (f.xd, f.yd)) {
        let cells: Vec<(usize, usize)> =
            line.iter().map(|&(x, y)| (f.col_of(x), f.row_of(y))).collect();
        for w in cells.windows(2) {
            for (col, row) in stroke::cells_between(w[0], w[1]) {
                c.put(col, row, LINE_GLYPH, ink);
            }
        }
    }
}

/// One point layer: one glyph per point inside the domain.
fn draw_point_layer(c: &mut Canvas, f: &Frame, layer: &BuiltLayer) {
    let aes = layer.points.as_deref().unwrap_or_default();
    for ((x, y), p) in layer.x.iter().zip(&layer.y).zip(aes) {
        if in_domain(*x, f.xd) && in_domain(*y, f.yd) {
            let ink = Ink::Point(p.colour.r, p.colour.g, p.colour.b);
            c.put(f.col_of(*x), f.row_of(*y), POINT_GLYPH, ink);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Rgba;
    use crate::grammar::{scale_continuous, scale_log10, Aes, DataFrame, GGPlot, Geom};

    /// Row 1's fixture data (`fixtures/ggplot2/geom_point/data.csv`).
    const X: [f32; 10] = [1.5, 2.25, 3.0, 4.75, 6.5, 8.125, 9.5, 11.0, 12.25, 13.75];
    const Y: [f32; 10] = [-2.25, 0.5, 1.75, 3.0, 2.5, 4.125, 5.5, 4.75, 6.25, 7.5];

    fn row1() -> GGPlot {
        GGPlot::new().data_xy(&X, &Y).geom(Geom::point())
    }

    fn plain(w: u32, h: u32) -> TerminalPlot {
        TerminalPlot { width: w, height: h, color: false }
    }

    fn render(p: GGPlot, opts: TerminalPlot) -> String {
        p.build().expect("builds").to_terminal(opts).expect("renders")
    }

    #[test]
    fn has_exactly_height_lines_none_wider_than_width() {
        let out = render(row1().title("T").xlab("x").ylab("y"), plain(80, 24));
        assert_eq!(out.lines().count(), 24);
        assert!(out.lines().all(|l| l.chars().count() <= 80));
        assert!(!out.contains('\x1b'), "plain output carries an escape");
    }

    #[test]
    fn draws_every_point_and_the_oracle_break_labels() {
        let out = render(row1(), plain(80, 24));
        assert_eq!(out.matches(POINT_GLYPH).count(), 10, "\n{out}");
        let lines: Vec<&str> = out.lines().collect();
        // ggplot2 4.0.3 labels this data x: 5, 10 and y: -2.5 0.0 2.5 5.0 7.5.
        let x_labels: Vec<&str> = lines[23].split_whitespace().collect();
        assert_eq!(x_labels, ["5", "10"]);
        let y_labels: Vec<&str> = lines
            .iter()
            .filter(|l| l.contains('┤'))
            .map(|l| l.split('┤').next().unwrap_or_default().trim())
            .collect();
        assert_eq!(y_labels, ["7.5", "5.0", "2.5", "0.0", "-2.5"]);
    }

    #[test]
    fn is_deterministic() {
        assert_eq!(render(row1(), plain(80, 24)), render(row1(), plain(80, 24)));
    }

    #[test]
    fn colour_wraps_runs_and_resets() {
        let out = render(
            row1()
                .title("T")
                .aes(Aes::new().x("x").y("y").color_value(Rgba::new(200, 10, 20, 255))),
            TerminalPlot { width: 60, height: 16, color: true },
        );
        assert!(out.contains("\x1b[38;2;200;10;20m●"), "{out:?}");
        assert!(out.contains("\x1b[1mT\x1b[0m"), "{out:?}");
        assert!(out.contains("\x1b[90m"), "{out:?}");
        assert_eq!(Ink::Plain.sgr(), "");
        let black = render(row1(), TerminalPlot { width: 60, height: 16, color: true });
        assert!(black.contains("\x1b[1m●"), "{black:?}");
    }

    #[test]
    fn labels_land_where_they_belong() {
        let out = render(row1().title("Title").xlab("xx").ylab("yy"), plain(40, 12));
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0].trim(), "Title");
        assert_eq!(lines[1], "yy");
        assert_eq!(lines[11].trim(), "xx");
        assert!(lines[9].contains('└') && lines[9].contains('┬'));
    }

    #[test]
    fn crowded_x_labels_are_skipped_not_overprinted() {
        // 21 cells: x ticks at 5 and 10 land 4 cells apart, so both labels still fit; at the
        // minimum width with long labels the second must be dropped.
        let p = GGPlot::new().data_xy(&[1_000_000.5, 1_000_010.5], &[0.0, 1.0]).geom(Geom::point());
        let out = render(p, plain(20, 8));
        let x_line = out.lines().nth(7).unwrap_or_default();
        assert!(!x_line.contains("00 1"), "labels overprinted: {x_line:?}");
        assert!(x_line.trim().len() >= 9, "{out}");
    }

    #[test]
    fn coord_limits_drop_points_and_breaks_outside() {
        let p = row1().coord(Coord::cartesian().xlim(0.0, 7.0).ylim(0.0, 3.0));
        let out = render(p, plain(60, 16));
        let pts = X.iter().zip(&Y).filter(|(x, y)| **x <= 7.0 && **y >= 0.0 && **y <= 3.0).count();
        assert_eq!(out.matches(POINT_GLYPH).count(), pts);
        assert!(!out.contains("10"), "{out}");
    }

    #[test]
    fn refuses_what_it_cannot_draw() {
        let err = |p: GGPlot, o: TerminalPlot| {
            p.build().expect("builds").to_terminal(o).expect_err("must refuse").to_string()
        };
        assert!(err(row1(), plain(19, 24)).contains("at least 20x8"));
        assert!(err(row1(), plain(80, 7)).contains("at least 20x8"));
        // The canvas is width x height cells, so an unbounded size is an unbounded allocation:
        // refused before a single cell exists.
        for (w, h) in [(1001, 24), (80, 1001), (u32::MAX, u32::MAX)] {
            assert!(err(row1(), plain(w, h)).contains("at most 1000x1000"), "{w}x{h}");
        }
        assert!(err(row1().facet(Facet::wrap("g", 2)), plain(80, 24)).contains("facets"));
        assert!(err(row1().coord(Coord::cartesian().flip()), plain(80, 24)).contains("coord_flip"));
        assert!(err(row1().coord(Coord::polar()), plain(80, 24)).contains("polar"));
        // Bars refuse in layout() over a numeric x, and over a text x in the terminal itself.
        assert!(err(row1().geom(Geom::bar()), plain(80, 24)).contains("geom_bar()/geom_col()"));
        let scaled = row1().scale_x(scale_log10());
        assert!(err(scaled, plain(80, 24)).contains("scale_x/scale_y"));
        let scaled_y = row1().scale_y(scale_continuous());
        assert!(err(scaled_y, plain(80, 24)).contains("scale_x/scale_y"));
        let mut text = DataFrame::new();
        text.add_column_str("x", &["a", "b"]);
        text.add_column_f32("y", &[1.0, 2.0]);
        let discrete = GGPlot::new().data(text).aes(Aes::new().x("x").y("y"));
        let col_msg = err(discrete.clone().geom(Geom::col()), plain(80, 24));
        assert!(col_msg.contains("reference layers only"), "{col_msg}");
        assert!(err(discrete.geom(Geom::point()), plain(80, 24)).contains("text (discrete)"));
        let wide =
            // R's format() prints these breaks as 13-character labels (-1.000002e+14), too wide
            // for a 20-cell canvas.
            GGPlot::new().data_xy(&[0.0, 1.0], &[-1.0e14, -1.000_005e14]).geom(Geom::point());
        assert!(err(wide, plain(20, 8)).contains("no room for the panel"));
    }

    #[test]
    fn text_longer_than_the_width_is_clipped_at_the_edge() {
        let title = "abcdefghijklmnopqrstuvwxyz0123";
        let out = render(row1().title(title), plain(20, 8));
        assert_eq!(out.lines().next(), Some("abcdefghijklmnopqrst"));
        assert!(out.lines().all(|l| l.chars().count() <= 20));
    }

    #[test]
    fn draws_at_the_largest_canvas() {
        let out = render(row1(), plain(MAX_TERMINAL_WIDTH, MAX_TERMINAL_HEIGHT));
        assert_eq!(out.lines().count(), MAX_TERMINAL_HEIGHT as usize);
        assert_eq!(out.matches(POINT_GLYPH).count(), X.len());
    }

    #[test]
    fn empty_rows_are_blank_lines() {
        let mut df = DataFrame::new();
        df.add_column_f32("x", &X);
        df.add_column_f32("y", &Y);
        let out = render(GGPlot::new().data(df).geom(Geom::point()), plain(30, 10));
        assert!(out.ends_with('\n'));
        assert_eq!(out.lines().count(), 10);
    }

    #[test]
    fn check_canvas_bounds() {
        assert!(check_canvas(plain(MIN_TERMINAL_WIDTH, MIN_TERMINAL_HEIGHT)).is_ok());
        assert!(check_canvas(plain(MAX_TERMINAL_WIDTH, MAX_TERMINAL_HEIGHT)).is_ok());
        let small = check_canvas(plain(19, 8)).expect_err("small").to_string();
        assert!(small.contains("at least 20x8 cells, got 19x8"), "{small}");
        let big = check_canvas(plain(80, 1001)).expect_err("big").to_string();
        assert!(big.contains("at most 1000x1000 cells, got 80x1001"), "{big}");
    }

    #[test]
    fn frame_maps_data_to_cells_and_refuses_a_squeezed_panel() {
        let f = Frame::new((40, 12), (2, 3), 3, ((0.0, 10.0), (0.0, 4.0))).expect("fits");
        assert_eq!((f.axis_col, f.plot_left, f.plot_w, f.plot_h), (4, 5, 35, 7));
        assert_eq!((f.col_of(0.0), f.col_of(10.0)), (5, 39));
        assert_eq!((f.row_of(0.0), f.row_of(4.0)), (8, 2));
        let e = Frame::new((12, 12), (0, 2), 3, ((0.0, 1.0), (0.0, 1.0)));
        let e = e.err().expect("too narrow").to_string();
        assert!(e.contains("12 cells wide; its y labels need 3"), "{e}");
    }

    #[test]
    fn y_ticks_keep_only_the_visible_domain_and_gutter_is_the_widest_label() {
        let built = row1().build().expect("builds");
        let layout = built.layout().expect("layout");
        let all = y_ticks(&layout, (-100.0, 100.0));
        assert_eq!(all.len(), layout.y.breaks.len());
        let some = y_ticks(&layout, (0.0, 3.0));
        assert!(some.len() < all.len() && some.iter().all(|(b, _)| (0.0..=3.0).contains(b)));
        assert_eq!(gutter_of(&all), 4);
        assert_eq!(gutter_of(&[]), 0);
    }

    #[test]
    fn put_x_label_centres_clamps_and_skips_a_touching_label() {
        let mut c = Canvas::new(10, 2);
        let mut free = 0;
        // Centred: starts one cell left of column 3, and the next free column follows it.
        put_x_label(&mut c, (3, 1), "abc", &mut free);
        assert_eq!(free, 6);
        // Would start at column 4, before the free column: skipped, nothing moves.
        put_x_label(&mut c, (5, 1), "abc", &mut free);
        assert_eq!(free, 6);
        // Too close to the right edge: pulled back so it ends on the last column.
        let mut edge = 0;
        put_x_label(&mut c, (9, 0), "wxyz", &mut edge);
        assert_eq!(edge, 11);
        assert_eq!(
            c.render(false),
            "      wxyz
  abc
"
        );
    }

    #[test]
    fn draw_header_returns_the_rows_it_used() {
        let mut c = Canvas::new(20, 8);
        let both = row1().title("T").ylab("y").build().expect("builds");
        assert_eq!(both.draw_header(&mut c), 2);
        let none = row1().build().expect("builds");
        assert_eq!(none.draw_header(&mut c), 0);
        let only_y = row1().ylab("y").build().expect("builds");
        assert_eq!(only_y.draw_header(&mut c), 1);
    }

    #[test]
    fn check_drawable_names_the_refusal() {
        let ok = row1().build().expect("builds");
        assert!(ok.check_drawable().is_ok());
        let facet = row1().facet(Facet::wrap("g", 2)).build().expect("builds");
        assert!(facet.check_drawable().expect_err("facet").to_string().contains("facets"));
        let flip = row1().coord(Coord::cartesian().flip()).build().expect("builds");
        assert!(flip.check_drawable().expect_err("flip").to_string().contains("coord_flip"));
    }
}
