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
//! Supported today: point layers, `coord_cartesian()` (with limits, without flip), no facets.
//! Anything else is refused with an error naming what is missing; nothing falls back silently.

use crate::error::{Error, Result};

use super::coord::{apply_limits, Coord};
use super::facet::Facet;
use super::ggplot::BuiltGGPlot;

/// Smallest canvas, in character cells, the terminal target draws on.
pub const MIN_TERMINAL_WIDTH: u32 = 20;
/// Smallest canvas height, in lines.
pub const MIN_TERMINAL_HEIGHT: u32 = 8;
/// The glyph a point is drawn with (ggplot2's default `shape = 19`, a solid circle).
pub const POINT_GLYPH: char = '●';
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

impl BuiltGGPlot {
    /// Render to text for a terminal of `opts.width` x `opts.height` cells.
    ///
    /// # Errors
    ///
    /// [`Error::Rendering`] for a canvas below [`MIN_TERMINAL_WIDTH`] x [`MIN_TERMINAL_HEIGHT`]
    /// (or too narrow for the y labels), a facet, `coord_flip()`, or a layer that is not a point
    /// layer; [`Error::UnsupportedCoord`] for polar or fixed coordinates.
    pub fn to_terminal(&self, opts: TerminalPlot) -> Result<String> {
        if opts.width < MIN_TERMINAL_WIDTH || opts.height < MIN_TERMINAL_HEIGHT {
            return Err(Error::Rendering(format!(
                "terminal plot needs at least {MIN_TERMINAL_WIDTH}x{MIN_TERMINAL_HEIGHT} cells, got {}x{}",
                opts.width, opts.height
            )));
        }
        if !matches!(self.facet(), Facet::None) {
            return Err(Error::Rendering("the terminal target does not draw facets yet".into()));
        }
        if matches!(self.coord(), Coord::Cartesian { flip: true, .. }) {
            return Err(Error::Rendering(
                "the terminal target does not draw coord_flip() yet".into(),
            ));
        }
        let layout = self.layout()?;
        let (xd, yd) =
            apply_limits(self.coord(), layout.x.continuous_range, layout.y.continuous_range)?;
        if layout.layers.iter().any(|l| l.points.is_none()) {
            return Err(Error::Rendering("the terminal target draws point layers only".into()));
        }

        let (w, h) = (opts.width as usize, opts.height as usize);
        let mut c = Canvas::new(w, h);

        // Rows from the top: title, y label, the panel; from the bottom: x label, x break
        // labels, the x axis.
        let mut top = 0;
        if let Some(t) = self.title() {
            c.text(w.saturating_sub(t.chars().count()) / 2, top, t, Ink::Title);
            top += 1;
        }
        if let Some(t) = self.ylab() {
            c.text(0, top, t, Ink::Plain);
            top += 1;
        }
        let below = 2 + usize::from(self.xlab().is_some());
        let plot_h = h - top - below;

        let y_ticks: Vec<(f64, &str)> = layout
            .y
            .breaks
            .iter()
            .zip(&layout.y.labels)
            .filter(|(b, _)| in_domain(**b, yd))
            .map(|(b, l)| (*b, l.as_str()))
            .collect();
        let gutter = y_ticks.iter().map(|(_, l)| l.chars().count()).max().unwrap_or(0);
        let axis_col = gutter + 1;
        let plot_left = axis_col + 1;
        if plot_left + MIN_PANEL_WIDTH > w {
            return Err(Error::Rendering(format!(
                "terminal plot is {w} cells wide; its y labels need {gutter}, leaving no room for the panel"
            )));
        }
        let plot_w = w - plot_left;
        let row_of = |y: f64| top + plot_h - 1 - cell(y, yd, plot_h);
        let col_of = |x: f64| plot_left + cell(x, xd, plot_w);

        // y axis, ticks and right-aligned break labels.
        for r in top..top + plot_h {
            c.put(axis_col, r, '│', Ink::Axis);
        }
        for (b, label) in &y_ticks {
            let r = row_of(*b);
            c.put(axis_col, r, '┤', Ink::Axis);
            c.text(gutter - label.chars().count(), r, label, Ink::Axis);
        }

        // x axis, ticks and centred break labels; a label that would touch the previous one is
        // skipped rather than overprinted.
        let axis_row = top + plot_h;
        c.put(axis_col, axis_row, '└', Ink::Axis);
        for col in plot_left..w {
            c.put(col, axis_row, '─', Ink::Axis);
        }
        let mut free = 0;
        for (b, label) in layout.x.breaks.iter().zip(&layout.x.labels) {
            if !in_domain(*b, xd) {
                continue;
            }
            let col = col_of(*b);
            c.put(col, axis_row, '┬', Ink::Axis);
            let len = label.chars().count();
            let start = col.saturating_sub(len / 2).min(w.saturating_sub(len));
            if start >= free {
                c.text(start, axis_row + 1, label, Ink::Axis);
                free = start + len + 1;
            }
        }
        if let Some(t) = self.xlab() {
            c.text(
                plot_left + plot_w.saturating_sub(t.chars().count()) / 2,
                axis_row + 2,
                t,
                Ink::Plain,
            );
        }

        // Points last, so they sit on top of nothing but the panel.
        for layer in &layout.layers {
            let aes = layer.points.as_deref().unwrap_or_default();
            for ((x, y), p) in layer.x.iter().zip(&layer.y).zip(aes) {
                if in_domain(*x, xd) && in_domain(*y, yd) {
                    let ink = Ink::Point(p.colour.r, p.colour.g, p.colour.b);
                    c.put(col_of(*x), row_of(*y), POINT_GLYPH, ink);
                }
            }
        }

        Ok(c.render(opts.color))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::Rgba;
    use crate::grammar::{Aes, DataFrame, GGPlot, Geom};

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
        assert!(err(row1().facet(Facet::wrap("g", 2)), plain(80, 24)).contains("facets"));
        assert!(err(row1().coord(Coord::cartesian().flip()), plain(80, 24)).contains("coord_flip"));
        assert!(err(row1().coord(Coord::polar()), plain(80, 24)).contains("polar"));
        assert!(err(row1().geom(Geom::line()), plain(80, 24)).contains("point layers only"));
        let wide =
            GGPlot::new().data_xy(&[0.0, 1.0], &[0.0, 123_456_789_012_345.0]).geom(Geom::point());
        assert!(err(wide, plain(20, 8)).contains("no room for the panel"));
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
}
