//! PLOT-GG-001 (paiml/aprender#6909): the line family and the reference lines are DRAWN, in every
//! render target. This is not a layout test: it reads the rendered SVG, the rendered terminal
//! text and the rendered pixels.
//!
//! The expected positions are computed independently of the code under test, from the axis. The
//! SVG carries a `<line>` tick and a `<text>` label for every break, drawn by the axis code and
//! not by the stroke code, so `pixel(v) = tick("0.0") + v * (tick("10.0") - tick("0.0")) / 10`
//! is a scale read off the picture. A stroke drawn at the wrong place, clipped wrongly or not
//! drawn at all cannot satisfy it.
//!
//! The data is x = y = (0, 5, 10), so ggplot2's breaks are 0.0 2.5 5.0 7.5 10.0 on both axes and
//! the visible domain is -0.5 to 10.5 (the 5% expansion) on both.

use std::collections::BTreeMap;
use std::sync::mpsc;
use std::time::Duration;

use trueno_viz::color::Rgba;
use trueno_viz::grammar::{
    Aes, Coord, DataFrame, GGPlot, Geom, StepDirection, TerminalPlot, LINE_GLYPH,
};

const TOL: f64 = 0.01;

fn base(x: &[f32], y: &[f32]) -> GGPlot {
    GGPlot::new().data_xy(x, y)
}

fn diagonal() -> GGPlot {
    base(&[0.0, 5.0, 10.0], &[0.0, 5.0, 10.0])
}

/// The diagonal data drawn as points (raster, not polylines), so reference lines have a scale.
fn dotted() -> GGPlot {
    diagonal().geom(Geom::point())
}

fn svg_of(p: GGPlot) -> String {
    p.build().expect("builds").to_svg().expect("renders")
}

/// What an SVG drew: its polylines, and the axis ticks by label.
struct Picture {
    polylines: Vec<Vec<(f64, f64)>>,
    x_ticks: BTreeMap<String, f64>,
    y_ticks: BTreeMap<String, f64>,
    texts: Vec<String>,
    strokes: Vec<String>,
    widths: Vec<f64>,
}

fn read(svg: &str) -> Picture {
    let doc = roxmltree::Document::parse(svg).expect("well-formed SVG");
    let els: Vec<_> = doc.root_element().children().filter(|n| n.is_element()).collect();
    let num = |n: &roxmltree::Node, a: &str| -> f64 {
        n.attribute(a).unwrap_or_default().parse().unwrap_or(f64::NAN)
    };
    let mut p = Picture {
        polylines: vec![],
        x_ticks: BTreeMap::new(),
        y_ticks: BTreeMap::new(),
        texts: vec![],
        strokes: vec![],
        widths: vec![],
    };
    for (i, el) in els.iter().enumerate() {
        match el.tag_name().name() {
            "polyline" => {
                let pts = el.attribute("points").unwrap_or_default();
                p.polylines.push(
                    pts.split_whitespace()
                        .map(|pair| {
                            let (a, b) = pair.split_once(',').expect("x,y");
                            (a.parse().expect("x"), b.parse().expect("y"))
                        })
                        .collect(),
                );
                p.strokes.push(el.attribute("stroke").unwrap_or_default().to_string());
                p.widths.push(num(el, "stroke-width"));
            }
            "text" => p.texts.push(el.text().unwrap_or_default().to_string()),
            "line" => {
                let Some(t) = els.get(i + 1).filter(|t| t.tag_name().name() == "text") else {
                    continue;
                };
                let label = t.text().unwrap_or_default().to_string();
                match t.attribute("text-anchor") {
                    Some("middle") => drop(p.x_ticks.insert(label, num(el, "x1"))),
                    Some("end") => drop(p.y_ticks.insert(label, num(el, "y1"))),
                    _ => {}
                }
            }
            _ => {}
        }
    }
    p
}

impl Picture {
    /// Pixel x of data x, read off the 0.0 and 10.0 ticks.
    fn px(&self, v: f64) -> f64 {
        let (a, b) = (self.x_ticks["0.0"], self.x_ticks["10.0"]);
        a + v * (b - a) / 10.0
    }

    /// Pixel y of data y.
    fn py(&self, v: f64) -> f64 {
        let (a, b) = (self.y_ticks["0.0"], self.y_ticks["10.0"]);
        a + v * (b - a) / 10.0
    }
}

fn assert_line(got: &[(f64, f64)], want: &[(f64, f64)]) {
    assert_eq!(got.len(), want.len(), "vertices: got {got:?}, want {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert!((g.0 - w.0).abs() < TOL && (g.1 - w.1).abs() < TOL, "got {got:?}, want {want:?}");
    }
}

/// The data-space vertices as the picture's pixels.
fn pixels(p: &Picture, pts: &[(f64, f64)]) -> Vec<(f64, f64)> {
    pts.iter().map(|&(x, y)| (p.px(x), p.py(y))).collect()
}

#[test]
fn line_is_a_polyline_through_the_data_at_the_axis_pixels() {
    let p = read(&svg_of(diagonal().geom(Geom::line())));
    assert_eq!(p.polylines.len(), 1);
    assert_line(&p.polylines[0], &pixels(&p, &[(0.0, 0.0), (5.0, 5.0), (10.0, 10.0)]));
}

#[test]
fn line_sorts_by_x_and_path_keeps_the_data_order() {
    let (x, y) = ([5.0, 0.0, 10.0], [5.0, 10.0, 0.0]);
    let line = read(&svg_of(base(&x, &y).geom(Geom::line())));
    assert_line(&line.polylines[0], &pixels(&line, &[(0.0, 10.0), (5.0, 5.0), (10.0, 0.0)]));
    let path = read(&svg_of(base(&x, &y).geom(Geom::path())));
    assert_line(&path.polylines[0], &pixels(&path, &[(5.0, 5.0), (0.0, 10.0), (10.0, 0.0)]));
}

#[test]
fn step_draws_the_stairs_of_each_direction() {
    let cases = [
        (StepDirection::Hv, vec![(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (10.0, 5.0), (10.0, 10.0)]),
        (StepDirection::Vh, vec![(0.0, 0.0), (0.0, 5.0), (5.0, 5.0), (5.0, 10.0), (10.0, 10.0)]),
        (
            StepDirection::Mid,
            vec![(0.0, 0.0), (2.5, 0.0), (2.5, 5.0), (7.5, 5.0), (7.5, 10.0), (10.0, 10.0)],
        ),
    ];
    for (dir, want) in cases {
        let p = read(&svg_of(diagonal().geom(Geom::step().direction(dir))));
        assert_eq!(p.polylines.len(), 1, "{dir:?}");
        assert_line(&p.polylines[0], &pixels(&p, &want));
    }
}

#[test]
fn reference_lines_span_the_panel_at_the_axis_pixels() {
    let svg =
        svg_of(dotted().geom(Geom::hline(2.5)).geom(Geom::vline(7.5)).geom(Geom::abline(0.5, 1.0)));
    let p = read(&svg);
    assert_eq!(p.polylines.len(), 3, "{svg}");
    // The visible domain is -0.5..10.5 on both axes.
    assert_line(&p.polylines[0], &pixels(&p, &[(-0.5, 2.5), (10.5, 2.5)]));
    assert_line(&p.polylines[1], &pixels(&p, &[(7.5, -0.5), (7.5, 10.5)]));
    // 0.5 * -0.5 + 1 = 0.75 and 0.5 * 10.5 + 1 = 6.25: both ends inside, so edge to edge.
    assert_line(&p.polylines[2], &pixels(&p, &[(-0.5, 0.75), (10.5, 6.25)]));
}

#[test]
fn an_abline_leaving_the_panel_is_clipped_to_it() {
    // y = 2x: at x = -0.5 it is at y = -1, below the panel, so it enters at y = -0.5, x = -0.25;
    // it leaves through the top at y = 10.5, x = 5.25.
    let p = read(&svg_of(dotted().geom(Geom::abline(2.0, 0.0))));
    assert_eq!(p.polylines.len(), 1);
    assert_line(&p.polylines[0], &pixels(&p, &[(-0.25, -0.5), (5.25, 10.5)]));
}

#[test]
fn an_hline_trains_the_scale_like_ggplot2_and_an_abline_that_misses_draws_nothing() {
    // geom_hline(yintercept = 50) widens the y axis to reach 50 (an intercept is a mapped
    // aesthetic in ggplot2), so the line sits on the tick labelled 50.
    let p = read(&svg_of(dotted().geom(Geom::hline(50.0))));
    let tick50 =
        p.y_ticks.iter().find(|(l, _)| l.parse::<f64>().is_ok_and(|v| (v - 50.0).abs() < 1e-9));
    let (label, y) = tick50.expect("a tick labelled 50");
    assert_eq!(p.polylines.len(), 1, "label {label}");
    assert_line(&p.polylines[0][..1], &[(p.polylines[0][0].0, *y)]);
    assert!((p.polylines[0][1].1 - *y).abs() < TOL, "{:?}", p.polylines);

    // An abline never trains a scale. y = x + 100 stays above the 0..10 panel everywhere.
    let missed = read(&svg_of(dotted().geom(Geom::abline(1.0, 100.0))));
    assert!(missed.polylines.is_empty(), "{:?}", missed.polylines);
}

#[test]
fn a_missing_value_breaks_the_line_and_groups_are_separate_lines() {
    let nan = f32::NAN;
    let broken = read(&svg_of(
        base(&[0.0, 2.5, 5.0, 7.5, 10.0], &[0.0, 5.0, nan, 5.0, 10.0]).geom(Geom::line()),
    ));
    assert_eq!(broken.polylines.len(), 2, "{:?}", broken.polylines);
    assert_line(&broken.polylines[0], &pixels(&broken, &[(0.0, 0.0), (2.5, 5.0)]));
    assert_line(&broken.polylines[1], &pixels(&broken, &[(7.5, 5.0), (10.0, 10.0)]));

    // A lone vertex between two gaps has no segment.
    let lone = read(&svg_of(base(&[0.0, 5.0, 10.0], &[0.0, nan, 10.0]).geom(Geom::line())));
    assert!(lone.polylines.is_empty(), "{:?}", lone.polylines);

    // Two groups: two polylines, each in its own order, never joined.
    let mut df = DataFrame::new();
    df.add_column_f32("x", &[0.0, 5.0, 10.0, 0.0, 5.0, 10.0]);
    df.add_column_f32("y", &[0.0, 5.0, 10.0, 10.0, 5.0, 0.0]);
    df.add_column_str("g", &["a", "a", "a", "b", "b", "b"]);
    let two = read(&svg_of(
        GGPlot::new().data(df).aes(Aes::new().x("x").y("y").group("g")).geom(Geom::line()),
    ));
    assert_eq!(two.polylines.len(), 2, "{:?}", two.polylines);
    assert_line(&two.polylines[0], &pixels(&two, &[(0.0, 0.0), (5.0, 5.0), (10.0, 10.0)]));
    assert_line(&two.polylines[1], &pixels(&two, &[(0.0, 10.0), (5.0, 5.0), (10.0, 0.0)]));
}

#[test]
fn colour_and_width_reach_the_stroke() {
    let aes = Aes::new().x("x").y("y").color_value(Rgba::new(200, 10, 20, 255));
    let p = read(&svg_of(diagonal().aes(aes).geom(Geom::line())));
    assert_eq!(p.strokes, ["rgb(200,10,20)"]);
    // ggplot2's default linewidth 0.5 mm is 0.5 * 72.27 / 25.4 px.
    assert!((p.widths[0] - 0.5 * 72.27 / 25.4).abs() < 0.001, "{:?}", p.widths);
}

#[test]
fn axis_titles_and_the_title_are_still_drawn_with_lines() {
    let p = read(&svg_of(
        diagonal()
            .geom(Geom::line())
            .geom(Geom::hline(5.0))
            .title("The Title")
            .xlab("The X Title")
            .ylab("The Y Title"),
    ));
    for t in ["The Title", "The X Title", "The Y Title"] {
        assert!(p.texts.iter().any(|x| x == t), "missing {t:?} in {:?}", p.texts);
    }
}

#[test]
fn svg_is_deterministic_with_strokes() {
    let one = || svg_of(diagonal().geom(Geom::line()).geom(Geom::abline(1.0, 0.0)));
    assert_eq!(one(), one());
}

fn term(p: GGPlot, w: u32, h: u32) -> String {
    p.build()
        .expect("builds")
        .to_terminal(TerminalPlot { width: w, height: h, color: false })
        .expect("renders")
}

/// The panel rows of a terminal plot: the lines with a y axis `│` or `┤`, as chars.
fn panel_rows(out: &str) -> Vec<Vec<char>> {
    out.lines()
        .map(|l| l.chars().collect::<Vec<char>>())
        .filter(|cs| cs.iter().any(|c| matches!(c, '│' | '┤')))
        .collect()
}

fn axis_col(row: &[char]) -> usize {
    row.iter().position(|c| matches!(c, '│' | '┤')).expect("axis")
}

#[test]
fn terminal_hline_fills_the_row_of_its_axis_label() {
    let out = term(dotted().geom(Geom::hline(2.5)), 41, 15);
    let rows = panel_rows(&out);
    let labelled = rows
        .iter()
        .find(|r| r.iter().collect::<String>().trim_start().starts_with("2.5 ┤"))
        .expect("the 2.5 label row");
    let from = axis_col(labelled) + 1;
    assert!(labelled[from..].iter().all(|&c| c == LINE_GLYPH), "{out}");
    // No other panel row is a full line.
    let full = rows
        .iter()
        .filter(|r| {
            let panel = &r[axis_col(r) + 1..];
            !panel.is_empty() && panel.iter().all(|&c| c == LINE_GLYPH)
        })
        .count();
    assert_eq!(full, 1, "{out}");
}

#[test]
fn terminal_vline_fills_the_column_of_its_tick() {
    let out = term(dotted().geom(Geom::vline(5.0)), 41, 15);
    let ticks: Vec<usize> = out
        .lines()
        .find(|l| l.contains('┬'))
        .expect("x axis")
        .chars()
        .enumerate()
        .filter_map(|(i, c)| (c == '┬').then_some(i))
        .collect();
    assert_eq!(ticks.len(), 5, "{out}");
    let col = ticks[2]; // the 5.0 tick
    let rows = panel_rows(&out);
    assert!(!rows.is_empty());
    assert!(rows.iter().all(|r| r.get(col) == Some(&LINE_GLYPH)), "{out}");
}

#[test]
fn terminal_line_marks_its_data_cells_and_joins_them() {
    let out = term(diagonal().geom(Geom::line()), 41, 15);
    let rows = panel_rows(&out);
    let marks: Vec<(usize, usize)> = rows
        .iter()
        .enumerate()
        .flat_map(|(r, cs)| {
            cs.iter().enumerate().filter(|(_, &c)| c == LINE_GLYPH).map(move |(c, _)| (r, c))
        })
        .collect();
    // Rising line: the first mark (top row) is right of the last (bottom row).
    let top = marks.iter().filter(|m| m.0 == marks[0].0).map(|m| m.1).max().expect("top");
    let bottom = marks.iter().filter(|m| m.0 == rows.len() - 1 - 0).map(|m| m.1).min();
    assert!(marks.len() > 20, "a joined line, not three dots: {out}");
    assert!(bottom.map_or(true, |b| b < top), "{out}");
    // A tick-aligned vertex: the data point (5, 5) lies on the 5.0 row at the 5.0 column.
    let ticks: Vec<usize> = out
        .lines()
        .find(|l| l.contains('┬'))
        .expect("x axis")
        .chars()
        .enumerate()
        .filter_map(|(i, c)| (c == '┬').then_some(i))
        .collect();
    let row5 = rows
        .iter()
        .find(|r| r.iter().collect::<String>().trim_start().starts_with("5.0 ┤"))
        .expect("5.0 row");
    assert_eq!(row5[ticks[2]], LINE_GLYPH, "{out}");
}

#[test]
fn terminal_breaks_the_line_at_a_missing_value_and_draws_the_titles() {
    let nan = f32::NAN;
    let out = term(
        base(&[0.0, 2.5, 5.0, 7.5, 10.0], &[0.0, 5.0, nan, 5.0, 10.0])
            .geom(Geom::line())
            .title("The Title")
            .xlab("The X")
            .ylab("The Y"),
        61,
        20,
    );
    let rows = panel_rows(&out);
    // The gap: the column of the missing x = 5 has no mark.
    let ticks: Vec<usize> = out
        .lines()
        .find(|l| l.contains('┬'))
        .expect("x axis")
        .chars()
        .enumerate()
        .filter_map(|(i, c)| (c == '┬').then_some(i))
        .collect();
    assert!(rows.iter().all(|r| r.get(ticks[2]) != Some(&LINE_GLYPH)), "{out}");
    for t in ["The Title", "The X", "The Y"] {
        assert!(out.contains(t), "missing {t:?}\n{out}");
    }
}

#[test]
fn terminal_refuses_a_flipped_coord_for_strokes() {
    let e = diagonal()
        .geom(Geom::line())
        .coord(Coord::cartesian().flip())
        .build()
        .expect("builds")
        .to_terminal(TerminalPlot { width: 41, height: 15, color: false })
        .expect_err("refuses");
    assert!(e.to_string().contains("coord_flip"), "{e}");
}

#[test]
fn raster_draws_the_stroke_where_the_svg_puts_it() {
    let plot = dotted().geom(Geom::hline(2.5));
    let p = read(&svg_of(plot.clone()));
    let fb = plot.build().expect("builds").to_framebuffer().expect("renders");
    let y = p.py(2.5).round() as u32;
    let x = p.px(5.0).round() as u32;
    let luminance = |px: Rgba| u32::from(px.r) + u32::from(px.g) + u32::from(px.b);
    // The hline is black over the grey panel (92% grey): darkest within a pixel of the tick.
    let on = [y.saturating_sub(1), y, y + 1]
        .iter()
        .map(|&row| luminance(fb.get_pixel(x, row).expect("pixel")))
        .min()
        .expect("min");
    let off = luminance(fb.get_pixel(x, y + 40).expect("pixel"));
    assert!(on + 150 < off, "on-line {on} vs panel {off}");
    // And the line reaches both panel edges.
    let (left, right) = (p.px(-0.5).ceil() as u32 + 1, p.px(10.5).floor() as u32 - 1);
    for edge in [left, right] {
        let m = [y.saturating_sub(1), y, y + 1]
            .iter()
            .map(|&row| luminance(fb.get_pixel(edge, row).expect("pixel")))
            .min()
            .expect("min");
        assert!(m + 150 < off, "edge x={edge}: {m} vs {off}");
    }
}

/// Run `f` on a thread; fail on a panic or when it takes longer than 30 s.
fn bounded(name: &str, f: impl FnOnce() + Send + 'static) {
    let (tx, rx) = mpsc::channel();
    let handle = std::thread::spawn(move || {
        f();
        let _ = tx.send(());
    });
    match rx.recv_timeout(Duration::from_secs(30)) {
        Ok(()) => handle.join().unwrap_or_else(|_| panic!("{name}: panicked")),
        Err(mpsc::RecvTimeoutError::Disconnected) => panic!("{name}: panicked"),
        Err(mpsc::RecvTimeoutError::Timeout) => panic!("{name}: still running after 30 s"),
    }
}

#[test]
fn extreme_inputs_neither_panic_nor_hang() {
    let big = f32::MAX;
    let tiny = f32::MIN_POSITIVE;
    let nan = f32::NAN;
    let inf = f32::INFINITY;
    let many: Vec<f32> = (0..5000).map(|i| (i as f32).sin() * 1.0e6).collect();
    let cases: Vec<(&str, GGPlot)> = vec![
        ("line to 1e30", base(&[0.0, 1.0e30], &[0.0, 1.0]).geom(Geom::line())),
        ("path across f32", base(&[-big, big, -big], &[big, -big, big]).geom(Geom::path())),
        (
            "step with nan and inf",
            base(&[0.0, nan, inf, 5.0], &[nan, 1.0, 2.0, -inf]).geom(Geom::step()),
        ),
        ("tiny", base(&[tiny, 2.0 * tiny], &[tiny, 3.0 * tiny]).geom(Geom::line())),
        ("denormal", base(&[1.0e-45, 2.0e-45], &[1.0e-45, 3.0e-45]).geom(Geom::line())),
        ("steep abline", diagonal().geom(Geom::abline(3.0e38, 3.0e38))),
        ("abline nan", diagonal().geom(Geom::abline(nan, 1.0))),
        ("hline max", diagonal().geom(Geom::hline(big))),
        ("vline -max", diagonal().geom(Geom::vline(-big))),
        ("hline nan", diagonal().geom(Geom::hline(nan))),
        ("vline inf", diagonal().geom(Geom::vline(inf))),
        ("5000 vertices", base(&many, &many).geom(Geom::line()).geom(Geom::step())),
        ("one vertex", base(&[1.0], &[1.0]).geom(Geom::line())),
        ("equal x", base(&[1.0, 1.0, 1.0], &[0.0, 1.0, 2.0]).geom(Geom::line())),
        ("all nan", base(&[nan, nan], &[nan, nan]).geom(Geom::line())),
    ];
    for (name, plot) in cases {
        bounded(name, move || {
            // Ok or a typed error are both fine; a panic or a hang is not.
            let _ = plot.clone().build().and_then(|b| b.to_svg());
            let _ = plot.clone().build().and_then(|b| b.to_framebuffer());
            let _ = plot
                .clone()
                .build()
                .and_then(|b| b.to_terminal(TerminalPlot { width: 80, height: 24, color: false }));
            let _ = plot.build().and_then(|b| b.layout());
        });
    }
}

#[test]
fn a_plot_of_reference_lines_alone_is_a_typed_error_not_a_picture() {
    // No data layer, no scale: ggplot2 would also have nothing to draw them on.
    let plot = GGPlot::new().geom(Geom::hline(1.0));
    let err = plot.build().and_then(|b| b.to_svg()).expect_err("refused");
    assert!(
        err.to_string().contains("degenerate range") || err.to_string().contains("no finite"),
        "{err}"
    );
}

#[test]
fn svg_and_raster_refuse_a_flipped_coord_for_strokes() {
    let plot = diagonal().geom(Geom::line()).coord(Coord::cartesian().flip());
    let e = plot.clone().build().expect("builds").to_svg().expect_err("svg refuses");
    assert!(e.to_string().contains("coord_flip"), "{e}");
    let e = plot.build().expect("builds").to_framebuffer().expect_err("raster refuses");
    assert!(e.to_string().contains("coord_flip"), "{e}");
    // Points under the same coord are not strokes and still draw.
    assert!(diagonal()
        .geom(Geom::point())
        .coord(Coord::cartesian().flip())
        .build()
        .expect("builds")
        .to_svg()
        .is_ok());
}

#[test]
fn a_degenerate_zoom_is_an_error_for_strokes_not_a_panic() {
    for plot in [
        diagonal().geom(Geom::line()).coord(Coord::cartesian().xlim(1.0, 1.0)),
        diagonal()
            .geom(Geom::hline(1.0))
            .geom(Geom::point())
            .coord(Coord::cartesian().ylim(2.0, 2.0)),
    ] {
        let built = plot.build().expect("builds");
        let _ = built.to_svg();
        let _ = built.to_framebuffer();
    }
}

/// Pixel of data value `v` on one axis, fitted to the lowest and highest tick label of `ticks`.
fn fit(ticks: &BTreeMap<String, f64>, v: f64) -> f64 {
    let mut t: Vec<(f64, f64)> =
        ticks.iter().map(|(l, &px)| (l.parse::<f64>().expect("numeric tick"), px)).collect();
    t.sort_by(|a, b| a.0.total_cmp(&b.0));
    let ((v0, p0), (v1, p1)) = (t[0], t[t.len() - 1]);
    p0 + (v - v0) * (p1 - p0) / (v1 - v0)
}

/// `geom_path` over `x = 0..n`, `y` and a text `group`, drawn to SVG.
fn grouped_path(y: &[f32], groups: &[&str]) -> Picture {
    let mut df = DataFrame::new();
    let x: Vec<f32> = (0..y.len()).map(|i| i as f32).collect();
    df.add_column_f32("x", &x);
    df.add_column_f32("y", y);
    df.add_column_str("g", groups);
    read(&svg_of(
        GGPlot::new().data(df).aes(Aes::new().x("x").y("y").group("g")).geom(Geom::path()),
    ))
}

fn in_pixels(p: &Picture, pts: &[(f64, f64)]) -> Vec<(f64, f64)> {
    pts.iter().map(|&(x, y)| (fit(&p.x_ticks, x), fit(&p.y_ticks, y))).collect()
}

// ggplot2 4.0.3, R 4.6.2, LC_ALL=C, `layer_grob()` polylines in data space (the review's C06):
//   P(d(0:6, c(1,2,3,4,5,6,0), c("b","B","a","b","B","a","b"))) + geom_path()
// GeomPath$draw_panel orders rows by group first, so R draws 3 polylines: B, a, b.
#[test]
fn path_with_interleaved_groups_draws_one_polyline_per_group_as_r_does() {
    let p = grouped_path(&[1., 2., 3., 4., 5., 6., 0.], &["b", "B", "a", "b", "B", "a", "b"]);
    assert_eq!(p.polylines.len(), 3, "{:?}", p.polylines);
    assert_line(&p.polylines[0], &in_pixels(&p, &[(1.0, 2.0), (4.0, 5.0)]));
    assert_line(&p.polylines[1], &in_pixels(&p, &[(2.0, 3.0), (5.0, 6.0)]));
    assert_line(&p.polylines[2], &in_pixels(&p, &[(0.0, 1.0), (3.0, 4.0), (6.0, 0.0)]));
}

// The review's C07: text groups sort as strings ("10" < "2" < "9"), rows interleaved.
//   P(d(0:5, c(5,1,4,2,3,0), c("10","9","2","10","9","2"))) + geom_path()
#[test]
fn path_with_interleaved_text_groups_sorts_them_as_text() {
    let p = grouped_path(&[5., 1., 4., 2., 3., 0.], &["10", "9", "2", "10", "9", "2"]);
    assert_eq!(p.polylines.len(), 3, "{:?}", p.polylines);
    assert_line(&p.polylines[0], &in_pixels(&p, &[(0.0, 5.0), (3.0, 2.0)]));
    assert_line(&p.polylines[1], &in_pixels(&p, &[(2.0, 4.0), (5.0, 0.0)]));
    assert_line(&p.polylines[2], &in_pixels(&p, &[(1.0, 1.0), (4.0, 3.0)]));
}
