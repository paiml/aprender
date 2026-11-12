//! The renderers are total (paiml/aprender#6941, Refs #6909): no public plotting path panics or
//! runs unbounded for any f32 input.
//!
//! Twelve inputs panicked (`attempt to add with overflow` in `draw_circle` and `fill_rect`) or ran
//! for over a minute (`render_area` walking 2^31 rows, `draw_line_aa` walking off-canvas pixels).
//! Each is a named test with a hard time bound. The property tests then sweep arbitrary f32
//! (NaN, +/-Inf, +/-MAX, subnormals), arbitrary coord limits and canvas sizes 1x1..512x512 across
//! every geom, through `to_framebuffer`, `to_svg` and `to_terminal`. The last group proves the
//! clipping changes nothing for data on or near the canvas: the clipped rasterisers are compared
//! pixel for pixel with reference copies of the unclipped algorithms.

use std::sync::mpsc;
use std::time::Duration;

use proptest::prelude::*;
use trueno_viz::color::Rgba;
use trueno_viz::framebuffer::Framebuffer;
use trueno_viz::grammar::{Coord, GGPlot, Geom, Layer, TerminalPlot};
use trueno_viz::render::{draw_circle, draw_line, draw_line_aa, draw_rect};

/// Generous: every case below takes milliseconds; the defects ran for minutes.
const BOUND: Duration = Duration::from_secs(20);

const INF: f32 = f32::INFINITY;
const NAN: f32 = f32::NAN;

/// Run `f` on its own thread and fail the test (with `name`) if it panics or outlives [`BOUND`].
fn bounded<T: Send + 'static>(name: &str, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    match rx.recv_timeout(BOUND) {
        Ok(v) => v,
        Err(mpsc::RecvTimeoutError::Disconnected) => panic!("{name}: the renderer panicked"),
        Err(mpsc::RecvTimeoutError::Timeout) => panic!("{name}: the renderer ran past {BOUND:?}"),
    }
}

/// The SVG with the embedded PNG removed (base64 can spell "inf" or "nan"), so what is left is
/// attributes and text only.
fn svg_markup(svg: &str) -> String {
    let mut out = String::new();
    let mut rest = svg;
    while let Some(i) = rest.find("href=\"") {
        out.push_str(&rest[..i]);
        let tail = &rest[i + 6..];
        rest = tail.find('"').map_or("", |j| &tail[j..]);
    }
    out.push_str(rest);
    out
}

/// Render `plot` through every output and check nothing non-finite reaches the SVG.
fn render_all(plot: GGPlot) {
    let Ok(built) = plot.build() else { return };
    let _ = built.to_framebuffer();
    if let Ok(svg) = built.to_svg() {
        let markup = svg_markup(&svg);
        assert!(!markup.contains("NaN"), "NaN in SVG: {markup}");
        assert!(!markup.to_lowercase().contains("inf"), "inf in SVG: {markup}");
    }
    let _ = built.to_terminal(TerminalPlot { width: 80, height: 24, color: false });
}

fn xy(x: &[f32], y: &[f32], g: Geom) -> GGPlot {
    GGPlot::new().data_xy(x, y).geom(g)
}

/// The data every "mixed" repro uses: finite rows beside NaN and +/-Inf in both columns.
fn mixed() -> (Vec<f32>, Vec<f32>) {
    (vec![INF, -INF, NAN, 1e38, -1e38, 1.0, 2.0], vec![1.0, 2.0, 3.0, -1e38, 1e38, INF, NAN])
}

fn limits() -> Coord {
    Coord::cartesian().xlim(0.0, 1.0).ylim(0.0, 1.0)
}

macro_rules! repro {
    ($name:ident, $geom:expr, $x:expr, $y:expr, $coord:expr) => {
        #[test]
        fn $name() {
            bounded(stringify!($name), || {
                let (x, y): (Vec<f32>, Vec<f32>) = ($x, $y);
                let plot = xy(&x, &y, $geom).coord($coord);
                let built = plot.build().expect("builds");
                built.to_framebuffer().expect("renders");
                let svg = built.to_svg().expect("svg");
                let markup = svg_markup(&svg);
                assert!(!markup.contains("NaN") && !markup.to_lowercase().contains("inf"));
            });
        }
    };
}

// 1. draw_circle: `cx + x` overflowed i32.
repro!(
    point_coordlim_outside,
    Geom::point(),
    vec![0.5, 1e38, -1e38],
    vec![0.5, 1e38, -1e38],
    limits()
);
// 2. fill_rect: `x + w` overflowed u32.
repro!(bar_coordlim_outside, Geom::bar(), vec![0.5, 1e38, -1e38], vec![0.5, 1e38, -1e38], limits());
// 3. and 4. line and area with finite points far off the canvas walked billions of pixels.
repro!(
    line_coordlim_outside,
    Geom::line(),
    vec![0.5, 1e38, -1e38],
    vec![0.5, 1e38, -1e38],
    limits()
);
repro!(
    area_coordlim_outside,
    Geom::area(),
    vec![0.5, 1e38, -1e38],
    vec![0.5, 1e38, -1e38],
    limits()
);
// 5. the y = 0 baseline of this area maps to an extreme pixel: the fill looped about 2^31 rows.
repro!(
    area_huge_y_close,
    Geom::area(),
    vec![1.0, 2.0, 3.0],
    vec![3.0e38, 3.0000002e38, 3.0000004e38],
    Coord::cartesian()
);
// 6. to 12. non-finite rows, with and without coord_flip().
repro!(point_mixed, Geom::point(), mixed().0, mixed().1, Coord::cartesian());
repro!(line_mixed, Geom::line(), mixed().0, mixed().1, Coord::cartesian());
repro!(area_mixed, Geom::area(), mixed().0, mixed().1, Coord::cartesian());
repro!(point_flip_inf, Geom::point(), mixed().0, mixed().1, Coord::cartesian().flip());
repro!(bar_flip_inf, Geom::bar(), mixed().0, mixed().1, Coord::cartesian().flip());
repro!(line_flip_inf, Geom::line(), mixed().0, mixed().1, Coord::cartesian().flip());
repro!(area_flip_inf, Geom::area(), mixed().0, mixed().1, Coord::cartesian().flip());

#[test]
fn non_finite_rows_are_dropped_not_drawn_at_pixel_zero() {
    // main drew a NaN row at pixel 0; the row is now absent, so the render equals the one
    // without it.
    let with = xy(&[1.0, NAN, 3.0], &[1.0, 2.0, 3.0], Geom::point()).build().expect("builds");
    let without = xy(&[1.0, 3.0], &[1.0, 3.0], Geom::point()).build().expect("builds");
    assert_eq!(
        with.to_framebuffer().expect("fb").pixels(),
        without.to_framebuffer().expect("fb").pixels()
    );
    let all_bad = xy(&[NAN, INF], &[1.0, NAN], Geom::point()).build().expect("builds");
    assert!(all_bad.layout().expect("layout").layers[0].x.is_empty());
}

#[test]
fn reference_lines_with_non_finite_intercepts_draw_nothing() {
    for v in [NAN, INF, -INF, 1e38, -1e38] {
        for layer in [Layer::new(Geom::hline(v)), Layer::new(Geom::vline(v))] {
            bounded("intercept", move || {
                render_all(xy(&[1.0, 2.0], &[1.0, 2.0], Geom::point()).layer(layer));
            });
        }
    }
}

#[test]
fn non_finite_limits_draw_nothing_and_do_not_panic() {
    for (lo, hi) in [(f64::NAN, 1.0), (0.0, f64::INFINITY), (-1e300, 1e300), (f64::NAN, f64::NAN)] {
        bounded("limits", move || {
            render_all(
                xy(&[1.0, 2.0], &[1.0, 2.0], Geom::point())
                    .coord(Coord::cartesian().xlim(lo, hi).ylim(lo, hi)),
            );
        });
    }
}

// ---------------------------------------------------------------------------
// Property tests: arbitrary f32, limits and canvas sizes, every geom, every output.
// ---------------------------------------------------------------------------

fn any_f32() -> impl Strategy<Value = f32> {
    prop_oneof![
        3 => proptest::num::f32::ANY,
        2 => prop_oneof![
            Just(f32::MAX),
            Just(-f32::MAX),
            Just(f32::MIN_POSITIVE),
            Just(-f32::MIN_POSITIVE),
            Just(1e-45_f32),
            Just(-1e-45_f32),
            Just(0.0_f32),
            Just(-0.0_f32),
            Just(INF),
            Just(-INF),
            Just(NAN),
        ],
        3 => -2.0_f32..2.0,
    ]
}

fn any_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        3 => proptest::num::f64::ANY,
        3 => -2.0_f64..2.0,
        1 => Just(f64::MAX),
        1 => Just(f64::NAN),
    ]
}

fn any_geom() -> impl Strategy<Value = Geom> {
    prop_oneof![
        Just(Geom::point()),
        Just(Geom::line()),
        Just(Geom::area()),
        Just(Geom::bar()),
        Just(Geom::histogram()),
        Just(Geom::boxplot()),
        Just(Geom::violin()),
        Just(Geom::tile()),
        Just(Geom::text()),
        any_f32().prop_map(Geom::hline),
        any_f32().prop_map(Geom::vline),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(160))]

    #[test]
    fn every_geom_renders_total(
        rows in proptest::collection::vec((any_f32(), any_f32()), 0..9),
        geom in any_geom(),
        xlim in proptest::option::of((any_f64(), any_f64())),
        ylim in proptest::option::of((any_f64(), any_f64())),
        flip in any::<bool>(),
        w in 1_u32..=512,
        h in 1_u32..=512,
    ) {
        let (x, y): (Vec<f32>, Vec<f32>) = rows.into_iter().unzip();
        let mut coord = Coord::cartesian();
        if let Some((lo, hi)) = xlim { coord = coord.xlim(lo, hi); }
        if let Some((lo, hi)) = ylim { coord = coord.ylim(lo, hi); }
        if flip { coord = coord.flip(); }
        let plot = xy(&x, &y, geom).coord(coord).dimensions(w, h);
        bounded("every_geom_renders_total", move || render_all(plot));
    }

    #[test]
    fn point_line_area_bar_layers_stack_total(
        rows in proptest::collection::vec((any_f32(), any_f32()), 1..9),
        w in 1_u32..=512,
        h in 1_u32..=512,
        lim in (any_f64(), any_f64()),
    ) {
        let (x, y): (Vec<f32>, Vec<f32>) = rows.into_iter().unzip();
        let plot = xy(&x, &y, Geom::point())
            .layer(Layer::new(Geom::line()))
            .layer(Layer::new(Geom::area()))
            .layer(Layer::new(Geom::bar()))
            .coord(Coord::cartesian().xlim(lim.0, lim.1).ylim(lim.0, lim.1))
            .dimensions(w, h);
        bounded("layers", move || render_all(plot));
    }
}

// ---------------------------------------------------------------------------
// Clipping is invisible for data on or near the canvas: reference copies of the unclipped
// algorithms (as on main) against the shipped, clipped ones.
// ---------------------------------------------------------------------------

fn blank(w: u32, h: u32) -> Framebuffer {
    let mut fb = Framebuffer::new(w, h).expect("fb");
    fb.clear(Rgba::WHITE);
    fb
}

fn fpart(x: f32) -> f32 {
    x - x.floor()
}

fn rfpart(x: f32) -> f32 {
    1.0 - fpart(x)
}

fn ref_plot(fb: &mut Framebuffer, x: i32, y: i32, color: Rgba, intensity: f32) {
    if x >= 0 && y >= 0 && x < fb.width() as i32 && y < fb.height() as i32 {
        let alpha = (f32::from(color.a) * intensity) as u8;
        fb.blend_pixel(x as u32, y as u32, color.with_alpha(alpha));
    }
}

/// `draw_line_aa` exactly as on main, with no clipping.
fn ref_wu(fb: &mut Framebuffer, x0: f32, y0: f32, x1: f32, y1: f32, color: Rgba) {
    let steep = (y1 - y0).abs() > (x1 - x0).abs();
    let (x0, y0, x1, y1) = if steep { (y0, x0, y1, x1) } else { (x0, y0, x1, y1) };
    let (x0, y0, x1, y1) = if x0 > x1 { (x1, y1, x0, y0) } else { (x0, y0, x1, y1) };
    let dx = x1 - x0;
    let dy = y1 - y0;
    let gradient = if dx.abs() < f32::EPSILON { 1.0 } else { dy / dx };
    let xend = x0.round();
    let yend = y0 + gradient * (xend - x0);
    let xgap = rfpart(x0 + 0.5);
    let xpxl1 = xend as i32;
    let ypxl1 = yend.floor() as i32;
    if steep {
        ref_plot(fb, ypxl1, xpxl1, color, rfpart(yend) * xgap);
        ref_plot(fb, ypxl1 + 1, xpxl1, color, fpart(yend) * xgap);
    } else {
        ref_plot(fb, xpxl1, ypxl1, color, rfpart(yend) * xgap);
        ref_plot(fb, xpxl1, ypxl1 + 1, color, fpart(yend) * xgap);
    }
    let mut intery = yend + gradient;
    let xend = x1.round();
    let yend = y1 + gradient * (xend - x1);
    let xgap = fpart(x1 + 0.5);
    let xpxl2 = xend as i32;
    let ypxl2 = yend.floor() as i32;
    if steep {
        ref_plot(fb, ypxl2, xpxl2, color, rfpart(yend) * xgap);
        ref_plot(fb, ypxl2 + 1, xpxl2, color, fpart(yend) * xgap);
    } else {
        ref_plot(fb, xpxl2, ypxl2, color, rfpart(yend) * xgap);
        ref_plot(fb, xpxl2, ypxl2 + 1, color, fpart(yend) * xgap);
    }
    for x in (xpxl1 + 1)..xpxl2 {
        let ipart = intery.floor() as i32;
        if steep {
            ref_plot(fb, ipart, x, color, rfpart(intery));
            ref_plot(fb, ipart + 1, x, color, fpart(intery));
        } else {
            ref_plot(fb, x, ipart, color, rfpart(intery));
            ref_plot(fb, x, ipart + 1, color, fpart(intery));
        }
        intery += gradient;
    }
}

/// `draw_line` exactly as on main, with no clipping (coordinates kept small by the caller).
fn ref_bresenham(fb: &mut Framebuffer, x0: i32, y0: i32, x1: i32, y1: i32, color: Rgba) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let (mut x, mut y, mut err) = (x0, y0, dx + dy);
    loop {
        if x >= 0 && y >= 0 {
            fb.set_pixel(x as u32, y as u32, color);
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            if x == x1 {
                break;
            }
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            if y == y1 {
                break;
            }
            err += dx;
            y += sy;
        }
    }
}

/// `draw_circle` exactly as on main (i32 arithmetic; caller keeps the sums in range).
fn ref_circle(fb: &mut Framebuffer, cx: i32, cy: i32, radius: i32, color: Rgba) {
    let hline = |fb: &mut Framebuffer, x1: i32, x2: i32, y: i32| {
        if y < 0 || y >= fb.height() as i32 {
            return;
        }
        let x_start = x1.max(0) as u32;
        let x_end = (x2 + 1).max(0).min(fb.width() as i32) as u32;
        if x_start < x_end {
            fb.fill_rect(x_start, y as u32, x_end - x_start, 1, color);
        }
    };
    let mut x = radius;
    let mut y = 0;
    let mut err = 1 - radius;
    while x >= y {
        hline(fb, cx - x, cx + x, cy + y);
        hline(fb, cx - x, cx + x, cy - y);
        hline(fb, cx - y, cx + y, cy + x);
        hline(fb, cx - y, cx + y, cy - x);
        y += 1;
        if err < 0 {
            err += 2 * y + 1;
        } else {
            x -= 1;
            err += 2 * (y - x) + 1;
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    #[test]
    fn wu_line_equals_the_unclipped_line_near_the_canvas(
        w in 1_u32..=200, h in 1_u32..=200,
        a in (-1000.0_f32..1000.0, -1000.0_f32..1000.0),
        b in (-1000.0_f32..1000.0, -1000.0_f32..1000.0),
    ) {
        // Endpoints in -1000..1000 stay inside the 1024 px guard box of any canvas
        // (the canvas is at least 1 px), so the segment is drawn as given.
        let (mut got, mut want) = (blank(w, h), blank(w, h));
        draw_line_aa(&mut got, a.0, a.1, b.0, b.1, Rgba::BLACK);
        ref_wu(&mut want, a.0, a.1, b.0, b.1, Rgba::BLACK);
        prop_assert_eq!(got.pixels(), want.pixels());
    }

    #[test]
    fn bresenham_line_equals_the_unclipped_line_near_the_canvas(
        w in 1_u32..=200, h in 1_u32..=200,
        a in (-1000_i32..1000, -1000_i32..1000),
        b in (-1000_i32..1000, -1000_i32..1000),
    ) {
        let (mut got, mut want) = (blank(w, h), blank(w, h));
        draw_line(&mut got, a.0, a.1, b.0, b.1, Rgba::BLACK);
        ref_bresenham(&mut want, a.0, a.1, b.0, b.1, Rgba::BLACK);
        prop_assert_eq!(got.pixels(), want.pixels());
    }

    #[test]
    fn circle_equals_the_midpoint_circle_up_to_the_radius_limit(
        w in 1_u32..=200, h in 1_u32..=200,
        c in (-5000_i32..5200, -5000_i32..5200),
        r in 1_i32..=4096,
    ) {
        let (mut got, mut want) = (blank(w, h), blank(w, h));
        draw_circle(&mut got, c.0, c.1, r, Rgba::BLACK);
        ref_circle(&mut want, c.0, c.1, r, Rgba::BLACK);
        prop_assert_eq!(got.pixels(), want.pixels());
    }

    #[test]
    fn rect_on_the_canvas_equals_fill_rect(
        w in 1_u32..=200, h in 1_u32..=200,
        x in 0_i32..300, y in 0_i32..300, rw in 0_u32..300, rh in 0_u32..300,
    ) {
        let (mut got, mut want) = (blank(w, h), blank(w, h));
        draw_rect(&mut got, x, y, rw, rh, Rgba::BLACK);
        want.fill_rect(x as u32, y as u32, rw, rh, Rgba::BLACK);
        prop_assert_eq!(got.pixels(), want.pixels());
    }

    // Bounded work: whatever the endpoints, the shipped rasterisers return (the test would hang
    // otherwise) and never touch memory off the canvas (they would panic).
    #[test]
    fn rasterisers_return_for_any_f32_and_i32(
        w in 1_u32..=64, h in 1_u32..=64,
        f in proptest::array::uniform4(any_f32()),
        i in proptest::array::uniform4(any::<i32>()),
        r in any::<i32>(),
        rect in (any::<i32>(), any::<i32>(), any::<u32>(), any::<u32>()),
    ) {
        bounded("rasterisers", move || {
            let mut fb = blank(w, h);
            draw_line_aa(&mut fb, f[0], f[1], f[2], f[3], Rgba::BLACK);
            draw_line(&mut fb, i[0], i[1], i[2], i[3], Rgba::BLACK);
            draw_circle(&mut fb, i[0], i[1], r, Rgba::BLACK);
            trueno_viz::render::draw_circle_outline(&mut fb, i[2], i[3], r, Rgba::BLACK);
            draw_rect(&mut fb, rect.0, rect.1, rect.2, rect.3, Rgba::BLACK);
            trueno_viz::render::draw_rect_outline(&mut fb, rect.0, rect.1, rect.2, rect.3, Rgba::BLACK, rect.2);
            trueno_viz::render::draw_point(&mut fb, f[0], f[1], f[2], Rgba::BLACK);
        });
    }
}
