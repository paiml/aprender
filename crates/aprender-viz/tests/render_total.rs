//! The GGPlot grammar renderers (framebuffer, SVG, terminal) and the draw primitives are total
//! (paiml/aprender#6941, Refs #6909): none panics or runs unbounded for any f32 input. The
//! `plots/` renderers (ScatterPlot, LineChart, Histogram, `from_prompt`) and the wasm entry points
//! that call them are NOT covered; they are tracked in paiml/aprender#6952.
//!
//! Twelve inputs panicked (`attempt to add with overflow` in `draw_circle` and `fill_rect`) or ran
//! for over a minute (`render_area` walking 2^31 rows, `draw_line_aa` walking off-canvas pixels).
//! Each is a named test with a hard time bound. The property tests then sweep arbitrary f32
//! (NaN, +/-Inf, +/-MAX, subnormals), arbitrary coord limits and canvas sizes 1x1..512x512 across
//! every geom, through `to_framebuffer`, `to_svg` and `to_terminal`. The last group proves the
//! clipping changes nothing for data on or near the canvas: the clipped rasterisers are compared
//! pixel for pixel with reference copies of the unclipped algorithms. Far off the canvas the
//! aliased line is still pixel-identical to the unclipped walk (it is exact in closed form, no
//! clipping), and the antialiased line is bounded against the ideal segment (RT-DEV-001).

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

// Each repro is written as `#[test] fn name() = (geom, x, y, coord);` so the test name is
// spelled out in the source, where a contract's `cargo test <name>` citation can resolve it.
macro_rules! repro {
    ($(#[test] fn $name:ident() = ($geom:expr, $x:expr, $y:expr, $coord:expr $(,)?);)+) => {$(
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
    )+};
}

repro! {
    // 1. draw_circle: `cx + x` overflowed i32.
    #[test]
    fn point_coordlim_outside() = (
        Geom::point(),
        vec![0.5, 1e38, -1e38],
        vec![0.5, 1e38, -1e38],
        limits(),
    );
    // 2. fill_rect: `x + w` overflowed u32.
    #[test]
    fn bar_coordlim_outside() = (
        Geom::bar(),
        vec![0.5, 1e38, -1e38],
        vec![0.5, 1e38, -1e38],
        limits(),
    );
    // 3. and 4. line and area with finite points far off the canvas walked billions of pixels.
    #[test]
    fn line_coordlim_outside() = (
        Geom::line(),
        vec![0.5, 1e38, -1e38],
        vec![0.5, 1e38, -1e38],
        limits(),
    );
    #[test]
    fn area_coordlim_outside() = (
        Geom::area(),
        vec![0.5, 1e38, -1e38],
        vec![0.5, 1e38, -1e38],
        limits(),
    );
    // 5. the y = 0 baseline of this area maps to an extreme pixel: the fill looped about 2^31 rows.
    #[test]
    fn area_huge_y_close() = (
        Geom::area(),
        vec![1.0, 2.0, 3.0],
        vec![3.0e38, 3.0000002e38, 3.0000004e38],
        Coord::cartesian(),
    );
    // 6. to 12. non-finite rows, with and without coord_flip().
    #[test]
    fn point_mixed() = (Geom::point(), mixed().0, mixed().1, Coord::cartesian());
    #[test]
    fn line_mixed() = (Geom::line(), mixed().0, mixed().1, Coord::cartesian());
    #[test]
    fn area_mixed() = (Geom::area(), mixed().0, mixed().1, Coord::cartesian());
    #[test]
    fn point_flip_inf() = (Geom::point(), mixed().0, mixed().1, Coord::cartesian().flip());
    #[test]
    fn bar_flip_inf() = (Geom::bar(), mixed().0, mixed().1, Coord::cartesian().flip());
    #[test]
    fn line_flip_inf() = (Geom::line(), mixed().0, mixed().1, Coord::cartesian().flip());
    #[test]
    fn area_flip_inf() = (Geom::area(), mixed().0, mixed().1, Coord::cartesian().flip());
}

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
    let (sx, sy) = (if x0 < x1 { 1 } else { -1 }, if y0 < y1 { 1 } else { -1 });
    let (mut x, mut y, mut err) = (x0, y0, dx + dy);
    loop {
        if x >= 0 && y >= 0 {
            fb.set_pixel(x as u32, y as u32, color);
        }
        let e2 = 2 * err;
        let (step_x, step_y) = (e2 >= dy, e2 <= dx);
        // Reaching the end point on either axis stops the walk before the step is taken.
        if (x == x1 && (y == y1 || step_x)) || (y == y1 && step_y) {
            break;
        }
        if step_x {
            err += dy;
            x += sx;
        }
        if step_y {
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

// ---------------------------------------------------------------------------
// Dropped rows leave no trace in any geom (FALSIFY-RT-001, the `panel_points` filter).
// ---------------------------------------------------------------------------

#[test]
fn dropped_rows_change_nothing_for_every_geom() {
    // The filter in `panel_points` is not redundant: a bar's width is `range / n * width`, and a
    // polyline joins its neighbours, so a kept NaN row would change both. Rendering with the bad
    // rows must equal rendering without them, byte for byte.
    let clean = (vec![1.0_f32, 2.0, 3.0], vec![2.0_f32, 5.0, 3.0]);
    let dirty = (vec![1.0_f32, NAN, 2.0, INF, 3.0, 2.5], vec![2.0_f32, 4.0, 5.0, 4.0, 3.0, NAN]);
    for (name, geom) in [
        ("bar", Geom::bar()),
        ("line", Geom::line()),
        ("area", Geom::area()),
        ("point", Geom::point()),
    ] {
        let want = xy(&clean.0, &clean.1, geom.clone()).build().expect("builds");
        let got = xy(&dirty.0, &dirty.1, geom).build().expect("builds");
        assert_eq!(
            got.to_framebuffer().expect("fb").pixels(),
            want.to_framebuffer().expect("fb").pixels(),
            "{name}: dropped rows changed the picture"
        );
    }
}

// ---------------------------------------------------------------------------
// The radius threshold of draw_circle / draw_circle_outline (4096).
// ---------------------------------------------------------------------------
//
// Up to 4096 the midpoint circle; above it, one span (or two end points) per canvas row at the
// exact integer half-width. Centres are chosen so the arc crosses a 400x300 canvas: the right
// arc, the bottom arc and a diagonal arc. The exact disc is computed here from `dx^2 + dy^2 <= r^2`.

const CW: u32 = 400;
const CH: u32 = 300;
const PLACEMENTS: [(i32, i32); 3] = [(-3700, 150), (200, -3900), (-2700, -2700)];

fn canvas() -> Framebuffer {
    blank(CW, CH)
}

fn is_lit(fb: &Framebuffer, x: u32, y: u32) -> bool {
    fb.get_pixel(x, y) != Some(Rgba::WHITE)
}

/// Largest `s` with `s * s <= n`.
fn int_sqrt(n: i64) -> i64 {
    let mut s = (n as f64).sqrt() as i64;
    while s * s > n {
        s -= 1;
    }
    while (s + 1) * (s + 1) <= n {
        s += 1;
    }
    s
}

/// The exact filled disc, straight from the definition.
fn exact_disc(cx: i32, cy: i32, r: i32) -> Framebuffer {
    let mut fb = canvas();
    let r2 = i64::from(r) * i64::from(r);
    for y in 0..CH {
        for x in 0..CW {
            let (dx, dy) = (i64::from(x) - i64::from(cx), i64::from(y) - i64::from(cy));
            if dx * dx + dy * dy <= r2 {
                fb.set_pixel(x, y, Rgba::BLACK);
            }
        }
    }
    fb
}

/// The two end points of the exact disc on every canvas row it reaches.
fn exact_row_ends(cx: i32, cy: i32, r: i32) -> Framebuffer {
    let mut fb = canvas();
    let r2 = i64::from(r) * i64::from(r);
    for y in 0..CH {
        let dy = i64::from(y) - i64::from(cy);
        if dy * dy > r2 {
            continue;
        }
        let half = int_sqrt(r2 - dy * dy);
        for x in [i64::from(cx) - half, i64::from(cx) + half] {
            if (0..i64::from(CW)).contains(&x) {
                fb.set_pixel(x as u32, y, Rgba::BLACK);
            }
        }
    }
    fb
}

/// `draw_circle_outline` exactly as on main (midpoint, eight points per step).
fn ref_circle_outline(fb: &mut Framebuffer, cx: i32, cy: i32, radius: i32, color: Rgba) {
    let plot = |fb: &mut Framebuffer, x: i32, y: i32| {
        if x >= 0 && y >= 0 && x < fb.width() as i32 && y < fb.height() as i32 {
            fb.set_pixel(x as u32, y as u32, color);
        }
    };
    let (mut x, mut y, mut err) = (radius, 0, 1 - radius);
    while x >= y {
        for (px, py) in [(x, y), (-x, y), (x, -y), (-x, -y), (y, x), (-y, x), (y, -x), (-y, -x)] {
            plot(fb, cx + px, cy + py);
        }
        y += 1;
        if err < 0 {
            err += 2 * y + 1;
        } else {
            x -= 1;
            err += 2 * (y - x) + 1;
        }
    }
}

fn filled(cx: i32, cy: i32, r: i32) -> Framebuffer {
    let mut fb = canvas();
    draw_circle(&mut fb, cx, cy, r, Rgba::BLACK);
    fb
}

fn outlined(cx: i32, cy: i32, r: i32) -> Framebuffer {
    let mut fb = canvas();
    trueno_viz::render::draw_circle_outline(&mut fb, cx, cy, r, Rgba::BLACK);
    fb
}

fn midpoint_filled(cx: i32, cy: i32, r: i32) -> Framebuffer {
    let mut fb = canvas();
    ref_circle(&mut fb, cx, cy, r, Rgba::BLACK);
    fb
}

fn midpoint_outline(cx: i32, cy: i32, r: i32) -> Framebuffer {
    let mut fb = canvas();
    ref_circle_outline(&mut fb, cx, cy, r, Rgba::BLACK);
    fb
}

/// Pixels that differ, as `(x, y)`.
fn differing(a: &Framebuffer, b: &Framebuffer) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for y in 0..CH {
        for x in 0..CW {
            if is_lit(a, x, y) != is_lit(b, x, y) {
                out.push((x, y));
            }
        }
    }
    out
}

#[test]
fn filled_circle_uses_the_midpoint_up_to_4096_and_the_exact_rows_above() {
    for (cx, cy) in PLACEMENTS {
        // Not vacuous: at 4096 the midpoint circle and the exact disc really are different
        // pictures here, so the two comparisons below tell the two paths apart.
        assert!(
            !differing(&midpoint_filled(cx, cy, 4096), &exact_disc(cx, cy, 4096)).is_empty(),
            "placement ({cx}, {cy}): midpoint and exact disc coincide, the test cannot see the path"
        );
        for r in [4095, 4096] {
            assert!(
                differing(&filled(cx, cy, r), &midpoint_filled(cx, cy, r)).is_empty(),
                "r={r} at ({cx}, {cy}) is not the midpoint circle"
            );
        }
        assert!(
            differing(&filled(cx, cy, 4097), &exact_disc(cx, cy, 4097)).is_empty(),
            "r=4097 at ({cx}, {cy}) is not the exact disc"
        );
    }
}

#[test]
fn circle_outline_uses_the_midpoint_up_to_4096_and_the_row_ends_above() {
    for (cx, cy) in PLACEMENTS {
        assert!(
            !differing(&midpoint_outline(cx, cy, 4096), &exact_row_ends(cx, cy, 4096)).is_empty(),
            "placement ({cx}, {cy}): the two outlines coincide, the test cannot see the path"
        );
        for r in [4095, 4096] {
            assert!(
                differing(&outlined(cx, cy, r), &midpoint_outline(cx, cy, r)).is_empty(),
                "outline r={r} at ({cx}, {cy}) is not the midpoint outline"
            );
        }
        assert!(
            differing(&outlined(cx, cy, 4097), &exact_row_ends(cx, cy, 4097)).is_empty(),
            "outline r=4097 at ({cx}, {cy}) is not the row end points"
        );
    }
}

#[test]
fn the_two_filled_circle_paths_agree_within_one_pixel_of_the_edge() {
    // Stated tolerance (RT-DEV-002): the paths differ only on pixels whose centre is within one
    // pixel of the circle (`|distance - r| < 1`). Output does not jump by more than that where
    // the path changes. (A row's end can move by more than a pixel where the arc is nearly
    // horizontal; the distance to the circle is the right measure.)
    for (cx, cy) in PLACEMENTS {
        let r = 4096_i64;
        let mid = midpoint_filled(cx, cy, 4096);
        let exact = exact_disc(cx, cy, 4096);
        let bad = differing(&mid, &exact);
        for &(x, y) in &bad {
            let (dx, dy) = (i64::from(x) - i64::from(cx), i64::from(y) - i64::from(cy));
            let d2 = dx * dx + dy * dy;
            assert!(
                (r - 1) * (r - 1) < d2 && d2 < (r + 1) * (r + 1),
                "({x}, {y}) differs but is {} px from the circle",
                (d2 as f64).sqrt() - r as f64
            );
        }
        // 4097 against 4096: the same bound plus the one pixel the radius grew.
        let up = filled(cx, cy, 4097);
        for &(x, y) in &differing(&up, &mid) {
            let (dx, dy) = (i64::from(x) - i64::from(cx), i64::from(y) - i64::from(cy));
            let d2 = dx * dx + dy * dy;
            assert!(
                (r - 1) * (r - 1) < d2 && d2 < (r + 2) * (r + 2),
                "({x}, {y}) differs between r=4096 and r=4097 but is {} px from r",
                (d2 as f64).sqrt() - r as f64
            );
        }
    }
}
// ---------------------------------------------------------------------------
// Segments far off the canvas. Bresenham is exact for every i32 (RT-EQV-001); Wu, clipped at the
// 1024 px guard, stays on the ideal segment (RT-DEV-001).
// ---------------------------------------------------------------------------

/// A deterministic stream in `[0, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1_u64 << 53) as f64
    }
}

type Seg = (f64, f64, f64, f64);

/// One end on a `w` x `h` canvas, the other `near..far` px away in any direction (the family
/// the round-2 review drew).
fn one_end_on(rng: &mut Lcg, w: f64, h: f64, near: f64, far: f64) -> Seg {
    let (ax, ay) = (rng.next() * w, rng.next() * h);
    let (angle, dist) = (rng.next() * std::f64::consts::TAU, near + rng.next() * (far - near));
    (ax, ay, ax + dist * angle.cos(), ay + dist * angle.sin())
}

/// A segment that hugs edge `edge` (0 right, 1 left, 2 bottom, 3 top) of a `w` x `h` canvas: it
/// starts within 8 px of the edge and runs within 0.1 rad of parallel to it, out to 2000..20000 px.
/// This is where the old guard clip moved Bresenham's step phase by tens of rows.
fn edge_hugging(rng: &mut Lcg, w: f64, h: f64, edge: u32) -> Seg {
    let inset = rng.next() * 8.0;
    let along = rng.next();
    let (ax, ay, base) = match edge {
        0 => (w - inset, along * h, std::f64::consts::FRAC_PI_2),
        1 => (inset, along * h, std::f64::consts::FRAC_PI_2),
        2 => (along * w, h - inset, 0.0),
        _ => (along * w, inset, 0.0),
    };
    let flip = if rng.next() < 0.5 { 0.0 } else { std::f64::consts::PI };
    let angle = base + flip + (rng.next() - 0.5) * 0.2;
    let dist = 2000.0 + rng.next() * 18000.0;
    (ax, ay, ax + dist * angle.cos(), ay + dist * angle.sin())
}

/// Both ends `near..far` px off a `w` x `h` canvas on opposite sides, through a canvas point.
fn through(rng: &mut Lcg, w: f64, h: f64, near: f64, far: f64) -> Seg {
    let (cx, cy) = (rng.next() * w, rng.next() * h);
    let angle = rng.next() * std::f64::consts::TAU;
    let (d0, d1) = (near + rng.next() * (far - near), near + rng.next() * (far - near));
    let (c, s) = (angle.cos(), angle.sin());
    (cx - d0 * c, cy - d0 * s, cx + d1 * c, cy + d1 * s)
}

/// The three segments the round-2 review found, then 1000 of its family, 500 hugging each edge,
/// and 500 passing through, on 512x512 unless stated.
fn far_segments() -> Vec<(u32, u32, Seg)> {
    let mut out = vec![
        (512, 512, (503.0, 483.0, 606.0, -4592.0)),
        (800, 600, (792.0, 286.0, 1231.0, -15902.0)),
        (
            512,
            512,
            (
                511.100_590_923_432_55,
                301.022_343_328_706_4,
                940.659_806_548_316_5,
                -12_009.057_765_658_255,
            ),
        ),
    ];
    let mut rng = Lcg(0xA5A5_0001);
    out.extend((0..1000).map(|_| (512, 512, one_end_on(&mut rng, 512.0, 512.0, 2000.0, 20000.0))));
    let mut rng = Lcg(0xA5A5_0003);
    out.extend((0..300).map(|_| (800, 600, one_end_on(&mut rng, 800.0, 600.0, 2000.0, 20000.0))));
    let mut rng = Lcg(0xED6E);
    for edge in 0..4 {
        out.extend((0..500).map(|_| (512, 512, edge_hugging(&mut rng, 512.0, 512.0, edge))));
    }
    let mut rng = Lcg(0x7480_0001);
    out.extend((0..500).map(|_| (512, 512, through(&mut rng, 512.0, 512.0, 1100.0, 20000.0))));
    out
}

#[test]
fn bresenham_line_equals_the_unclipped_line_far_off_the_canvas() {
    // Every segment above against main's walk, unclipped: pixel for pixel, the three review
    // repros and the edge-hugging lines included.
    let r = |v: f64| v.round() as i32;
    let mut fail = Vec::new();
    for (w, h, (ax, ay, bx, by)) in far_segments() {
        let (mut got, mut want) = (blank(w, h), blank(w, h));
        draw_line(&mut got, r(ax), r(ay), r(bx), r(by), Rgba::BLACK);
        ref_bresenham(&mut want, r(ax), r(ay), r(bx), r(by), Rgba::BLACK);
        if got.pixels() != want.pixels() {
            fail.push((w, h, (r(ax), r(ay), r(bx), r(by))));
        }
    }
    assert!(
        fail.is_empty(),
        "{} segments differ from the unclipped walk, first {:?}",
        fail.len(),
        fail.first()
    );
}

/// For each major-axis line of the canvas, the pixel the ideal line of `(x0, y0) -> (x1, y1)`
/// rounds to there (a tie away from the start), if it is on the canvas. Brute force over the
/// canvas in `i128`, sharing no code with `draw_line`.
fn nearest_pixels(w: u32, h: u32, x0: i32, y0: i32, x1: i32, y1: i32) -> Framebuffer {
    let mut fb = blank(w, h);
    let (x0, y0, x1, y1) = (i128::from(x0), i128::from(y0), i128::from(x1), i128::from(y1));
    let (dx, dy) = ((x1 - x0).abs(), (y1 - y0).abs());
    let x_major = dx >= dy;
    let (u0, v0, du, dv, ulen, vlen) =
        if x_major { (x0, y0, x1 - x0, y1 - y0, w, h) } else { (y0, x0, y1 - y0, x1 - x0, h, w) };
    let n = du.abs();
    for u in 0..i128::from(ulen) {
        // Steps from the start to this line; a single point (n = 0) has only u0.
        let k = if n == 0 { u - u0 } else { (u - u0) * du.signum() };
        if !(0..=n).contains(&k) {
            continue;
        }
        // minor offset = round(|dv| k / n), ties away from the start; 0 for a single point.
        let j = if n == 0 { 0 } else { (2 * dv.abs() * k + n).div_euclid(2 * n) };
        let v = v0 + if dv < 0 { -j } else { j };
        if (0..i128::from(vlen)).contains(&v) {
            let (x, y) = if x_major { (u, v) } else { (v, u) };
            fb.set_pixel(x as u32, y as u32, Rgba::BLACK);
        }
    }
    fb
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    // Any i32 at all, where main's walk would overflow or take 2^32 steps: the shipped line is
    // the nearest pixel on every canvas column (row, for a steep line) the segment spans.
    #[test]
    fn bresenham_line_is_the_nearest_pixel_per_column_for_any_i32(
        w in 1_u32..=48, h in 1_u32..=48,
        a in (any::<i32>(), any::<i32>()), b in (any::<i32>(), any::<i32>()),
        near in (-60_i32..110, -60_i32..110),
        pick in 0_u8..3,
    ) {
        // Mix wholly random ends with one end near the canvas, so the canvas is often crossed.
        let (x0, y0) = if pick == 0 { a } else { near };
        let (x1, y1) = b;
        let mut got = blank(w, h);
        draw_line(&mut got, x0, y0, x1, y1, Rgba::BLACK);
        let want = nearest_pixels(w, h, x0, y0, x1, y1);
        prop_assert_eq!(got.pixels(), want.pixels());
    }

    #[test]
    fn nearest_pixels_equals_the_walk_near_the_canvas(
        w in 1_u32..=64, h in 1_u32..=64,
        a in (-200_i32..260, -200_i32..260), b in (-200_i32..260, -200_i32..260),
    ) {
        // Ties the brute-force reference above to main's walk where the walk can run.
        let mut want = blank(w, h);
        ref_bresenham(&mut want, a.0, a.1, b.0, b.1, Rgba::BLACK);
        let got = nearest_pixels(w, h, a.0, a.1, b.0, b.1);
        prop_assert_eq!(got.pixels(), want.pixels());
    }
}

/// Distance from the pixel centre `(px, py)` to the segment `s` (Wu's pixels sit on integers).
fn seg_dist(px: f64, py: f64, (x0, y0, x1, y1): Seg) -> f64 {
    let (dx, dy) = (x1 - x0, y1 - y0);
    let len2 = dx * dx + dy * dy;
    let t =
        if len2 == 0.0 { 0.0 } else { (((px - x0) * dx + (py - y0) * dy) / len2).clamp(0.0, 1.0) };
    (px - (x0 + t * dx)).hypot(py - (y0 + t * dy))
}

/// The lit pixels of `fb`, grouped by major-axis line: `lines[u]` holds the minor coordinates.
fn lit_by_line(fb: &Framebuffer, steep: bool) -> Vec<Vec<u32>> {
    let (w, h) = (fb.width(), fb.height());
    let mut lines = vec![Vec::new(); if steep { h } else { w } as usize];
    for (i, px) in fb.pixels().chunks_exact(4).enumerate() {
        if px != [255, 255, 255, 255] {
            let (x, y) = ((i as u32) % w, (i as u32) / w);
            let (u, v) = if steep { (y, x) } else { (x, y) };
            lines[u as usize].push(v);
        }
    }
    lines
}

/// How far the antialiased line on `fb` strays from the ideal segment `s`: the largest distance
/// from a drawn pixel to `s` (`.0`), and the largest distance from `s` to the nearest drawn
/// pixel on the same major-axis line (`.1`), over every line where `s` is at least 2 px inside
/// the canvas and 2 px from its ends.
fn wu_stray(fb: &Framebuffer, s: Seg) -> (f64, f64) {
    let (x0, y0, x1, y1) = s;
    let steep = (y1 - y0).abs() > (x1 - x0).abs();
    let lines = lit_by_line(fb, steep);
    let (u0, v0, u1, v1, vlen) =
        if steep { (y0, x0, y1, x1, fb.width()) } else { (x0, y0, x1, y1, fb.height()) };
    let at =
        |u: usize, v: u32| if steep { (f64::from(v), u as f64) } else { (u as f64, f64::from(v)) };
    let mut off = 0.0_f64;
    for (u, vs) in lines.iter().enumerate() {
        for &v in vs {
            let (px, py) = at(u, v);
            off = off.max(seg_dist(px, py, s));
        }
    }
    let (umin, umax) = (u0.min(u1) + 2.0, u0.max(u1) - 2.0);
    let mut gap = 0.0_f64;
    for (u, vs) in lines.iter().enumerate() {
        let uf = u as f64;
        if uf < umin || uf > umax {
            continue;
        }
        let v = v0 + (uf - u0) * (v1 - v0) / (u1 - u0);
        if v < 2.0 || v > f64::from(vlen) - 3.0 {
            continue;
        }
        gap = gap.max(vs.iter().map(|&p| (f64::from(p) - v).abs()).fold(f64::INFINITY, f64::min));
    }
    (off, gap)
}

/// RT-DEV-001, drawn-to-ideal: Wu lights the two pixels around the line on each major-axis line
/// (each within 1 px of it), and at an end point the line's own end-pixel rule can put one
/// pixel up to sqrt(0.5^2 + 1.5^2) = 1.58 px from the segment. Measured: 1.31 px.
const WU_DRAWN_BOUND: f64 = 2.0;

/// RT-DEV-001, ideal-to-drawn: one of the two pixels has at least half the intensity, so the
/// nearest drawn pixel is within 0.5 px of the line on every major-axis line. Measured: 0.50 px.
const WU_COVER_BOUND: f64 = 1.0;

/// The worst of each [`wu_stray`] measure of `draw` over `segs`, each with the segment that set it.
fn worst_wu_stray(
    segs: &[(u32, u32, Seg)],
    draw: fn(&mut Framebuffer, f32, f32, f32, f32, Rgba),
) -> ((f64, Seg), (f64, Seg)) {
    let none = (0.0, 0.0, 0.0, 0.0);
    let (mut off, mut gap) = ((0.0_f64, none), (0.0_f64, none));
    for &(w, h, s) in segs {
        let f = (s.0 as f32, s.1 as f32, s.2 as f32, s.3 as f32);
        let mut fb = blank(w, h);
        draw(&mut fb, f.0, f.1, f.2, f.3, Rgba::BLACK);
        let (o, g) =
            wu_stray(&fb, (f64::from(f.0), f64::from(f.1), f64::from(f.2), f64::from(f.3)));
        if o > off.0 {
            off = (o, s);
        }
        if g > gap.0 {
            gap = (g, s);
        }
    }
    (off, gap)
}

#[test]
fn wu_line_stays_on_the_ideal_segment_far_off_the_canvas() {
    // The f32 walk restarts at the guard cut, so a segment that crosses the guard is not
    // bit-exact with main's walk from the far end. What is bounded is the drawing against the
    // true segment, both ways: no drawn pixel farther than the bound from it, and no stretch of
    // it on the canvas without a drawn pixel within the bound.
    let mut segs = far_segments();
    // And ends far beyond anything main could walk.
    let mut rng = Lcg(0x7480_0002);
    segs.extend((0..300).map(|_| (512, 512, through(&mut rng, 512.0, 512.0, 1e6, 1e30))));
    let ((off, off_at), (gap, gap_at)) = worst_wu_stray(&segs, draw_line_aa);
    println!(
        "RT-DEV-001 measured on {} segments: drawn-to-ideal {off:.4} px at {off_at:?}, \
         ideal-to-drawn {gap:.4} px at {gap_at:?}",
        segs.len()
    );
    assert!(
        off <= WU_DRAWN_BOUND,
        "a drawn pixel is {off:.3} px from the ideal segment {off_at:?}"
    );
    assert!(
        gap <= WU_COVER_BOUND,
        "the ideal segment is {gap:.3} px from any drawn pixel {gap_at:?}"
    );
}
