//! ggplot2 conformance oracle (PLOT-GG-001, paiml/aprender#6909).
//!
//! Each row of the conformance matrix is one fixture under `fixtures/ggplot2/<row>/`:
//! `data.csv` (the input), `generate.R` (the ggplot2 spec), and `expected.json` (what
//! `ggplot_build()` in ggplot2 4.0.3 computed). The fixture is committed; R is never run here.
//! The test rebuilds the same spec with the grammar in this module and compares
//! [`BuiltGGPlot::layout`] against the fixture.
//!
//! # Tolerance
//!
//! Positions, limits, expanded ranges and break positions: `|a - b| <= 1e-9 * max(|a|, |b|) + 1e-12`.
//! Both sides run the same few binary64 operations on inputs that are exact in binary32, so an
//! honest implementation differs by operation-order rounding only (a few ulp, ~1e-15 relative).
//! 1e-9 leaves six orders of magnitude for that, and is still six orders below the smallest
//! default we guard (the 5% expansion moves a limit by 0.6 on this data; a 4% expansion moves it
//! by 0.12). The 1e-12 absolute floor exists only for a break at exactly 0.
//! Labels, shapes, colours and counts compare exactly.
//!
//! Lives in `src/` (a `#[cfg(test)]` module, fixture via `include_str!`) so that it runs in
//! `cargo test -p aprender-viz --lib`, which CI runs; a new `tests/*.rs` target would not be.

use serde_json::Value;

use super::{Aes, BuiltLayout, DataFrame, GGPlot, Geom, PositionScale};

/// Relative tolerance for every continuous value. See the module docs.
const REL_TOL: f64 = 1e-9;
/// Absolute floor, for values at exactly zero.
const ABS_TOL: f64 = 1e-12;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= REL_TOL * a.abs().max(b.abs()) + ABS_TOL
}

fn assert_close_slice(what: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}: length got {got:?} want {want:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(close(*g, *w), "{what}[{i}]: got {g:?} want {w:?} (tol {REL_TOL} rel)");
    }
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("fixture field is an array")
        .iter()
        .map(|e| e.as_f64().expect("fixture array holds numbers"))
        .collect()
}

fn strs(v: &Value) -> Vec<String> {
    v.as_array()
        .expect("fixture field is an array")
        .iter()
        .map(|e| e.as_str().expect("fixture array holds strings").to_string())
        .collect()
}

/// Parse a two-column numeric CSV (header `x,y`) into a frame. Values must be exact in f32;
/// the parse asserts it, so a fixture edit cannot quietly turn the test into a rounding test.
fn frame_from_csv(csv: &str) -> DataFrame {
    let mut lines = csv.lines();
    let header: Vec<&str> = lines.next().expect("csv has a header").split(',').collect();
    let mut cols: Vec<Vec<f32>> = vec![Vec::new(); header.len()];
    for line in lines.filter(|l| !l.trim().is_empty()) {
        for (c, cell) in line.split(',').enumerate() {
            let v: f64 = cell.trim().parse().expect("csv cell is numeric");
            let f = v as f32;
            assert_eq!(f64::from(f), v, "fixture value {v} is not exact in f32");
            cols[c].push(f);
        }
    }
    let mut df = DataFrame::new();
    for (name, col) in header.iter().zip(&cols) {
        df.add_column_f32(name.trim(), col);
    }
    df
}

fn check_axis(name: &str, got: &PositionScale, want: &Value) {
    assert_close_slice(
        &format!("{name}.limits"),
        &[got.limits.0, got.limits.1],
        &nums(&want["limits"]),
    );
    assert_close_slice(
        &format!("{name}.continuous_range"),
        &[got.continuous_range.0, got.continuous_range.1],
        &nums(&want["continuous_range"]),
    );
    assert_close_slice(&format!("{name}.breaks"), &got.breaks, &nums(&want["breaks"]));
    assert_eq!(got.labels, strs(&want["labels"]), "{name}.labels");
    let censored = want["n_breaks_censored"].as_u64().expect("n_breaks_censored is a count");
    assert_eq!(got.n_breaks_censored as u64, censored, "{name}.n_breaks_censored");
}

// --- Row 1: geom_point + scale_x_continuous / scale_y_continuous -------------------------------

const POINT_JSON: &str = include_str!("../../fixtures/ggplot2/geom_point/expected.json");
const POINT_CSV: &str = include_str!("../../fixtures/ggplot2/geom_point/data.csv");

fn point_layout() -> (BuiltLayout, Value) {
    let want: Value = serde_json::from_str(POINT_JSON).expect("fixture is valid JSON");
    assert_eq!(
        want["versions"]["ggplot2"], "4.0.3",
        "fixture was not produced by the pinned ggplot2"
    );
    let layout = GGPlot::new()
        .data(frame_from_csv(POINT_CSV))
        .aes(Aes::new().x("x").y("y"))
        .geom(Geom::point())
        .build()
        .expect("plot builds")
        .layout()
        .expect("layout resolves");
    (layout, want)
}

#[test]
fn ggplot2_row1_point_layer_positions() {
    let (got, want) = point_layout();
    assert_eq!(got.layers.len(), 1);
    let (g, w) = (&got.layers[0], &want["layers"][0]);
    assert_eq!(w["geom"], "GeomPoint");
    assert_close_slice("x", &g.x, &nums(&w["x"]));
    assert_close_slice("y", &g.y, &nums(&w["y"]));
}

#[test]
fn ggplot2_row1_point_default_aesthetics() {
    let (got, want) = point_layout();
    let w = &want["layers"][0];
    let points = got.layers[0].points.as_ref().expect("a point layer resolves point aesthetics");
    assert_eq!(points.len(), nums(&w["x"]).len());
    for (i, p) in points.iter().enumerate() {
        assert_eq!(f64::from(p.shape), w["shape"][i].as_f64().expect("shape"), "shape[{i}]");
        let rgba = nums(&w["colour_rgba"][i]);
        assert_eq!(
            [p.colour.r, p.colour.g, p.colour.b, p.colour.a].map(f64::from).to_vec(),
            rgba,
            "colour[{i}] (R: {})",
            w["colour"][i]
        );
        assert_eq!(p.fill.is_none(), w["fill"][i].is_null(), "fill[{i}]");
        assert_eq!(p.alpha.is_none(), w["alpha"][i].is_null(), "alpha[{i}]");
        assert!(close(p.size, w["size"][i].as_f64().expect("size")), "size[{i}]");
        assert!(close(p.stroke, w["stroke"][i].as_f64().expect("stroke")), "stroke[{i}]");
    }
}

#[test]
fn ggplot2_row1_scale_x_continuous() {
    let (got, want) = point_layout();
    check_axis("x", &got.x, &want["panel"]["x"]);
}

#[test]
fn ggplot2_row1_scale_y_continuous() {
    let (got, want) = point_layout();
    check_axis("y", &got.y, &want["panel"]["y"]);
}
