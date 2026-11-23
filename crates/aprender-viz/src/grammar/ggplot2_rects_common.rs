//! Shared helpers of the tile, rect and raster oracle modules (PLOT-GG-001, paiml/aprender#6909).
//! Fixtures are `fixtures/ggplot2/<row>/expected_<case>.json`, made by ggplot2 4.0.3 under C
//! collation (`rects_oracle.R`); R is never run here.
//!
//! Tolerance is the row-1 pattern: `|a - b| <= 1e-9 * max(|a|, |b|) + 1e-12` for every position,
//! bound, limit, range, break, width, linewidth and alpha; labels, colours, groups, panels,
//! linetypes, warnings and counts compare exactly. `NaN` matches `null` (R's `NA`) by position.

use serde_json::Value;

use super::rects::{BuiltRects, RectKind};
use super::{Aes, BuiltLayer, BuiltLayout, DataFrame, Facet, GGPlot, Geom, Layer, PositionScale};
use crate::color::Rgba;

pub(super) const REL_TOL: f64 = 1e-9;
pub(super) const ABS_TOL: f64 = 1e-12;

pub(super) const RED: Rgba = Rgba { r: 255, g: 0, b: 0, a: 255 };
pub(super) const BLUE: Rgba = Rgba { r: 0, g: 0, b: 255, a: 255 };

pub(super) fn close(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= REL_TOL * a.abs().max(b.abs()) + ABS_TOL
}

pub(super) fn assert_close_slice(what: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}: length got {got:?} want {want:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(close(*g, *w), "{what}[{i}]: got {g:?} want {w:?} (tol {REL_TOL} rel)");
    }
}

/// A JSON array of numbers; `null` (R's `NA`) is `NaN`.
pub(super) fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("fixture field is an array")
        .iter()
        .map(|e| if e.is_null() { f64::NAN } else { e.as_f64().expect("fixture number") })
        .collect()
}

pub(super) fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .expect("fixture field is an array")
        .iter()
        .map(|e| e.as_i64().expect("fixture integer"))
        .collect()
}

pub(super) fn strs(v: &Value) -> Vec<String> {
    v.as_array()
        .expect("fixture field is an array")
        .iter()
        .map(|e| e.as_str().expect("fixture string").to_string())
        .collect()
}

/// Parse a CSV into a frame: every cell must be a number (`NA` is `NaN`) exact in f32.
pub(super) fn frame_from_csv(csv: &str) -> DataFrame {
    let mut lines = csv.lines();
    let header: Vec<&str> = lines.next().expect("csv has a header").split(',').collect();
    let rows: Vec<Vec<&str>> = lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').map(str::trim).collect())
        .collect();
    let mut df = DataFrame::new();
    for (c, name) in header.iter().enumerate() {
        let col: Vec<f32> = rows
            .iter()
            .map(|r| {
                if r[c] == "NA" {
                    return f32::NAN;
                }
                let v: f64 = r[c].parse().expect("numeric cell");
                let f = v as f32;
                assert_eq!(f64::from(f), v, "fixture value {v} is not exact in f32");
                f
            })
            .collect();
        df.add_column_f32(name.trim(), &col);
    }
    df
}

macro_rules! fixture {
    ($row:literal, $case:literal) => {
        include_str!(concat!("../../fixtures/ggplot2/", $row, "/expected_", $case, ".json"))
    };
}

macro_rules! csv {
    ($row:literal, $file:literal) => {
        $crate::grammar::ggplot2_rects_common::frame_from_csv(include_str!(concat!(
            "../../fixtures/ggplot2/",
            $row,
            "/",
            $file,
            ".csv"
        )))
    };
}

pub(super) use csv;
pub(super) use fixture;

pub(super) fn want(json: &str, row: &str, case: &str) -> Value {
    let v: Value = serde_json::from_str(json).expect("fixture is valid JSON");
    assert_eq!(v["versions"]["ggplot2"], "4.0.3", "fixture was not produced by the pinned ggplot2");
    assert_eq!(v["collation"], "C", "fixture was not made under C collation");
    assert_eq!(v["row"], row);
    assert_eq!(v["case"], case);
    v
}

pub(super) fn xy() -> Aes {
    Aes::new().x("x").y("y")
}

pub(super) fn bounds() -> Aes {
    Aes::new().xmin("xmin").xmax("xmax").ymin("ymin").ymax("ymax")
}

pub(super) fn build(
    data: DataFrame,
    aes: Aes,
    layers: Vec<Layer>,
) -> crate::error::Result<BuiltLayout> {
    build_with(data, aes, layers, Facet::None)
}

pub(super) fn build_with(
    data: DataFrame,
    aes: Aes,
    layers: Vec<Layer>,
    facet: Facet,
) -> crate::error::Result<BuiltLayout> {
    let mut p = GGPlot::new().data(data).aes(aes).facet(facet);
    for l in layers {
        p = p.layer(l);
    }
    p.build()?.layout()
}

pub(super) fn layer(g: Geom) -> Layer {
    Layer::new(g)
}

pub(super) fn check_axis(name: &str, got: &PositionScale, w: &Value) {
    assert_close_slice(
        &format!("{name}.limits"),
        &[got.limits.0, got.limits.1],
        &nums(&w["limits"]),
    );
    assert_close_slice(
        &format!("{name}.continuous_range"),
        &[got.continuous_range.0, got.continuous_range.1],
        &nums(&w["continuous_range"]),
    );
    assert_close_slice(&format!("{name}.breaks"), &got.breaks, &nums(&w["breaks"]));
    assert_eq!(got.labels, strs(&w["labels"]), "{name}.labels");
    let censored = w["n_breaks_censored"].as_u64().expect("count");
    assert_eq!(got.n_breaks_censored as u64, censored, "{name}.n_breaks_censored");
}

fn rgba(c: Rgba) -> Vec<f64> {
    [c.r, c.g, c.b, c.a].map(f64::from).to_vec()
}

/// One tile, rect or raster layer against its fixture layer: every numeric field.
pub(super) fn check_rects(tag: &str, got: &BuiltLayer, w: &Value, geom: &str, kind: RectKind) {
    assert_eq!(w["geom"], geom, "{tag}: R geom class");
    let r = got.rects.as_ref().expect("a tile, rect or raster layer resolves rect data");
    assert_eq!(r.kind, kind, "{tag}: kind");
    let n = nums(&w["xmin"]).len();
    assert!(n > 0, "{tag}: a fixture with no rows proves nothing");
    assert_eq!(r.xmin.len(), n, "{tag}: row count");
    assert_eq!(ints(&w["panel"]), vec![1; n], "{tag}.panel: one panel");
    assert_eq!(ints(&w["group"]), vec![-1; n], "{tag}.group: no group");
    assert_close_slice(&format!("{tag}.xmin"), &r.xmin, &nums(&w["xmin"]));
    assert_close_slice(&format!("{tag}.xmax"), &r.xmax, &nums(&w["xmax"]));
    assert_close_slice(&format!("{tag}.ymin"), &r.ymin, &nums(&w["ymin"]));
    assert_close_slice(&format!("{tag}.ymax"), &r.ymax, &nums(&w["ymax"]));
    check_rect_position(tag, got, w, kind);
    check_rect_extent(tag, r, w, kind, n);
    check_rect_alpha(tag, r, w, n);
    assert_eq!(w["fill_rgba"].as_array().map(Vec::len), Some(n), "{tag}.fill rows");
    for i in 0..n {
        assert_eq!(rgba(r.fill), nums(&w["fill_rgba"][i]), "{tag}.fill[{i}]");
    }
    check_rect_outline(tag, r, w, kind, n);
}

/// A rect layer carries no x/y; tile and raster layers do.
fn check_rect_position(tag: &str, got: &BuiltLayer, w: &Value, kind: RectKind) {
    if kind == RectKind::Rect {
        assert!(got.x.is_empty() && got.y.is_empty(), "{tag}: a rect layer carries no x/y");
        assert!(w.get("x").is_none() && w.get("y").is_none(), "{tag}: R has x/y on a rect");
    } else {
        assert_close_slice(&format!("{tag}.x"), &got.x, &nums(&w["x"]));
        assert_close_slice(&format!("{tag}.y"), &got.y, &nums(&w["y"]));
    }
}

/// Tiles have width/height columns; rects and rasters have none.
fn check_rect_extent(tag: &str, r: &BuiltRects, w: &Value, kind: RectKind, n: usize) {
    match kind {
        RectKind::Tile => {
            assert_close_slice(
                &format!("{tag}.width"),
                &vec![r.width.expect("tile width"); n],
                &nums(&w["width"]),
            );
            assert_close_slice(
                &format!("{tag}.height"),
                &vec![r.height.expect("tile height"); n],
                &nums(&w["height"]),
            );
        }
        _ => {
            assert!(r.width.is_none() && r.height.is_none(), "{tag}: no width/height columns");
            assert!(w.get("width").is_none() && w.get("height").is_none(), "{tag}: R has width");
        }
    }
}

fn check_rect_alpha(tag: &str, r: &BuiltRects, w: &Value, n: usize) {
    let alpha = nums(&w["alpha"]);
    assert_eq!(alpha.len(), n, "{tag}.alpha rows");
    for (i, a) in alpha.iter().enumerate() {
        match r.alpha {
            None => assert!(a.is_nan(), "{tag}.alpha[{i}]: R has {a}, got NA"),
            Some(v) => assert!(close(v, *a), "{tag}.alpha[{i}]: got {v} want {a}"),
        }
    }
}

/// Rasters have no outline; tiles and rects carry colour, linewidth and linetype.
fn check_rect_outline(tag: &str, r: &BuiltRects, w: &Value, kind: RectKind, n: usize) {
    if kind == RectKind::Raster {
        assert!(r.colour.is_none() && r.linewidth.is_none() && r.linetype.is_none(), "{tag}");
        let absent = ["colour_rgba", "linewidth", "linetype"].iter().all(|k| w.get(*k).is_none());
        assert!(absent, "{tag}: R raster has an outline column");
    } else {
        assert_eq!(w["colour_rgba"].as_array().map(Vec::len), Some(n), "{tag}.colour rows");
        for i in 0..n {
            let want = &w["colour_rgba"][i];
            match r.colour {
                None => assert!(want.is_null(), "{tag}.colour[{i}]: R has {want}, got NA"),
                Some(c) => assert_eq!(rgba(c), nums(want), "{tag}.colour[{i}]"),
            }
            let lw = r.linewidth.expect("outline width");
            assert!(close(lw, nums(&w["linewidth"])[i]), "{tag}.linewidth[{i}]");
            let lt = f64::from(r.linetype.expect("outline linetype"));
            assert_eq!(lt, nums(&w["linetype"])[i], "{tag}.linetype[{i}]");
        }
    }
}

/// The `geom_point` companion layer of a fixture: positions, shape, size, stroke.
pub(super) fn check_point(tag: &str, got: &BuiltLayer, w: &Value) {
    assert_eq!(w["geom"], "GeomPoint", "{tag}: R geom class");
    assert!(got.rects.is_none(), "{tag}: a point layer has no rect data");
    assert_close_slice(&format!("{tag}.x"), &got.x, &nums(&w["x"]));
    assert_close_slice(&format!("{tag}.y"), &got.y, &nums(&w["y"]));
    let p = got.points.as_ref().expect("point aesthetics");
    assert_eq!(p.len(), got.x.len(), "{tag}: one point aes per row");
    for (i, a) in p.iter().enumerate() {
        assert_eq!(f64::from(a.shape), nums(&w["shape"])[i], "{tag}.shape[{i}]");
        assert!(close(a.size, nums(&w["size"])[i]), "{tag}.size[{i}]");
        assert!(close(a.stroke, nums(&w["stroke"])[i]), "{tag}.stroke[{i}]");
        assert_eq!(rgba(a.colour), nums(&w["colour_rgba"][i]), "{tag}.colour[{i}]");
    }
}

/// Compare a whole layout with a fixture: every layer, then both panel scales.
pub(super) fn check(got: &BuiltLayout, w: &Value, geom: &str, kind: RectKind) {
    let want_layers = w["layers"].as_array().expect("layers");
    assert_eq!(got.layers.len(), want_layers.len(), "layer count");
    check_rects("layer0", &got.layers[0], &want_layers[0], geom, kind);
    let warnings = &got.layers[0].rects.as_ref().expect("rect data").warnings;
    assert_eq!(*warnings, strs(&w["warnings"]), "ggplot2's warnings");
    for (i, l) in want_layers.iter().enumerate().skip(1) {
        check_point(&format!("layer{i}"), &got.layers[i], l);
    }
    check_axis("x", &got.x, &w["panel"]["x"]);
    check_axis("y", &got.y, &w["panel"]["y"]);
}

/// The case ggplot2 widens by its own rule: our scale code refuses it with a typed error, and the
/// fixture proves the refused range really is degenerate in R.
pub(super) fn check_refused(res: crate::error::Result<BuiltLayout>, w: &Value) {
    let degenerate = ["x", "y"].iter().any(|a| {
        let l = nums(&w["panel"][*a]["limits"]);
        l[0] == l[1]
    });
    assert!(degenerate, "fixture is not a degenerate-range case");
    assert!(
        matches!(res, Err(crate::error::Error::ScaleDomain(_))),
        "expected ScaleDomain, got {res:?}"
    );
}
