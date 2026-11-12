//! ggplot2 conformance oracle for the segment, polygon, smooth (`method = "lm"`), raster and
//! violin rows (PLOT-GG-001, paiml/aprender#6909). Fixtures are
//! `fixtures/ggplot2/<row>/expected_<case>.json`, produced by ggplot2 4.0.3 under C collation
//! (`g3_oracle.R` and each row's `generate.R`); R is never run here.
//!
//! Tolerance is the row-1 pattern: `|a - b| <= 1e-9 * max(|a|, |b|) + 1e-12` for every position,
//! limit, range, break, fitted value and band; labels, colours, groups, panels, linetypes and
//! counts compare exactly. `NaN` matches `null` (R's `NA`) by position.
//!
//! Every case is built through the public builder and read back with `layout()`. The violin row
//! is a typed-error row: its fixtures record the *shape* of the refused computation only.

use serde_json::Value;

use super::coord::Coord;
use super::geom::SmoothMethod;
use super::shapes::ShapeKind;
use super::stat::Stat;
use super::{Aes, BuiltLayer, BuiltLayout, DataFrame, Facet, GGPlot, Geom, Layer, PositionScale};
use crate::color::Rgba;
use crate::error::Error;

const REL_TOL: f64 = 1e-9;
const ABS_TOL: f64 = 1e-12;

fn close(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= REL_TOL * a.abs().max(b.abs()) + ABS_TOL
}

fn assert_close_slice(what: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}: length got {got:?} want {want:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(close(*g, *w), "{what}[{i}]: got {g:?} want {w:?} (tol {REL_TOL} rel)");
    }
}

/// A JSON array of numbers; `null` (R's `NA`) is `NaN`.
fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("fixture field is an array")
        .iter()
        .map(|e| if e.is_null() { f64::NAN } else { e.as_f64().expect("fixture number") })
        .collect()
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .expect("fixture field is an array")
        .iter()
        .map(|e| e.as_i64().expect("fixture integer"))
        .collect()
}

fn strs(v: &Value) -> Vec<String> {
    v.as_array()
        .expect("fixture field is an array")
        .iter()
        .map(|e| e.as_str().expect("fixture string").to_string())
        .collect()
}

/// Parse a CSV into a frame. A column whose every non-`NA` cell parses as a number is numeric
/// (and each value must be exact in f32); any other column is text. `NA` in a numeric column is
/// `NaN`.
fn frame_from_csv(csv: &str) -> DataFrame {
    let mut lines = csv.lines();
    let header: Vec<&str> = lines.next().expect("csv has a header").split(',').collect();
    let rows: Vec<Vec<&str>> = lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').map(str::trim).collect())
        .collect();
    let mut df = DataFrame::new();
    for (c, name) in header.iter().enumerate() {
        let cells: Vec<&str> = rows.iter().map(|r| r[c]).collect();
        let numeric = cells.iter().all(|s| *s == "NA" || s.parse::<f64>().is_ok());
        if numeric {
            let col: Vec<f32> = cells
                .iter()
                .map(|s| {
                    if *s == "NA" {
                        return f32::NAN;
                    }
                    let v: f64 = s.parse().expect("numeric cell");
                    let f = v as f32;
                    assert_eq!(f64::from(f), v, "fixture value {v} is not exact in f32");
                    f
                })
                .collect();
            df.add_column_f32(name.trim(), &col);
        } else {
            df.add_column_str(name.trim(), &cells);
        }
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
        frame_from_csv(include_str!(concat!("../../fixtures/ggplot2/", $row, "/", $file, ".csv")))
    };
}

fn want(json: &str, row: &str, case: &str) -> Value {
    let v: Value = serde_json::from_str(json).expect("fixture is valid JSON");
    assert_eq!(v["versions"]["ggplot2"], "4.0.3", "fixture was not produced by the pinned ggplot2");
    assert_eq!(v["collation"], "C", "fixture was not made under C collation");
    assert_eq!(v["row"], row);
    assert_eq!(v["case"], case);
    v
}

fn xy() -> Aes {
    Aes::new().x("x").y("y")
}

fn seg_aes() -> Aes {
    xy().xend("xe").yend("ye")
}

fn wrap_f() -> Facet {
    Facet::Wrap { var: "f".into(), ncol: 2 }
}

fn build(
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

fn layer(g: Geom) -> Layer {
    Layer::new(g)
}

fn lm() -> Geom {
    Geom::smooth().method(SmoothMethod::Linear)
}

fn check_axis(name: &str, got: &PositionScale, w: &Value) {
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

fn rgba(c: Option<Rgba>) -> Option<Vec<f64>> {
    c.map(|c| vec![f64::from(c.r), f64::from(c.g), f64::from(c.b), f64::from(c.a)])
}

fn scalar(w: &Value, key: &str) -> Option<f64> {
    w.get(key).and_then(Value::as_f64)
}

/// Compare one shape layer with its fixture layer: positions, the geom's extra columns (the exact
/// set, no more and no fewer), panels, groups and every resolved style value.
fn check_layer(tag: &str, got: &BuiltLayer, w: &Value, geom: &str, kind: ShapeKind, cols: &[&str]) {
    assert_eq!(w["geom"], geom, "{tag}: R geom class");
    let s = got.shape.as_ref().expect("a shape layer resolves shape data");
    assert_eq!(s.kind, kind, "{tag}.kind");
    let panel: Vec<i64> = s.panel.iter().map(|v| *v as i64 + 1).collect();
    assert_eq!(panel, ints(&w["panel"]), "{tag}.panel");
    assert_eq!(s.group, ints(&w["group"]), "{tag}.group");
    assert_eq!(w["n_rows"].as_u64(), Some(panel.len() as u64), "{tag}: fixture row count");
    assert_close_slice(&format!("{tag}.x"), &got.x, &nums(&w["x"]));
    assert_close_slice(&format!("{tag}.y"), &got.y, &nums(&w["y"]));
    let mut have: Vec<&str> = s.cols.keys().copied().collect();
    have.sort_unstable();
    let mut expect: Vec<&str> = cols.to_vec();
    expect.sort_unstable();
    assert_eq!(have, expect, "{tag}: extra columns");
    for c in cols {
        assert_close_slice(&format!("{tag}.{c}"), &s.cols[c], &nums(&w[*c]));
    }
    let fixture_rgba = |key: &str| w.get(key).map(nums);
    assert_eq!(rgba(s.colour), fixture_rgba("colour_rgba"), "{tag}.colour");
    assert_eq!(rgba(s.fill), fixture_rgba("fill_rgba"), "{tag}.fill");
    match (s.linewidth, scalar(w, "linewidth")) {
        (None, None) => {}
        (Some(g), Some(r)) => assert!(close(g, r), "{tag}.linewidth: got {g} want {r}"),
        other => panic!("{tag}.linewidth: got/want {other:?}"),
    }
    assert_eq!(s.linetype.map(f64::from), scalar(w, "linetype"), "{tag}.linetype");
    match (s.alpha, scalar(w, "alpha")) {
        (None, None) => {}
        (Some(g), Some(r)) => assert!(close(g, r), "{tag}.alpha: got {g} want {r}"),
        other => panic!("{tag}.alpha: got/want {other:?}"),
    }
}

/// Compare a whole layout with a one-layer fixture, then both panel scales.
fn check(got: &BuiltLayout, w: &Value, geom: &str, kind: ShapeKind, cols: &[&str]) {
    assert_eq!(got.layers.len(), 1, "layer count");
    assert_eq!(w["layers"].as_array().map(Vec::len), Some(1), "fixture layer count");
    check_layer("layer0", &got.layers[0], &w["layers"][0], geom, kind, cols);
    check_axis("x", &got.x, &w["panel"]["x"]);
    check_axis("y", &got.y, &w["panel"]["y"]);
}

/// A point layer followed by a shape layer with its own data: the shape layer and both scales
/// match R, and the point layer is the plain one.
fn check_second_layer(got: &BuiltLayout, w: &Value, geom: &str, kind: ShapeKind, cols: &[&str]) {
    assert_eq!(got.layers.len(), 2, "layer count");
    assert!(got.layers[0].shape.is_none() && got.layers[0].points.is_some());
    assert_close_slice("layer0.x", &got.layers[0].x, &nums(&w["layers"][0]["x"]));
    assert_close_slice("layer0.y", &got.layers[0].y, &nums(&w["layers"][0]["y"]));
    check_layer("layer1", &got.layers[1], &w["layers"][1], geom, kind, cols);
    check_axis("x", &got.x, &w["panel"]["x"]);
    check_axis("y", &got.y, &w["panel"]["y"]);
}

// --- geom_segment ------------------------------------------------------------------------------

mod segment {
    use super::*;

    const ROW: &str = "geom_segment";
    const G: &str = "GeomSegment";
    const COLS: [&str; 2] = ["xend", "yend"];

    fn run(case: &str, json: &str, data: DataFrame, aes: Aes, facet: Facet) {
        let w = want(json, ROW, case);
        let got = build(data, aes, vec![layer(Geom::segment())], facet).expect("layout resolves");
        check(&got, &w, G, ShapeKind::Segment, &COLS);
    }

    #[test]
    fn plain_keeps_data_order_and_trains_the_end_positions() {
        run(
            "plain",
            fixture!("geom_segment", "plain"),
            csv!("geom_segment", "data"),
            seg_aes(),
            Facet::None,
        );
    }

    /// The edges of the label range the contract claims (non-zero data magnitude from 2^-8 to
    /// 10^4): the tick labels equal R's.
    #[test]
    fn tick_labels_match_r_at_the_edges_of_the_claimed_magnitude_range() {
        for (case, json, file) in [
            (
                "label_lo",
                fixture!("geom_segment", "label_lo"),
                csv!("geom_segment", "data_label_lo"),
            ),
            (
                "label_hi",
                fixture!("geom_segment", "label_hi"),
                csv!("geom_segment", "data_label_hi"),
            ),
        ] {
            run(case, json, file, seg_aes(), Facet::None);
        }
    }

    #[test]
    fn numeric_group_ids_rank_sorted_values() {
        run(
            "group_num",
            fixture!("geom_segment", "group_num"),
            csv!("geom_segment", "data"),
            seg_aes().group("gn"),
            Facet::None,
        );
    }

    #[test]
    fn text_group_ids_rank_sorted_levels() {
        run(
            "group_text",
            fixture!("geom_segment", "group_text"),
            csv!("geom_segment", "data"),
            seg_aes().group("g"),
            Facet::None,
        );
    }

    #[test]
    fn na_rows_are_kept_in_place() {
        run(
            "na",
            fixture!("geom_segment", "na"),
            csv!("geom_segment", "data_na"),
            seg_aes(),
            Facet::None,
        );
    }

    #[test]
    fn facet_wrap_keeps_data_order_with_panel_ids() {
        run(
            "facet",
            fixture!("geom_segment", "facet"),
            csv!("geom_segment", "data"),
            seg_aes(),
            wrap_f(),
        );
    }

    #[test]
    fn fixed_colour_and_alpha() {
        let w = want(fixture!("geom_segment", "colour_alpha"), ROW, "colour_alpha");
        let g = Geom::segment().aes(Aes::new().color_value(Rgba::RED).alpha_value(0.5));
        let got = build(csv!("geom_segment", "data"), seg_aes(), vec![layer(g)], Facet::None)
            .expect("layout resolves");
        check(&got, &w, G, ShapeKind::Segment, &COLS);
    }

    #[test]
    fn a_second_layer_with_its_own_data_widens_the_scales() {
        let w = want(fixture!("geom_segment", "layer_data"), ROW, "layer_data");
        let own = layer(Geom::segment())
            .data(csv!("geom_segment", "data_layer"))
            .aes(Aes::new().xend("xe").yend("ye"));
        let got = build(
            csv!("geom_segment", "data_pt"),
            xy(),
            vec![layer(Geom::point()), own],
            Facet::None,
        )
        .expect("layout resolves");
        check_second_layer(&got, &w, G, ShapeKind::Segment, &COLS);
    }

    #[test]
    fn a_missing_end_column_is_a_typed_error() {
        let r =
            build(csv!("geom_segment", "data"), xy(), vec![layer(Geom::segment())], Facet::None);
        assert!(matches!(r, Err(Error::UnknownColumn(_))), "{r:?}");
    }
}

// --- geom_polygon ------------------------------------------------------------------------------

mod polygon {
    use super::*;

    const ROW: &str = "geom_polygon";
    const G: &str = "GeomPolygon";

    fn run(case: &str, json: &str, data: DataFrame, aes: Aes, facet: Facet) {
        let w = want(json, ROW, case);
        let got = build(data, aes, vec![layer(Geom::polygon())], facet).expect("layout resolves");
        check(&got, &w, G, ShapeKind::Polygon, &[]);
    }

    #[test]
    fn plain_keeps_data_order_and_default_fill() {
        run(
            "plain",
            fixture!("geom_polygon", "plain"),
            csv!("geom_polygon", "data"),
            xy(),
            Facet::None,
        );
    }

    #[test]
    fn tick_labels_match_r_at_the_edges_of_the_claimed_magnitude_range() {
        for (case, json, file) in [
            (
                "label_lo",
                fixture!("geom_polygon", "label_lo"),
                csv!("geom_polygon", "data_label_lo"),
            ),
            (
                "label_hi",
                fixture!("geom_polygon", "label_hi"),
                csv!("geom_polygon", "data_label_hi"),
            ),
        ] {
            run(case, json, file, xy(), Facet::None);
        }
    }

    #[test]
    fn numeric_group_ids_rank_sorted_values() {
        run(
            "group_num",
            fixture!("geom_polygon", "group_num"),
            csv!("geom_polygon", "data"),
            xy().group("gn"),
            Facet::None,
        );
    }

    #[test]
    fn text_group_ids_rank_sorted_levels() {
        run(
            "group_text",
            fixture!("geom_polygon", "group_text"),
            csv!("geom_polygon", "data"),
            xy().group("g"),
            Facet::None,
        );
    }

    #[test]
    fn two_row_groups() {
        run(
            "two_groups",
            fixture!("geom_polygon", "two_groups"),
            csv!("geom_polygon", "data_two"),
            xy().group("gn"),
            Facet::None,
        );
    }

    #[test]
    fn na_rows_are_kept_in_place() {
        run(
            "na",
            fixture!("geom_polygon", "na"),
            csv!("geom_polygon", "data_na"),
            xy(),
            Facet::None,
        );
    }

    #[test]
    fn facet_wrap_keeps_data_order_with_panel_ids() {
        run(
            "facet",
            fixture!("geom_polygon", "facet"),
            csv!("geom_polygon", "data"),
            xy(),
            wrap_f(),
        );
    }

    #[test]
    fn fixed_colour_and_alpha() {
        let w = want(fixture!("geom_polygon", "colour_alpha"), ROW, "colour_alpha");
        let g = Geom::polygon().aes(Aes::new().color_value(Rgba::RED).alpha_value(0.5));
        let got = build(csv!("geom_polygon", "data"), xy(), vec![layer(g)], Facet::None)
            .expect("layout resolves");
        check(&got, &w, G, ShapeKind::Polygon, &[]);
    }

    #[test]
    fn a_second_layer_with_its_own_data_widens_the_scales() {
        let w = want(fixture!("geom_polygon", "layer_data"), ROW, "layer_data");
        let own = layer(Geom::polygon()).data(csv!("geom_polygon", "data_layer"));
        let got = build(
            csv!("geom_polygon", "data_pt"),
            xy(),
            vec![layer(Geom::point()), own],
            Facet::None,
        )
        .expect("layout resolves");
        check_second_layer(&got, &w, G, ShapeKind::Polygon, &[]);
    }
}

// --- geom_raster -------------------------------------------------------------------------------

mod raster {
    use super::*;

    const ROW: &str = "geom_raster";
    const G: &str = "GeomRaster";
    const COLS: [&str; 4] = ["xmin", "xmax", "ymin", "ymax"];

    fn run(case: &str, json: &str, data: DataFrame, aes: Aes, facet: Facet) {
        let w = want(json, ROW, case);
        let got = build(data, aes, vec![layer(Geom::raster())], facet).expect("layout resolves");
        check(&got, &w, G, ShapeKind::Raster, &COLS);
    }

    #[test]
    fn plain_derives_pixel_edges_from_even_spacing_with_gaps() {
        run(
            "plain",
            fixture!("geom_raster", "plain"),
            csv!("geom_raster", "data"),
            xy(),
            Facet::None,
        );
    }

    #[test]
    fn numeric_group_ids_rank_sorted_values() {
        run(
            "group_num",
            fixture!("geom_raster", "group_num"),
            csv!("geom_raster", "data"),
            xy().group("gn"),
            Facet::None,
        );
    }

    #[test]
    fn na_rows_are_kept_in_place() {
        run("na", fixture!("geom_raster", "na"), csv!("geom_raster", "data_na"), xy(), Facet::None);
    }

    #[test]
    fn facet_wrap_keeps_data_order_with_panel_ids() {
        run("facet", fixture!("geom_raster", "facet"), csv!("geom_raster", "data"), xy(), wrap_f());
    }

    #[test]
    fn fixed_alpha() {
        let w = want(fixture!("geom_raster", "colour_alpha"), ROW, "colour_alpha");
        let g = Geom::raster().aes(Aes::new().alpha_value(0.5));
        let got = build(csv!("geom_raster", "data"), xy(), vec![layer(g)], Facet::None)
            .expect("layout resolves");
        check(&got, &w, G, ShapeKind::Raster, &COLS);
    }

    #[test]
    fn a_second_layer_with_its_own_data_widens_the_scales() {
        let w = want(fixture!("geom_raster", "layer_data"), ROW, "layer_data");
        let own = layer(Geom::raster()).data(csv!("geom_raster", "data_layer"));
        let got = build(
            csv!("geom_raster", "data_pt"),
            xy(),
            vec![layer(Geom::point()), own],
            Facet::None,
        )
        .expect("layout resolves");
        check_second_layer(&got, &w, G, ShapeKind::Raster, &COLS);
    }

    #[test]
    fn fractional_and_negative_spacing() {
        run(
            "fractional",
            fixture!("geom_raster", "fractional"),
            csv!("geom_raster", "data_frac"),
            xy(),
            Facet::None,
        );
        run(
            "negative",
            fixture!("geom_raster", "negative"),
            csv!("geom_raster", "data_neg"),
            xy(),
            Facet::None,
        );
    }

    #[test]
    fn one_distinct_value_gives_a_unit_pixel() {
        run(
            "single",
            fixture!("geom_raster", "single"),
            csv!("geom_raster", "data_single"),
            xy(),
            Facet::None,
        );
    }

    /// R warns and shifts the pixels of an unevenly spaced axis; we refuse. The fixture proves
    /// the data really is uneven to R.
    #[test]
    fn uneven_spacing_is_refused_and_the_fixture_proves_it_is_uneven() {
        let w = want(fixture!("geom_raster", "uneven"), ROW, "uneven");
        let mut xs = nums(&w["layers"][0]["x"]);
        xs.sort_by(f64::total_cmp);
        let d: Vec<f64> = xs.windows(2).map(|p| p[1] - p[0]).collect();
        assert!(d.windows(2).any(|p| (p[1] - p[0]).abs() > 1e-6), "spacing {d:?} is even");
        let r = build(
            csv!("geom_raster", "data_uneven"),
            xy(),
            vec![layer(Geom::raster())],
            Facet::None,
        );
        assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
    }
}

// --- geom_smooth(method = "lm") ----------------------------------------------------------------

mod smooth {
    use super::*;

    const ROW: &str = "geom_smooth";
    const G: &str = "GeomSmooth";
    const BAND: [&str; 3] = ["ymin", "ymax", "se"];

    fn run(case: &str, json: &str, data: DataFrame, aes: Aes, facet: Facet, se: bool) {
        let w = want(json, ROW, case);
        let got = build(data, aes, vec![layer(lm().se(se))], facet).expect("layout resolves");
        check(&got, &w, G, ShapeKind::Smooth, if se { &BAND } else { &[] });
    }

    #[test]
    fn se_false_is_the_fitted_line_on_an_80_point_grid() {
        run(
            "se_false",
            fixture!("geom_smooth", "se_false"),
            csv!("geom_smooth", "data_g1"),
            xy(),
            Facet::None,
            false,
        );
    }

    #[test]
    fn se_true_adds_the_t_band_and_trains_the_scale_with_it() {
        run(
            "se_true",
            fixture!("geom_smooth", "se_true"),
            csv!("geom_smooth", "data_g1"),
            xy(),
            Facet::None,
            true,
        );
    }

    #[test]
    fn groups_fit_separately_in_ascending_order() {
        run(
            "group_se",
            fixture!("geom_smooth", "group_se"),
            csv!("geom_smooth", "data"),
            xy().group("gn"),
            Facet::None,
            true,
        );
        run(
            "group_text_nose",
            fixture!("geom_smooth", "group_text_nose"),
            csv!("geom_smooth", "data"),
            xy().group("g"),
            Facet::None,
            false,
        );
    }

    #[test]
    fn two_row_groups_fit_a_line_through_both_points() {
        run(
            "group_two_row_nose",
            fixture!("geom_smooth", "group_two_row_nose"),
            csv!("geom_smooth", "data_g2"),
            xy().group("gn"),
            Facet::None,
            false,
        );
    }

    #[test]
    fn facet_wrap_fits_each_panel() {
        run(
            "facet_se",
            fixture!("geom_smooth", "facet_se"),
            csv!("geom_smooth", "data_facet"),
            xy(),
            wrap_f(),
            true,
        );
    }

    #[test]
    fn na_rows_are_removed_before_the_fit() {
        run(
            "na_se",
            fixture!("geom_smooth", "na_se"),
            csv!("geom_smooth", "data_na"),
            xy(),
            Facet::None,
            true,
        );
    }

    #[test]
    fn three_rows_give_a_band_with_one_residual_degree_of_freedom() {
        run(
            "three_rows_se",
            fixture!("geom_smooth", "three_rows_se"),
            csv!("geom_smooth", "data_three"),
            xy(),
            Facet::None,
            true,
        );
    }

    #[test]
    fn two_rows_fit_without_a_band() {
        run(
            "two_rows_nose",
            fixture!("geom_smooth", "two_rows_nose"),
            csv!("geom_smooth", "data_two"),
            xy(),
            Facet::None,
            false,
        );
    }

    #[test]
    fn fixed_colour_and_alpha() {
        let w = want(fixture!("geom_smooth", "colour_alpha"), ROW, "colour_alpha");
        let g = lm().aes(Aes::new().color_value(Rgba::RED).alpha_value(0.5));
        let got = build(csv!("geom_smooth", "data_g1"), xy(), vec![layer(g)], Facet::None)
            .expect("layout resolves");
        check(&got, &w, G, ShapeKind::Smooth, &BAND);
    }

    #[test]
    fn a_second_layer_with_its_own_data_widens_the_scales() {
        let w = want(fixture!("geom_smooth", "layer_data"), ROW, "layer_data");
        let own = layer(lm().se(true)).data(csv!("geom_smooth", "data_layer"));
        let got = build(
            csv!("geom_smooth", "data_pt"),
            xy(),
            vec![layer(Geom::point()), own],
            Facet::None,
        )
        .expect("layout resolves");
        check_second_layer(&got, &w, G, ShapeKind::Smooth, &BAND);
    }

    /// `qt(0.975, df)` against R's table: the band's multiplier, small df through very large.
    #[test]
    fn student_t_quantile_matches_r_across_degrees_of_freedom() {
        let w = want(fixture!("geom_smooth", "qt_table"), ROW, "qt_table");
        let (dfs, qts) = (nums(&w["df"]), nums(&w["qt"]));
        assert!(dfs.len() >= 30 && dfs.len() == qts.len());
        for (df, q) in dfs.iter().zip(&qts) {
            let got = super::super::tdist::qt_upper(0.025, *df).expect("in domain");
            assert!((got - q).abs() <= 1e-9 * q.abs(), "qt(0.975, {df}): got {got:?} want {q:?}");
        }
    }

    #[test]
    fn loess_and_other_methods_are_refused() {
        let r =
            build(csv!("geom_smooth", "data_g1"), xy(), vec![layer(Geom::smooth())], Facet::None);
        assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
    }

    #[test]
    fn groups_ggplot2_drops_or_nans_are_refused() {
        // Two rows with se = TRUE: R's band is NaN.
        let r =
            build(csv!("geom_smooth", "data_two"), xy(), vec![layer(lm().se(true))], Facet::None);
        assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
        // One distinct x: R drops the group without a word.
        let mut df = DataFrame::new();
        df.add_column_f32("x", &[2.0, 2.0, 2.0]);
        df.add_column_f32("y", &[1.0, 2.0, 3.0]);
        for se in [false, true] {
            let r = build(df.clone(), xy(), vec![layer(lm().se(se))], Facet::None);
            assert!(matches!(r, Err(Error::Rendering(_))), "se={se}: {r:?}");
        }
        // Every row has an NA: nothing is left to fit.
        let mut nas = DataFrame::new();
        nas.add_column_f32("x", &[1.0, 2.0, f32::NAN]);
        nas.add_column_f32("y", &[f32::NAN, f32::NAN, 3.0]);
        let r = build(nas, xy(), vec![layer(lm().se(false))], Facet::None);
        assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
    }

    #[test]
    fn a_stat_other_than_smooth_is_refused() {
        let r = build(
            csv!("geom_smooth", "data_g1"),
            xy(),
            vec![layer(lm().stat(Stat::Count))],
            Facet::None,
        );
        assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
    }
}

// --- geom_violin -------------------------------------------------------------------------------

mod violin {
    use super::*;

    const ROW: &str = "geom_violin";

    /// `stat_ydensity` is not implemented; the row's claim is the typed error. Each fixture shows
    /// R's answer is a real per-group density (hundreds of rows per group), not a trivial one.
    #[test]
    fn the_row_is_a_typed_error_and_the_fixtures_show_what_is_refused() {
        for case in [
            ("plain", fixture!("geom_violin", "plain")),
            ("two_groups", fixture!("geom_violin", "two_groups")),
            ("three_groups", fixture!("geom_violin", "three_groups")),
            ("group_num", fixture!("geom_violin", "group_num")),
            ("facet", fixture!("geom_violin", "facet")),
            ("small", fixture!("geom_violin", "small")),
            ("layer_data", fixture!("geom_violin", "layer_data")),
            ("untrimmed", fixture!("geom_violin", "untrimmed")),
        ] {
            let w = want(case.1, ROW, case.0);
            let l = &w["layer"];
            assert_eq!(l["geom"], "GeomViolin", "{}", case.0);
            assert_eq!(l["has_density"], true, "{}", case.0);
            let per = ints(&l["rows_per_group"]);
            assert_eq!(per.len() as u64, l["n_groups"].as_u64().expect("groups"), "{}", case.0);
            assert_eq!(per.iter().sum::<i64>() as u64, l["n_rows"].as_u64().expect("rows"));
            assert!(per.iter().all(|n| *n >= 100), "{}: rows per group {per:?}", case.0);
        }
        let r = build(
            csv!("geom_violin", "data"),
            Aes::new().x("gn").y("y"),
            vec![layer(Geom::violin())],
            Facet::None,
        );
        assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
    }
}

// --- refusals and robustness shared by every shape geom -----------------------------------------

mod robustness {
    use super::*;

    fn aes_for(g: &Geom) -> Aes {
        if matches!(g.geom_type, super::super::geom::GeomType::Segment) {
            seg_aes()
        } else {
            xy()
        }
    }

    fn all_shape_geoms() -> Vec<Geom> {
        vec![Geom::segment(), Geom::polygon(), Geom::raster(), lm().se(false), lm().se(true)]
    }

    fn ok_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column_f32("x", &[1.0, 2.0, 3.0, 4.0]);
        df.add_column_f32("y", &[1.0, 3.0, 2.0, 4.0]);
        df.add_column_f32("xe", &[2.0, 3.0, 4.0, 5.0]);
        df.add_column_f32("ye", &[2.0, 5.0, 10.0, 4.0]);
        df.add_column_str("f", &["a", "b", "a", "b"]);
        df
    }

    fn one(g: &Geom, data: DataFrame, aes: Aes, facet: Facet) -> crate::error::Result<BuiltLayout> {
        build(data, aes, vec![layer(g.clone())], facet)
    }

    fn is_rendering<T: std::fmt::Debug>(r: &crate::error::Result<T>) -> bool {
        matches!(r, Err(Error::Rendering(_)))
    }

    #[test]
    fn every_geom_resolves_the_ok_frame() {
        for g in all_shape_geoms() {
            let r = one(&g, ok_df(), aes_for(&g), Facet::None);
            assert!(r.is_ok(), "{:?}: {r:?}", g.geom_type);
        }
    }

    #[test]
    fn empty_frame_and_zero_row_columns_are_errors() {
        for g in all_shape_geoms() {
            let r = one(&g, DataFrame::new(), aes_for(&g), Facet::None);
            assert!(r.is_err(), "{:?} on an empty frame returned {r:?}", g.geom_type);
            let mut zero = DataFrame::new();
            for c in ["x", "y", "xe", "ye"] {
                zero.add_column_f32(c, &[]);
            }
            let r = one(&g, zero, aes_for(&g), Facet::None);
            assert!(r.is_err(), "{:?} on zero rows returned {r:?}", g.geom_type);
        }
    }

    /// A short column is a typed error, never an `Ok` padded with `NA`.
    #[test]
    fn a_short_column_is_a_typed_error_never_padded() {
        for g in all_shape_geoms() {
            for short in ["x", "y", "xe", "ye"] {
                let mut df = ok_df();
                df.add_column_f32(short, &[1.0, 2.0]);
                let r = one(&g, df, aes_for(&g), Facet::None);
                let used = short.len() == 1
                    || matches!(g.geom_type, super::super::geom::GeomType::Segment);
                if used {
                    assert!(is_rendering(&r), "{:?} with short {short}: {r:?}", g.geom_type);
                } else {
                    assert!(r.is_ok(), "{:?} ignores {short}: {r:?}", g.geom_type);
                }
            }
            // A short group column.
            let mut df = ok_df();
            df.add_column_f32("g", &[1.0, 2.0]);
            let r = one(&g, df, aes_for(&g).group("g"), Facet::None);
            assert!(r.is_err(), "{:?} with a short group column returned {r:?}", g.geom_type);
        }
    }

    #[test]
    fn ragged_columns_under_a_facet_do_not_panic() {
        let mut df = DataFrame::new();
        df.add_column_f32("x", &[1.0, 2.0, 3.0, 4.0]);
        df.add_column_f32("y", &[1.0, 2.0]);
        df.add_column_f32("xe", &[1.0]);
        df.add_column_f32("ye", &[1.0, 2.0, 3.0]);
        df.add_column_str("f", &["a", "b", "a"]);
        for g in all_shape_geoms() {
            let r = one(&g, df.clone(), aes_for(&g), wrap_f());
            assert!(r.is_err(), "{:?}: {r:?}", g.geom_type);
        }
    }

    #[test]
    fn infinite_and_text_positions_are_typed_errors() {
        for g in all_shape_geoms() {
            for col in ["x", "y", "xe", "ye"] {
                let mut inf = ok_df();
                inf.add_column_f32(col, &[1.0, f32::INFINITY, 2.0, 3.0]);
                let r = one(&g, inf, aes_for(&g), Facet::None);
                let mut text = ok_df();
                text.add_column_str(col, &["a", "b", "c", "d"]);
                let rt = one(&g, text, aes_for(&g), Facet::None);
                let used =
                    col.len() == 1 || matches!(g.geom_type, super::super::geom::GeomType::Segment);
                if used {
                    assert!(is_rendering(&r), "{:?} inf {col}: {r:?}", g.geom_type);
                    assert!(is_rendering(&rt), "{:?} text {col}: {rt:?}", g.geom_type);
                }
            }
        }
    }

    #[test]
    fn nan_positions_are_values_for_segment_polygon_and_raster() {
        for g in [Geom::segment(), Geom::polygon(), Geom::raster()] {
            let mut df = ok_df();
            df.add_column_f32("x", &[f32::NAN, 2.0, 3.0, 4.0]);
            df.add_column_f32("y", &[1.0, f32::NAN, 3.0, 2.0]);
            let l = one(&g, df, aes_for(&g), Facet::None).expect("a value");
            assert_eq!(l.layers[0].x.len(), 4);
            assert!(l.layers[0].x[0].is_nan() && l.layers[0].y[1].is_nan());
        }
    }

    #[test]
    fn all_nan_positions_are_a_typed_error_not_a_panic() {
        for g in all_shape_geoms() {
            let mut df = ok_df();
            df.add_column_f32("x", &[f32::NAN; 4]);
            df.add_column_f32("y", &[f32::NAN; 4]);
            df.add_column_f32("xe", &[f32::NAN; 4]);
            df.add_column_f32("ye", &[f32::NAN; 4]);
            let r = one(&g, df, aes_for(&g), Facet::None);
            assert!(r.is_err(), "{:?}: {r:?}", g.geom_type);
        }
    }

    #[test]
    fn extreme_magnitudes_never_panic_or_hang() {
        for scale in [1e-30f32, 1e-8, 1e8, 1e30, f32::MAX / 4.0] {
            for g in all_shape_geoms() {
                let mut df = ok_df();
                for c in ["x", "y", "xe", "ye"] {
                    let base = [1.0f32, 2.0, 3.0, 4.5];
                    let col: Vec<f32> = base.iter().map(|v| v * scale).collect();
                    df.add_column_f32(c, &col);
                }
                let r = one(&g, df, aes_for(&g), Facet::None);
                if let Ok(l) = r {
                    assert!(!l.x.continuous_range.0.is_nan(), "{:?}", g.geom_type);
                }
            }
        }
    }

    #[test]
    fn missing_columns_are_unknown_column_errors() {
        for g in all_shape_geoms() {
            let r = one(&g, ok_df(), Aes::new().x("nope").y("y"), Facet::None);
            assert!(matches!(r, Err(Error::UnknownColumn(_))), "{:?}: {r:?}", g.geom_type);
            let r = one(&g, ok_df(), aes_for(&g).group("nope"), Facet::None);
            assert!(r.is_err(), "{:?}: {r:?}", g.geom_type);
        }
    }

    #[test]
    fn unsupported_aesthetics_are_refused_not_ignored() {
        for g in all_shape_geoms() {
            for a in [
                aes_for(&g).color("f"),
                aes_for(&g).size("x"),
                aes_for(&g).alpha("x"),
                aes_for(&g).fill("f"),
                aes_for(&g).shape("f"),
                aes_for(&g).label("f"),
            ] {
                let r = one(&g, ok_df(), a, Facet::None);
                assert!(is_rendering(&r), "{:?}: {r:?}", g.geom_type);
            }
            let sized = g.clone().aes(Aes::new().size_value(2.0));
            let r = one(&sized, ok_df(), aes_for(&g), Facet::None);
            assert!(is_rendering(&r), "{:?}: {r:?}", g.geom_type);
        }
    }

    /// Any stat other than the geom's own changes R's numbers.
    #[test]
    fn a_non_default_stat_is_refused() {
        for g in [Geom::segment(), Geom::polygon(), Geom::raster()] {
            for stat in [Stat::Count, Stat::Density, Stat::Smooth] {
                let r = one(&g.clone().stat(stat.clone()), ok_df(), aes_for(&g), Facet::None);
                assert!(is_rendering(&r), "{:?} with {stat:?}: {r:?}", g.geom_type);
            }
            let ok = one(&g.clone().stat(Stat::Identity), ok_df(), aes_for(&g), Facet::None);
            assert!(ok.is_ok(), "identity is the default: {ok:?}");
        }
    }

    #[test]
    fn unsupported_facets_and_facet_columns_are_refused() {
        let grid = Facet::Grid { row: "f".into(), col: "f".into() };
        for g in all_shape_geoms() {
            for facet in
                [Facet::Row { var: "f".into() }, Facet::Col { var: "f".into() }, grid.clone()]
            {
                let r = one(&g, ok_df(), aes_for(&g), facet);
                assert!(is_rendering(&r), "{:?}: {r:?}", g.geom_type);
            }
            let r = one(&g, ok_df(), aes_for(&g), Facet::Wrap { var: "x".into(), ncol: 2 });
            assert!(is_rendering(&r), "{:?}: {r:?}", g.geom_type);
        }
    }

    #[test]
    fn layer_data_must_match_the_plot_panels() {
        for g in [Geom::polygon(), Geom::segment()] {
            let mut own = DataFrame::new();
            for c in ["x", "y", "xe", "ye"] {
                own.add_column_f32(c, &[1.0, 2.0]);
            }
            let r = build(ok_df(), aes_for(&g), vec![layer(g.clone()).data(own.clone())], wrap_f());
            assert!(matches!(r, Err(Error::UnknownColumn(_))), "{r:?}");
            own.add_column_str("f", &["a", "zzz"]);
            let r = build(ok_df(), aes_for(&g), vec![layer(g.clone()).data(own)], wrap_f());
            assert!(is_rendering(&r), "{r:?}");
        }
    }

    fn non_default_coords() -> Vec<(&'static str, Coord)> {
        vec![
            ("flip", Coord::cartesian().flip()),
            ("xlim", Coord::cartesian().xlim(0.0, 10.0)),
            ("ylim", Coord::cartesian().ylim(0.0, 10.0)),
            ("xlim+flip", Coord::cartesian().xlim(0.0, 10.0).flip()),
            ("polar", Coord::polar()),
            ("fixed", Coord::fixed(1.0)),
        ]
    }

    fn with_coord(g: Geom, coord: Coord) -> crate::error::Result<BuiltLayout> {
        GGPlot::new()
            .data(ok_df())
            .aes(aes_for(&g))
            .layer(layer(Geom::point()))
            .layer(layer(g))
            .coord(coord)
            .build()?
            .layout()
    }

    /// `layout()` applies no coordinate system, so one it would silently ignore is refused for
    /// every geom of these rows (R gives different panel ranges and breaks for each).
    #[test]
    fn every_non_default_coord_is_refused_for_every_geom() {
        let mut geoms = all_shape_geoms();
        geoms.push(Geom::violin());
        for g in geoms {
            for (name, coord) in non_default_coords() {
                let r = with_coord(g.clone(), coord);
                assert!(
                    matches!(r, Err(Error::UnsupportedCoord(_))),
                    "{:?} under {name} returned {r:?}",
                    g.geom_type
                );
            }
        }
    }

    #[test]
    fn the_default_coord_is_accepted_for_every_supported_geom() {
        for g in all_shape_geoms() {
            let r = with_coord(g.clone(), Coord::cartesian());
            assert!(r.is_ok(), "{:?} under the default coord returned {r:?}", g.geom_type);
        }
    }

    /// The refusal is scoped to these geoms: a point-only plot keeps its behaviour.
    #[test]
    fn a_point_only_plot_is_not_refused_by_a_coord() {
        let r = GGPlot::new()
            .data(ok_df())
            .aes(xy())
            .layer(layer(Geom::point()))
            .coord(Coord::cartesian().flip())
            .build()
            .and_then(|p| p.layout());
        assert!(r.is_ok(), "{r:?}");
    }

    #[test]
    fn rendering_a_plot_with_these_geoms_still_works() {
        let mut geoms = all_shape_geoms();
        geoms.push(Geom::violin());
        geoms.push(Geom::smooth());
        for g in geoms {
            let p = GGPlot::new().data(ok_df()).aes(aes_for(&g)).geom(g).build().expect("builds");
            let _ = p.to_svg();
        }
    }

    #[test]
    fn a_degenerate_axis_is_a_scale_domain_error() {
        let mut df = ok_df();
        df.add_column_f32("x", &[5.0; 4]);
        df.add_column_f32("xe", &[5.0; 4]);
        let r = one(&Geom::polygon(), df, xy(), Facet::None);
        assert!(matches!(r, Err(Error::ScaleDomain(_))), "{r:?}");
    }
}
