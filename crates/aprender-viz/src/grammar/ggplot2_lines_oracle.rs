//! ggplot2 conformance oracle for the path and reference-line rows (PLOT-GG-001,
//! paiml/aprender#6909): `geom_line`, `geom_path`, `geom_step`, `geom_hline`, `geom_vline`,
//! `geom_abline`. Fixtures are `fixtures/ggplot2/<row>/expected_<case>.json`, produced by
//! ggplot2 4.0.3 under C collation (`lines_oracle.R`); R is never run here.
//!
//! Tolerance is the row-1 pattern: `|a - b| <= 1e-9 * max(|a|, |b|) + 1e-12` for every position,
//! limit, range, break, intercept, slope and linewidth; labels, colours, groups, panels, linetypes
//! and counts compare exactly. `NaN` matches `null` (R's `NA`) by position.
//!
//! Every case is built through the public builder and read back with `layout()`.

use serde_json::Value;

use super::coord::Coord;
use super::stat::Stat;
use super::{
    Aes, BuiltLayer, BuiltLayout, DataFrame, Facet, GGPlot, Geom, Layer, PositionScale,
    ReferenceKind, StepDirection,
};
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

fn styled(g: Geom, colour: Rgba, alpha: f32) -> Geom {
    g.aes(Aes::new().color_value(colour).alpha_value(alpha))
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

fn check_layer(tag: &str, got: &BuiltLayer, w: &Value, geom: &str) {
    assert_eq!(w["geom"], geom, "{tag}: R geom class");
    let p = got.path.as_ref().expect("a path-family layer resolves path data");
    let panel: Vec<i64> = p.panel.iter().map(|v| *v as i64 + 1).collect();
    assert_eq!(panel, ints(&w["panel"]), "{tag}.panel");
    assert_eq!(p.group, ints(&w["group"]), "{tag}.group");
    let n = panel.len();
    assert_eq!(w["linewidth"].as_array().map(Vec::len), Some(n), "{tag}: fixture row count");
    for i in 0..n {
        let rgba = nums(&w["colour_rgba"][i]);
        let c = p.aes.colour;
        assert_eq!([c.r, c.g, c.b, c.a].map(f64::from).to_vec(), rgba, "{tag}.colour[{i}]");
        assert!(close(p.aes.linewidth, nums(&w["linewidth"])[i]), "{tag}.linewidth[{i}]");
        assert_eq!(f64::from(p.aes.linetype), nums(&w["linetype"])[i], "{tag}.linetype[{i}]");
        let a = nums(&w["alpha"])[i];
        match p.aes.alpha {
            None => assert!(a.is_nan(), "{tag}.alpha[{i}]: R has {a}, got NA"),
            Some(v) => assert!(close(v, a), "{tag}.alpha[{i}]: got {v} want {a}"),
        }
    }
    if let Some(r) = &p.reference {
        assert!(got.x.is_empty() && got.y.is_empty(), "{tag}: reference rows carry no x/y");
        match r.kind {
            ReferenceKind::Horizontal => {
                assert_close_slice(
                    &format!("{tag}.yintercept"),
                    &r.intercept,
                    &nums(&w["yintercept"]),
                );
            }
            ReferenceKind::Vertical => {
                assert_close_slice(
                    &format!("{tag}.xintercept"),
                    &r.intercept,
                    &nums(&w["xintercept"]),
                );
            }
            ReferenceKind::Diagonal => {
                assert_close_slice(
                    &format!("{tag}.intercept"),
                    &r.intercept,
                    &nums(&w["intercept"]),
                );
                assert_close_slice(&format!("{tag}.slope"), &r.slope, &nums(&w["slope"]));
            }
        }
    } else {
        assert_close_slice(&format!("{tag}.x"), &got.x, &nums(&w["x"]));
        assert_close_slice(&format!("{tag}.y"), &got.y, &nums(&w["y"]));
    }
    assert_eq!(p.stairs.is_some(), w.get("stairs").is_some(), "{tag}: stairs presence");
    if let (Some(s), Some(ws)) = (&p.stairs, w.get("stairs")) {
        let dir = match s.direction {
            StepDirection::Hv => "hv",
            StepDirection::Vh => "vh",
            StepDirection::Mid => "mid",
        };
        assert_eq!(ws["direction"], dir, "{tag}.stairs.direction");
        assert_close_slice(&format!("{tag}.stairs.x"), &s.x, &nums(&ws["x"]));
        assert_close_slice(&format!("{tag}.stairs.y"), &s.y, &nums(&ws["y"]));
        let sp: Vec<i64> = s.panel.iter().map(|v| *v as i64 + 1).collect();
        assert_eq!(sp, ints(&ws["panel"]), "{tag}.stairs.panel");
        assert_eq!(s.group, ints(&ws["group"]), "{tag}.stairs.group");
    }
}

/// Compare a whole layout with a fixture: every layer, then both panel scales.
fn check(got: &BuiltLayout, w: &Value, geoms: &[&str]) {
    let want_layers = w["layers"].as_array().expect("layers");
    assert_eq!(got.layers.len(), want_layers.len(), "layer count");
    assert_eq!(geoms.len(), want_layers.len(), "geom names");
    for (i, g) in geoms.iter().enumerate() {
        check_layer(&format!("layer{i}"), &got.layers[i], &want_layers[i], g);
    }
    check_axis("x", &got.x, &w["panel"]["x"]);
    check_axis("y", &got.y, &w["panel"]["y"]);
}

/// The case ggplot2 widens by its own rule: our scale code refuses it with a typed error, and the
/// fixture proves the refused range really is degenerate in R.
fn check_refused(res: crate::error::Result<BuiltLayout>, w: &Value) {
    let which = ["x", "y"].iter().any(|a| {
        let l = nums(&w["panel"][*a]["limits"]);
        l[0] == l[1]
    });
    assert!(which, "fixture is not a degenerate-range case");
    assert!(matches!(res, Err(Error::ScaleDomain(_))), "expected ScaleDomain, got {res:?}");
}

// --- geom_line ---------------------------------------------------------------------------------

mod line {
    use super::*;

    const ROW: &str = "geom_line";
    const G: &str = "GeomLine";

    fn run(case: &str, json: &str, data: DataFrame, aes: Aes, facet: Facet) {
        let w = want(json, ROW, case);
        let got = build(data, aes, vec![layer(Geom::line())], facet).expect("layout resolves");
        check(&got, &w, &[G]);
    }

    #[test]
    fn plain_sorts_by_x_with_ties_in_data_order() {
        run("plain", fixture!("geom_line", "plain"), csv!("geom_line", "data"), xy(), Facet::None);
    }

    /// The edges of the label range the contracts claim (non-zero data magnitude from 2^-8 to
    /// 10^4): the tick labels equal R's. Beyond it the shared formatter is known to differ from R's
    /// fixed-versus-scientific choice (`1e+05`, `1e-04`); that is a separate ticket, not claimed.
    #[test]
    fn tick_labels_match_r_at_the_edges_of_the_claimed_magnitude_range() {
        for (case, json, file) in [
            ("label_lo", fixture!("geom_line", "label_lo"), csv!("geom_line", "data_label_lo")),
            ("label_hi", fixture!("geom_line", "label_hi"), csv!("geom_line", "data_label_hi")),
            ("label_neg", fixture!("geom_line", "label_neg"), csv!("geom_line", "data_label_neg")),
        ] {
            run(case, json, file, xy(), Facet::None);
        }
    }

    #[test]
    fn numeric_group_ids_rank_sorted_values() {
        run(
            "group_num",
            fixture!("geom_line", "group_num"),
            csv!("geom_line", "data"),
            xy().group("gn"),
            Facet::None,
        );
    }

    #[test]
    fn text_group_ids_rank_sorted_levels() {
        run(
            "group_text",
            fixture!("geom_line", "group_text"),
            csv!("geom_line", "data"),
            xy().group("g"),
            Facet::None,
        );
    }

    #[test]
    fn na_x_sorts_last_and_na_y_is_kept() {
        run("na", fixture!("geom_line", "na"), csv!("geom_line", "data_na"), xy(), Facet::None);
    }

    #[test]
    fn facet_wrap_orders_rows_by_panel_first() {
        run("facet", fixture!("geom_line", "facet"), csv!("geom_line", "data"), xy(), wrap_f());
    }

    #[test]
    fn facet_wrap_with_groups() {
        run(
            "facet_group",
            fixture!("geom_line", "facet_group"),
            csv!("geom_line", "data"),
            xy().group("gn"),
            wrap_f(),
        );
    }

    #[test]
    fn fixed_colour_and_alpha() {
        let w = want(fixture!("geom_line", "colour_alpha"), ROW, "colour_alpha");
        let got = build(
            csv!("geom_line", "data"),
            xy(),
            vec![layer(styled(Geom::line(), Rgba::RED, 0.5))],
            Facet::None,
        )
        .expect("layout resolves");
        check(&got, &w, &[G]);
    }

    #[test]
    fn a_second_layer_with_its_own_data_widens_the_scales() {
        let w = want(fixture!("geom_line", "layer_data"), ROW, "layer_data");
        let got = build(
            csv!("geom_line", "data"),
            xy(),
            vec![layer(Geom::point()), layer(Geom::line()).data(csv!("geom_line", "data_layer"))],
            Facet::None,
        )
        .expect("layout resolves");
        assert!(got.layers[0].path.is_none() && got.layers[0].points.is_some());
        assert_eq!(w["layers"][1]["geom"], G);
        check_layer("layer1", &got.layers[1], &w["layers"][1], G);
        check_axis("x", &got.x, &w["panel"]["x"]);
        check_axis("y", &got.y, &w["panel"]["y"]);
    }

    #[test]
    fn a_single_row_is_a_typed_error_not_a_panic() {
        let w = want(fixture!("geom_line", "single"), ROW, "single");
        check_refused(
            build(csv!("geom_line", "data_single"), xy(), vec![layer(Geom::line())], Facet::None),
            &w,
        );
    }
}

// --- geom_path ---------------------------------------------------------------------------------

mod path {
    use super::*;

    const ROW: &str = "geom_path";
    const G: &str = "GeomPath";

    fn run(case: &str, json: &str, file: &str, aes: Aes, facet: Facet) {
        let w = want(json, ROW, case);
        let data =
            if file == "data" { csv!("geom_path", "data") } else { csv!("geom_path", "data_na") };
        let got = build(data, aes, vec![layer(Geom::path())], facet).expect("layout resolves");
        check(&got, &w, &[G]);
    }

    #[test]
    fn plain_keeps_data_order() {
        run("plain", fixture!("geom_path", "plain"), "data", xy(), Facet::None);
    }

    #[test]
    fn numeric_group() {
        run("group_num", fixture!("geom_path", "group_num"), "data", xy().group("gn"), Facet::None);
    }

    #[test]
    fn text_group() {
        run(
            "group_text",
            fixture!("geom_path", "group_text"),
            "data",
            xy().group("g"),
            Facet::None,
        );
    }

    #[test]
    fn na_rows_are_kept_in_place() {
        run("na", fixture!("geom_path", "na"), "na", xy(), Facet::None);
    }

    #[test]
    fn facet_wrap_keeps_panels_interleaved() {
        run("facet", fixture!("geom_path", "facet"), "data", xy(), wrap_f());
    }

    #[test]
    fn facet_wrap_with_groups() {
        run(
            "facet_group",
            fixture!("geom_path", "facet_group"),
            "data",
            xy().group("gn"),
            wrap_f(),
        );
    }

    #[test]
    fn fixed_colour_and_alpha() {
        let w = want(fixture!("geom_path", "colour_alpha"), ROW, "colour_alpha");
        let got = build(
            csv!("geom_path", "data"),
            xy(),
            vec![layer(styled(Geom::path(), Rgba::RED, 0.5))],
            Facet::None,
        )
        .expect("layout resolves");
        check(&got, &w, &[G]);
    }

    #[test]
    fn a_second_layer_with_its_own_data_widens_the_scales() {
        let w = want(fixture!("geom_path", "layer_data"), ROW, "layer_data");
        let got = build(
            csv!("geom_path", "data"),
            xy(),
            vec![layer(Geom::point()), layer(Geom::path()).data(csv!("geom_path", "data_layer"))],
            Facet::None,
        )
        .expect("layout resolves");
        check_layer("layer1", &got.layers[1], &w["layers"][1], G);
        check_axis("x", &got.x, &w["panel"]["x"]);
        check_axis("y", &got.y, &w["panel"]["y"]);
    }

    #[test]
    fn a_single_row_is_a_typed_error_not_a_panic() {
        let w = want(fixture!("geom_path", "single"), ROW, "single");
        check_refused(
            build(csv!("geom_path", "data_single"), xy(), vec![layer(Geom::path())], Facet::None),
            &w,
        );
    }
}

// --- geom_step ---------------------------------------------------------------------------------

mod step {
    use super::*;

    const ROW: &str = "geom_step";
    const G: &str = "GeomStep";

    fn run(case: &str, json: &str, geom: Geom, aes: Aes, facet: Facet) {
        let w = want(json, ROW, case);
        let got = build(csv!("geom_step", "data"), aes, vec![layer(geom)], facet)
            .expect("layout resolves");
        check(&got, &w, &[G]);
    }

    #[test]
    fn hv() {
        run("hv", fixture!("geom_step", "hv"), Geom::step(), xy(), Facet::None);
    }

    #[test]
    fn vh() {
        run(
            "vh",
            fixture!("geom_step", "vh"),
            Geom::step().direction(StepDirection::Vh),
            xy(),
            Facet::None,
        );
    }

    #[test]
    fn mid() {
        run(
            "mid",
            fixture!("geom_step", "mid"),
            Geom::step().direction(StepDirection::Mid),
            xy(),
            Facet::None,
        );
    }

    #[test]
    fn numeric_groups_get_their_own_stairs() {
        run(
            "group_hv",
            fixture!("geom_step", "group_hv"),
            Geom::step(),
            xy().group("gn"),
            Facet::None,
        );
    }

    #[test]
    fn text_groups_with_mid() {
        run(
            "group_mid",
            fixture!("geom_step", "group_mid"),
            Geom::step().direction(StepDirection::Mid),
            xy().group("g"),
            Facet::None,
        );
    }

    #[test]
    fn facet_wrap_makes_stairs_per_panel() {
        run(
            "facet_vh",
            fixture!("geom_step", "facet_vh"),
            Geom::step().direction(StepDirection::Vh),
            xy(),
            wrap_f(),
        );
    }

    #[test]
    fn a_one_row_group_has_no_stairs() {
        let w = want(fixture!("geom_step", "single_row_group"), ROW, "single_row_group");
        let got = build(
            csv!("geom_step", "data_single_group"),
            xy().group("gn"),
            vec![layer(Geom::step())],
            Facet::None,
        )
        .expect("layout resolves");
        check(&got, &w, &[G]);
    }

    /// A group of exactly two rows has stairs (3 vertices for hv, 4 for mid); only a one-row group
    /// has none. The fixture mixes a 2-row group with a 3-row group.
    fn two_row_group(case: &str, json: &str, geom: Geom) {
        let w = want(json, ROW, case);
        let got = build(
            csv!("geom_step", "data_two_row_group"),
            xy().group("gn"),
            vec![layer(geom)],
            Facet::None,
        )
        .expect("layout resolves");
        check(&got, &w, &[G]);
    }

    #[test]
    fn a_two_row_group_has_stairs_hv() {
        two_row_group("two_row_group_hv", fixture!("geom_step", "two_row_group_hv"), Geom::step());
    }

    #[test]
    fn a_two_row_group_has_stairs_mid() {
        two_row_group(
            "two_row_group_mid",
            fixture!("geom_step", "two_row_group_mid"),
            Geom::step().direction(StepDirection::Mid),
        );
    }

    #[test]
    fn fixed_colour_and_alpha() {
        let w = want(fixture!("geom_step", "colour_alpha"), ROW, "colour_alpha");
        let got = build(
            csv!("geom_step", "data"),
            xy(),
            vec![layer(styled(Geom::step(), Rgba::BLUE, 0.25))],
            Facet::None,
        )
        .expect("layout resolves");
        check(&got, &w, &[G]);
    }

    #[test]
    fn na_positions_are_a_typed_error() {
        let mut df = DataFrame::new();
        df.add_column_f32("x", &[1.0, f32::NAN, 3.0]);
        df.add_column_f32("y", &[1.0, 2.0, 3.0]);
        let r = build(df, xy(), vec![layer(Geom::step())], Facet::None);
        assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
    }
}

// --- reference lines ---------------------------------------------------------------------------

fn points_plus(extra: Vec<Layer>, facet: Facet, row: &str) -> crate::error::Result<BuiltLayout> {
    let mut layers = vec![layer(Geom::point())];
    layers.extend(extra);
    build(csv_for(row), xy(), layers, facet)
}

fn csv_for(row: &str) -> DataFrame {
    match row {
        "geom_hline" => csv!("geom_hline", "data"),
        "geom_vline" => csv!("geom_vline", "data"),
        _ => csv!("geom_abline", "data"),
    }
}

mod hline {
    use super::*;

    const ROW: &str = "geom_hline";
    const P: &str = "GeomPoint";
    const H: &str = "GeomHline";

    /// The point layer is checked for its scale effect only (row 1 owns its aesthetics).
    fn run(case: &str, json: &str, extra: Vec<Layer>, geoms: &[&str], facet: Facet) {
        let w = want(json, ROW, case);
        let got = points_plus(extra, facet, ROW).expect("layout resolves");
        let wl = w["layers"].as_array().expect("layers");
        assert_eq!(got.layers.len(), wl.len());
        assert_eq!(wl[0]["geom"], P);
        for (i, g) in geoms.iter().enumerate() {
            check_layer(&format!("layer{}", i + 1), &got.layers[i + 1], &wl[i + 1], g);
        }
        check_axis("x", &got.x, &w["panel"]["x"]);
        check_axis("y", &got.y, &w["panel"]["y"]);
    }

    #[test]
    fn inside_the_data_range_changes_nothing() {
        run(
            "inside",
            fixture!("geom_hline", "inside"),
            vec![layer(Geom::hline(3.0))],
            &[H],
            Facet::None,
        );
    }

    #[test]
    fn above_the_range_widens_y() {
        run(
            "above",
            fixture!("geom_hline", "above"),
            vec![layer(Geom::hline(30.0))],
            &[H],
            Facet::None,
        );
    }

    #[test]
    fn below_the_range_widens_y() {
        run(
            "below",
            fixture!("geom_hline", "below"),
            vec![layer(Geom::hline(-20.0))],
            &[H],
            Facet::None,
        );
    }

    #[test]
    fn two_layers_widen_both_ends() {
        run(
            "two_layers",
            fixture!("geom_hline", "two_layers"),
            vec![
                layer(Geom::hline(30.0)),
                layer(Geom::hline(-20.0).aes(Aes::new().color_value(Rgba::RED))),
            ],
            &[H, H],
            Facet::None,
        )
    }

    #[test]
    fn facet_wrap_replicates_the_line_per_panel() {
        run(
            "facet",
            fixture!("geom_hline", "facet"),
            vec![layer(Geom::hline(30.0))],
            &[H],
            wrap_f(),
        );
    }

    #[test]
    fn fixed_colour_and_alpha() {
        run(
            "colour_alpha",
            fixture!("geom_hline", "colour_alpha"),
            vec![layer(styled(Geom::hline(3.0), Rgba::RED, 0.5))],
            &[H],
            Facet::None,
        );
    }

    #[test]
    fn without_data_the_axes_are_degenerate_and_refused() {
        let w = want(fixture!("geom_hline", "only_ref"), ROW, "only_ref");
        let r = build(
            DataFrame::new(),
            Aes::new(),
            vec![layer(Geom::hline(2.0)), layer(Geom::vline(5.0))],
            Facet::None,
        );
        check_refused(r, &w);
    }
}

mod vline {
    use super::*;

    const ROW: &str = "geom_vline";
    const P: &str = "GeomPoint";
    const V: &str = "GeomVline";

    fn run(case: &str, json: &str, extra: Vec<Layer>, geoms: &[&str], facet: Facet) {
        let w = want(json, ROW, case);
        let got = points_plus(extra, facet, ROW).expect("layout resolves");
        let wl = w["layers"].as_array().expect("layers");
        assert_eq!(got.layers.len(), wl.len());
        assert_eq!(wl[0]["geom"], P);
        for (i, g) in geoms.iter().enumerate() {
            check_layer(&format!("layer{}", i + 1), &got.layers[i + 1], &wl[i + 1], g);
        }
        check_axis("x", &got.x, &w["panel"]["x"]);
        check_axis("y", &got.y, &w["panel"]["y"]);
    }

    #[test]
    fn inside_the_data_range_changes_nothing() {
        run(
            "inside",
            fixture!("geom_vline", "inside"),
            vec![layer(Geom::vline(3.0))],
            &[V],
            Facet::None,
        );
    }

    #[test]
    fn above_the_range_widens_x() {
        run(
            "above",
            fixture!("geom_vline", "above"),
            vec![layer(Geom::vline(30.0))],
            &[V],
            Facet::None,
        );
    }

    #[test]
    fn below_the_range_widens_x() {
        run(
            "below",
            fixture!("geom_vline", "below"),
            vec![layer(Geom::vline(-20.0))],
            &[V],
            Facet::None,
        );
    }

    #[test]
    fn two_layers_widen_both_ends() {
        run(
            "two_layers",
            fixture!("geom_vline", "two_layers"),
            vec![
                layer(Geom::vline(30.0)),
                layer(Geom::vline(-20.0).aes(Aes::new().color_value(Rgba::RED))),
            ],
            &[V, V],
            Facet::None,
        )
    }

    #[test]
    fn facet_wrap_replicates_the_line_per_panel() {
        run(
            "facet",
            fixture!("geom_vline", "facet"),
            vec![layer(Geom::vline(30.0))],
            &[V],
            wrap_f(),
        );
    }

    #[test]
    fn fixed_colour_and_alpha() {
        run(
            "colour_alpha",
            fixture!("geom_vline", "colour_alpha"),
            vec![layer(styled(Geom::vline(3.0), Rgba::RED, 0.5))],
            &[V],
            Facet::None,
        );
    }

    #[test]
    fn without_data_the_axes_are_degenerate_and_refused() {
        let w = want(fixture!("geom_vline", "only_ref"), ROW, "only_ref");
        let r = build(
            DataFrame::new(),
            Aes::new(),
            vec![layer(Geom::vline(2.0)), layer(Geom::hline(5.0))],
            Facet::None,
        );
        check_refused(r, &w);
    }
}

mod abline {
    use super::*;

    const ROW: &str = "geom_abline";
    const A: &str = "GeomAbline";

    fn run(case: &str, json: &str, g: Geom, facet: Facet) {
        let w = want(json, ROW, case);
        let got = points_plus(vec![layer(g)], facet, ROW).expect("layout resolves");
        let wl = w["layers"].as_array().expect("layers");
        assert_eq!(got.layers.len(), 2);
        check_layer("layer1", &got.layers[1], &wl[1], A);
        check_axis("x", &got.x, &w["panel"]["x"]);
        check_axis("y", &got.y, &w["panel"]["y"]);
    }

    #[test]
    fn plain() {
        run("plain", fixture!("geom_abline", "plain"), Geom::abline(1.5, 0.5), Facet::None);
    }

    #[test]
    fn negative_slope() {
        run("negative", fixture!("geom_abline", "negative"), Geom::abline(-2.0, 4.0), Facet::None);
    }

    #[test]
    fn a_line_far_outside_the_data_trains_no_scale() {
        run(
            "far_outside",
            fixture!("geom_abline", "far_outside"),
            Geom::abline(1000.0, 5000.0),
            Facet::None,
        );
    }

    #[test]
    fn zero_slope_outside_the_data_trains_no_scale() {
        run(
            "zero_slope",
            fixture!("geom_abline", "zero_slope"),
            Geom::abline(0.0, -50.0),
            Facet::None,
        );
    }

    /// An abline trains no scale, so a plot whose only layer is a reference line has no range on
    /// either axis. The plot's own data and aesthetics must not train it behind the layer's back
    /// (ggplot2: `inherit.aes = FALSE`); the typed error is the contract.
    #[test]
    fn only_a_reference_line_trains_nothing_and_the_axes_are_refused() {
        let r = build(
            csv!("geom_abline", "data"),
            xy(),
            vec![layer(Geom::abline(1.5, 0.5))],
            Facet::None,
        );
        assert!(matches!(r, Err(Error::ScaleDomain(_))), "{r:?}");
    }

    #[test]
    fn an_abline_with_its_own_data_still_trains_nothing() {
        let mut far = DataFrame::new();
        far.add_column_f32("x", &[1000.0, 2000.0]);
        far.add_column_f32("y", &[-500.0, 500.0]);
        let l = Layer::new(Geom::abline(1.0, 1.0)).data(far);
        let r = build(DataFrame::new(), xy(), vec![l], Facet::None);
        assert!(matches!(r, Err(Error::ScaleDomain(_))), "{r:?}");
    }

    #[test]
    fn facet_wrap_replicates_the_line_per_panel() {
        run("facet", fixture!("geom_abline", "facet"), Geom::abline(1.0, 0.0), wrap_f());
    }

    #[test]
    fn fixed_colour_and_alpha() {
        run(
            "colour_alpha",
            fixture!("geom_abline", "colour_alpha"),
            styled(Geom::abline(1.0, 1.0), Rgba::RED, 0.5),
            Facet::None,
        );
    }
}

// --- never panic, never silently wrong ---------------------------------------------------------

mod robustness {
    use super::*;

    fn all_path_geoms() -> Vec<Geom> {
        vec![
            Geom::line(),
            Geom::path(),
            Geom::step(),
            Geom::step().direction(StepDirection::Mid),
            Geom::hline(1.0),
            Geom::vline(1.0),
            Geom::abline(1.0, 0.0),
        ]
    }

    fn ok_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column_f32("x", &[1.0, 2.0, 3.0]);
        df.add_column_f32("y", &[1.0, 4.0, 9.0]);
        df.add_column_str("f", &["a", "b", "a"]);
        df
    }

    #[test]
    fn empty_frame_and_zero_row_columns_are_errors_for_every_geom() {
        for g in all_path_geoms() {
            let r = build(DataFrame::new(), xy(), vec![layer(g.clone())], Facet::None);
            assert!(r.is_err(), "{:?} on an empty frame returned {r:?}", g.geom_type);
            let mut zero = DataFrame::new();
            zero.add_column_f32("x", &[]);
            zero.add_column_f32("y", &[]);
            let r = build(zero, xy(), vec![layer(g.clone())], Facet::None);
            assert!(r.is_err(), "{:?} on zero rows returned {r:?}", g.geom_type);
        }
    }

    #[test]
    fn ragged_columns_pad_with_na_instead_of_panicking() {
        let mut df = DataFrame::new();
        df.add_column_f32("x", &[1.0, 2.0, 3.0, 4.0, 5.0]);
        df.add_column_f32("y", &[1.0, 2.0, 3.0]);
        for g in [Geom::line(), Geom::path()] {
            let l = build(df.clone(), xy(), vec![layer(g)], Facet::None).expect("a value");
            assert_eq!(l.layers[0].x.len(), 5);
            assert!(l.layers[0].y[3].is_nan() && l.layers[0].y[4].is_nan());
        }
        // A step needs every position, so a short column is its typed error.
        let r = build(df.clone(), xy(), vec![layer(Geom::step())], Facet::None);
        assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
        // A short group column is refused, not indexed past its end.
        let r = build(df.clone(), xy().group("y"), vec![layer(Geom::line())], Facet::None);
        assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
    }

    #[test]
    fn ragged_columns_under_a_facet_do_not_panic() {
        let mut df = DataFrame::new();
        df.add_column_f32("x", &[1.0, 2.0, 3.0, 4.0]);
        df.add_column_f32("y", &[1.0, 2.0]);
        df.add_column_str("f", &["a", "b", "a"]);
        for g in all_path_geoms() {
            let _ = build(df.clone(), xy(), vec![layer(g)], wrap_f());
        }
    }

    #[test]
    fn infinite_and_text_positions_are_typed_errors() {
        let mut inf = ok_df();
        inf.add_column_f32("y", &[1.0, f32::INFINITY, 2.0]);
        let r = build(inf, xy(), vec![layer(Geom::line())], Facet::None);
        assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
        let mut text = ok_df();
        text.add_column_str("x", &["a", "b", "c"]);
        let r = build(text, xy(), vec![layer(Geom::path())], Facet::None);
        assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
    }

    #[test]
    fn several_nan_positions_keep_their_order_after_the_finite_ones() {
        let mut df = ok_df();
        df.add_column_f32("x", &[f32::NAN, 2.0, f32::NAN, 1.0]);
        df.add_column_f32("y", &[7.0, 8.0, 9.0, 6.0]);
        df.add_column_str("f", &["a", "a", "a", "a"]);
        let l = build(df, xy(), vec![layer(Geom::line())], Facet::None).expect("a value");
        assert_eq!(l.layers[0].y, [6.0, 8.0, 7.0, 9.0]);
    }

    #[test]
    fn nan_positions_are_values_for_line_and_path() {
        let mut df = ok_df();
        df.add_column_f32("x", &[f32::NAN, 2.0, 1.0]);
        let l = build(df, xy(), vec![layer(Geom::line())], Facet::None).expect("a value");
        assert_eq!(l.layers[0].x[..2], [1.0, 2.0]);
        assert!(l.layers[0].x[2].is_nan());
    }

    #[test]
    fn missing_columns_are_unknown_column_errors() {
        let r = build(ok_df(), Aes::new().x("x").y("nope"), vec![layer(Geom::line())], Facet::None);
        assert!(matches!(r, Err(Error::UnknownColumn(_))), "{r:?}");
        let r = build(ok_df(), xy().group("nope"), vec![layer(Geom::path())], Facet::None);
        assert!(matches!(r, Err(Error::UnknownColumn(_))), "{r:?}");
        let r = build(
            ok_df(),
            xy(),
            vec![layer(Geom::line())],
            Facet::Wrap { var: "nope".into(), ncol: 2 },
        );
        assert!(matches!(r, Err(Error::UnknownColumn(_))), "{r:?}");
    }

    #[test]
    fn unsupported_aesthetics_are_refused_not_ignored() {
        for a in [
            xy().color("f"),
            xy().size("x"),
            xy().alpha("x"),
            xy().fill("f"),
            xy().shape("f"),
            xy().label("f"),
        ] {
            let r = build(ok_df(), a, vec![layer(Geom::line())], Facet::None);
            assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
        }
        let r = build(
            ok_df(),
            xy(),
            vec![layer(Geom::line().aes(Aes::new().size_value(2.0)))],
            Facet::None,
        );
        assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
        let r = build(ok_df(), xy(), vec![layer(Geom::line().width(3.0))], Facet::None);
        assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
        // A reference line takes no x/y/group.
        let r =
            build(ok_df(), xy(), vec![layer(Geom::hline(1.0).aes(Aes::new().x("x")))], Facet::None);
        assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
    }

    #[test]
    fn non_finite_reference_parameters_are_refused() {
        for g in [
            Geom::hline(f32::NAN),
            Geom::vline(f32::INFINITY),
            Geom::abline(f32::NAN, 0.0),
            Geom::abline(1.0, f32::INFINITY),
        ] {
            let r = build(ok_df(), xy(), vec![layer(Geom::point()), layer(g)], Facet::None);
            assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
        }
    }

    #[test]
    fn unsupported_facets_and_facet_columns_are_refused() {
        let grid = Facet::Grid { row: "f".into(), col: "f".into() };
        for facet in [Facet::Row { var: "f".into() }, Facet::Col { var: "f".into() }, grid] {
            let r = build(ok_df(), xy(), vec![layer(Geom::line())], facet);
            assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
        }
        let r = build(
            ok_df(),
            xy(),
            vec![layer(Geom::line())],
            Facet::Wrap { var: "x".into(), ncol: 2 },
        );
        assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
    }

    #[test]
    fn layer_data_must_match_the_plot_panels() {
        // Its own data lacks the facet column.
        let mut own = DataFrame::new();
        own.add_column_f32("x", &[1.0, 2.0]);
        own.add_column_f32("y", &[1.0, 2.0]);
        let r = build(ok_df(), xy(), vec![layer(Geom::line()).data(own.clone())], wrap_f());
        assert!(matches!(r, Err(Error::UnknownColumn(_))), "{r:?}");
        // Its own data has a level the plot data lacks.
        own.add_column_str("f", &["a", "zzz"]);
        let r = build(ok_df(), xy(), vec![layer(Geom::line()).data(own.clone())], wrap_f());
        assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
        // Matching levels land in the plot's panels.
        own.add_column_str("f", &["b", "a"]);
        let l =
            build(ok_df(), xy(), vec![layer(Geom::path()).data(own)], wrap_f()).expect("a value");
        let p = l.layers[0].path.as_ref().expect("path");
        assert_eq!(p.panel, [1, 0]);
    }

    #[test]
    fn group_columns_must_be_complete_and_homogeneous() {
        let mut df = ok_df();
        df.add_column_f32("g", &[1.0, f32::NAN, 2.0]);
        let r = build(df.clone(), xy().group("g"), vec![layer(Geom::line())], Facet::None);
        assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
        df.add_column_str("g", &["b", "a", "b"]);
        let l = build(df, xy().group("g"), vec![layer(Geom::path())], Facet::None).expect("value");
        assert_eq!(l.layers[0].path.as_ref().expect("path").group, [2, 1, 2]);
        // Negative zero and zero are one group.
        let mut z = ok_df();
        z.add_column_f32("g", &[0.0, -0.0, 1.0]);
        let l = build(z, xy().group("g"), vec![layer(Geom::path())], Facet::None).expect("value");
        assert_eq!(l.layers[0].path.as_ref().expect("path").group, [1, 1, 2]);
    }

    #[test]
    fn a_default_aes_with_no_columns_named_x_y_is_unknown_column() {
        let mut df = DataFrame::new();
        df.add_column_f32("a", &[1.0, 2.0]);
        let r = build(df, Aes::new(), vec![layer(Geom::line())], Facet::None);
        assert!(matches!(r, Err(Error::UnknownColumn(_))), "{r:?}");
    }

    #[test]
    fn a_missing_text_cell_in_the_facet_column_is_refused() {
        let mut df = ok_df();
        df.add_column_f32("f", &[1.0, 2.0, 3.0]);
        let r = build(df, xy(), vec![layer(Geom::line())], wrap_f());
        assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
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
            .aes(xy())
            .layer(layer(Geom::point()))
            .layer(layer(g))
            .coord(coord)
            .build()?
            .layout()
    }

    /// `layout()` applies no coordinate system, so a coordinate system it would silently ignore is
    /// refused for every geom of these rows (R gives different panel ranges and breaks for each).
    #[test]
    fn every_non_default_coord_is_refused_for_every_geom() {
        for g in all_path_geoms() {
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
    fn the_refusal_names_the_coord() {
        let msg = |c: Coord| match with_coord(Geom::line(), c) {
            Err(Error::UnsupportedCoord(m)) => m,
            other => panic!("expected UnsupportedCoord, got {other:?}"),
        };
        assert!(msg(Coord::cartesian().flip()).contains("coord_flip"));
        assert!(msg(Coord::cartesian().xlim(0.0, 1.0)).contains("limits"));
        assert!(msg(Coord::polar()).contains("coord_polar"));
        assert!(msg(Coord::fixed(2.0)).contains("coord_fixed"));
    }

    #[test]
    fn the_default_coord_is_accepted_for_every_geom() {
        for g in all_path_geoms() {
            let r = with_coord(g.clone(), Coord::cartesian());
            assert!(r.is_ok(), "{:?} under the default coord returned {r:?}", g.geom_type);
        }
    }

    /// The refusal is scoped to these geoms: a plot with only a point layer keeps its behaviour.
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

    // --- stats and companion layers (paiml/aprender#6978) ---------------------------------------

    /// Every stat but identity, with the parameters the review measured against R
    /// (`geom_line(stat = "smooth")` is 80 rows, `stat = "bin", bins = 3` is 3 rows).
    fn non_identity_stats() -> Vec<Stat> {
        vec![
            Stat::Count,
            Stat::Bin { bins: 3 },
            Stat::Bin { bins: 30 },
            Stat::Density,
            Stat::Boxplot,
            Stat::Smooth,
            Stat::Bin2d { bins_x: 3, bins_y: 3 },
            Stat::Sum,
            Stat::Mean,
        ]
    }

    /// `geom` alone, and after a point layer (so the check is not on the first layer only).
    fn with_layers(geom: &Geom, lead_point: bool) -> crate::error::Result<BuiltLayout> {
        let mut p = GGPlot::new().data(ok_df()).aes(xy());
        if lead_point {
            p = p.layer(layer(Geom::point()));
        }
        p.layer(layer(geom.clone())).build()?.layout()
    }

    /// `layout()` computes no stat, so each stat it would silently skip is refused.
    fn assert_every_stat_refused(geoms: &[Geom]) {
        for g in geoms {
            for s in non_identity_stats() {
                for lead in [false, true] {
                    let r = with_layers(&g.clone().stat(s.clone()), lead);
                    assert!(
                        matches!(r, Err(Error::UnsupportedStat(_))),
                        "{:?} with {s:?} (after a point layer: {lead}) returned {r:?}",
                        g.geom_type
                    );
                }
            }
        }
    }

    #[test]
    fn every_non_identity_stat_is_refused_on_line() {
        assert_every_stat_refused(&[Geom::line()]);
    }

    #[test]
    fn every_non_identity_stat_is_refused_on_path() {
        assert_every_stat_refused(&[Geom::path()]);
    }

    #[test]
    fn every_non_identity_stat_is_refused_on_step() {
        assert_every_stat_refused(&[
            Geom::step(),
            Geom::step().direction(StepDirection::Vh),
            Geom::step().direction(StepDirection::Mid),
        ]);
    }

    #[test]
    fn every_non_identity_stat_is_refused_on_hline() {
        assert_every_stat_refused(&[Geom::hline(1.0)]);
    }

    #[test]
    fn every_non_identity_stat_is_refused_on_vline() {
        assert_every_stat_refused(&[Geom::vline(1.0)]);
    }

    #[test]
    fn every_non_identity_stat_is_refused_on_abline() {
        assert_every_stat_refused(&[Geom::abline(1.0, 0.0)]);
    }

    /// `stat = "identity"` is ggplot2's default for these geoms: stated explicitly, it changes
    /// nothing, so the layout equals the one without a stat.
    #[test]
    fn the_identity_stat_is_accepted_and_changes_nothing() {
        for g in all_path_geoms() {
            for lead in [false, true] {
                let plain = with_layers(&g, lead);
                let ident = with_layers(&g.clone().stat(Stat::Identity), lead);
                match (plain, ident) {
                    (Ok(a), Ok(b)) => assert_eq!(format!("{a:?}"), format!("{b:?}")),
                    // A reference line alone is refused for its degenerate axes, with or without
                    // the identity stat.
                    (Err(Error::ScaleDomain(_)), Err(Error::ScaleDomain(_))) if !lead => {}
                    (a, b) => panic!("{:?}: without stat {a:?}, identity {b:?}", g.geom_type),
                }
            }
        }
    }

    /// A stat on the point layer would change the shared scales too, so it is refused while one
    /// of these layers is present, whichever layer comes first.
    #[test]
    fn a_stat_on_a_companion_point_layer_is_refused() {
        for g in all_path_geoms() {
            for s in non_identity_stats() {
                let pt = layer(Geom::point().stat(s.clone()));
                for layers in [vec![pt.clone(), layer(g.clone())], vec![layer(g.clone()), pt]] {
                    let r = build(ok_df(), xy(), layers, Facet::None);
                    assert!(
                        matches!(r, Err(Error::UnsupportedStat(_))),
                        "point with {s:?} beside {:?} returned {r:?}",
                        g.geom_type
                    );
                }
            }
        }
    }

    /// R trains different scales for a line plus `geom_bar`/`geom_col`, `geom_histogram`,
    /// `geom_area` or `geom_tile` (bar and tile widths, a zero baseline, `stat_align`), and
    /// different layer data for the stat geoms; `layout()` would train raw x/y. Refused.
    #[test]
    fn every_companion_geom_but_point_is_refused() {
        let companions = [
            Geom::area(),
            Geom::bar(),
            Geom::bar().stat(Stat::Identity),
            Geom::histogram(),
            Geom::boxplot(),
            Geom::violin(),
            Geom::tile(),
            Geom::text(),
            Geom::smooth(),
        ];
        for g in all_path_geoms() {
            for c in &companions {
                for layers in [
                    vec![layer(c.clone()), layer(g.clone())],
                    vec![layer(g.clone()), layer(c.clone())],
                ] {
                    let r = build(ok_df(), xy(), layers, Facet::None);
                    assert!(
                        matches!(r, Err(Error::UnsupportedLayer(_))),
                        "{:?} beside {:?} returned {r:?}",
                        c.geom_type,
                        g.geom_type
                    );
                }
            }
        }
    }

    #[test]
    fn the_stat_refusal_names_the_stat() {
        let msg = |s: Stat| match with_layers(&Geom::line().stat(s), true) {
            Err(Error::UnsupportedStat(m)) => m,
            other => panic!("expected UnsupportedStat, got {other:?}"),
        };
        assert!(msg(Stat::Smooth).starts_with("stat_smooth "));
        assert!(msg(Stat::Bin { bins: 3 }).starts_with("stat_bin "));
        assert!(msg(Stat::Count).starts_with("stat_count "));
        assert!(msg(Stat::Density).starts_with("stat_density "));
        assert!(msg(Stat::Boxplot).starts_with("stat_boxplot "));
        assert!(msg(Stat::Bin2d { bins_x: 2, bins_y: 2 }).starts_with("stat_bin_2d "));
        assert!(msg(Stat::Sum).starts_with("stat_summary(fun = sum) "));
        assert!(msg(Stat::Mean).starts_with("stat_summary(fun = mean) "));
    }

    /// The refusals are scoped to plots with these geoms: other plots keep their behaviour.
    #[test]
    fn a_plot_without_these_geoms_is_not_refused_for_a_stat_or_companion() {
        let layers = vec![layer(Geom::point().stat(Stat::Smooth)), layer(Geom::bar())];
        assert!(build(ok_df(), xy(), layers, Facet::None).is_ok());
    }

    #[test]
    fn no_path_layer_leaves_other_geoms_unchanged() {
        let l = build(ok_df(), xy(), vec![layer(Geom::point())], Facet::None).expect("value");
        assert!(l.layers[0].path.is_none());
    }

    #[test]
    fn step_geom_defaults_to_hv_and_direction_ignores_other_geoms() {
        let l = build(ok_df(), xy(), vec![layer(Geom::step())], Facet::None).expect("value");
        assert_eq!(
            l.layers[0].path.as_ref().expect("path").stairs.as_ref().expect("stairs").direction,
            StepDirection::Hv
        );
        let g = Geom::line().direction(StepDirection::Mid);
        assert!(matches!(g.geom_type, super::super::geom::GeomType::Line { .. }));
    }

    #[test]
    fn rendering_a_plot_with_these_geoms_still_works() {
        for g in all_path_geoms() {
            let p = GGPlot::new().data(ok_df()).aes(xy()).geom(g).build().expect("builds");
            let _ = p.to_svg();
        }
    }
}
