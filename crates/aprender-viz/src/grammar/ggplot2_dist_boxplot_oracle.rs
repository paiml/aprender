//! ggplot2 conformance oracle for `geom_boxplot()` (PLOT-GG-001, paiml/aprender#6965, master
//! #6909): `stat_boxplot`, `GeomBoxplot$setup_data`, `position_dodge2` and the retrained position
//! scales. `fixtures/ggplot2/geom_boxplot/expected.json` holds every case, produced by ggplot2
//! 4.0.3 under R 4.6.2 and C collation (`generate.R`); R is never run here. Each case names the
//! rows of `data.csv` it used.
//!
//! Tolerance is the row-1 pattern: `|a - b| <= 1e-9 * max(|a|, |b|) + 1e-12`; groups, labels and
//! counts compare exactly. Every case is built through the public builder and read back with
//! `layout()`.

use serde_json::Value;

use super::ggplot2_geom_area_oracle::{
    assert_close_slice, frame_from_csv, ints, nums, report_max_err, reset_max_err, strs,
};
use super::{
    Aes, BoxRow, BuiltLayout, DataFrame, GGPlot, Geom, Layer, PositionScale, TerminalPlot,
};
use crate::error::Error;

const DATA: &str = include_str!("../../fixtures/ggplot2/geom_boxplot/data.csv");
const EXPECTED: &str = include_str!("../../fixtures/ggplot2/geom_boxplot/expected.json");

/// The number of cases in the fixture; `every_case_is_run` ties it to the table below.
const CASES: usize = 29;

fn fixture() -> Value {
    let v: Value = serde_json::from_str(EXPECTED).expect("fixture is valid JSON");
    assert_eq!(v["versions"]["ggplot2"], "4.0.3", "fixture was not produced by the pinned ggplot2");
    assert_eq!(v["versions"]["collate"], "C");
    assert_eq!(v["row"], "geom_boxplot");
    v
}

/// The frame of the case's rows: `rows` are 1-based indices into `data.csv`.
fn subset(rows: &[i64]) -> DataFrame {
    let lines: Vec<&str> = DATA.lines().collect();
    let mut out = String::from(lines.first().copied().expect("header"));
    for r in rows {
        out.push('\n');
        out.push_str(lines.get(usize::try_from(*r).expect("row index")).expect("row exists"));
    }
    frame_from_csv(&out)
}

/// The R call of a case: the aesthetics and the `geom_boxplot()` arguments.
fn setup(name: &str) -> (Aes, Geom) {
    let xy = || Aes::new().x("x").y("y");
    let nxy = || Aes::new().x("n").y("y");
    let g = Geom::gg_boxplot;
    match name {
        "no_x_single" => (Aes::new().y("y"), g()),
        "coef_6" => (xy(), g().coef(6.0)),
        "coef_zero" => (xy(), g().coef(0.0)),
        "coef_large" => (xy(), g().coef(100.0)),
        "width_half" => (xy(), g().width(0.5)),
        "width_wide_overlap" => (xy(), g().width(2.0)),
        "fill_split" | "three_boxes_one_x" => (xy().fill("f"), g()),
        "fill_split_width_half" => (xy().fill("f"), g().width(0.5)),
        "colour_split" => (xy().color("f"), g()),
        "group_split" => (xy().group("gr"), g()),
        "group_split_width_half" => (xy().group("gr"), g().width(0.5)),
        "continuous_x_one_box_width_half" => (nxy(), g().width(0.5)),
        "no_x_fill" => (Aes::new().y("y").fill("f"), g()),
        "continuous_x_one_box" => (nxy(), g()),
        "continuous_x_grouped" | "resolution_counts_zero" | "na_numeric_x_dropped" => {
            (nxy().group("n"), g())
        }
        "continuous_x_fill" => (nxy().fill("f"), g()),
        _ => (xy(), g()),
    }
}

/// Cases whose mappings are split between `ggplot(aes())` and `geom_boxplot(aes())`:
/// `(plot aes, layer aes)`. R interacts the layer's discrete mappings before the plot's.
fn layered(name: &str) -> Option<(Aes, Aes)> {
    let xy = || Aes::new().x("x").y("y");
    match name {
        "layer_colour_plot_fill" => Some((xy().fill("f"), Aes::new().color("c"))),
        "layer_fill_plot_colour" => Some((xy().color("c"), Aes::new().fill("f"))),
        "layer_fill_only" => Some((xy(), Aes::new().fill("f"))),
        "layer_x_plot_fill" => Some((Aes::new().fill("f"), xy())),
        _ => None,
    }
}

fn build(data: DataFrame, aes: Aes, geom: Geom) -> crate::error::Result<BuiltLayout> {
    GGPlot::new().data(data).aes(aes).layer(Layer::new(geom)).build()?.layout()
}

fn build_layered(
    data: DataFrame,
    plot: Aes,
    layer: Aes,
    geom: Geom,
) -> crate::error::Result<BuiltLayout> {
    GGPlot::new().data(data).aes(plot).layer(Layer::new(geom).aes(layer)).build()?.layout()
}

fn check_scale(name: &str, got: &PositionScale, w: &Value) {
    assert_close_slice(
        &format!("{name}.continuous_range"),
        &[got.continuous_range.0, got.continuous_range.1],
        &nums(&w["continuous_range"]),
    );
    assert_eq!(got.labels, strs(&w["labels"]), "{name}.labels");
    if w["kind"] == "discrete" {
        assert_eq!(got.levels.as_deref(), Some(strs(&w["limits"]).as_slice()), "{name}.levels");
        assert_close_slice(&format!("{name}.breaks"), &got.breaks, &nums(&w["break_positions"]));
        return;
    }
    assert!(got.levels.is_none(), "{name}: continuous in R");
    assert_close_slice(
        &format!("{name}.limits"),
        &[got.limits.0, got.limits.1],
        &nums(&w["limits"]),
    );
    assert_close_slice(&format!("{name}.breaks"), &got.breaks, &nums(&w["breaks"]));
    assert_close_slice(&format!("{name}.minor"), &got.minor_breaks, &nums(&w["minor_breaks"]));
    let censored = w["n_breaks_censored"].as_u64().expect("count");
    assert_eq!(got.n_breaks_censored as u64, censored, "{name}.n_breaks_censored");
}

fn check_boxes(case: &str, got: &[BoxRow], w: &Value) {
    let col = |k: &str| nums(&w[k]);
    assert_eq!(got.iter().map(|b| b.group).collect::<Vec<_>>(), ints(&w["group"]), "{case}.group");
    let fields: [(&str, fn(&BoxRow) -> f64); 13] = [
        ("x", |b| b.x),
        ("xmin", |b| b.xmin),
        ("xmax", |b| b.xmax),
        ("ymin", |b| b.ymin),
        ("lower", |b| b.lower),
        ("middle", |b| b.middle),
        ("upper", |b| b.upper),
        ("ymax", |b| b.ymax),
        ("notchlower", |b| b.notchlower),
        ("notchupper", |b| b.notchupper),
        ("ymin_final", |b| b.ymin_final),
        ("ymax_final", |b| b.ymax_final),
        ("relvarwidth", |b| b.relvarwidth),
    ];
    for (name, f) in fields {
        let g: Vec<f64> = got.iter().map(f).collect();
        assert_close_slice(&format!("{case}.{name}"), &g, &col(name));
    }
    let want = w["outliers"].as_array().expect("outliers array");
    assert_eq!(got.len(), want.len(), "{case}: outlier lists");
    for (i, (b, o)) in got.iter().zip(want).enumerate() {
        assert_close_slice(&format!("{case}.outliers[{i}]"), &b.outliers, &nums(o));
    }
}

fn run_case(c: &Value) {
    let name = c["name"].as_str().expect("case name");
    reset_max_err();
    let data = subset(&ints(&c["rows"]));
    let got = match layered(name) {
        Some((plot, layer)) => build_layered(data, plot, layer, Geom::gg_boxplot()),
        None => {
            let (aes, geom) = setup(name);
            build(data, aes, geom)
        }
    }
    .unwrap_or_else(|e| panic!("{name}: {e:?}"));
    assert_eq!(got.layers.len(), 1, "{name}: layer count");
    let boxes = got.layers[0].boxes.as_deref().expect("a boxplot layer resolves its boxes");
    check_boxes(name, boxes, &c["layer"]);
    assert_eq!(c["layer"]["geom"], "GeomBoxplot");
    check_scale(&format!("{name}.x"), &got.x, &c["panel"]["x"]);
    check_scale(&format!("{name}.y"), &got.y, &c["panel"]["y"]);
    report_max_err("geom_boxplot", name);
}

fn case(name: &str) {
    let f = fixture();
    let c = f["cases"]
        .as_array()
        .and_then(|a| a.iter().find(|c| c["name"] == name))
        .unwrap_or_else(|| panic!("fixture lacks case {name}"));
    run_case(c);
}

#[test]
fn one_box_without_x() {
    case("no_x_single");
}
#[test]
fn one_box_with_a_text_x() {
    case("text_x_single");
}
#[test]
fn several_categories_with_outliers_on_both_sides() {
    case("categories_outliers");
}
#[test]
fn groups_of_one_to_five_values() {
    case("small_n_1_to_5");
}
#[test]
fn ties_and_all_equal_values() {
    case("ties_and_all_equal");
}
#[test]
fn coef_six_moves_the_fences() {
    case("coef_6");
}
#[test]
fn coef_zero_makes_everything_outside_the_box_an_outlier() {
    case("coef_zero");
}
#[test]
fn a_huge_coef_has_no_outliers() {
    case("coef_large");
}
#[test]
fn a_given_width_sets_the_box_width() {
    case("width_half");
}
#[test]
fn a_wide_width_makes_neighbours_overlap_and_dodge() {
    case("width_wide_overlap");
}
#[test]
fn fill_splits_each_category_into_dodged_boxes() {
    case("fill_split");
}
#[test]
fn colour_splits_like_fill() {
    case("colour_split");
}
#[test]
fn a_group_column_splits_a_category() {
    case("group_split");
}
#[test]
fn a_group_spanning_several_x_ignores_width() {
    case("group_split_width_half");
}
#[test]
fn a_continuous_x_group_ignores_width() {
    case("continuous_x_one_box_width_half");
}
#[test]
fn fill_with_a_given_width() {
    case("fill_split_width_half");
}
#[test]
fn three_boxes_share_one_x() {
    case("three_boxes_one_x");
}
#[test]
fn fill_without_x() {
    case("no_x_fill");
}
#[test]
fn a_continuous_x_is_one_box() {
    case("continuous_x_one_box");
}
#[test]
fn a_continuous_x_grouped_by_itself() {
    case("continuous_x_grouped");
}
#[test]
fn a_continuous_x_with_fill() {
    case("continuous_x_fill");
}
#[test]
fn resolution_counts_zero_as_a_value() {
    case("resolution_counts_zero");
}
#[test]
fn a_missing_y_is_dropped() {
    case("na_y_dropped");
}
#[test]
fn a_missing_numeric_x_is_dropped() {
    case("na_numeric_x_dropped");
}

#[test]
fn a_group_with_no_finite_y_vanishes() {
    case("group_all_missing_y_vanishes");
}

#[test]
fn a_layer_colour_is_interacted_before_a_plot_fill() {
    case("layer_colour_plot_fill");
}
#[test]
fn a_layer_fill_is_interacted_before_a_plot_colour() {
    case("layer_fill_plot_colour");
}
#[test]
fn a_layer_fill_comes_before_a_plot_x() {
    case("layer_fill_only");
}
#[test]
fn a_layer_x_comes_before_a_plot_fill() {
    case("layer_x_plot_fill");
}

/// The table above covers every fixture case, and the fixture holds `CASES` of them.
#[test]
fn every_case_is_run() {
    let f = fixture();
    let names: Vec<&str> =
        f["cases"].as_array().expect("cases").iter().filter_map(|c| c["name"].as_str()).collect();
    assert_eq!(names.len(), CASES);
    let tested = include_str!("ggplot2_dist_boxplot_oracle.rs");
    for n in names {
        assert!(tested.contains(&format!("case(\"{n}\")")), "case {n} has no test");
    }
}

fn refused(r: crate::error::Result<BuiltLayout>) -> bool {
    matches!(r, Err(Error::Rendering(_)))
}

fn frame() -> DataFrame {
    subset(&(1..=10).collect::<Vec<i64>>())
}

#[test]
fn out_of_scope_input_is_a_typed_error() {
    let g = Geom::gg_boxplot;
    let mut inf = frame();
    inf.add_column_f32("inf", &[f32::INFINITY; 10]);
    let mut nulls = DataFrame::new();
    nulls.add_column_f32("y", &[1.0, 2.0, f32::NAN]);
    nulls.add_column_str("g", &["a", "b", "b"]);
    nulls.add_column_f32("k", &[1.0, f32::NAN, 3.0]);
    let mut allnan = DataFrame::new();
    allnan.add_column_f32("y", &[f32::NAN; 3]);
    allnan.add_column_str("g", &["a", "b", "b"]);
    let cases: Vec<(&str, crate::error::Result<BuiltLayout>)> = vec![
        ("flipped: x only", build(frame(), Aes::new().x("y"), g())),
        ("width zero", build(frame(), Aes::new().y("y"), g().width(0.0))),
        ("width negative", build(frame(), Aes::new().y("y"), g().width(-1.0))),
        ("coef nan", build(frame(), Aes::new().y("y"), g().coef(f32::NAN))),
        ("coef inf", build(frame(), Aes::new().y("y"), g().coef(f32::INFINITY))),
        ("inf y", build(inf, Aes::new().y("inf"), g())),
        ("text y", build(frame(), Aes::new().y("x"), g())),
        ("continuous fill", build(nulls.clone(), Aes::new().y("y").fill("k"), g())),
        ("no finite y at all", build(allnan, Aes::new().x("g").y("y"), g())),
        ("missing group value survives", build(nulls, Aes::new().y("y").group("k"), g())),
    ];
    for (name, r) in cases {
        assert!(refused(r), "{name}");
    }
}

/// A discrete `fill` and `colour` on two columns in one `aes()` call: R orders them as written
/// (measured: the group ids and the box positions change), which `Aes` does not record.
#[test]
fn fill_and_colour_in_one_aes_call_are_refused() {
    let g = Geom::gg_boxplot;
    let rows = ints(&fixture()["cases"][CASES - 1]["rows"]);
    assert_eq!(fixture()["cases"][CASES - 1]["name"], "layer_x_plot_fill");
    let d = || subset(&rows);
    let xy = || Aes::new().x("x").y("y");
    let both = || xy().fill("f").color("c");
    let same_col = build(d(), xy().fill("f").color("f"), g());
    let cases = [
        ("both in the plot call", build(d(), both(), g())),
        ("both in the layer call", build_layered(d(), xy(), Aes::new().fill("f").color("c"), g())),
        (
            "both in the layer, x in the plot",
            build_layered(d(), Aes::new().x("x").y("y"), Aes::new().color("c").fill("f"), g()),
        ),
    ];
    for (name, r) in cases {
        assert!(matches!(r, Err(Error::Rendering(m)) if m.contains("same aes() call")), "{name}");
    }
    // the same column twice has no order to get wrong
    assert!(same_col.is_ok(), "fill and colour on one column");
    // one in each call is ordered (layer first) and built
    assert!(build_layered(d(), xy().fill("f"), Aes::new().color("c"), g()).is_ok());
}

/// An infinite `x` is refused (R's `remove_missing(finite = TRUE)` drops the row).
#[test]
fn an_infinite_x_is_refused() {
    let mut d = frame();
    d.add_column_f32("inf", &[f32::INFINITY; 10]);
    assert!(refused(build(d, Aes::new().x("inf").y("y"), Geom::gg_boxplot())));
}

#[test]
fn ragged_and_unknown_columns_are_typed_errors() {
    let mut ragged = frame();
    ragged.add_column_str("short", &["a", "b"]);
    assert!(refused(build(ragged, Aes::new().x("short").y("y"), Geom::gg_boxplot())));
    let unknown = build(frame(), Aes::new().x("nope").y("y"), Geom::gg_boxplot());
    assert!(matches!(unknown, Err(Error::UnknownColumn(_))));
}

/// The renderers refuse the layer with `NOT_DRAWN` rather than draw nothing.
#[test]
fn the_renderers_refuse_a_boxplot() {
    let p = GGPlot::new()
        .data(frame())
        .aes(Aes::new().x("x").y("y"))
        .layer(Layer::new(Geom::gg_boxplot()))
        .build()
        .expect("builds");
    let drawn = |r: crate::error::Result<()>| matches!(r, Err(Error::UnsupportedLayer(m)) if m == super::dist::NOT_DRAWN);
    assert!(drawn(p.to_framebuffer().map(drop)), "framebuffer");
    assert!(drawn(p.to_svg().map(drop)), "svg");
}

/// The terminal target refuses the layer too (it draws point and path layers only; a text `x` is
/// refused earlier, as a position scale).
#[test]
fn the_terminal_refuses_a_boxplot() {
    let opts = TerminalPlot { width: 60, height: 20, color: false };
    for (name, aes) in [
        ("numeric x", Aes::new().x("n").y("y")),
        ("text x", Aes::new().x("x").y("y")),
        ("no x", Aes::new().y("y")),
    ] {
        let p = GGPlot::new()
            .data(frame())
            .aes(aes)
            .layer(Layer::new(Geom::gg_boxplot()))
            .build()
            .expect("builds");
        assert!(matches!(p.to_terminal(opts), Err(Error::Rendering(_))), "{name}");
    }
}
