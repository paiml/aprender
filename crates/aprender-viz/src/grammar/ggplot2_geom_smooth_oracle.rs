//! ggplot2 conformance oracle for `geom_smooth(method = "lm")` (PLOT-GG-001, paiml/aprender#6983,
//! master #6909): `stat_smooth` with `formula = y ~ x`. Fixtures are
//! `fixtures/ggplot2/geom_smooth/expected_<case>.json` and `qt_table.json`, produced by ggplot2
//! 4.0.3 under R 4.6.2 with C collation (`generate.R`); R is never run here.
//!
//! Tolerance and helpers are those of `ggplot2_geom_area_oracle`: `|a - b| <= 1e-9 * max(|a|, |b|)
//! + 1e-12` for every position, limit, range, break, fitted value, band edge and standard error;
//! labels, groups, flags and counts compare exactly. `NaN` matches `null` (R's `NA`) by position.
//! Every case is built through the public builder and read back with `layout()`.

use serde_json::Value;

use super::geom::SmoothMethod;
use super::ggplot2_geom_area_oracle::{
    assert_close_slice, build, check_axis, csv, fixture, frame_from_csv, ints, nums,
    report_max_err, reset_max_err, want,
};
use super::stat::Stat;
use super::t_dist::qt_upper;
use super::{Aes, BuiltLayer, BuiltLayout, Coord, DataFrame, Facet, GGPlot, Geom, Layer};
use crate::error::Error;

const CASES: usize = 16;

fn lm() -> Geom {
    Geom::smooth().method(SmoothMethod::Linear)
}

fn xy() -> Aes {
    Aes::new().x("x").y("y")
}

/// One smooth layer against R's `ggplot_build()` data: every column the stat produced, the group
/// ids and `flipped_aes`. A column R lacks (`se = FALSE`) must be absent from ours, and the other
/// way round.
fn check_layer(tag: &str, got: &BuiltLayer, w: &Value) {
    assert_eq!(w["geom"], "GeomSmooth", "{tag}: R geom class");
    let s = got.smooth.as_ref().expect("a smooth layer resolves smooth data");
    assert_eq!(s.group, ints(&w["group"]), "{tag}.group");
    assert_eq!(got.x.len() as u64, w["n_rows"].as_u64().expect("row count"), "{tag}.n_rows");
    assert_close_slice(&format!("{tag}.x"), &got.x, &nums(&w["x"]));
    assert_close_slice(&format!("{tag}.y"), &got.y, &nums(&w["y"]));
    for (name, col) in [("ymin", &s.ymin), ("ymax", &s.ymax), ("se", &s.se)] {
        assert_eq!(col.is_some(), w.get(name).is_some(), "{tag}.{name} presence");
        if let (Some(g), Some(r)) = (col, w.get(name)) {
            assert_close_slice(&format!("{tag}.{name}"), g, &nums(r));
        }
    }
    assert_eq!(i64::from(s.flipped_aes), ints(&w["flipped_aes"])[0], "{tag}.flipped_aes");
}

fn run_with(case: &str, data: DataFrame, aes: Aes, geom: Geom, json: &str) {
    reset_max_err();
    let w = want(json, "geom_smooth", case);
    let got = build(data, aes, vec![Layer::new(geom)], Facet::None)
        .unwrap_or_else(|e| panic!("{case}: {e:?}"));
    assert_eq!(got.layers.len(), 1, "{case}: layer count");
    check_layer(case, &got.layers[0], &w["layers"][0]);
    check_axis("x", &got.x, &w["panel"]["x"]);
    check_axis("y", &got.y, &w["panel"]["y"]);
    report_max_err("geom_smooth", case);
}

fn run(case: &str, data: DataFrame, json: &str) {
    run_with(case, data, xy(), lm(), json);
}

#[test]
fn a_clean_line_has_no_band() {
    run("clean", csv!("geom_smooth", "data_clean"), fixture!("geom_smooth", "clean"));
}
#[test]
fn noisy_data_gets_a_fitted_band() {
    run("noisy", csv!("geom_smooth", "data_noisy"), fixture!("geom_smooth", "noisy"));
}
#[test]
fn two_points_have_no_residual_degrees_of_freedom() {
    run("n2", csv!("geom_smooth", "data_n2"), fixture!("geom_smooth", "n2"));
}
#[test]
fn three_points_have_one_residual_degree_of_freedom() {
    run("n3", csv!("geom_smooth", "data_n3"), fixture!("geom_smooth", "n3"));
}
#[test]
fn a_constant_y_gives_a_zero_width_band_and_the_zero_range_scale() {
    run(
        "constant_y",
        csv!("geom_smooth", "data_constant_y"),
        fixture!("geom_smooth", "constant_y"),
    );
}
#[test]
fn a_negative_slope_is_fitted() {
    run("negative", csv!("geom_smooth", "data_negative"), fixture!("geom_smooth", "negative"));
}
#[test]
fn x_near_a_million_keeps_its_digits() {
    run("offset", csv!("geom_smooth", "data_offset"), fixture!("geom_smooth", "offset"));
}
#[test]
fn missing_and_infinite_rows_are_dropped_before_the_fit() {
    run("na", csv!("geom_smooth", "data_na"), fixture!("geom_smooth", "na"));
}
#[test]
fn se_false_leaves_x_and_y_only() {
    run_with(
        "se_false",
        csv!("geom_smooth", "data_noisy"),
        xy(),
        lm().se(false),
        fixture!("geom_smooth", "se_false"),
    );
}
#[test]
fn a_non_default_level_widens_the_band() {
    run_with(
        "level90",
        csv!("geom_smooth", "data_noisy"),
        xy(),
        lm().level(0.9),
        fixture!("geom_smooth", "level90"),
    );
}
#[test]
fn a_non_default_n_changes_the_grid() {
    run_with(
        "n5",
        csv!("geom_smooth", "data_noisy"),
        xy(),
        lm().n(5),
        fixture!("geom_smooth", "n5"),
    );
}
#[test]
fn each_group_is_fitted_over_its_own_range() {
    run_with(
        "groups",
        csv!("geom_smooth", "data_groups"),
        xy().group("g"),
        lm(),
        fixture!("geom_smooth", "groups"),
    );
}
#[test]
fn a_group_with_one_distinct_x_vanishes() {
    run_with(
        "group_vanishes",
        csv!("geom_smooth", "data_groups_vanish"),
        xy().group("g"),
        lm(),
        fixture!("geom_smooth", "group_vanishes"),
    );
}

#[test]
fn a_smooth_beside_point_layers_shares_their_scales() {
    reset_max_err();
    let w = want(fixture!("geom_smooth", "with_points"), "geom_smooth", "with_points");
    let got = build(
        csv!("geom_smooth", "data_noisy"),
        xy(),
        vec![
            Layer::new(Geom::point()),
            Layer::new(lm()),
            Layer::new(Geom::point()).data(csv!("geom_smooth", "data_point")),
        ],
        Facet::None,
    )
    .expect("a smooth and point layers share a plot");
    assert_eq!(got.layers.len(), 3);
    assert!(got.layers[0].smooth.is_none() && got.layers[2].smooth.is_none());
    check_layer("with_points", &got.layers[1], &w["layers"][1]);
    check_axis("x", &got.x, &w["panel"]["x"]);
    check_axis("y", &got.y, &w["panel"]["y"]);
    report_max_err("geom_smooth", "with_points");
}

/// The smooth layer's own raw rows must not train the scales (ggplot2 retrains them on the stat's
/// output): the points' data is narrow and the smooth layer's raw y is far wider than its band.
#[test]
fn a_smooth_layers_raw_rows_do_not_train_the_scales() {
    reset_max_err();
    let w = want(fixture!("geom_smooth", "own_data"), "geom_smooth", "own_data");
    let got = build(
        csv!("geom_smooth", "data_point"),
        xy(),
        vec![Layer::new(Geom::point()), Layer::new(lm()).data(csv!("geom_smooth", "data_wide"))],
        Facet::None,
    )
    .expect("a smooth with its own data beside a point layer");
    check_layer("own_data", &got.layers[1], &w["layers"][1]);
    check_axis("x", &got.x, &w["panel"]["x"]);
    check_axis("y", &got.y, &w["panel"]["y"]);
    report_max_err("geom_smooth", "own_data");
}

/// R returns an empty layer when no group has two distinct x; the engine refuses with a typed error
/// rather than an `Ok` with no rows. The fixture proves R's side of that statement.
#[test]
fn a_layer_with_no_fittable_group_is_refused_where_r_returns_it_empty() {
    let w = want(fixture!("geom_smooth", "constant_x"), "geom_smooth", "constant_x");
    assert_eq!(w["layers"][0]["geom"], "GeomSmooth");
    assert_eq!(w["layers"][0]["n_rows"], 0, "R's layer is empty");
    let r =
        build(csv!("geom_smooth", "data_constant_x"), xy(), vec![Layer::new(lm())], Facet::None);
    assert!(matches!(&r, Err(Error::Rendering(m)) if m.contains("empty")), "{r:?}");
}

#[test]
fn every_fixture_case_has_a_test() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/ggplot2/geom_smooth");
    let n = std::fs::read_dir(dir)
        .expect("fixture dir")
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().starts_with("expected_"))
        .count();
    assert_eq!(n, CASES, "a fixture was added without a test (or the other way round)");
}

// --- the Student-t quantile, against R's qt() -----------------------------------------------------

#[test]
fn the_t_quantile_matches_r_for_every_df_and_level() {
    let w: Value =
        serde_json::from_str(include_str!("../../fixtures/ggplot2/geom_smooth/qt_table.json"))
            .expect("fixture is valid JSON");
    assert_eq!(w["versions"]["ggplot2"], "4.0.3");
    let (qs, dfs, ts) = (nums(&w["q"]), nums(&w["df"]), nums(&w["t"]));
    assert_eq!(qs.len(), 64);
    for df in [1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 78.0, 1000.0] {
        assert!(dfs.contains(&df), "df {df} missing from the table");
    }
    let got: Vec<f64> = qs.iter().zip(&dfs).map(|(q, d)| qt_upper(*q, *d)).collect();
    assert_close_slice("qt", &got, &ts);
}

// --- refusals: out-of-scope input is a typed error, never a plausible wrong answer ---------------

fn noisy() -> DataFrame {
    csv!("geom_smooth", "data_noisy")
}

fn lay(aes: Aes, geom: Geom) -> crate::error::Result<BuiltLayout> {
    build(noisy(), aes, vec![Layer::new(geom)], Facet::None)
}

#[test]
fn a_bare_geom_smooth_is_refused_not_fitted_with_lm() {
    let r = lay(xy(), Geom::smooth());
    assert!(matches!(&r, Err(Error::UnsupportedStat(m)) if m.contains("lm")), "{r:?}");
}

#[test]
fn a_stat_other_than_smooth_is_refused() {
    let r = lay(xy(), lm().stat(Stat::Identity));
    assert!(matches!(r, Err(Error::UnsupportedStat(_))), "{r:?}");
}

#[test]
fn fullrange_is_refused() {
    let r = lay(xy(), lm().fullrange(true));
    assert!(matches!(&r, Err(Error::Rendering(m)) if m.contains("fullrange")), "{r:?}");
}

#[test]
fn a_level_outside_zero_one_is_refused() {
    for level in [0.0, 1.0, -0.5, 1.5, f64::NAN, f64::INFINITY] {
        let r = lay(xy(), lm().level(level));
        assert!(matches!(&r, Err(Error::Rendering(m)) if m.contains("level")), "{level}: {r:?}");
    }
}

#[test]
fn an_n_of_zero_or_past_the_cap_is_refused() {
    for n in [0, super::smooth::MAX_N + 1] {
        let r = lay(xy(), lm().n(n));
        assert!(matches!(&r, Err(Error::Rendering(m)) if m.contains("n ")), "{n}: {r:?}");
    }
}

#[test]
fn a_data_mapped_style_aesthetic_is_refused() {
    let mapped = [
        xy().color("x"),
        xy().fill("x"),
        xy().size("x"),
        xy().shape("x"),
        xy().alpha("x"),
        xy().label("x"),
        xy().ymin("x"),
        xy().ymax("x"),
    ];
    for aes in mapped {
        let r = lay(aes, lm());
        assert!(matches!(&r, Err(Error::Rendering(m)) if m.contains("geom_smooth")), "{r:?}");
    }
}

#[test]
fn a_facet_is_refused() {
    let mut d = noisy();
    d.add_column_str("f", &["u"; 12]);
    let r = build(d, xy(), vec![Layer::new(lm())], Facet::Wrap { var: "f".into(), ncol: 2 });
    assert!(matches!(&r, Err(Error::Rendering(m)) if m.contains("facet")), "{r:?}");
}

#[test]
fn a_missing_or_text_position_is_refused() {
    let r = lay(Aes::new().x("x").y("nope"), lm());
    assert!(matches!(r, Err(Error::UnknownColumn(_))), "{r:?}");
    let mut d = noisy();
    d.add_column_str("t", &["a"; 12]);
    let r = build(d, Aes::new().x("t").y("y"), vec![Layer::new(lm())], Facet::None);
    assert!(matches!(&r, Err(Error::Rendering(m)) if m.contains("text")), "{r:?}");
}

#[test]
fn a_single_evaluation_point_has_no_degenerate_x_answer() {
    let r = lay(xy(), lm().n(1));
    assert!(matches!(r, Err(Error::ScaleDomain(_))), "{r:?}");
}

/// A smooth layer ALONE under a coordinate system `layout()` does not apply: no other layer family
/// has to be present for the refusal to fire.
#[test]
fn every_non_default_coord_is_refused_for_a_smooth_alone() {
    let coords = [
        ("flip", Coord::cartesian().flip()),
        ("xlim", Coord::cartesian().xlim(0.0, 10.0)),
        ("ylim", Coord::cartesian().ylim(0.0, 10.0)),
        ("polar", Coord::polar()),
        ("fixed", Coord::fixed(1.0)),
    ];
    for (name, coord) in coords {
        let r = GGPlot::new()
            .data(noisy())
            .aes(xy())
            .layer(Layer::new(lm()))
            .coord(coord)
            .build()
            .and_then(|b| b.layout());
        assert!(matches!(r, Err(Error::UnsupportedCoord(_))), "{name}: {r:?}");
    }
}

#[test]
fn every_companion_but_point_is_refused() {
    for companion in [Geom::bar(), Geom::line(), Geom::ribbon(), Geom::area(), Geom::boxplot()] {
        let r = build(
            noisy(),
            xy(),
            vec![Layer::new(lm()), Layer::new(companion.clone())],
            Facet::None,
        );
        assert!(matches!(r, Err(Error::UnsupportedLayer(_))), "{:?}: {r:?}", companion.geom_type);
    }
}

#[test]
fn a_bad_point_companion_beside_a_smooth_is_refused() {
    let r = build(
        noisy(),
        xy(),
        vec![Layer::new(lm()), Layer::new(Geom::point()).aes(Aes::new().x("x").y("nope"))],
        Facet::None,
    );
    assert!(matches!(r, Err(Error::UnknownColumn(_))), "{r:?}");
    let r = build(
        noisy(),
        xy(),
        vec![Layer::new(lm()), Layer::new(Geom::point().stat(Stat::Count))],
        Facet::None,
    );
    assert!(matches!(r, Err(Error::UnsupportedStat(_))), "{r:?}");
}

#[test]
fn a_position_scale_spec_is_refused() {
    let r = GGPlot::new()
        .data(noisy())
        .aes(xy())
        .scale_x(super::scale_log10())
        .layer(Layer::new(lm()))
        .build()
        .and_then(|b| b.layout());
    assert!(matches!(r, Err(Error::UnsupportedLayer(_))), "{r:?}");
}

/// The renderer does not draw a smooth layer, so both outputs refuse it with the typed error the
/// other undrawn layers use, never a plot with the fit silently missing.
#[test]
fn drawing_a_smooth_layer_is_refused() {
    let p = GGPlot::new().data(noisy()).aes(xy()).layer(Layer::new(lm())).build().expect("builds");
    let svg = p.to_svg();
    assert!(
        matches!(&svg, Err(Error::UnsupportedLayer(m)) if *m == super::smooth::NOT_DRAWN),
        "{svg:?}"
    );
    let fb = p.to_framebuffer();
    assert!(
        matches!(&fb, Err(Error::UnsupportedLayer(m)) if *m == super::smooth::NOT_DRAWN),
        "{:?}",
        fb.map(|_| ())
    );
}

/// `frame_from_csv` is shared; keep the import used when a case is added without one.
#[test]
fn the_csv_helper_reads_infinite_cells() {
    let d = frame_from_csv("x,y\n1,Inf\n2,-Inf\n");
    assert_eq!(d.nrow(), 2);
}
