//! ggplot2 conformance oracle for `geom_ribbon()` (PLOT-GG-001, paiml/aprender#6957, master
//! #6909): `GeomRibbon$setup_data` with the identity stat and position. Fixtures are
//! `fixtures/ggplot2/geom_ribbon/expected_<case>.json`, produced by ggplot2 4.0.3 under C collation
//! (`generate.R`); R is never run here. Tolerance and helpers are those of
//! `ggplot2_geom_area_oracle`.

use super::ggplot2_geom_area_oracle::{
    build, check_axis, check_layer, csv, fixture, frame_from_csv, report_max_err, reset_max_err,
    want, wrap_f,
};
use super::{Aes, DataFrame, Facet, Geom, Layer};
use crate::error::Error;

const CASES: usize = 8;

fn band() -> Aes {
    Aes::new().x("x").ymin("lo").ymax("hi")
}

fn band_g() -> Aes {
    band().group("g")
}

fn run(case: &str, data: DataFrame, aes: Aes, facet: Facet, json: &str) {
    reset_max_err();
    let w = want(json, "geom_ribbon", case);
    let got = build(data, aes, vec![Layer::new(Geom::ribbon())], facet)
        .unwrap_or_else(|e| panic!("{case}: {e:?}"));
    assert_eq!(got.layers.len(), 1, "{case}: layer count");
    check_layer(case, &got.layers[0], &w["layers"][0], "GeomRibbon");
    check_axis("x", &got.x, &w["panel"]["x"]);
    check_axis("y", &got.y, &w["panel"]["y"]);
    report_max_err("geom_ribbon", case);
}

#[test]
fn a_plain_band_keeps_its_rows() {
    run(
        "plain",
        csv!("geom_ribbon", "data"),
        band(),
        Facet::None,
        fixture!("geom_ribbon", "plain"),
    );
}
#[test]
fn rows_are_ordered_by_x() {
    run(
        "unsorted",
        csv!("geom_ribbon", "data_unsorted"),
        band(),
        Facet::None,
        fixture!("geom_ribbon", "unsorted"),
    );
}
#[test]
fn missing_values_are_kept_with_x_last() {
    run("na", csv!("geom_ribbon", "data_na"), band(), Facet::None, fixture!("geom_ribbon", "na"));
}
#[test]
fn rows_are_ordered_by_group_then_x() {
    run(
        "groups",
        csv!("geom_ribbon", "data_groups"),
        band_g(),
        Facet::None,
        fixture!("geom_ribbon", "groups"),
    );
}
#[test]
fn rows_are_ordered_inside_each_panel() {
    run(
        "facet",
        csv!("geom_ribbon", "data_facet"),
        band_g(),
        wrap_f(),
        fixture!("geom_ribbon", "facet"),
    );
}
#[test]
fn an_inverted_band_is_not_swapped() {
    run(
        "inverted",
        csv!("geom_ribbon", "data_inverted"),
        band(),
        Facet::None,
        fixture!("geom_ribbon", "inverted"),
    );
}
#[test]
fn a_negative_band_is_not_clamped() {
    run(
        "negative",
        csv!("geom_ribbon", "data_negative"),
        band(),
        Facet::None,
        fixture!("geom_ribbon", "negative"),
    );
}

#[test]
fn a_ribbon_beside_a_point_layer_with_its_own_data() {
    reset_max_err();
    let w = want(fixture!("geom_ribbon", "layer_data"), "geom_ribbon", "layer_data");
    let points = csv!("geom_ribbon", "data_point");
    let band_data = csv!("geom_ribbon", "data_layer");
    let got = build(
        DataFrame::new(),
        Aes::new(),
        vec![
            Layer::new(Geom::point()).data(points).aes(Aes::new().x("x").y("y")),
            Layer::new(Geom::ribbon()).data(band_data).aes(band()),
        ],
        Facet::None,
    )
    .expect("a ribbon and a point layer share a plot");
    assert_eq!(got.layers.len(), 2);
    assert!(got.layers[0].filled.is_none());
    check_layer("layer_data", &got.layers[1], &w["layers"][1], "GeomRibbon");
    check_axis("x", &got.x, &w["panel"]["x"]);
    check_axis("y", &got.y, &w["panel"]["y"]);
    report_max_err("geom_ribbon", "layer_data");
}

#[test]
fn every_fixture_case_has_a_test() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/ggplot2/geom_ribbon");
    let n = std::fs::read_dir(dir)
        .expect("fixture dir")
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().starts_with("expected_"))
        .count();
    assert_eq!(n, CASES, "a fixture was added without a test (or the other way round)");
}

// --- refusals: out-of-scope input is a typed error, never a plausible wrong answer ---------------

fn tiny() -> DataFrame {
    csv!("geom_ribbon", "data")
}

fn lay(aes: Aes, geom: Geom) -> crate::error::Result<super::BuiltLayout> {
    build(tiny(), aes, vec![Layer::new(geom)], Facet::None)
}

#[test]
fn a_missing_edge_mapping_is_refused() {
    let r = lay(Aes::new().x("x").ymin("lo"), Geom::ribbon());
    assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
    let r = lay(Aes::new().x("x").ymax("hi"), Geom::ribbon());
    assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
}

#[test]
fn a_data_mapped_fill_is_refused() {
    let r = lay(band().fill("g"), Geom::ribbon());
    assert!(matches!(&r, Err(Error::Rendering(m)) if m.contains("geom_ribbon")), "{r:?}");
}

#[test]
fn a_missing_edge_column_is_refused() {
    let r = lay(Aes::new().x("x").ymin("nope").ymax("hi"), Geom::ribbon());
    assert!(matches!(r, Err(Error::UnknownColumn(_))), "{r:?}");
}

#[test]
fn a_mapped_y_is_ignored_as_ggplot2_does() {
    let with_y = lay(band().y("nope"), Geom::ribbon()).expect("y is not read");
    let without = lay(band(), Geom::ribbon()).expect("plain band");
    assert_eq!(with_y.layers[0], without.layers[0]);
    assert_eq!(with_y.y, without.y);
}

#[test]
fn a_ribbon_beside_a_bar_or_a_line_is_refused() {
    for companion in [Geom::bar(), Geom::line()] {
        let r = build(
            tiny(),
            band().y("lo"),
            vec![Layer::new(Geom::ribbon()), Layer::new(companion.clone())],
            Facet::None,
        );
        assert!(matches!(r, Err(Error::UnsupportedLayer(_))), "{:?}: {r:?}", companion.geom_type);
    }
}

#[test]
fn a_single_position_has_no_degenerate_scale_answer() {
    let r = build(
        csv!("geom_ribbon", "data_single_row"),
        band(),
        vec![Layer::new(Geom::ribbon())],
        Facet::None,
    );
    assert!(matches!(r, Err(Error::ScaleDomain(_))), "{r:?}");
}

/// A `geom_point` beside a ribbon gets the companion refusals: a missing column is a typed error,
/// never an `Ok` with an empty point layer.
#[test]
fn a_bad_point_companion_beside_a_ribbon_is_refused() {
    let r = build(
        tiny(),
        band(),
        vec![
            Layer::new(Geom::ribbon()),
            Layer::new(Geom::point()).aes(Aes::new().x("x").y("nope")),
        ],
        Facet::None,
    );
    assert!(matches!(r, Err(Error::UnknownColumn(_))), "{r:?}");
}
