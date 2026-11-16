//! ggplot2 conformance oracle for `geom_segment` (PLOT-GG-001, paiml/aprender#6909). Fixtures are
//! `fixtures/ggplot2/geom_segment/expected_<case>.json`, produced by ggplot2 4.0.3 under C
//! collation (`generate.R`); R is never run here. Tolerance and helpers: `ggplot2_pack_common`.

use serde_json::Value;

use super::ggplot2_pack_common::{
    assert_close_slice, build, check_axis, check_common, frame_from_csv, nums, want, wrap_f, xy,
};
use super::{Aes, BuiltLayout, DataFrame, Facet, Geom, Layer};
use crate::color::Rgba;
use crate::error::Error;

const ROW: &str = "geom_segment";

macro_rules! fixture {
    ($case:literal) => {
        include_str!(concat!("../../fixtures/ggplot2/geom_segment/expected_", $case, ".json"))
    };
}

macro_rules! csv {
    ($file:literal) => {
        frame_from_csv(include_str!(concat!("../../fixtures/ggplot2/geom_segment/", $file, ".csv")))
    };
}

fn full() -> Aes {
    xy().xend("xend").yend("yend")
}

fn check(got: &BuiltLayout, w: &Value) {
    let layers = w["layers"].as_array().expect("layers");
    assert_eq!(got.layers.len(), 1);
    let (l, wl) = (&got.layers[0], &layers[0]);
    assert_eq!(wl["geom"], "GeomSegment", "R geom class");
    let p = l.path.as_ref().expect("a segment layer resolves path data");
    let c = p.aes.colour;
    check_common("segment", &p.panel, &p.group, &l.x, &l.y, p.aes.alpha, [c.r, c.g, c.b, c.a], wl);
    let n = p.panel.len();
    for i in 0..n {
        assert!(super::ggplot2_pack_common::close(p.aes.linewidth, nums(&wl["linewidth"])[i]));
        assert_eq!(f64::from(p.aes.linetype), nums(&wl["linetype"])[i], "linetype[{i}]");
    }
    for (name, got_end) in [("xend", &p.xend), ("yend", &p.yend)] {
        assert_eq!(got_end.is_some(), wl.get(name).is_some(), "{name} presence");
        if let Some(v) = got_end {
            assert_close_slice(name, v, &nums(&wl[name]));
        }
    }
    assert!(p.fill.is_none(), "a segment has no fill");
    check_axis("x", &got.x, &w["panel"]["x"]);
    check_axis("y", &got.y, &w["panel"]["y"]);
}

fn run(case: &str, json: &str, data: DataFrame, aes: Aes, facet: Facet) {
    run_geom(case, json, data, aes, facet, Geom::segment());
}

fn run_geom(case: &str, json: &str, data: DataFrame, aes: Aes, facet: Facet, g: Geom) {
    let w = want(json, ROW, case);
    let got = build(data, aes, vec![Layer::new(g)], facet).expect("layout resolves");
    check(&got, &w);
}

#[test]
fn plain() {
    run("plain", fixture!("plain"), csv!("data"), full(), Facet::None);
}

#[test]
fn reversed_ends_keep_row_order() {
    run("reversed", fixture!("reversed"), csv!("data_reversed"), full(), Facet::None);
}

#[test]
fn zero_length_segments_are_kept() {
    run("zero_length", fixture!("zero_length"), csv!("data_zero"), full(), Facet::None);
}

#[test]
fn negative_positions() {
    run("negative", fixture!("negative"), csv!("data_negative"), full(), Facet::None);
}

/// The scales train on `xend`/`yend` too: the far ends set the limits.
#[test]
fn ends_beyond_the_start_points_widen_the_scales() {
    run("ends_far", fixture!("ends_far"), csv!("data_ends_far"), full(), Facet::None);
}

#[test]
fn xend_alone_leaves_no_yend_column() {
    run("xend_only", fixture!("xend_only"), csv!("data"), xy().xend("xend"), Facet::None);
}

#[test]
fn yend_alone_leaves_no_xend_column() {
    run("yend_only", fixture!("yend_only"), csv!("data"), xy().yend("yend"), Facet::None);
}

#[test]
fn numeric_groups_in_facet_wrap() {
    run("group_facet", fixture!("group_facet"), csv!("data"), full().group("gn"), wrap_f());
}

#[test]
fn text_group_ids_rank_sorted_levels() {
    run("group_text", fixture!("group_text"), csv!("data"), full().group("g"), Facet::None);
}

#[test]
fn fixed_colour_and_alpha() {
    let g = Geom::segment().aes(Aes::new().color_value(Rgba::BLUE).alpha_value(0.25));
    run_geom("colour_alpha", fixture!("colour_alpha"), csv!("data"), full(), Facet::None, g);
}

#[test]
fn na_rows_are_kept_and_not_trained() {
    run("na", fixture!("na"), csv!("data_na"), full(), Facet::None);
}

#[test]
fn a_segment_without_xend_or_yend_is_refused() {
    let res = build(csv!("data"), xy(), vec![Layer::new(Geom::segment())], Facet::None);
    assert!(matches!(res, Err(Error::Rendering(_))), "got {res:?}");
}

#[test]
fn a_missing_end_column_is_an_unknown_column() {
    let res =
        build(csv!("data"), xy().xend("nope"), vec![Layer::new(Geom::segment())], Facet::None);
    assert!(matches!(res, Err(Error::UnknownColumn(_))), "got {res:?}");
}

#[test]
fn a_degenerate_scale_is_refused() {
    let mut d = DataFrame::new();
    d.add_column_f32("x", &[2.0, 2.0]);
    d.add_column_f32("y", &[1.0, 3.0]);
    d.add_column_f32("xend", &[2.0, 2.0]);
    d.add_column_f32("yend", &[2.0, 4.0]);
    let res = build(d, full(), vec![Layer::new(Geom::segment())], Facet::None);
    assert!(matches!(res, Err(Error::ScaleDomain(_))), "got {res:?}");
}

#[test]
fn only_point_companions_are_allowed() {
    let res = build(
        csv!("data"),
        full(),
        vec![Layer::new(Geom::segment()), Layer::new(Geom::bar())],
        Facet::None,
    );
    assert!(matches!(res, Err(Error::UnsupportedLayer(_))), "got {res:?}");
    let ok = build(
        csv!("data"),
        full(),
        vec![Layer::new(Geom::segment()), Layer::new(Geom::point())],
        Facet::None,
    );
    assert!(ok.is_ok(), "a point companion resolves: {ok:?}");
}

#[test]
fn empty_data_is_a_typed_error_not_a_panic() {
    let mut d = DataFrame::new();
    d.add_column_f32("x", &[]);
    d.add_column_f32("y", &[]);
    d.add_column_f32("xend", &[]);
    d.add_column_f32("yend", &[]);
    let res = build(d, full(), vec![Layer::new(Geom::segment())], Facet::None);
    assert!(matches!(res, Err(Error::ScaleDomain(_))), "got {res:?}");
}
