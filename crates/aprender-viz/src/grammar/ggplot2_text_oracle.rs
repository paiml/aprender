//! ggplot2 conformance oracle for `geom_text` (PLOT-GG-001, paiml/aprender#6909). Fixtures are
//! `fixtures/ggplot2/geom_text/expected_<case>.json`, produced by ggplot2 4.0.3 under C collation
//! (`generate.R`); R is never run here. Tolerance and helpers: `ggplot2_pack_common`.

use serde_json::Value;

use super::ggplot2_pack_common::{
    build, check_axis, check_common, close, frame_from_csv, nums, strs, want, wrap_f, xy,
};
use super::stat::Stat;
use super::{Aes, BuiltLayout, DataFrame, Facet, Geom, Layer};
use crate::color::Rgba;
use crate::error::Error;

const ROW: &str = "geom_text";

macro_rules! fixture {
    ($case:literal) => {
        include_str!(concat!("../../fixtures/ggplot2/geom_text/expected_", $case, ".json"))
    };
}

macro_rules! csv {
    ($file:literal) => {
        frame_from_csv(include_str!(concat!("../../fixtures/ggplot2/geom_text/", $file, ".csv")))
    };
}

fn lab() -> Aes {
    xy().label("lab")
}

fn check(got: &BuiltLayout, w: &Value) {
    let layers = w["layers"].as_array().expect("layers");
    assert_eq!(got.layers.len(), 1);
    let (l, wl) = (&got.layers[0], &layers[0]);
    assert_eq!(wl["geom"], "GeomText", "R geom class");
    assert!(l.path.is_none() && l.points.is_none(), "a text layer is neither path nor points");
    let t = l.text.as_ref().expect("a text layer resolves text data");
    let c = t.colour;
    check_common("text", &t.panel, &t.group, &l.x, &l.y, t.alpha, [c.r, c.g, c.b, c.a], wl);
    let labels: Vec<String> = t.label.iter().map(|s| s.clone().unwrap_or_default()).collect();
    assert_eq!(labels, strs(&wl["label"]), "label");
    for i in 0..t.panel.len() {
        for (name, got_v) in [
            ("nudge_x", t.nudge_x),
            ("nudge_y", t.nudge_y),
            ("size", t.size),
            ("angle", t.angle),
            ("hjust", t.hjust),
            ("vjust", t.vjust),
        ] {
            assert!(close(got_v, nums(&wl[name])[i]), "{name}[{i}]: got {got_v}");
        }
    }
    check_axis("x", &got.x, &w["panel"]["x"]);
    check_axis("y", &got.y, &w["panel"]["y"]);
}

fn run(case: &str, json: &str, data: DataFrame, aes: Aes, facet: Facet, g: Geom) {
    let w = want(json, ROW, case);
    let got = build(data, aes, vec![Layer::new(g)], facet).expect("layout resolves");
    check(&got, &w);
}

#[test]
fn plain() {
    run("plain", fixture!("plain"), csv!("data"), lab(), Facet::None, Geom::text());
}

/// `position_nudge` shifts the positions and the scales retrain on the shifted values.
#[test]
fn nudge_shifts_positions_and_retrains_the_scales() {
    let g = Geom::text().nudge_x(0.5).nudge_y(-1.0);
    run("nudge", fixture!("nudge"), csv!("data"), lab(), Facet::None, g);
}

#[test]
fn nudge_x_alone() {
    run(
        "nudge_x_only",
        fixture!("nudge_x_only"),
        csv!("data"),
        lab(),
        Facet::None,
        Geom::text().nudge_x(0.25),
    );
}

#[test]
fn hjust_vjust_and_angle_pass_through() {
    let g = Geom::text().hjust(0.0).vjust(1.0).angle(45.0);
    run("hjust_vjust_angle", fixture!("hjust_vjust_angle"), csv!("data"), lab(), Facet::None, g);
}

#[test]
fn fixed_size_colour_and_alpha() {
    let g = Geom::text().aes(Aes::new().size_value(5.0).color_value(Rgba::BLUE).alpha_value(0.5));
    run("size_colour_alpha", fixture!("size_colour_alpha"), csv!("data"), lab(), Facet::None, g);
}

#[test]
fn facet_wrap_with_nudge() {
    run("facet", fixture!("facet"), csv!("data"), lab(), wrap_f(), Geom::text().nudge_y(0.5));
}

#[test]
fn numeric_group_ids_rank_sorted_values() {
    run(
        "group_num",
        fixture!("group_num"),
        csv!("data"),
        lab().group("gn"),
        Facet::None,
        Geom::text(),
    );
}

#[test]
fn text_group_ids_rank_sorted_levels() {
    run(
        "group_text",
        fixture!("group_text"),
        csv!("data"),
        lab().group("g"),
        Facet::None,
        Geom::text(),
    );
}

#[test]
fn negative_positions_with_negative_nudge() {
    let g = Geom::text().nudge_x(-0.5).nudge_y(0.25);
    run("negative", fixture!("negative"), csv!("data_negative"), lab(), Facet::None, g);
}

#[test]
fn na_rows_are_kept_and_not_trained() {
    run("na", fixture!("na"), csv!("data_na"), lab(), Facet::None, Geom::text());
}

#[test]
fn a_text_layer_without_a_label_is_refused() {
    let res = build(csv!("data"), xy(), vec![Layer::new(Geom::text())], Facet::None);
    assert!(matches!(res, Err(Error::Rendering(_))), "got {res:?}");
}

#[test]
fn a_missing_label_column_is_an_unknown_column() {
    let res = build(csv!("data"), xy().label("nope"), vec![Layer::new(Geom::text())], Facet::None);
    assert!(matches!(res, Err(Error::UnknownColumn(_))), "got {res:?}");
}

#[test]
fn a_data_mapped_colour_is_refused() {
    let res = build(csv!("data"), lab().color("g"), vec![Layer::new(Geom::text())], Facet::None);
    assert!(matches!(res, Err(Error::Rendering(_))), "got {res:?}");
}

#[test]
fn a_degenerate_scale_is_refused() {
    let mut d = DataFrame::new();
    d.add_column_f32("x", &[2.0]);
    d.add_column_f32("y", &[1.0]);
    d.add_column_str("lab", &["a"]);
    let res = build(d, lab(), vec![Layer::new(Geom::text())], Facet::None);
    assert!(matches!(res, Err(Error::ScaleDomain(_))), "got {res:?}");
}

#[test]
fn text_beside_a_path_layer_is_still_refused() {
    let res = build(
        csv!("data"),
        lab(),
        vec![Layer::new(Geom::text()), Layer::new(Geom::line())],
        Facet::None,
    );
    assert!(matches!(res, Err(Error::UnsupportedLayer(_))), "got {res:?}");
}

#[test]
fn text_beside_an_unresolved_mark_is_refused() {
    for other in [Geom::area(), Geom::tile()] {
        let res = build(
            csv!("data"),
            lab(),
            vec![Layer::new(Geom::text()), Layer::new(other)],
            Facet::None,
        );
        assert!(matches!(res, Err(Error::UnsupportedLayer(_))), "got {res:?}");
    }
}

#[test]
fn text_beside_a_point_layer_resolves() {
    let res = build(
        csv!("data"),
        lab(),
        vec![Layer::new(Geom::point()), Layer::new(Geom::text())],
        Facet::None,
    )
    .expect("layout resolves");
    assert!(res.layers[0].points.is_some() && res.layers[1].text.is_some());
}

#[test]
fn a_non_identity_stat_is_refused() {
    let g = Geom::text().stat(Stat::Count);
    let res = build(csv!("data"), lab(), vec![Layer::new(g)], Facet::None);
    assert!(res.is_err(), "a stat on geom_text must not be answered with the raw data");
}

#[test]
fn text_params_on_another_geom_are_a_no_op() {
    let g = Geom::point().nudge_x(1.0).hjust(0.0);
    assert!(matches!(g.geom_type, super::geom::GeomType::Point { .. }));
}
