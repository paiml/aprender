//! ggplot2 conformance oracle for `geom_text` (PLOT-GG-001, paiml/aprender#6909). Fixtures are
//! `fixtures/ggplot2/geom_text/expected_<case>.json`, produced by ggplot2 4.0.3 under C collation
//! (`generate.R`); R is never run here. Tolerance and helpers: `ggplot2_pack_common`.

use serde_json::Value;

use super::coord::Coord;
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

/// FALSIFY-GG18-006: the path-family guard alone refuses text beside a path layer.
#[test]
fn the_path_guard_refuses_text_beside_a_path_layer() {
    let layers = vec![Layer::new(Geom::text()), Layer::new(Geom::line())];
    let res = super::path::require_identity_stats(&layers);
    assert!(matches!(res, Err(Error::UnsupportedLayer(_))), "got {res:?}");
}

/// FALSIFY-GG18-007: the marks guard alone refuses a path layer beside text.
#[test]
fn the_marks_guard_refuses_a_path_layer_beside_text() {
    let layers = vec![Layer::new(Geom::text()), Layer::new(Geom::line())];
    let data = csv!("data");
    let res = super::marks::require_modelled_companions(&layers, &lab(), &data);
    assert!(matches!(res, Err(Error::UnsupportedLayer(_))), "got {res:?}");
}

#[test]
fn text_beside_a_path_layer_is_refused_end_to_end() {
    let res = build(
        csv!("data"),
        lab(),
        vec![Layer::new(Geom::text()), Layer::new(Geom::line())],
        Facet::None,
    );
    assert!(matches!(res, Err(Error::UnsupportedLayer(_))), "got {res:?}");
}

fn coords() -> Vec<(&'static str, Coord, &'static str)> {
    vec![
        ("flip", Coord::cartesian().flip(), "coord_flip"),
        ("xlim", Coord::cartesian().xlim(0.0, 10.0), "limits"),
        ("ylim", Coord::cartesian().ylim(0.0, 10.0), "limits"),
        ("xlim+flip", Coord::cartesian().xlim(0.0, 10.0).flip(), "coord_flip"),
        ("polar", Coord::polar(), "coord_polar"),
        ("fixed", Coord::fixed(1.0), "coord_fixed"),
    ]
}

fn with_coord(layers: Vec<Layer>, coord: Coord) -> crate::error::Result<BuiltLayout> {
    let mut p = super::GGPlot::new().data(csv!("data")).aes(lab());
    for l in layers {
        p = p.layer(l);
    }
    p.coord(coord).build()?.layout()
}

/// FALSIFY-GG18-008: a text layer under any non-default coord is refused, alone and beside a
/// `geom_point`, with the variant and the coord named.
#[test]
fn every_non_default_coord_is_refused_for_text() {
    let routes: Vec<(&str, Vec<Layer>)> = vec![
        ("text alone", vec![Layer::new(Geom::text())]),
        ("text + point", vec![Layer::new(Geom::point()), Layer::new(Geom::text())]),
        ("point + text", vec![Layer::new(Geom::text()), Layer::new(Geom::point())]),
    ];
    for (route, layers) in routes {
        for (name, coord, want_msg) in coords() {
            match with_coord(layers.clone(), coord) {
                Err(Error::UnsupportedCoord(m)) => {
                    assert!(m.contains(want_msg), "{route} under {name}: message {m:?}");
                    assert!(m.contains("text"), "{route} under {name}: message {m:?}");
                }
                other => panic!("{route} under {name}: expected UnsupportedCoord, got {other:?}"),
            }
        }
    }
}

#[test]
fn the_default_coord_is_accepted_for_text() {
    let r = with_coord(vec![Layer::new(Geom::text())], Coord::cartesian());
    assert!(r.is_ok(), "got {r:?}");
}

#[test]
fn non_finite_text_params_are_refused() {
    for g in [
        Geom::text().nudge_x(f32::INFINITY),
        Geom::text().nudge_y(f32::NAN),
        Geom::text().hjust(f32::INFINITY),
        Geom::text().vjust(f32::NEG_INFINITY),
        Geom::text().angle(f32::NAN),
    ] {
        let res = build(csv!("data"), lab(), vec![Layer::new(g)], Facet::None);
        assert!(matches!(res, Err(Error::Rendering(_))), "got {res:?}");
    }
}

/// Stated limit (row 18 contract): the layer positions are f64 sums, the trained limits are those
/// sums rounded once to f32. This pins that limit exactly; it does not claim 1e-9 agreement with R
/// for a non-f32-representable `x + nudge`.
#[test]
fn a_non_dyadic_nudge_limits_are_the_f64_sum_rounded_once_to_f32() {
    let g = Geom::text().nudge_x(0.1).nudge_y(0.3);
    let got = build(csv!("data"), lab(), vec![Layer::new(g)], Facet::None).expect("resolves");
    let xs = &got.layers[0].x;
    let (lo, hi) = (
        xs.iter().copied().fold(f64::INFINITY, f64::min),
        xs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
    assert_eq!(got.x.limits, (f64::from(lo as f32), f64::from(hi as f32)));
}

#[test]
fn a_numeric_label_column_is_refused() {
    let mut d = DataFrame::new();
    d.add_column_f32("x", &[1.0, 2.0]);
    d.add_column_f32("y", &[1.0, 3.0]);
    d.add_column_f32("lab", &[0.1, 1e5]);
    let res = build(d, lab(), vec![Layer::new(Geom::text())], Facet::None);
    assert!(matches!(res, Err(Error::Rendering(_))), "got {res:?}");
}

#[test]
fn a_point_beside_text_with_a_missing_column_is_refused() {
    let res = build(
        csv!("data"),
        lab(),
        vec![Layer::new(Geom::text()), Layer::new(Geom::point()).aes(Aes::new().x("nope").y("y"))],
        Facet::None,
    );
    assert!(res.is_err(), "got {res:?}");
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

/// A `geom_point` beside a path-family layer is first routed by `uses_position_scales`: a text
/// position column, or a `scale_x`/`scale_y` spec, selects the scaled layout, whose scope check
/// refuses the non-point layer with `UnsupportedLayer` (rows 2 and 4), as on `main`. The path
/// companion guard must not run first and change that variant.
#[test]
fn a_point_routed_to_the_scaled_layout_keeps_the_scoped_refusal_beside_a_line() {
    let mut text_x = DataFrame::new();
    text_x.add_column_str("x", &["a", "b", "c"]);
    text_x.add_column_f32("y", &[1.0, 2.0, 3.0]);
    let layers = || vec![Layer::new(Geom::point()), Layer::new(Geom::line())];
    let res = build(text_x, xy(), layers(), Facet::None);
    assert!(matches!(res, Err(Error::UnsupportedLayer(_))), "text column: {res:?}");

    let mut df = DataFrame::new();
    df.add_column_f32("x", &[1.0, 2.0, 3.0]);
    df.add_column_f32("y", &[1.0, 2.0, 3.0]);
    for missing in ["nope", "x"] {
        let mut p = super::GGPlot::new().data(df.clone()).aes(xy()).scale_x(super::scale_log10());
        p = p.layer(Layer::new(Geom::line()));
        let point = Aes::new().x(missing).y(if missing == "x" { "short" } else { "y" });
        p = p.layer(Layer::new(Geom::point()).aes(point));
        let res = p.build().and_then(|b| b.layout());
        assert!(matches!(res, Err(Error::UnsupportedLayer(_))), "{missing}: {res:?}");
    }
}

/// ggplot2 4.0.3 keeps a `geom_point` with an infinite position (the row stays in `layer_data`, it
/// is clamped to the panel edge and the scale ignores it); `layout()` does not model that, so an
/// infinite value beside a segment or polygon is refused rather than silently dropped.
#[test]
fn an_infinite_point_position_beside_segment_or_polygon_is_refused() {
    for other in [Geom::segment(), Geom::polygon()] {
        let mut inf = DataFrame::new();
        inf.add_column_f32("px", &[1.0, f32::INFINITY, 3.0]);
        inf.add_column_f32("py", &[1.0, 2.0, 3.0]);
        let point = Layer::new(Geom::point()).data(inf).aes(Aes::new().x("px").y("py"));
        let mut base = DataFrame::new();
        for (name, v) in [("x", 1.0), ("y", 1.0), ("xend", 2.0), ("yend", 3.0)] {
            base.add_column_f32(name, &[v, v + 1.0, v + 2.0]);
        }
        let aes = xy().xend("xend").yend("yend");
        let res = build(base, aes, vec![Layer::new(other.clone()), point], Facet::None);
        assert!(matches!(res, Err(Error::Rendering(_))), "{:?}: {res:?}", other.geom_type);
    }
}

/// Without a segment or polygon layer an infinite `geom_point` position behaves as on `main`: the
/// row is dropped and does not train the x scale, alone or beside a line, as when it is absent.
#[test]
fn an_infinite_point_without_segment_or_polygon_behaves_as_on_main() {
    let frame = |xs: &[f32]| {
        let mut df = DataFrame::new();
        df.add_column_f32("x", xs);
        let ys: Vec<f32> = xs.iter().map(|v| if v.is_finite() { *v + 1.0 } else { 2.0 }).collect();
        df.add_column_f32("y", &ys);
        df
    };
    let x_range = |res: crate::error::Result<BuiltLayout>| match res {
        Ok(l) => l.x.continuous_range,
        Err(e) => panic!("expected Ok, got {e:?}"),
    };
    let with_inf = frame(&[1.0, f32::INFINITY, 3.0, 4.0]);
    let without = frame(&[1.0, 3.0, 4.0]);
    let alone = |df: DataFrame| build(df, xy(), vec![Layer::new(Geom::point())], Facet::None);
    assert_eq!(x_range(alone(with_inf.clone())), x_range(alone(without.clone())));
    let beside = |df: DataFrame| {
        let line = frame(&[1.0, 3.0]);
        let point = Layer::new(Geom::point()).data(df).aes(xy());
        build(line, xy(), vec![Layer::new(Geom::line()), point], Facet::None)
    };
    assert_eq!(x_range(beside(with_inf)), x_range(beside(without)));
}
