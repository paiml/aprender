//! ggplot2 conformance for the filled marks and text (PLOT-GG-001, paiml/aprender#6909):
//! `geom_area`, `geom_ribbon`, `geom_tile`, `geom_rect`, `geom_text`.
//!
//! This slice holds the refusals only: `layout()` refuses each of these geoms until its
//! `ggplot_build()` resolution lands, and refuses every companion layer it would not resolve
//! beside them. The fixture-compared cases arrive with each geom.

use super::{Aes, BuiltLayout, DataFrame, GGPlot, Geom};
use crate::error::Error;

// --- refusals: out-of-scope input is a typed error, never a plausible wrong answer ---------------

fn frame(cols: &[(&str, &[f32])]) -> DataFrame {
    let mut df = DataFrame::new();
    for (n, v) in cols {
        df.add_column_f32(n, v);
    }
    df
}

/// The marks geoms that have a constructor, with a mapping each would lay out from [`wide`].
fn three() -> Vec<(&'static str, Geom, Aes)> {
    vec![
        ("area", Geom::area(), Aes::new().x("x").y("y").group("g")),
        ("tile", Geom::tile(), Aes::new().x("x").y("y").group("g")),
        ("text", Geom::text(), Aes::new().x("x").y("y").label("lab").group("g")),
    ]
}

fn wide() -> DataFrame {
    let mut df =
        frame(&[("x", &[1.0, 2.0, 3.0]), ("y", &[2.0, 5.0, 3.0]), ("g", &[1.0, 1.0, 1.0])]);
    df.add_column_str("lab", &["p", "q", "r"]);
    df
}

/// What `layout()` does with a marks layer `name` once the companion guard has passed: refuses
/// `geom_area` and `geom_tile` as not resolved yet, and resolves `geom_text` (`text.rs`, proven
/// in `ggplot2_text_oracle.rs`).
fn unresolved(name: &str, r: &Result<BuiltLayout, Error>) -> bool {
    if name == "text" {
        return r.is_ok();
    }
    matches!(r, Err(Error::UnsupportedLayer(m))
        if *m == format!("geom_{name}: layout() does not resolve it yet"))
}

#[test]
fn refuses_area_and_tile_and_resolves_text() {
    for (name, geom, aes) in three() {
        let r = GGPlot::new().data(wide()).aes(aes).geom(geom).build().and_then(|b| b.layout());
        assert!(unresolved(name, &r), "{name}: {r:?}");
    }
}

#[test]
fn a_plot_without_marks_is_not_refused() {
    use super::Layer;
    // The companion guard is off without a marks layer: a bar beside a point lays out as before.
    let r = GGPlot::new()
        .data(wide())
        .layer(Layer::new(Geom::bar()).aes(Aes::new().x("x")))
        .layer(Layer::new(Geom::point()).aes(Aes::new().x("x").y("y")))
        .build()
        .and_then(|b| b.layout());
    assert!(r.is_ok(), "{r:?}");
}

/// `marks` and `companion` in both orders over [`wide`], laid out.
fn both_orders(
    geom: &Geom,
    aes: &Aes,
    companion: &super::Layer,
) -> [Result<BuiltLayout, Error>; 2] {
    use super::Layer;
    let marks = Layer::new(geom.clone()).aes(aes.clone());
    let lay = |a: &Layer, b: &Layer| {
        GGPlot::new()
            .data(wide())
            .layer(a.clone())
            .layer(b.clone())
            .build()
            .and_then(|b| b.layout())
    };
    [lay(&marks, companion), lay(companion, &marks)]
}

#[test]
fn refuses_every_companion_geom_layout_does_not_model_for_every_geom() {
    use super::Layer;
    let xy = Aes::new().x("x").y("y");
    // Each widens a shared scale in ggplot2 4.0.3 through its stat or setup (measured beside
    // geom_area: bar y to 0 and x to the bar edges, histogram x to the bin edges, boxplot x to the
    // box width, smooth y to its fit).
    let modelled_elsewhere = [
        ("geom_bar", Layer::new(Geom::bar()).aes(Aes::new().x("x"))),
        ("geom_histogram", Layer::new(Geom::histogram()).aes(Aes::new().x("x"))),
        ("geom_boxplot", Layer::new(Geom::boxplot()).aes(xy.clone())),
        ("geom_violin", Layer::new(Geom::violin()).aes(xy.clone())),
        ("geom_smooth", Layer::new(Geom::smooth()).aes(xy.clone())),
    ];
    for (name, geom, aes) in three() {
        for (companion, layer) in &modelled_elsewhere {
            for r in both_orders(&geom, &aes, layer) {
                match &r {
                    Err(Error::UnsupportedLayer(msg)) => {
                        assert!(msg.starts_with(companion), "{name} + {companion}: {msg}");
                    }
                    other => panic!("{name} + {companion}: {other:?}"),
                }
            }
        }
        // The path family and reference lines are refused too (by the path guards, first).
        for layer in [
            Layer::new(Geom::line()).aes(xy.clone()),
            Layer::new(Geom::path()).aes(xy.clone()),
            Layer::new(Geom::step()).aes(xy.clone()),
            Layer::new(Geom::hline(1.0)),
            Layer::new(Geom::vline(1.0)),
            Layer::new(Geom::abline(1.0, 0.0)),
        ] {
            for r in both_orders(&geom, &aes, &layer) {
                assert!(matches!(r, Err(Error::UnsupportedLayer(_))), "{name} + {layer:?}: {r:?}");
            }
        }
    }
}

#[test]
fn refuses_every_stat_on_a_point_beside_every_geom() {
    use super::{Layer, Stat};
    let stats = [
        (Stat::count(), "stat_count"),
        (Stat::bin(5), "stat_bin "),
        (Stat::density(), "stat_density"),
        (Stat::boxplot(), "stat_boxplot"),
        (Stat::smooth(), "stat_smooth"),
        (Stat::bin2d(2, 2), "stat_bin_2d"),
        (Stat::sum(), "stat_summary(fun = sum)"),
        (Stat::mean(), "stat_summary(fun = mean)"),
    ];
    let point = |stat: Option<Stat>| {
        let g = Geom::point();
        Layer::new(stat.map_or(g.clone(), |s| g.stat(s))).aes(Aes::new().x("x").y("y"))
    };
    for (name, geom, aes) in three() {
        for (stat, want) in &stats {
            for r in both_orders(&geom, &aes, &point(Some(stat.clone()))) {
                match &r {
                    Err(Error::UnsupportedStat(msg)) => {
                        assert!(msg.starts_with(want), "{name} + point {stat:?}: {msg}");
                    }
                    other => panic!("{name} + point {stat:?}: {other:?}"),
                }
            }
        }
        // The identity stat is the point's default: the guard passes it, as it passes no stat,
        // and the marks layer itself is what is refused.
        for r in both_orders(&geom, &aes, &point(Some(Stat::identity())))
            .into_iter()
            .chain(both_orders(&geom, &aes, &point(None)))
        {
            assert!(unresolved(name, &r), "{name} + point identity: {r:?}");
        }
    }
}

/// `companion` beside each of [`three`] over `df` with plot-level `plot_aes`, in both orders.
fn beside_every_geom(
    df: &DataFrame,
    plot_aes: &Aes,
    companion: &super::Layer,
) -> Vec<(&'static str, String, Result<BuiltLayout, Error>)> {
    use super::Layer;
    let mut out = Vec::new();
    for (name, geom, aes) in three() {
        let marks = Layer::new(geom).aes(aes);
        for (order, pair) in [("after", [&marks, companion]), ("before", [companion, &marks])] {
            let r = GGPlot::new()
                .data(df.clone())
                .aes(plot_aes.clone())
                .layer(pair[0].clone())
                .layer(pair[1].clone())
                .build()
                .and_then(|b| b.layout());
            out.push((name, format!("{name} ({order})"), r));
        }
    }
    out
}

fn point_layer(aes: Aes) -> super::Layer {
    super::Layer::new(Geom::point()).aes(aes)
}

/// [`wide`] plus a column `v` = `[1, cell, 3]`.
fn wide_with_v(cell: f32) -> DataFrame {
    let mut df = wide();
    df.add_column_f32("v", &[1.0, cell, 3.0]);
    df
}

#[test]
fn refuses_a_point_companion_layout_does_not_model_for_every_geom() {
    let mut short = wide();
    short.add_column_f32("s", &[1.0, 2.0]);
    let none = Aes::new();
    // (what, frame, plot-level aes, point aes, text the Error::Rendering must contain)
    let cases = [
        // Unmapped x or y: ggplot2 refuses the point; nothing here may guess one.
        ("no x", wide(), none.clone(), Aes::new().y("y"), "mapped x".to_string()),
        ("no y", wide(), none.clone(), Aes::new().x("x"), "mapped y".to_string()),
        // A short column: R cannot build that frame.
        ("short x", short.clone(), none.clone(), Aes::new().x("s").y("y"), "shorter".to_string()),
        ("short y", short, none, Aes::new().x("x").y("s"), "shorter".to_string()),
    ];
    for (what, df, plot_aes, aes, needle) in cases {
        for (_, name, r) in beside_every_geom(&df, &plot_aes, &point_layer(aes)) {
            match &r {
                Err(Error::Rendering(m)) if m.contains(&needle) => {}
                other => panic!("{name} + point {what}: {other:?}"),
            }
        }
    }
}

#[test]
fn refuses_a_text_point_column_for_every_geom() {
    use super::Layer;
    // A text column needs a discrete scale (ggplot2 4.0.3 makes x/y discrete). layout() takes a
    // text position to the position-scale path, which refuses every non-point layer first; the
    // companion guard must refuse it on its own too.
    for aes in [Aes::new().x("lab").y("y"), Aes::new().x("x").y("lab")] {
        for (_, name, r) in beside_every_geom(&wide(), &Aes::new(), &point_layer(aes.clone())) {
            assert!(
                matches!(&r, Err(Error::UnsupportedLayer(m)) if m.contains("geom_point only")),
                "{name} {aes:?}: {r:?}"
            );
        }
        for (name, geom, marks) in three() {
            let layers = [Layer::new(geom).aes(marks), point_layer(aes.clone())];
            let r = super::marks::require_modelled_companions(&layers, &Aes::new(), &wide());
            assert!(
                matches!(&r, Err(Error::Rendering(m)) if m.contains("is text")),
                "{name}: {r:?}"
            );
        }
    }
}

#[test]
fn refuses_a_missing_or_infinite_point_column_and_keeps_an_na_cell_for_every_geom() {
    let none = Aes::new();
    let unknown = point_layer(Aes::new().x("nope").y("y"));
    for (_, name, r) in beside_every_geom(&wide(), &none, &unknown) {
        assert!(matches!(&r, Err(Error::UnknownColumn(c)) if c == "nope"), "{name}: {r:?}");
    }
    // An infinite position: ggplot2 squishes it to the panel, which layout() does not model.
    for inf in [f32::INFINITY, f32::NEG_INFINITY] {
        let df = wide_with_v(inf);
        for aes in [Aes::new().x("v").y("y"), Aes::new().x("x").y("v")] {
            for (_, name, r) in beside_every_geom(&df, &none, &point_layer(aes.clone())) {
                assert!(matches!(r, Err(Error::ScaleDomain(_))), "{name} {aes:?}: {r:?}");
            }
        }
    }
    // The point's own frame is checked, not the plot's.
    let own = point_layer(Aes::new().x("x").y("y"))
        .data(frame(&[("x", &[1.0]), ("y", &[f32::INFINITY])]));
    for (_, name, r) in beside_every_geom(&wide(), &none, &own) {
        assert!(matches!(r, Err(Error::ScaleDomain(_))), "{name} own data: {r:?}");
    }
    // An NA cell is not refused by the guard: the scale trains on the other rows. The marks
    // layer itself is what is refused.
    let na = wide_with_v(f32::NAN);
    for (geom, name, r) in beside_every_geom(&na, &none, &point_layer(Aes::new().x("v").y("y"))) {
        assert!(unresolved(geom, &r), "{name} NA: {r:?}");
    }
}

#[test]
fn reports_a_companion_refusal_before_the_coord() {
    use super::{Coord, Layer};
    for (name, geom, aes) in three() {
        let r = GGPlot::new()
            .data(wide())
            .layer(Layer::new(geom).aes(aes))
            .layer(Layer::new(Geom::bar()).aes(Aes::new().x("x")))
            .coord(Coord::cartesian().flip())
            .build()
            .and_then(|b| b.layout());
        assert!(matches!(r, Err(Error::UnsupportedLayer(_))), "{name} + bar + flip: {r:?}");
    }
}

#[test]
fn the_companion_guard_refuses_the_path_family_on_its_own() {
    use super::Layer;
    // layout() runs the path guards first, so check the companion guard alone: it must not rely on
    // their order to refuse a path-family or reference layer.
    let xy = Aes::new().x("x").y("y");
    for (name, geom, aes) in three() {
        for other in [
            Layer::new(Geom::line()).aes(xy.clone()),
            Layer::new(Geom::path()).aes(xy.clone()),
            Layer::new(Geom::step()).aes(xy.clone()),
            Layer::new(Geom::hline(1.0)),
            Layer::new(Geom::vline(1.0)),
            Layer::new(Geom::abline(1.0, 0.0)),
        ] {
            let layers = [Layer::new(geom.clone()).aes(aes.clone()), other];
            let r = super::marks::require_modelled_companions(&layers, &Aes::new(), &wide());
            match &r {
                Err(Error::UnsupportedLayer(m)) => assert!(m.starts_with("a line, path"), "{m}"),
                other => panic!("{name}: {other:?}"),
            }
        }
    }
}

#[test]
fn the_companion_guard_passes_marks_beside_an_identity_point() {
    use super::Layer;
    // layout() refuses the marks layer after the guard, so check the guard's pass alone.
    for (name, geom, aes) in three() {
        let layers = [
            Layer::new(geom).aes(aes),
            point_layer(Aes::new().x("x").y("y")),
            point_layer(Aes::new().x("x").y("y")).data(wide_with_v(f32::NAN)),
        ];
        let r = super::marks::require_modelled_companions(&layers, &Aes::new(), &wide());
        assert!(r.is_ok(), "{name}: {r:?}");
    }
    // A point inherits x and y from the plot-level mapping.
    let layers = [Layer::new(Geom::tile()), point_layer(Aes::new())];
    let r = super::marks::require_modelled_companions(&layers, &Aes::new().x("x").y("y"), &wide());
    assert!(r.is_ok(), "inherited x/y: {r:?}");
}
