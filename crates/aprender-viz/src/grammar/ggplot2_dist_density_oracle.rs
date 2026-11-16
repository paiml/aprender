//! ggplot2 4.0.3 oracle for the `geom_density` row (PLOT-GG-001, paiml/aprender#6964):
//! `stat_density` (`bw.nrd0`, gaussian, 512 points) through [`GGPlot::layout`]. Fixtures are
//! committed under `fixtures/ggplot2/geom_density/`; R is never run here. Tolerance and helpers
//! are those of `ggplot2_oracle.rs` (1e-9 relative, 1e-12 absolute).

use serde_json::Value;

use super::ggplot2_oracle::{assert_close_slice, check_axis, frame_from_csv, nums};
use super::{Aes, BuiltLayout, GGPlot, Geom, Layer};

fn run(csv: &str) -> BuiltLayout {
    GGPlot::new()
        .data(frame_from_csv(csv))
        .aes(Aes::new().x("x"))
        .layer(Layer::new(Geom::gg_density()))
        .build()
        .expect("builds")
        .layout()
        .expect("layout")
}

fn check(case: &str, json: &str, csv: &str) {
    let want: Value = serde_json::from_str(json).expect("fixture is valid JSON");
    assert_eq!(want["versions"]["ggplot2"], "4.0.3", "{case}: not the pinned ggplot2");
    assert_eq!(want["versions"]["collate"], "C", "{case}: fixture not made under C collation");
    let got = run(csv);
    let (g, w) = (&got.layers[0], &want["layers"][0]);
    assert_eq!(w["geom"], "GeomDensity", "{case}");
    let d = g.density.as_ref().expect("a density layer carries its columns");
    assert_close_slice(&format!("{case}.x"), &g.x, &nums(&w["x"]));
    assert_close_slice(&format!("{case}.y"), &g.y, &nums(&w["y"]));
    assert_close_slice(&format!("{case}.density"), &g.y, &nums(&w["density"]));
    assert_close_slice(&format!("{case}.scaled"), &d.scaled, &nums(&w["scaled"]));
    assert_close_slice(&format!("{case}.ndensity"), &d.scaled, &nums(&w["ndensity"]));
    assert_close_slice(&format!("{case}.count"), &d.count, &nums(&w["count"]));
    assert_close_slice(&format!("{case}.wdensity"), &d.wdensity, &nums(&w["wdensity"]));
    let n: Vec<f64> = vec![d.n as f64; g.x.len()];
    assert_close_slice(&format!("{case}.n"), &n, &nums(&w["n"]));
    assert_eq!(g.x.len(), 512, "{case}: 512 points");
    check_axis(&format!("{case}.x"), &got.x, &want["panel"]["x"]);
    check_axis(&format!("{case}.y"), &got.y, &want["panel"]["y"]);
}

macro_rules! case {
    ($name:ident, $csv:literal) => {
        #[test]
        fn $name() {
            let json = include_str!(concat!(
                "../../fixtures/ggplot2/geom_density/expected_",
                stringify!($name),
                ".json"
            ));
            check(
                stringify!($name),
                json,
                include_str!(concat!("../../fixtures/ggplot2/geom_density/", $csv)),
            );
        }
    };
}

case!(default, "data.csv");
case!(three, "data_three.csv");
case!(outlier, "data_outlier.csv");
case!(negative, "data_negative.csv");
// IQR 0 with sd > 0: bw.nrd0 falls back to sd. Constant samples fall back to |x[1]|, then 1.
case!(iqr0, "data_iqr0.csv");
case!(constant, "data_constant.csv");
case!(zero, "data_zero.csv");
