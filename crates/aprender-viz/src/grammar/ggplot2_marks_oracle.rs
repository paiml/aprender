//! ggplot2 conformance oracle for the filled marks and text (PLOT-GG-001, paiml/aprender#6909):
//! `geom_area`, `geom_ribbon`, `geom_tile`, `geom_rect`, `geom_text`.
//!
//! Each case is one `<case>.csv` plus the `expected_<case>.json` that `ggplot_build()` in
//! ggplot2 4.0.3 produced from it (`fixtures/ggplot2/<row>/generate.R`, run with `LC_ALL=C`). The
//! fixtures are committed; R is never run here. The test rebuilds the same plot through the public
//! [`GGPlot`] API and compares [`BuiltGGPlot::layout`] column by column: the set of columns
//! ggplot2 emitted must equal the set built here, and every value must match.
//!
//! # Tolerance
//!
//! As in `ggplot2_oracle`: `|a - b| <= 1e-9 * max(|a|, |b|) + 1e-12` for positions, ranges and
//! breaks. Labels, colours, group ids, column names and counts compare exactly. `NA` is `null` in
//! the fixture and `NaN`/`None` here, and must match as `NA`.

use std::collections::BTreeMap;

use serde_json::Value;

use super::{Aes, BuiltLayout, DataFrame, DataValue, GGPlot, Geom};
use crate::color::Rgba;
use crate::error::Error;

const REL_TOL: f64 = 1e-9;
const ABS_TOL: f64 = 1e-12;

fn close(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= REL_TOL * a.abs().max(b.abs()) + ABS_TOL
}

/// A cell of a case CSV: the text and whether it was quoted (a quoted `NA` is the string "NA").
fn split_csv_line(line: &str) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    let (mut cur, mut quoted, mut in_q) = (String::new(), false, false);
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match (c, in_q) {
            ('"', true) if chars.peek() == Some(&'"') => {
                cur.push('"');
                chars.next();
            }
            ('"', _) => {
                in_q = !in_q;
                quoted = true;
            }
            (',', false) => out.push((std::mem::take(&mut cur), std::mem::take(&mut quoted))),
            (ch, _) => cur.push(ch),
        }
    }
    out.push((cur, quoted));
    out
}

/// Parse a case CSV. A column whose every cell is a number or an unquoted `NA` is numeric (`NA` is
/// `NaN`; every number must be exact in f32, so a fixture edit cannot turn this into a rounding
/// test); any other column is text (an unquoted `NA` is a missing cell).
fn frame_from_csv(csv: &str) -> DataFrame {
    let mut lines = csv.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<String> =
        split_csv_line(lines.next().expect("csv has a header")).into_iter().map(|c| c.0).collect();
    let rows: Vec<Vec<(String, bool)>> = lines.map(split_csv_line).collect();
    let mut df = DataFrame::new();
    for (c, name) in header.iter().enumerate() {
        let cells: Vec<&(String, bool)> = rows.iter().map(|r| &r[c]).collect();
        let numeric = cells.iter().all(|(s, q)| !q && (s == "NA" || s.parse::<f64>().is_ok()));
        if numeric {
            let vals: Vec<f32> = cells
                .iter()
                .map(|(s, _)| {
                    if s == "NA" {
                        return f32::NAN;
                    }
                    let v: f64 = s.parse().expect("numeric cell");
                    assert_eq!(f64::from(v as f32), v, "fixture value {v} is not exact in f32");
                    v as f32
                })
                .collect();
            df.add_column_f32(name.trim(), &vals);
        } else {
            let vals = cells
                .iter()
                .map(
                    |(s, q)| {
                        if !q && s == "NA" {
                            DataValue::Null
                        } else {
                            DataValue::Text(s.clone())
                        }
                    },
                )
                .collect();
            df.add_column_values(name.trim(), vals);
        }
    }
    df
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|e| if e.is_null() { f64::NAN } else { e.as_f64().expect("number") })
        .collect()
}

fn check_axis(name: &str, got: &super::PositionScale, want: &Value) {
    let pair = |t: (f64, f64)| vec![t.0, t.1];
    for (what, g, w) in [
        ("limits", pair(got.limits), nums(&want["limits"])),
        ("continuous_range", pair(got.continuous_range), nums(&want["continuous_range"])),
        ("breaks", got.breaks.clone(), nums(&want["breaks"])),
    ] {
        assert_eq!(g.len(), w.len(), "{name}.{what}: got {g:?} want {w:?}");
        for (i, (a, b)) in g.iter().zip(&w).enumerate() {
            assert!(close(*a, *b), "{name}.{what}[{i}]: got {a:?} want {b:?}");
        }
    }
    let labels: Vec<String> = want["labels"]
        .as_array()
        .expect("labels")
        .iter()
        .map(|s| s.as_str().expect("str").to_string())
        .collect();
    assert_eq!(got.labels, labels, "{name}.labels");
    assert_eq!(got.n_breaks_censored as u64, want["n_breaks_censored"].as_u64().expect("count"));
}

/// One column of the Rust-built layer, in the shape the fixture stores it.
enum Col {
    Num(Vec<f64>),
    Text(Vec<Option<String>>),
    Colour(Vec<Option<[u8; 4]>>),
    Group(Vec<i32>),
}

/// The columns of the marks layer at `index`.
fn layer_columns(layout: &BuiltLayout, index: usize) -> (usize, BTreeMap<String, Col>) {
    let m = layout.layers[index].marks.as_ref().expect("a marks layer");
    let n = m.rows();
    let mut cols = BTreeMap::new();
    cols.insert("group".to_string(), Col::Group(m.group.clone()));
    for (k, v) in &m.num {
        cols.insert((*k).to_string(), Col::Num(v.clone()));
    }
    for (k, v) in &m.texts {
        cols.insert((*k).to_string(), Col::Text(v.clone()));
    }
    for (k, v) in &m.scalars {
        cols.insert((*k).to_string(), Col::Num(vec![*v; n]));
    }
    for (k, v) in &m.colours {
        cols.insert((*k).to_string(), Col::Colour(vec![*v; n]));
    }
    (n, cols)
}

fn check_column(name: &str, got: &Col, layer: &Value) {
    match got {
        Col::Group(g) => {
            let want: Vec<i32> = layer["group"]
                .as_array()
                .expect("group")
                .iter()
                .map(|v| v.as_i64().expect("int") as i32)
                .collect();
            assert_eq!(*g, want, "group");
        }
        Col::Colour(c) => {
            let want =
                layer[format!("{name}_rgba")].as_array().unwrap_or_else(|| panic!("{name}_rgba"));
            assert_eq!(c.len(), want.len(), "{name} length");
            for (i, (g, w)) in c.iter().zip(want).enumerate() {
                let w: Option<[u8; 4]> = (!w.is_null()).then(|| {
                    let a = nums(w);
                    [a[0] as u8, a[1] as u8, a[2] as u8, a[3] as u8]
                });
                assert_eq!(*g, w, "{name}[{i}]");
            }
        }
        Col::Text(t) => {
            let want = layer[name].as_array().unwrap_or_else(|| panic!("{name}"));
            let want: Vec<Option<String>> =
                want.iter().map(|v| v.as_str().map(str::to_string)).collect();
            assert_eq!(*t, want, "{name}");
        }
        Col::Num(v) => {
            let want: Vec<f64> = layer[name]
                .as_array()
                .unwrap_or_else(|| panic!("{name}"))
                .iter()
                .map(|e| match e {
                    Value::Null => f64::NAN,
                    Value::Bool(b) => f64::from(u8::from(*b)),
                    other => other.as_f64().expect("number"),
                })
                .collect();
            assert_eq!(v.len(), want.len(), "{name}: got {v:?} want {want:?}");
            for (i, (a, b)) in v.iter().zip(&want).enumerate() {
                assert!(close(*a, *b), "{name}[{i}]: got {a:?} want {b:?}");
            }
        }
    }
}

/// Build the case's plot through the public API.
fn plot(csv: &str, aes: Aes, geom: Geom) -> GGPlot {
    GGPlot::new().data(frame_from_csv(csv)).aes(aes).geom(geom)
}

fn layout_of(csv: &str, aes: Aes, geom: Geom) -> BuiltLayout {
    plot(csv, aes, geom).build().expect("plot builds").layout().expect("layout resolves")
}

fn check_case(json: &str, geom_class: &str, row: &str, got: &BuiltLayout) {
    let want: Value = serde_json::from_str(json).expect("fixture is valid JSON");
    assert_eq!(
        want["versions"]["ggplot2"], "4.0.3",
        "fixture was not produced by the pinned ggplot2"
    );
    assert_eq!(want["collation"], "C", "fixture must be generated with LC_ALL=C");
    assert_eq!(want["row"], row);
    assert_eq!(got.layers.len(), 1);
    let layer = &want["layers"][0];
    assert_eq!(layer["geom"], geom_class);
    check_layer(got, 0, layer);
    check_axis("x", &got.x, &want["panel"]["x"]);
    check_axis("y", &got.y, &want["panel"]["y"]);
}

/// The marks layer at `index` against the fixture's layer: column set, row count, every value.
fn check_layer(got: &BuiltLayout, index: usize, layer: &Value) {
    let (n, cols) = layer_columns(got, index);
    let mut want_names: Vec<String> = layer["columns"]
        .as_array()
        .expect("columns")
        .iter()
        .map(|c| c.as_str().expect("name").to_string())
        .collect();
    want_names.sort();
    let got_names: Vec<String> = cols.keys().cloned().collect();
    assert_eq!(got_names, want_names, "column set");
    assert_eq!(n, layer["group"].as_array().expect("group").len(), "row count");
    for (name, col) in &cols {
        check_column(name, col, layer);
    }
}

fn steelblue_red(a: Aes) -> Aes {
    a.fill_value(Rgba::rgb(70, 130, 180)).color_value(Rgba::rgb(255, 0, 0)).alpha_value(0.5)
}

/// `aes` for a row's CSV: positions per row, plus `group = g` when the CSV has a `g` column.
fn row_aes(row: &str, csv: &str) -> Aes {
    let base = match row {
        "geom_area" | "geom_tile" => Aes::new().x("x").y("y"),
        "geom_ribbon" => Aes::new().x("x").ymin("lo").ymax("hi"),
        "geom_rect" => Aes::new().xmin("x0").xmax("x1").ymin("y0").ymax("y1"),
        "geom_text" => Aes::new().x("x").y("y").label("lab"),
        other => panic!("unknown row {other}"),
    };
    if csv.lines().next().is_some_and(|h| h.split(',').any(|c| c == "g")) {
        base.group("g")
    } else {
        base
    }
}

fn row_geom(row: &str) -> (Geom, &'static str) {
    match row {
        "geom_area" => (Geom::area(), "GeomArea"),
        "geom_ribbon" => (Geom::ribbon(), "GeomRibbon"),
        "geom_tile" => (Geom::tile(), "GeomTile"),
        "geom_rect" => (Geom::rect(), "GeomRect"),
        "geom_text" => (Geom::text(), "GeomText"),
        other => panic!("unknown row {other}"),
    }
}

fn run_case(row: &str, csv: &str, json: &str, style: bool) {
    let (geom, class) = row_geom(row);
    let mut aes = row_aes(row, csv);
    if style {
        aes = if row == "geom_text" {
            aes.color_value(Rgba::rgb(255, 0, 0)).alpha_value(0.5).size_value(5.0)
        } else {
            steelblue_red(aes)
        };
    }
    let got = layout_of(csv, aes, geom);
    check_case(json, class, row, &got);
}

mod area {
    use super::run_case;

    #[test]
    fn groups_text() {
        run_case(
            "geom_area",
            include_str!("../../fixtures/ggplot2/geom_area/groups_text.csv"),
            include_str!("../../fixtures/ggplot2/geom_area/expected_groups_text.json"),
            false,
        );
    }

    #[test]
    fn groups_num() {
        run_case(
            "geom_area",
            include_str!("../../fixtures/ggplot2/geom_area/groups_num.csv"),
            include_str!("../../fixtures/ggplot2/geom_area/expected_groups_num.json"),
            false,
        );
    }

    #[test]
    fn single_unsorted() {
        run_case(
            "geom_area",
            include_str!("../../fixtures/ggplot2/geom_area/single_unsorted.csv"),
            include_str!("../../fixtures/ggplot2/geom_area/expected_single_unsorted.json"),
            false,
        );
    }

    #[test]
    fn single_dup_x() {
        run_case(
            "geom_area",
            include_str!("../../fixtures/ggplot2/geom_area/single_dup_x.csv"),
            include_str!("../../fixtures/ggplot2/geom_area/expected_single_dup_x.json"),
            false,
        );
    }

    #[test]
    fn na_rows() {
        run_case(
            "geom_area",
            include_str!("../../fixtures/ggplot2/geom_area/na_rows.csv"),
            include_str!("../../fixtures/ggplot2/geom_area/expected_na_rows.json"),
            false,
        );
    }

    #[test]
    fn two_row_groups() {
        run_case(
            "geom_area",
            include_str!("../../fixtures/ggplot2/geom_area/two_row_groups.csv"),
            include_str!("../../fixtures/ggplot2/geom_area/expected_two_row_groups.json"),
            false,
        );
    }

    #[test]
    fn zero_y() {
        run_case(
            "geom_area",
            include_str!("../../fixtures/ggplot2/geom_area/zero_y.csv"),
            include_str!("../../fixtures/ggplot2/geom_area/expected_zero_y.json"),
            false,
        );
    }

    #[test]
    fn style() {
        run_case(
            "geom_area",
            include_str!("../../fixtures/ggplot2/geom_area/style.csv"),
            include_str!("../../fixtures/ggplot2/geom_area/expected_style.json"),
            true,
        );
    }

    #[test]
    fn dup_in_group_multi() {
        run_case(
            "geom_area",
            include_str!("../../fixtures/ggplot2/geom_area/dup_in_group_multi.csv"),
            include_str!("../../fixtures/ggplot2/geom_area/expected_dup_in_group_multi.json"),
            false,
        );
    }

    #[test]
    fn tiny_gap() {
        run_case(
            "geom_area",
            include_str!("../../fixtures/ggplot2/geom_area/tiny_gap.csv"),
            include_str!("../../fixtures/ggplot2/geom_area/expected_tiny_gap.json"),
            false,
        );
    }
}
mod ribbon {
    use super::run_case;

    #[test]
    fn basic_unsorted() {
        run_case(
            "geom_ribbon",
            include_str!("../../fixtures/ggplot2/geom_ribbon/basic_unsorted.csv"),
            include_str!("../../fixtures/ggplot2/geom_ribbon/expected_basic_unsorted.json"),
            false,
        );
    }

    #[test]
    fn groups_text() {
        run_case(
            "geom_ribbon",
            include_str!("../../fixtures/ggplot2/geom_ribbon/groups_text.csv"),
            include_str!("../../fixtures/ggplot2/geom_ribbon/expected_groups_text.json"),
            false,
        );
    }

    #[test]
    fn groups_num() {
        run_case(
            "geom_ribbon",
            include_str!("../../fixtures/ggplot2/geom_ribbon/groups_num.csv"),
            include_str!("../../fixtures/ggplot2/geom_ribbon/expected_groups_num.json"),
            false,
        );
    }

    #[test]
    fn na_ymin() {
        run_case(
            "geom_ribbon",
            include_str!("../../fixtures/ggplot2/geom_ribbon/na_ymin.csv"),
            include_str!("../../fixtures/ggplot2/geom_ribbon/expected_na_ymin.json"),
            false,
        );
    }

    #[test]
    fn na_x() {
        run_case(
            "geom_ribbon",
            include_str!("../../fixtures/ggplot2/geom_ribbon/na_x.csv"),
            include_str!("../../fixtures/ggplot2/geom_ribbon/expected_na_x.json"),
            false,
        );
    }

    #[test]
    fn dup_x() {
        run_case(
            "geom_ribbon",
            include_str!("../../fixtures/ggplot2/geom_ribbon/dup_x.csv"),
            include_str!("../../fixtures/ggplot2/geom_ribbon/expected_dup_x.json"),
            false,
        );
    }

    #[test]
    fn reversed() {
        run_case(
            "geom_ribbon",
            include_str!("../../fixtures/ggplot2/geom_ribbon/reversed.csv"),
            include_str!("../../fixtures/ggplot2/geom_ribbon/expected_reversed.json"),
            false,
        );
    }

    #[test]
    fn two_row_groups() {
        run_case(
            "geom_ribbon",
            include_str!("../../fixtures/ggplot2/geom_ribbon/two_row_groups.csv"),
            include_str!("../../fixtures/ggplot2/geom_ribbon/expected_two_row_groups.json"),
            false,
        );
    }

    #[test]
    fn style() {
        run_case(
            "geom_ribbon",
            include_str!("../../fixtures/ggplot2/geom_ribbon/style.csv"),
            include_str!("../../fixtures/ggplot2/geom_ribbon/expected_style.json"),
            true,
        );
    }
}
mod tile {
    use super::run_case;

    #[test]
    fn grid_regular() {
        run_case(
            "geom_tile",
            include_str!("../../fixtures/ggplot2/geom_tile/grid_regular.csv"),
            include_str!("../../fixtures/ggplot2/geom_tile/expected_grid_regular.json"),
            false,
        );
    }

    #[test]
    fn fractional_res() {
        run_case(
            "geom_tile",
            include_str!("../../fixtures/ggplot2/geom_tile/fractional_res.csv"),
            include_str!("../../fixtures/ggplot2/geom_tile/expected_fractional_res.json"),
            false,
        );
    }

    #[test]
    fn unsorted_dup() {
        run_case(
            "geom_tile",
            include_str!("../../fixtures/ggplot2/geom_tile/unsorted_dup.csv"),
            include_str!("../../fixtures/ggplot2/geom_tile/expected_unsorted_dup.json"),
            false,
        );
    }

    #[test]
    fn na_cells() {
        run_case(
            "geom_tile",
            include_str!("../../fixtures/ggplot2/geom_tile/na_cells.csv"),
            include_str!("../../fixtures/ggplot2/geom_tile/expected_na_cells.json"),
            false,
        );
    }

    #[test]
    fn single_cell() {
        run_case(
            "geom_tile",
            include_str!("../../fixtures/ggplot2/geom_tile/single_cell.csv"),
            include_str!("../../fixtures/ggplot2/geom_tile/expected_single_cell.json"),
            false,
        );
    }

    #[test]
    fn same_x() {
        run_case(
            "geom_tile",
            include_str!("../../fixtures/ggplot2/geom_tile/same_x.csv"),
            include_str!("../../fixtures/ggplot2/geom_tile/expected_same_x.json"),
            false,
        );
    }

    #[test]
    fn groups_text() {
        run_case(
            "geom_tile",
            include_str!("../../fixtures/ggplot2/geom_tile/groups_text.csv"),
            include_str!("../../fixtures/ggplot2/geom_tile/expected_groups_text.json"),
            false,
        );
    }

    #[test]
    fn style() {
        run_case(
            "geom_tile",
            include_str!("../../fixtures/ggplot2/geom_tile/style.csv"),
            include_str!("../../fixtures/ggplot2/geom_tile/expected_style.json"),
            true,
        );
    }

    #[test]
    fn wide_gaps() {
        run_case(
            "geom_tile",
            include_str!("../../fixtures/ggplot2/geom_tile/wide_gaps.csv"),
            include_str!("../../fixtures/ggplot2/geom_tile/expected_wide_gaps.json"),
            false,
        );
    }

    #[test]
    fn subtol_gap() {
        run_case(
            "geom_tile",
            include_str!("../../fixtures/ggplot2/geom_tile/subtol_gap.csv"),
            include_str!("../../fixtures/ggplot2/geom_tile/expected_subtol_gap.json"),
            false,
        );
    }
}
mod rect {
    use super::run_case;

    #[test]
    fn basic() {
        run_case(
            "geom_rect",
            include_str!("../../fixtures/ggplot2/geom_rect/basic.csv"),
            include_str!("../../fixtures/ggplot2/geom_rect/expected_basic.json"),
            false,
        );
    }

    #[test]
    fn reversed() {
        run_case(
            "geom_rect",
            include_str!("../../fixtures/ggplot2/geom_rect/reversed.csv"),
            include_str!("../../fixtures/ggplot2/geom_rect/expected_reversed.json"),
            false,
        );
    }

    #[test]
    fn na_edge() {
        run_case(
            "geom_rect",
            include_str!("../../fixtures/ggplot2/geom_rect/na_edge.csv"),
            include_str!("../../fixtures/ggplot2/geom_rect/expected_na_edge.json"),
            false,
        );
    }

    #[test]
    fn duplicate_rows() {
        run_case(
            "geom_rect",
            include_str!("../../fixtures/ggplot2/geom_rect/duplicate_rows.csv"),
            include_str!("../../fixtures/ggplot2/geom_rect/expected_duplicate_rows.json"),
            false,
        );
    }

    #[test]
    fn groups_text() {
        run_case(
            "geom_rect",
            include_str!("../../fixtures/ggplot2/geom_rect/groups_text.csv"),
            include_str!("../../fixtures/ggplot2/geom_rect/expected_groups_text.json"),
            false,
        );
    }

    #[test]
    fn zero_width() {
        run_case(
            "geom_rect",
            include_str!("../../fixtures/ggplot2/geom_rect/zero_width.csv"),
            include_str!("../../fixtures/ggplot2/geom_rect/expected_zero_width.json"),
            false,
        );
    }

    #[test]
    fn negative() {
        run_case(
            "geom_rect",
            include_str!("../../fixtures/ggplot2/geom_rect/negative.csv"),
            include_str!("../../fixtures/ggplot2/geom_rect/expected_negative.json"),
            false,
        );
    }

    #[test]
    fn style() {
        run_case(
            "geom_rect",
            include_str!("../../fixtures/ggplot2/geom_rect/style.csv"),
            include_str!("../../fixtures/ggplot2/geom_rect/expected_style.json"),
            true,
        );
    }

    #[test]
    fn single_row() {
        run_case(
            "geom_rect",
            include_str!("../../fixtures/ggplot2/geom_rect/single_row.csv"),
            include_str!("../../fixtures/ggplot2/geom_rect/expected_single_row.json"),
            false,
        );
    }

    #[test]
    fn group_order_c() {
        run_case(
            "geom_rect",
            include_str!("../../fixtures/ggplot2/geom_rect/group_order_c.csv"),
            include_str!("../../fixtures/ggplot2/geom_rect/expected_group_order_c.json"),
            false,
        );
    }
}
mod text {
    use super::run_case;

    #[test]
    fn basic() {
        run_case(
            "geom_text",
            include_str!("../../fixtures/ggplot2/geom_text/basic.csv"),
            include_str!("../../fixtures/ggplot2/geom_text/expected_basic.json"),
            false,
        );
    }

    #[test]
    fn unsorted_dup() {
        run_case(
            "geom_text",
            include_str!("../../fixtures/ggplot2/geom_text/unsorted_dup.csv"),
            include_str!("../../fixtures/ggplot2/geom_text/expected_unsorted_dup.json"),
            false,
        );
    }

    #[test]
    fn na_xy() {
        run_case(
            "geom_text",
            include_str!("../../fixtures/ggplot2/geom_text/na_xy.csv"),
            include_str!("../../fixtures/ggplot2/geom_text/expected_na_xy.json"),
            false,
        );
    }

    #[test]
    fn groups_text() {
        run_case(
            "geom_text",
            include_str!("../../fixtures/ggplot2/geom_text/groups_text.csv"),
            include_str!("../../fixtures/ggplot2/geom_text/expected_groups_text.json"),
            false,
        );
    }

    #[test]
    fn style() {
        run_case(
            "geom_text",
            include_str!("../../fixtures/ggplot2/geom_text/style.csv"),
            include_str!("../../fixtures/ggplot2/geom_text/expected_style.json"),
            true,
        );
    }

    #[test]
    fn odd_labels() {
        run_case(
            "geom_text",
            include_str!("../../fixtures/ggplot2/geom_text/odd_labels.csv"),
            include_str!("../../fixtures/ggplot2/geom_text/expected_odd_labels.json"),
            false,
        );
    }

    #[test]
    fn single_row() {
        run_case(
            "geom_text",
            include_str!("../../fixtures/ggplot2/geom_text/single_row.csv"),
            include_str!("../../fixtures/ggplot2/geom_text/expected_single_row.json"),
            false,
        );
    }

    #[test]
    fn negative() {
        run_case(
            "geom_text",
            include_str!("../../fixtures/ggplot2/geom_text/negative.csv"),
            include_str!("../../fixtures/ggplot2/geom_text/expected_negative.json"),
            false,
        );
    }

    #[test]
    fn na_label() {
        run_case(
            "geom_text",
            include_str!("../../fixtures/ggplot2/geom_text/na_label.csv"),
            include_str!("../../fixtures/ggplot2/geom_text/expected_na_label.json"),
            false,
        );
    }
}

// --- refusals: out-of-scope input is a typed error, never a plausible wrong answer ---------------

fn frame(cols: &[(&str, &[f32])]) -> DataFrame {
    let mut df = DataFrame::new();
    for (n, v) in cols {
        df.add_column_f32(n, v);
    }
    df
}

fn run(df: DataFrame, aes: Aes, geom: Geom) -> Result<BuiltLayout, Error> {
    GGPlot::new().data(df).aes(aes).geom(geom).build()?.layout()
}

fn xy() -> DataFrame {
    frame(&[("x", &[1.0, 2.0, 3.0]), ("y", &[1.0, 2.0, 3.0]), ("c", &[1.0, 2.0, 3.0])])
}

/// Every one of the five geoms with a frame and mapping that lays out cleanly under the default
/// coord. The frame carries the columns of all five (`x y lo hi x0 x1 y0 y1 lab g`), three rows.
fn five() -> Vec<(&'static str, Geom, Aes)> {
    vec![
        ("area", Geom::area(), Aes::new().x("x").y("y").group("g")),
        ("ribbon", Geom::ribbon(), Aes::new().x("x").ymin("lo").ymax("hi").group("g")),
        ("tile", Geom::tile(), Aes::new().x("x").y("y").group("g")),
        ("rect", Geom::rect(), Aes::new().xmin("x0").xmax("x1").ymin("y0").ymax("y1").group("g")),
        ("text", Geom::text(), Aes::new().x("x").y("y").label("lab").group("g")),
    ]
}

fn wide() -> DataFrame {
    let mut df = frame(&[
        ("x", &[1.0, 2.0, 3.0]),
        ("y", &[2.0, 5.0, 3.0]),
        ("lo", &[1.0, 1.0, 1.0]),
        ("hi", &[2.0, 3.0, 4.0]),
        ("x0", &[0.0, 1.0, 2.0]),
        ("x1", &[1.0, 2.0, 3.0]),
        ("y0", &[1.0, 1.0, 1.0]),
        ("y1", &[2.0, 3.0, 4.0]),
        ("g", &[1.0, 1.0, 1.0]),
    ]);
    df.add_column_str("lab", &["p", "q", "r"]);
    df
}

fn layout_with(
    df: &DataFrame,
    aes: Aes,
    geom: Geom,
    coord: super::Coord,
    facet: super::Facet,
) -> Result<BuiltLayout, Error> {
    GGPlot::new().data(df.clone()).aes(aes).geom(geom).coord(coord).facet(facet).build()?.layout()
}

#[test]
fn refuses_every_non_default_coord_and_facet_for_every_geom() {
    use super::{Coord, Facet};
    let coords = [
        ("xlim", Coord::cartesian().xlim(0.0, 2.0)),
        ("ylim", Coord::cartesian().ylim(0.0, 2.0)),
        ("xlim+ylim", Coord::cartesian().xlim(0.0, 2.0).ylim(0.0, 2.0)),
        ("flip", Coord::cartesian().flip()),
        ("polar", Coord::polar()),
        ("polar start", Coord::polar().start_angle(1.0).direction(-1)),
        ("fixed 1", Coord::fixed(1.0)),
        ("fixed 2", Coord::fixed(2.0)),
    ];
    let facets = [
        ("wrap", Facet::wrap("g", 2)),
        ("grid", Facet::grid("g", "g")),
        ("row", Facet::row("g")),
        ("col", Facet::col("g")),
    ];
    let df = wide();
    for (name, geom, aes) in five() {
        // The default coord and facet lay the same plot out, so each refusal below is the coord's.
        let ok = layout_with(&df, aes.clone(), geom.clone(), Coord::cartesian(), Facet::none());
        assert!(ok.is_ok(), "{name}: default coord and facet must lay out: {ok:?}");
        for (what, coord) in &coords {
            let r = layout_with(&df, aes.clone(), geom.clone(), coord.clone(), Facet::none());
            assert!(matches!(r, Err(Error::UnsupportedCoord(_))), "{name} {what}: {r:?}");
        }
        for (what, facet) in &facets {
            let r = layout_with(&df, aes.clone(), geom.clone(), Coord::cartesian(), facet.clone());
            assert!(matches!(r, Err(Error::UnsupportedCoord(_))), "{name} {what}: {r:?}");
        }
    }
}

#[test]
fn refuses_non_default_coord_when_a_marks_layer_is_not_the_only_layer() {
    use super::{Coord, Layer};
    let df = wide();
    for (name, geom, aes) in five() {
        let r = GGPlot::new()
            .data(df.clone())
            .layer(Layer::new(Geom::point()).aes(Aes::new().x("x").y("y")))
            .layer(Layer::new(geom).aes(aes))
            .coord(Coord::cartesian().flip())
            .build()
            .and_then(|b| b.layout());
        assert!(matches!(r, Err(Error::UnsupportedCoord(_))), "{name}: {r:?}");
    }
}

#[test]
fn path_guards_leave_marks_layouts_alone_and_refuse_mixes_with_typed_errors() {
    use super::{Coord, Layer};
    let df = wide();
    for (name, geom, aes) in five() {
        // A marks layer with a point layer is laid out (the path guards are inert without a
        // path-family layer).
        let ok = GGPlot::new()
            .data(df.clone())
            .layer(Layer::new(Geom::point()).aes(Aes::new().x("x").y("y")))
            .layer(Layer::new(geom.clone()).aes(aes.clone()))
            .build()
            .and_then(|b| b.layout());
        assert!(ok.is_ok(), "{name} + point: {ok:?}");
        // Sharing a plot with a path-family layer is refused, never laid out as raw data.
        let mixed = |coord: Coord| {
            GGPlot::new()
                .data(df.clone())
                .layer(Layer::new(Geom::line()).aes(Aes::new().x("x").y("y")))
                .layer(Layer::new(geom.clone()).aes(aes.clone()))
                .coord(coord)
                .build()
                .and_then(|b| b.layout())
        };
        let r = mixed(Coord::cartesian());
        assert!(matches!(r, Err(Error::UnsupportedLayer(_))), "{name} + line: {r:?}");
        // The coord refusal is reported before the layer refusal, and is a coord error.
        let r = mixed(Coord::cartesian().flip());
        assert!(matches!(r, Err(Error::UnsupportedCoord(_))), "{name} + line + flip: {r:?}");
    }
}

#[test]
fn refuses_a_short_column_for_every_geom() {
    // R cannot build a data frame with columns of different lengths; NA-padding one would invent
    // numbers. Each mapped column of each geom is cut to two cells in a three-row frame.
    let full = wide();
    for (name, geom, aes) in five() {
        let mapped: Vec<&str> =
            [&aes.x, &aes.y, &aes.ymin, &aes.ymax, &aes.xmin, &aes.xmax, &aes.label, &aes.group]
                .iter()
                .filter_map(|s| s.as_deref())
                .collect();
        assert!(mapped.len() >= 3, "{name}: {mapped:?}");
        for col in mapped {
            let mut df = full.clone();
            if col == "lab" {
                df.add_column_str(col, &["p", "q"]);
            } else {
                df.add_column_f32(col, &[1.0, 2.0]);
            }
            assert_eq!(df.nrow(), 3, "{name} {col}: the frame keeps its longest column");
            let r = run(df, aes.clone(), geom.clone());
            assert!(matches!(r, Err(Error::Rendering(_))), "{name} {col}: {r:?}");
        }
    }
}

#[test]
fn refuses_infinite_and_text_positions_for_every_geom() {
    let full = wide();
    for (name, geom, aes) in five() {
        let positions: Vec<&str> = [&aes.x, &aes.y, &aes.ymin, &aes.ymax, &aes.xmin, &aes.xmax]
            .iter()
            .filter_map(|s| s.as_deref())
            .collect();
        for col in positions {
            for inf in [f32::INFINITY, f32::NEG_INFINITY] {
                let mut df = full.clone();
                df.add_column_f32(col, &[1.0, inf, 3.0]);
                let r = run(df, aes.clone(), geom.clone());
                assert!(matches!(r, Err(Error::ScaleDomain(_))), "{name} {col} {inf}: {r:?}");
            }
            let mut df = full.clone();
            df.add_column_str(col, &["a", "b", "c"]);
            let r = run(df, aes.clone(), geom.clone());
            assert!(matches!(r, Err(Error::Rendering(_))), "{name} {col} text: {r:?}");
        }
    }
}

#[test]
fn refuses_a_stat_the_layer_does_not_model() {
    use super::Stat;
    let df = wide();
    for (name, geom, aes) in five() {
        for stat in [
            Stat::count(),
            Stat::bin(5),
            Stat::density(),
            Stat::boxplot(),
            Stat::smooth(),
            Stat::bin2d(2, 2),
            Stat::sum(),
            Stat::mean(),
        ] {
            let r = run(df.clone(), aes.clone(), geom.clone().stat(stat.clone()));
            assert!(matches!(r, Err(Error::Rendering(_))), "{name} {stat:?}: {r:?}");
        }
        // stat = "identity" is the default of ribbon, tile, rect and text; geom_area's default is
        // stat_align, so an explicit identity is a different layer (measured: no padding rows).
        let r = run(df.clone(), aes.clone(), geom.clone().stat(Stat::identity()));
        if name == "area" {
            assert!(matches!(r, Err(Error::Rendering(_))), "{name} identity: {r:?}");
        } else {
            // Debug text, not ==: a built NaN (alpha NA) is not equal to itself.
            let (a, b) =
                (format!("{:?}", r.ok()), format!("{:?}", run(df.clone(), aes, geom).ok()));
            assert_eq!(a, b, "{name} identity");
        }
    }
}

#[test]
fn refuses_a_geom_alpha_that_layout_does_not_read() {
    let df = wide();
    let aes = Aes::new().x("x").y("y");
    let r = run(df.clone(), aes.clone(), Geom::area().alpha(0.5));
    assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
    // 0.3 is what Geom::area() starts with: the same layer, so it is not refused.
    assert!(run(df.clone(), aes.clone(), Geom::area().alpha(0.3)).is_ok());
    // Aes::alpha_value is the supported route, and the built column carries it.
    let ok = run(df, aes.alpha_value(0.5), Geom::area()).expect("alpha_value lays out");
    let m = ok.layers[0].marks.as_ref().expect("marks");
    assert!((m.scalars["alpha"] - 0.5).abs() < 1e-12);
}

#[test]
fn refuses_aesthetics_ggplot2_keeps_and_this_layer_would_drop() {
    let df = wide();
    // geom_rect(aes(x, y)) keeps x and y and trains both scales on them (measured).
    for aes in [
        Aes::new().x("x").xmin("x0").xmax("x1").ymin("y0").ymax("y1"),
        Aes::new().y("y").xmin("x0").xmax("x1").ymin("y0").ymax("y1"),
    ] {
        let r = run(df.clone(), aes, Geom::rect());
        assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
    }
    // geom_text(aes(ymin = ...)) keeps the column and trains y on it (measured).
    let base = Aes::new().x("x").y("y").label("lab");
    for aes in
        [base.clone().xmin("x0"), base.clone().xmax("x1"), base.clone().ymin("y0"), base.ymax("y1")]
    {
        let r = run(df.clone(), aes, Geom::text());
        assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
    }
    // A fixed size is linewidth (deprecated) on every geom but text.
    for (name, geom, aes) in five() {
        let r = run(df.clone(), aes.clone().size_value(3.0), geom);
        if name == "text" {
            assert!(r.is_ok(), "{name}: {r:?}");
        } else {
            assert!(matches!(r, Err(Error::Rendering(_))), "{name}: {r:?}");
        }
    }
}

#[test]
fn overwritten_aesthetics_equal_default() {
    // Measured on ggplot2 4.0.3 (ggplot_build): setup_data overwrites these, so the built data and
    // both scales equal the layer without them. Area: ymin, ymax. Tile: xmin, xmax, ymin, ymax.
    // Ribbon: an inherited y.
    let df = wide();
    let same = |geom: fn() -> Geom, plain: Aes, extra: Aes| {
        // Debug text, not ==: a built NaN (alpha NA) is not equal to itself.
        let (a, b) = (run(df.clone(), plain, geom()), run(df.clone(), extra, geom()));
        assert!(a.is_ok(), "{a:?}");
        assert_eq!(
            format!("{a:?}"),
            format!("{b:?}"),
            "an overwritten aesthetic changed the layout"
        );
    };
    let xy = || Aes::new().x("x").y("y");
    same(Geom::area, xy(), xy().ymin("lo").ymax("hi"));
    same(Geom::tile, xy(), xy().xmin("x0").xmax("x1").ymin("lo").ymax("hi"));
    let rib = || Aes::new().x("x").ymin("lo").ymax("hi");
    same(Geom::ribbon, rib(), rib().y("y"));
    assert!(run(df, xy(), Geom::area()).is_ok());
}

#[test]
fn refuses_mapped_style_aesthetics() {
    for aes in [
        Aes::new().x("x").y("y").color("c"),
        Aes::new().x("x").y("y").fill("c"),
        Aes::new().x("x").y("y").size("c"),
        Aes::new().x("x").y("y").alpha("c"),
        Aes::new().x("x").y("y").shape("c"),
    ] {
        for geom in [Geom::area(), Geom::tile(), Geom::text()] {
            assert!(matches!(run(xy(), aes.clone(), geom), Err(Error::Rendering(_))));
        }
    }
}

#[test]
fn refuses_missing_required_aesthetics() {
    let r = |aes: Aes, g: Geom| run(xy(), aes, g);
    assert!(matches!(r(Aes::new().x("x").ymin("y"), Geom::ribbon()), Err(Error::Rendering(_))));
    assert!(matches!(r(Aes::new().x("x").ymax("y"), Geom::ribbon()), Err(Error::Rendering(_))));
    assert!(matches!(
        r(Aes::new().xmin("x").xmax("x").ymin("y"), Geom::rect()),
        Err(Error::Rendering(_))
    ));
    assert!(matches!(r(Aes::new().x("x").y("y"), Geom::text()), Err(Error::Rendering(_))));
    assert!(matches!(r(Aes::new().y("y"), Geom::area()), Err(Error::Rendering(_))));
}

#[test]
fn refuses_unknown_column() {
    let r = run(xy(), Aes::new().x("nope").y("y"), Geom::tile());
    assert!(matches!(r, Err(Error::UnknownColumn(_))), "{r:?}");
}

#[test]
fn refuses_negative_area_and_degenerate_groups() {
    let neg = frame(&[("x", &[1.0, 2.0]), ("y", &[1.0, -2.0])]);
    assert!(matches!(run(neg, Aes::new().x("x").y("y"), Geom::area()), Err(Error::Rendering(_))));
    let same_x = frame(&[("x", &[1.0, 1.0]), ("y", &[1.0, 2.0]), ("g", &[1.0, 2.0])]);
    let r = run(same_x, Aes::new().x("x").y("y").group("g"), Geom::area());
    assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
}

#[test]
fn refuses_infinite_and_text_positions_and_numeric_labels() {
    let inf = frame(&[("x", &[1.0, f32::INFINITY]), ("y", &[1.0, 2.0])]);
    assert!(matches!(run(inf, Aes::new().x("x").y("y"), Geom::tile()), Err(Error::ScaleDomain(_))));
    let mut txt = frame(&[("y", &[1.0, 2.0])]);
    txt.add_column_str("x", &["a", "b"]);
    assert!(matches!(run(txt, Aes::new().x("x").y("y"), Geom::tile()), Err(Error::Rendering(_))));
    let r = run(xy(), Aes::new().x("x").y("y").label("c"), Geom::text());
    assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
}

#[test]
fn refuses_empty_and_all_na() {
    let empty = frame(&[("x", &[]), ("y", &[])]);
    for g in [Geom::area(), Geom::tile()] {
        assert!(matches!(run(empty.clone(), Aes::new().x("x").y("y"), g), Err(Error::EmptyData)));
    }
    let all_na = frame(&[("x", &[f32::NAN]), ("y", &[f32::NAN])]);
    assert!(matches!(
        run(all_na.clone(), Aes::new().x("x").y("y"), Geom::area()),
        Err(Error::EmptyData)
    ));
    let r = run(all_na, Aes::new().x("x").y("y"), Geom::tile());
    assert!(matches!(r, Err(Error::ScaleDomain(_))), "{r:?}");
}

// --- robustness: no input panics ----------------------------------------------------------------

const EXTREME: [f32; 10] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    f32::MAX,
    f32::MIN,
    f32::MIN_POSITIVE,
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
];

fn all_geoms() -> Vec<(Geom, Aes)> {
    vec![
        (Geom::area(), Aes::new().x("a").y("b").group("g")),
        (Geom::area(), Aes::new().x("a").y("b")),
        (Geom::ribbon(), Aes::new().x("a").ymin("b").ymax("c").group("g")),
        (Geom::tile(), Aes::new().x("a").y("b")),
        (Geom::rect(), Aes::new().xmin("a").xmax("b").ymin("c").ymax("a")),
    ]
}

#[test]
fn no_panic_on_extreme_and_non_finite_values() {
    for a in EXTREME {
        for b in EXTREME {
            for c in EXTREME {
                let df = frame(&[
                    ("a", &[a, b, c, a]),
                    ("b", &[b, c, a, a]),
                    ("c", &[c, a, b, b]),
                    ("g", &[1.0, 1.0, 2.0, 2.0]),
                ]);
                for (geom, aes) in all_geoms() {
                    let _ = run(df.clone(), aes, geom);
                }
                let mut t = df.clone();
                t.add_column_str("lab", &["p", "q", "r", "s"]);
                let _ = run(t, Aes::new().x("a").y("b").label("lab"), Geom::text());
            }
        }
    }
}

#[test]
fn no_panic_on_empty_and_tiny_frames() {
    for n in 0..4 {
        let v: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let df = frame(&[("a", &v), ("b", &v), ("c", &v), ("g", &v)]);
        for (geom, aes) in all_geoms() {
            let _ = run(df.clone(), aes, geom);
        }
    }
    let _ = run(DataFrame::new(), Aes::new().x("a").y("b"), Geom::area());
}

#[test]
fn a_null_position_cell_is_an_na_like_nan() {
    // A missing cell and a NaN cell are both NA to ggplot2, so both lay out the same way.
    let full = wide();
    for (name, geom, aes) in five() {
        let positions: Vec<&str> = [&aes.x, &aes.y, &aes.ymin, &aes.ymax, &aes.xmin, &aes.xmax]
            .iter()
            .filter_map(|s| s.as_deref())
            .collect();
        for col in positions {
            let mut with_null = full.clone();
            with_null.add_column_values(
                col,
                vec![DataValue::Number(1.0), DataValue::Null, DataValue::Number(3.0)],
            );
            let mut with_nan = full.clone();
            with_nan.add_column_f32(col, &[1.0, f32::NAN, 3.0]);
            let a = run(with_null, aes.clone(), geom.clone());
            let b = run(with_nan, aes.clone(), geom.clone());
            // The raw `x`/`y` copies of a layer are not the marks; compare the marks and scales.
            let key = |r: &Result<BuiltLayout, Error>| match r {
                Ok(l) => format!("{:?} {:?} {:?}", l.layers[0].marks, l.x, l.y),
                Err(e) => format!("{e:?}"),
            };
            assert_eq!(key(&a), key(&b), "{name} {col}");
        }
    }
}

#[test]
fn refuses_a_missing_group_for_every_geom() {
    let full = wide();
    for (name, geom, aes) in five() {
        for bad in [DataValue::Null, DataValue::Number(f32::NAN), DataValue::Number(f32::INFINITY)]
        {
            let mut df = full.clone();
            df.add_column_values(
                "g",
                vec![DataValue::Number(1.0), bad.clone(), DataValue::Number(1.0)],
            );
            let r = run(df, aes.clone(), geom.clone());
            assert!(matches!(r, Err(Error::Rendering(_))), "{name} {bad:?}: {r:?}");
        }
    }
}

#[test]
fn numbers_sort_before_text_in_a_mixed_group_column() {
    // The cells are rows 1..3; the rank is of the sorted distinct value: 7, then "a", then "b".
    let mut df = wide();
    df.add_column_values(
        "g",
        vec![DataValue::Text("b".into()), DataValue::Number(7.0), DataValue::Text("a".into())],
    );
    let got =
        run(df, Aes::new().x("x").y("y").label("lab").group("g"), Geom::text()).expect("lays out");
    let m = got.layers[0].marks.as_ref().expect("marks");
    // Text layout keeps the row order: "b" is rank 3, 7 is rank 1, "a" is rank 2.
    assert_eq!(m.group, vec![3, 1, 2]);
}

#[test]
fn ribbon_refuses_the_flipped_form() {
    let r = run(wide(), Aes::new().y("y").xmin("x0").xmax("x1"), Geom::ribbon());
    assert!(matches!(r, Err(Error::Rendering(_))), "{r:?}");
}

// --- companions: the five beside each other and beside geom_point (G1, shared scales) -----------

/// The layer named `part` in `marks_companions/generate.R`: one of [`five`], `point` = `aes(px,
/// py)`, or `point_na` = `aes(qx, qy)` (an NA in each point column).
fn companion_layer(part: &str) -> super::Layer {
    use super::Layer;
    match part {
        "point" => Layer::new(Geom::point()).aes(Aes::new().x("px").y("py")),
        "point_na" => Layer::new(Geom::point()).aes(Aes::new().x("qx").y("qy")),
        name => {
            let (_, geom, aes) =
                five().into_iter().find(|(n, _, _)| *n == name).expect("one of the five");
            Layer::new(geom).aes(aes)
        }
    }
}

/// Lay `parts` out over `wide.csv` (no plot-level mapping, as in R) and compare every layer and
/// both panel scales with `ggplot_build()`.
fn run_companions(parts: &[&str], json: &str) {
    let want: Value = serde_json::from_str(json).expect("fixture is valid JSON");
    assert_eq!(want["versions"]["ggplot2"], "4.0.3", "fixture ggplot2");
    assert_eq!(want["collation"], "C", "fixture collation");
    assert_eq!(want["row"], "marks_companions");
    let mut p = GGPlot::new()
        .data(frame_from_csv(include_str!("../../fixtures/ggplot2/marks_companions/wide.csv")));
    for part in parts {
        p = p.layer(companion_layer(part));
    }
    let got = p.build().expect("plot builds").layout().expect("layout resolves");
    let layers = want["layers"].as_array().expect("layers");
    assert_eq!(got.layers.len(), layers.len(), "layer count");
    for (i, (part, layer)) in parts.iter().zip(layers).enumerate() {
        if part.starts_with("point") {
            assert_eq!(layer["geom"], "GeomPoint");
            // A built point layer keeps the complete rows only (`BuiltLayer`: ggplot2 keeps the
            // incomplete ones until draw time), so compare R's complete rows. The panel scales
            // below are trained on every non-missing cell, per axis, as in R.
            let (wx, wy) = (nums(&layer["x"]), nums(&layer["y"]));
            let complete: Vec<(f64, f64)> =
                wx.into_iter().zip(wy).filter(|(a, b)| !a.is_nan() && !b.is_nan()).collect();
            let got_rows: Vec<(f64, f64)> =
                got.layers[i].x.iter().copied().zip(got.layers[i].y.iter().copied()).collect();
            assert_eq!(got_rows.len(), complete.len(), "{part} rows");
            for ((gx, gy), (x, y)) in got_rows.iter().zip(&complete) {
                assert!(close(*gx, *x) && close(*gy, *y), "{part}: ({gx}, {gy}) vs ({x}, {y})");
            }
        } else {
            let (_, class) = row_geom(&format!("geom_{part}"));
            assert_eq!(layer["geom"], class, "layer {i}");
            check_layer(&got, i, layer);
        }
    }
    check_axis("x", &got.x, &want["panel"]["x"]);
    check_axis("y", &got.y, &want["panel"]["y"]);
}

mod companions {
    use super::run_companions;

    macro_rules! case {
        ($name:ident, [$($part:literal),+]) => {
            #[test]
            fn $name() {
                run_companions(
                    &[$($part),+],
                    include_str!(concat!(
                        "../../fixtures/ggplot2/marks_companions/expected_",
                        stringify!($name),
                        ".json"
                    )),
                );
            }
        };
    }

    case!(point_area, ["point", "area"]);
    case!(point_ribbon, ["point", "ribbon"]);
    case!(point_tile, ["point", "tile"]);
    case!(point_rect, ["point", "rect"]);
    case!(point_text, ["point", "text"]);
    case!(area_point, ["area", "point"]);
    case!(ribbon_point, ["ribbon", "point"]);
    case!(tile_point, ["tile", "point"]);
    case!(rect_point, ["rect", "point"]);
    case!(text_point, ["text", "point"]);
    case!(area_ribbon, ["area", "ribbon"]);
    case!(area_tile, ["area", "tile"]);
    case!(area_rect, ["area", "rect"]);
    case!(area_text, ["area", "text"]);
    case!(ribbon_tile, ["ribbon", "tile"]);
    case!(ribbon_rect, ["ribbon", "rect"]);
    case!(ribbon_text, ["ribbon", "text"]);
    case!(tile_rect, ["tile", "rect"]);
    case!(tile_text, ["tile", "text"]);
    case!(rect_text, ["rect", "text"]);
    case!(point_na_area, ["point_na", "area"]);
    case!(point_na_text, ["point_na", "text"]);
    case!(area_text_point, ["area", "text", "point"]);
    case!(rect_tile_ribbon, ["rect", "tile", "ribbon"]);
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
    for (name, geom, aes) in five() {
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
    for (name, geom, aes) in five() {
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
        // The identity stat is the point's default: the same layout as no stat at all.
        let [a, b] = both_orders(&geom, &aes, &point(Some(Stat::identity())));
        let [c, d] = both_orders(&geom, &aes, &point(None));
        assert!(a.is_ok() && b.is_ok(), "{name} + point identity: {a:?} {b:?}");
        assert_eq!(format!("{:?}", a.ok()), format!("{:?}", c.ok()), "{name} identity");
        assert_eq!(format!("{:?}", b.ok()), format!("{:?}", d.ok()), "{name} identity");
    }
}

#[test]
fn refuses_a_point_companion_layout_does_not_model_for_every_geom() {
    use super::Layer;
    let with = |extra: &[(&str, DataValue)]| {
        let mut df = wide();
        for (name, v) in extra {
            df.add_column_values(
                name,
                vec![DataValue::Number(1.0), v.clone(), DataValue::Number(3.0)],
            );
        }
        df
    };
    let point = |aes: Aes| Layer::new(Geom::point()).aes(aes);
    for (name, geom, aes) in five() {
        let marks = Layer::new(geom.clone()).aes(aes.clone());
        let lay = |df: DataFrame, plot_aes: Aes, companion: Layer| {
            let a = GGPlot::new()
                .data(df.clone())
                .aes(plot_aes.clone())
                .layer(marks.clone())
                .layer(companion.clone())
                .build()
                .and_then(|b| b.layout());
            let b = GGPlot::new()
                .data(df)
                .aes(plot_aes)
                .layer(companion)
                .layer(marks.clone())
                .build()
                .and_then(|b| b.layout());
            [a, b]
        };
        let rendering = |rs: [Result<BuiltLayout, Error>; 2], what: &str, needle: &str| {
            for r in rs {
                match &r {
                    Err(Error::Rendering(m)) if m.contains(needle) => {}
                    other => panic!("{name} + point {what}: {other:?}"),
                }
            }
        };
        // Unmapped x or y: ggplot2 refuses the point; nothing here may guess one.
        rendering(lay(wide(), Aes::new(), point(Aes::new().y("y"))), "no x", "mapped x");
        rendering(lay(wide(), Aes::new(), point(Aes::new().x("x"))), "no y", "mapped y");
        // A text column needs a discrete scale (ggplot2 4.0.3 makes x/y discrete).
        rendering(lay(wide(), Aes::new(), point(Aes::new().x("lab").y("y"))), "text x", "is text");
        rendering(lay(wide(), Aes::new(), point(Aes::new().x("x").y("lab"))), "text y", "is text");
        // A short column: R cannot build that frame.
        let mut short = wide();
        short.add_column_f32("s", &[1.0, 2.0]);
        rendering(
            lay(short.clone(), Aes::new(), point(Aes::new().x("s").y("y"))),
            "short x",
            "shorter",
        );
        rendering(lay(short, Aes::new(), point(Aes::new().x("x").y("s"))), "short y", "shorter");
        // Bounds the point inherits: ggplot2 keeps them in its data and trains the scales on them.
        for bound in ["xmin", "xmax", "ymin", "ymax"] {
            let plot_aes = match bound {
                "xmin" => Aes::new().xmin("x0"),
                "xmax" => Aes::new().xmax("x1"),
                "ymin" => Aes::new().ymin("lo"),
                _ => Aes::new().ymax("hi"),
            };
            rendering(
                lay(wide(), plot_aes, point(Aes::new().x("x").y("y"))),
                bound,
                &format!("mapped {bound}"),
            );
        }
        // A missing column.
        for r in lay(wide(), Aes::new(), point(Aes::new().x("nope").y("y"))) {
            assert!(matches!(&r, Err(Error::UnknownColumn(c)) if c == "nope"), "{name}: {r:?}");
        }
        // An infinite position: ggplot2 squishes it to the panel, which layout() does not model.
        for inf in [f32::INFINITY, f32::NEG_INFINITY] {
            for axis in ["x", "y"] {
                let df = with(&[("v", DataValue::Number(inf))]);
                let a =
                    if axis == "x" { Aes::new().x("v").y("y") } else { Aes::new().x("x").y("v") };
                for r in lay(df, Aes::new(), point(a)) {
                    assert!(matches!(r, Err(Error::ScaleDomain(_))), "{name} {axis} {inf}: {r:?}");
                }
            }
        }
        // The point's own frame is checked, not the plot's.
        let own = Layer::new(Geom::point())
            .data(frame(&[("x", &[1.0]), ("y", &[f32::INFINITY])]))
            .aes(Aes::new().x("x").y("y"));
        for r in lay(wide(), Aes::new(), own) {
            assert!(matches!(r, Err(Error::ScaleDomain(_))), "{name} own data: {r:?}");
        }
        // And an NA cell is not refused: the scale trains on the other rows (fixture point_na_*).
        for r in lay(with(&[("v", DataValue::Null)]), Aes::new(), point(Aes::new().x("v").y("y"))) {
            assert!(r.is_ok(), "{name} NA: {r:?}");
        }
    }
}

#[test]
fn refuses_a_layout_that_trains_one_axis_only() {
    use super::Layer;
    // An all-NA tile trains neither axis; a point with finite x and all-NA y trains x only. One
    // untrained axis is an empty scale, refused like two.
    let df = frame(&[
        ("x", &[f32::NAN, f32::NAN]),
        ("y", &[f32::NAN, f32::NAN]),
        ("px", &[1.0, 2.0]),
        ("py", &[f32::NAN, f32::NAN]),
    ]);
    let r = GGPlot::new()
        .data(df.clone())
        .layer(Layer::new(Geom::tile()).aes(Aes::new().x("x").y("y")))
        .layer(Layer::new(Geom::point()).aes(Aes::new().x("px").y("py")))
        .build()
        .and_then(|b| b.layout());
    assert!(matches!(r, Err(Error::ScaleDomain(_))), "x only: {r:?}");
    let r = GGPlot::new()
        .data(df)
        .layer(Layer::new(Geom::tile()).aes(Aes::new().x("x").y("y")))
        .layer(Layer::new(Geom::point()).aes(Aes::new().x("py").y("px")))
        .build()
        .and_then(|b| b.layout());
    assert!(matches!(r, Err(Error::ScaleDomain(_))), "y only: {r:?}");
}

#[test]
fn reports_a_companion_refusal_before_the_coord() {
    use super::{Coord, Layer};
    for (name, geom, aes) in five() {
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
    for (name, geom, aes) in five() {
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
