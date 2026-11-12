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

fn rust_columns(layout: &BuiltLayout) -> (usize, BTreeMap<String, Col>) {
    let m = layout.layers[0].marks.as_ref().expect("a marks layer");
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
    let (n, cols) = rust_columns(got);
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
    check_axis("x", &got.x, &want["panel"]["x"]);
    check_axis("y", &got.y, &want["panel"]["y"]);
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

#[test]
fn refuses_non_default_coord() {
    use super::Coord;
    for coord in [Coord::cartesian().xlim(0.0, 2.0), Coord::cartesian().flip(), Coord::polar()] {
        let r = GGPlot::new()
            .data(xy())
            .aes(Aes::new().x("x").y("y"))
            .geom(Geom::tile())
            .coord(coord)
            .build()
            .and_then(|b| b.layout());
        assert!(matches!(r, Err(Error::UnsupportedCoord(_))), "{r:?}");
    }
}

#[test]
fn refuses_facets() {
    use super::Facet;
    let r = GGPlot::new()
        .data(xy())
        .aes(Aes::new().x("x").y("y"))
        .geom(Geom::area())
        .facet(Facet::wrap("c", 2))
        .build()
        .and_then(|b| b.layout());
    assert!(matches!(r, Err(Error::UnsupportedCoord(_))), "{r:?}");
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
