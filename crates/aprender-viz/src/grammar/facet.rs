//! Faceting for Grammar of Graphics.
//!
//! Creates small multiples by splitting data on one or more variables.
//!
//! # The partition
//!
//! [`panels`] is the reader for [`Facet`]: it turns a facet specification plus a frame into the
//! panel grid. It is **total** — every source row lands in exactly one panel — which is what makes
//! `Σ panel rows == input rows` an assertion rather than a hope. Totality needs care at two points:
//!
//! - A cell that is not text still has a level: a number formats to its shortest round-tripping
//!   form and [`DataValue::Null`] becomes `NA`, so no row is silently dropped for having the wrong
//!   type in the faceting column. (ggplot2 likewise gives `NA` its own panel.)
//! - A row shorter than `n_rows` (a ragged column) is padded with `NA` rather than truncated.
//!
//! # Panel order is lexicographic, and that is a decision
//!
//! Levels are sorted by their string form. Panel order decides layout, layout decides the output
//! bytes, and APEX-001 renders the same figure on two architectures and compares them with `cmp` —
//! so the order must be a total function of the *set* of levels, never of the order rows happened
//! to arrive in or of a `HashMap`'s iteration. Sorting gives that; first-appearance order does not,
//! because permuting the input rows would then change the image.
//!
//! The cost is that numeric levels sort as text: `"10"` precedes `"9"`. Faceting is a discrete
//! operation and this matches ggplot2's default for character vectors, but a caller who wants
//! `"Low" < "Medium" < "High"` cannot express it yet. An explicit level override belongs with a
//! real factor/categorical type in the data layer, which this crate does not have.

use super::data::{DataFrame, DataValue};
use crate::error::{Error, Result};

/// One panel of a faceted plot.
///
/// `rows` holds indices into the *source* frame rather than copied data: the panel is a view, the
/// frame stays the single owner, and the partition laws are checkable by counting indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Panel {
    /// The level of each faceting variable, in the order the [`Facet`] names them.
    ///
    /// Empty for [`Facet::None`], one entry for `Row`/`Col`/`Wrap`, two for `Grid` (row var first).
    pub levels: Vec<String>,
    /// Row of this panel in the panel grid, from 0.
    pub row: usize,
    /// Column of this panel in the panel grid, from 0.
    pub col: usize,
    /// Indices of the source frame's rows belonging to this panel, ascending.
    pub rows: Vec<usize>,
}

/// The level a cell contributes to, as a total function — every cell has one.
fn level_of(value: Option<&DataValue>) -> String {
    match value {
        Some(DataValue::Text(s)) => s.clone(),
        Some(DataValue::Number(n)) => format!("{n}"),
        Some(DataValue::Null) | None => "NA".to_string(),
    }
}

/// The level of every row of `var`, padded to `n_rows` so the result indexes by row.
fn levels_of_column(data: &DataFrame, var: &str) -> Result<Vec<String>> {
    let col = data
        .get(var)
        .ok_or_else(|| Error::UnknownColumn(format!("no column {var:?} to facet by")))?;
    Ok((0..data.nrow()).map(|i| level_of(col.get(i))).collect())
}

/// The distinct levels of `per_row`, lexicographically ordered.
fn distinct_sorted(per_row: &[String]) -> Vec<String> {
    let mut out: Vec<String> = per_row.to_vec();
    out.sort();
    out.dedup();
    out
}

/// Group row indices by level, in the order `levels` gives.
fn group_by_level(per_row: &[String], levels: &[String]) -> Vec<Vec<usize>> {
    let mut groups = vec![Vec::new(); levels.len()];
    for (i, lvl) in per_row.iter().enumerate() {
        // `levels` comes from `distinct_sorted(per_row)`, so the search always succeeds.
        if let Ok(slot) = levels.binary_search(lvl) {
            groups[slot].push(i);
        }
    }
    groups
}

/// One panel per level, placed by `place(index) -> (row, col)`.
fn panels_1d(
    data: &DataFrame,
    var: &str,
    place: impl Fn(usize) -> (usize, usize),
) -> Result<Vec<Panel>> {
    let per_row = levels_of_column(data, var)?;
    let levels = distinct_sorted(&per_row);
    let groups = group_by_level(&per_row, &levels);
    Ok(levels
        .into_iter()
        .zip(groups)
        .enumerate()
        .map(|(i, (lvl, rows))| {
            let (row, col) = place(i);
            Panel { levels: vec![lvl], row, col, rows }
        })
        .collect())
}

/// The full row × column grid, including cells no row falls into.
fn panels_grid(data: &DataFrame, row_var: &str, col_var: &str) -> Result<Vec<Panel>> {
    let per_row_r = levels_of_column(data, row_var)?;
    let per_row_c = levels_of_column(data, col_var)?;
    let row_levels = distinct_sorted(&per_row_r);
    let col_levels = distinct_sorted(&per_row_c);

    let mut out: Vec<Panel> = Vec::with_capacity(row_levels.len() * col_levels.len());
    for (r, rl) in row_levels.iter().enumerate() {
        for (c, cl) in col_levels.iter().enumerate() {
            out.push(Panel { levels: vec![rl.clone(), cl.clone()], row: r, col: c, rows: vec![] });
        }
    }
    for i in 0..data.nrow() {
        let (Ok(r), Ok(c)) =
            (row_levels.binary_search(&per_row_r[i]), col_levels.binary_search(&per_row_c[i]))
        else {
            continue; // unreachable: both level sets come from these very columns
        };
        out[r * col_levels.len() + c].rows.push(i);
    }
    Ok(out)
}

/// Partition `data` into panels according to `facet`.
///
/// Every row of `data` appears in exactly one panel, so `Σ panel rows == data.nrow()`. Panels are
/// ordered by level, lexicographically; see the module docs for why that order and not data order.
///
/// [`Facet::Grid`] returns the **full** cross product of the two variables' levels, so a
/// combination no row exhibits is present with an empty `rows` — that is what makes a facet grid a
/// grid. `Row`, `Col` and `Wrap` return one panel per observed level, never an empty one.
///
/// # Errors
///
/// [`Error::UnknownColumn`] if a faceting variable names a column the frame does not have. A
/// silently empty plot is the failure mode this replaces.
///
/// ```
/// use trueno_viz::grammar::{panels, DataFrame, Facet};
///
/// let mut df = DataFrame::new();
/// df.add_column_f32("y", &[1.0, 2.0, 3.0, 4.0]);
/// df.add_column_str("lineage", &["b", "a", "b", "c"]);
///
/// let ps = panels(&Facet::wrap("lineage", 2), &df).unwrap();
/// assert_eq!(ps.len(), 3); // three lineages, three panels
/// assert_eq!(ps[0].levels, vec!["a".to_string()]); // sorted, not first-seen
/// assert_eq!(ps[0].rows, vec![1]);
/// assert_eq!(ps.iter().map(|p| p.rows.len()).sum::<usize>(), df.nrow());
/// ```
pub fn panels(facet: &Facet, data: &DataFrame) -> Result<Vec<Panel>> {
    match facet {
        Facet::None => {
            Ok(vec![Panel { levels: vec![], row: 0, col: 0, rows: (0..data.nrow()).collect() }])
        }
        Facet::Row { var } => panels_1d(data, var, |i| (i, 0)),
        Facet::Col { var } => panels_1d(data, var, |i| (0, i)),
        Facet::Wrap { var, ncol } => {
            let n = (*ncol).max(1);
            panels_1d(data, var, move |i| (i / n, i % n))
        }
        Facet::Grid { row, col } => panels_grid(data, row, col),
    }
}

/// Faceting specification.
#[derive(Debug, Clone, Default)]
pub enum Facet {
    /// No faceting.
    #[default]
    None,
    /// Facet into a row of panels.
    Row {
        /// Column to facet by.
        var: String,
    },
    /// Facet into a column of panels.
    Col {
        /// Column to facet by.
        var: String,
    },
    /// Facet into a grid of panels.
    Grid {
        /// Row variable.
        row: String,
        /// Column variable.
        col: String,
    },
    /// Facet into wrapped panels.
    Wrap {
        /// Variable to facet by.
        var: String,
        /// Number of columns.
        ncol: usize,
    },
}

impl Facet {
    /// No faceting.
    #[must_use]
    pub fn none() -> Self {
        Facet::None
    }

    /// Facet into rows.
    #[must_use]
    pub fn row(var: &str) -> Self {
        Facet::Row { var: var.to_string() }
    }

    /// Facet into columns.
    #[must_use]
    pub fn col(var: &str) -> Self {
        Facet::Col { var: var.to_string() }
    }

    /// Facet into a grid.
    #[must_use]
    pub fn grid(row: &str, col: &str) -> Self {
        Facet::Grid { row: row.to_string(), col: col.to_string() }
    }

    /// Facet with wrapping.
    #[must_use]
    pub fn wrap(var: &str, ncol: usize) -> Self {
        Facet::Wrap { var: var.to_string(), ncol }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_facet_grid() {
        let f = Facet::grid("category", "year");
        match f {
            Facet::Grid { row, col } => {
                assert_eq!(row, "category");
                assert_eq!(col, "year");
            }
            _ => panic!("Expected Grid"),
        }
    }

    #[test]
    fn test_facet_wrap() {
        let f = Facet::wrap("category", 3);
        match f {
            Facet::Wrap { var, ncol } => {
                assert_eq!(var, "category");
                assert_eq!(ncol, 3);
            }
            _ => panic!("Expected Wrap"),
        }
    }

    #[test]
    fn test_facet_none() {
        let f = Facet::none();
        assert!(matches!(f, Facet::None));
    }

    #[test]
    fn test_facet_row() {
        let f = Facet::row("group");
        match f {
            Facet::Row { var } => assert_eq!(var, "group"),
            _ => panic!("Expected Row"),
        }
    }

    #[test]
    fn test_facet_col() {
        let f = Facet::col("region");
        match f {
            Facet::Col { var } => assert_eq!(var, "region"),
            _ => panic!("Expected Col"),
        }
    }

    #[test]
    fn test_facet_default() {
        let f = Facet::default();
        assert!(matches!(f, Facet::None));
    }

    #[test]
    fn test_facet_debug_clone() {
        let facets = vec![
            Facet::none(),
            Facet::row("a"),
            Facet::col("b"),
            Facet::grid("a", "b"),
            Facet::wrap("c", 2),
        ];
        for f in facets {
            let f2 = f.clone();
            let _ = format!("{f2:?}");
        }
    }
}
