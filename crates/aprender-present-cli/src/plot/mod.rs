//! `plot` verb, phase 1: the two input readers (PMAT-6917, PMAT-6918).
//!
//! * [`spec`] parses ggplot2 text syntax into a typed [`spec::PlotSpec`].
//! * [`csv`] reads a CSV table (with `#` comment lines) into typed columns.
//!
//! Pure Rust, no rendering. Phase 2 wires these to the `aprender-present plot` verb.

pub mod csv;
pub mod spec;

use std::fmt;

/// A spec or a table was rejected; each side keeps its own positioned error.
#[derive(Debug, Clone, PartialEq)]
pub enum PlotError {
    Spec(spec::SpecError),
    Csv(csv::CsvError),
}

impl fmt::Display for PlotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spec(e) => write!(f, "plot spec: {e}"),
            Self::Csv(e) => write!(f, "data table: {e}"),
        }
    }
}

impl std::error::Error for PlotError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Spec(e) => Some(e),
            Self::Csv(e) => Some(e),
        }
    }
}

impl From<spec::SpecError> for PlotError {
    fn from(e: spec::SpecError) -> Self {
        Self::Spec(e)
    }
}

impl From<csv::CsvError> for PlotError {
    fn from(e: csv::CsvError) -> Self {
        Self::Csv(e)
    }
}

/// Parse the spec text and read the table text; the first failure wins.
pub fn load(spec_text: &str, csv_text: &str) -> Result<(spec::PlotSpec, csv::Table), PlotError> {
    Ok((spec::parse(spec_text)?, csv::read_table(csv_text)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn load_returns_both_halves() {
        let (spec, table) =
            load("ggplot(d, aes(x = a)) + geom_point()", "# c\na\n1\n2\n").expect("loads");
        assert_eq!(spec.layers.len(), 1);
        assert_eq!(table.row_count(), 2);
    }

    #[test]
    fn load_names_which_side_failed_and_where() {
        let e = load("ggplot(d) + geom_histogram()", "a\n1\n").expect_err("spec side fails");
        assert!(
            matches!(e, PlotError::Spec(spec::SpecError::UnknownCall { .. })),
            "{e}"
        );
        let msg = e.to_string();
        assert!(
            msg.starts_with("plot spec: ")
                && msg.contains("geom_histogram")
                && msg.contains("line 1, column 14"),
            "{msg}"
        );
        assert!(e.source().is_some());

        let e = load("ggplot(d)", "a,b\n1\n").expect_err("data side fails");
        assert_eq!(
            e,
            PlotError::Csv(csv::CsvError::Ragged {
                line: 2,
                expected: 2,
                got: 1
            })
        );
        let msg = e.to_string();
        assert!(
            msg.starts_with("data table: ") && msg.contains("line 2"),
            "{msg}"
        );
        assert!(e.source().is_some());
    }

    #[test]
    fn the_spec_side_is_checked_first() {
        let e = load("", "").expect_err("both sides are bad");
        assert!(matches!(e, PlotError::Spec(_)), "{e}");
    }
}
