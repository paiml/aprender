//! Faceting for Grammar of Graphics.
//!
//! Creates small multiples by splitting data on one or more variables.

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
