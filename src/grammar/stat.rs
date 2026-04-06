//! Statistical transformations for Grammar of Graphics.
//!
//! Transforms data before visualization.

/// Statistical transformation type.
#[derive(Debug, Clone)]
pub enum Stat {
    /// No transformation (identity).
    Identity,
    /// Count occurrences.
    Count,
    /// Bin continuous data.
    Bin {
        /// Number of bins.
        bins: usize,
    },
    /// Compute density estimation.
    Density,
    /// Compute summary statistics (for boxplot).
    Boxplot,
    /// Fit a smooth curve.
    Smooth,
    /// Compute 2D binning (for tile/heatmap).
    Bin2d {
        /// Number of x bins.
        bins_x: usize,
        /// Number of y bins.
        bins_y: usize,
    },
    /// Sum values.
    Sum,
    /// Mean values.
    Mean,
}

impl Stat {
    /// Create an identity stat (no transformation).
    #[must_use]
    pub fn identity() -> Self {
        Stat::Identity
    }

    /// Create a count stat.
    #[must_use]
    pub fn count() -> Self {
        Stat::Count
    }

    /// Create a binning stat.
    #[must_use]
    pub fn bin(bins: usize) -> Self {
        Stat::Bin { bins }
    }

    /// Create a density estimation stat.
    #[must_use]
    pub fn density() -> Self {
        Stat::Density
    }

    /// Create a boxplot stat.
    #[must_use]
    pub fn boxplot() -> Self {
        Stat::Boxplot
    }

    /// Create a smooth stat.
    #[must_use]
    pub fn smooth() -> Self {
        Stat::Smooth
    }

    /// Create a 2D binning stat.
    #[must_use]
    pub fn bin2d(bins_x: usize, bins_y: usize) -> Self {
        Stat::Bin2d { bins_x, bins_y }
    }

    /// Create a sum stat.
    #[must_use]
    pub fn sum() -> Self {
        Stat::Sum
    }

    /// Create a mean stat.
    #[must_use]
    pub fn mean() -> Self {
        Stat::Mean
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stat_bin() {
        let stat = Stat::bin(20);
        match stat {
            Stat::Bin { bins } => assert_eq!(bins, 20),
            _ => panic!("Expected Bin stat"),
        }
    }

    #[test]
    fn test_stat_bin2d() {
        let stat = Stat::bin2d(10, 15);
        match stat {
            Stat::Bin2d { bins_x, bins_y } => {
                assert_eq!(bins_x, 10);
                assert_eq!(bins_y, 15);
            }
            _ => panic!("Expected Bin2d stat"),
        }
    }

    #[test]
    fn test_stat_identity() {
        assert!(matches!(Stat::identity(), Stat::Identity));
    }

    #[test]
    fn test_stat_count() {
        assert!(matches!(Stat::count(), Stat::Count));
    }

    #[test]
    fn test_stat_density() {
        assert!(matches!(Stat::density(), Stat::Density));
    }

    #[test]
    fn test_stat_boxplot() {
        assert!(matches!(Stat::boxplot(), Stat::Boxplot));
    }

    #[test]
    fn test_stat_smooth() {
        assert!(matches!(Stat::smooth(), Stat::Smooth));
    }

    #[test]
    fn test_stat_sum() {
        assert!(matches!(Stat::sum(), Stat::Sum));
    }

    #[test]
    fn test_stat_mean() {
        assert!(matches!(Stat::mean(), Stat::Mean));
    }

    #[test]
    fn test_stat_debug() {
        // Verify Debug impl works for all variants
        let variants: Vec<Stat> = vec![
            Stat::identity(),
            Stat::count(),
            Stat::bin(10),
            Stat::density(),
            Stat::boxplot(),
            Stat::smooth(),
            Stat::bin2d(5, 5),
            Stat::sum(),
            Stat::mean(),
        ];
        for v in variants {
            let _ = format!("{v:?}");
        }
    }

    #[test]
    fn test_stat_clone() {
        let s1 = Stat::bin(42);
        let s2 = s1.clone();
        match s2 {
            Stat::Bin { bins } => assert_eq!(bins, 42),
            _ => panic!("Clone failed"),
        }
    }
}
