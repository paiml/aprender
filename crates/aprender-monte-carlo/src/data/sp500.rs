//! Embedded S&P 500 historical data
//!
//! Contains monthly returns from 1928 to present, compiled from:
//! - Robert Shiller's online data
//! - Yahoo Finance historical prices
//!
//! Data includes both nominal and inflation-adjusted (real) returns.

/// S&P 500 historical data container
#[derive(Debug, Clone)]
pub struct Sp500Data {
    /// Monthly returns (nominal)
    nominal_returns: Vec<f64>,
    /// Monthly returns (real, inflation-adjusted)
    real_returns: Vec<f64>,
    /// Start year
    start_year: u32,
    /// End year
    end_year: u32,
}

/// Time period for filtering S&P 500 data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sp500Period {
    /// All available data (1928-present)
    All,
    /// Post-war era (1946-present)
    PostWar,
    /// Modern era (1970-present)
    Modern,
    /// Recent decades (1990-present)
    Recent,
    /// Custom range
    Custom {
        /// Start year
        start: u32,
        /// End year
        end: u32,
    },
}

impl Sp500Data {
    /// Load embedded S&P 500 data
    ///
    /// # Returns
    /// Historical monthly returns from 1928 to present
    #[must_use]
    pub fn load() -> Self {
        Self {
            nominal_returns: NOMINAL_MONTHLY_RETURNS.to_vec(),
            real_returns: REAL_MONTHLY_RETURNS.to_vec(),
            start_year: 1928,
            end_year: 2024,
        }
    }

    /// Get monthly returns for the specified period
    ///
    /// # Arguments
    /// * `period` - Time period to filter
    /// * `real` - If true, return inflation-adjusted returns
    #[must_use]
    pub fn monthly_returns(&self, period: Sp500Period, real: bool) -> Vec<f64> {
        let base = if real {
            &self.real_returns
        } else {
            &self.nominal_returns
        };

        match period {
            Sp500Period::All => base.clone(),
            Sp500Period::PostWar => {
                // 1946 starts at month (1946-1928)*12 = 216
                let start = 216;
                base[start..].to_vec()
            }
            Sp500Period::Modern => {
                // 1970 starts at month (1970-1928)*12 = 504
                let start = 504;
                base[start..].to_vec()
            }
            Sp500Period::Recent => {
                // 1990 starts at month (1990-1928)*12 = 744
                let start = 744;
                base[start..].to_vec()
            }
            Sp500Period::Custom { start, end } => {
                let start_idx = ((start.saturating_sub(self.start_year)) * 12) as usize;
                let end_idx = ((end.saturating_sub(self.start_year) + 1) * 12) as usize;
                let end_idx = end_idx.min(base.len());
                if start_idx >= base.len() {
                    return Vec::new();
                }
                base[start_idx..end_idx].to_vec()
            }
        }
    }

    /// Get all nominal monthly returns
    #[must_use]
    pub fn all_nominal(&self) -> &[f64] {
        &self.nominal_returns
    }

    /// Get all real (inflation-adjusted) monthly returns
    #[must_use]
    pub fn all_real(&self) -> &[f64] {
        &self.real_returns
    }

    /// Number of months of data
    #[must_use]
    pub fn len(&self) -> usize {
        self.nominal_returns.len()
    }

    /// Get the year range of the data
    #[must_use]
    pub fn year_range(&self) -> (u32, u32) {
        (self.start_year, self.end_year)
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nominal_returns.is_empty()
    }

    /// Calculate summary statistics
    #[must_use]
    pub fn statistics(&self, period: Sp500Period, real: bool) -> Sp500Stats {
        let returns = self.monthly_returns(period, real);
        if returns.is_empty() {
            return Sp500Stats::default();
        }

        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std = variance.sqrt();

        // Annualized metrics
        let annual_mean = (1.0 + mean).powf(12.0) - 1.0;
        let annual_std = std * 12.0_f64.sqrt();

        // Worst month
        let worst = returns.iter().copied().fold(f64::INFINITY, f64::min);
        let best = returns.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        // Calculate max drawdown
        let mut peak = 1.0;
        let mut max_dd = 0.0;
        let mut value = 1.0;

        for &ret in &returns {
            value *= 1.0 + ret;
            if value > peak {
                peak = value;
            }
            let dd = (peak - value) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        Sp500Stats {
            n_months: returns.len(),
            monthly_mean: mean,
            monthly_std: std,
            annual_mean,
            annual_std,
            worst_month: worst,
            best_month: best,
            max_drawdown: max_dd,
        }
    }

    /// Get decade-by-decade statistics
    #[must_use]
    pub fn decade_stats(&self, real: bool) -> Vec<(String, Sp500Stats)> {
        let decades = [
            ("1930s", 1930, 1939),
            ("1940s", 1940, 1949),
            ("1950s", 1950, 1959),
            ("1960s", 1960, 1969),
            ("1970s", 1970, 1979),
            ("1980s", 1980, 1989),
            ("1990s", 1990, 1999),
            ("2000s", 2000, 2009),
            ("2010s", 2010, 2019),
            ("2020s", 2020, 2024),
        ];

        decades
            .iter()
            .map(|(label, start, end)| {
                let period = Sp500Period::Custom {
                    start: *start,
                    end: *end,
                };
                (label.to_string(), self.statistics(period, real))
            })
            .collect()
    }
}

/// Summary statistics for S&P 500 data
#[derive(Debug, Clone, Default)]
pub struct Sp500Stats {
    /// Number of months
    pub n_months: usize,
    /// Monthly mean return
    pub monthly_mean: f64,
    /// Monthly standard deviation
    pub monthly_std: f64,
    /// Annualized mean return
    pub annual_mean: f64,
    /// Annualized standard deviation
    pub annual_std: f64,
    /// Worst monthly return
    pub worst_month: f64,
    /// Best monthly return
    pub best_month: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
}

include!("sp500_display_and_returns.rs");
