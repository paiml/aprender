//! Risk-Adjusted Return Ratios
//!
//! Implements Sharpe, Sortino, Calmar, and other risk-adjusted performance metrics.
//!
//! References:
//! - Sharpe (1966), "Mutual Fund Performance"
//! - Sortino & van der Meer (1991), "Downside Risk"
//! - Young (1991), "Calmar Ratio"

/// Calculate Sharpe Ratio
///
/// Sharpe = (E[R] - Rf) / σ(R)
///
/// Measures excess return per unit of total risk.
///
/// # Arguments
/// * `returns` - Vector of period returns
/// * `risk_free_rate` - Risk-free rate for the same period
///
/// # Example
/// ```
/// use aprender::monte_carlo::risk::sharpe_ratio;
///
/// let returns = vec![0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.02, 0.01];
/// let sharpe = sharpe_ratio(&returns, 0.001);
/// assert!(sharpe.is_finite());
/// ```
#[must_use]
pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let excess_return = mean - risk_free_rate;

    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = variance.sqrt();

    if std > 1e-10 {
        excess_return / std
    } else if excess_return > 0.0 {
        f64::INFINITY
    } else if excess_return < 0.0 {
        f64::NEG_INFINITY
    } else {
        0.0
    }
}

/// Calculate annualized Sharpe Ratio
///
/// Assumes returns are at a given frequency and annualizes.
///
/// # Arguments
/// * `returns` - Vector of period returns
/// * `risk_free_rate` - Risk-free rate (annualized)
/// * `periods_per_year` - Number of periods per year (252 for daily, 12 for monthly)
#[must_use]
pub fn sharpe_ratio_annualized(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let annualized_return = mean * periods_per_year;
    let excess_return = annualized_return - risk_free_rate;

    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let annualized_vol = variance.sqrt() * periods_per_year.sqrt();

    if annualized_vol > 1e-10 {
        excess_return / annualized_vol
    } else if excess_return > 0.0 {
        f64::INFINITY
    } else if excess_return < 0.0 {
        f64::NEG_INFINITY
    } else {
        0.0
    }
}

/// Calculate Sortino Ratio
///
/// Sortino = (E[R] - Rf) / σ_d(R)
///
/// Uses downside deviation instead of total standard deviation.
/// Only penalizes negative volatility.
///
/// # Arguments
/// * `returns` - Vector of period returns
/// * `risk_free_rate` - Risk-free rate for the same period
/// * `target_return` - Minimum acceptable return (MAR)
///
/// # Example
/// ```
/// use aprender::monte_carlo::risk::sortino_ratio;
///
/// let returns = vec![0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.02, 0.01];
/// let sortino = sortino_ratio(&returns, 0.001, 0.0);
/// assert!(sortino.is_finite());
/// ```
#[must_use]
pub fn sortino_ratio(returns: &[f64], risk_free_rate: f64, target_return: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let excess_return = mean - risk_free_rate;

    // Downside deviation: only consider returns below target
    let downside_sq_sum: f64 = returns
        .iter()
        .filter(|&&r| r < target_return)
        .map(|&r| (r - target_return).powi(2))
        .sum();

    let downside_deviation = (downside_sq_sum / n).sqrt();

    if downside_deviation > 1e-10 {
        excess_return / downside_deviation
    } else if excess_return > 0.0 {
        f64::INFINITY
    } else if excess_return < 0.0 {
        f64::NEG_INFINITY
    } else {
        0.0
    }
}

/// Calculate Calmar Ratio
///
/// Calmar = Annualized Return / Maximum Drawdown
///
/// Measures return per unit of drawdown risk.
///
/// # Arguments
/// * `annualized_return` - Annualized return
/// * `max_drawdown` - Maximum drawdown (as positive fraction)
///
/// # Example
/// ```
/// use aprender::monte_carlo::risk::calmar_ratio;
///
/// let calmar = calmar_ratio(0.15, 0.10); // 15% return, 10% max DD
/// assert!((calmar - 1.5).abs() < 0.01);
/// ```
#[must_use]
pub fn calmar_ratio(annualized_return: f64, max_drawdown: f64) -> f64 {
    if max_drawdown > 1e-10 {
        annualized_return / max_drawdown
    } else if annualized_return > 0.0 {
        f64::INFINITY
    } else if annualized_return < 0.0 {
        f64::NEG_INFINITY
    } else {
        0.0
    }
}

/// Calculate Treynor Ratio
///
/// Treynor = (E[R] - Rf) / β
///
/// Measures excess return per unit of systematic risk (beta).
///
/// # Arguments
/// * `returns` - Vector of portfolio returns
/// * `benchmark_returns` - Vector of benchmark returns
/// * `risk_free_rate` - Risk-free rate
#[must_use]
pub fn treynor_ratio(returns: &[f64], benchmark_returns: &[f64], risk_free_rate: f64) -> f64 {
    let beta = calculate_beta(returns, benchmark_returns);

    if returns.is_empty() {
        return 0.0;
    }

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let excess_return = mean_return - risk_free_rate;

    if beta.abs() > 1e-10 {
        excess_return / beta
    } else {
        0.0
    }
}

/// Calculate Information Ratio
///
/// IR = (E[R] - E[Rb]) / σ(R - Rb)
///
/// Measures active return per unit of tracking error.
///
/// # Arguments
/// * `returns` - Vector of portfolio returns
/// * `benchmark_returns` - Vector of benchmark returns
#[must_use]
pub fn information_ratio(returns: &[f64], benchmark_returns: &[f64]) -> f64 {
    if returns.len() != benchmark_returns.len() || returns.len() < 2 {
        return 0.0;
    }

    let n = returns.len() as f64;

    // Active returns (excess over benchmark)
    let active_returns: Vec<f64> = returns
        .iter()
        .zip(benchmark_returns.iter())
        .map(|(r, b)| r - b)
        .collect();

    let mean_active = active_returns.iter().sum::<f64>() / n;

    let tracking_variance = active_returns
        .iter()
        .map(|r| (r - mean_active).powi(2))
        .sum::<f64>()
        / (n - 1.0);

    let tracking_error = tracking_variance.sqrt();

    if tracking_error > 1e-10 {
        mean_active / tracking_error
    } else {
        0.0
    }
}

/// Calculate Omega Ratio
///
/// Omega(L) = ∫_L^∞ (1-F(x))dx / ∫_{-∞}^L F(x)dx
///
/// Ratio of upside probability-weighted gains to downside probability-weighted losses.
///
/// # Arguments
/// * `returns` - Vector of returns
/// * `threshold` - Return threshold (minimum acceptable return)
#[must_use]
pub fn omega_ratio(returns: &[f64], threshold: f64) -> f64 {
    if returns.is_empty() {
        return 1.0;
    }

    let gains: f64 = returns
        .iter()
        .filter(|&&r| r > threshold)
        .map(|&r| r - threshold)
        .sum();

    let losses: f64 = returns
        .iter()
        .filter(|&&r| r < threshold)
        .map(|&r| threshold - r)
        .sum();

    if losses > 1e-10 {
        gains / losses
    } else if gains > 0.0 {
        f64::INFINITY
    } else {
        1.0
    }
}

/// Calculate Beta (systematic risk)
///
/// β = Cov(R, Rm) / Var(Rm)
fn calculate_beta(returns: &[f64], benchmark_returns: &[f64]) -> f64 {
    if returns.len() != benchmark_returns.len() || returns.len() < 2 {
        return 1.0;
    }

    let n = returns.len() as f64;
    let mean_r = returns.iter().sum::<f64>() / n;
    let mean_b = benchmark_returns.iter().sum::<f64>() / n;

    let covariance: f64 = returns
        .iter()
        .zip(benchmark_returns.iter())
        .map(|(r, b)| (r - mean_r) * (b - mean_b))
        .sum::<f64>()
        / (n - 1.0);

    let var_benchmark: f64 = benchmark_returns
        .iter()
        .map(|b| (b - mean_b).powi(2))
        .sum::<f64>()
        / (n - 1.0);

    if var_benchmark > 1e-10 {
        covariance / var_benchmark
    } else {
        1.0
    }
}

/// Calculate Alpha (Jensen's Alpha)
///
/// α = E[R] - (Rf + β × (E[Rm] - Rf))
#[must_use]
pub fn jensens_alpha(returns: &[f64], benchmark_returns: &[f64], risk_free_rate: f64) -> f64 {
    if returns.is_empty() || benchmark_returns.is_empty() {
        return 0.0;
    }

    let mean_r = returns.iter().sum::<f64>() / returns.len() as f64;
    let mean_b = benchmark_returns.iter().sum::<f64>() / benchmark_returns.len() as f64;
    let beta = calculate_beta(returns, benchmark_returns);

    mean_r - (risk_free_rate + beta * (mean_b - risk_free_rate))
}

/// Calculate Gain-to-Pain Ratio
///
/// Sum of positive returns / absolute sum of negative returns
#[must_use]
pub fn gain_to_pain_ratio(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let gains: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
    let losses: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();

    if losses > 1e-10 {
        gains / losses
    } else if gains > 0.0 {
        f64::INFINITY
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_returns() -> Vec<f64> {
        vec![
            0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.02, 0.01, 0.005, -0.015,
        ]
    }

    fn sample_benchmark() -> Vec<f64> {
        vec![
            0.008, 0.015, -0.005, 0.025, 0.01, -0.01, 0.018, 0.008, 0.003, -0.012,
        ]
    }

    #[test]
    fn test_sharpe_ratio_positive() {
        let returns = sample_returns();
        let sharpe = sharpe_ratio(&returns, 0.001);

        // Should be positive (positive mean return exceeds risk-free rate)
        assert!(sharpe > 0.0, "Sharpe should be positive: {sharpe}");
        assert!(sharpe.is_finite());
    }

    #[test]
    fn test_sharpe_ratio_zero_variance() {
        let returns = vec![0.01, 0.01, 0.01, 0.01, 0.01];
        let sharpe = sharpe_ratio(&returns, 0.001);

        // With zero variance and positive excess return, should be infinity
        assert!(
            sharpe.is_infinite() && sharpe > 0.0,
            "Sharpe with zero variance: {sharpe}"
        );
    }

    #[test]
    fn test_sharpe_ratio_annualized() {
        let returns = sample_returns();

        // Daily returns, annualize with 252 trading days
        let sharpe_daily = sharpe_ratio_annualized(&returns, 0.05, 252.0);
        let sharpe_monthly = sharpe_ratio_annualized(&returns, 0.05, 12.0);

        assert!(sharpe_daily.is_finite());
        assert!(sharpe_monthly.is_finite());
    }

    #[test]
    fn test_sortino_ratio() {
        let returns = sample_returns();
        let sortino = sortino_ratio(&returns, 0.001, 0.0);

        // Should be positive and typically higher than Sharpe
        assert!(sortino > 0.0, "Sortino should be positive: {sortino}");
        assert!(sortino.is_finite());
    }

    #[test]
    fn test_sortino_no_downside() {
        let returns = vec![0.01, 0.02, 0.03, 0.04, 0.05];
        let sortino = sortino_ratio(&returns, 0.0, 0.0);

        // No negative returns, downside deviation = 0, infinite Sortino
        assert!(
            sortino.is_infinite() && sortino > 0.0,
            "Sortino with no downside: {sortino}"
        );
    }

    #[test]
    fn test_calmar_ratio() {
        let calmar = calmar_ratio(0.15, 0.10);
        assert!((calmar - 1.5).abs() < 0.001);

        let calmar_high = calmar_ratio(0.30, 0.10);
        assert!((calmar_high - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_calmar_ratio_zero_drawdown() {
        let calmar = calmar_ratio(0.10, 0.0);
        assert!(calmar.is_infinite() && calmar > 0.0);
    }

    #[test]
    fn test_treynor_ratio() {
        let returns = sample_returns();
        let benchmark = sample_benchmark();
        let treynor = treynor_ratio(&returns, &benchmark, 0.001);

        assert!(treynor.is_finite());
    }

    #[test]
    fn test_information_ratio() {
        let returns = sample_returns();
        let benchmark = sample_benchmark();
        let ir = information_ratio(&returns, &benchmark);

        assert!(ir.is_finite());
    }

    #[test]
    fn test_omega_ratio() {
        let returns = sample_returns();
        let omega = omega_ratio(&returns, 0.0);

        // With positive mean, omega should be > 1
        assert!(omega > 0.0, "Omega should be positive: {omega}");
    }

    #[test]
    fn test_omega_ratio_threshold() {
        let returns = sample_returns();

        // Higher threshold should give lower omega
        let omega_0 = omega_ratio(&returns, 0.0);
        let omega_high = omega_ratio(&returns, 0.02);

        assert!(omega_high < omega_0, "Higher threshold = lower omega");
    }

    #[test]
    fn test_jensens_alpha() {
        let returns = sample_returns();
        let benchmark = sample_benchmark();
        let alpha = jensens_alpha(&returns, &benchmark, 0.001);

        assert!(alpha.is_finite());
    }

    #[test]
    fn test_gain_to_pain_ratio() {
        let returns = sample_returns();
        let gpr = gain_to_pain_ratio(&returns);

        assert!(gpr > 0.0);
        assert!(gpr.is_finite());
    }

    #[test]
    fn test_gain_to_pain_all_positive() {
        let returns = vec![0.01, 0.02, 0.03, 0.04];
        let gpr = gain_to_pain_ratio(&returns);

        assert!(gpr.is_infinite() && gpr > 0.0);
    }

    #[test]
    fn test_calculate_beta() {
        let returns = sample_returns();
        let benchmark = sample_benchmark();
        let beta = calculate_beta(&returns, &benchmark);

        // Portfolio with similar behavior to benchmark should have beta near 1
        assert!(
            beta > 0.0 && beta < 3.0,
            "Beta should be reasonable: {beta}"
        );
    }

    #[test]
    fn test_empty_inputs() {
        assert!(sharpe_ratio(&[], 0.0).abs() < 1e-10);
        assert!(sortino_ratio(&[], 0.0, 0.0).abs() < 1e-10);
        assert!(information_ratio(&[], &[]).abs() < 1e-10);
        assert!((omega_ratio(&[], 0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_value() {
        let returns = vec![0.01];
        assert!(sharpe_ratio(&returns, 0.0).abs() < 1e-10);
        assert!(sortino_ratio(&returns, 0.0, 0.0).abs() < 1e-10);
    }

    // Property-based tests
    #[cfg(test)]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_sharpe_finite(returns in prop::collection::vec(-0.5..0.5f64, 10..100)) {
                let sharpe = sharpe_ratio(&returns, 0.01);
                prop_assert!(sharpe.is_finite() || sharpe.is_infinite(), "Sharpe must be defined");
            }

            #[test]
            fn prop_sortino_geq_zero_or_negative(
                returns in prop::collection::vec(-0.5..0.5f64, 10..100)
            ) {
                let sortino = sortino_ratio(&returns, 0.0, 0.0);
                // Sortino can be negative if mean is negative
                prop_assert!(sortino.is_finite() || sortino.is_infinite());
            }

            #[test]
            fn prop_calmar_sign_matches_return(
                ret in -0.5..0.5f64,
                dd in 0.01..0.5f64,
            ) {
                let calmar = calmar_ratio(ret, dd);
                if ret > 0.0 {
                    prop_assert!(calmar > 0.0);
                } else if ret < 0.0 {
                    prop_assert!(calmar < 0.0);
                }
            }

            #[test]
            fn prop_omega_positive(
                returns in prop::collection::vec(-0.1..0.1f64, 10..100),
                threshold in -0.1..0.1f64,
            ) {
                let omega = omega_ratio(&returns, threshold);
                prop_assert!(omega >= 0.0 || omega.is_infinite(), "Omega must be non-negative");
            }

            #[test]
            fn prop_information_ratio_finite(
                returns in prop::collection::vec(-0.1..0.1f64, 20..100),
                benchmark in prop::collection::vec(-0.1..0.1f64, 20..100),
            ) {
                let len = returns.len().min(benchmark.len());
                let ir = information_ratio(&returns[..len], &benchmark[..len]);
                prop_assert!(ir.is_finite() || ir.abs() < 1e-10, "IR should be finite: {ir}");
            }
        }
    }
}
