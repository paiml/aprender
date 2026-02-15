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
/// `Sharpe = (E[R] - Rf) / σ(R)`
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
/// `Sortino = (E[R] - Rf) / σ_d(R)`
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
/// `Treynor = (E[R] - Rf) / β`
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
/// `IR = (E[R] - E[Rb]) / σ(R - Rb)`
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
/// `α = E[R] - (Rf + β × (E[Rm] - Rf))`
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

include!("ratios_part_02.rs");
