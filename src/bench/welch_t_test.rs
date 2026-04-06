
/// Perform Welch's t-test to compare two sample means
///
/// Welch's t-test is used when samples may have unequal variances.
/// Returns statistical significance information.
///
/// # Arguments
/// * `sample_a` - First sample
/// * `sample_b` - Second sample
/// * `alpha` - Significance level (e.g., 0.05 for 95% confidence)
///
/// # Example
/// ```
/// use realizar::bench::welch_t_test;
///
/// let a = vec![10.0, 11.0, 10.5, 10.2, 10.8];
/// let b = vec![20.0, 21.0, 20.5, 20.2, 20.8];
/// let result = welch_t_test(&a, &b, 0.05);
/// assert!(result.significant); // Clearly different means
/// ```
pub fn welch_t_test(sample_a: &[f64], sample_b: &[f64], alpha: f64) -> WelchTTestResult {
    let n1 = sample_a.len() as f64;
    let n2 = sample_b.len() as f64;

    // Calculate means
    let mean1 = sample_a.iter().sum::<f64>() / n1;
    let mean2 = sample_b.iter().sum::<f64>() / n2;

    // Calculate sample variances (using n-1 for unbiased estimator)
    let var1 = if n1 > 1.0 {
        sample_a.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0)
    } else {
        0.0
    };
    let var2 = if n2 > 1.0 {
        sample_b.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0)
    } else {
        0.0
    };

    // Handle zero variance case
    let se1 = var1 / n1;
    let se2 = var2 / n2;
    let se_diff = (se1 + se2).sqrt();

    if se_diff < f64::EPSILON {
        // Both samples have zero variance - cannot compute t-statistic
        return WelchTTestResult {
            t_statistic: 0.0,
            degrees_of_freedom: n1 + n2 - 2.0,
            p_value: 1.0,
            significant: false,
        };
    }

    // Calculate t-statistic
    let t_stat = (mean1 - mean2) / se_diff;

    // Welch-Satterthwaite degrees of freedom
    let df_num = (se1 + se2).powi(2);
    let df_denom = if n1 > 1.0 && se1 > f64::EPSILON {
        se1.powi(2) / (n1 - 1.0)
    } else {
        0.0
    } + if n2 > 1.0 && se2 > f64::EPSILON {
        se2.powi(2) / (n2 - 1.0)
    } else {
        0.0
    };

    let df = if df_denom > f64::EPSILON {
        df_num / df_denom
    } else {
        n1 + n2 - 2.0
    };

    // Approximate p-value using normal distribution for large df
    // For small df, we use a more conservative approximation
    let p_value = approximate_t_pvalue(t_stat.abs(), df);

    WelchTTestResult {
        t_statistic: t_stat,
        degrees_of_freedom: df,
        p_value,
        significant: p_value < alpha,
    }
}

/// Approximate two-tailed p-value from t-distribution
///
/// Uses normal approximation for large df, conservative approximation for small df
fn approximate_t_pvalue(t_abs: f64, df: f64) -> f64 {
    // For very large df, use normal approximation
    if df > 100.0 {
        // Use error function approximation for normal CDF
        let z = t_abs;
        let p = erfc_approx(z / std::f64::consts::SQRT_2);
        return p;
    }

    // For smaller df, use a polynomial approximation of t-distribution CDF
    // Based on Abramowitz and Stegun approximation
    let ratio = df / (df + t_abs * t_abs);
    incomplete_beta_approx(ratio, df / 2.0, 0.5)
}

/// Approximate complementary error function
fn erfc_approx(x: f64) -> f64 {
    // Horner form coefficients for erfc approximation
    // From Abramowitz and Stegun, formula 7.1.26
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591_1;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    if sign < 0.0 {
        2.0 - y
    } else {
        y
    }
}

/// Approximate incomplete beta function (simplified for t-test)
fn incomplete_beta_approx(x: f64, a: f64, b: f64) -> f64 {
    // Use continued fraction expansion for better accuracy
    // Simplified approximation suitable for t-distribution p-values
    if x < (a + 1.0) / (a + b + 2.0) {
        let beta_factor =
            gamma_ln(a + b) - gamma_ln(a) - gamma_ln(b) + a * x.ln() + b * (1.0 - x).ln();
        let beta_factor = beta_factor.exp();
        beta_factor * cf_beta(x, a, b) / a
    } else {
        1.0 - incomplete_beta_approx(1.0 - x, b, a)
    }
}

/// Continued fraction for incomplete beta
#[allow(clippy::many_single_char_names)] // Standard math notation for beta function
fn cf_beta(x: f64, a: f64, b: f64) -> f64 {
    let max_iter = 100;
    let eps = 1e-10;
    let tiny = 1e-30;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=max_iter {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;

        // Even step
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < eps {
            break;
        }
    }

    h
}

/// Approximate log-gamma function (Stirling's approximation)
#[allow(clippy::excessive_precision)] // Lanczos coefficients require high precision
fn gamma_ln(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }

    // Lanczos approximation coefficients
    let g = 7.0;
    let c = [
        0.999_999_999_999_81,
        676.520_368_121_885,
        -1_259.139_216_722_403,
        771.323_428_777_653,
        -176.615_029_162_141,
        12.507_343_278_687,
        -0.138_571_095_265_72,
        9.984_369_578_02e-6,
        1.505_632_735_15e-7,
    ];

    let x = x - 1.0;
    let mut sum = c[0];
    for (i, &coef) in c.iter().enumerate().skip(1) {
        sum += coef / (x + i as f64);
    }

    let t = x + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}
