//! Statistical moments and jitter calculations.

/// Calculate jitter (inter-packet delay variation)
pub(crate) fn calculate_jitter(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }

    let diffs: Vec<f64> = samples.windows(2).map(|w| (w[1] - w[0]).abs()).collect();

    if diffs.is_empty() {
        return 0.0;
    }

    let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let variance = diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f64>() / diffs.len() as f64;

    variance.sqrt()
}

/// Calculate skewness and kurtosis
pub(crate) fn calculate_moments(samples: &[f64], mean: f64, std_dev: f64) -> (f64, f64) {
    if samples.len() < 4 || std_dev == 0.0 {
        return (0.0, 3.0);
    }

    let n = samples.len() as f64;

    let mut m3 = 0.0;
    let mut m4 = 0.0;

    for &x in samples {
        let z = (x - mean) / std_dev;
        m3 += z.powi(3);
        m4 += z.powi(4);
    }

    let skewness = (n / ((n - 1.0) * (n - 2.0))) * m3;

    let kurtosis = ((n * (n + 1.0)) / ((n - 1.0) * (n - 2.0) * (n - 3.0))) * m4
        - (3.0 * (n - 1.0).powi(2)) / ((n - 2.0) * (n - 3.0));

    (skewness, kurtosis + 3.0)
}
