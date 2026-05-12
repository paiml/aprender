// SHIP-TWO-001 — `arima-v1` algorithm-level PARTIAL discharge for
// FALSIFY-ARIMA-001..004 (closes 4/4 sweep).
//
// Contract: `contracts/arima-v1.yaml`.
// Spec: ARIMA(p, d, q) time series forecasting (Box & Jenkins 1970;
// Hamilton 1994 Time Series Analysis Ch. 3-5).

// ===========================================================================
// Helper — d-th order differencing (in-module reference impl)
// ===========================================================================

/// Δ^d y_t: applies d-order differencing iteratively.
/// Length: T - d for input of length T.
#[must_use]
pub fn difference(series: &[f32], d: u64) -> Option<Vec<f32>> {
    if series.is_empty() { return None; }
    if d as usize >= series.len() { return None; }
    let mut current: Vec<f32> = series.to_vec();
    for _ in 0..d {
        let mut next: Vec<f32> = Vec::with_capacity(current.len() - 1);
        for i in 1..current.len() {
            let v = current[i] - current[i - 1];
            if !v.is_finite() { return None; }
            next.push(v);
        }
        current = next;
    }
    Some(current)
}

/// Pure-Rust AR(p) forecast: y_hat_t = Σ_{i=1}^p phi_i * y_{t-i}.
/// Forecast n_periods steps ahead by recursively appending forecasts to history.
#[must_use]
pub fn ar_forecast(history: &[f32], phi: &[f32], n_periods: u64) -> Option<Vec<f32>> {
    if history.is_empty() || phi.is_empty() { return None; }
    if !history.iter().all(|v| v.is_finite()) { return None; }
    if !phi.iter().all(|v| v.is_finite()) { return None; }
    let p = phi.len();
    if history.len() < p { return None; }
    let mut working: Vec<f32> = history.to_vec();
    let mut out: Vec<f32> = Vec::with_capacity(n_periods as usize);
    for _ in 0..n_periods {
        let n = working.len();
        let mut acc = 0.0_f32;
        for i in 0..p {
            acc += phi[i] * working[n - 1 - i];
        }
        if !acc.is_finite() { return None; }
        out.push(acc);
        working.push(acc);
    }
    Some(out)
}

// ===========================================================================
// ARIMA-001 — Forecast length: |forecast(model, n_periods)| == n_periods
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arima001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_forecast_length(
    history: &[f32],
    phi: &[f32],
    n_periods: u64,
) -> Arima001Verdict {
    if n_periods == 0 { return Arima001Verdict::Fail; }
    match ar_forecast(history, phi, n_periods) {
        Some(forecasts) if forecasts.len() as u64 == n_periods => Arima001Verdict::Pass,
        _ => Arima001Verdict::Fail,
    }
}

// ===========================================================================
// ARIMA-002 — Forecast finiteness: all forecast values are finite
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arima002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_forecast_finiteness(
    history: &[f32],
    phi: &[f32],
    n_periods: u64,
) -> Arima002Verdict {
    if n_periods == 0 { return Arima002Verdict::Fail; }
    match ar_forecast(history, phi, n_periods) {
        Some(forecasts) if forecasts.iter().all(|v| v.is_finite()) => Arima002Verdict::Pass,
        _ => Arima002Verdict::Fail,
    }
}

// ===========================================================================
// ARIMA-003 — Differencing length: |Δ^d y| == |y| - d
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arima003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_differencing_length(series: &[f32], d: u64) -> Arima003Verdict {
    if series.is_empty() { return Arima003Verdict::Fail; }
    if d as usize >= series.len() { return Arima003Verdict::Fail; }
    match difference(series, d) {
        Some(diff) if diff.len() as u64 == series.len() as u64 - d => Arima003Verdict::Pass,
        _ => Arima003Verdict::Fail,
    }
}

// ===========================================================================
// ARIMA-004 — Forecast deterministic: same input → same output (byte-exact)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arima004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_forecast_deterministic(
    history: &[f32],
    phi: &[f32],
    n_periods: u64,
) -> Arima004Verdict {
    if n_periods == 0 { return Arima004Verdict::Fail; }
    let f1 = match ar_forecast(history, phi, n_periods) {
        Some(v) => v,
        None => return Arima004Verdict::Fail,
    };
    let f2 = match ar_forecast(history, phi, n_periods) {
        Some(v) => v,
        None => return Arima004Verdict::Fail,
    };
    if f1.len() != f2.len() { return Arima004Verdict::Fail; }
    for (a, b) in f1.iter().zip(f2.iter()) {
        if a.to_bits() != b.to_bits() { return Arima004Verdict::Fail; }
    }
    Arima004Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // ARIMA-001 (forecast length)
    #[test] fn arima001_pass_canonical() {
        // AR(2) forecast 5 steps ahead.
        let history = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let phi = vec![0.5_f32, 0.3];
        assert_eq!(
            verdict_from_forecast_length(&history, &phi, 5),
            Arima001Verdict::Pass
        );
    }
    #[test] fn arima001_pass_one_step() {
        let history = vec![1.0_f32, 2.0];
        let phi = vec![0.5_f32];
        assert_eq!(verdict_from_forecast_length(&history, &phi, 1), Arima001Verdict::Pass);
    }
    #[test] fn arima001_fail_zero_periods() {
        let history = vec![1.0_f32, 2.0];
        let phi = vec![0.5_f32];
        assert_eq!(verdict_from_forecast_length(&history, &phi, 0), Arima001Verdict::Fail);
    }
    #[test] fn arima001_fail_history_too_short() {
        // p = 5 but history only has 3 points.
        let history = vec![1.0_f32, 2.0, 3.0];
        let phi = vec![0.1_f32, 0.1, 0.1, 0.1, 0.1];
        assert_eq!(verdict_from_forecast_length(&history, &phi, 5), Arima001Verdict::Fail);
    }
    #[test] fn arima001_fail_empty_phi() {
        let history = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_forecast_length(&history, &[], 5), Arima001Verdict::Fail);
    }

    // ARIMA-002 (forecast finiteness)
    #[test] fn arima002_pass_stable() {
        // Stable AR(1): φ = 0.5 (|φ| < 1) on bounded series.
        let history = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let phi = vec![0.5_f32];
        assert_eq!(
            verdict_from_forecast_finiteness(&history, &phi, 10),
            Arima002Verdict::Pass
        );
    }
    #[test] fn arima002_pass_zero_phi() {
        // φ = 0 produces all-zero forecasts (still finite).
        let history = vec![1.0_f32];
        let phi = vec![0.0_f32];
        assert_eq!(
            verdict_from_forecast_finiteness(&history, &phi, 5),
            Arima002Verdict::Pass
        );
    }
    #[test] fn arima002_fail_unstable_diverges() {
        // Unstable AR(1): φ = 100 with positive history → forecasts grow
        // exponentially and overflow to inf within ~5 steps for large initial.
        let history = vec![1e30_f32];
        let phi = vec![100.0_f32];
        assert_eq!(
            verdict_from_forecast_finiteness(&history, &phi, 5),
            Arima002Verdict::Fail
        );
    }
    #[test] fn arima002_fail_nan_history() {
        let history = vec![1.0_f32, f32::NAN];
        let phi = vec![0.5_f32];
        assert_eq!(
            verdict_from_forecast_finiteness(&history, &phi, 5),
            Arima002Verdict::Fail
        );
    }

    // ARIMA-003 (differencing length)
    #[test] fn arima003_pass_d_1() {
        let series = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        // Δ^1 has length T - 1 = 4.
        assert_eq!(verdict_from_differencing_length(&series, 1), Arima003Verdict::Pass);
    }
    #[test] fn arima003_pass_d_2() {
        let series = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        // Δ^2 has length T - 2 = 3.
        assert_eq!(verdict_from_differencing_length(&series, 2), Arima003Verdict::Pass);
    }
    #[test] fn arima003_pass_d_0_identity() {
        let series = vec![1.0_f32, 2.0, 3.0];
        // Δ^0 is identity (length unchanged).
        assert_eq!(verdict_from_differencing_length(&series, 0), Arima003Verdict::Pass);
    }
    #[test] fn arima003_fail_d_too_large() {
        // d == series.len() leaves zero-length series; verdict rejects.
        let series = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(verdict_from_differencing_length(&series, 3), Arima003Verdict::Fail);
    }
    #[test] fn arima003_fail_empty() {
        assert_eq!(verdict_from_differencing_length(&[], 1), Arima003Verdict::Fail);
    }

    // ARIMA-004 (forecast determinism)
    #[test] fn arima004_pass_canonical() {
        let history = vec![1.0_f32, 2.0, 3.0, 4.0];
        let phi = vec![0.5_f32, 0.3];
        assert_eq!(
            verdict_from_forecast_deterministic(&history, &phi, 5),
            Arima004Verdict::Pass
        );
    }
    #[test] fn arima004_pass_long_horizon() {
        let history = vec![1.0_f32, 2.0];
        let phi = vec![0.5_f32];
        assert_eq!(
            verdict_from_forecast_deterministic(&history, &phi, 100),
            Arima004Verdict::Pass
        );
    }
    #[test] fn arima004_fail_zero_periods() {
        let history = vec![1.0_f32];
        let phi = vec![0.5_f32];
        assert_eq!(
            verdict_from_forecast_deterministic(&history, &phi, 0),
            Arima004Verdict::Fail
        );
    }

    // Helper sanity
    #[test] fn difference_d_1_canonical() {
        // Δ^1 [1, 2, 4, 8] = [1, 2, 4].
        let d = difference(&[1.0_f32, 2.0, 4.0, 8.0], 1).unwrap();
        assert_eq!(d, vec![1.0_f32, 2.0, 4.0]);
    }
    #[test] fn difference_d_2_canonical() {
        // Δ^2 [1, 2, 4, 8] = Δ[1, 2, 4] = [1, 2].
        let d = difference(&[1.0_f32, 2.0, 4.0, 8.0], 2).unwrap();
        assert_eq!(d, vec![1.0_f32, 2.0]);
    }
    #[test] fn ar_forecast_ar1_doubles() {
        // AR(1) with φ=2 doubles each step: [1] → [2, 4, 8, 16].
        let f = ar_forecast(&[1.0_f32], &[2.0], 4).unwrap();
        assert_eq!(f, vec![2.0_f32, 4.0, 8.0, 16.0]);
    }
}
