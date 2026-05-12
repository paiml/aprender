// SHIP-TWO-001 — `metrics-regression-v1` algorithm-level PARTIAL
// discharge for FALSIFY-RM-001..007 (closes 7/7 sweep).
//
// Contract: `contracts/metrics-regression-v1.yaml`.

// ===========================================================================
// Reference regression metrics
// ===========================================================================

#[must_use]
pub fn mse(y: &[f32], y_hat: &[f32]) -> Option<f64> {
    if y.is_empty() || y.len() != y_hat.len() { return None; }
    if y.iter().chain(y_hat.iter()).any(|v| !v.is_finite()) { return None; }
    let n = y.len() as f64;
    let s: f64 = y.iter().zip(y_hat).map(|(a, b)| ((*a as f64) - (*b as f64)).powi(2)).sum();
    Some(s / n)
}

#[must_use]
pub fn mae(y: &[f32], y_hat: &[f32]) -> Option<f64> {
    if y.is_empty() || y.len() != y_hat.len() { return None; }
    if y.iter().chain(y_hat.iter()).any(|v| !v.is_finite()) { return None; }
    let n = y.len() as f64;
    let s: f64 = y.iter().zip(y_hat).map(|(a, b)| ((*a as f64) - (*b as f64)).abs()).sum();
    Some(s / n)
}

#[must_use]
pub fn rmse(y: &[f32], y_hat: &[f32]) -> Option<f64> {
    mse(y, y_hat).map(f64::sqrt)
}

#[must_use]
pub fn r_squared(y: &[f32], y_hat: &[f32]) -> Option<f64> {
    if y.is_empty() || y.len() != y_hat.len() { return None; }
    if y.iter().chain(y_hat.iter()).any(|v| !v.is_finite()) { return None; }
    let n = y.len() as f64;
    let mean = y.iter().map(|v| *v as f64).sum::<f64>() / n;
    let ss_tot: f64 = y.iter().map(|v| ((*v as f64) - mean).powi(2)).sum();
    let ss_res: f64 = y.iter().zip(y_hat).map(|(a, b)| ((*a as f64) - (*b as f64)).powi(2)).sum();
    if ss_tot == 0.0 {
        // Degenerate: y is constant. Conventionally R² = 1 if ŷ == y; else 0.
        return Some(if ss_res == 0.0 { 1.0 } else { 0.0 });
    }
    Some(1.0 - ss_res / ss_tot)
}

// ===========================================================================
// RM-001 — R² <= 1.0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rm001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_r2_upper_bound(y: &[f32], y_hat: &[f32]) -> Rm001Verdict {
    match r_squared(y, y_hat) {
        Some(r) if r.is_finite() && r <= 1.0 + 1e-9 => Rm001Verdict::Pass,
        _ => Rm001Verdict::Fail,
    }
}

// ===========================================================================
// RM-002 — MSE >= 0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rm002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_mse_nonneg(y: &[f32], y_hat: &[f32]) -> Rm002Verdict {
    match mse(y, y_hat) {
        Some(m) if m.is_finite() && m >= -1e-12 => Rm002Verdict::Pass,
        _ => Rm002Verdict::Fail,
    }
}

// ===========================================================================
// RM-003 — MAE <= RMSE (Jensen's inequality)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rm003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_mae_le_rmse(y: &[f32], y_hat: &[f32]) -> Rm003Verdict {
    let m = match mae(y, y_hat) { Some(v) => v, None => return Rm003Verdict::Fail };
    let r = match rmse(y, y_hat) { Some(v) => v, None => return Rm003Verdict::Fail };
    if m <= r + 1e-9 { Rm003Verdict::Pass } else { Rm003Verdict::Fail }
}

// ===========================================================================
// RM-004 — Perfect prediction: ŷ = y ⇒ R²=1, MSE=0, MAE=0, RMSE=0
// ===========================================================================

pub const AC_RM_004_TOLERANCE: f64 = 1e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rm004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_perfect_prediction(y: &[f32]) -> Rm004Verdict {
    if y.len() < 2 { return Rm004Verdict::Fail; }
    if y.iter().any(|v| !v.is_finite()) { return Rm004Verdict::Fail; }
    // Check non-constant — for constant y, ss_tot == 0 makes R² conventional.
    let first = y[0];
    let all_constant = y.iter().all(|v| (*v - first).abs() < f32::EPSILON);
    if all_constant { return Rm004Verdict::Fail; }
    let m = mse(y, y).unwrap_or(f64::NAN);
    let a = mae(y, y).unwrap_or(f64::NAN);
    let rs = rmse(y, y).unwrap_or(f64::NAN);
    let r2 = r_squared(y, y).unwrap_or(f64::NAN);
    if m.abs() > AC_RM_004_TOLERANCE { return Rm004Verdict::Fail; }
    if a.abs() > AC_RM_004_TOLERANCE { return Rm004Verdict::Fail; }
    if rs.abs() > AC_RM_004_TOLERANCE { return Rm004Verdict::Fail; }
    if (r2 - 1.0).abs() > AC_RM_004_TOLERANCE { return Rm004Verdict::Fail; }
    Rm004Verdict::Pass
}

// ===========================================================================
// RM-005 — MSE symmetry: MSE(y, ŷ) == MSE(ŷ, y)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rm005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_mse_symmetry(y: &[f32], y_hat: &[f32]) -> Rm005Verdict {
    let a = match mse(y, y_hat) { Some(v) => v, None => return Rm005Verdict::Fail };
    let b = match mse(y_hat, y) { Some(v) => v, None => return Rm005Verdict::Fail };
    if (a - b).abs() < 1e-9 { Rm005Verdict::Pass } else { Rm005Verdict::Fail }
}

// ===========================================================================
// RM-006 — MAE >= 0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rm006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_mae_nonneg(y: &[f32], y_hat: &[f32]) -> Rm006Verdict {
    match mae(y, y_hat) {
        Some(m) if m.is_finite() && m >= -1e-12 => Rm006Verdict::Pass,
        _ => Rm006Verdict::Fail,
    }
}

// ===========================================================================
// RM-007 — RMSE >= 0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rm007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_rmse_nonneg(y: &[f32], y_hat: &[f32]) -> Rm007Verdict {
    match rmse(y, y_hat) {
        Some(r) if r.is_finite() && r >= -1e-12 => Rm007Verdict::Pass,
        _ => Rm007Verdict::Fail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(n: usize) -> (Vec<f32>, Vec<f32>) {
        let y: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let y_hat: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 + 0.1).collect();
        (y, y_hat)
    }

    // Reference impl
    #[test] fn ref_perfect() {
        let y = vec![1.0_f32, 2.0, 3.0];
        assert!(mse(&y, &y).unwrap().abs() < 1e-9);
        assert!(mae(&y, &y).unwrap().abs() < 1e-9);
        assert!(rmse(&y, &y).unwrap().abs() < 1e-9);
        assert!((r_squared(&y, &y).unwrap() - 1.0).abs() < 1e-9);
    }

    #[test] fn ref_mae_le_rmse() {
        let y = vec![1.0_f32, 2.0, 3.0];
        let y_hat = vec![1.5_f32, 1.5, 4.0];
        let m = mae(&y, &y_hat).unwrap();
        let r = rmse(&y, &y_hat).unwrap();
        assert!(m <= r);
    }

    // RM-001
    #[test] fn rm001_pass_normal() {
        let (y, h) = pair(50);
        assert_eq!(verdict_from_r2_upper_bound(&y, &h), Rm001Verdict::Pass);
    }
    #[test] fn rm001_pass_perfect() {
        let y = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(verdict_from_r2_upper_bound(&y, &y), Rm001Verdict::Pass);
    }
    #[test] fn rm001_fail_dim_mismatch() {
        assert_eq!(verdict_from_r2_upper_bound(&[1.0, 2.0], &[1.0]), Rm001Verdict::Fail);
    }

    // RM-002
    #[test] fn rm002_pass_normal() {
        let (y, h) = pair(20);
        assert_eq!(verdict_from_mse_nonneg(&y, &h), Rm002Verdict::Pass);
    }
    #[test] fn rm002_pass_extreme() {
        let y = vec![1e3_f32, -1e3];
        let h = vec![1e3_f32, 0.0];
        assert_eq!(verdict_from_mse_nonneg(&y, &h), Rm002Verdict::Pass);
    }
    #[test] fn rm002_fail_nan() {
        assert_eq!(verdict_from_mse_nonneg(&[f32::NAN], &[1.0]), Rm002Verdict::Fail);
    }

    // RM-003
    #[test] fn rm003_pass_random() {
        let (y, h) = pair(30);
        assert_eq!(verdict_from_mae_le_rmse(&y, &h), Rm003Verdict::Pass);
    }
    #[test] fn rm003_pass_constant_diff() {
        // |x|² == |x| only when |x| in {0, 1}; for constant difference,
        // mae == rmse (Jensen's becomes equality).
        let y = vec![0.0_f32, 0.0, 0.0];
        let h = vec![1.0_f32, 1.0, 1.0];
        assert_eq!(verdict_from_mae_le_rmse(&y, &h), Rm003Verdict::Pass);
    }

    // RM-004
    #[test] fn rm004_pass_canonical() {
        let y = vec![1.0_f32, 2.0, 3.0, 4.0];
        assert_eq!(verdict_from_perfect_prediction(&y), Rm004Verdict::Pass);
    }
    #[test] fn rm004_fail_constant() {
        // Constant y has undefined R² in the standard formula.
        let y = vec![5.0_f32; 4];
        assert_eq!(verdict_from_perfect_prediction(&y), Rm004Verdict::Fail);
    }
    #[test] fn rm004_fail_too_short() {
        let y = vec![1.0_f32];
        assert_eq!(verdict_from_perfect_prediction(&y), Rm004Verdict::Fail);
    }

    // RM-005
    #[test] fn rm005_pass_normal() {
        let (y, h) = pair(20);
        assert_eq!(verdict_from_mse_symmetry(&y, &h), Rm005Verdict::Pass);
    }
    #[test] fn rm005_pass_zero_diff() {
        let y = vec![1.0_f32, 2.0];
        assert_eq!(verdict_from_mse_symmetry(&y, &y), Rm005Verdict::Pass);
    }

    // RM-006
    #[test] fn rm006_pass_normal() {
        let (y, h) = pair(20);
        assert_eq!(verdict_from_mae_nonneg(&y, &h), Rm006Verdict::Pass);
    }

    // RM-007
    #[test] fn rm007_pass_normal() {
        let (y, h) = pair(20);
        assert_eq!(verdict_from_rmse_nonneg(&y, &h), Rm007Verdict::Pass);
    }

    // Provenance pin
    #[test] fn provenance_tolerance() {
        assert!((AC_RM_004_TOLERANCE - 1e-9).abs() < 1e-15);
    }
}
