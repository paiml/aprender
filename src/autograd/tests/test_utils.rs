//! Test utilities for gradient checking

/// Finite difference gradient checker
///
/// Computes numerical gradient using central difference:
/// f'(x) ≈ (f(x + h) - f(x - h)) / (2h)
pub fn finite_difference<F>(f: F, x: &[f32], epsilon: f32) -> Vec<f32>
where
    F: Fn(&[f32]) -> f32,
{
    let mut grad = vec![0.0; x.len()];
    let mut x_plus = x.to_vec();
    let mut x_minus = x.to_vec();

    for i in 0..x.len() {
        x_plus[i] = x[i] + epsilon;
        x_minus[i] = x[i] - epsilon;

        let f_plus = f(&x_plus);
        let f_minus = f(&x_minus);

        grad[i] = (f_plus - f_minus) / (2.0 * epsilon);

        x_plus[i] = x[i];
        x_minus[i] = x[i];
    }

    grad
}
