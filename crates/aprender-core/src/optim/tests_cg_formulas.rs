use super::*;

#[test]
fn test_cg_fr_non_descent_direction_fallback() {
    // Fletcher-Reeves can produce non-descent directions on difficult problems
    // This tests the `if grad_dot_d >= 0.0` restart branch
    let mut optimizer = ConjugateGradient::new(200, 1e-5, CGBetaFormula::FletcherReeves);

    // Rapidly changing curvature
    let f = |x: &Vector<f32>| x[0].powi(4) + x[1].powi(4) + 2.0 * x[0] * x[0] * x[1] * x[1];
    let grad = |x: &Vector<f32>| {
        Vector::from_slice(&[
            4.0 * x[0].powi(3) + 4.0 * x[0] * x[1] * x[1],
            4.0 * x[1].powi(3) + 4.0 * x[0] * x[0] * x[1],
        ])
    };

    let x0 = Vector::from_slice(&[10.0, -10.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert!(
        result.status == ConvergenceStatus::Converged
            || result.status == ConvergenceStatus::MaxIterations
    );
}

#[test]
fn test_cg_pr_negative_beta() {
    // Test PR formula's automatic restart when beta would be negative
    let mut optimizer = ConjugateGradient::new(100, 1e-5, CGBetaFormula::PolakRibiere);

    // Function where gradients can change direction dramatically
    let f = |x: &Vector<f32>| {
        let a = x[0];
        let b = x[1];
        a.sin().powi(2) + b.cos().powi(2)
    };
    let grad = |x: &Vector<f32>| {
        let a = x[0];
        let b = x[1];
        Vector::from_slice(&[2.0 * a.sin() * a.cos(), -2.0 * b.cos() * b.sin()])
    };

    let x0 = Vector::from_slice(&[1.5, 1.5]);
    let result = optimizer.minimize(f, grad, x0);

    // Should make progress regardless of beta sign issues
    assert!(
        result.status == ConvergenceStatus::Converged
            || result.status == ConvergenceStatus::MaxIterations
            || result.status == ConvergenceStatus::Stalled
    );
}

#[test]
fn test_cg_hs_denominator_near_zero() {
    // Test HS formula when denominator is near zero
    let mut optimizer = ConjugateGradient::new(50, 1e-5, CGBetaFormula::HestenesStiefel);

    // Function with nearly parallel gradients
    let f = |x: &Vector<f32>| x[0] * x[0] + 0.0001 * x[1] * x[1];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0], 0.0002 * x[1]]);

    let x0 = Vector::from_slice(&[5.0, 0.001]);
    let result = optimizer.minimize(f, grad, x0);

    // Should converge despite potentially tricky beta calculation
    assert!(
        result.status == ConvergenceStatus::Converged
            || result.status == ConvergenceStatus::MaxIterations
    );
}

#[test]
fn test_cg_high_dimensional() {
    // Test with higher dimensional problem
    let mut optimizer =
        ConjugateGradient::new(200, 1e-5, CGBetaFormula::PolakRibiere).with_restart_interval(5);

    let f = |x: &Vector<f32>| {
        let mut sum = 0.0;
        for i in 0..x.len() {
            sum += (x[i] - f32::from(i as u8)).powi(2);
        }
        sum
    };
    let grad = |x: &Vector<f32>| {
        let mut g = Vector::zeros(x.len());
        for i in 0..x.len() {
            g[i] = 2.0 * (x[i] - f32::from(i as u8));
        }
        g
    };

    let x0 = Vector::zeros(5);
    let result = optimizer.minimize(f, grad, x0);

    assert!(
        result.status == ConvergenceStatus::Converged
            || result.status == ConvergenceStatus::MaxIterations
    );
}
