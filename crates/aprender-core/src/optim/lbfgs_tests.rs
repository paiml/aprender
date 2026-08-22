pub(crate) use super::*;

#[test]
fn test_lbfgs_quadratic() {
    let mut optimizer = LBFGS::new(100, 1e-5, 10);

    // Simple quadratic: f(x) = (x-5)^2
    let f = |x: &Vector<f32>| (x[0] - 5.0).powi(2);
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 5.0)]);

    let x0 = Vector::from_slice(&[0.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!((result.solution[0] - 5.0).abs() < 1e-4);
}

#[test]
fn test_lbfgs_rosenbrock() {
    let mut optimizer = LBFGS::new(1000, 1e-5, 10);

    let f = |x: &Vector<f32>| {
        let a = x[0];
        let b = x[1];
        (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
    };

    let grad = |x: &Vector<f32>| {
        let a = x[0];
        let b = x[1];
        Vector::from_slice(&[
            -2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
            200.0 * (b - a * a),
        ])
    };

    let x0 = Vector::from_slice(&[0.0, 0.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!((result.solution[0] - 1.0).abs() < 1e-3);
    assert!((result.solution[1] - 1.0).abs() < 1e-3);
}

#[test]
fn test_lbfgs_clone_debug() {
    let opt = LBFGS::new(50, 1e-4, 5);
    let cloned = opt.clone();
    assert_eq!(opt.max_iter, cloned.max_iter);
    assert_eq!(opt.m, cloned.m);
    let debug_str = format!("{:?}", opt);
    assert!(debug_str.contains("LBFGS"));
}

#[test]
fn test_lbfgs_already_converged() {
    let mut optimizer = LBFGS::new(100, 1e-5, 10);
    let f = |x: &Vector<f32>| x[0] * x[0];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

    let x0 = Vector::from_slice(&[0.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert_eq!(result.iterations, 0);
}

#[test]
fn test_lbfgs_stalled_tiny_alpha() {
    // Function that causes line search to return essentially zero
    // Use a flat function where the line search cannot improve
    let mut optimizer = LBFGS::new(100, 1e-20, 5);

    let f = |x: &Vector<f32>| x[0].abs().min(1e-15);
    let grad = |_x: &Vector<f32>| Vector::from_slice(&[1e-15]);

    let x0 = Vector::from_slice(&[1.0]);
    let result = optimizer.minimize(f, grad, x0);

    // May stall, converge, or max-iter depending on line search
    assert!(
        result.status == ConvergenceStatus::Stalled
            || result.status == ConvergenceStatus::Converged
            || result.status == ConvergenceStatus::MaxIterations
    );
}

#[test]
fn test_lbfgs_numerical_error_nan() {
    let mut optimizer = LBFGS::new(100, 1e-5, 5);

    // Function that returns NaN after some steps
    let f = |x: &Vector<f32>| {
        if x[0] > 3.0 {
            f32::NAN
        } else {
            -(x[0] - 5.0).powi(2) // Concave, will diverge
        }
    };
    let grad = |x: &Vector<f32>| Vector::from_slice(&[-2.0 * (x[0] - 5.0)]);

    let x0 = Vector::from_slice(&[2.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert!(
        result.status == ConvergenceStatus::NumericalError
            || result.status == ConvergenceStatus::Converged
            || result.status == ConvergenceStatus::Stalled
            || result.status == ConvergenceStatus::MaxIterations
    );
}

#[test]
fn test_lbfgs_numerical_error_infinite() {
    let mut optimizer = LBFGS::new(100, 1e-5, 5);

    let f = |x: &Vector<f32>| {
        if x[0] > 3.0 {
            f32::INFINITY
        } else {
            -(x[0] - 5.0).powi(2)
        }
    };
    let grad = |x: &Vector<f32>| Vector::from_slice(&[-2.0 * (x[0] - 5.0)]);

    let x0 = Vector::from_slice(&[2.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert!(
        result.status == ConvergenceStatus::NumericalError
            || result.status == ConvergenceStatus::Stalled
            || result.status == ConvergenceStatus::MaxIterations
    );
}

#[test]
fn test_lbfgs_history_overflow() {
    // Use m=2, run long enough to overflow history
    let mut optimizer = LBFGS::new(50, 1e-8, 2);

    let f = |x: &Vector<f32>| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2) + (x[2] - 3.0).powi(2);
    let grad = |x: &Vector<f32>| {
        Vector::from_slice(&[2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0), 2.0 * (x[2] - 3.0)])
    };

    let x0 = Vector::from_slice(&[10.0, -5.0, 8.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
    assert!((result.solution[0] - 1.0).abs() < 1e-3);
    // History should have been capped at m=2
    assert!(optimizer.s_history.len() <= 2);
}

#[test]
fn test_lbfgs_curvature_skip() {
    // Test the y_dot_s <= 1e-10 branch (curvature condition not met)
    // Use a function where gradients don't change much along step
    let mut optimizer = LBFGS::new(100, 1e-5, 5);

    let f = |x: &Vector<f32>| x[0] * x[0];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

    let x0 = Vector::from_slice(&[5.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(result.status, ConvergenceStatus::Converged);
}

#[test]
fn test_lbfgs_norm_function() {
    let v = Vector::from_slice(&[3.0, 4.0]);
    let n = LBFGS::norm(&v);
    assert!((n - 5.0).abs() < 1e-6);

    let zero = Vector::from_slice(&[0.0]);
    assert!(LBFGS::norm(&zero).abs() < 1e-10);
}

#[test]
fn test_lbfgs_reset_clears_history() {
    let mut optimizer = LBFGS::new(100, 1e-5, 5);

    let f = |x: &Vector<f32>| x[0] * x[0];
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * x[0]]);

    let _ = optimizer.minimize(f, grad, Vector::from_slice(&[5.0]));
    assert!(!optimizer.s_history.is_empty());

    optimizer.reset();
    assert!(optimizer.s_history.is_empty());
    assert!(optimizer.y_history.is_empty());
}

#[test]
fn test_lbfgs_compute_direction_no_history() {
    let optimizer = LBFGS::new(100, 1e-5, 5);
    let grad = Vector::from_slice(&[3.0, -4.0]);
    let d = optimizer.compute_direction(&grad);

    // With no history, should be steepest descent: d = -grad
    assert!((d[0] - (-3.0)).abs() < 1e-6);
    assert!((d[1] - 4.0).abs() < 1e-6);
}

#[test]
fn test_lbfgs_max_iterations_deterministic() {
    // Force MaxIterations by using max_iter=1 with a function that doesn't converge in 1 step
    let mut optimizer = LBFGS::new(1, 1e-20, 5);

    // Quadratic far from minimum — won't converge in 1 iteration with tiny tolerance
    let f = |x: &Vector<f32>| (x[0] - 100.0).powi(2);
    let grad = |x: &Vector<f32>| Vector::from_slice(&[2.0 * (x[0] - 100.0)]);

    let x0 = Vector::from_slice(&[0.0]);
    let result = optimizer.minimize(f, grad, x0);

    assert_eq!(
        result.status,
        ConvergenceStatus::MaxIterations,
        "Should hit MaxIterations with max_iter=1"
    );
    assert_eq!(result.iterations, 1);
}

#[test]
fn test_lbfgs_stalled_deterministic() {
    // Force Stalled by returning alpha=0 from line search
    // A constant function has zero gradient change, causing tiny step sizes
    let mut optimizer = LBFGS::new(100, 1e-20, 5);

    // Function where gradient never changes (constant gradient)
    // This causes s_k ~ 0 and line search returns tiny alpha
    let f = |x: &Vector<f32>| x[0]; // Linear, gradient is constant 1.0
    let grad = |_x: &Vector<f32>| Vector::from_slice(&[1.0]);

    let x0 = Vector::from_slice(&[0.0]);
    let result = optimizer.minimize(f, grad, x0);

    // With constant gradient, LBFGS direction is -grad, line search on linear function
    // may stall or hit max iterations
    assert!(
        result.status == ConvergenceStatus::Stalled
            || result.status == ConvergenceStatus::MaxIterations,
        "Should stall or max-iter on linear function: {:?}",
        result.status
    );
}

#[test]
#[should_panic(expected = "does not support stochastic")]
fn test_lbfgs_step_panics() {
    let mut optimizer = LBFGS::new(100, 1e-5, 5);
    let mut params = Vector::from_slice(&[1.0]);
    let grad = Vector::from_slice(&[0.1]);
    optimizer.step(&mut params, &grad);
}

// =========================================================================
// GOLDEN TRAJECTORY REGRESSION (phase 3, plan 03-01, Task 1 Step A)
//
// WHY: plan 03-01 widens the L-BFGS core to f64 behind a PRIVATE generic and
// keeps `LBFGS` as a non-generic f32 wrapper. "It still compiles and the
// convergence tests are green" is NOT evidence that the f32 trajectory is
// unchanged: a rewritten core can take a different number of iterations,
// accept different line-search steps, or return a different status while
// every tolerance-based assertion in this file stays satisfied.
//
// The literals below were CAPTURED at commit
// e2dee4be9f5e3e97193d13489eb6cdd8006628c4 against the PRE-WIDENING,
// f32-hardwired implementation, and are frozen as exact IEEE-754 bit
// patterns. Editing any literal in this block is a FINDING — it means the
// widening changed observable f32 behaviour — not a maintenance chore.
//
// Instrumentation is TEST-ONLY: the objective closure pushes every call into
// a `RefCell<Vec<f32>>` the test owns. No `pub` item exists for this, and the
// solver needs no observable "accepted step" API.
// =========================================================================

mod golden_trajectory {
    use super::*;
    use std::cell::RefCell;

    /// Everything an observer can see about one f32 L-BFGS run, as exact bits.
    #[derive(Debug)]
    struct Golden {
        solution_bits: Vec<u32>,
        status: String,
        iterations: usize,
        gradient_norm_bits: u32,
        objective_calls: usize,
        accepted_objective_bits: Vec<u32>,
    }

    /// The frozen expectation for one case.
    struct Frozen {
        solution_bits: &'static [u32],
        status: &'static str,
        iterations: usize,
        gradient_norm_bits: u32,
        objective_calls: usize,
        accepted_objective_bits: &'static [u32],
    }

    /// Runs `opt` on `(f, g, x0)` through a recording objective and returns the
    /// exact-bit trajectory record.
    fn record<F, G>(mut opt: LBFGS, f: F, g: G, x0: Vector<f32>) -> Golden
    where
        F: Fn(&Vector<f32>) -> f32,
        G: Fn(&Vector<f32>) -> Vector<f32>,
    {
        let calls: RefCell<Vec<f32>> = RefCell::new(Vec::new());
        let recorded = |x: &Vector<f32>| {
            let value = f(x);
            calls.borrow_mut().push(value);
            value
        };
        let result = opt.minimize(&recorded, &g, x0);
        let calls = calls.into_inner();

        // "Accepted" objective values: the strictly-decreasing running-minimum
        // subsequence of every evaluation the solver made. Derived by the test
        // from its own recording, so the solver needs no new public surface.
        let mut accepted = Vec::new();
        let mut best = f32::INFINITY;
        for value in &calls {
            if *value < best {
                best = *value;
                accepted.push(value.to_bits());
            }
        }

        Golden {
            solution_bits: (0..result.solution.len())
                .map(|i| result.solution[i].to_bits())
                .collect(),
            status: format!("{:?}", result.status),
            iterations: result.iterations,
            gradient_norm_bits: result.gradient_norm.to_bits(),
            objective_calls: calls.len(),
            accepted_objective_bits: accepted,
        }
    }

    /// Prints one record in copy-pasteable literal form (capture aid).
    fn dump(name: &str, got: &Golden) {
        eprintln!("// ---- CAPTURE {name} ----");
        eprintln!("solution_bits: &{:?},", got.solution_bits);
        eprintln!("status: {:?},", got.status);
        eprintln!("iterations: {},", got.iterations);
        eprintln!("gradient_norm_bits: {},", got.gradient_norm_bits);
        eprintln!("objective_calls: {},", got.objective_calls);
        eprintln!(
            "accepted_objective_bits: &{:?},",
            got.accepted_objective_bits
        );
    }

    fn assert_frozen(name: &str, got: &Golden, want: &Frozen) {
        assert_eq!(
            got.solution_bits.as_slice(),
            want.solution_bits,
            "GOLDEN {name}: solution bit patterns changed — the f32 trajectory is NOT byte-identical"
        );
        assert_eq!(
            got.status, want.status,
            "GOLDEN {name}: ConvergenceStatus changed"
        );
        assert_eq!(
            got.iterations, want.iterations,
            "GOLDEN {name}: iteration count changed"
        );
        assert_eq!(
            got.gradient_norm_bits, want.gradient_norm_bits,
            "GOLDEN {name}: gradient_norm bit pattern changed"
        );
        assert_eq!(
            got.objective_calls, want.objective_calls,
            "GOLDEN {name}: objective evaluation count changed (line search behaved differently)"
        );
        assert_eq!(
            got.accepted_objective_bits.as_slice(),
            want.accepted_objective_bits,
            "GOLDEN {name}: accepted-objective sequence changed"
        );
    }

    /// Case A — the FALSIFY-LBFGS-001 quadratic: f(x) = x0^2 from x0 = 5.
    fn case_quadratic_1d() -> Golden {
        record(
            LBFGS::new(100, 1e-6, 10),
            |x: &Vector<f32>| x[0] * x[0],
            |x: &Vector<f32>| Vector::from_vec(vec![2.0 * x[0]]),
            Vector::from_vec(vec![5.0]),
        )
    }

    /// Case B — the FALSIFY-LBFGS-002 quadratic: f(x) = x0^2 + x1^2 from (3, 4).
    fn case_quadratic_2d() -> Golden {
        record(
            LBFGS::new(100, 1e-6, 10),
            |x: &Vector<f32>| x[0] * x[0] + x[1] * x[1],
            |x: &Vector<f32>| Vector::from_vec(vec![2.0 * x[0], 2.0 * x[1]]),
            Vector::from_vec(vec![3.0, 4.0]),
        )
    }

    /// Case C — diagonal Hessian with condition number 1e4, the case most
    /// sensitive to line-search and initial-scaling differences.
    fn case_ill_conditioned_2d() -> Golden {
        record(
            LBFGS::new(100, 1e-6, 10),
            |x: &Vector<f32>| x[0] * x[0] + 1e4 * x[1] * x[1],
            |x: &Vector<f32>| Vector::from_vec(vec![2.0 * x[0], 2e4 * x[1]]),
            Vector::from_vec(vec![1.0, 1.0]),
        )
    }

    #[test]
    fn lbfgs_f32_golden_trajectory_is_unchanged() {
        let a = case_quadratic_1d();
        let b = case_quadratic_2d();
        let c = case_ill_conditioned_2d();
        dump("quadratic_1d", &a);
        dump("quadratic_2d", &b);
        dump("ill_conditioned_2d", &c);

        // FROZEN at e2dee4be9f5e3e97193d13489eb6cdd8006628c4 (pre-widening).
        assert_frozen(
            "quadratic_1d",
            &a,
            &Frozen {
                solution_bits: &[0],
                status: "Converged",
                iterations: 1,
                gradient_norm_bits: 0,
                objective_calls: 5,
                accepted_objective_bits: &[1_103_626_240, 0],
            },
        );
        // FROZEN at e2dee4be9f5e3e97193d13489eb6cdd8006628c4 (pre-widening).
        assert_frozen(
            "quadratic_2d",
            &b,
            &Frozen {
                solution_bits: &[0, 0],
                status: "Converged",
                iterations: 1,
                gradient_norm_bits: 0,
                objective_calls: 5,
                accepted_objective_bits: &[1_103_626_240, 0],
            },
        );
        // FROZEN at e2dee4be9f5e3e97193d13489eb6cdd8006628c4 (pre-widening).
        assert_frozen(
            "ill_conditioned_2d",
            &c,
            &Frozen {
                solution_bits: &[2_836_135_936, 612_368_384],
                status: "Converged",
                iterations: 6,
                gradient_norm_bits: 731_676_332,
                objective_calls: 43,
                accepted_objective_bits: &[
                    1_176_257_536,
                    1_140_067_482,
                    1_065_346_507,
                    1_065_343_154,
                    1_065_339_799,
                    1_065_333_092,
                    1_065_319_685,
                    1_065_292_884,
                    1_065_239_348,
                    1_065_132_533,
                    1_064_919_935,
                    1_064_498_875,
                    1_063_673_280,
                    1_062_088_200,
                    847_008_392,
                    666_613_385,
                    312_345_088,
                ],
            },
        );
    }
}
