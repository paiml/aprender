// =========================================================================
// FALSIFY-MULTINOMIAL: contracts/multinomial-head-v1.yaml
//
// Two independent gates on the ONE numerical algorithm phase 3 hand-writes:
//
//   FALSIFY-MULTINOMIAL-001  the sklearn relation  lambda = 1/(2*C*n)
//   FALSIFY-MULTINOMIAL-002  the analytic gradient vs central differences
//
// 002 exists because 001 alone is not enough. 001 compares a converged OPTIMUM,
// and a subtly wrong gradient can still reach the right optimum on a particular
// configuration while being wrong everywhere else — the optimizer would simply
// take a different path to the same place. Central differences check the
// derivative pointwise, at a non-degenerate point, which is a strictly stronger
// statement and costs no Python.
//
// References:
//   - contracts/multinomial-head-v1.yaml
//   - scripts/gen_multinomial_sklearn_fixture.py (the 001 fixture generator)
//   - glm_tests.rs::falsify_glm_irls_link_derivative (the RED-value-in-comment
//     precedent this file follows)
// =========================================================================

use super::multinomial::{
    HeadFitError, MultinomialLogisticRegression, Regularization, SoftmaxNllProblem,
};
use crate::primitives::Vector;

// =========================================================================
// FALSIFY-MULTINOMIAL-002 — central-difference gradient suite
// =========================================================================

/// Central-difference step. In f64 the truncation error of a central difference is
/// O(h^2) ~ 1e-12 and the roundoff is O(eps/h) ~ 2e-10, so 1e-6 sits comfortably in
/// the valley between them.
const GRAD_H: f64 = 1e-6;

/// Relative band for `|g_analytic - g_central| / max(1, |g_analytic|)`.
///
/// Loose enough to ignore the ~2e-10 differencing noise, tight enough that a factor
/// of 2, a sign flip, a missing 1/n, or a penalty leaking onto the intercept is
/// caught by many orders of magnitude.
const GRAD_REL_TOL: f64 = 1e-6;

const GRAD_N: usize = 7;
const GRAD_D: usize = 3;

/// The four configurations the suite must cover.
///
/// `lambda = 0` isolates the NLL; `lambda > 0` brings in the penalty term. A gradient
/// bug in the penalty is invisible at `lambda = 0`, which is exactly why both appear.
const GRAD_CONFIGS: [(usize, f64); 4] = [(2, 0.0), (2, 0.07), (3, 0.0), (3, 0.07)];

/// `X[i][j] = ((7*i + 5*j) mod 25) / 8 - 1.5`, values in [-1.5, 1.5].
///
/// Eighths, so every value is exact in f32 and f64 alike.
fn grad_design_matrix() -> Vec<Vec<f32>> {
    (0..GRAD_N)
        .map(|i| {
            (0..GRAD_D)
                .map(|j| ((7 * i + 5 * j) % 25) as f32 / 8.0 - 1.5)
                .collect()
        })
        .collect()
}

fn round_robin_labels(n: usize, k: usize) -> Vec<usize> {
    (0..n).map(|i| i % k).collect()
}

/// A deliberately NON-ZERO parameter point.
///
/// `x[p] = ((p mod 5) - 2) * 0.37 + 0.11`, i.e. it cycles through
/// `{-0.63, -0.26, 0.11, 0.48, 0.85}`.
///
/// Zeros would be a useless place to test: the penalty gradient `2*lambda*W` vanishes
/// there for ANY factor, so the factor-2 bug this suite exists to catch is invisible
/// at the origin. The cycle length 5 is coprime with neither K nor d for these
/// configurations, so the pattern does not accidentally align with the class or
/// feature stride and leave whole blocks constant.
fn perturbed_point(n_params: usize) -> Vec<f64> {
    (0..n_params)
        .map(|p| ((p % 5) as f64 - 2.0) * 0.37 + 0.11)
        .collect()
}

fn central_difference(problem: &SoftmaxNllProblem<'_>, point: &[f64], j: usize) -> f64 {
    let mut plus = point.to_vec();
    plus[j] += GRAD_H;
    let mut minus = point.to_vec();
    minus[j] -= GRAD_H;
    let f_plus = problem.objective(&Vector::from_vec(plus));
    let f_minus = problem.objective(&Vector::from_vec(minus));
    (f_plus - f_minus) / (2.0 * GRAD_H)
}

fn relative_error(analytic: f64, central: f64) -> f64 {
    (analytic - central).abs() / analytic.abs().max(1.0)
}

/// Largest relative error over `range`, comparing `candidate` against central
/// differences of the objective.
fn max_relative_error(
    problem: &SoftmaxNllProblem<'_>,
    point: &[f64],
    candidate: &[f64],
    range: std::ops::Range<usize>,
) -> f64 {
    range
        .map(|j| relative_error(candidate[j], central_difference(problem, point, j)))
        .fold(0.0_f64, f64::max)
}

/// The head's real gradient, with the penalty term's factor retargeted.
///
/// Derived FROM the real gradient rather than reimplemented: the shipped gradient
/// carries `+2*lambda*W`, so adding `(factor - 2)*lambda*W` yields exactly
/// `+factor*lambda*W` with the NLL part untouched. That keeps the mutation surgical —
/// a hand-written second copy of the NLL could drift and then this file would be
/// testing itself rather than the head.
fn gradient_with_penalty_factor(
    problem: &SoftmaxNllProblem<'_>,
    point: &[f64],
    factor: f64,
) -> Vec<f64> {
    let mut g = problem
        .gradient(&Vector::from_vec(point.to_vec()))
        .as_slice()
        .to_vec();
    for (p, value) in g.iter_mut().enumerate().take(problem.intercept_offset()) {
        *value += (factor - 2.0) * problem.lambda * point[p];
    }
    g
}

/// The head's real gradient, with the penalty wrongly ALSO applied to the intercept.
fn gradient_penalizing_the_intercept(problem: &SoftmaxNllProblem<'_>, point: &[f64]) -> Vec<f64> {
    let mut g = problem
        .gradient(&Vector::from_vec(point.to_vec()))
        .as_slice()
        .to_vec();
    let off = problem.intercept_offset();
    for (p, value) in g.iter_mut().enumerate().skip(off) {
        *value += 2.0 * problem.lambda * point[p];
    }
    g
}

/// Runs `body` with a configured problem for each of the four `{K} x {lambda}` cases.
fn for_each_grad_config(mut body: impl FnMut(usize, f64, &SoftmaxNllProblem<'_>, &[f64])) {
    let features = grad_design_matrix();
    for (k, lambda) in GRAD_CONFIGS {
        let class_indices = round_robin_labels(GRAD_N, k);
        let problem = SoftmaxNllProblem {
            features: &features,
            class_indices: &class_indices,
            n_classes: k,
            n_features: GRAD_D,
            lambda,
        };
        let point = perturbed_point(problem.n_params());
        body(k, lambda, &problem, &point);
    }
}

/// FALSIFY-MULTINOMIAL-002: the analytic gradient agrees with central differences at
/// EVERY parameter index, in all four `{K = 2, 3} x {lambda = 0, 0.07}` configurations,
/// at a non-zero parameter point.
#[test]
fn falsify_multinomial_002_gradient_central_difference() {
    let mut configs_seen = 0;
    for_each_grad_config(|k, lambda, problem, point| {
        configs_seen += 1;
        let analytic = problem
            .gradient(&Vector::from_vec(point.to_vec()))
            .as_slice()
            .to_vec();
        assert_eq!(analytic.len(), problem.n_params());
        assert!(
            point.iter().any(|v| v.abs() > 0.1),
            "the evaluation point must not be degenerate"
        );
        for j in 0..problem.n_params() {
            let central = central_difference(problem, point, j);
            let rel = relative_error(analytic[j], central);
            assert!(
                rel < GRAD_REL_TOL,
                "FALSIFIED MULTINOMIAL-002: K={k}, lambda={lambda}, index {j}: \
                 analytic {analytic:?}[{j}] = {a:e} vs central {c:e} (relative {rel:e} \
                 >= {GRAD_REL_TOL:e})",
                a = analytic[j],
                c = central,
            );
        }
    });
    assert_eq!(configs_seen, 4, "all four configurations must be exercised");
}

/// FALSIFY-MULTINOMIAL-002b: the penalty contributes EXACTLY ZERO to the intercept
/// block.
///
/// This is its own test because it is the one assertion a
/// penalize-everything gradient cannot pass: such a gradient matches central
/// differences on every W entry and fails only here.
#[test]
fn falsify_multinomial_002_intercept_block_excluded_from_penalty() {
    let features = grad_design_matrix();
    for k in [2_usize, 3] {
        let class_indices = round_robin_labels(GRAD_N, k);
        let lambda = 0.07;
        let problem = SoftmaxNllProblem {
            features: &features,
            class_indices: &class_indices,
            n_classes: k,
            n_features: GRAD_D,
            lambda,
        };
        let point = perturbed_point(problem.n_params());
        let off = problem.intercept_offset();
        let analytic = problem
            .gradient(&Vector::from_vec(point.to_vec()))
            .as_slice()
            .to_vec();

        // Every intercept entry of the point is non-zero, so a leaked penalty term
        // 2*lambda*b would be a visible 2*0.07*|b| >= 0.015 offset.
        for p in off..problem.n_params() {
            assert!(
                point[p].abs() > 0.1,
                "K={k}: intercept entry {p} is {v}, too small to expose a leaked penalty",
                v = point[p]
            );
        }

        let worst = max_relative_error(&problem, &point, &analytic, off..problem.n_params());
        assert!(
            worst < GRAD_REL_TOL,
            "FALSIFIED MULTINOMIAL-002b: K={k}: the intercept block disagrees with \
             central differences by {worst:e}, i.e. the L2 penalty is leaking onto the \
             unpenalized intercept"
        );
    }
}

/// The RED observation for MULTINOMIAL-002, encoded as a permanent test rather than
/// left as a line in a log.
///
/// A gradient that uses `lambda*W` instead of `2*lambda*W` is:
///
/// * **GREEN at lambda = 0** — the two differ by `lambda*W`, which is zero there;
/// * **RED at lambda = 0.07** — measured relative error ~1e-2, four orders of
///   magnitude above the 1e-6 band.
///
/// That ASYMMETRY is the point. It proves the suite is actually measuring the penalty
/// term and not just the NLL: a suite that only ran at lambda = 0 would be green
/// against the broken gradient and would have proved nothing about regularization.
#[test]
fn falsify_multinomial_002_broken_penalty_factor_is_red_only_when_lambda_is_positive() {
    let mut checked_zero = 0;
    let mut checked_positive = 0;
    for_each_grad_config(|k, lambda, problem, point| {
        let broken = gradient_with_penalty_factor(problem, point, 1.0);
        let worst = max_relative_error(problem, point, &broken, 0..problem.intercept_offset());
        if lambda == 0.0 {
            checked_zero += 1;
            assert!(
                worst < GRAD_REL_TOL,
                "K={k}: at lambda=0 the broken factor must be INDISTINGUISHABLE \
                 (worst {worst:e}); if it is not, the mutation is not the one claimed"
            );
        } else {
            checked_positive += 1;
            assert!(
                worst > 1e-3,
                "K={k}, lambda={lambda}: the dropped factor of 2 must be caught, but \
                 the worst relative error is only {worst:e}. A gate that cannot see \
                 this mutation is not defending the penalty term."
            );
        }
    });
    assert_eq!(checked_zero, 2);
    assert_eq!(checked_positive, 2);
}

/// The second RED observation: a gradient that penalizes the intercept passes every
/// W-block check and is caught ONLY by the intercept block.
#[test]
fn falsify_multinomial_002_intercept_penalty_bug_is_caught_only_by_the_intercept_block() {
    let features = grad_design_matrix();
    let class_indices = round_robin_labels(GRAD_N, 3);
    let problem = SoftmaxNllProblem {
        features: &features,
        class_indices: &class_indices,
        n_classes: 3,
        n_features: GRAD_D,
        lambda: 0.07,
    };
    let point = perturbed_point(problem.n_params());
    let off = problem.intercept_offset();
    let bugged = gradient_penalizing_the_intercept(&problem, &point);

    let w_block = max_relative_error(&problem, &point, &bugged, 0..off);
    assert!(
        w_block < GRAD_REL_TOL,
        "the intercept-penalty bug must leave the W block untouched (worst {w_block:e})"
    );

    let b_block = max_relative_error(&problem, &point, &bugged, off..problem.n_params());
    assert!(
        b_block > 1e-3,
        "the intercept-penalty bug must be caught by the intercept block, but the \
         worst relative error there is only {b_block:e}"
    );
}

// =========================================================================
// FALSIFY-MULTINOMIAL-001 — the sklearn relation lambda = 1/(2*C*n)
// =========================================================================
//
// GENERATED by scripts/gen_multinomial_sklearn_fixture.py — DO NOT HAND-EDIT the
// constants below. Regenerate with `uv run scripts/gen_multinomial_sklearn_fixture.py`.
//
//   python       : 3.13.0
//   scikit-learn : 1.9.0
//   numpy        : 2.3.5
//   n_iter_      : 15 (max_iter=5000) — converged, asserted by the generator
//
//   get_params():
//     C = 1.0                     class_weight = null        dual = false
//     fit_intercept = true        intercept_scaling = 1      l1_ratio = 0.0
//     max_iter = 5000             n_jobs = null              penalty = "deprecated"
//     random_state = null         solver = "lbfgs"           tol = 1e-10
//     verbose = 0                 warm_start = false
//
//   n = 24 ROWS, d = 4, K = 3, C = 1.0
//   contracted lambda = 1/(2*C*n) = 0.020833333333333332
//   the factor-2 RED value  1/(C*n) = 0.041666666666666664
//
//   Convention proof, computed by the generator — aprender's analytic gradient
//   evaluated AT sklearn's own converged optimum:
//     max|grad| at lambda = 1/(2*C*n) : 2.7934921420329217e-09   (stationary)
//     max|grad| at lambda = 1/(C*n)   : 0.014778266913130668     (NOT stationary)
//   Five million to one. The half inside sklearn's r(W) = (1/2)||W||_F^2 is real,
//   and dropping it does not merely shift the answer slightly.
// =========================================================================

/// `X[i][j] = ((7*i + 5*j) mod 25) / 8 - 1.5` — eighths, exact in f32 and f64.
const SKLEARN_X: [[f64; 4]; 24] = [
    [-1.5, -0.875, -0.25, 0.375],
    [-0.625, 0.0, 0.625, 1.25],
    [0.25, 0.875, 1.5, -1.0],
    [1.125, -1.375, -0.75, -0.125],
    [-1.125, -0.5, 0.125, 0.75],
    [-0.25, 0.375, 1.0, -1.5],
    [0.625, 1.25, -1.25, -0.625],
    [1.5, -1.0, -0.375, 0.25],
    [-0.75, -0.125, 0.5, 1.125],
    [0.125, 0.75, 1.375, -1.125],
    [1.0, -1.5, -0.875, -0.25],
    [-1.25, -0.625, 0.0, 0.625],
    [-0.375, 0.25, 0.875, 1.5],
    [0.5, 1.125, -1.375, -0.75],
    [1.375, -1.125, -0.5, 0.125],
    [-0.875, -0.25, 0.375, 1.0],
    [0.0, 0.625, 1.25, -1.25],
    [0.875, 1.5, -1.0, -0.375],
    [-1.375, -0.75, -0.125, 0.5],
    [-0.5, 0.125, 0.75, 1.375],
    [0.375, 1.0, -1.5, -0.875],
    [1.25, -1.25, -0.625, 0.0],
    [-1.0, -0.375, 0.25, 0.875],
    [-0.125, 0.5, 1.125, -1.375],
];

const SKLEARN_Y: [usize; 24] = [
    0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
];

const SKLEARN_COEF: [[f64; 4]; 3] = [
    [
        -0.12090497247805271,
        -0.21602255166086262,
        -0.11789435905680899,
        0.05478950802820239,
    ],
    [
        0.1281920391502422,
        -0.03681701400784295,
        0.0504668411628533,
        0.2998888935316587,
    ],
    [
        -0.007287066672189499,
        0.25283956566870547,
        0.06742751789395555,
        -0.3546784015598612,
    ],
];

const SKLEARN_INTERCEPT: [f64; 3] = [
    0.009803259354931093,
    0.015838254535947914,
    -0.025641513890878577,
];

const SKLEARN_PROBA: [[f64; 3]; 24] = [
    [0.4844182470450334, 0.301374983618384, 0.21420676933658273],
    [0.344293537781093, 0.44739592728739863, 0.20831053493150825],
    [0.19090987522910247, 0.2408942324381883, 0.5681958923327092],
    [0.41377476274772834, 0.3680706745285579, 0.21815456272371386],
    [
        0.42393625352666725,
        0.36164059093164636,
        0.21442315554168642,
    ],
    [0.23167498875513787, 0.19190691174458438, 0.5764180995002778],
    [0.2544493413897121, 0.26013239110171904, 0.4854182675085689],
    [0.3542620701420464, 0.4320975278335247, 0.2136404020244289],
    [0.36387275909348804, 0.4256141551050617, 0.21051308580145037],
    [0.2007354666480837, 0.22799508493373083, 0.5712694484181855],
    [0.43387013488541504, 0.3474004206678705, 0.2187294444467144],
    [0.4441499225681561, 0.341043336060649, 0.21480674137119493],
    [0.3062258284931597, 0.4911320143843479, 0.2026421571224925],
    [
        0.26706647124201727,
        0.24576262346374844,
        0.48717090529423424,
    ],
    [0.3738966683940034, 0.41049898279021324, 0.21560434881578341],
    [0.383722302417226, 0.40400517260784713, 0.212272524974927],
    [0.21081103314032826, 0.2155252084529376, 0.5736637584067342],
    [
        0.22997505857979572,
        0.29017975597321616,
        0.47984518544698807,
    ],
    [0.4643348311063943, 0.3209331687536963, 0.2147320001399095],
    [0.32505554094326544, 0.46926413603024714, 0.2056803230264874],
    [0.27991015236908273, 0.23185607593859406, 0.4882337716923232],
    [0.39375877935179154, 0.389129453720178, 0.21711176692803055],
    [
        0.40376853158676906,
        0.38265352353312015,
        0.21357794488011078,
    ],
    [0.2211274623489, 0.2034936134197917, 0.5753789242313083],
];

/// Frozen BEFORE the first comparison was run (Phase 1 D-14 discipline).
const PROBA_TOL: f64 = 1e-4;
/// Frozen BEFORE the first comparison was run.
const COEF_TOL: f64 = 1e-4;
/// The wrong-lambda control must miss by at least this much, or it is not a gate.
const WRONG_LAMBDA_MIN_DIVERGENCE: f64 = 1e-3;

/// REGRESSION TRIPWIRE — deliberately NOT the contracted tolerance.
///
/// The contracted claim is [`PROBA_TOL`] = 1e-4, frozen before any comparison ran and
/// left alone afterwards. But the agreement actually MEASURED is 7.74e-9, four orders
/// of magnitude better, so a real regression could degrade the fit by a factor of ten
/// thousand and still sit inside the contracted band unnoticed.
///
/// These two constants therefore say different things: `PROBA_TOL` is the promise,
/// this is the alarm. Widening this one is a legitimate response to a deliberate
/// change (say, dropping the fixture's tolerance back to sklearn's 1e-4 default);
/// widening it silently to make a red build green is not.
const PROBA_MEASURED_TRIPWIRE: f64 = 1e-6;

/// REGRESSION TRIPWIRE for the coefficients; see [`PROBA_MEASURED_TRIPWIRE`].
///
/// Measured 1.45e-8, which is essentially the f32 storage floor: the stored weights
/// are ~0.35 in magnitude and f32 carries ~6e-8 relative precision, so ~2e-8 is the
/// best any f32-stored artifact could do. In other words the f64 fit agrees with
/// scikit-learn to within the deliberate downcast and nothing else.
const COEF_MEASURED_TRIPWIRE: f64 = 1e-6;

/// Solver settings mirroring the generator's.
///
/// Both sides run L-BFGS to a tolerance far below the default 1e-4 ON PURPOSE. At the
/// default, aprender stops on the L2 norm of the full gradient and scipy stops on the
/// max-abs projected gradient; the gap between the two halt points would then dominate
/// the comparison, and the test would be measuring two stopping rules rather than the
/// contracted objective. Tightening the GENERATOR (rather than loosening the assertion)
/// is the direction the plan requires.
const FIXTURE_TOL: f64 = 1e-10;
const FIXTURE_MAX_ITER: usize = 5000;

fn sklearn_features() -> Vec<Vec<f32>> {
    SKLEARN_X
        .iter()
        .map(|row| row.iter().map(|&v| v as f32).collect())
        .collect()
}

fn sklearn_labels() -> Vec<String> {
    (0..3).map(|c| format!("class{c}")).collect()
}

fn fit_at(regularization: Regularization) -> MultinomialLogisticRegression {
    let features = sklearn_features();
    let mut head = MultinomialLogisticRegression::new(3)
        .with_tol(FIXTURE_TOL)
        .with_max_iter(FIXTURE_MAX_ITER);
    let report = head.fit(&features, &SKLEARN_Y, &sklearn_labels(), regularization);
    match report {
        Ok(_) => head,
        Err(e) => panic!("fixture fit failed: {e}"),
    }
}

/// Largest absolute probability difference against the frozen sklearn reference.
fn max_proba_deviation(head: &MultinomialLogisticRegression) -> f64 {
    let probs = head
        .predict_proba(&sklearn_features())
        .expect("predict_proba");
    let mut worst = 0.0_f64;
    for (i, row) in probs.iter().enumerate() {
        for (c, &p) in row.iter().enumerate() {
            worst = worst.max((p - SKLEARN_PROBA[i][c]).abs());
        }
    }
    worst
}

/// FALSIFY-MULTINOMIAL-001: `Regularization::SklearnEquivalentC { c: 1.0 }` over 24
/// rows resolves to `lambda = 1/48` and reproduces scikit-learn's fit.
///
/// The features cross the API as f32 and the fitted weights are STORED as f32, so this
/// compares the artifact a caller actually gets, not a privileged f64 view of it.
#[test]
fn falsify_multinomial_001_sklearn_relation() {
    let head = fit_at(Regularization::SklearnEquivalentC { c: 1.0 });

    assert!(
        (Regularization::SklearnEquivalentC { c: 1.0 }.resolve_lambda(24) - 1.0 / 48.0).abs()
            < 1e-15,
        "the relation itself must resolve to 1/48 at n = 24 rows"
    );

    let worst_proba = max_proba_deviation(&head);
    assert!(
        worst_proba < PROBA_TOL,
        "FALSIFIED MULTINOMIAL-001: worst probability deviation {worst_proba:e} exceeds \
         {PROBA_TOL:e} against scikit-learn 1.9.0"
    );
    assert!(
        worst_proba < PROBA_MEASURED_TRIPWIRE,
        "TRIPWIRE: probability agreement degraded to {worst_proba:e}. This is still \
         inside the contracted {PROBA_TOL:e}, but it is far worse than the 7.74e-9 \
         measured when the fixture was frozen — something regressed. Read the tripwire \
         comment before touching this constant."
    );

    // Coefficients: the L2 penalty fixes the W gauge, so those compare directly.
    let d = head.n_features().expect("fitted");
    let w = head.weights();
    let mut worst_coef = 0.0_f64;
    for (c, row) in SKLEARN_COEF.iter().enumerate() {
        for (j, &expected) in row.iter().enumerate() {
            worst_coef = worst_coef.max((f64::from(w[c * d + j]) - expected).abs());
        }
    }
    assert!(
        worst_coef < COEF_TOL,
        "FALSIFIED MULTINOMIAL-001: worst coefficient deviation {worst_coef:e} exceeds \
         {COEF_TOL:e}"
    );
    assert!(
        worst_coef < COEF_MEASURED_TRIPWIRE,
        "TRIPWIRE: coefficient agreement degraded to {worst_coef:e}, far worse than the \
         1.45e-8 f32-storage floor measured when the fixture was frozen"
    );

    // Intercepts: unpenalized, hence gauge-free. Mean-centre BOTH sides before
    // comparing — an uncentred comparison would be testing where two solvers happened
    // to leave a degree of freedom neither objective constrains.
    let ours: Vec<f64> = head.intercepts().iter().map(|&b| f64::from(b)).collect();
    let ours_mean = ours.iter().sum::<f64>() / ours.len() as f64;
    let theirs_mean = SKLEARN_INTERCEPT.iter().sum::<f64>() / SKLEARN_INTERCEPT.len() as f64;
    let mut worst_intercept = 0.0_f64;
    for (c, &expected) in SKLEARN_INTERCEPT.iter().enumerate() {
        worst_intercept =
            worst_intercept.max(((ours[c] - ours_mean) - (expected - theirs_mean)).abs());
    }
    assert!(
        worst_intercept < COEF_TOL,
        "FALSIFIED MULTINOMIAL-001: worst mean-centred intercept deviation \
         {worst_intercept:e} exceeds {COEF_TOL:e}"
    );
}

/// The in-band control that makes MULTINOMIAL-001 a gate rather than a formality.
///
/// Fitting at the factor-2 RED value `lambda = 1/(C*n) = 1/24` instead of the correct
/// `1/(2*C*n) = 1/48` must MISS the reference by far more than the tolerance. Without
/// this, a test that passed at both lambdas would be asserting nothing about the
/// constant it exists to pin.
#[test]
fn falsify_multinomial_001_wrong_lambda_detected() {
    let correct = fit_at(Regularization::Lambda(1.0 / 48.0));
    let wrong = fit_at(Regularization::Lambda(1.0 / 24.0));

    let good = max_proba_deviation(&correct);
    let bad = max_proba_deviation(&wrong);

    assert!(
        good < PROBA_TOL,
        "the correct lambda = 1/48 must match the reference (got {good:e})"
    );
    assert!(
        bad > WRONG_LAMBDA_MIN_DIVERGENCE,
        "FALSIFIED: the factor-2 error lambda = 1/24 deviates by only {bad:e}, which is \
         under the {WRONG_LAMBDA_MIN_DIVERGENCE:e} the control needs to bite. A gate \
         that passes at both lambdas pins nothing."
    );
    assert!(
        bad > good * 100.0,
        "the wrong lambda ({bad:e}) must be dramatically worse than the right one ({good:e})"
    );
}

/// The head's own objective evaluated at scikit-learn's frozen optimum is stationary
/// at `lambda = 1/(2*C*n)` and NOT at `1/(C*n)`.
///
/// This is the Rust twin of the generator's convention proof, and it is a stronger
/// statement than the end-to-end comparison above: it does not run an optimizer at all,
/// so it cannot be satisfied by two solvers coincidentally drifting to the same place.
#[test]
fn falsify_multinomial_001_sklearn_optimum_is_stationary_only_at_the_halved_lambda() {
    let features = sklearn_features();
    let n = features.len() as f64;
    let c = 1.0;

    // Flatten sklearn's solution into this crate's parameter layout.
    let mut point = Vec::with_capacity(3 * 4 + 3);
    for row in &SKLEARN_COEF {
        point.extend_from_slice(row);
    }
    point.extend_from_slice(&SKLEARN_INTERCEPT);

    let evaluate = |lambda: f64| -> f64 {
        let problem = SoftmaxNllProblem {
            features: &features,
            class_indices: &SKLEARN_Y,
            n_classes: 3,
            n_features: 4,
            lambda,
        };
        problem
            .gradient(&Vector::from_vec(point.clone()))
            .as_slice()
            .iter()
            .fold(0.0_f64, |acc, v| acc.max(v.abs()))
    };

    let correct = evaluate(1.0 / (2.0 * c * n));
    let wrong = evaluate(1.0 / (c * n));

    assert!(
        correct < 1e-6,
        "sklearn's optimum is not stationary for aprender's objective at \
         lambda = 1/(2*C*n): max|grad| = {correct:e}"
    );
    assert!(
        wrong > 1e-3,
        "the factor-2 error must break stationarity, but max|grad| at \
         lambda = 1/(C*n) is only {wrong:e}"
    );
}

/// A non-converged fit is an error even on the fixture data, so the comparison above
/// can never be silently made against a half-optimized head.
#[test]
fn falsify_multinomial_001_fixture_fit_fails_loudly_without_budget() {
    let features = sklearn_features();
    let mut head = MultinomialLogisticRegression::new(3)
        .with_tol(FIXTURE_TOL)
        .with_max_iter(2);
    let err = head
        .fit(
            &features,
            &SKLEARN_Y,
            &sklearn_labels(),
            Regularization::SklearnEquivalentC { c: 1.0 },
        )
        .expect_err("two iterations cannot reach 1e-10");
    assert!(
        matches!(err, HeadFitError::NotConverged { .. }),
        "expected NotConverged, got {err:?}"
    );
}
