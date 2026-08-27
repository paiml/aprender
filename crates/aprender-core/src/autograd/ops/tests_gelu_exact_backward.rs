//! Falsifier: `Tensor::gelu_exact` MUST be the EXACT erf GELU, graph-connected, with an
//! analytic backward that matches central finite differences.
//!
//! Contract: `setfit-encoder-conformance-v1`, equation `gelu_exact`. Amendment A-03.
//! Obligations: OBLIG-ENC-03-ACTIVATION-PARITY, D-04 (per-element gradcheck).
//!
//! # Why this file exists
//!
//! The pinned MiniLM config sets `hidden_act: "gelu"`, which in HuggingFace means the
//! EXACT erf form `0.5*x*(1 + erf(x/sqrt(2)))`. The pre-existing `Tensor::gelu` is the
//! tanh APPROXIMATION `0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))` — a DIFFERENT
//! function. Substituting one for the other is a parity defect, not a rounding
//! difference, and widening the FFN tolerance to absorb it is explicitly forbidden.
//!
//! Three independent guards:
//!
//! 1. **DIFFERENTIAL** — `gelu_exact` and `gelu` must measurably DISAGREE. This is the
//!    anti-tampering gate (T-1-19): a future edit cannot silently route `gelu_exact`
//!    back to the tanh approximation and still pass. Note that point values near x=1
//!    agree to ~2e-7, so a naive spot-check CANNOT tell the two apart — only a scan
//!    over the region where they diverge can.
//! 2. **ORACLE** — an f64 erf implemented HERE, independently of the production path
//!    (Taylor series for |t| <= 2, continued fraction for |t| > 2), versus the
//!    production path's Cody rational approximation. Agreement between two
//!    independently derived algorithms is evidence; agreement of an implementation
//!    with itself is a tautology.
//! 3. **FINITE DIFFERENCE** — the analytic backward versus central differences at every
//!    element of a grid spanning [-6, 6].

use crate::autograd::{self, Tensor};

const FD_EPS: f32 = 1e-3;
const TOL: f32 = 2e-2;

// ===========================================================================
// Independent f64 erf oracle — NOT the production implementation.
//
// Production uses Cody's rational Chebyshev approximation (aprender-common
// `erf_precise` / `erfc_precise`). This oracle uses a completely different
// derivation: the Maclaurin series for small |t| and the Laplace continued
// fraction for large |t|. Neither shares code, coefficients, or structure with
// the production path, so agreement is genuine evidence of correctness.
// ===========================================================================

/// erf via the Maclaurin series: erf(t) = (2/sqrt(pi)) * sum (-1)^n t^(2n+1) / (n!(2n+1)).
/// Converges rapidly in f64 for |t| <= 2 (max term ~3.2 at t=2, so cancellation costs
/// about one decimal digit out of ~16).
fn oracle_erf_series(t: f64) -> f64 {
    // u_n = (-1)^n t^(2n+1) / n!, with u_0 = t and u_n = u_{n-1} * (-t^2 / n).
    // The n-th series term is u_n / (2n+1).
    let mut u = t;
    let mut sum = t;
    let mut n = 1.0_f64;
    while n <= 200.0 {
        u *= -(t * t) / n;
        let add = u / (2.0 * n + 1.0);
        sum += add;
        if add == 0.0 || add.abs() < 1e-18 * sum.abs() {
            break;
        }
        n += 1.0;
    }
    sum * 2.0 / std::f64::consts::PI.sqrt()
}

/// erfc via the Laplace continued fraction, evaluated by backward recurrence:
/// erfc(t) = exp(-t^2)/sqrt(pi) * 1/(t + (1/2)/(t + 1/(t + (3/2)/(t + 2/(t + ...)))))
/// Valid and rapidly convergent for t > 2.
fn oracle_erfc_cf(t: f64) -> f64 {
    debug_assert!(t > 0.0);
    let mut cf = 0.0_f64;
    let mut k = 80_i32;
    while k >= 1 {
        cf = (f64::from(k) / 2.0) / (t + cf);
        k -= 1;
    }
    (-t * t).exp() / std::f64::consts::PI.sqrt() / (t + cf)
}

/// Independent f64 erf: series for |t| <= 2, continued-fraction erfc beyond.
fn oracle_erf(t: f64) -> f64 {
    let a = t.abs();
    let v = if a <= 2.0 {
        oracle_erf_series(a)
    } else {
        1.0 - oracle_erfc_cf(a)
    };
    if t < 0.0 {
        -v
    } else {
        v
    }
}

/// Reference GELU at f64, computed from the oracle erf.
fn oracle_gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + oracle_erf(x / std::f64::consts::SQRT_2))
}

/// Dense grid spanning [-6, 6].
fn grid() -> Vec<f32> {
    (0..=240).map(|i| -6.0 + 0.05 * (i as f32)).collect()
}

// ===========================================================================
// Oracle self-verification — an oracle that is itself wrong is worse than none.
// ===========================================================================

#[test]
fn gelu_exact_oracle_erf_matches_known_high_precision_values() {
    // Reference values (f64, from the standard library erf of a mature libm).
    let cases: [(f64, f64); 7] = [
        (0.0, 0.0),
        (0.5, 0.520_499_877_813_046_5),
        (1.0, 0.842_700_792_949_714_9),
        (1.5, 0.966_105_146_475_310_7),
        (2.0, 0.995_322_265_018_952_7),
        (3.0, 0.999_977_909_503_001_4),
        (4.0, 0.999_999_984_582_742_1),
    ];
    for (t, want) in cases {
        let got = oracle_erf(t);
        assert!(
            (got - want).abs() < 1e-14,
            "oracle_erf({t}) = {got}, expected {want} (dev {:.3e})",
            (got - want).abs()
        );
        // Odd symmetry.
        let got_neg = oracle_erf(-t);
        assert!(
            (got_neg + want).abs() < 1e-14,
            "oracle_erf({}) = {got_neg}, expected {}",
            -t,
            -want
        );
    }
}

// ===========================================================================
// Forward correctness
// ===========================================================================

#[test]
fn gelu_exact_matches_known_point_values() {
    let x = Tensor::new(&[0.0, 1.0, -1.0], &[3]);
    let y = x.gelu_exact();

    assert!(
        y.data()[0].abs() < 1e-7,
        "gelu_exact(0) = {}, expected 0",
        y.data()[0]
    );
    assert!(
        (y.data()[1] - 0.841_344_7).abs() < 1e-6,
        "gelu_exact(1) = {}, expected 0.8413447",
        y.data()[1]
    );
    assert!(
        (y.data()[2] - (-0.158_655_25)).abs() < 1e-6,
        "gelu_exact(-1) = {}, expected -0.15865525",
        y.data()[2]
    );
}

#[test]
fn gelu_exact_has_the_correct_asymptotic_shape() {
    let x = Tensor::new(&[-8.0, -6.0, 0.0, 6.0, 8.0], &[5]);
    let y = x.gelu_exact();

    assert!(
        y.data()[0].abs() < 1e-6,
        "gelu_exact(-8) should approach 0, got {}",
        y.data()[0]
    );
    assert!(
        y.data()[1].abs() < 1e-6,
        "gelu_exact(-6) should approach 0, got {}",
        y.data()[1]
    );
    assert!(y.data()[2].abs() < 1e-7, "gelu_exact(0) must be exactly 0");
    assert!(
        (y.data()[3] - 6.0).abs() < 1e-5,
        "gelu_exact(6) should approach x=6, got {}",
        y.data()[3]
    );
    assert!(
        (y.data()[4] - 8.0).abs() < 1e-5,
        "gelu_exact(8) should approach x=8, got {}",
        y.data()[4]
    );
}

#[test]
fn gelu_exact_has_a_shallow_negative_minimum_near_minus_three_quarters() {
    // The exact GELU is NOT monotone: it dips below zero around x = -0.75.
    // Pins the characteristic shape, so a monotone stand-in cannot pass.
    let x = Tensor::new(&[-0.75], &[1]);
    let y = x.gelu_exact();
    assert!(
        y.data()[0] < -0.16 && y.data()[0] > -0.18,
        "gelu_exact(-0.75) = {}, expected the shallow minimum near -0.17",
        y.data()[0]
    );
}

// ===========================================================================
// ORACLE — production (Cody) vs independently derived (series + CF)
// ===========================================================================

#[test]
fn gelu_exact_matches_the_independent_f64_oracle_within_f32_noise() {
    let g = grid();
    let x = Tensor::new(&g, &[g.len()]);
    let y = x.gelu_exact();

    let mut max_dev = 0.0_f64;
    let mut max_at = 0.0_f32;
    for (i, &xi) in g.iter().enumerate() {
        let want = oracle_gelu(f64::from(xi));
        let dev = (f64::from(y.data()[i]) - want).abs();
        if dev > max_dev {
            max_dev = dev;
            max_at = xi;
        }
    }

    // f32 has ~6e-8 relative precision; over [-6,6] the largest representable
    // magnitude is 6, so a correct f32 implementation lands within ~1e-6 absolute.
    assert!(
        max_dev < 1e-6,
        "gelu_exact deviates from the independent oracle by {max_dev:.4e} at x={max_at} \
         — exceeds the f32 noise floor, so this is an ALGORITHM error, not rounding"
    );
    println!("ORACLE max |gelu_exact - oracle| = {max_dev:.4e} at x = {max_at}");
}

#[test]
fn gelu_exact_relative_accuracy_holds_in_the_negative_tail() {
    // The negative tail is where `1 + erf(x/sqrt(2))` cancels catastrophically:
    // at x = -2.67 the sum of two ~1.0 quantities leaves ~0.0077. An erf good only
    // to 1.5e-7 ABSOLUTE therefore yields ~2e-5 RELATIVE error here — a systematic
    // bias that compounds across six FFN layers. This test pins the relative error.
    let pts: [f32; 6] = [-1.5, -2.0, -2.5, -2.67, -3.0, -3.5];
    let x = Tensor::new(&pts, &[pts.len()]);
    let y = x.gelu_exact();

    for (i, &xi) in pts.iter().enumerate() {
        let want = oracle_gelu(f64::from(xi));
        let got = f64::from(y.data()[i]);
        let rel = (got - want).abs() / want.abs();
        assert!(
            rel < 1e-5,
            "gelu_exact({xi}) = {got}, oracle {want}, relative error {rel:.3e} \
             — the negative-tail cancellation is not being handled accurately"
        );
    }
}

// ===========================================================================
// DIFFERENTIAL — gelu_exact is provably NOT the tanh approximation
// ===========================================================================

#[test]
fn gelu_exact_is_a_different_function_from_the_tanh_gelu() {
    let g = grid();
    let x = Tensor::new(&g, &[g.len()]);
    let exact = x.gelu_exact();
    let tanh_v = x.gelu();

    let mut max_diff = 0.0_f32;
    let mut max_at = 0.0_f32;
    for (i, &xi) in g.iter().enumerate() {
        let d = (exact.data()[i] - tanh_v.data()[i]).abs();
        if d > max_diff {
            max_diff = d;
            max_at = xi;
        }
    }

    // MEASURED (plan 01-09): the true max |exact - tanh| over [-6,6] is 4.73e-4 at
    // x ~= -2.70. The plan's provisional ">1e-3" figure is NOT attainable by a correct
    // implementation — asserting it would fail the phase on correct code. 3e-4 sits
    // safely below the measured maximum and ~3 orders of magnitude above f32 noise,
    // so it cannot be satisfied by rounding.
    assert!(
        max_diff > 3e-4,
        "gelu_exact and gelu differ by only {max_diff:.4e} (at x={max_at}) — \
         gelu_exact appears to BE the tanh approximation"
    );

    // And confirm the divergence is where theory says it is.
    assert!(
        max_at < -2.0 && max_at > -3.5,
        "max divergence at x={max_at}, expected the region near x = -2.7"
    );
    println!("DIFFERENTIAL max |gelu_exact - gelu| = {max_diff:.4e} at x = {max_at}");
}

#[test]
fn gelu_exact_tracks_the_oracle_more_closely_than_the_tanh_gelu_does() {
    // Two-sided: not only do they differ, but gelu_exact is the one that is RIGHT.
    let g = grid();
    let x = Tensor::new(&g, &[g.len()]);
    let exact = x.gelu_exact();
    let tanh_v = x.gelu();

    let mut worst_exact = 0.0_f64;
    let mut worst_tanh = 0.0_f64;
    for (i, &xi) in g.iter().enumerate() {
        let want = oracle_gelu(f64::from(xi));
        worst_exact = worst_exact.max((f64::from(exact.data()[i]) - want).abs());
        worst_tanh = worst_tanh.max((f64::from(tanh_v.data()[i]) - want).abs());
    }
    assert!(
        worst_exact * 100.0 < worst_tanh,
        "gelu_exact (worst {worst_exact:.3e}) must be far closer to the oracle than \
         the tanh gelu (worst {worst_tanh:.3e})"
    );
}

// ===========================================================================
// Graph connectivity + finite-difference gradcheck
// ===========================================================================

#[test]
fn gelu_exact_records_a_gelu_exact_backward_edge() {
    autograd::clear_graph();

    let x = Tensor::new(&[-1.0, 0.0, 0.5, 2.0], &[4]).requires_grad();
    let y = x.gelu_exact();

    assert!(
        y.requires_grad_enabled(),
        "gelu_exact output lost requires_grad — graph severed"
    );
    let gf = y.grad_fn().expect("gelu_exact recorded no grad_fn");
    assert_eq!(
        gf.name(),
        "GeluExactBackward",
        "gelu_exact must record GeluExactBackward, not the tanh GeluBackward"
    );
}

/// Fixed, non-uniform detached coefficients so dL/dx is non-degenerate.
fn coeff(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.21 + 0.013 * (i as f32)).collect()
}

fn scalar_loss(y: &Tensor, c: &[f32]) -> Tensor {
    let c_tensor = Tensor::new(c, y.shape());
    y.mul(&c_tensor).sum()
}

fn perturbed_loss(x_data: &[f32], i: usize, eps: f32, c: &[f32]) -> f32 {
    autograd::no_grad(|| {
        let mut d = x_data.to_vec();
        d[i] += eps;
        let x = Tensor::new(&d, &[d.len()]);
        let y = x.gelu_exact();
        scalar_loss(&y, c).data()[0]
    })
}

#[test]
fn gelu_exact_backward_matches_central_finite_differences_over_the_grid() {
    autograd::clear_graph();

    let g = grid();
    let c = coeff(g.len());

    let x = Tensor::new(&g, &[g.len()]).requires_grad();
    let xid = x.id();
    let y = x.gelu_exact();
    let loss = scalar_loss(&y, &c);
    loss.backward();

    let grad = autograd::get_grad(xid)
        .expect("gelu_exact input received NO gradient — autograd graph severed");

    assert_eq!(grad.shape(), &[g.len()], "grad shape mismatch");
    assert!(
        grad.data().iter().all(|v| v.is_finite()),
        "gelu_exact produced a non-finite gradient"
    );
    assert!(
        grad.data().iter().any(|&v| v.abs() > 1e-9),
        "gelu_exact gradient is all zero"
    );

    for i in 0..g.len() {
        let num = (perturbed_loss(&g, i, FD_EPS, &c) - perturbed_loss(&g, i, -FD_EPS, &c))
            / (2.0 * FD_EPS);
        let analytic = grad.data()[i];
        let denom = analytic.abs().max(num.abs()).max(1.0);
        let rel = (analytic - num).abs() / denom;
        assert!(
            rel < TOL,
            "dL/dx[{i}] (x = {}): analytic {analytic} != finite-diff {num} (rel {rel})",
            g[i]
        );
    }
}

#[test]
fn gelu_exact_backward_matches_the_closed_form_derivative() {
    // d/dx gelu_exact(x) = Phi(x) + x * phi(x), computed here from the INDEPENDENT
    // oracle erf rather than the production erf.
    autograd::clear_graph();

    let pts: [f32; 9] = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
    let x = Tensor::new(&pts, &[pts.len()]).requires_grad();
    let xid = x.id();
    let y = x.gelu_exact();
    y.sum().backward();

    let grad = autograd::get_grad(xid).expect("no gradient recorded");

    for (i, &xi) in pts.iter().enumerate() {
        let xd = f64::from(xi);
        let phi_cap = 0.5 * (1.0 + oracle_erf(xd / std::f64::consts::SQRT_2));
        let phi = (-xd * xd / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let want = phi_cap + xd * phi;
        let got = f64::from(grad.data()[i]);
        assert!(
            (got - want).abs() < 1e-5,
            "d/dx gelu_exact({xi}) = {got}, closed form {want}"
        );
    }
}

#[test]
fn gelu_exact_propagates_non_finite_input_without_panicking() {
    // Records the OBSERVED behavior rather than asserting a policy: a non-finite
    // activation is a training-dynamics signal, and this op must not panic on it.
    let x = Tensor::new(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0], &[4]);
    let y = x.gelu_exact();

    assert!(y.data()[0].is_nan(), "NaN input must yield NaN, not a panic");
    assert!(
        y.data()[1].is_infinite() && y.data()[1] > 0.0,
        "+inf input yields {} (expected +inf)",
        y.data()[1]
    );
    assert!(
        y.data()[2].abs() < 1e-6 || y.data()[2].is_nan(),
        "-inf input yields {} (expected 0 or NaN, never a panic)",
        y.data()[2]
    );
    assert!(
        (y.data()[3] - 0.841_344_7).abs() < 1e-6,
        "a finite element alongside non-finite ones must still be correct"
    );
}
