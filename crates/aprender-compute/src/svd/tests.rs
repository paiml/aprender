//! Falsification tests for the SVD substrate (#3147, contracts/svd-v1.yaml).

use super::{gaussian, matmul, transpose, RandomizedSvdConfig, Svd};
use crate::TruenoError;
use proptest::prelude::*;

const TOL: f64 = 1e-10;

fn frobenius(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn relative_reconstruction_error(a: &[f64], svd: &Svd) -> f64 {
    let back = svd.reconstruct();
    let diff: Vec<f64> = a.iter().zip(&back).map(|(x, y)| x - y).collect();
    let norm = frobenius(a);
    if norm == 0.0 {
        frobenius(&diff)
    } else {
        frobenius(&diff) / norm
    }
}

/// `max |XᵀX − I|` over the `cols` columns of the row-major `rows × cols` matrix `x`.
fn column_orthonormality_error(x: &[f64], rows: usize, cols: usize) -> f64 {
    let gram = matmul(&transpose(x, rows, cols), x, cols, rows, cols);
    let mut worst = 0.0_f64;
    for i in 0..cols {
        for j in 0..cols {
            let want = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((gram[i * cols + j] - want).abs());
        }
    }
    worst
}

/// Every structural property a decomposition must hold, whatever its input.
fn assert_well_formed(a: &[f64], m: usize, n: usize, svd: &Svd, label: &str) {
    let k = svd.len();
    assert_eq!((svd.rows(), svd.cols(), k), (m, n, m.min(n)), "{label}: shape");
    assert_eq!((svd.u().len(), svd.vt().len()), (m * k, k * n), "{label}: factor sizes");
    let rel = relative_reconstruction_error(a, svd);
    assert!(rel <= TOL, "{label}: ||A - U S Vt||_F / ||A||_F = {rel:e} > {TOL:e}");
    let eu = column_orthonormality_error(svd.u(), m, k);
    assert!(eu <= TOL, "{label}: U^T U deviates from I by {eu:e}");
    let ev = column_orthonormality_error(&transpose(svd.vt(), k, n), n, k);
    assert!(ev <= TOL, "{label}: V^T V deviates from I by {ev:e}");
    let s = svd.singular_values();
    assert!(s.iter().all(|&x| x >= 0.0), "{label}: negative singular value in {s:?}");
    assert!(s.windows(2).all(|w| w[0] >= w[1]), "{label}: not descending: {s:?}");
    assert_sign_convention(svd, label);
}

fn assert_sign_convention(svd: &Svd, label: &str) {
    let k = svd.len();
    for j in 0..k {
        let col: Vec<f64> = (0..svd.rows()).map(|i| svd.u()[i * k + j]).collect();
        let max = col.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()));
        let first = col.iter().find(|x| x.abs() == max).copied().unwrap_or(0.0);
        assert!(first >= 0.0, "{label}: U column {j} has its largest entry negative ({first})");
    }
}

/// Deterministic case generator: a small LCG over the case index.
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % n as u64) as usize
    }
}

/// A random `n × n` orthogonal matrix, from nalgebra's QR (an independent code path).
fn orthogonal(n: usize, seed: u64) -> Vec<f64> {
    let g = nalgebra::DMatrix::from_row_slice(n, n, &gaussian(n, n, seed));
    let q = g.qr().q();
    (0..n).flat_map(|i| (0..n).map(move |j| (i, j))).map(|(i, j)| q[(i, j)]).collect()
}

/// `m × n` matrix with prescribed singular values `sigma` (length `min(m, n)`).
fn with_spectrum(m: usize, n: usize, sigma: &[f64], seed: u64) -> Vec<f64> {
    let r = sigma.len();
    let (u, v) = (orthogonal(m, seed), orthogonal(n, seed ^ 0xabcd));
    let mut a = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            a[i * n + j] = (0..r).map(|t| u[i * m + t] * sigma[t] * v[j * n + t]).sum();
        }
    }
    a
}

fn nalgebra_condition(a: &[f64], m: usize, n: usize) -> f64 {
    let s = nalgebra::DMatrix::from_row_slice(m, n, a).singular_values();
    let (hi, lo) = s.iter().fold((0.0_f64, f64::MAX), |(hi, lo), &x| (hi.max(x), lo.min(x)));
    hi / lo
}

#[derive(Debug, Clone, Copy)]
enum Class {
    Wide,
    Square,
    Tall,
    RankDeficient,
    IllConditioned,
}

/// The 200-matrix acceptance set: 40 each of m<n, m=n, m>n, rank-deficient and cond ≥ 1e8.
fn acceptance_set() -> Vec<(Class, usize, usize, usize, Vec<f64>)> {
    let classes =
        [Class::Wide, Class::Square, Class::Tall, Class::RankDeficient, Class::IllConditioned];
    let mut rng = Lcg(3147);
    (0..200)
        .map(|case| {
            let class = classes[case % classes.len()];
            let seed = 1000 + case as u64;
            let small = 1 + rng.below(12);
            let extra = 1 + rng.below(12);
            let (m, n) = match class {
                Class::Wide => (small, small + extra),
                Class::Square => (small, small),
                Class::Tall => (small + extra, small),
                Class::RankDeficient | Class::IllConditioned => {
                    let (a, b) = (2 + rng.below(12), 2 + rng.below(12));
                    (a, b)
                }
            };
            let r = m.min(n);
            let (rank, a) = match class {
                Class::RankDeficient => {
                    let rank = 1 + rng.below(r - 1);
                    let left = gaussian(m, rank, seed);
                    let right = gaussian(rank, n, seed ^ 0x55);
                    (rank, matmul(&left, &right, m, rank, n))
                }
                Class::IllConditioned => {
                    let sigma: Vec<f64> =
                        (0..r).map(|t| 10f64.powf(-10.0 * t as f64 / (r - 1) as f64)).collect();
                    (r, with_spectrum(m, n, &sigma, seed))
                }
                _ => (r, gaussian(m, n, seed).iter().map(|x| x * 10.0).collect()),
            };
            (class, m, n, rank, a)
        })
        .collect()
}

// FALSIFY-SVD-001: reconstruction, orthonormality, ordering and signs on 200 matrices.
#[test]
fn falsify_svd_001_reconstruction_over_200_matrices() {
    let set = acceptance_set();
    assert_eq!(set.len(), 200);
    for (case, (class, m, n, _, a)) in set.iter().enumerate() {
        let svd = Svd::new(a, *m, *n).expect("finite non-empty input decomposes");
        assert_well_formed(a, *m, *n, &svd, &format!("case {case} {class:?} {m}x{n}"));
    }
}

/// The ill-conditioned class must really be ill-conditioned, measured by nalgebra,
/// or FALSIFY-SVD-001 would pass on easy inputs and claim coverage it lacks.
#[test]
fn acceptance_set_covers_every_shape_class() {
    let set = acceptance_set();
    let count = |f: &dyn Fn(&(Class, usize, usize, usize, Vec<f64>)) -> bool| {
        set.iter().filter(|c| f(c)).count()
    };
    assert!(count(&|c| c.1 < c.2) >= 40, "m < n");
    assert!(count(&|c| c.1 == c.2) >= 40, "m = n");
    assert!(count(&|c| c.1 > c.2) >= 40, "m > n");
    assert!(count(&|c| c.3 < c.1.min(c.2)) >= 40, "rank-deficient");
    for (_, m, n, _, a) in set.iter().filter(|c| matches!(c.0, Class::IllConditioned)) {
        let cond = nalgebra_condition(a, *m, *n);
        assert!(cond >= 1e8, "{m}x{n} ill-conditioned case has cond {cond:e} < 1e8");
    }
}

// FALSIFY-SVD-005: past the rank, singular values fall under the documented tolerance.
#[test]
fn falsify_svd_005_rank_deficient_values_are_within_tolerance() {
    for (case, (_, m, n, rank, a)) in acceptance_set().iter().enumerate() {
        if *rank == (*m).min(*n) {
            continue;
        }
        let svd = Svd::new(a, *m, *n).expect("decomposes");
        let tol = svd.rank_tolerance();
        let s = svd.singular_values();
        assert!(
            s[*rank..].iter().all(|&x| x <= tol),
            "case {case}: tail {:?} above {tol:e}",
            &s[*rank..]
        );
        assert_eq!(svd.rank(), *rank, "case {case} {m}x{n}: numerical rank of {s:?}");
    }
}

#[test]
fn zero_matrix_has_zero_values_and_orthonormal_factors() {
    let a = vec![0.0; 12];
    for (m, n) in [(3, 4), (4, 3)] {
        let svd = Svd::new(&a, m, n).expect("decomposes");
        assert!(svd.singular_values().iter().all(|&x| x == 0.0));
        assert_eq!(svd.rank(), 0);
        assert_well_formed(&a, m, n, &svd, &format!("zero {m}x{n}"));
    }
}

#[test]
fn identical_input_gives_identical_bits() {
    let a = gaussian(7, 5, 42);
    assert_eq!(Svd::new(&a, 7, 5).expect("ok"), Svd::new(&a, 7, 5).expect("ok"));
}

#[test]
fn invalid_input_is_rejected() {
    let bad = |r: Result<Svd, TruenoError>| matches!(r, Err(TruenoError::InvalidInput(_)));
    assert!(bad(Svd::new(&[], 0, 3)));
    assert!(bad(Svd::new(&[], 3, 0)));
    assert!(bad(Svd::new(&[1.0; 5], 2, 3)));
    assert!(bad(Svd::new(&[1.0, f64::NAN, 1.0, 1.0], 2, 2)));
    assert!(bad(Svd::new(&[1.0, f64::INFINITY, 1.0, 1.0], 2, 2)));
    let a = gaussian(4, 6, 1);
    assert!(bad(Svd::randomized(&a, 4, 6, RandomizedSvdConfig::new(0))));
    assert!(bad(Svd::randomized(&a, 4, 6, RandomizedSvdConfig::new(5))));
}

#[derive(serde::Deserialize)]
struct OracleCase {
    name: String,
    rows: usize,
    cols: usize,
    a: Vec<f64>,
    s: Vec<f64>,
    u: Vec<f64>,
    vt: Vec<f64>,
    vectors: usize,
}

#[derive(serde::Deserialize)]
struct Oracle {
    cases: Vec<OracleCase>,
}

// FALSIFY-SVD-004: parity with scipy.linalg.svd within 1e-9 relative.
#[test]
fn falsify_svd_004_scipy_oracle_parity() {
    let oracle: Oracle =
        serde_json::from_str(include_str!("../../tests/fixtures/svd_scipy_v1.json"))
            .expect("fixture parses");
    assert!(oracle.cases.len() >= 10, "fixture lost cases");
    for c in &oracle.cases {
        let svd = Svd::new(&c.a, c.rows, c.cols).expect("decomposes");
        let (k, scale) = (c.s.len(), c.s[0]);
        assert_eq!(svd.len(), k, "{}", c.name);
        for (j, (got, want)) in svd.singular_values().iter().zip(&c.s).enumerate() {
            let rel = (got - want).abs() / scale;
            assert!(rel <= 1e-9, "{}: sigma_{j} = {got} vs scipy {want} (rel {rel:e})", c.name);
        }
        for j in 0..c.vectors {
            for i in 0..c.rows {
                let (got, want) = (svd.u()[i * k + j], c.u[i * k + j]);
                assert!(
                    (got - want).abs() <= 1e-9,
                    "{}: U[{i},{j}] = {got} vs scipy {want}",
                    c.name
                );
            }
            for i in 0..c.cols {
                let (got, want) = (svd.vt()[j * c.cols + i], c.vt[j * c.cols + i]);
                assert!(
                    (got - want).abs() <= 1e-9,
                    "{}: Vt[{j},{i}] = {got} vs scipy {want}",
                    c.name
                );
            }
        }
    }
}

/// Top-`k` values of a randomized SVD against the exact ones, as the worst relative error.
fn randomized_error(a: &[f64], m: usize, n: usize, k: usize) -> f64 {
    let exact = Svd::new(a, m, n).expect("exact");
    let approx = Svd::randomized(a, m, n, RandomizedSvdConfig::new(k)).expect("randomized");
    assert_eq!((approx.len(), approx.u().len(), approx.vt().len()), (k, m * k, k * n));
    assert!(column_orthonormality_error(approx.u(), m, k) <= TOL);
    assert!(column_orthonormality_error(&transpose(approx.vt(), k, n), n, k) <= TOL);
    assert_sign_convention(&approx, "randomized");
    approx
        .singular_values()
        .iter()
        .zip(exact.singular_values())
        .map(|(x, y)| (x - y).abs() / y)
        .fold(0.0, f64::max)
}

// FALSIFY-SVD-006: p = 10, q = 2 recovers the top-k singular values within 1%
// wherever the spectrum has decayed by half across the oversampling window,
// sigma_{k+p+1} / sigma_k <= 0.5 (the domain `RandomizedSvdConfig` documents).
#[test]
fn falsify_svd_006_randomized_top_k_within_one_percent() {
    let (m, n) = (160, 120);
    // (name, spectrum, k values inside the domain): geometric 0.9^i has ratio
    // 0.9^11 = 0.31 at every k; harmonic 1/(i+1) has ratio (k+1)/(k+11) <= 0.5 for k <= 10.
    let geometric: Vec<f64> = (0..n).map(|i| 0.9f64.powi(i as i32)).collect();
    let harmonic: Vec<f64> = (0..n).map(|i| 1.0 / (1 + i) as f64).collect();
    for (name, sigma, ks) in
        [("geometric", &geometric, &[1, 5, 10, 20][..]), ("harmonic", &harmonic, &[1, 5, 10][..])]
    {
        let a = with_spectrum(m, n, sigma, 77);
        for &k in ks {
            let ratio = sigma[k + 10] / sigma[k - 1];
            assert!(ratio <= 0.5, "{name} k={k} is outside the documented domain (ratio {ratio})");
            let err = randomized_error(&a, m, n, k);
            assert!(err <= 0.01, "{name} spectrum, k={k}: worst top-k relative error {err:e} > 1%");
        }
        let wide = transpose(&a, m, n);
        let err = randomized_error(&wide, n, m, 10);
        assert!(err <= 0.01, "{name} spectrum, wide, k=10: {err:e} > 1%");
    }
}

/// Outside the domain the estimate degrades but stays bounded: harmonic k = 20
/// has ratio 21/31 = 0.68 and measured 1.8% at p = 10, q = 2. More power
/// iterations are the documented remedy, and they must bring it back under 1%.
#[test]
fn randomized_outside_the_decay_domain_needs_more_power_iterations() {
    let (m, n) = (160, 120);
    let harmonic: Vec<f64> = (0..n).map(|i| 1.0 / (1 + i) as f64).collect();
    let a = with_spectrum(m, n, &harmonic, 77);
    let exact = Svd::new(&a, m, n).expect("exact");
    let worst = |q: usize| {
        let cfg = RandomizedSvdConfig { power_iters: q, ..RandomizedSvdConfig::new(20) };
        let approx = Svd::randomized(&a, m, n, cfg).expect("randomized");
        approx
            .singular_values()
            .iter()
            .zip(exact.singular_values())
            .map(|(x, y)| (x - y).abs() / y)
            .fold(0.0, f64::max)
    };
    let (default_q, more_q) = (worst(2), worst(6));
    assert!(default_q <= 0.05, "q=2 outside the domain: {default_q:e} > 5%");
    assert!(more_q <= 0.01, "q=6 should recover 1% on harmonic k=20, got {more_q:e}");
}

#[test]
fn randomized_is_seeded_and_exact_at_full_rank() {
    let a = gaussian(9, 6, 5);
    let cfg = RandomizedSvdConfig::new(6);
    assert_eq!(
        Svd::randomized(&a, 9, 6, cfg).expect("ok"),
        Svd::randomized(&a, 9, 6, cfg).expect("ok")
    );
    let full = Svd::randomized(&a, 9, 6, cfg).expect("ok");
    assert!(relative_reconstruction_error(&a, &full) <= TOL);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    // FALSIFY-SVD-002 / 003: orthonormal factors, non-negative descending values.
    #[test]
    fn falsify_svd_002_003_orthonormal_and_sorted(
        (m, n, a) in (1usize..10, 1usize..10).prop_flat_map(|(m, n)| {
            (Just(m), Just(n), proptest::collection::vec(-1e3f64..1e3, m * n))
        })
    ) {
        let svd = Svd::new(&a, m, n).expect("decomposes");
        assert_well_formed(&a, m, n, &svd, &format!("proptest {m}x{n}"));
    }
}
