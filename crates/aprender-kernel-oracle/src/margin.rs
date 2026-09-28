//! The per-element margin test (KTEST-001 §3.2), with NMSE reported as a secondary signal.
//!
//! mᵢ = |ŷᵢ − yᵢ| / Bᵢ. A kernel passes iff
//! - max mᵢ ≤ 1,
//! - it produces no NaN the oracle does not (and propagates every NaN the oracle does), and
//! - its Inf set and signs match the oracle's, where the oracle's value counts as Inf when it
//!   overflows the kernel's f32 output.
//!
//! An element with Bᵢ = 0 (bit-exact models, all-zero inputs) passes only on an exact match.

use crate::error_model::{ErrorModel, Refusal};

/// llama.cpp `test-backend-ops` default max NMSE, for receipts comparable with it (§1).
pub const LLAMA_CPP_NMSE_DEFAULT: f64 = 1e-7;
/// llama.cpp's override for `mul_mat`, `mul_mat_id` and `out_prod`.
pub const LLAMA_CPP_NMSE_MUL_MAT: f64 = 5e-4;

/// The first reason a kernel output is RED.
#[derive(Debug, Clone, PartialEq)]
pub enum Failure {
    /// mᵢ > 1 at `index` (the element with the largest margin).
    MarginExceeded { index: usize, margin: f64 },
    /// The kernel produced NaN where the oracle is a number.
    UnexpectedNaN { index: usize },
    /// The oracle is NaN (a NaN input) and the kernel did not propagate it.
    MissingNaN { index: usize },
    /// Exactly one side is ±Inf, or both are with opposite signs.
    InfMismatch { index: usize },
}

/// Pass, or the first failure found.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    Pass,
    Fail(Failure),
}

/// What a kernel receipt records per shape class (§3.2).
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    /// Elements judged.
    pub n: usize,
    /// max mᵢ over the elements where a margin is defined (finite on both sides).
    pub max_margin: f64,
    /// Where `max_margin` occurred: the tile to look at when it is RED.
    pub argmax: Option<usize>,
    /// The 99.9th percentile of mᵢ (nearest rank).
    pub p999_margin: f64,
    /// Σ(ŷ − y)² / Σy² over finite elements: llama.cpp-comparable, never the gate.
    pub nmse: f64,
    pub verdict: Verdict,
}

impl Report {
    #[must_use]
    pub fn passed(&self) -> bool {
        self.verdict == Verdict::Pass
    }
}

/// Classify one element: `Ok(Some(m))` for a margin, `Ok(None)` for a matched NaN/Inf.
fn element(i: usize, yhat: f32, y: f64, bound: f64) -> Result<Option<f64>, Failure> {
    let yh = f64::from(yhat);
    if y.is_nan() {
        return if yh.is_nan() {
            Ok(None)
        } else {
            Err(Failure::MissingNaN { index: i })
        };
    }
    if yh.is_nan() {
        return Err(Failure::UnexpectedNaN { index: i });
    }
    #[allow(clippy::cast_possible_truncation)] // the point: does y overflow the f32 output?
    let y_out = y as f32;
    if y_out.is_infinite() || yh.is_infinite() {
        return if y_out == yhat {
            Ok(None)
        } else {
            Err(Failure::InfMismatch { index: i })
        };
    }
    let diff = (yh - y).abs();
    let m = if bound > 0.0 {
        diff / bound
    } else if diff == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    Ok(Some(m))
}

/// Judge a kernel's outputs against the oracle with explicit per-element bounds.
///
/// # Errors
/// [`Refusal::LengthMismatch`] when the three buffers differ in length.
pub fn judge(yhat: &[f32], oracle: &[f64], bounds: &[f64]) -> Result<Report, Refusal> {
    if yhat.len() != oracle.len() || oracle.len() != bounds.len() {
        return Err(Refusal::LengthMismatch {
            yhat: yhat.len(),
            oracle: oracle.len(),
            bound: bounds.len(),
        });
    }
    let mut first_failure = None;
    let mut margins = Vec::with_capacity(yhat.len());
    let (mut max_margin, mut argmax) = (0.0_f64, None);
    let (mut err2, mut ref2) = (0.0_f64, 0.0_f64);
    for (i, ((&yh, &y), &b)) in yhat.iter().zip(oracle).zip(bounds).enumerate() {
        match element(i, yh, y, b) {
            Ok(Some(m)) => {
                if argmax.is_none() || m > max_margin {
                    max_margin = m;
                    argmax = Some(i);
                }
                margins.push(m);
                err2 += (f64::from(yh) - y).powi(2);
                ref2 += y * y;
            }
            Ok(None) => {}
            Err(f) => {
                first_failure.get_or_insert(f);
            }
        }
    }
    margins.sort_by(f64::total_cmp);
    let p999_margin = if margins.is_empty() {
        0.0
    } else {
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )] // nearest rank of a slice length: positive and far below 2^53
        let rank = ((0.999 * margins.len() as f64).ceil() as usize).clamp(1, margins.len());
        margins[rank - 1]
    };
    let nmse = if ref2 > 0.0 {
        err2 / ref2
    } else if err2 == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    let verdict = match (first_failure, argmax) {
        (Some(f), _) => Verdict::Fail(f),
        (None, Some(index)) if max_margin > 1.0 => Verdict::Fail(Failure::MarginExceeded {
            index,
            margin: max_margin,
        }),
        (None, _) => Verdict::Pass,
    };
    Ok(Report {
        n: yhat.len(),
        max_margin,
        argmax,
        p999_margin,
        nmse,
        verdict,
    })
}

/// Judge against a declared error model: Bᵢ = model bound at `magnitudes[i]` (Σ|a||b| per output
/// for `EM-DOT`, Σ|x| for a sum; anything for the zero-bound models).
///
/// # Errors
/// The model's [`Refusal`] (vacuous accumulator, unimplemented model), or a length mismatch.
pub fn judge_model(
    yhat: &[f32],
    oracle: &[f64],
    magnitudes: &[f64],
    model: &ErrorModel,
) -> Result<Report, Refusal> {
    let bounds = model.bound()?.over(magnitudes);
    judge(yhat, oracle, &bounds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error_model::Dtype;
    use crate::oracle;

    /// splitmix64: a seeded, dependency-free generator so every failure reproduces from its seed.
    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Uniform in [lo, hi).
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
            let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            (f64::from(lo) + u * f64::from(hi - lo)) as f32
        }

        #[allow(clippy::cast_possible_truncation)]
        fn below(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// A GEMV "kernel" summing in f32 in the given order, with or without FMA.
    fn f32_dot_in_order(w: &[f32], x: &[f32], order: &[usize], fma: bool) -> f32 {
        order.iter().fold(0.0_f32, |acc, &j| {
            if fma {
                w[j].mul_add(x[j], acc)
            } else {
                acc + w[j] * x[j]
            }
        })
    }

    fn f32_pairwise(terms: &[f32]) -> f32 {
        match terms.len() {
            0 => 0.0,
            1 => terms[0],
            n => f32_pairwise(&terms[..n / 2]) + f32_pairwise(&terms[n / 2..]),
        }
    }

    fn dot_model(k: usize, acc: Dtype) -> ErrorModel {
        ErrorModel::Dot {
            k,
            acc,
            out: Dtype::F32,
            ftz: false,
        }
    }

    /// F-12 (KTEST-001 §7): one bad tile in 10⁶ outputs. A global NMSE passes llama.cpp's
    /// strictest default threshold; the per-element margin is RED and names the tile.
    /// Anti-vacuity arm: the same kernel without the planted tile is GREEN.
    #[test]
    fn f12_nmse_passes_one_bad_tile_the_margin_does_not() {
        const ROWS: usize = 1_000_000;
        const K: usize = 16;
        const TILE: usize = 16;
        let mut rng = Rng(0x4B54_4553_5402);
        let w: Vec<f32> = (0..ROWS * K).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let x: Vec<f32> = (0..K).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let order: Vec<usize> = (0..K).collect();
        let good: Vec<f32> = w
            .chunks_exact(K)
            .map(|row| f32_dot_in_order(row, &x, &order, false))
            .collect();
        let y = oracle::gemv(&w, ROWS, K, &x);
        let mags = oracle::gemv_abs(&w, ROWS, K, &x);
        let model = dot_model(K, Dtype::F32);

        let clean = judge_model(&good, &y, &mags, &model).expect("shapes match");
        assert!(
            clean.passed(),
            "anti-vacuity: a correct f32 GEMV must pass: {clean:?}"
        );
        assert!(clean.max_margin <= 1.0);

        // the planted defect: one 16-output tile off by 0.1% (a wrong scale in one block)
        let start = 424_242 / TILE * TILE;
        let mut bad = good.clone();
        for v in &mut bad[start..start + TILE] {
            *v *= 1.001;
        }
        let r = judge_model(&bad, &y, &mags, &model).expect("shapes match");
        assert!(
            r.nmse < LLAMA_CPP_NMSE_DEFAULT,
            "F-12 premise: NMSE must PASS llama.cpp's 1e-7, got {:e}",
            r.nmse
        );
        match r.verdict {
            Verdict::Fail(Failure::MarginExceeded { index, margin }) => {
                assert!(
                    (start..start + TILE).contains(&index),
                    "names the tile: {index}"
                );
                assert!(margin > 1.0);
            }
            v => panic!("F-12: the margin must be RED while NMSE passes, got {v:?}"),
        }
    }

    /// Theorem K1a: one bound holds for every summation order and bracketing, with or without FMA,
    /// on the adversarial inputs of §3.3 (cancellation pairs, wide dynamic range).
    #[test]
    fn k1a_bound_holds_for_every_order() {
        let mut rng = Rng(0x004B_3141);
        for &k in &[1_usize, 2, 7, 64, 1000, 4096, 65_536] {
            let mut w: Vec<f32> = Vec::with_capacity(k);
            let mut x: Vec<f32> = Vec::with_capacity(k);
            for j in 0..k {
                #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                let scale = 10f32.powi(rng.below(9) as i32 - 4); // 1e-4 … 1e4
                let v = rng.uniform(-1.0, 1.0) * scale;
                // every other term cancels its neighbour's product exactly
                if j % 2 == 1 {
                    w.push(-w[j - 1]);
                    x.push(x[j - 1]);
                } else {
                    w.push(v);
                    x.push(rng.uniform(0.5, 2.0));
                }
            }
            let y = [oracle::dot(&w, &x)];
            let mag = [oracle::dot_abs(&w, &x)];
            let model = dot_model(k, Dtype::F32);
            let mut order: Vec<usize> = (0..k).collect();
            for trial in 0..8 {
                // Fisher–Yates: a fresh random order per trial
                for i in (1..k).rev() {
                    order.swap(i, rng.below(i + 1));
                }
                for fma in [false, true] {
                    let yh = [f32_dot_in_order(&w, &x, &order, fma)];
                    let r = judge_model(&yh, &y, &mag, &model).expect("one element");
                    assert!(r.passed(), "K = {k}, trial {trial}, fma {fma}: {r:?}");
                }
            }
            let products: Vec<f32> = w.iter().zip(&x).map(|(a, b)| a * b).collect();
            let r = judge_model(&[f32_pairwise(&products)], &y, &mag, &model).expect("one");
            assert!(r.passed(), "K = {k}, pairwise tree: {r:?}");
        }
    }

    /// Rounds an f32 in the f16 normal range to f16 precision (11 significant bits).
    fn to_f16_precision(v: f32) -> f32 {
        if v == 0.0 {
            return 0.0;
        }
        let e = v.abs().log2().floor();
        let q = (e - 10.0).exp2();
        (v / q).round_ties_even() * q
    }

    /// F-2 (KTEST-001 §7): an f16 accumulator in a K = 4096 GEMV. Declared honestly it is refused
    /// (K1b); declared as f32 the margin is RED. Anti-vacuity: the f32 accumulator is GREEN.
    #[test]
    fn f2_f16_accumulator_is_red_or_refused() {
        const K: usize = 4096;
        let mut rng = Rng(0xF2);
        let w: Vec<f32> = (0..K).map(|_| rng.uniform(0.0, 1.0)).collect();
        let x: Vec<f32> = (0..K).map(|_| rng.uniform(0.0, 1.0)).collect();
        let order: Vec<usize> = (0..K).collect();
        let y = [oracle::dot(&w, &x)];
        let mag = [oracle::dot_abs(&w, &x)];
        let f16_acc = (0..K).fold(0.0_f32, |acc, j| {
            to_f16_precision(acc + to_f16_precision(w[j] * x[j]))
        });
        let f32_acc = f32_dot_in_order(&w, &x, &order, false);

        let honest = judge_model(&[f16_acc], &y, &mag, &dot_model(K, Dtype::F16));
        assert!(matches!(
            honest,
            Err(Refusal::Vacuous {
                n: K,
                acc: Dtype::F16,
                ..
            })
        ));
        let lying = judge_model(&[f16_acc], &y, &mag, &dot_model(K, Dtype::F32)).expect("one");
        assert!(
            matches!(lying.verdict, Verdict::Fail(Failure::MarginExceeded { .. })),
            "an f16 accumulator declared f32 must be RED: {lying:?}"
        );
        let good = judge_model(&[f32_acc], &y, &mag, &dot_model(K, Dtype::F32)).expect("one");
        assert!(good.passed(), "anti-vacuity: {good:?}");
    }

    #[test]
    fn nan_and_inf_rules() {
        let b = [1.0, 1.0];
        let red = |yhat: [f32; 2], y: [f64; 2]| judge(&yhat, &y, &b).expect("lengths").verdict;
        assert_eq!(
            red([f32::NAN, 0.0], [0.0, 0.0]),
            Verdict::Fail(Failure::UnexpectedNaN { index: 0 })
        );
        assert_eq!(
            red([0.0, 1.0], [0.0, f64::NAN]),
            Verdict::Fail(Failure::MissingNaN { index: 1 })
        );
        assert_eq!(red([0.0, f32::NAN], [0.0, f64::NAN]), Verdict::Pass);
        assert_eq!(
            red([f32::NEG_INFINITY, 0.0], [f64::INFINITY, 0.0]),
            Verdict::Fail(Failure::InfMismatch { index: 0 })
        );
        assert_eq!(
            red([f32::INFINITY, 0.0], [0.0, 0.0]),
            Verdict::Fail(Failure::InfMismatch { index: 0 })
        );
        // an oracle value past f32::MAX is Inf in the kernel's output type
        assert_eq!(red([f32::INFINITY, 0.0], [1e39, 0.0]), Verdict::Pass);
        assert_eq!(
            red([f32::MAX, 0.0], [1e39, 0.0]),
            Verdict::Fail(Failure::InfMismatch { index: 0 })
        );
    }

    #[test]
    fn zero_bound_is_bit_exact() {
        let model = ErrorModel::Dequant;
        let exact = judge_model(&[0.5, -2.0], &[0.5, -2.0], &[9.0, 9.0], &model).expect("len");
        assert!(exact.passed());
        let off =
            judge_model(&[0.5, -2.0], &[0.5, -2.000_000_1], &[9.0, 9.0], &model).expect("len");
        assert!(matches!(
            off.verdict,
            Verdict::Fail(Failure::MarginExceeded { index: 1, margin }) if margin.is_infinite()
        ));
    }

    #[test]
    fn margin_boundary_is_inclusive() {
        let at = judge(&[1.5], &[1.0], &[0.5]).expect("len");
        assert!(at.passed(), "m = 1 passes: {at:?}");
        let over = judge(&[1.5], &[1.0], &[0.499_999]).expect("len");
        assert!(!over.passed());
    }

    #[test]
    fn report_statistics() {
        // margins 0.000, 0.001, …, 0.999 → p99.9 (nearest rank 999 of 1000) = 0.998
        let n = 1000;
        let y = vec![0.0; n];
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        let yhat: Vec<f32> = (0..n).map(|i| i as f32 / 1000.0).collect();
        let r = judge(&yhat, &y, &vec![1.0; n]).expect("len");
        assert!(r.passed());
        assert_eq!(r.argmax, Some(n - 1));
        assert!((r.max_margin - f64::from(yhat[n - 1])).abs() < 1e-12);
        assert!((r.p999_margin - f64::from(yhat[n - 2])).abs() < 1e-12);
        assert!(r.nmse.is_infinite(), "Σy² = 0 with nonzero error");
        let empty = judge(&[], &[], &[]).expect("len");
        assert!(empty.passed() && empty.nmse == 0.0 && empty.argmax.is_none());
    }

    #[test]
    fn length_mismatch_is_refused() {
        assert_eq!(
            judge(&[1.0], &[1.0, 2.0], &[1.0]),
            Err(Refusal::LengthMismatch {
                yhat: 1,
                oracle: 2,
                bound: 1
            })
        );
    }
}
