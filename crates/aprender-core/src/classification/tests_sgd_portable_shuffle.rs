//! Falsifiers for the SGD epoch-shuffle LCG. Refs #2310.
//!
//! `fit_stochastic` / `fit_minibatch` shuffle the sample order each epoch with a
//! Fisher-Yates pass whose partner index comes from the MMIX LCG constants
//! (`6364136223846793005`, `1442695040888963407`). Those constants are 64-bit.
//! They were written as bare integer literals in a `usize` expression, which:
//!
//!   * is a **hard compile error** on 32-bit targets — `error: literal out of
//!     range for usize` on `wasm32-unknown-unknown` (the #2310 report), and
//!   * **panics** in any overflow-checked (debug/test) build on 64-bit, because
//!     `seed * 6364136223846793005` overflows `u64` from `seed == 3` onwards and
//!     `i * 1442695040888963407` overflows from `i == 13` onwards.
//!
//! Both `FitMode::Stochastic` and `FitMode::MiniBatch` had zero test coverage, so
//! the 64-bit panic shipped undetected. These tests state what the code must NOT
//! do: it must not panic, must not return a non-permutation, and must not change
//! the 64-bit release-mode partner sequence that the fix preserves bit-for-bit.

use super::{shuffle_partner, FitMode, LogisticRegression};
use crate::primitives::Matrix;

/// Linearly separable 2-D problem with enough rows that the Fisher-Yates loop
/// reaches `i == 13`, the first index at which `i * 1442695040888963407`
/// overflows `u64`.
fn separable_dataset() -> (Matrix<f32>, Vec<usize>) {
    let mut rows = Vec::new();
    let mut labels = Vec::new();
    for k in 0..20 {
        let t = k as f32;
        rows.push(vec![t * 0.1 - 2.0, 0.5]);
        labels.push(usize::from(k >= 10));
    }
    let flat: Vec<f32> = rows.into_iter().flatten().collect();
    let x = Matrix::from_vec(20, 2, flat).expect("20x2 matrix");
    (x, labels)
}

/// The partner index must always land inside `[0, i]`, otherwise `indices.swap`
/// would either panic or reach outside the unshuffled prefix and destroy the
/// permutation. Exercised well past the overflow thresholds (`seed >= 3`,
/// `i >= 13`).
#[test]
fn test_shuffle_partner_never_exceeds_i() {
    for seed in 0..64usize {
        for i in 1..256usize {
            let j = shuffle_partner(seed, i);
            assert!(
                j <= i,
                "shuffle_partner({seed}, {i}) = {j} escaped the [0, {i}] window"
            );
        }
    }
}

/// Contract `apr-stochastic-lr-v1.yaml` — `minibatch_gradient` invariant
/// "Each sample seen exactly once per epoch". A Fisher-Yates pass over
/// `shuffle_partner` must yield a permutation, never a multiset with repeats.
#[test]
fn test_epoch_shuffle_is_a_permutation() {
    let n_samples = 64usize;
    for seed in 0..16usize {
        let mut indices: Vec<usize> = (0..n_samples).collect();
        for i in (1..n_samples).rev() {
            let j = shuffle_partner(seed, i);
            indices.swap(i, j);
        }
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            (0..n_samples).collect::<Vec<usize>>(),
            "epoch seed {seed} produced a non-permutation: {indices:?}"
        );
    }
}

/// Behaviour pin: the fix moves the arithmetic into explicit `u64` wrapping ops,
/// which must reproduce the pre-#2310 64-bit **release** result exactly, so no
/// already-trained model's epoch order shifts. Expected values computed from the
/// wrapping 64-bit definition `(seed*MUL + i*INC) mod (i+1)`.
#[test]
fn test_shuffle_partner_matches_64bit_wrapping_reference() {
    let cases: [(usize, usize, usize); 7] = [
        (0, 1, 1),
        (1, 1, 0),
        (3, 2, 1),
        (3, 7, 0),
        (4, 12, 6),
        (7, 13, 8),
        (999, 255, 76),
    ];
    for (seed, i, expected) in cases {
        assert_eq!(
            shuffle_partner(seed, i),
            expected,
            "shuffle_partner({seed}, {i}) drifted from the 64-bit wrapping reference"
        );
    }
}

/// #2310 regression, 64-bit half: with the default `max_iter` of 1000 the epoch
/// seed passes 3 and the Fisher-Yates index passes 13, so the pre-fix `usize`
/// multiplication aborts the process in any overflow-checked build.
#[test]
fn test_stochastic_fit_survives_overflowing_epoch_and_index() {
    let (x, y) = separable_dataset();
    let mut model = LogisticRegression::new().with_fit_mode(FitMode::Stochastic);
    model.fit(&x, &y).expect("stochastic fit must succeed");
    let acc = model.score(&x, &y);
    assert!(
        acc > 0.9,
        "stochastic fit on a separable set scored {acc}, below the 0.9 floor"
    );
}

/// #2310 regression, mini-batch path (a second, independently-compiled copy of
/// the same expression lived in `fit_minibatch`).
#[test]
fn test_minibatch_fit_survives_overflowing_epoch_and_index() {
    let (x, y) = separable_dataset();
    let mut model = LogisticRegression::new().with_fit_mode(FitMode::MiniBatch(4));
    model.fit(&x, &y).expect("mini-batch fit must succeed");
    let acc = model.score(&x, &y);
    assert!(
        acc > 0.9,
        "mini-batch fit on a separable set scored {acc}, below the 0.9 floor"
    );
}
