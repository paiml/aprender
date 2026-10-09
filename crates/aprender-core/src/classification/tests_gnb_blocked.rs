// =========================================================================
// GaussianNB row-major fit and blocked predict (#4965) are bit-identical
// to the per-feature strided loops they replaced.
//
// The reference functions below are those loops, kept verbatim. Every
// statistic and every log-posterior is compared by `to_bits`, so a change
// of one rounding step anywhere is RED, not only a flipped prediction.
// =========================================================================

use super::*;

/// Deterministic data: `n × d` values in about [-4, 4], shifted per class, with
/// exact zeros of both signs and repeated values sprinkled in.
fn data(n: usize, d: usize, n_classes: usize, seed: u64) -> (Matrix<f32>, Vec<usize>) {
    let mut s = seed;
    let mut next = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (s >> 33) as u32
    };
    let mut y = Vec::with_capacity(n);
    let mut v = Vec::with_capacity(n * d);
    for i in 0..n {
        let class = if i < n_classes {
            i
        } else {
            next() as usize % n_classes
        };
        y.push(class);
        for f in 0..d {
            let r = next();
            let x = match r % 16 {
                0 => -0.0,
                1 => 0.0,
                2 => 1.5,
                _ => (r % 8001) as f32 / 1000.0 - 4.0 + class as f32 * 0.37 + f as f32 * 0.01,
            };
            v.push(x);
        }
    }
    (Matrix::from_vec(n, d, v).expect("n * d values"), y)
}

/// The fit statistics as the strided loops computed them before #4965.
fn ref_fit(
    x: &Matrix<f32>,
    y: &[usize],
    var_smoothing: f32,
) -> (Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let (n_samples, n_features) = x.shape();
    let mut classes: Vec<usize> = y.to_vec();
    classes.sort_unstable();
    classes.dedup();
    let n_classes = classes.len();
    let mut max_feature_var = 0.0_f32;
    for feature_idx in 0..n_features {
        let mut sum = 0.0_f32;
        for sample_idx in 0..n_samples {
            sum += x.get(sample_idx, feature_idx);
        }
        let mean = sum / n_samples as f32;
        let mut sum_sq_diff = 0.0_f32;
        for sample_idx in 0..n_samples {
            let diff = x.get(sample_idx, feature_idx) - mean;
            sum_sq_diff += diff * diff;
        }
        let feature_var = sum_sq_diff / n_samples as f32;
        if feature_var > max_feature_var {
            max_feature_var = feature_var;
        }
    }
    let epsilon = var_smoothing * max_feature_var;
    let mut class_priors = vec![0.0; n_classes];
    let mut means = vec![vec![0.0; n_features]; n_classes];
    let mut variances = vec![vec![0.0; n_features]; n_classes];
    for (class_idx, &class_label) in classes.iter().enumerate() {
        let class_samples: Vec<usize> = y
            .iter()
            .enumerate()
            .filter_map(|(i, &label)| if label == class_label { Some(i) } else { None })
            .collect();
        let n_class_samples = class_samples.len() as f32;
        class_priors[class_idx] = n_class_samples / n_samples as f32;
        for (feature_idx, mean_val) in means[class_idx].iter_mut().enumerate() {
            let sum: f32 = class_samples
                .iter()
                .map(|&sample_idx| x.get(sample_idx, feature_idx))
                .sum();
            *mean_val = sum / n_class_samples;
        }
        for (feature_idx, variance_val) in variances[class_idx].iter_mut().enumerate() {
            let mean = means[class_idx][feature_idx];
            let sum_sq_diff: f32 = class_samples
                .iter()
                .map(|&sample_idx| {
                    let diff = x.get(sample_idx, feature_idx) - mean;
                    diff * diff
                })
                .sum();
            *variance_val = sum_sq_diff / n_class_samples + epsilon;
        }
    }
    (class_priors, means, variances)
}

/// Every (sample, class) log-posterior as the per-sample scalar loop computed it before #4965.
fn ref_log_posteriors(model: &GaussianNB, x: &Matrix<f32>) -> Vec<Vec<f32>> {
    let means = model.means.as_ref().expect("fitted");
    let variances = model.variances.as_ref().expect("fitted");
    let priors = model.class_priors.as_ref().expect("fitted");
    let const_term = GaussianNB::log_const_terms(priors, variances);
    let (n_samples, n_features) = x.shape();
    (0..n_samples)
        .map(|sample_idx| {
            (0..means.len())
                .map(|class_idx| {
                    let mut lp = const_term[class_idx];
                    for feature_idx in 0..n_features {
                        let inv = 1.0 / (2.0 * variances[class_idx][feature_idx]);
                        let diff = x.get(sample_idx, feature_idx) - means[class_idx][feature_idx];
                        lp -= diff * diff * inv;
                    }
                    lp
                })
                .collect()
        })
        .collect()
}

/// Every (sample, class) log-posterior through the blocked kernel `predict` uses.
fn blocked_log_posteriors(model: &GaussianNB, x: &Matrix<f32>) -> Vec<Vec<f32>> {
    let means = model.means.as_ref().expect("fitted");
    let variances = model.variances.as_ref().expect("fitted");
    let priors = model.class_priors.as_ref().expect("fitted");
    let const_term = GaussianNB::log_const_terms(priors, variances);
    let inv: Vec<Vec<f32>> = variances
        .iter()
        .map(|vc| vc.iter().map(|&v| 1.0 / (2.0 * v)).collect())
        .collect();
    let (n_samples, d) = x.shape();
    let mut out = vec![vec![0.0_f32; means.len()]; n_samples];
    for start in (0..n_samples).step_by(GNB_PREDICT_BLOCK) {
        let b = GNB_PREDICT_BLOCK.min(n_samples - start);
        let mut tile = vec![f32::NAN; d * b];
        gnb_transpose_block(&x.as_slice()[start * d..(start + b) * d], b, &mut tile);
        let mut lp = vec![f32::NAN; b];
        for c in 0..means.len() {
            gnb_class_log_posterior(&tile, const_term[c], &means[c], &inv[c], &mut lp);
            for (j, &l) in lp.iter().enumerate() {
                out[start + j][c] = l;
            }
        }
    }
    out
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Shapes that cross the block edge (127/128/129 rows) and a partial last block.
const SHAPES: [(usize, usize, usize); 8] = [
    (2, 1, 2),
    (127, 3, 2),
    (128, 30, 8),
    (129, 30, 8),
    (300, 7, 3),
    (1000, 30, 8),
    (257, 1, 5),
    (64, 2, 2),
];

#[test]
fn row_major_fit_is_bit_identical_to_the_strided_fit_4965() {
    for (i, &(n, d, k)) in SHAPES.iter().enumerate() {
        for vs in [1e-9_f32, 1e-2] {
            let (x, y) = data(n, d, k, 7 + i as u64);
            let mut model = GaussianNB::new().with_var_smoothing(vs);
            model.fit(&x, &y).expect("fit");
            let (priors, means, variances) = ref_fit(&x, &y, vs);
            let got_priors = model.class_priors.as_ref().expect("fitted");
            assert_eq!(bits(got_priors), bits(&priors), "priors n={n} d={d} k={k}");
            for c in 0..k {
                let got_mean = &model.means.as_ref().expect("fitted")[c];
                let got_var = &model.variances.as_ref().expect("fitted")[c];
                assert_eq!(
                    bits(got_mean),
                    bits(&means[c]),
                    "means n={n} d={d} k={k} c={c}"
                );
                assert_eq!(
                    bits(got_var),
                    bits(&variances[c]),
                    "variances n={n} d={d} k={k} c={c}"
                );
            }
        }
    }
}

#[test]
fn an_all_negative_zero_column_keeps_its_sign_4965() {
    // Iterator::sum of only -0.0 is -0.0; the row-major fit must start from the same value.
    let x = Matrix::from_vec(4, 2, vec![-0.0, 1.0, -0.0, 2.0, -0.0, 5.0, -0.0, 6.0]).expect("4x2");
    let y = vec![0, 0, 1, 1];
    let mut model = GaussianNB::new();
    model.fit(&x, &y).expect("fit");
    let (_, means, _) = ref_fit(&x, &y, 1e-9);
    let got = model.means.as_ref().expect("fitted");
    for c in 0..2 {
        assert_eq!(bits(&got[c]), bits(&means[c]), "class {c}");
    }
}

#[test]
fn blocked_log_posterior_is_bit_identical_to_the_scalar_loop_4965() {
    for (i, &(n, d, k)) in SHAPES.iter().enumerate() {
        let (x, y) = data(n, d, k, 101 + i as u64);
        let mut model = GaussianNB::new();
        model.fit(&x, &y).expect("fit");
        // Score rows the model was not fitted on too.
        let (xt, _) = data(n, d, k, 9001 + i as u64);
        for m in [&x, &xt] {
            let want = ref_log_posteriors(&model, m);
            let got = blocked_log_posteriors(&model, m);
            for (s, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(bits(g), bits(w), "n={n} d={d} k={k} sample {s}");
            }
        }
    }
}

#[test]
fn blocked_predict_equals_the_scalar_argmax_4965() {
    for (i, &(n, d, k)) in SHAPES.iter().enumerate() {
        let (x, y) = data(n, d, k, 333 + i as u64);
        let mut model = GaussianNB::new();
        model.fit(&x, &y).expect("fit");
        let (xt, _) = data(n, d, k, 4444 + i as u64);
        for m in [&x, &xt] {
            let classes = model.classes.as_ref().expect("fitted");
            let want: Vec<usize> = ref_log_posteriors(&model, m)
                .iter()
                .map(|lps| {
                    let (mut best, mut best_lp) = (0, f32::NEG_INFINITY);
                    for (c, &lp) in lps.iter().enumerate() {
                        if lp > best_lp {
                            best_lp = lp;
                            best = c;
                        }
                    }
                    classes[best]
                })
                .collect();
            assert_eq!(
                model.predict(m).expect("predict"),
                want,
                "n={n} d={d} k={k}"
            );
        }
    }
}

#[test]
fn an_exact_tie_goes_to_the_first_class_4965() {
    // Labels 3 and 9 see the same rows, so their priors, means and variances are equal and
    // every log-posterior ties; the scalar loop's strict `>` keeps the first class.
    let rows = [0.5_f32, -1.0, 2.0, 0.25, -0.5, 1.0];
    let mut v = Vec::new();
    for _ in 0..2 {
        v.extend_from_slice(&rows);
    }
    let x = Matrix::from_vec(6, 2, v).expect("6x2");
    let y = vec![3, 3, 3, 9, 9, 9];
    let mut model = GaussianNB::new();
    model.fit(&x, &y).expect("fit");
    assert_eq!(model.predict(&x).expect("predict"), vec![3; 6]);
}
