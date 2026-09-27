//! FALSIFY-NBR-007: rewiring KNN, DBSCAN and LOF onto the neighbor index
//! changed no output (#3149).
//!
//! One class of output is exempt from bit-identity, by construction: KNN's
//! distance-WEIGHTED `predict_proba`. It sums `1/d` over the k nearest, and
//! the old code summed in `select_nth_unstable` partition order, which no
//! contract defined; it now sums in `(distance, index)` order. So those rows
//! are checked in two steps instead: an in-test reference rebuilds the legacy
//! path from its own retained helpers and must reproduce the PINNED digest
//! bit for bit, and the new output must match that reference to 1e-6.
//!
//! `PINNED` holds FNV-1a digests of every output bit, captured from the
//! PRE-rewire implementations (origin/main a016cee94, exhaustive O(n²) scans)
//! with `print_rewire_digests`. The fixtures are sized so `Auto` picks a real
//! tree (n > 64): kd-tree for d = 2 and d = 5, ball tree for d = 20. Their
//! grid coordinates make exact distance ties, and a DBSCAN `eps` that lands
//! exactly on a neighbor distance, common.

use crate::classification::{select_k_nearest, DistanceMetric, KNearestNeighbors};
use crate::cluster::{LocalOutlierFactor, DBSCAN};
use crate::primitives::Matrix;
use crate::traits::UnsupervisedEstimator;

fn fixture(seed: u64, n: usize, d: usize, grid: bool) -> Matrix<f32> {
    let mut s = seed;
    let data = (0..n * d)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            if grid {
                (z % 9) as f32
            } else {
                ((z >> 40) as f32 / (1u64 << 24) as f32) * 10.0
            }
        })
        .collect();
    Matrix::from_vec(n, d, data).expect("n * d values")
}

/// FNV-1a over 32-bit words.
fn digest(words: impl IntoIterator<Item = u32>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for w in words {
        for b in w.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// The pre-#3149 weighted `predict_proba`, rebuilt from the helpers it used:
/// a full distance scan, the partial select, and the same summation.
fn legacy_weighted_proba(
    knn: &KNearestNeighbors,
    k: usize,
    x_train: &Matrix<f32>,
    labels: &[usize],
    queries: &Matrix<f32>,
) -> Vec<f32> {
    let (n_query, d) = queries.shape();
    let n_classes = labels.iter().max().expect("labels") + 1;
    let mut out = Vec::new();
    for i in 0..n_query {
        let mut distances: Vec<(f32, usize, usize)> = labels
            .iter()
            .enumerate()
            .map(|(j, &label)| (knn.compute_distance(queries, i, x_train, j, d), j, label))
            .collect();
        let k_nearest = select_k_nearest(&mut distances, k);
        let mut class_counts = vec![0.0; n_classes];
        if k_nearest.iter().any(|(dist, _)| *dist < 1e-10) {
            for (dist, label) in &k_nearest {
                if *dist < 1e-10 {
                    class_counts[*label] += 1.0;
                }
            }
        } else {
            for (dist, label) in &k_nearest {
                class_counts[*label] += 1.0 / dist;
            }
        }
        let total: f32 = class_counts.iter().sum();
        for count in &mut class_counts {
            *count /= total;
        }
        out.extend(class_counts);
    }
    out
}

/// One output: its name, the digest of what the code produces now, and for a
/// weighted-proba row the `(now, legacy reference)` values.
struct Output {
    name: String,
    digest: u64,
    weighted: Option<(Vec<f32>, Vec<f32>)>,
}

impl Output {
    fn exact(name: String, digest: u64) -> Self {
        Self {
            name,
            digest,
            weighted: None,
        }
    }

    /// The digest PINNED holds for this row: the legacy reference's where
    /// there is one.
    fn pinned_digest(&self) -> u64 {
        self.weighted.as_ref().map_or(self.digest, |(_, legacy)| {
            digest(legacy.iter().map(|p| p.to_bits()))
        })
    }
}

/// Every output of every configuration.
fn outputs() -> Vec<Output> {
    let mut out = Vec::new();
    let sets = [
        ("g2", fixture(11, 300, 2, true)),
        ("c5", fixture(12, 300, 5, false)),
        ("g20", fixture(13, 200, 20, true)),
    ];
    for (name, x) in &sets {
        let (n, _) = x.shape();
        let labels: Vec<usize> = (0..n).map(|i| (i * 7 + i / 3) % 3).collect();
        let queries = fixture(99, 80, x.shape().1, name.starts_with('g'));

        for metric in [
            DistanceMetric::Euclidean,
            DistanceMetric::Manhattan,
            DistanceMetric::Minkowski(3.0),
        ] {
            for k in [1, 5, 15] {
                for weights in [false, true] {
                    let mut knn = KNearestNeighbors::new(k)
                        .with_metric(metric)
                        .with_weights(weights);
                    knn.fit(x, &labels).expect("fit");
                    let pred = knn.predict(&queries).expect("predict");
                    let proba: Vec<f32> = knn
                        .predict_proba(&queries)
                        .expect("proba")
                        .into_iter()
                        .flatten()
                        .collect();
                    let tag = format!("knn/{name}/{metric:?}/k{k}/w{weights}");
                    out.push(Output::exact(
                        format!("{tag}/labels"),
                        digest(pred.iter().map(|&p| p as u32)),
                    ));
                    let mut row = Output::exact(
                        format!("{tag}/proba"),
                        digest(proba.iter().map(|p| p.to_bits())),
                    );
                    if weights {
                        let legacy = legacy_weighted_proba(&knn, k, x, &labels, &queries);
                        row.weighted = Some((proba, legacy));
                    }
                    out.push(row);
                }
            }
        }

        for eps in [0.5_f32, 1.0, 1.5, 2.0, 3.0, 12.0] {
            for min_samples in [3, 5, 10] {
                let mut db = DBSCAN::new(eps, min_samples);
                db.fit(x).expect("fit");
                out.push(Output::exact(
                    format!("dbscan/{name}/eps{eps}/m{min_samples}"),
                    digest(db.labels().iter().map(|&l| l as u32)),
                ));
            }
        }

        for k in [5, 20] {
            let mut lof = LocalOutlierFactor::new().with_n_neighbors(k);
            lof.fit(x).expect("fit");
            let tag = format!("lof/{name}/k{k}");
            out.push(Output::exact(
                format!("{tag}/nof"),
                digest(lof.negative_outlier_factor().iter().map(|v| v.to_bits())),
            ));
            out.push(Output::exact(
                format!("{tag}/score"),
                digest(lof.score_samples(&queries).iter().map(|v| v.to_bits())),
            ));
            out.push(Output::exact(
                format!("{tag}/predict"),
                digest(lof.predict(&queries).iter().map(|&v| v as u32)),
            ));
        }
    }
    out
}

/// Regenerates `PINNED` (captured on the pre-rewire tree; the legacy
/// reference makes the weighted rows reproducible after it too).
#[test]
#[ignore = "fixture generator"]
fn print_rewire_digests() {
    for o in outputs() {
        println!("    (\"{}\", 0x{:016x}),", o.name, o.pinned_digest());
    }
}

#[test]
fn falsify_nbr_007_rewire_changes_no_output() {
    let got = outputs();
    assert_eq!(got.len(), PINNED.len(), "configuration count changed");
    let mut diffs = Vec::new();
    for (o, &(pname, pdigest)) in got.iter().zip(PINNED) {
        if o.name != pname {
            diffs.push(format!("{}: pinned row is {pname}", o.name));
            continue;
        }
        match &o.weighted {
            // Exact rows: every output bit unchanged.
            None if o.digest != pdigest => {
                diffs.push(format!(
                    "{pname}: got 0x{:016x}, pinned 0x{pdigest:016x}",
                    o.digest
                ));
            }
            None => {}
            Some((now, legacy)) => {
                // The reference must BE the old code, or the tolerance below
                // compares against nothing.
                if o.pinned_digest() != pdigest {
                    diffs.push(format!(
                        "{pname}: legacy reference does not reproduce the pin"
                    ));
                }
                let worst = now
                    .iter()
                    .zip(legacy)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max);
                if now.len() != legacy.len() || worst > 1e-6 {
                    diffs.push(format!("{pname}: weighted proba moved by {worst:e}"));
                }
            }
        }
    }
    assert!(
        diffs.is_empty(),
        "{} outputs changed:\n{}",
        diffs.len(),
        diffs.join("\n")
    );
}

const PINNED: &[(&str, u64)] = &[
    ("knn/g2/Euclidean/k1/wfalse/labels", 0x563165028630d3f7),
    ("knn/g2/Euclidean/k1/wfalse/proba", 0x1659bf3d3bfc68a5),
    ("knn/g2/Euclidean/k1/wtrue/labels", 0x563165028630d3f7),
    ("knn/g2/Euclidean/k1/wtrue/proba", 0x1659bf3d3bfc68a5),
    ("knn/g2/Euclidean/k5/wfalse/labels", 0x0bd013f875f18b94),
    ("knn/g2/Euclidean/k5/wfalse/proba", 0x211edeab22cd3152),
    ("knn/g2/Euclidean/k5/wtrue/labels", 0x35f10dc46e211464),
    ("knn/g2/Euclidean/k5/wtrue/proba", 0x969cf9663fa74f9d),
    ("knn/g2/Euclidean/k15/wfalse/labels", 0x2923b568ffba8514),
    ("knn/g2/Euclidean/k15/wfalse/proba", 0x7d20084405e32327),
    ("knn/g2/Euclidean/k15/wtrue/labels", 0xb5c9016453906eb7),
    ("knn/g2/Euclidean/k15/wtrue/proba", 0x839c893615b3a439),
    ("knn/g2/Manhattan/k1/wfalse/labels", 0x563165028630d3f7),
    ("knn/g2/Manhattan/k1/wfalse/proba", 0x1659bf3d3bfc68a5),
    ("knn/g2/Manhattan/k1/wtrue/labels", 0x563165028630d3f7),
    ("knn/g2/Manhattan/k1/wtrue/proba", 0x1659bf3d3bfc68a5),
    ("knn/g2/Manhattan/k5/wfalse/labels", 0x0bd013f875f18b94),
    ("knn/g2/Manhattan/k5/wfalse/proba", 0x211edeab22cd3152),
    ("knn/g2/Manhattan/k5/wtrue/labels", 0x35f10dc46e211464),
    ("knn/g2/Manhattan/k5/wtrue/proba", 0x969cf9663fa74f9d),
    ("knn/g2/Manhattan/k15/wfalse/labels", 0x66f57d6ebba497e7),
    ("knn/g2/Manhattan/k15/wfalse/proba", 0x2073b30c7989d6c1),
    ("knn/g2/Manhattan/k15/wtrue/labels", 0xb5c9016453906eb7),
    ("knn/g2/Manhattan/k15/wtrue/proba", 0x9215cc4b49c8d810),
    ("knn/g2/Minkowski(3.0)/k1/wfalse/labels", 0x563165028630d3f7),
    ("knn/g2/Minkowski(3.0)/k1/wfalse/proba", 0x1659bf3d3bfc68a5),
    ("knn/g2/Minkowski(3.0)/k1/wtrue/labels", 0x563165028630d3f7),
    ("knn/g2/Minkowski(3.0)/k1/wtrue/proba", 0x1659bf3d3bfc68a5),
    ("knn/g2/Minkowski(3.0)/k5/wfalse/labels", 0x0bd013f875f18b94),
    ("knn/g2/Minkowski(3.0)/k5/wfalse/proba", 0x211edeab22cd3152),
    ("knn/g2/Minkowski(3.0)/k5/wtrue/labels", 0x35f10dc46e211464),
    ("knn/g2/Minkowski(3.0)/k5/wtrue/proba", 0x969cf9663fa74f9d),
    (
        "knn/g2/Minkowski(3.0)/k15/wfalse/labels",
        0x2923b568ffba8514,
    ),
    ("knn/g2/Minkowski(3.0)/k15/wfalse/proba", 0x7d20084405e32327),
    ("knn/g2/Minkowski(3.0)/k15/wtrue/labels", 0xb5c9016453906eb7),
    ("knn/g2/Minkowski(3.0)/k15/wtrue/proba", 0xa25c51cd7402dc42),
    ("dbscan/g2/eps0.5/m3", 0xdd89cf256b00cf12),
    ("dbscan/g2/eps0.5/m5", 0x805d4630e2a0463e),
    ("dbscan/g2/eps0.5/m10", 0x68eeaffc3a0c1a9d),
    ("dbscan/g2/eps1/m3", 0x74b4429a2fdd70e5),
    ("dbscan/g2/eps1/m5", 0x74b4429a2fdd70e5),
    ("dbscan/g2/eps1/m10", 0x74b4429a2fdd70e5),
    ("dbscan/g2/eps1.5/m3", 0x74b4429a2fdd70e5),
    ("dbscan/g2/eps1.5/m5", 0x74b4429a2fdd70e5),
    ("dbscan/g2/eps1.5/m10", 0x74b4429a2fdd70e5),
    ("dbscan/g2/eps2/m3", 0x74b4429a2fdd70e5),
    ("dbscan/g2/eps2/m5", 0x74b4429a2fdd70e5),
    ("dbscan/g2/eps2/m10", 0x74b4429a2fdd70e5),
    ("dbscan/g2/eps3/m3", 0x74b4429a2fdd70e5),
    ("dbscan/g2/eps3/m5", 0x74b4429a2fdd70e5),
    ("dbscan/g2/eps3/m10", 0x74b4429a2fdd70e5),
    ("dbscan/g2/eps12/m3", 0x74b4429a2fdd70e5),
    ("dbscan/g2/eps12/m5", 0x74b4429a2fdd70e5),
    ("dbscan/g2/eps12/m10", 0x74b4429a2fdd70e5),
    ("lof/g2/k5/nof", 0xc24069c54a5a0f05),
    ("lof/g2/k5/score", 0xa980f887ed8a9263),
    ("lof/g2/k5/predict", 0x56adf77bd660a5a5),
    ("lof/g2/k20/nof", 0x131fd8f25a25e2f8),
    ("lof/g2/k20/score", 0x0a7fe01864c12eca),
    ("lof/g2/k20/predict", 0x56adf77bd660a5a5),
    ("knn/c5/Euclidean/k1/wfalse/labels", 0x84471c96fac1ced4),
    ("knn/c5/Euclidean/k1/wfalse/proba", 0x239e48539b2c3155),
    ("knn/c5/Euclidean/k1/wtrue/labels", 0x84471c96fac1ced4),
    ("knn/c5/Euclidean/k1/wtrue/proba", 0x239e48539b2c3155),
    ("knn/c5/Euclidean/k5/wfalse/labels", 0xfd395328328eec34),
    ("knn/c5/Euclidean/k5/wfalse/proba", 0x02c6a99b290ec65c),
    ("knn/c5/Euclidean/k5/wtrue/labels", 0xb47f4046a4971274),
    ("knn/c5/Euclidean/k5/wtrue/proba", 0xc4023e86bd7f8dfa),
    ("knn/c5/Euclidean/k15/wfalse/labels", 0x9622fc26153b2674),
    ("knn/c5/Euclidean/k15/wfalse/proba", 0x3278de2984701718),
    ("knn/c5/Euclidean/k15/wtrue/labels", 0xe4347f6ae4a06ad6),
    ("knn/c5/Euclidean/k15/wtrue/proba", 0xb31f4268e9f22430),
    ("knn/c5/Manhattan/k1/wfalse/labels", 0x2240d8344554a1b6),
    ("knn/c5/Manhattan/k1/wfalse/proba", 0xb5aa934184a41d95),
    ("knn/c5/Manhattan/k1/wtrue/labels", 0x2240d8344554a1b6),
    ("knn/c5/Manhattan/k1/wtrue/proba", 0xb5aa934184a41d95),
    ("knn/c5/Manhattan/k5/wfalse/labels", 0xc9a38b0a9ca666c5),
    ("knn/c5/Manhattan/k5/wfalse/proba", 0xf42bdd264ccb29b9),
    ("knn/c5/Manhattan/k5/wtrue/labels", 0x5e35d7ca6bc7d394),
    ("knn/c5/Manhattan/k5/wtrue/proba", 0xe29fa5308ec725bb),
    ("knn/c5/Manhattan/k15/wfalse/labels", 0x5a7427400aee4207),
    ("knn/c5/Manhattan/k15/wfalse/proba", 0x4e8d42413c488810),
    ("knn/c5/Manhattan/k15/wtrue/labels", 0x187a0e6e0b2a6557),
    ("knn/c5/Manhattan/k15/wtrue/proba", 0xf18eb2406e987716),
    ("knn/c5/Minkowski(3.0)/k1/wfalse/labels", 0xa955b635e2e8d627),
    ("knn/c5/Minkowski(3.0)/k1/wfalse/proba", 0xfb47e42e20c279a5),
    ("knn/c5/Minkowski(3.0)/k1/wtrue/labels", 0xa955b635e2e8d627),
    ("knn/c5/Minkowski(3.0)/k1/wtrue/proba", 0xfb47e42e20c279a5),
    ("knn/c5/Minkowski(3.0)/k5/wfalse/labels", 0x920dc836a97beb06),
    ("knn/c5/Minkowski(3.0)/k5/wfalse/proba", 0x5703706e5dc0c0f3),
    ("knn/c5/Minkowski(3.0)/k5/wtrue/labels", 0x7d9bd9c0f0a65e17),
    ("knn/c5/Minkowski(3.0)/k5/wtrue/proba", 0x6d888804eda84468),
    (
        "knn/c5/Minkowski(3.0)/k15/wfalse/labels",
        0xfa49e87cba7d2206,
    ),
    ("knn/c5/Minkowski(3.0)/k15/wfalse/proba", 0x63b180079ccd4fe1),
    ("knn/c5/Minkowski(3.0)/k15/wtrue/labels", 0xdb399a66a53b2b17),
    ("knn/c5/Minkowski(3.0)/k15/wtrue/proba", 0x33674b87a0748b3c),
    ("dbscan/c5/eps0.5/m3", 0xd339907f3b080c35),
    ("dbscan/c5/eps0.5/m5", 0xd339907f3b080c35),
    ("dbscan/c5/eps0.5/m10", 0xd339907f3b080c35),
    ("dbscan/c5/eps1/m3", 0xd339907f3b080c35),
    ("dbscan/c5/eps1/m5", 0xd339907f3b080c35),
    ("dbscan/c5/eps1/m10", 0xd339907f3b080c35),
    ("dbscan/c5/eps1.5/m3", 0xd339907f3b080c35),
    ("dbscan/c5/eps1.5/m5", 0xd339907f3b080c35),
    ("dbscan/c5/eps1.5/m10", 0xd339907f3b080c35),
    ("dbscan/c5/eps2/m3", 0x0d5b9f86a9bfe93e),
    ("dbscan/c5/eps2/m5", 0xd339907f3b080c35),
    ("dbscan/c5/eps2/m10", 0xd339907f3b080c35),
    ("dbscan/c5/eps3/m3", 0x2326af3752037174),
    ("dbscan/c5/eps3/m5", 0xcfea036de849e364),
    ("dbscan/c5/eps3/m10", 0xd339907f3b080c35),
    ("dbscan/c5/eps12/m3", 0x74b4429a2fdd70e5),
    ("dbscan/c5/eps12/m5", 0x74b4429a2fdd70e5),
    ("dbscan/c5/eps12/m10", 0x74b4429a2fdd70e5),
    ("lof/c5/k5/nof", 0xa703a18bb88a79e4),
    ("lof/c5/k5/score", 0x2ad75aa9db3fdb84),
    ("lof/c5/k5/predict", 0xe8b35988308aa195),
    ("lof/c5/k20/nof", 0x59be0b3a3496c830),
    ("lof/c5/k20/score", 0x233534fc92b9862b),
    ("lof/c5/k20/predict", 0x263af4ce105147e8),
    ("knn/g20/Euclidean/k1/wfalse/labels", 0x841116e15f5a8455),
    ("knn/g20/Euclidean/k1/wfalse/proba", 0x8ff751cf69381665),
    ("knn/g20/Euclidean/k1/wtrue/labels", 0x841116e15f5a8455),
    ("knn/g20/Euclidean/k1/wtrue/proba", 0x8ff751cf69381665),
    ("knn/g20/Euclidean/k5/wfalse/labels", 0x92a9d8058ffee4a7),
    ("knn/g20/Euclidean/k5/wfalse/proba", 0xd152e13700b44f9f),
    ("knn/g20/Euclidean/k5/wtrue/labels", 0xaad04c847ef4c985),
    ("knn/g20/Euclidean/k5/wtrue/proba", 0x31c7523bb55d680e),
    ("knn/g20/Euclidean/k15/wfalse/labels", 0x4f3d7c995aacb564),
    ("knn/g20/Euclidean/k15/wfalse/proba", 0x3a7e912990798395),
    ("knn/g20/Euclidean/k15/wtrue/labels", 0x24345d367cf64655),
    ("knn/g20/Euclidean/k15/wtrue/proba", 0xa9153b86bdaed7d0),
    ("knn/g20/Manhattan/k1/wfalse/labels", 0x962742cec2752e47),
    ("knn/g20/Manhattan/k1/wfalse/proba", 0xc4a2e3e094519065),
    ("knn/g20/Manhattan/k1/wtrue/labels", 0x962742cec2752e47),
    ("knn/g20/Manhattan/k1/wtrue/proba", 0xc4a2e3e094519065),
    ("knn/g20/Manhattan/k5/wfalse/labels", 0x038a75d82b2e83b6),
    ("knn/g20/Manhattan/k5/wfalse/proba", 0x102892d25a2ca198),
    ("knn/g20/Manhattan/k5/wtrue/labels", 0x17d5360273332904),
    ("knn/g20/Manhattan/k5/wtrue/proba", 0x8f12aa82aa46ff57),
    ("knn/g20/Manhattan/k15/wfalse/labels", 0x1bd4957e0e679375),
    ("knn/g20/Manhattan/k15/wfalse/proba", 0x46e16ccc3a25e87d),
    ("knn/g20/Manhattan/k15/wtrue/labels", 0x9023858adaf6f1a7),
    ("knn/g20/Manhattan/k15/wtrue/proba", 0xe652612cf25501f6),
    (
        "knn/g20/Minkowski(3.0)/k1/wfalse/labels",
        0x03e69e22c6d538f5,
    ),
    ("knn/g20/Minkowski(3.0)/k1/wfalse/proba", 0x3c8b893380cb0a25),
    ("knn/g20/Minkowski(3.0)/k1/wtrue/labels", 0x03e69e22c6d538f5),
    ("knn/g20/Minkowski(3.0)/k1/wtrue/proba", 0x3c8b893380cb0a25),
    (
        "knn/g20/Minkowski(3.0)/k5/wfalse/labels",
        0x61b4d7def72c32b7,
    ),
    ("knn/g20/Minkowski(3.0)/k5/wfalse/proba", 0xcdc5d7be8fe668f5),
    ("knn/g20/Minkowski(3.0)/k5/wtrue/labels", 0xcd9fffebb5d1c4e4),
    ("knn/g20/Minkowski(3.0)/k5/wtrue/proba", 0xcd63c0e7d21b4a42),
    (
        "knn/g20/Minkowski(3.0)/k15/wfalse/labels",
        0x38a1e9590a7a8974,
    ),
    (
        "knn/g20/Minkowski(3.0)/k15/wfalse/proba",
        0xdd203602ced0a64e,
    ),
    (
        "knn/g20/Minkowski(3.0)/k15/wtrue/labels",
        0x918ba0d7aac782b5,
    ),
    ("knn/g20/Minkowski(3.0)/k15/wtrue/proba", 0x65247d801e1abccb),
    ("dbscan/g20/eps0.5/m3", 0xcea8dd2c0abfd685),
    ("dbscan/g20/eps0.5/m5", 0xcea8dd2c0abfd685),
    ("dbscan/g20/eps0.5/m10", 0xcea8dd2c0abfd685),
    ("dbscan/g20/eps1/m3", 0xcea8dd2c0abfd685),
    ("dbscan/g20/eps1/m5", 0xcea8dd2c0abfd685),
    ("dbscan/g20/eps1/m10", 0xcea8dd2c0abfd685),
    ("dbscan/g20/eps1.5/m3", 0xcea8dd2c0abfd685),
    ("dbscan/g20/eps1.5/m5", 0xcea8dd2c0abfd685),
    ("dbscan/g20/eps1.5/m10", 0xcea8dd2c0abfd685),
    ("dbscan/g20/eps2/m3", 0xcea8dd2c0abfd685),
    ("dbscan/g20/eps2/m5", 0xcea8dd2c0abfd685),
    ("dbscan/g20/eps2/m10", 0xcea8dd2c0abfd685),
    ("dbscan/g20/eps3/m3", 0xcea8dd2c0abfd685),
    ("dbscan/g20/eps3/m5", 0xcea8dd2c0abfd685),
    ("dbscan/g20/eps3/m10", 0xcea8dd2c0abfd685),
    ("dbscan/g20/eps12/m3", 0x6d7328bacbd4387d),
    ("dbscan/g20/eps12/m5", 0x309f9e922dcf6b99),
    ("dbscan/g20/eps12/m10", 0x125e8d41565222b9),
    ("lof/g20/k5/nof", 0x97b2940d24be23c5),
    ("lof/g20/k5/score", 0x44ae466ad03633f1),
    ("lof/g20/k5/predict", 0xcfc7609e2b96e198),
    ("lof/g20/k20/nof", 0x1d84c5abc490221e),
    ("lof/g20/k20/score", 0x3c03b354ad66e87a),
    ("lof/g20/k20/predict", 0x507eaadb0789eab0),
];
