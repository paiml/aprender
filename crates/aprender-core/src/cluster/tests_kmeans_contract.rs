// =========================================================================
// FALSIFY-KM: kmeans-kernel-v1.yaml contract (aprender KMeans)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had proptest KMeans tests but zero inline FALSIFY-KM-* tests
//   Why 2: proptests live in tests/contracts/, not near the implementation
//   Why 3: no mapping from kmeans-kernel-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: KMeans was "obviously correct" (standard Lloyd's algorithm)
//
// References:
//   - provable-contracts/contracts/kmeans-kernel-v1.yaml
//   - Lloyd (1982) "Least Squares Quantization in PCM"
// =========================================================================

use super::*;

/// FALSIFY-KM-001: Valid cluster indices — all labels in [0, K-1]
#[test]
fn falsify_km_001_valid_indices() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 8.0, 8.0, 1.0, 0.6, 9.0, 11.0],
    )
    .expect("valid matrix");

    let k = 2;
    let mut km = KMeans::new(k).with_random_state(42);
    km.fit(&data).expect("fit succeeds");

    let labels = km.predict(&data);
    for (i, &label) in labels.iter().enumerate() {
        assert!(
            label < k,
            "FALSIFIED KM-001: label[{i}] = {label}, expected < {k}"
        );
    }
}

/// FALSIFY-KM-002: Objective non-negative — inertia >= 0
#[test]
fn falsify_km_002_inertia_non_negative() {
    let data =
        Matrix::from_vec(4, 2, vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).expect("valid matrix");

    let mut km = KMeans::new(2).with_random_state(42);
    km.fit(&data).expect("fit succeeds");

    assert!(
        km.inertia() >= 0.0,
        "FALSIFIED KM-002: inertia = {} < 0",
        km.inertia()
    );
}

/// FALSIFY-KM-003: Nearest centroid assignment — each point assigned to closest
#[test]
fn falsify_km_003_nearest_centroid() {
    let data = Matrix::from_vec(
        6,
        2,
        vec![
            0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 10.0, 10.0, 10.1, 10.1, 10.2, 10.2,
        ],
    )
    .expect("valid matrix");

    let mut km = KMeans::new(2).with_random_state(42);
    km.fit(&data).expect("fit succeeds");

    let labels = km.predict(&data);
    let centroids = km.centroids();
    let n_features = 2;

    for i in 0..6 {
        let assigned = labels[i];
        // Distance to assigned centroid
        let d_assigned: f32 = (0..n_features)
            .map(|f| {
                let diff = data.get(i, f) - centroids.get(assigned, f);
                diff * diff
            })
            .sum();

        // Distance to other centroids
        for c in 0..2 {
            if c == assigned {
                continue;
            }
            let d_other: f32 = (0..n_features)
                .map(|f| {
                    let diff = data.get(i, f) - centroids.get(c, f);
                    diff * diff
                })
                .sum();
            assert!(
                d_assigned <= d_other + 1e-5,
                "FALSIFIED KM-003: point[{i}] assigned to c={assigned} (d={d_assigned}) but c={c} is closer (d={d_other})"
            );
        }
    }
}

/// FALSIFY-KM-004: K=1 — all points in same cluster, centroid is mean
#[test]
fn falsify_km_004_single_cluster() {
    let data = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid matrix");

    let mut km = KMeans::new(1).with_random_state(42);
    km.fit(&data).expect("fit succeeds");

    let labels = km.predict(&data);
    for (i, &l) in labels.iter().enumerate() {
        assert_eq!(l, 0, "FALSIFIED KM-004: point[{i}] not in cluster 0");
    }

    let centroids = km.centroids();
    // Centroid should be mean: (3.0, 4.0)
    assert!(
        (centroids.get(0, 0) - 3.0).abs() < 1e-4,
        "FALSIFIED KM-004: centroid[0][0] = {}, expected 3.0",
        centroids.get(0, 0)
    );
    assert!(
        (centroids.get(0, 1) - 4.0).abs() < 1e-4,
        "FALSIFIED KM-004: centroid[0][1] = {}, expected 4.0",
        centroids.get(0, 1)
    );
}

// =========================================================================
// FALSIFY-KM-007/008: n_init restarts (E9 finding, #3146)
// =========================================================================

/// Deterministic blob data: `n_blobs` Gaussian-ish blobs in `d` dims (LCG).
fn km_blobs(seed: u64, n: usize, d: usize, n_blobs: usize) -> Matrix<f32> {
    let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mut next = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 33) as f32) / ((1u64 << 31) as f32)
    };
    let centers: Vec<f32> = (0..n_blobs * d).map(|_| next() * 10.0).collect();
    let mut data = Vec::with_capacity(n * d);
    for i in 0..n {
        let b = i % n_blobs;
        for j in 0..d {
            data.push(centers[b * d + j] + (next() + next() + next() - 1.5) * 1.5);
        }
    }
    Matrix::from_vec(n, d, data).expect("valid matrix")
}

fn km_fit(x: &Matrix<f32>, k: usize, n_init: usize) -> KMeans {
    let mut km = KMeans::new(k).with_random_state(42).with_n_init(n_init);
    km.fit(x).expect("fit succeeds");
    km
}

/// FALSIFY-KM-007: best-of-n never loses to a single init.
/// inertia(n_init = 10) <= inertia(n_init = 1), and n_init = 1 is the
/// restart-0 run a pre-n_init fit produced (same start row, same labels).
#[test]
fn falsify_km_007_n_init_never_worse_than_single() {
    for seed in 0..24u64 {
        let x = km_blobs(seed, 60, 3, 6);
        for k in [2usize, 3, 5, 8] {
            let one = km_fit(&x, k, 1);
            let ten = km_fit(&x, k, 10);
            assert!(
                ten.inertia() <= one.inertia(),
                "FALSIFIED KM-007: seed {seed} k {k}: n_init=10 {} > n_init=1 {}",
                ten.inertia(),
                one.inertia()
            );
            assert_eq!(
                one.restart_start_row(0, x.n_rows()),
                42 % x.n_rows(),
                "FALSIFIED KM-007: restart 0 no longer starts at seed % n"
            );
        }
    }
}

/// FALSIFY-KM-008: restarts actually explore. Over a grid of datasets the
/// restarts must strictly beat the single init somewhere (a no-op restart
/// loop, or every restart reusing row seed % n, scores zero here).
#[test]
fn falsify_km_008_n_init_strictly_improves_somewhere() {
    let mut strict = 0;
    let mut total = 0;
    for seed in 0..24u64 {
        let x = km_blobs(seed, 60, 3, 6);
        for k in [2usize, 3, 5, 8] {
            total += 1;
            if km_fit(&x, k, 10).inertia() < km_fit(&x, k, 1).inertia() {
                strict += 1;
            }
        }
    }
    assert!(
        strict > 0,
        "FALSIFIED KM-008: n_init=10 beat n_init=1 on 0/{total} fits"
    );
}

/// FALSIFY-KM-009: n_init is seeded — two fits with the same seed agree.
#[test]
fn falsify_km_009_n_init_reproducible() {
    let x = km_blobs(7, 60, 3, 6);
    let a = km_fit(&x, 5, 10);
    let b = km_fit(&x, 5, 10);
    assert_eq!(a.inertia().to_bits(), b.inertia().to_bits());
    assert_eq!(a.predict(&x), b.predict(&x));
    assert_eq!(KMeans::new(3).n_init(), 10, "sklearn default n_init");
    assert_eq!(KMeans::new(3).with_n_init(0).n_init(), 1);
}
