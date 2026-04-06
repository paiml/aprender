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
