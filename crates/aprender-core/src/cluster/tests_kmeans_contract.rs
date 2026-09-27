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
            // The kept run IS the best of the ten: its inertia equals the
            // minimum over each restart run alone. Deleting the restart loop
            // (every fit = restart 0) fails this wherever restart 0 is not
            // the minimum, independently of KM-008.
            let runs_min = (0..10)
                .map(|r| ten.lloyd(&x, r).2)
                .fold(f32::INFINITY, f32::min);
            assert_eq!(
                ten.inertia(),
                runs_min,
                "FALSIFIED KM-007: seed {seed} k {k}: kept inertia is not the min over the 10 restarts"
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
    assert_eq!(
        KMeans::new(3).n_init(),
        1,
        "sklearn >=1.4 n_init=\"auto\" for k-means++-family seeding"
    );
    assert_eq!(KMeans::new(3).with_n_init(0).n_init(), 1);
}

// =========================================================================
// FALSIFY-KM-011/012: D² (k-means++) seeding (E9 finding 5d-E9-kmeans-d2-seeding)
// =========================================================================

/// The pre-#3146 init, kept here as the reference D² must beat: farthest-point
/// seeding from `first_idx`, then the SAME Lloyd iterations (`lloyd_from`).
fn farthest_point_lloyd(km: &KMeans, x: &Matrix<f32>, first_idx: usize) -> f32 {
    let (n, d) = x.shape();
    let mut data = Vec::with_capacity(km.n_clusters * d);
    append_row(&mut data, x, first_idx, d);
    let mut closest = distances_sq_to_sample(x, first_idx);
    for _ in 1..km.n_clusters {
        let far = (0..n).fold(0, |b, i| if closest[i] > closest[b] { i } else { b });
        append_row(&mut data, x, far, d);
        let d_far = distances_sq_to_sample(x, far);
        for (c, f) in closest.iter_mut().zip(d_far) {
            *c = c.min(f);
        }
    }
    let centroids = Matrix::from_vec(km.n_clusters, d, data).expect("valid matrix");
    km.lloyd_from(x, centroids).2
}

/// FALSIFY-KM-011: over the KM-007 grid (96 single-init fits) D² seeding
/// reaches a strictly lower total inertia than farthest-point seeding
/// (measured: 29635.9 vs 30066.0, 30 wins / 31 losses — the margin is in the
/// aggregate, not in a per-fit majority). An argmax mutant IS
/// farthest-point and ties, so it fails the strict inequality.
#[test]
fn falsify_km_011_d2_beats_farthest_point() {
    let (mut sum_d2, mut sum_fp, mut wins, mut losses) = (0.0_f64, 0.0_f64, 0, 0);
    for seed in 0..24u64 {
        let x = km_blobs(seed, 60, 3, 6);
        for k in [2usize, 3, 5, 8] {
            let km = km_fit(&x, k, 1);
            let fp = farthest_point_lloyd(&km, &x, km.restart_start_row(0, x.n_rows()));
            sum_d2 += f64::from(km.inertia());
            sum_fp += f64::from(fp);
            wins += usize::from(km.inertia() < fp);
            losses += usize::from(km.inertia() > fp);
        }
    }
    assert!(
        sum_d2 < sum_fp,
        "FALSIFIED KM-011: D² total inertia {sum_d2} >= farthest-point {sum_fp} over 96 fits (wins {wins}, losses {losses})"
    );
}

/// FALSIFY-KM-012: a draw is proportional to D², so a row at distance 0 from
/// a chosen centroid is never drawn while any other row has weight. 50 copies
/// of one point plus two distinct points, k = 3: every seed must seed all
/// three distinct points and reach inertia 0. Uniform sampling draws a copy
/// with probability ~50/52 per step.
#[test]
fn falsify_km_012_zero_weight_rows_never_drawn() {
    let mut data = vec![1.0_f32; 100];
    data.extend([5.0, -3.0, -4.0, 7.0]);
    let x = Matrix::from_vec(52, 2, data).expect("valid matrix");
    for seed in 0..32u64 {
        let km = KMeans::new(3).with_random_state(seed);
        let centroids = km.kmeans_plusplus_init(&x, 0);
        let mut rows: Vec<(u32, u32)> = (0..3)
            .map(|c| (centroids.get(c, 0).to_bits(), centroids.get(c, 1).to_bits()))
            .collect();
        rows.sort_unstable();
        rows.dedup();
        assert_eq!(
            rows.len(),
            3,
            "FALSIFIED KM-012: seed {seed}: D² drew a zero-weight duplicate row"
        );
    }
}

/// FALSIFY-KM-013: two restarts that happen to share a start row must still draw
/// different candidates. `restart_start_row` is `% n`, so over 200 runs on 60 rows
/// collisions are certain; an RNG keyed on the start row instead of the run makes
/// every colliding pair seed identically, and n_init silently repeats a restart.
#[test]
fn falsify_km_013_restarts_sharing_a_start_row_still_differ() {
    let x = km_blobs(3, 60, 3, 6);
    let n = x.shape().0;
    let km = KMeans::new(5).with_random_state(11);
    let mut first_run_at: Vec<Option<usize>> = vec![None; n];
    let (mut pairs, mut differ) = (0usize, 0usize);
    for run in 0..200 {
        let row = km.restart_start_row(run, n);
        let Some(prev) = first_run_at[row] else {
            first_run_at[row] = Some(run);
            continue;
        };
        pairs += 1;
        if km.kmeans_plusplus_init(&x, prev).as_slice()
            != km.kmeans_plusplus_init(&x, run).as_slice()
        {
            differ += 1;
        }
    }
    assert!(
        pairs > 0,
        "KM-013 is vacuous: no two of 200 runs shared a start row"
    );
    assert!(
        differ * 2 > pairs,
        "FALSIFIED KM-013: only {differ} of {pairs} restart pairs sharing a start row drew differently"
    );
}

/// Written by the pre-`n_init` code (d868ec946^, `KMeans::new(2).with_random_state(7)`
/// fitted on these six points). Regenerating them with the current code would
/// make the test vacuous: they must stay the OLD bytes.
const LEGACY_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/kmeans");

fn legacy_points() -> Matrix<f32> {
    Matrix::from_vec(
        6,
        2,
        vec![0.0, 0.0, 0.1, 0.2, 0.2, 0.1, 5.0, 5.0, 5.1, 5.2, 5.2, 5.1],
    )
    .expect("6x2")
}

fn assert_legacy_model(km: &KMeans) {
    assert_eq!(
        km.n_init(),
        1,
        "a pre-n_init model was fitted from one start"
    );
    assert_eq!(km.random_state(), Some(7));
    let c = km.centroids();
    let got: Vec<f32> = (0..2)
        .flat_map(|i| (0..2).map(move |j| (i, j)))
        .map(|(i, j)| c.get(i, j))
        .collect();
    for (g, w) in got.iter().zip([0.1_f32, 0.1, 5.1, 5.1]) {
        assert!((g - w).abs() < 1e-5, "centroids {got:?}");
    }
    assert_eq!(km.predict(&legacy_points()), vec![0, 0, 0, 1, 1, 1]);
}

#[test]
fn falsify_km_010_pre_n_init_bincode_loads() {
    let km = KMeans::load(format!("{LEGACY_DIR}/kmeans_pre_n_init.bincode"))
        .expect("a model saved before n_init existed must still load");
    assert_legacy_model(&km);
}

#[test]
fn falsify_km_010_pre_n_init_safetensors_loads() {
    let km = KMeans::load_safetensors(format!("{LEGACY_DIR}/kmeans_pre_n_init.st-fixture"))
        .expect("a safetensors model saved before n_init existed must still load");
    assert_legacy_model(&km);
}

#[test]
fn falsify_km_010_current_bincode_round_trips_n_init() {
    let dir = tempfile::tempdir().expect("tmp");
    let p = dir.path().join("km.bin");
    let mut km = KMeans::new(2).with_random_state(7).with_n_init(4);
    km.fit(&legacy_points()).expect("fit");
    km.save(&p).expect("save");
    let back = KMeans::load(&p).expect("load");
    assert_eq!(
        back.n_init(),
        4,
        "a current file must not be read as legacy"
    );
    assert_eq!(back.predict(&legacy_points()), km.predict(&legacy_points()));
}

#[test]
fn falsify_km_010_layout_is_legacy_plus_trailing_n_init() {
    // n_init must be the LAST bincode field: a current n_init=1 fit of the
    // fixture's data is the old file's exact bytes followed by 1u64. A field
    // anywhere else shifts old bytes under the current decode.
    let legacy = std::fs::read(format!("{LEGACY_DIR}/kmeans_pre_n_init.bincode")).expect("fixture");
    let mut km = KMeans::new(2).with_random_state(7).with_n_init(1);
    km.fit(&legacy_points()).expect("fit");
    let mut want = legacy;
    want.extend_from_slice(&1u64.to_le_bytes());
    assert_eq!(bincode::serialize(&km).expect("ser"), want);
}
