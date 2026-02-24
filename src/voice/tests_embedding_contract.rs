// =========================================================================
// FALSIFY-EMB: embedding-algebra-v1.yaml contract (voice SpeakerEmbedding)
//
// Five-Whys (PMAT-354):
//   Why 1: voice had SpeakerEmbedding tests but zero FALSIFY-EMB-* tests
//   Why 2: unit tests verify API behavior, not algebraic invariants
//   Why 3: no mapping from embedding-algebra-v1.yaml to voice test names
//   Why 4: voice/embedding predates the provable-contracts YAML convention
//   Why 5: normalize/cosine operations were "obviously correct" (stdlib math)
//
// References:
//   - provable-contracts/contracts/embedding-algebra-v1.yaml
//   - Snyder et al. (2018) "X-Vectors: Robust DNN Embeddings"
// =========================================================================

use super::*;

/// FALSIFY-EMB-001: Normalize idempotent — normalize(normalize(x)) = normalize(x)
///
/// A second normalization must not change the result.
#[test]
fn falsify_emb_001_normalize_idempotent() {
    let emb = SpeakerEmbedding::from_vec(vec![3.0, 4.0, 0.0, -1.0, 2.5]);
    let once = normalize_embedding(&emb);
    let twice = normalize_embedding(&once);

    for (i, (&a, &b)) in once
        .as_slice()
        .iter()
        .zip(twice.as_slice().iter())
        .enumerate()
    {
        assert!(
            (a - b).abs() < 1e-6,
            "FALSIFIED EMB-001: normalize not idempotent at dim {i}: {a} vs {b}"
        );
    }
}

/// FALSIFY-EMB-002: Cosine self-similarity = 1.0
///
/// cos(x, x) = 1.0 for any non-zero embedding.
#[test]
fn falsify_emb_002_cosine_self_similarity() {
    let test_vecs = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.5, -0.3, 0.8, 0.1],
        vec![100.0, -200.0, 300.0],
        vec![0.01, 0.01], // small but above epsilon threshold
    ];

    for (idx, v) in test_vecs.iter().enumerate() {
        let emb = SpeakerEmbedding::from_vec(v.clone());
        let sim = cosine_similarity(&emb, &emb);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "FALSIFIED EMB-002: cosine_similarity(x, x) = {sim} != 1.0 for test case {idx}"
        );
    }
}

/// FALSIFY-EMB-003: Average preserves dimension — dim(avg(xs)) = dim(xs[0])
#[test]
fn falsify_emb_003_average_preserves_dimension() {
    let dims = [1, 3, 64, 192, 512];
    for &dim in &dims {
        let embeddings: Vec<SpeakerEmbedding> = (0..5)
            .map(|i| {
                SpeakerEmbedding::from_vec((0..dim).map(|j| (i * dim + j) as f32 * 0.01).collect())
            })
            .collect();

        let avg = average_embeddings(&embeddings).unwrap_or_else(|_| {
            panic!("FALSIFIED EMB-003: average_embeddings failed for dim={dim}")
        });
        assert_eq!(
            avg.dim(),
            dim,
            "FALSIFIED EMB-003: avg dim {} != input dim {dim}",
            avg.dim()
        );
    }
}

/// FALSIFY-EMB-004: Cosine similarity symmetry — cos(a, b) = cos(b, a)
#[test]
fn falsify_emb_004_similarity_symmetry() {
    let a = SpeakerEmbedding::from_vec(vec![1.0, 2.0, -3.0, 0.5]);
    let b = SpeakerEmbedding::from_vec(vec![-0.5, 1.5, 0.0, 4.0]);

    let sim_ab = cosine_similarity(&a, &b);
    let sim_ba = cosine_similarity(&b, &a);

    assert!(
        (sim_ab - sim_ba).abs() < 1e-7,
        "FALSIFIED EMB-004: cos(a,b) = {sim_ab} != cos(b,a) = {sim_ba}"
    );
}

/// FALSIFY-EMB-005: Normalize produces unit norm — ||normalize(x)|| = 1.0
///
/// Exception: zero vector stays zero (norm = 0).
#[test]
fn falsify_emb_005_normalize_unit_norm() {
    let test_vecs = vec![
        vec![3.0, 4.0],
        vec![1.0, 1.0, 1.0, 1.0],
        vec![100.0, -200.0, 300.0, -400.0, 500.0],
        vec![1e-4, 1e-4],
    ];

    for (idx, v) in test_vecs.iter().enumerate() {
        let normalized = normalize_embedding(&SpeakerEmbedding::from_vec(v.clone()));
        let norm = normalized.l2_norm();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "FALSIFIED EMB-005: ||normalize(x)|| = {norm} != 1.0 for test case {idx}"
        );
    }
}

/// FALSIFY-EMB-006: Cosine similarity bounded — -1 <= cos(a, b) <= 1
#[test]
fn falsify_emb_006_cosine_bounded() {
    let test_pairs = vec![
        (vec![1.0, 0.0], vec![0.0, 1.0]),
        (vec![1.0, 1.0], vec![-1.0, -1.0]),
        (vec![100.0, -50.0, 25.0], vec![-30.0, 80.0, -10.0]),
        (vec![1e-8, 1e-8], vec![1e8, 1e8]),
    ];

    for (idx, (a, b)) in test_pairs.iter().enumerate() {
        let ea = SpeakerEmbedding::from_vec(a.clone());
        let eb = SpeakerEmbedding::from_vec(b.clone());
        let sim = cosine_similarity(&ea, &eb);
        assert!(
            (-1.0 - 1e-6..=1.0 + 1e-6).contains(&sim),
            "FALSIFIED EMB-006: cos(a,b) = {sim} out of [-1,1] for test case {idx}"
        );
    }
}

/// FALSIFY-EMB-007: Average is element-wise mean
///
/// avg([1,0], [0,1]) must equal [0.5, 0.5].
#[test]
fn falsify_emb_007_average_correctness() {
    let embeddings = vec![
        SpeakerEmbedding::from_vec(vec![2.0, 4.0, 6.0]),
        SpeakerEmbedding::from_vec(vec![4.0, 6.0, 8.0]),
        SpeakerEmbedding::from_vec(vec![6.0, 8.0, 10.0]),
    ];
    let avg = average_embeddings(&embeddings).expect("average should succeed");

    let expected = [4.0, 6.0, 8.0];
    for (i, (&actual, &exp)) in avg.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - exp).abs() < 1e-5,
            "FALSIFIED EMB-007: avg[{i}] = {actual}, expected {exp}"
        );
    }
}

mod voice_emb_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-EMB-002-prop: Cosine self-similarity = 1.0 for random vectors
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_emb_002_prop_cosine_self(
            a in -50.0f32..50.0,
            b in -50.0f32..50.0,
            c in -50.0f32..50.0,
        ) {
            prop_assume!(a.abs() + b.abs() + c.abs() > 1e-4);

            let emb = SpeakerEmbedding::from_vec(vec![a, b, c]);
            let sim = cosine_similarity(&emb, &emb);
            prop_assert!(
                (sim - 1.0).abs() < 1e-4,
                "FALSIFIED EMB-002-prop: cos(x,x)={} != 1.0 for [{},{},{}]",
                sim, a, b, c
            );
        }
    }

    /// FALSIFY-EMB-005-prop: Normalize produces unit norm for random vectors
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn falsify_emb_005_prop_unit_norm(
            a in -100.0f32..100.0,
            b in -100.0f32..100.0,
            c in -100.0f32..100.0,
        ) {
            prop_assume!(a.abs() + b.abs() + c.abs() > 1e-4);

            let emb = SpeakerEmbedding::from_vec(vec![a, b, c]);
            let normalized = normalize_embedding(&emb);
            let norm = normalized.l2_norm();
            prop_assert!(
                (norm - 1.0).abs() < 1e-4,
                "FALSIFIED EMB-005-prop: ||normalize(x)||={} != 1.0 for [{},{},{}]",
                norm, a, b, c
            );
        }
    }
}
