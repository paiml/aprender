//! Similarity Metrics Contract Falsification Tests
//!
//! Popperian falsification of NLP spec §2.1.6 claims:
//!   - Cosine similarity: self-similarity = 1, symmetric, range [-1, 1]
//!   - Jaccard similarity: identity property, symmetric, range [0, 1]
//!   - Edit distance: identity = 0, triangle inequality, non-negative
//!   - Pairwise similarity matrix: symmetric, diagonal = 1
//!
//! Five-Whys (PMAT-349):
//!   Why #1: similarity module has unit tests but zero FALSIFY-SIM-* contract tests
//!   Why #2: unit tests verify examples, not metric space axioms
//!   Why #3: no provable-contract YAML for similarity metrics
//!   Why #4: similarity module was built before DbC methodology
//!   Why #5: no systematic check that metric axioms (symmetry, triangle inequality) hold
//!
//! References:
//!   - docs/specifications/nlp-models-techniques-spec.md §2.1.6
//!   - src/text/similarity.rs

use crate::primitives::Vector;
use crate::text::similarity::*;

// ============================================================================
// FALSIFY-SIM-001: Cosine self-similarity
// Contract: cosine_similarity(v, v) = 1.0 for any non-zero vector
// ============================================================================

#[test]
fn falsify_sim_001_cosine_self_similarity() {
    let vectors = vec![
        Vector::from_slice(&[1.0, 0.0, 0.0]),
        Vector::from_slice(&[1.0, 2.0, 3.0]),
        Vector::from_slice(&[-1.0, -2.0, -3.0]),
        Vector::from_slice(&[0.5, 0.5, 0.5, 0.5]),
    ];

    for (i, v) in vectors.iter().enumerate() {
        let sim = cosine_similarity(v, v).expect("cosine self-similarity");
        assert!(
            (sim - 1.0).abs() < 1e-12,
            "FALSIFIED SIM-001: cosine_similarity(v{i}, v{i}) = {sim}, expected 1.0"
        );
    }
}

// ============================================================================
// FALSIFY-SIM-002: Cosine symmetry
// Contract: cosine_similarity(a, b) == cosine_similarity(b, a)
// ============================================================================

#[test]
fn falsify_sim_002_cosine_symmetry() {
    let pairs = vec![
        (
            Vector::from_slice(&[1.0, 2.0, 3.0]),
            Vector::from_slice(&[4.0, 5.0, 6.0]),
        ),
        (
            Vector::from_slice(&[1.0, 0.0]),
            Vector::from_slice(&[0.0, 1.0]),
        ),
        (
            Vector::from_slice(&[-1.0, 2.0, -3.0]),
            Vector::from_slice(&[3.0, -2.0, 1.0]),
        ),
    ];

    for (i, (a, b)) in pairs.iter().enumerate() {
        let ab = cosine_similarity(a, b).expect("cos(a,b)");
        let ba = cosine_similarity(b, a).expect("cos(b,a)");
        assert!(
            (ab - ba).abs() < 1e-12,
            "FALSIFIED SIM-002: cos(a{i},b{i}) = {ab} != cos(b{i},a{i}) = {ba}"
        );
    }
}

// ============================================================================
// FALSIFY-SIM-003: Cosine range
// Contract: cosine_similarity always returns value in [-1, 1]
// ============================================================================

#[test]
fn falsify_sim_003_cosine_range() {
    let vectors = vec![
        Vector::from_slice(&[1.0, 0.0]),
        Vector::from_slice(&[0.0, 1.0]),
        Vector::from_slice(&[-1.0, 0.0]),
        Vector::from_slice(&[1.0, 1.0]),
        Vector::from_slice(&[100.0, -200.0, 300.0]),
        Vector::from_slice(&[-0.001, 0.002, -0.003]),
    ];

    for (i, a) in vectors.iter().enumerate() {
        for (j, b) in vectors.iter().enumerate() {
            if a.len() != b.len() {
                continue;
            }
            let sim = cosine_similarity(a, b).expect("cosine");
            assert!(
                (-1.0 - 1e-12..=1.0 + 1e-12).contains(&sim),
                "FALSIFIED SIM-003: cosine(v{i}, v{j}) = {sim} out of [-1, 1]"
            );
        }
    }
}

// ============================================================================
// FALSIFY-SIM-004: Jaccard metric axioms
// Contract: jaccard(a,a) = 1, jaccard(a,b) = jaccard(b,a), range [0, 1]
// ============================================================================

#[test]
fn falsify_sim_004_jaccard_self_identity() {
    let tokens = vec!["cat", "dog", "bird"];
    let sim = jaccard_similarity(&tokens, &tokens).expect("jaccard self");
    assert!(
        (sim - 1.0).abs() < 1e-12,
        "FALSIFIED SIM-004: jaccard(a, a) = {sim}, expected 1.0"
    );
}

#[test]
fn falsify_sim_004_jaccard_symmetry() {
    let a = vec!["cat", "dog", "bird"];
    let b = vec!["dog", "fish", "snake"];

    let ab = jaccard_similarity(&a, &b).expect("jaccard(a,b)");
    let ba = jaccard_similarity(&b, &a).expect("jaccard(b,a)");

    assert!(
        (ab - ba).abs() < 1e-12,
        "FALSIFIED SIM-004: jaccard(a,b) = {ab} != jaccard(b,a) = {ba}"
    );
}

#[test]
fn falsify_sim_004_jaccard_range() {
    let cases = vec![
        (vec!["a", "b", "c"], vec!["d", "e", "f"]), // no overlap
        (vec!["a", "b", "c"], vec!["a", "b", "c"]), // identical
        (vec!["a", "b", "c"], vec!["b", "c", "d"]), // partial
    ];

    for (i, (a, b)) in cases.iter().enumerate() {
        let sim = jaccard_similarity(a, b).expect("jaccard");
        assert!(
            (0.0..=1.0).contains(&sim),
            "FALSIFIED SIM-004: jaccard case {i} = {sim} out of [0, 1]"
        );
    }
}

#[test]
fn falsify_sim_004_jaccard_empty_sets() {
    let empty: Vec<&str> = vec![];
    let non_empty = vec!["a"];

    // Two empty sets are identical
    let sim_ee = jaccard_similarity(&empty, &empty).expect("empty,empty");
    assert!(
        (sim_ee - 1.0).abs() < 1e-12,
        "FALSIFIED SIM-004: jaccard(∅, ∅) = {sim_ee}, expected 1.0"
    );

    // Empty vs non-empty = 0
    let sim_en = jaccard_similarity(&empty, &non_empty).expect("empty,nonempty");
    assert!(
        sim_en.abs() < 1e-12,
        "FALSIFIED SIM-004: jaccard(∅, {{a}}) = {sim_en}, expected 0.0"
    );
}

// ============================================================================
// FALSIFY-SIM-005: Edit distance identity
// Contract: edit_distance(a, a) = 0
// ============================================================================

#[test]
fn falsify_sim_005_edit_distance_identity() {
    let strings = ["", "hello", "rust programming", "日本語", "a b c"];

    for s in &strings {
        let dist = edit_distance(s, s).expect("edit_distance(s,s)");
        assert_eq!(
            dist, 0,
            "FALSIFIED SIM-005: edit_distance('{s}', '{s}') = {dist}, expected 0"
        );
    }
}

#[test]
fn falsify_sim_005_edit_distance_non_negative() {
    let pairs = [
        ("kitten", "sitting"),
        ("", "hello"),
        ("abc", ""),
        ("rust", "dust"),
    ];

    for (a, b) in &pairs {
        let dist = edit_distance(a, b).expect("edit_distance");
        // usize is always ≥ 0, but this documents the contract
        assert!(
            dist <= a.len() + b.len(),
            "FALSIFIED SIM-005: edit_distance('{a}', '{b}') = {dist} > max possible {}",
            a.len() + b.len()
        );
    }
}

// ============================================================================
// FALSIFY-SIM-006: Edit distance triangle inequality
// Contract: d(a,c) ≤ d(a,b) + d(b,c) for all strings a, b, c
// ============================================================================

#[test]
fn falsify_sim_006_triangle_inequality() {
    let triples = [
        ("kitten", "sitting", "knitting"),
        ("abc", "abd", "xyz"),
        ("rust", "dust", "must"),
        ("hello", "", "world"),
    ];

    for (a, b, c) in &triples {
        let d_ac = edit_distance(a, c).expect("d(a,c)");
        let d_ab = edit_distance(a, b).expect("d(a,b)");
        let d_bc = edit_distance(b, c).expect("d(b,c)");

        assert!(
            d_ac <= d_ab + d_bc,
            "FALSIFIED SIM-006: d('{a}','{c}')={d_ac} > d('{a}','{b}')+d('{b}','{c}')={} (triangle inequality violated)",
            d_ab + d_bc
        );
    }
}

// ============================================================================
// FALSIFY-SIM-007: Pairwise similarity matrix properties
// Contract: diagonal = 1, symmetric
// ============================================================================

#[test]
fn falsify_sim_007_pairwise_matrix_properties() {
    let vectors = vec![
        Vector::from_slice(&[1.0, 2.0, 3.0]),
        Vector::from_slice(&[4.0, 5.0, 6.0]),
        Vector::from_slice(&[7.0, 8.0, 9.0]),
    ];

    let matrix = pairwise_cosine_similarity(&vectors).expect("pairwise");

    // Diagonal must be 1.0
    for i in 0..vectors.len() {
        assert!(
            (matrix[i][i] - 1.0).abs() < 1e-12,
            "FALSIFIED SIM-007: diagonal[{i}] = {}, expected 1.0",
            matrix[i][i]
        );
    }

    // Must be symmetric
    for i in 0..vectors.len() {
        for j in 0..vectors.len() {
            assert!(
                (matrix[i][j] - matrix[j][i]).abs() < 1e-12,
                "FALSIFIED SIM-007: matrix[{i}][{j}]={} != matrix[{j}][{i}]={}",
                matrix[i][j],
                matrix[j][i]
            );
        }
    }
}

// ============================================================================
// FALSIFY-SIM-008: Edit distance similarity normalized range
// Contract: edit_distance_similarity returns value in [0, 1]
// ============================================================================

#[test]
fn falsify_sim_008_edit_distance_similarity_range() {
    let pairs = [
        ("hello", "hello"), // identical = 1.0
        ("abc", "xyz"),     // completely different
        ("", ""),           // both empty = 1.0
        ("kitten", "sitting"),
        ("rust", "dust"),
    ];

    for (a, b) in &pairs {
        let sim = edit_distance_similarity(a, b).expect("edit_distance_similarity");
        assert!(
            (0.0..=1.0 + 1e-12).contains(&sim),
            "FALSIFIED SIM-008: edit_distance_similarity('{a}', '{b}') = {sim} out of [0, 1]"
        );
    }
}
