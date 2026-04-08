#![allow(clippy::disallowed_methods)]
//! Chapter 19: Text Processing and Tokenization
//!
//! Demonstrates cosine similarity, Jaccard similarity, and edit distance.
//! Citation: Sennrich et al., "Subword Units," arXiv:1508.07909
//! Contract: contracts/apr-book-ch19-v1.yaml (v2 — api_calls enforced)

use aprender::primitives::Vector;
use aprender::text::similarity::{cosine_similarity, edit_distance, jaccard_similarity};

fn main() {
    // --- Cosine similarity ---
    let a = Vector::from_vec(vec![1.0_f64, 2.0, 3.0]);
    let b = Vector::from_vec(vec![1.0_f64, 2.0, 3.0]);
    let cos_identical = cosine_similarity(&a, &b).expect("cosine similarity");
    println!("Cosine similarity:");
    println!("  Identical vectors: {cos_identical:.4}");
    assert!(
        (cos_identical - 1.0).abs() < 1e-6,
        "Identical vectors must have cosine = 1.0"
    );

    // Orthogonal vectors -> cosine = 0.0
    let c = Vector::from_vec(vec![1.0_f64, 0.0, 0.0]);
    let d = Vector::from_vec(vec![0.0_f64, 1.0, 0.0]);
    let cos_orthogonal = cosine_similarity(&c, &d).expect("cosine similarity");
    println!("  Orthogonal vectors: {cos_orthogonal:.4}");
    assert!(
        cos_orthogonal.abs() < 1e-6,
        "Orthogonal vectors must have cosine = 0.0"
    );

    // --- Jaccard similarity ---
    let tokens_a = vec!["the", "cat", "sat", "on", "the", "mat"];
    let tokens_b = vec!["the", "cat", "sat", "on", "a", "hat"];
    let jaccard = jaccard_similarity(&tokens_a, &tokens_b).expect("jaccard similarity");
    println!("\nJaccard similarity:");
    println!("  '{}' vs '{}'", tokens_a.join(" "), tokens_b.join(" "));
    println!("  Jaccard: {jaccard:.4}");
    assert!(
        jaccard > 0.0 && jaccard < 1.0,
        "Jaccard in (0,1) for overlapping sets"
    );

    let jaccard_self = jaccard_similarity(&tokens_a, &tokens_a).expect("jaccard self-similarity");
    assert!(
        (jaccard_self - 1.0).abs() < 1e-6,
        "Self-similarity must be 1.0"
    );

    // --- Edit distance (Levenshtein) ---
    let s1 = "kitten";
    let s2 = "sitting";
    let dist = edit_distance(s1, s2).expect("edit distance");
    println!("\nEdit distance:");
    println!("  '{s1}' -> '{s2}': {dist}");
    assert_eq!(dist, 3, "kitten->sitting requires 3 edits");

    let dist_self = edit_distance("hello", "hello").expect("edit distance");
    assert_eq!(dist_self, 0, "Same string must have distance 0");

    println!("\nChapter 19 contracts: PASSED");
}
