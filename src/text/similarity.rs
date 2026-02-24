//! Document similarity metrics.
//!
//! This module provides various similarity measures for comparing documents:
//! - Cosine similarity (TF-IDF vectors)
//! - Jaccard similarity (token overlap)
//! - Edit distance (Levenshtein)
//!
//! # Quick Start
//!
//! ```
//! use aprender::text::similarity::cosine_similarity;
//! use aprender::primitives::Vector;
//!
//! let v1 = Vector::from_slice(&[1.0, 2.0, 3.0]);
//! let v2 = Vector::from_slice(&[2.0, 3.0, 4.0]);
//!
//! let similarity = cosine_similarity(&v1, &v2).expect("cosine similarity should succeed");
//! println!("Cosine similarity: {:.3}", similarity);
//! ```

use crate::primitives::Vector;
use crate::AprenderError;

/// Compute cosine similarity between two vectors.
///
/// Cosine similarity measures the angle between two vectors in n-dimensional space.
/// Returns a value between -1 and 1, where:
/// - 1 = identical direction (very similar)
/// - 0 = orthogonal (unrelated)
/// - -1 = opposite direction (very dissimilar)
///
/// # Formula
/// ```text
/// cosine_sim(A, B) = (A · B) / (||A|| * ||B||)
/// ```
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Examples
///
/// ```
/// use aprender::text::similarity::cosine_similarity;
/// use aprender::primitives::Vector;
///
/// let v1 = Vector::from_slice(&[1.0, 2.0, 3.0]);
/// let v2 = Vector::from_slice(&[2.0, 3.0, 4.0]);
///
/// let sim = cosine_similarity(&v1, &v2).expect("cosine similarity should succeed");
/// assert!(sim > 0.9); // Very similar
/// ```
pub fn cosine_similarity(a: &Vector<f64>, b: &Vector<f64>) -> Result<f64, AprenderError> {
    if a.len() != b.len() {
        return Err(AprenderError::Other(
            "Vectors must have same length".to_string(),
        ));
    }

    if a.is_empty() {
        return Err(AprenderError::Other("Vectors cannot be empty".to_string()));
    }

    // Compute dot product
    let dot_product: f64 = a
        .as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(x, y)| x * y)
        .sum();

    // Compute norms
    let norm_a: f64 = a.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0); // Zero vector is orthogonal to everything
    }

    Ok(dot_product / (norm_a * norm_b))
}

/// Compute Jaccard similarity between two sets of tokens.
///
/// Jaccard similarity is the size of the intersection divided by the size of the union.
/// Returns a value between 0 and 1, where:
/// - 1 = identical sets
/// - 0 = no overlap
///
/// # Formula
/// ```text
/// jaccard(A, B) = |A ∩ B| / |A ∪ B|
/// ```
///
/// # Arguments
///
/// * `a` - First set of tokens
/// * `b` - Second set of tokens
///
/// # Examples
///
/// ```
/// use aprender::text::similarity::jaccard_similarity;
///
/// let tokens1 = vec!["the", "cat", "sat"];
/// let tokens2 = vec!["the", "dog", "sat"];
///
/// let sim = jaccard_similarity(&tokens1, &tokens2).expect("jaccard similarity should succeed");
/// assert!((sim - 0.5).abs() < 0.01); // 2 common / 4 total = 0.5
/// ```
pub fn jaccard_similarity<S: AsRef<str>>(a: &[S], b: &[S]) -> Result<f64, AprenderError> {
    if a.is_empty() && b.is_empty() {
        return Ok(1.0); // Empty sets are identical
    }

    if a.is_empty() || b.is_empty() {
        return Ok(0.0); // No overlap possible
    }

    // Convert to sets (unique tokens)
    let set_a: std::collections::HashSet<&str> = a.iter().map(AsRef::as_ref).collect();
    let set_b: std::collections::HashSet<&str> = b.iter().map(AsRef::as_ref).collect();

    // Intersection
    let intersection_size = set_a.intersection(&set_b).count();

    // Union
    let union_size = set_a.union(&set_b).count();

    if union_size == 0 {
        return Ok(0.0);
    }

    Ok(intersection_size as f64 / union_size as f64)
}

/// Compute Levenshtein edit distance between two strings.
///
/// Edit distance is the minimum number of single-character edits (insertions,
/// deletions, or substitutions) required to transform one string into another.
///
/// Lower values indicate more similar strings.
///
/// # Arguments
///
/// * `a` - First string
/// * `b` - Second string
///
/// # Examples
///
/// ```
/// use aprender::text::similarity::edit_distance;
///
/// let dist = edit_distance("kitten", "sitting").expect("edit distance should succeed");
/// assert_eq!(dist, 3); // 3 edits: k->s, e->i, +g
/// ```
pub fn edit_distance(a: &str, b: &str) -> Result<usize, AprenderError> {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let m = a_chars.len();
    let n = b_chars.len();

    // Handle empty strings
    if m == 0 {
        return Ok(n);
    }
    if n == 0 {
        return Ok(m);
    }

    // Dynamic programming matrix
    let mut dp = vec![vec![0usize; n + 1]; m + 1];

    // Initialize first row and column
    #[allow(clippy::needless_range_loop)]
    for i in 0..=m {
        dp[i][0] = i;
    }
    #[allow(clippy::needless_range_loop)]
    for j in 0..=n {
        dp[0][j] = j;
    }

    // Fill the matrix
    for i in 1..=m {
        for j in 1..=n {
            let cost = usize::from(a_chars[i - 1] != b_chars[j - 1]);

            dp[i][j] = std::cmp::min(
                std::cmp::min(
                    dp[i - 1][j] + 1, // Deletion
                    dp[i][j - 1] + 1, // Insertion
                ),
                dp[i - 1][j - 1] + cost, // Substitution
            );
        }
    }

    Ok(dp[m][n])
}

/// Compute normalized edit distance similarity.
///
/// Normalizes edit distance to a similarity score between 0 and 1 by dividing
/// by the maximum possible distance (length of longer string).
///
/// # Arguments
///
/// * `a` - First string
/// * `b` - Second string
///
/// # Examples
///
/// ```
/// use aprender::text::similarity::edit_distance_similarity;
///
/// let sim = edit_distance_similarity("kitten", "sitting").expect("edit distance similarity should succeed");
/// assert!(sim > 0.5); // Somewhat similar
/// ```
pub fn edit_distance_similarity(a: &str, b: &str) -> Result<f64, AprenderError> {
    let distance = edit_distance(a, b)?;
    let max_len = std::cmp::max(a.len(), b.len());

    if max_len == 0 {
        return Ok(1.0); // Empty strings are identical
    }

    Ok(1.0 - (distance as f64 / max_len as f64))
}

/// Compute pairwise cosine similarities between all documents.
///
/// Returns a similarity matrix where element (i, j) is the cosine similarity
/// between document i and document j.
///
/// # Arguments
///
/// * `vectors` - Document vectors (TF-IDF or embeddings)
///
/// # Examples
///
/// ```
/// use aprender::text::similarity::pairwise_cosine_similarity;
/// use aprender::primitives::Vector;
///
/// let docs = vec![
///     Vector::from_slice(&[1.0, 2.0, 3.0]),
///     Vector::from_slice(&[2.0, 3.0, 4.0]),
///     Vector::from_slice(&[0.0, 1.0, 0.0]),
/// ];
///
/// let similarities = pairwise_cosine_similarity(&docs).expect("pairwise cosine similarity should succeed");
/// assert_eq!(similarities.len(), 3); // 3 documents
/// assert_eq!(similarities[0].len(), 3); // 3 similarities per doc
/// ```
/// Initialize similarity matrix with self-similarity on diagonal.
fn init_similarity_matrix(n: usize) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; n]; n];
    for (i, row) in matrix.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    matrix
}

/// Compute and store symmetric similarity for a pair.
fn store_symmetric_similarity(
    similarities: &mut [Vec<f64>],
    vectors: &[Vector<f64>],
    i: usize,
    j: usize,
) -> Result<(), AprenderError> {
    let sim = cosine_similarity(&vectors[i], &vectors[j])?;
    similarities[i][j] = sim;
    similarities[j][i] = sim;
    Ok(())
}

pub fn pairwise_cosine_similarity(vectors: &[Vector<f64>]) -> Result<Vec<Vec<f64>>, AprenderError> {
    if vectors.is_empty() {
        return Ok(Vec::new());
    }

    let n = vectors.len();
    let mut similarities = init_similarity_matrix(n);

    // Compute upper triangle only (use symmetry)
    for i in 0..n {
        for j in (i + 1)..n {
            store_symmetric_similarity(&mut similarities, vectors, i, j)?;
        }
    }

    Ok(similarities)
}

/// Find top-k most similar documents to a query.
///
/// # Arguments
///
/// * `query` - Query vector
/// * `documents` - Document vectors
/// * `k` - Number of top results to return
///
/// # Returns
///
/// List of (index, similarity) pairs sorted by similarity (highest first)
///
/// # Examples
///
/// ```
/// use aprender::text::similarity::top_k_similar;
/// use aprender::primitives::Vector;
///
/// let query = Vector::from_slice(&[1.0, 2.0, 3.0]);
/// let docs = vec![
///     Vector::from_slice(&[2.0, 3.0, 4.0]),  // Similar
///     Vector::from_slice(&[0.0, 0.0, 1.0]),  // Less similar
///     Vector::from_slice(&[1.0, 2.0, 2.9]),  // Very similar
/// ];
///
/// let top = top_k_similar(&query, &docs, 2).expect("top_k_similar should succeed");
/// assert_eq!(top.len(), 2);
/// assert_eq!(top[0].0, 2); // Doc 2 is most similar
/// ```
pub fn top_k_similar(
    query: &Vector<f64>,
    documents: &[Vector<f64>],
    k: usize,
) -> Result<Vec<(usize, f64)>, AprenderError> {
    if documents.is_empty() {
        return Ok(Vec::new());
    }

    // Compute all similarities
    let mut similarities: Vec<(usize, f64)> = documents
        .iter()
        .enumerate()
        .map(|(idx, doc)| {
            let sim = cosine_similarity(query, doc).unwrap_or(0.0);
            (idx, sim)
        })
        .collect();

    // Sort by similarity (descending)
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top k
    similarities.truncate(k);

    Ok(similarities)
}

#[cfg(test)]
#[path = "similarity_tests.rs"]
mod tests;

// Similarity metrics contract falsification (FALSIFY-SIM-001..008)
// Refs: NLP spec §2.1.6, PMAT-349
#[cfg(test)]
#[path = "similarity_contract_falsify.rs"]
mod similarity_contract_falsify;
