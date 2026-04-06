//! Flash Attention: IO-aware tiled attention.
//!
//! Matches `flash-attention-v1.yaml`.
//! Online softmax with running max and running sum per tile.
//!
//! Each function provides one of three backends:
//! - `fn flash_attention_scalar(...)` -- Pure Rust scalar reference (ground truth)
//! - `unsafe fn flash_attention_avx2(...)` -- AVX2 SIMD implementation
//! - `fn flash_attention_ptx() -> &'static str` -- PTX assembly source string

use super::ops;

// ────────────────────────────────────────────────────────────────────────────
// Naive attention helper (for comparison in tests)
// ────────────────────────────────────────────────────────────────────────────

/// Naive scaled dot-product self-attention (for verification only).
///
/// Q, K, V are all n x d. Output is n x d.
#[cfg(test)]
fn naive_attention(q: &[f32], k: &[f32], v: &[f32], n: usize, d: usize, output: &mut [f32]) {
    let mut scores = vec![0.0f32; n * n];
    ops::score_matrix(q, k, n, n, d, &mut scores);
    ops::softmax_rows(&mut scores, n, n);
    ops::matmul_sv(&scores, v, n, n, d, output);
}

// ────────────────────────────────────────────────────────────────────────────
// Scalar implementation
// ────────────────────────────────────────────────────────────────────────────

/// Flash Attention: IO-aware tiled attention with online softmax.
///
/// Q, K, V are all n x d (row-major). Output is n x d.
///
/// For each query row i, processes KV in tiles of `tile_size` rows, maintaining
/// a running max and running sum for numerically stable online softmax.
///
/// # Panics
/// Panics if `q.len() != n*d`, `k.len() != n*d`, `v.len() != n*d`,
/// `output.len() != n*d`, or `tile_size == 0`.
pub fn flash_attention_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    d: usize,
    tile_size: usize,
    output: &mut [f32],
) {
    assert_eq!(
        q.len(),
        n * d,
        "Q dimension mismatch: expected {} got {}",
        n * d,
        q.len()
    );
    assert_eq!(
        k.len(),
        n * d,
        "K dimension mismatch: expected {} got {}",
        n * d,
        k.len()
    );
    assert_eq!(
        v.len(),
        n * d,
        "V dimension mismatch: expected {} got {}",
        n * d,
        v.len()
    );
    assert_eq!(
        output.len(),
        n * d,
        "output dimension mismatch: expected {} got {}",
        n * d,
        output.len()
    );
    assert!(tile_size > 0, "tile_size must be > 0");

    let scale = 1.0 / (d as f32).sqrt();

    for i in 0..n {
        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;
        let mut acc = vec![0.0f32; d];

        // Process KV in tiles
        let mut tile_start = 0;
        while tile_start < n {
            let tile_end = (tile_start + tile_size).min(n);
            process_tile(
                q,
                k,
                v,
                i,
                d,
                scale,
                tile_start,
                tile_end,
                &mut running_max,
                &mut running_sum,
                &mut acc,
            );
            tile_start = tile_end;
        }

        // Normalize
        normalize_row(&acc, running_sum, &mut output[i * d..(i + 1) * d]);
    }
}

/// Process a single tile of KV for one query row, updating online softmax state.
#[allow(clippy::too_many_arguments)]
fn process_tile(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    i: usize,
    d: usize,
    scale: f32,
    tile_start: usize,
    tile_end: usize,
    running_max: &mut f32,
    running_sum: &mut f32,
    acc: &mut [f32],
) {
    let tile_len = tile_end - tile_start;

    // Compute scores for this tile: Q[i] . K[j] / sqrt(d)
    let mut tile_scores = vec![0.0f32; tile_len];
    let q_row = &q[i * d..(i + 1) * d];
    for (tj, j) in (tile_start..tile_end).enumerate() {
        tile_scores[tj] = ops::dot(q_row, &k[j * d..(j + 1) * d]) * scale;
    }

    // Find max of this tile
    let tile_max = tile_scores
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let new_max = running_max.max(tile_max);

    // Correction factor: rescale previous accumulation
    let correction = (*running_max - new_max).exp();
    for a in acc.iter_mut() {
        *a *= correction;
    }
    *running_sum *= correction;

    // Accumulate this tile
    for (tj, j) in (tile_start..tile_end).enumerate() {
        let w = (tile_scores[tj] - new_max).exp();
        ops::weighted_accumulate(acc, w, &v[j * d..(j + 1) * d]);
        *running_sum += w;
    }

    *running_max = new_max;
}

/// Normalize accumulated attention output by the softmax denominator.
fn normalize_row(acc: &[f32], running_sum: f32, output: &mut [f32]) {
    if running_sum > 0.0 {
        for (o, a) in output.iter_mut().zip(acc.iter()) {
            *o = a / running_sum;
        }
    } else {
        for o in output.iter_mut() {
            *o = 0.0;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AVX2 implementation
// ────────────────────────────────────────────────────────────────────────────

/// AVX2 Flash Attention -- delegates to scalar.
///
/// The tiled online softmax algorithm is inherently sequential along the tile
/// dimension; the scalar implementation is the reference.
///
/// # Safety
/// Requires AVX2 support. Caller must verify with `is_x86_feature_detected!("avx2")`.
///
/// # Panics
/// Panics if dimensions do not match or `tile_size == 0`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn flash_attention_avx2(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    d: usize,
    tile_size: usize,
    output: &mut [f32],
) {
    flash_attention_scalar(q, k, v, n, d, tile_size, output);
}

// ────────────────────────────────────────────────────────────────────────────
// PTX implementation
// ────────────────────────────────────────────────────────────────────────────

include!("flash_attention_ptx.rs");

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    include!("flash_attention_tests.rs");
}
