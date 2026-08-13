//! CRUX-B-09 GPTQ quantization — algorithm-level classifiers.
//!
//! Partial discharge for the GPTQ quantization contract
//! (`contracts/crux-B-09-v1.yaml`). Two pure classifiers cover:
//!
//! 1. Compression ratio (GPTQ bytes ≤ 0.30 × fp16 bytes) — FALSIFY-001.
//! 2. Logit-fidelity cosine (mean cos ≥ 0.98 across N held-out prompts) — FALSIFY-002.
//!
//! The CLI-surface gate (FALSIFY-003) used to live here as `parse_gptq_flags` +
//! `validate_gptq_flags`: a hand-rolled `--method`/`--bits`/`--group-size`
//! matcher that decided whether the shipped `apr quantize` would accept an
//! argv. `apr quantize` takes none of those three flags, so the gate had never
//! once validated the command it claimed to. It is deleted; the verdict now
//! comes from the shipped clap parser via
//! `commands::quantize_flag_parity::shipped_quantize_verdict`
//! (aprender#2377 finding 2, `contracts/apr-lint-flag-parity-v1.yaml`).

/// Maximum GPTQ-to-fp16 byte ratio.
pub const GPTQ_MAX_COMPRESSION_RATIO: f64 = 0.30;

/// Minimum mean logit-cosine the contract demands.
pub const GPTQ_MIN_MEAN_COSINE: f64 = 0.98;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionOutcome {
    Compressed { ratio: f64 },
    Insufficient { ratio: f64, max_ratio: f64 },
}

#[must_use]
pub fn classify_compression_ratio(
    fp16_bytes: u64,
    gptq_bytes: u64,
    max_ratio: f64,
) -> CompressionOutcome {
    if fp16_bytes == 0 {
        return CompressionOutcome::Insufficient {
            ratio: f64::INFINITY,
            max_ratio,
        };
    }
    let ratio = gptq_bytes as f64 / fp16_bytes as f64;
    if ratio <= max_ratio {
        CompressionOutcome::Compressed { ratio }
    } else {
        CompressionOutcome::Insufficient { ratio, max_ratio }
    }
}

/// Cosine similarity of two equal-length vectors.
/// Returns `None` if lengths differ or either norm is zero.
#[must_use]
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return None;
    }
    Some(dot / (na.sqrt() * nb.sqrt()))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CosineFidelity {
    Ok {
        mean: f64,
        n: usize,
    },
    Degraded {
        mean: f64,
        threshold: f64,
        n: usize,
    },
    /// No valid per-prompt pairs (e.g. length mismatches only).
    NoSamples,
}

/// Classify per-prompt logit cosine fidelity across N prompts.
/// `pairs` holds matched (fp16_logits, gptq_logits) per prompt.
#[must_use]
pub fn classify_mean_cosine(pairs: &[(&[f64], &[f64])], threshold: f64) -> CosineFidelity {
    let cosines: Vec<f64> = pairs
        .iter()
        .filter_map(|(a, b)| cosine_similarity(a, b))
        .collect();
    let n = cosines.len();
    if n == 0 {
        return CosineFidelity::NoSamples;
    }
    let mean = cosines.iter().sum::<f64>() / n as f64;
    if mean >= threshold {
        CosineFidelity::Ok { mean, n }
    } else {
        CosineFidelity::Degraded { mean, threshold, n }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- FALSIFY-001 (compression) ----

    #[test]
    fn compression_under_ceiling_ok() {
        assert!(matches!(
            classify_compression_ratio(1_000_000, 200_000, GPTQ_MAX_COMPRESSION_RATIO),
            CompressionOutcome::Compressed { .. }
        ));
    }

    #[test]
    fn compression_at_exact_ceiling_ok() {
        match classify_compression_ratio(1_000_000, 300_000, GPTQ_MAX_COMPRESSION_RATIO) {
            CompressionOutcome::Compressed { ratio } => assert!((ratio - 0.30).abs() < 1e-9),
            _ => panic!("expected Compressed at exact ceiling"),
        }
    }

    #[test]
    fn compression_over_ceiling_flagged() {
        assert!(matches!(
            classify_compression_ratio(1_000_000, 400_000, GPTQ_MAX_COMPRESSION_RATIO),
            CompressionOutcome::Insufficient { .. }
        ));
    }

    #[test]
    fn compression_zero_source_is_insufficient() {
        assert!(matches!(
            classify_compression_ratio(0, 100, GPTQ_MAX_COMPRESSION_RATIO),
            CompressionOutcome::Insufficient { .. }
        ));
    }

    // ---- FALSIFY-002 (cosine fidelity) ----

    #[test]
    fn cosine_identical_vectors_is_one() {
        let v = vec![1.0, 2.0, 3.0];
        let c = cosine_similarity(&v, &v).unwrap();
        assert!((c - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let c = cosine_similarity(&a, &b).unwrap();
        assert!(c.abs() < 1e-12);
    }

    #[test]
    fn cosine_opposite_is_negative_one() {
        let a = vec![1.0, 2.0];
        let b = vec![-1.0, -2.0];
        let c = cosine_similarity(&a, &b).unwrap();
        assert!((c - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn cosine_mismatched_length_is_none() {
        assert!(cosine_similarity(&[1.0, 2.0], &[1.0]).is_none());
    }

    #[test]
    fn cosine_zero_norm_is_none() {
        assert!(cosine_similarity(&[0.0, 0.0], &[1.0, 2.0]).is_none());
    }

    #[test]
    fn mean_cosine_all_perfect_meets_threshold() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![0.5, 1.0, 1.5]; // same direction, scaled
        let pairs: Vec<(&[f64], &[f64])> = vec![
            (v1.as_slice(), v2.as_slice()),
            (v1.as_slice(), v2.as_slice()),
        ];
        let r = classify_mean_cosine(&pairs, GPTQ_MIN_MEAN_COSINE);
        match r {
            CosineFidelity::Ok { mean, n } => {
                assert!(mean >= GPTQ_MIN_MEAN_COSINE);
                assert_eq!(n, 2);
            }
            o => panic!("expected Ok, got {:?}", o),
        }
    }

    #[test]
    fn mean_cosine_degraded_below_threshold() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0]; // cosine = 0
        let pairs: Vec<(&[f64], &[f64])> = vec![(a.as_slice(), b.as_slice())];
        assert!(matches!(
            classify_mean_cosine(&pairs, GPTQ_MIN_MEAN_COSINE),
            CosineFidelity::Degraded { .. }
        ));
    }

    #[test]
    fn mean_cosine_no_valid_pairs_is_no_samples() {
        // All pairs length-mismatched → filtered away.
        let a = vec![1.0];
        let b = vec![1.0, 2.0];
        let pairs: Vec<(&[f64], &[f64])> = vec![(a.as_slice(), b.as_slice())];
        assert_eq!(
            classify_mean_cosine(&pairs, GPTQ_MIN_MEAN_COSINE),
            CosineFidelity::NoSamples
        );
    }

    #[test]
    fn mean_cosine_skips_invalid_pairs_but_counts_rest() {
        // One valid + one length-mismatch — mean is over the valid subset only.
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0];
        let bad_a = vec![1.0];
        let bad_b = vec![1.0, 2.0];
        let pairs: Vec<(&[f64], &[f64])> = vec![
            (a.as_slice(), b.as_slice()),
            (bad_a.as_slice(), bad_b.as_slice()),
        ];
        match classify_mean_cosine(&pairs, GPTQ_MIN_MEAN_COSINE) {
            CosineFidelity::Ok { n, .. } => assert_eq!(n, 1),
            o => panic!("expected Ok with n=1, got {:?}", o),
        }
    }
}
