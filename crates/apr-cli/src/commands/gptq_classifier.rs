//! CRUX-B-09 GPTQ quantization — algorithm-level classifiers.
//!
//! Partial discharge for the `apr quantize --method gptq` contract
//! (`contracts/crux-B-09-v1.yaml`). Three pure classifiers cover:
//!
//! 1. Compression ratio (GPTQ bytes ≤ 0.30 × fp16 bytes) — FALSIFY-001.
//! 2. Logit-fidelity cosine (mean cos ≥ 0.98 across N held-out prompts) — FALSIFY-002.
//! 3. CLI surface — `--method gptq --bits --group-size` — FALSIFY-003.

/// Maximum GPTQ-to-fp16 byte ratio.
pub const GPTQ_MAX_COMPRESSION_RATIO: f64 = 0.30;

/// Minimum mean logit-cosine the contract demands.
pub const GPTQ_MIN_MEAN_COSINE: f64 = 0.98;

/// Allowed GPTQ bit widths (matches auto-gptq reference).
pub const GPTQ_ALLOWED_BITS: &[u32] = &[2, 3, 4, 8];

/// Allowed GPTQ group sizes.
pub const GPTQ_ALLOWED_GROUP_SIZES: &[i32] = &[-1, 32, 64, 128];

/// Default group size (auto-gptq / GPTQ-for-LLaMa default).
pub const GPTQ_DEFAULT_GROUP_SIZE: i32 = 128;

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
        return CompressionOutcome::Insufficient { ratio: f64::INFINITY, max_ratio };
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
    Ok { mean: f64, n: usize },
    Degraded { mean: f64, threshold: f64, n: usize },
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

// ---- CLI surface ----

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GptqFlags {
    pub method: Option<String>,
    pub bits: Option<u32>,
    pub group_size: Option<i32>,
}

#[must_use]
pub fn parse_gptq_flags(argv: &[&str]) -> GptqFlags {
    let mut f = GptqFlags { method: None, bits: None, group_size: None };
    let mut i = 0;
    while i < argv.len() {
        let a = argv[i];
        match a {
            "--method" => f.method = argv.get(i + 1).map(|s| (*s).to_string()),
            "--bits" => f.bits = argv.get(i + 1).and_then(|s| s.parse::<u32>().ok()),
            "--group-size" => f.group_size = argv.get(i + 1).and_then(|s| s.parse::<i32>().ok()),
            _ => {
                if let Some(rest) = a.strip_prefix("--method=") {
                    f.method = Some(rest.to_string());
                } else if let Some(rest) = a.strip_prefix("--bits=") {
                    if let Ok(v) = rest.parse::<u32>() { f.bits = Some(v); }
                } else if let Some(rest) = a.strip_prefix("--group-size=") {
                    if let Ok(v) = rest.parse::<i32>() { f.group_size = Some(v); }
                }
            }
        }
        i += 1;
    }
    f
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GptqFlagValidation {
    Ok { bits: u32, group_size: i32 },
    MissingMethod,
    WrongMethod { got: String },
    InvalidBits { got: u32, allowed: &'static [u32] },
    MissingBits,
    InvalidGroupSize { got: i32, allowed: &'static [i32] },
}

#[must_use]
pub fn validate_gptq_flags(flags: &GptqFlags) -> GptqFlagValidation {
    let Some(method) = flags.method.as_deref() else {
        return GptqFlagValidation::MissingMethod;
    };
    if method != "gptq" {
        return GptqFlagValidation::WrongMethod { got: method.to_string() };
    }
    let Some(bits) = flags.bits else {
        return GptqFlagValidation::MissingBits;
    };
    if !GPTQ_ALLOWED_BITS.contains(&bits) {
        return GptqFlagValidation::InvalidBits { got: bits, allowed: GPTQ_ALLOWED_BITS };
    }
    let group_size = flags.group_size.unwrap_or(GPTQ_DEFAULT_GROUP_SIZE);
    if !GPTQ_ALLOWED_GROUP_SIZES.contains(&group_size) {
        return GptqFlagValidation::InvalidGroupSize {
            got: group_size,
            allowed: GPTQ_ALLOWED_GROUP_SIZES,
        };
    }
    GptqFlagValidation::Ok { bits, group_size }
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
        let pairs: Vec<(&[f64], &[f64])> =
            vec![(a.as_slice(), b.as_slice()), (bad_a.as_slice(), bad_b.as_slice())];
        match classify_mean_cosine(&pairs, GPTQ_MIN_MEAN_COSINE) {
            CosineFidelity::Ok { n, .. } => assert_eq!(n, 1),
            o => panic!("expected Ok with n=1, got {:?}", o),
        }
    }

    // ---- FALSIFY-003 (CLI) ----

    #[test]
    fn parse_all_three_space_form() {
        let argv = &["quantize", "--method", "gptq", "--bits", "4", "--group-size", "128"];
        let f = parse_gptq_flags(argv);
        assert_eq!(f.method.as_deref(), Some("gptq"));
        assert_eq!(f.bits, Some(4));
        assert_eq!(f.group_size, Some(128));
    }

    #[test]
    fn parse_group_size_neg_one_is_per_tensor() {
        // auto-gptq convention: group_size=-1 means per-tensor.
        let argv = &["--method=gptq", "--bits=4", "--group-size=-1"];
        let f = parse_gptq_flags(argv);
        assert_eq!(f.group_size, Some(-1));
    }

    #[test]
    fn validate_ok_with_default_group_size() {
        let f = GptqFlags { method: Some("gptq".into()), bits: Some(4), group_size: None };
        assert_eq!(
            validate_gptq_flags(&f),
            GptqFlagValidation::Ok { bits: 4, group_size: GPTQ_DEFAULT_GROUP_SIZE }
        );
    }

    #[test]
    fn validate_ok_with_per_tensor_group_size() {
        let f = GptqFlags { method: Some("gptq".into()), bits: Some(4), group_size: Some(-1) };
        assert_eq!(
            validate_gptq_flags(&f),
            GptqFlagValidation::Ok { bits: 4, group_size: -1 }
        );
    }

    #[test]
    fn validate_rejects_wrong_method() {
        let f = GptqFlags { method: Some("awq".into()), bits: Some(4), group_size: Some(128) };
        assert!(matches!(validate_gptq_flags(&f), GptqFlagValidation::WrongMethod { .. }));
    }

    #[test]
    fn validate_rejects_missing_bits() {
        let f = GptqFlags { method: Some("gptq".into()), bits: None, group_size: Some(128) };
        assert_eq!(validate_gptq_flags(&f), GptqFlagValidation::MissingBits);
    }

    #[test]
    fn validate_rejects_invalid_bits() {
        let f = GptqFlags { method: Some("gptq".into()), bits: Some(5), group_size: Some(128) };
        assert!(matches!(validate_gptq_flags(&f), GptqFlagValidation::InvalidBits { got: 5, .. }));
    }

    #[test]
    fn validate_rejects_invalid_group_size() {
        let f = GptqFlags { method: Some("gptq".into()), bits: Some(4), group_size: Some(96) };
        assert!(matches!(
            validate_gptq_flags(&f),
            GptqFlagValidation::InvalidGroupSize { got: 96, .. }
        ));
    }

    #[test]
    fn validate_is_deterministic() {
        let f = GptqFlags { method: Some("gptq".into()), bits: Some(4), group_size: None };
        assert_eq!(validate_gptq_flags(&f), validate_gptq_flags(&f));
    }

    #[test]
    fn reference_constants_match_spec() {
        assert!(GPTQ_ALLOWED_BITS.contains(&4));
        assert!(GPTQ_ALLOWED_GROUP_SIZES.contains(&128));
        assert!(GPTQ_ALLOWED_GROUP_SIZES.contains(&-1));
        assert_eq!(GPTQ_DEFAULT_GROUP_SIZE, 128);
    }
}
