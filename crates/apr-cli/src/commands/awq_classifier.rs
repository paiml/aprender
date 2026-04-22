//! CRUX-B-08 AWQ quantization — algorithm-level classifiers.
//!
//! Partial discharge for the `apr quantize --method awq` contract
//! (`contracts/crux-B-08-v1.yaml`). Three pure classifiers cover:
//!
//! 1. Quality retention (pass@1 AWQ ≥ 0.80 × pass@1 fp16) — FALSIFY-001.
//! 2. CLI flag parser + validation (`--method`, `--bits`, `--group-size`) — FALSIFY-002.
//! 3. Compression ratio (AWQ bytes ≤ 0.30 × fp16 bytes) — FALSIFY-003.
//!
//! Full discharge still requires a real AWQ quantizer and real
//! HumanEval scoring — neither lives in the CLI crate.

/// Default AWQ group size, matching vllm/awq reference implementation.
pub const AWQ_DEFAULT_GROUP_SIZE: u32 = 128;

/// Minimum quality-retention ratio the contract demands.
pub const AWQ_MIN_QUALITY_RETENTION: f64 = 0.80;

/// Maximum compressed-to-source byte ratio for 4-bit AWQ.
pub const AWQ_MAX_COMPRESSION_RATIO: f64 = 0.30;

/// Allowed AWQ bit widths.
pub const AWQ_ALLOWED_BITS: &[u32] = &[3, 4, 8];

/// Allowed AWQ group sizes.
pub const AWQ_ALLOWED_GROUP_SIZES: &[u32] = &[64, 128];

/// Outcome of comparing fp16 vs AWQ pass@1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QualityRetention {
    Retained { ratio: f64 },
    Degraded { ratio: f64, threshold: f64 },
}

/// Classify whether AWQ retained enough of fp16's pass@1 to meet contract.
///
/// Returns `Degraded` (not panic) when `p_fp16 <= 0.0` — the baseline
/// itself is broken, not the AWQ output, but the gate still fails.
#[must_use]
pub fn classify_quality_retention(p_fp16: f64, p_awq: f64, threshold: f64) -> QualityRetention {
    if !p_fp16.is_finite() || p_fp16 <= 0.0 {
        return QualityRetention::Degraded {
            ratio: f64::NAN,
            threshold,
        };
    }
    let ratio = p_awq / p_fp16;
    if ratio >= threshold {
        QualityRetention::Retained { ratio }
    } else {
        QualityRetention::Degraded { ratio, threshold }
    }
}

/// Outcome of comparing artifact size against the compression ceiling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionOutcome {
    Compressed { ratio: f64 },
    Insufficient { ratio: f64, max_ratio: f64 },
}

/// Classify whether the AWQ output is small enough relative to fp16.
/// `ratio = awq_bytes / fp16_bytes`; contract wants `ratio <= 0.30`.
#[must_use]
pub fn classify_compression_ratio(
    fp16_bytes: u64,
    awq_bytes: u64,
    max_ratio: f64,
) -> CompressionOutcome {
    if fp16_bytes == 0 {
        return CompressionOutcome::Insufficient {
            ratio: f64::INFINITY,
            max_ratio,
        };
    }
    let ratio = awq_bytes as f64 / fp16_bytes as f64;
    if ratio <= max_ratio {
        CompressionOutcome::Compressed { ratio }
    } else {
        CompressionOutcome::Insufficient { ratio, max_ratio }
    }
}

/// Parsed AWQ CLI surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AwqFlags {
    pub method: Option<String>,
    pub bits: Option<u32>,
    pub group_size: Option<u32>,
}

/// Parse `--method`, `--bits`, `--group-size` from argv.
/// Accepts both space and `=` separated forms.
#[must_use]
pub fn parse_awq_flags(argv: &[&str]) -> AwqFlags {
    let mut out = AwqFlags {
        method: None,
        bits: None,
        group_size: None,
    };
    let mut i = 0;
    while i < argv.len() {
        let a = argv[i];
        match a {
            "--method" => {
                out.method = argv.get(i + 1).map(|s| (*s).to_string());
            }
            "--bits" => {
                out.bits = argv.get(i + 1).and_then(|s| s.parse::<u32>().ok());
            }
            "--group-size" => {
                out.group_size = argv.get(i + 1).and_then(|s| s.parse::<u32>().ok());
            }
            _ => {
                if let Some(rest) = a.strip_prefix("--method=") {
                    out.method = Some(rest.to_string());
                } else if let Some(rest) = a.strip_prefix("--bits=") {
                    if let Ok(v) = rest.parse::<u32>() {
                        out.bits = Some(v);
                    }
                } else if let Some(rest) = a.strip_prefix("--group-size=") {
                    if let Ok(v) = rest.parse::<u32>() {
                        out.group_size = Some(v);
                    }
                }
            }
        }
        i += 1;
    }
    out
}

/// Validation outcome for a parsed AWQ flag set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AwqFlagValidation {
    Ok { bits: u32, group_size: u32 },
    MissingMethod,
    UnknownMethod { got: String },
    InvalidBits { got: u32, allowed: &'static [u32] },
    InvalidGroupSize { got: u32, allowed: &'static [u32] },
}

/// Validate parsed AWQ flags. Applies `AWQ_DEFAULT_GROUP_SIZE` when
/// `--group-size` is omitted. `--bits` has no default (must be given).
#[must_use]
pub fn validate_awq_flags(flags: &AwqFlags) -> AwqFlagValidation {
    let Some(method) = flags.method.as_deref() else {
        return AwqFlagValidation::MissingMethod;
    };
    if method != "awq" {
        return AwqFlagValidation::UnknownMethod {
            got: method.to_string(),
        };
    }
    let bits = match flags.bits {
        Some(b) if AWQ_ALLOWED_BITS.contains(&b) => b,
        Some(b) => {
            return AwqFlagValidation::InvalidBits {
                got: b,
                allowed: AWQ_ALLOWED_BITS,
            }
        }
        None => {
            return AwqFlagValidation::InvalidBits {
                got: 0,
                allowed: AWQ_ALLOWED_BITS,
            }
        }
    };
    let group_size = flags.group_size.unwrap_or(AWQ_DEFAULT_GROUP_SIZE);
    if !AWQ_ALLOWED_GROUP_SIZES.contains(&group_size) {
        return AwqFlagValidation::InvalidGroupSize {
            got: group_size,
            allowed: AWQ_ALLOWED_GROUP_SIZES,
        };
    }
    AwqFlagValidation::Ok { bits, group_size }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- FALSIFY-001 (quality retention) ----

    #[test]
    fn retention_above_threshold_is_retained() {
        let r = classify_quality_retention(0.50, 0.45, AWQ_MIN_QUALITY_RETENTION);
        assert!(matches!(r, QualityRetention::Retained { .. }));
    }

    #[test]
    fn retention_exactly_at_threshold_is_retained() {
        let r = classify_quality_retention(0.50, 0.40, AWQ_MIN_QUALITY_RETENTION);
        match r {
            QualityRetention::Retained { ratio } => assert!((ratio - 0.80).abs() < 1e-9),
            _ => panic!("expected Retained at exact threshold"),
        }
    }

    #[test]
    fn retention_below_threshold_is_degraded() {
        let r = classify_quality_retention(0.50, 0.30, AWQ_MIN_QUALITY_RETENTION);
        assert!(matches!(r, QualityRetention::Degraded { .. }));
    }

    #[test]
    fn retention_zero_baseline_is_degraded_not_panic() {
        let r = classify_quality_retention(0.0, 0.45, AWQ_MIN_QUALITY_RETENTION);
        assert!(matches!(r, QualityRetention::Degraded { .. }));
    }

    #[test]
    fn retention_is_deterministic() {
        let a = classify_quality_retention(0.42, 0.35, AWQ_MIN_QUALITY_RETENTION);
        let b = classify_quality_retention(0.42, 0.35, AWQ_MIN_QUALITY_RETENTION);
        assert_eq!(format!("{:?}", a), format!("{:?}", b));
    }

    // ---- FALSIFY-003 (compression) ----

    #[test]
    fn compression_well_under_ceiling_is_compressed() {
        let r = classify_compression_ratio(1_000_000, 200_000, AWQ_MAX_COMPRESSION_RATIO);
        assert!(matches!(r, CompressionOutcome::Compressed { .. }));
    }

    #[test]
    fn compression_exactly_at_ceiling_is_compressed() {
        let r = classify_compression_ratio(1_000_000, 300_000, AWQ_MAX_COMPRESSION_RATIO);
        match r {
            CompressionOutcome::Compressed { ratio } => assert!((ratio - 0.30).abs() < 1e-9),
            _ => panic!("expected Compressed at exact ceiling"),
        }
    }

    #[test]
    fn compression_over_ceiling_is_insufficient() {
        let r = classify_compression_ratio(1_000_000, 400_000, AWQ_MAX_COMPRESSION_RATIO);
        assert!(matches!(r, CompressionOutcome::Insufficient { .. }));
    }

    #[test]
    fn compression_zero_source_is_insufficient() {
        let r = classify_compression_ratio(0, 100, AWQ_MAX_COMPRESSION_RATIO);
        assert!(matches!(r, CompressionOutcome::Insufficient { .. }));
    }

    // ---- FALSIFY-002 (CLI parsing + validation) ----

    #[test]
    fn parse_all_three_space_form() {
        let argv = &[
            "quantize",
            "model.apr",
            "--method",
            "awq",
            "--bits",
            "4",
            "--group-size",
            "128",
        ];
        let f = parse_awq_flags(argv);
        assert_eq!(f.method.as_deref(), Some("awq"));
        assert_eq!(f.bits, Some(4));
        assert_eq!(f.group_size, Some(128));
    }

    #[test]
    fn parse_all_three_equals_form() {
        let argv = &["quantize", "--method=awq", "--bits=4", "--group-size=128"];
        let f = parse_awq_flags(argv);
        assert_eq!(f.method.as_deref(), Some("awq"));
        assert_eq!(f.bits, Some(4));
        assert_eq!(f.group_size, Some(128));
    }

    #[test]
    fn parse_absent_bits_yields_none() {
        let argv = &["quantize", "--method", "awq"];
        let f = parse_awq_flags(argv);
        assert_eq!(f.bits, None);
    }

    #[test]
    fn validate_ok_with_default_group_size() {
        let f = AwqFlags {
            method: Some("awq".into()),
            bits: Some(4),
            group_size: None,
        };
        assert_eq!(
            validate_awq_flags(&f),
            AwqFlagValidation::Ok {
                bits: 4,
                group_size: AWQ_DEFAULT_GROUP_SIZE
            }
        );
    }

    #[test]
    fn validate_ok_with_explicit_group_size() {
        let f = AwqFlags {
            method: Some("awq".into()),
            bits: Some(4),
            group_size: Some(64),
        };
        assert_eq!(
            validate_awq_flags(&f),
            AwqFlagValidation::Ok {
                bits: 4,
                group_size: 64
            }
        );
    }

    #[test]
    fn validate_rejects_missing_method() {
        let f = AwqFlags {
            method: None,
            bits: Some(4),
            group_size: Some(128),
        };
        assert_eq!(validate_awq_flags(&f), AwqFlagValidation::MissingMethod);
    }

    #[test]
    fn validate_rejects_unknown_method() {
        let f = AwqFlags {
            method: Some("gptq".into()),
            bits: Some(4),
            group_size: Some(128),
        };
        assert!(matches!(
            validate_awq_flags(&f),
            AwqFlagValidation::UnknownMethod { .. }
        ));
    }

    #[test]
    fn validate_rejects_invalid_bits() {
        let f = AwqFlags {
            method: Some("awq".into()),
            bits: Some(5),
            group_size: Some(128),
        };
        assert!(matches!(
            validate_awq_flags(&f),
            AwqFlagValidation::InvalidBits { got: 5, .. }
        ));
    }

    #[test]
    fn validate_rejects_missing_bits() {
        let f = AwqFlags {
            method: Some("awq".into()),
            bits: None,
            group_size: Some(128),
        };
        assert!(matches!(
            validate_awq_flags(&f),
            AwqFlagValidation::InvalidBits { got: 0, .. }
        ));
    }

    #[test]
    fn validate_rejects_invalid_group_size() {
        let f = AwqFlags {
            method: Some("awq".into()),
            bits: Some(4),
            group_size: Some(96),
        };
        assert!(matches!(
            validate_awq_flags(&f),
            AwqFlagValidation::InvalidGroupSize { got: 96, .. }
        ));
    }

    #[test]
    fn allowed_sets_include_reference_values() {
        assert!(AWQ_ALLOWED_BITS.contains(&4));
        assert!(AWQ_ALLOWED_GROUP_SIZES.contains(&128));
        assert_eq!(AWQ_DEFAULT_GROUP_SIZE, 128);
    }

    #[test]
    fn validate_is_deterministic() {
        let f = AwqFlags {
            method: Some("awq".into()),
            bits: Some(4),
            group_size: None,
        };
        let a = validate_awq_flags(&f);
        let b = validate_awq_flags(&f);
        assert_eq!(a, b);
    }
}
