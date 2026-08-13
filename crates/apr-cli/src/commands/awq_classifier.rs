//! CRUX-B-08 AWQ quantization — algorithm-level classifiers.
//!
//! Partial discharge for the AWQ quantization contract
//! (`contracts/crux-B-08-v1.yaml`). Two pure classifiers cover:
//!
//! 1. Quality retention (pass@1 AWQ ≥ 0.80 × pass@1 fp16) — FALSIFY-001.
//! 2. Compression ratio (AWQ bytes ≤ 0.30 × fp16 bytes) — FALSIFY-003.
//!
//! Full discharge still requires a real AWQ quantizer and real
//! HumanEval scoring — neither lives in the CLI crate.
//!
//! The CLI-surface gate (FALSIFY-002) used to live here as `parse_awq_flags` +
//! `validate_awq_flags`: a hand-rolled `--method`/`--bits`/`--group-size`
//! matcher that decided whether the shipped `apr quantize` would accept an
//! argv. `apr quantize` takes none of those three flags, so the gate had never
//! once validated the command it claimed to. It is deleted; the verdict now
//! comes from the shipped clap parser via
//! `commands::quantize_flag_parity::shipped_quantize_verdict`
//! (aprender#2377 finding 2, `contracts/apr-lint-flag-parity-v1.yaml`).

/// Minimum quality-retention ratio the contract demands.
pub const AWQ_MIN_QUALITY_RETENTION: f64 = 0.80;

/// Maximum compressed-to-source byte ratio for 4-bit AWQ.
pub const AWQ_MAX_COMPRESSION_RATIO: f64 = 0.30;

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
}
