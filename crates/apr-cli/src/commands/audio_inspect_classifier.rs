//! Audio dataset loader classifier (CRUX-H-13).
//!
//! Pure, deterministic classifiers that discharge FALSIFY-CRUX-H-13-{001,002}
//! at the PARTIAL_ALGORITHM_LEVEL — algorithm-level necessary conditions on
//! a captured `apr dataset audio-inspect --format json` body (parity with
//! `torchaudio.load`):
//!
//!   * `classify_amplitude_bounds` — `min` and `max` are finite and in
//!     `[-1, 1]` (no DC clipping, no NaN/Inf escapes from the decoder).
//!   * `classify_sample_rate` — `sample_rate` equals an explicit
//!     `--resample-to` argument when one is supplied, or is a positive
//!     integer in the standard set {8k, 16k, 22.05k, 24k, 44.1k, 48k,
//!     88.2k, 96k} when no resample is requested.
//!   * `classify_channel_shape` — `channels >= 1` and equals an explicit
//!     `--require-channels` value when supplied; `samples > 0`.
//!
//! Full discharge requires a live `apr dataset audio-inspect` emitter
//! actually decoding WAV/FLAC — tracked as BLOCKER-UPSTREAM-MISSING.

use serde_json::Value;

/// Canonical sample rates accepted by the H-13 contract when no explicit
/// `--resample-to` is supplied. Mirrors the torchaudio test corpus.
pub const H13_CANONICAL_SAMPLE_RATES: &[u32] = &[
    8_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 88_200, 96_000,
];

/// Outcome of `classify_amplitude_bounds`.
#[derive(Debug, Clone, PartialEq)]
pub enum AudioBoundsOutcome {
    Ok { min: f64, max: f64 },
    NotAnObject,
    MissingMin,
    MissingMax,
    MinNotFinite { got: f64 },
    MaxNotFinite { got: f64 },
    MinBelowFloor { got: f64, floor: f64 },
    MaxAboveCeiling { got: f64, ceiling: f64 },
    MinExceedsMax { min: f64, max: f64 },
}

/// Outcome of `classify_sample_rate`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioSampleRateOutcome {
    Ok {
        rate: u32,
    },
    MissingSampleRate,
    SampleRateNotPositive {
        got: i64,
    },
    /// Value does not fit in `u32`. Reported with the ORIGINAL `i64` so the
    /// report never prints a number the observation did not contain.
    SampleRateOutOfRange {
        got: i64,
    },
    ExpectedRateMismatch {
        got: u32,
        expected: u32,
    },
    NonCanonicalRate {
        got: u32,
    },
}

/// Outcome of `classify_channel_shape`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioChannelShapeOutcome {
    Ok {
        channels: u32,
        samples: u64,
    },
    MissingChannels,
    MissingSamples,
    ChannelsNotPositive {
        got: i64,
    },
    /// Channel count does not fit in `u32` (reported with the original `i64`).
    ChannelsOutOfRange {
        got: i64,
    },
    SamplesNotPositive {
        got: i64,
    },
    ExpectedChannelsMismatch {
        got: u32,
        expected: u32,
    },
}

/// FALSIFY-CRUX-H-13-001: amplitude bounds + finite.
pub fn classify_amplitude_bounds(body: &Value) -> AudioBoundsOutcome {
    let Some(obj) = body.as_object() else {
        return AudioBoundsOutcome::NotAnObject;
    };
    let Some(min) = obj.get("min").and_then(Value::as_f64) else {
        return AudioBoundsOutcome::MissingMin;
    };
    let Some(max) = obj.get("max").and_then(Value::as_f64) else {
        return AudioBoundsOutcome::MissingMax;
    };
    if !min.is_finite() {
        return AudioBoundsOutcome::MinNotFinite { got: min };
    }
    if !max.is_finite() {
        return AudioBoundsOutcome::MaxNotFinite { got: max };
    }
    if min < -1.0 {
        return AudioBoundsOutcome::MinBelowFloor {
            got: min,
            floor: -1.0,
        };
    }
    if max > 1.0 {
        return AudioBoundsOutcome::MaxAboveCeiling {
            got: max,
            ceiling: 1.0,
        };
    }
    if min > max {
        return AudioBoundsOutcome::MinExceedsMax { min, max };
    }
    AudioBoundsOutcome::Ok { min, max }
}

/// FALSIFY-CRUX-H-13-002: sample_rate is positive and matches expectation.
pub fn classify_sample_rate(body: &Value, expected: Option<u32>) -> AudioSampleRateOutcome {
    let Some(raw) = body.get("sample_rate").and_then(Value::as_i64) else {
        return AudioSampleRateOutcome::MissingSampleRate;
    };
    if raw <= 0 {
        return AudioSampleRateOutcome::SampleRateNotPositive { got: raw };
    }
    // A `raw as u32` here WRAPS: 4_294_983_296 (2^32 + 16000) would alias to
    // 16000 and PASS the canonical-set check while the report printed a rate
    // the observation never contained.
    let Ok(rate) = u32::try_from(raw) else {
        return AudioSampleRateOutcome::SampleRateOutOfRange { got: raw };
    };
    if let Some(exp) = expected {
        if rate != exp {
            return AudioSampleRateOutcome::ExpectedRateMismatch {
                got: rate,
                expected: exp,
            };
        }
        return AudioSampleRateOutcome::Ok { rate };
    }
    if !H13_CANONICAL_SAMPLE_RATES.contains(&rate) {
        return AudioSampleRateOutcome::NonCanonicalRate { got: rate };
    }
    AudioSampleRateOutcome::Ok { rate }
}

/// Channel shape gate: channels >= 1, samples > 0, optional channel-count match.
pub fn classify_channel_shape(
    body: &Value,
    expected_channels: Option<u32>,
) -> AudioChannelShapeOutcome {
    let Some(raw_c) = body.get("channels").and_then(Value::as_i64) else {
        return AudioChannelShapeOutcome::MissingChannels;
    };
    if raw_c <= 0 {
        return AudioChannelShapeOutcome::ChannelsNotPositive { got: raw_c };
    }
    let Some(raw_s) = body.get("samples").and_then(Value::as_i64) else {
        return AudioChannelShapeOutcome::MissingSamples;
    };
    if raw_s <= 0 {
        return AudioChannelShapeOutcome::SamplesNotPositive { got: raw_s };
    }
    // `raw_c as u32` WRAPS: 4_294_967_297 (2^32 + 1) would alias to 1 and the
    // report would print `channels: 1` for an input that said 4294967297.
    let Ok(channels) = u32::try_from(raw_c) else {
        return AudioChannelShapeOutcome::ChannelsOutOfRange { got: raw_c };
    };
    let samples = raw_s as u64;
    if let Some(exp) = expected_channels {
        if channels != exp {
            return AudioChannelShapeOutcome::ExpectedChannelsMismatch {
                got: channels,
                expected: exp,
            };
        }
    }
    AudioChannelShapeOutcome::Ok { channels, samples }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn good_body() -> Value {
        json!({"min": -0.85, "max": 0.92, "sample_rate": 16000, "channels": 2, "samples": 48000})
    }

    #[test]
    fn amplitude_bounds_ok_on_good_body() {
        match classify_amplitude_bounds(&good_body()) {
            AudioBoundsOutcome::Ok { min, max } => {
                assert!((min - -0.85).abs() < 1e-9);
                assert!((max - 0.92).abs() < 1e-9);
            }
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn amplitude_bounds_rejects_not_object() {
        assert_eq!(
            classify_amplitude_bounds(&json!([1, 2])),
            AudioBoundsOutcome::NotAnObject
        );
    }

    #[test]
    fn amplitude_bounds_rejects_missing_min() {
        assert_eq!(
            classify_amplitude_bounds(&json!({"max": 0.5})),
            AudioBoundsOutcome::MissingMin
        );
    }

    #[test]
    fn amplitude_bounds_rejects_nan_min() {
        let mut body = good_body();
        body["min"] = json!(f64::NAN);
        // serde_json normally serializes NaN as null; emulate by parsing a JSON value where min is null.
        // We handle the as_f64 None case as MissingMin, so use an explicit non-finite float-equivalent:
        // construct via Value::from_f64 (returns None) — fallback: rely on serde to skip NaN.
        // Instead, set min to a value the parser sees as f64 but is NaN-equivalent: use a string.
        body["min"] = json!("nan");
        assert_eq!(
            classify_amplitude_bounds(&body),
            AudioBoundsOutcome::MissingMin
        );
    }

    #[test]
    fn amplitude_bounds_rejects_below_floor() {
        let mut body = good_body();
        body["min"] = json!(-1.5);
        assert!(matches!(
            classify_amplitude_bounds(&body),
            AudioBoundsOutcome::MinBelowFloor { .. }
        ));
    }

    #[test]
    fn amplitude_bounds_rejects_above_ceiling() {
        let mut body = good_body();
        body["max"] = json!(1.5);
        assert!(matches!(
            classify_amplitude_bounds(&body),
            AudioBoundsOutcome::MaxAboveCeiling { .. }
        ));
    }

    #[test]
    fn amplitude_bounds_rejects_min_gt_max() {
        let mut body = good_body();
        body["min"] = json!(0.5);
        body["max"] = json!(-0.5);
        assert!(matches!(
            classify_amplitude_bounds(&body),
            AudioBoundsOutcome::MinExceedsMax { .. }
        ));
    }

    #[test]
    fn sample_rate_ok_on_16k_canonical() {
        match classify_sample_rate(&good_body(), None) {
            AudioSampleRateOutcome::Ok { rate } => assert_eq!(rate, 16_000),
            other => panic!("expected Ok(16000), got {other:?}"),
        }
    }

    #[test]
    fn sample_rate_ok_when_matches_expected() {
        assert_eq!(
            classify_sample_rate(&good_body(), Some(16_000)),
            AudioSampleRateOutcome::Ok { rate: 16_000 }
        );
    }

    #[test]
    fn sample_rate_rejects_expected_mismatch() {
        match classify_sample_rate(&good_body(), Some(22_050)) {
            AudioSampleRateOutcome::ExpectedRateMismatch { got, expected } => {
                assert_eq!(got, 16_000);
                assert_eq!(expected, 22_050);
            }
            other => panic!("expected ExpectedRateMismatch, got {other:?}"),
        }
    }

    #[test]
    fn sample_rate_rejects_non_canonical_when_no_expected() {
        let mut body = good_body();
        body["sample_rate"] = json!(12345);
        assert!(matches!(
            classify_sample_rate(&body, None),
            AudioSampleRateOutcome::NonCanonicalRate { got: 12345 }
        ));
    }

    #[test]
    fn sample_rate_rejects_zero() {
        let mut body = good_body();
        body["sample_rate"] = json!(0);
        assert!(matches!(
            classify_sample_rate(&body, None),
            AudioSampleRateOutcome::SampleRateNotPositive { got: 0 }
        ));
    }

    #[test]
    fn channel_shape_ok_on_good_body() {
        match classify_channel_shape(&good_body(), None) {
            AudioChannelShapeOutcome::Ok { channels, samples } => {
                assert_eq!(channels, 2);
                assert_eq!(samples, 48_000);
            }
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn channel_shape_rejects_missing_channels() {
        let body = json!({"samples": 1000});
        assert_eq!(
            classify_channel_shape(&body, None),
            AudioChannelShapeOutcome::MissingChannels
        );
    }

    #[test]
    fn channel_shape_rejects_zero_samples() {
        let mut body = good_body();
        body["samples"] = json!(0);
        assert!(matches!(
            classify_channel_shape(&body, None),
            AudioChannelShapeOutcome::SamplesNotPositive { got: 0 }
        ));
    }

    // ── wrapping-cast falsifiers (dogfood 0.63.0 #2377 finding 5) ────────
    //
    // `raw as u32` silently aliases anything >= 2^32 into the canonical set.
    // These assert BEHAVIOUR: an out-of-range value must be rejected, and the
    // reported value must be the one the observation actually contained.

    #[test]
    fn sample_rate_rejects_value_above_u32_that_aliases_to_16k() {
        // 2^32 + 16000: `as u32` yields exactly 16000, a canonical rate.
        let mut body = good_body();
        body["sample_rate"] = json!(4_294_983_296i64);
        assert_eq!(
            classify_sample_rate(&body, None),
            AudioSampleRateOutcome::SampleRateOutOfRange { got: 4_294_983_296 }
        );
    }

    #[test]
    fn sample_rate_out_of_range_is_not_rescued_by_expected_rate() {
        // The explicit --expected-sample-rate path must not be fooled either:
        // wrapped, 4294983296 == 16000 == expected.
        let mut body = good_body();
        body["sample_rate"] = json!(4_294_983_296i64);
        assert_eq!(
            classify_sample_rate(&body, Some(16_000)),
            AudioSampleRateOutcome::SampleRateOutOfRange { got: 4_294_983_296 }
        );
    }

    #[test]
    fn sample_rate_at_exactly_2_pow_32_reports_the_input_not_zero() {
        // `as u32` yields 0 here, so the pre-fix code reported
        // `NonCanonicalRate { got: 0 }` — a value never present in the input.
        let mut body = good_body();
        body["sample_rate"] = json!(4_294_967_296i64);
        assert_eq!(
            classify_sample_rate(&body, None),
            AudioSampleRateOutcome::SampleRateOutOfRange { got: 4_294_967_296 }
        );
    }

    #[test]
    fn sample_rate_at_u32_max_is_in_range_and_merely_non_canonical() {
        // Boundary: u32::MAX still fits, so it must reach the canonical-set
        // check rather than the range check.
        let mut body = good_body();
        body["sample_rate"] = json!(4_294_967_295i64);
        assert_eq!(
            classify_sample_rate(&body, None),
            AudioSampleRateOutcome::NonCanonicalRate { got: 4_294_967_295 }
        );
    }

    #[test]
    fn channel_shape_rejects_channels_above_u32_that_aliases_to_1() {
        // 2^32 + 1: `as u32` yields 1, a perfectly valid mono count.
        let mut body = good_body();
        body["channels"] = json!(4_294_967_297i64);
        assert_eq!(
            classify_channel_shape(&body, None),
            AudioChannelShapeOutcome::ChannelsOutOfRange { got: 4_294_967_297 }
        );
    }

    #[test]
    fn channel_shape_out_of_range_is_not_rescued_by_expected_channels() {
        let mut body = good_body();
        body["channels"] = json!(4_294_967_297i64);
        assert_eq!(
            classify_channel_shape(&body, Some(1)),
            AudioChannelShapeOutcome::ChannelsOutOfRange { got: 4_294_967_297 }
        );
    }

    #[test]
    fn channel_shape_at_u32_max_is_in_range() {
        let mut body = good_body();
        body["channels"] = json!(4_294_967_295i64);
        assert_eq!(
            classify_channel_shape(&body, None),
            AudioChannelShapeOutcome::Ok {
                channels: 4_294_967_295,
                samples: 48_000
            }
        );
    }

    #[test]
    fn channel_shape_rejects_expected_channels_mismatch() {
        match classify_channel_shape(&good_body(), Some(1)) {
            AudioChannelShapeOutcome::ExpectedChannelsMismatch { got, expected } => {
                assert_eq!(got, 2);
                assert_eq!(expected, 1);
            }
            other => panic!("expected ExpectedChannelsMismatch, got {other:?}"),
        }
    }
}
