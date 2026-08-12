//! `apr audio-inspect-lint` — CRUX-H-13 audio dataset loader gate.
//!
//! Reads a captured `apr dataset audio-inspect --format json` body and
//! dispatches the pure classifiers in `audio_inspect_classifier`. Exits
//! non-zero on any failure.
//!
//! Spec: `contracts/crux-H-13-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::audio_inspect_classifier::{
    classify_amplitude_bounds, classify_channel_shape, classify_sample_rate, AudioBoundsOutcome,
    AudioChannelShapeOutcome, AudioSampleRateOutcome,
};
use crate::error::{CliError, Result};

pub(crate) fn run(
    json_file: &Path,
    expected_sample_rate: Option<u32>,
    expected_channels: Option<u32>,
    json_out: bool,
) -> Result<()> {
    if !json_file.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(json_file)));
    }
    let body_text = std::fs::read_to_string(json_file)?;
    let body: Value = serde_json::from_str(&body_text).map_err(|e| {
        CliError::InvalidInput(format!(
            "apr audio-inspect-lint: failed to parse JSON from {}: {e}",
            json_file.display()
        ))
    })?;

    let bounds = classify_amplitude_bounds(&body);
    let rate = classify_sample_rate(&body, expected_sample_rate);
    let shape = classify_channel_shape(&body, expected_channels);

    print_report(json_file, &bounds, &rate, &shape, json_out);

    if !matches!(bounds, AudioBoundsOutcome::Ok { .. }) {
        return Err(CliError::ValidationFailed(format!(
            "audio-inspect-lint amplitude-bounds gate rejected body: {bounds:?}"
        )));
    }
    if !matches!(rate, AudioSampleRateOutcome::Ok { .. }) {
        return Err(CliError::ValidationFailed(format!(
            "audio-inspect-lint sample-rate gate rejected body: {rate:?}"
        )));
    }
    if !matches!(shape, AudioChannelShapeOutcome::Ok { .. }) {
        return Err(CliError::ValidationFailed(format!(
            "audio-inspect-lint channel-shape gate rejected body: {shape:?}"
        )));
    }
    Ok(())
}

fn print_report(
    path: &Path,
    bounds: &AudioBoundsOutcome,
    rate: &AudioSampleRateOutcome,
    shape: &AudioChannelShapeOutcome,
    json_out: bool,
) {
    if json_out {
        let obj = serde_json::json!({
            "file": path.display().to_string(),
            // aprender#2377(6): these were `format!("{x:?}")`, which puts a Rust
            // Debug rendering inside a JSON *string* — a consumer asking for the
            // sample rate got the characters `Ok { rate: 16000 }`. The outcome
            // enums are internally-tagged Serialize, so each is now an object
            // with a `status` discriminant and its real fields.
            "amplitude_bounds": bounds,
            "sample_rate": rate,
            "channel_shape": shape,
        });
        println!("{}", serde_json::to_string_pretty(&obj).unwrap_or_default());
        return;
    }
    println!("audio-inspect-lint report for {}", path.display());
    println!("  amplitude_bounds: {bounds:?}");
    println!("  sample_rate     : {rate:?}");
    println!("  channel_shape   : {shape:?}");
}

#[cfg(test)]
mod cov_tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    fn w(s: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(s.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }
    #[test]
    fn missing_file_is_file_not_found() {
        let err = run(Path::new("/no/such/audio.json"), None, None, false).unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }
    #[test]
    fn invalid_json_is_invalid_format() {
        let f = w("nope");
        let err = run(f.path(), None, None, false).unwrap_err();
        assert!(matches!(err, CliError::InvalidInput(_)));
    }
    #[test]
    fn empty_object_runs() {
        let f = w("{}");
        let _ = run(f.path(), None, None, true);
    }

    /// Dogfood 0.63.0 #2377 finding 5 at the command surface: the body says
    /// 4294983296, the pre-fix report said `Ok { rate: 16000 }` and exited 0.
    /// FALSIFIER (#2377-6): the JSON report carries FIELDS, not a Debug string.
    ///
    /// Before, `--json` emitted `"sample_rate": "ExpectedRateMismatch { got: \
    /// 8000, expected: 16000 }"` — a Rust rendering inside a JSON string, which
    /// no consumer can read and which changes whenever a variant is renamed.
    /// Asserting `!is_string()` is the half that would have caught the original
    /// defect; asserting the fields is the half that keeps it honest.
    #[test]
    fn json_outcomes_are_objects_with_fields_not_debug_strings() {
        let v = serde_json::to_value(AudioSampleRateOutcome::ExpectedRateMismatch {
            got: 8000,
            expected: 16000,
        })
        .expect("outcome serializes");

        assert!(
            !v.is_string(),
            "a JSON consumer must get an object, not a Debug rendering: {v}"
        );
        assert_eq!(v["status"], "expected_rate_mismatch");
        assert_eq!(v["got"], 8000);
        assert_eq!(v["expected"], 16000);

        // A unit variant still carries its discriminant rather than vanishing.
        let missing =
            serde_json::to_value(AudioSampleRateOutcome::MissingSampleRate).expect("serializes");
        assert_eq!(missing["status"], "missing_sample_rate");
    }

    #[test]
    fn out_of_range_sample_rate_fails_the_command() {
        let f = w(r#"{"min":-0.5,"max":0.5,"sample_rate":4294983296,"channels":1,"samples":100}"#);
        let err = run(f.path(), None, None, false).unwrap_err();
        assert!(matches!(err, CliError::ValidationFailed(_)), "{err:?}");
        let msg = err.to_string();
        assert!(msg.contains("4294983296"), "must echo the input: {msg}");
        assert!(!msg.contains("16000"), "must not invent a rate: {msg}");
    }

    /// The explicit assertion path was fooled identically: wrapped, the value
    /// equalled the `--expected-sample-rate` the user asked for.
    #[test]
    fn out_of_range_sample_rate_fails_even_with_expected_rate() {
        let f = w(r#"{"min":-0.5,"max":0.5,"sample_rate":4294983296,"channels":1,"samples":100}"#);
        let err = run(f.path(), Some(16_000), None, false).unwrap_err();
        assert!(err.to_string().contains("4294983296"), "{err}");
    }

    #[test]
    fn out_of_range_channel_count_fails_the_command() {
        let f =
            w(r#"{"min":-0.5,"max":0.5,"sample_rate":16000,"channels":4294967297,"samples":100}"#);
        let err = run(f.path(), None, None, false).unwrap_err();
        assert!(err.to_string().contains("4294967297"), "{err}");
    }

    #[test]
    fn a_well_formed_body_still_passes() {
        let f = w(r#"{"min":-0.5,"max":0.5,"sample_rate":16000,"channels":1,"samples":100}"#);
        assert!(run(f.path(), Some(16_000), Some(1), false).is_ok());
    }
}
