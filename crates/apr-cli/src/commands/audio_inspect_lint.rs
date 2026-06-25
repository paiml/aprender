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
        CliError::InvalidFormat(format!(
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
            "amplitude_bounds": format!("{bounds:?}"),
            "sample_rate": format!("{rate:?}"),
            "channel_shape": format!("{shape:?}"),
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
        assert!(matches!(err, CliError::InvalidFormat(_)));
    }
    #[test]
    fn empty_object_runs() {
        let f = w("{}");
        let _ = run(f.path(), None, None, true);
    }
}
