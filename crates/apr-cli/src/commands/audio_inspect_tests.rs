//! Tests for the `apr dataset audio-inspect` producer (aprender#2377 finding 3).
//!
//! The load-bearing test here is `round_trip_*`: the producer's own output is
//! fed to `apr audio-inspect-lint` and the lint must ACCEPT it. A shape
//! assertion ("the JSON has a `sample_rate` key") cannot prove producer and
//! consumer agree; running the consumer can.

use super::*;
use crate::commands::audio_inspect_lint;

// ── fixtures ─────────────────────────────────────────────────────────────

/// Build a RIFF/WAVE file in memory.
fn wav(format_tag: u16, channels: u16, rate: u32, bits: u16, payload: &[u8]) -> Vec<u8> {
    let mut fmt = Vec::new();
    fmt.extend_from_slice(&format_tag.to_le_bytes());
    fmt.extend_from_slice(&channels.to_le_bytes());
    fmt.extend_from_slice(&rate.to_le_bytes());
    let block_align = channels * (bits / 8);
    fmt.extend_from_slice(&(rate * u32::from(block_align)).to_le_bytes()); // byte rate
    fmt.extend_from_slice(&block_align.to_le_bytes());
    fmt.extend_from_slice(&bits.to_le_bytes());

    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + payload.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
    out.extend_from_slice(&fmt);
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

fn i16_payload(samples: &[i16]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

/// 16 kHz stereo i16, peaks at -0.5 and +0.5 of full scale.
fn stereo_16k() -> Vec<u8> {
    wav(
        1,
        2,
        16_000,
        16,
        &i16_payload(&[-16384, 16384, 0, 8192, -8192, 4096]),
    )
}

fn write(bytes: &[u8]) -> tempfile::NamedTempFile {
    use std::io::Write;
    let mut f = tempfile::NamedTempFile::new().expect("tempfile");
    f.write_all(bytes).expect("write");
    f.flush().expect("flush");
    f
}

// ── ROUND TRIP: producer output must be accepted by its own lint ─────────

#[test]
fn round_trip_producer_output_is_accepted_by_audio_inspect_lint() {
    let src = write(&stereo_16k());
    let dir = tempfile::tempdir().expect("tempdir");
    let obs = dir.path().join("audio.json");

    run(src.path(), true, Some(&obs), false).expect("producer must decode a valid 16-bit WAV");

    // The consumer, run exactly as its help documents, on exactly what the
    // producer wrote — with both optional assertions armed.
    audio_inspect_lint::run(&obs, Some(16_000), Some(2), false)
        .expect("audio-inspect-lint must accept the producer's own observation");
}

#[test]
fn round_trip_cannot_pass_vacuously_when_the_body_is_corrupted() {
    let src = write(&stereo_16k());
    let dir = tempfile::tempdir().expect("tempdir");
    let obs = dir.path().join("audio.json");
    run(src.path(), true, Some(&obs), false).expect("producer");

    let good: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&obs).expect("read")).expect("parse");

    // Each corruption is a DIFFERENT gate of the lint; all three must reject.
    for (label, mutate) in [
        (
            "amplitude above full scale",
            Box::new(|v: &mut serde_json::Value| v["max"] = serde_json::json!(1.5))
                as Box<dyn Fn(&mut serde_json::Value)>,
        ),
        (
            "non-canonical sample rate",
            Box::new(|v: &mut serde_json::Value| v["sample_rate"] = serde_json::json!(12_345)),
        ),
        (
            "zero frames",
            Box::new(|v: &mut serde_json::Value| v["samples"] = serde_json::json!(0)),
        ),
    ] {
        let mut bad = good.clone();
        mutate(&mut bad);
        let path = dir.path().join("bad.json");
        std::fs::write(&path, serde_json::to_string(&bad).expect("ser")).expect("write");
        let err = audio_inspect_lint::run(&path, None, None, false)
            .expect_err(&format!("lint must reject: {label}"));
        assert!(
            matches!(err, CliError::ValidationFailed(_)),
            "{label}: expected a validation refusal, got {err:?}"
        );
    }
}

// ── decode correctness ───────────────────────────────────────────────────

#[test]
fn decodes_i16_stereo_amplitudes_and_frame_count() {
    let src = write(&stereo_16k());
    let obs = inspect(src.path()).expect("decode");
    assert_eq!(obs.sample_rate, 16_000);
    assert_eq!(obs.channels, 2);
    assert_eq!(obs.samples, 3, "6 i16 samples over 2 channels is 3 frames");
    assert_eq!(obs.codec, "pcm_s16le");
    assert!((obs.min - -0.5).abs() < 1e-12, "min was {}", obs.min);
    assert!((obs.max - 0.5).abs() < 1e-12, "max was {}", obs.max);
}

#[test]
fn i16_full_negative_scale_normalises_to_exactly_minus_one() {
    let src = write(&wav(1, 1, 8_000, 16, &i16_payload(&[i16::MIN, i16::MAX])));
    let obs = inspect(src.path()).expect("decode");
    assert!((obs.min - -1.0).abs() < 1e-12, "min was {}", obs.min);
    assert!(
        obs.max < 1.0,
        "i16::MAX must stay below +1.0, got {}",
        obs.max
    );
}

#[test]
fn decodes_24_bit_pcm_with_sign_extension() {
    // -8_388_608 (full negative scale) then +8_388_607, little-endian 3-byte.
    let payload = vec![0x00, 0x00, 0x80, 0xFF, 0xFF, 0x7F];
    let src = write(&wav(1, 1, 48_000, 24, &payload));
    let obs = inspect(src.path()).expect("decode");
    assert_eq!(obs.codec, "pcm_s24le");
    assert!((obs.min - -1.0).abs() < 1e-12, "min was {}", obs.min);
    assert!(obs.max > 0.999_999, "max was {}", obs.max);
}

#[test]
fn decodes_u8_pcm_around_the_128_midpoint() {
    let src = write(&wav(1, 1, 8_000, 8, &[0u8, 128, 255]));
    let obs = inspect(src.path()).expect("decode");
    assert_eq!(obs.codec, "pcm_u8");
    assert!((obs.min - -1.0).abs() < 1e-12, "min was {}", obs.min);
    assert!(
        (obs.max - (127.0 / 128.0)).abs() < 1e-12,
        "max was {}",
        obs.max
    );
}

#[test]
fn reads_the_subformat_guid_of_wave_format_extensible() {
    // 0xFFFE with a 40-byte fmt whose SubFormat GUID starts with the PCM tag.
    let mut fmt = Vec::new();
    fmt.extend_from_slice(&0xFFFEu16.to_le_bytes());
    fmt.extend_from_slice(&1u16.to_le_bytes()); // channels
    fmt.extend_from_slice(&44_100u32.to_le_bytes());
    fmt.extend_from_slice(&88_200u32.to_le_bytes());
    fmt.extend_from_slice(&2u16.to_le_bytes()); // block align
    fmt.extend_from_slice(&16u16.to_le_bytes()); // bits
    fmt.extend_from_slice(&22u16.to_le_bytes()); // cbSize
    fmt.extend_from_slice(&16u16.to_le_bytes()); // valid bits
    fmt.extend_from_slice(&0u32.to_le_bytes()); // channel mask
    fmt.extend_from_slice(&1u16.to_le_bytes()); // SubFormat: PCM
    fmt.extend_from_slice(&[0u8; 14]);

    let payload = i16_payload(&[-16384, 16384]);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(20 + fmt.len() as u32 + payload.len() as u32).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&fmt);
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&payload);

    let src = write(&bytes);
    let obs = inspect(src.path()).expect("decode");
    assert_eq!(obs.sample_rate, 44_100);
    assert_eq!(obs.codec, "pcm_s16le");
}

// ── honest refusals: never a plausible number for a file we cannot read ──

#[test]
fn refuses_a_flac_stream_by_name() {
    let src = write(b"fLaC\x00\x00\x00\x22and then some");
    let err = inspect(src.path()).expect_err("FLAC must be refused, not guessed at");
    assert!(err.to_string().contains("FLAC"), "got: {err}");
}

#[test]
fn refuses_a_non_riff_file() {
    let src = write(b"this is not audio at all");
    let err = inspect(src.path()).expect_err("bad magic must be refused");
    assert!(err.to_string().contains("bad magic"), "got: {err}");
}

#[test]
fn refuses_a_truncated_data_chunk_instead_of_reporting_the_prefix() {
    let mut bytes = stereo_16k();
    bytes.truncate(bytes.len() - 4); // header still promises the full payload
    let src = write(&bytes);
    let err = inspect(src.path()).expect_err("a truncated file must be refused");
    assert!(err.to_string().contains("truncated"), "got: {err}");
}

#[test]
fn refuses_an_empty_stream_rather_than_reporting_zero_amplitude() {
    let src = write(&wav(1, 1, 16_000, 16, &[]));
    let err = inspect(src.path()).expect_err("0 frames has no amplitude to report");
    assert!(err.to_string().contains("0 frames"), "got: {err}");
}

#[test]
fn refuses_an_unsupported_codec_tag() {
    // 0x0011 = IMA ADPCM — a real WAV tag this decoder cannot decode.
    let src = write(&wav(0x0011, 1, 16_000, 16, &i16_payload(&[1, 2])));
    let err = inspect(src.path()).expect_err("ADPCM must be refused");
    assert!(err.to_string().contains("format tag 17"), "got: {err}");
}

#[test]
fn refuses_a_zero_channel_header() {
    let src = write(&wav(1, 0, 16_000, 16, &i16_payload(&[1, 2])));
    let err = inspect(src.path()).expect_err("0 channels is not decodable");
    assert!(err.to_string().contains("0 channels"), "got: {err}");
}

#[test]
fn refuses_a_missing_file() {
    let err = run(Path::new("/no/such/audio.wav"), true, None, false)
        .expect_err("a missing input must not be reported on");
    assert!(matches!(err, CliError::FileNotFound(_)), "{err:?}");
}

#[test]
fn refuses_to_clobber_an_existing_output_without_force() {
    let src = write(&stereo_16k());
    let existing = write(b"precious");
    let err = run(src.path(), true, Some(existing.path()), false)
        .expect_err("an existing output must not be overwritten silently");
    assert!(err.to_string().contains("--force"), "got: {err}");
}

// ── emitted shape ────────────────────────────────────────────────────────

#[test]
fn json_body_carries_the_five_keys_the_h13_classifier_reads() {
    let src = write(&stereo_16k());
    let obs = inspect(src.path()).expect("decode");
    let v = serde_json::to_value(&obs).expect("serialize");
    for key in ["min", "max", "sample_rate", "channels", "samples"] {
        assert!(
            v.get(key).is_some(),
            "the H-13 classifier reads `{key}` by name; it is absent from {v}"
        );
    }
    assert!(
        !v["sample_rate"].is_string(),
        "sample_rate must be a JSON number, not a rendering of one: {v}"
    );
}

#[test]
fn text_output_names_the_measured_fields() {
    let src = write(&stereo_16k());
    let obs = inspect(src.path()).expect("decode");
    let text = render_text(&obs);
    assert!(text.contains("sample_rate  : 16000"), "got: {text}");
    assert!(text.contains("channels     : 2"), "got: {text}");
}
