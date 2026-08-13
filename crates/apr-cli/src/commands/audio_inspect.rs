//! `apr dataset audio-inspect` — the PRODUCER whose output `apr audio-inspect-lint`
//! reads (aprender#2377 finding 3).
//!
//! CRUX-H-13 shipped a consumer with no producer: `audio-inspect-lint`'s help
//! documented `apr dataset audio-inspect --format json` and the shipped binary
//! had neither a `dataset` command nor an `audio-inspect` one, so the lint's
//! gates had never run on real data and could not. This module decodes a real
//! RIFF/WAVE file and emits exactly the observation the H-13 classifier
//! consumes: `min`, `max`, `sample_rate`, `channels`, `samples`.
//!
//! ## What this claims, and what it does not
//!
//! It claims to have decoded uncompressed RIFF/WAVE PCM (u8 / i16 / i24 / i32)
//! and IEEE-float32 payloads and to report the *measured* amplitude extrema of
//! the decoded samples. It does NOT claim torchaudio parity beyond that: there
//! is no resampling, no channel mixdown, and no compressed-codec support. Every
//! container or codec it cannot decode is REFUSED with a non-zero exit and a
//! message naming what it found — never a plausible-looking number.
//!
//! Amplitude normalisation follows the torchaudio `load(normalize=True)`
//! convention: integer PCM is divided by the *negative* full-scale magnitude
//! (2^(bits-1)), so the range is [-1.0, 1.0 - 1ulp] and never exceeds ±1 for
//! integer input. Float payloads are reported as stored — a float WAV that
//! overshoots ±1 is a real property of that file and the lint is entitled to
//! reject it.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::error::{refuse_overwrite, CliError, Result};

/// Decoded audio observation — the exact body `apr audio-inspect-lint` reads.
///
/// Field names are load-bearing: `min`/`max`/`sample_rate`/`channels`/`samples`
/// are what `audio_inspect_classifier` looks up by key.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct AudioObservation {
    /// Source file, as given on the command line.
    pub path: String,
    /// Decoded codec, e.g. `pcm_s16le` or `pcm_f32le`.
    pub codec: String,
    /// Sample rate in Hz, straight from the `fmt ` chunk.
    pub sample_rate: u32,
    /// Channel count, straight from the `fmt ` chunk.
    pub channels: u32,
    /// Frames per channel (torchaudio's `num_frames`).
    pub samples: u64,
    /// `samples / sample_rate`.
    pub duration_secs: f64,
    /// Smallest decoded amplitude across every channel.
    pub min: f64,
    /// Largest decoded amplitude across every channel.
    pub max: f64,
    /// Bits per stored sample.
    pub bits_per_sample: u16,
}

/// Parsed `fmt ` chunk fields this decoder acts on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WavFmt {
    format_tag: u16,
    channels: u16,
    sample_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
}

const FMT_PCM: u16 = 1;
const FMT_IEEE_FLOAT: u16 = 3;
const FMT_EXTENSIBLE: u16 = 0xFFFE;

/// Run the producer: decode `path` and emit the observation.
pub(crate) fn run(path: &Path, json: bool, output: Option<&Path>, force: bool) -> Result<()> {
    if !path.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(path)));
    }
    if let Some(out) = output {
        refuse_overwrite(out, force)?;
    }

    let obs = inspect(path)?;
    let rendered = if json {
        serde_json::to_string_pretty(&obs).map_err(|e| {
            CliError::InvalidInput(format!("apr dataset audio-inspect: cannot serialize: {e}"))
        })?
    } else {
        render_text(&obs)
    };

    match output {
        Some(out) => std::fs::write(out, format!("{rendered}\n"))?,
        None => println!("{rendered}"),
    }
    Ok(())
}

fn render_text(o: &AudioObservation) -> String {
    format!(
        "audio-inspect {}\n  codec        : {}\n  sample_rate  : {}\n  channels     : {}\n  \
         samples      : {}\n  duration_secs: {:.6}\n  min          : {}\n  max          : {}",
        o.path, o.codec, o.sample_rate, o.channels, o.samples, o.duration_secs, o.min, o.max
    )
}

/// Decode `path` into an observation, or refuse with a message naming what was found.
pub(crate) fn inspect(path: &Path) -> Result<AudioObservation> {
    let bytes = std::fs::read(path)?;
    reject_known_non_wav(&bytes, path)?;
    let fmt = parse_fmt(&bytes, path)?;
    let (data, declared) = find_chunk(&bytes, b"data").ok_or_else(|| {
        CliError::InvalidInput(format!(
            "apr dataset audio-inspect: {} has no `data` chunk",
            path.display()
        ))
    })?;
    if declared > data.len() {
        // The header promises more audio than the file holds. Reporting extrema
        // over the surviving prefix would answer a question about a file that
        // does not exist.
        return Err(CliError::InvalidInput(format!(
            "apr dataset audio-inspect: {} is truncated — its `data` chunk declares {declared} \
             bytes but only {} are present",
            path.display(),
            data.len()
        )));
    }
    let codec = codec_name(&fmt, path)?;
    let frame_bytes = frame_bytes(&fmt, path)?;
    let frames = frame_count(data.len(), frame_bytes, path)?;
    let (min, max) = decode_extrema(data, &fmt, path)?;

    Ok(AudioObservation {
        path: path.display().to_string(),
        codec,
        sample_rate: fmt.sample_rate,
        channels: u32::from(fmt.channels),
        samples: frames,
        duration_secs: frames as f64 / f64::from(fmt.sample_rate),
        min,
        max,
        bits_per_sample: fmt.bits_per_sample,
    })
}

/// Name the container we were actually handed instead of reporting "no fmt chunk".
fn reject_known_non_wav(bytes: &[u8], path: &Path) -> Result<()> {
    let named = |what: &str| {
        Err(CliError::InvalidInput(format!(
            "apr dataset audio-inspect: {} is {what}; this decoder reads uncompressed \
             RIFF/WAVE only (PCM u8/i16/i24/i32, IEEE float32)",
            path.display()
        )))
    };
    match bytes {
        b if b.starts_with(b"fLaC") => named("a FLAC stream"),
        b if b.starts_with(b"OggS") => named("an Ogg stream"),
        b if b.starts_with(b"ID3") || b.starts_with(&[0xFF, 0xFB]) => named("an MP3 stream"),
        b if b.len() >= 12 && b.starts_with(b"RIFF") && &b[8..12] == b"WAVE" => Ok(()),
        b if b.starts_with(b"RIFF") => named("a RIFF file that is not WAVE"),
        _ => named("not a RIFF/WAVE file (bad magic)"),
    }
}

/// Locate a top-level RIFF chunk by 4-byte id.
///
/// Returns the payload actually present in the file AND the size the header
/// declared, so callers can tell a complete chunk from a truncated one.
fn find_chunk<'a>(bytes: &'a [u8], id: &[u8; 4]) -> Option<(&'a [u8], usize)> {
    let mut pos = 12usize; // past "RIFF" <size> "WAVE"
    while pos + 8 <= bytes.len() {
        let this_id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        let start = pos + 8;
        let end = start.checked_add(size)?.min(bytes.len());
        if this_id == id {
            return Some((&bytes[start..end], size));
        }
        // Chunks are word-aligned: an odd size carries one pad byte.
        pos = start + size + (size & 1);
    }
    None
}

fn parse_fmt(bytes: &[u8], path: &Path) -> Result<WavFmt> {
    let (raw, _declared) = find_chunk(bytes, b"fmt ").ok_or_else(|| {
        CliError::InvalidInput(format!(
            "apr dataset audio-inspect: {} has no `fmt ` chunk",
            path.display()
        ))
    })?;
    if raw.len() < 16 {
        return Err(CliError::InvalidInput(format!(
            "apr dataset audio-inspect: {} has a truncated `fmt ` chunk ({} bytes, need >= 16)",
            path.display(),
            raw.len()
        )));
    }
    let u16at = |i: usize| u16::from_le_bytes([raw[i], raw[i + 1]]);
    let mut fmt = WavFmt {
        format_tag: u16at(0),
        channels: u16at(2),
        sample_rate: u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]),
        block_align: u16at(12),
        bits_per_sample: u16at(14),
    };
    if fmt.format_tag == FMT_EXTENSIBLE {
        // WAVE_FORMAT_EXTENSIBLE: the real tag is the first 2 bytes of the
        // 16-byte SubFormat GUID at offset 24.
        if raw.len() < 26 {
            return Err(CliError::InvalidInput(format!(
                "apr dataset audio-inspect: {} is WAVE_FORMAT_EXTENSIBLE but its `fmt ` chunk \
                 is too short ({} bytes) to carry a SubFormat GUID",
                path.display(),
                raw.len()
            )));
        }
        fmt.format_tag = u16at(24);
    }
    validate_fmt(&fmt, path)?;
    Ok(fmt)
}

fn validate_fmt(fmt: &WavFmt, path: &Path) -> Result<()> {
    let bad = |what: String| {
        Err(CliError::InvalidInput(format!(
            "{what} in {}",
            path.display()
        )))
    };
    if fmt.channels == 0 {
        return bad("apr dataset audio-inspect: `fmt ` declares 0 channels".to_string());
    }
    if fmt.sample_rate == 0 {
        return bad("apr dataset audio-inspect: `fmt ` declares a 0 Hz sample rate".to_string());
    }
    if fmt.bits_per_sample == 0 || fmt.bits_per_sample % 8 != 0 {
        return bad(format!(
            "apr dataset audio-inspect: unsupported bit depth {} (must be a positive multiple of 8)",
            fmt.bits_per_sample
        ));
    }
    Ok(())
}

fn codec_name(fmt: &WavFmt, path: &Path) -> Result<String> {
    match (fmt.format_tag, fmt.bits_per_sample) {
        (FMT_PCM, 8) => Ok("pcm_u8".to_string()),
        (FMT_PCM, 16) => Ok("pcm_s16le".to_string()),
        (FMT_PCM, 24) => Ok("pcm_s24le".to_string()),
        (FMT_PCM, 32) => Ok("pcm_s32le".to_string()),
        (FMT_IEEE_FLOAT, 32) => Ok("pcm_f32le".to_string()),
        (tag, bits) => Err(CliError::InvalidInput(format!(
            "apr dataset audio-inspect: {} carries format tag {tag} at {bits} bits, which this \
             decoder cannot decode (supported: PCM 8/16/24/32-bit, IEEE float 32-bit)",
            path.display()
        ))),
    }
}

fn frame_bytes(fmt: &WavFmt, path: &Path) -> Result<usize> {
    let computed = usize::from(fmt.channels) * usize::from(fmt.bits_per_sample / 8);
    if fmt.block_align != 0 && usize::from(fmt.block_align) != computed {
        return Err(CliError::InvalidInput(format!(
            "apr dataset audio-inspect: {} declares block_align {} but {} channels x {} bits \
             needs {computed}; this decoder reads only tightly-packed frames",
            path.display(),
            fmt.block_align,
            fmt.channels,
            fmt.bits_per_sample
        )));
    }
    Ok(computed)
}

fn frame_count(data_len: usize, frame_bytes: usize, path: &Path) -> Result<u64> {
    if data_len % frame_bytes != 0 {
        return Err(CliError::InvalidInput(format!(
            "apr dataset audio-inspect: {} has a truncated final frame ({data_len} data bytes is \
             not a multiple of the {frame_bytes}-byte frame)",
            path.display()
        )));
    }
    let frames = (data_len / frame_bytes) as u64;
    if frames == 0 {
        // Refusing beats inventing: min/max of an empty stream do not exist.
        return Err(CliError::ValidationFailed(format!(
            "apr dataset audio-inspect: {} decodes to 0 frames, so it has no amplitude to report",
            path.display()
        )));
    }
    Ok(frames)
}

/// Decode every sample and return the measured (min, max).
fn decode_extrema(data: &[u8], fmt: &WavFmt, path: &Path) -> Result<(f64, f64)> {
    let width = usize::from(fmt.bits_per_sample / 8);
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for raw in data.chunks_exact(width) {
        let v = decode_sample(raw, fmt.format_tag, fmt.bits_per_sample);
        if !v.is_finite() {
            return Err(CliError::ValidationFailed(format!(
                "apr dataset audio-inspect: {} contains a non-finite sample ({v})",
                path.display()
            )));
        }
        min = min.min(v);
        max = max.max(v);
    }
    Ok((min, max))
}

/// Decode one stored sample to the torchaudio-normalised amplitude domain.
fn decode_sample(raw: &[u8], format_tag: u16, bits: u16) -> f64 {
    match (format_tag, bits) {
        (FMT_PCM, 8) => (f64::from(raw[0]) - 128.0) / 128.0,
        (FMT_PCM, 16) => f64::from(i16::from_le_bytes([raw[0], raw[1]])) / 32_768.0,
        (FMT_PCM, 24) => {
            // Sign-extend 24 bits into an i32 by placing them in the high bytes.
            let v = i32::from_le_bytes([0, raw[0], raw[1], raw[2]]) >> 8;
            f64::from(v) / 8_388_608.0
        }
        (FMT_PCM, 32) => {
            f64::from(i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])) / 2_147_483_648.0
        }
        // Only (FMT_IEEE_FLOAT, 32) reaches here; `codec_name` refuses the rest.
        _ => f64::from(f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])),
    }
}

#[cfg(test)]
#[path = "audio_inspect_tests.rs"]
mod tests;
