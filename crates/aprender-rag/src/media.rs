//! Media types for subtitle and transcript support.
//!
//! Provides SRT and VTT subtitle parsing with timestamp preservation.
//! No additional dependencies — uses only `std` and `serde`.
//!
//! # Example
//!
//! ```rust
//! use trueno_rag::media::{parse_subtitles, SubtitleFormat};
//!
//! let srt = "1\n00:00:01,000 --> 00:00:04,500\nHello world.\n";
//! let track = parse_subtitles(srt).unwrap();
//! assert_eq!(track.format, SubtitleFormat::Srt);
//! assert_eq!(track.cues.len(), 1);
//! assert!((track.cues[0].start_secs - 1.0).abs() < 0.01);
//! ```

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Subtitle file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubtitleFormat {
    /// SubRip Text (.srt)
    Srt,
    /// Web Video Text Tracks (.vtt)
    Vtt,
}

impl fmt::Display for SubtitleFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Srt => write!(f, "srt"),
            Self::Vtt => write!(f, "vtt"),
        }
    }
}

/// A single timed text cue from a subtitle file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtitleCue {
    /// Sequence index (0-based internally)
    pub index: usize,
    /// Start time in seconds
    pub start_secs: f64,
    /// End time in seconds
    pub end_secs: f64,
    /// Text content (may contain multiple lines)
    pub text: String,
}

/// Parsed subtitle file.
#[derive(Debug, Clone)]
pub struct SubtitleTrack {
    /// Detected format
    pub format: SubtitleFormat,
    /// Ordered cues
    pub cues: Vec<SubtitleCue>,
}

impl SubtitleTrack {
    /// Total duration based on last cue end time.
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        self.cues.last().map(|c| c.end_secs).unwrap_or(0.0)
    }

    /// Concatenate all cue text into a plain transcript.
    #[must_use]
    pub fn to_plain_text(&self) -> String {
        self.cues.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(" ")
    }

    /// Get cues that overlap a time range.
    #[must_use]
    pub fn cues_in_range(&self, start: f64, end: f64) -> Vec<&SubtitleCue> {
        self.cues.iter().filter(|c| c.end_secs > start && c.start_secs < end).collect()
    }

    /// Serialize to SRT format string.
    #[must_use]
    pub fn to_srt_string(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        for (i, cue) in self.cues.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            let _ = writeln!(out, "{}", i + 1);
            let _ = writeln!(
                out,
                "{} --> {}",
                format_srt_time(cue.start_secs),
                format_srt_time(cue.end_secs),
            );
            out.push_str(&cue.text);
            out.push('\n');
        }
        out
    }
}

/// Format seconds as SRT timestamp `HH:MM:SS,mmm`.
#[allow(clippy::cast_sign_loss)]
fn format_srt_time(secs: f64) -> String {
    let total_ms = (secs.max(0.0) * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_secs = total_ms / 1000;
    let s = total_secs % 60;
    let total_mins = total_secs / 60;
    let m = total_mins % 60;
    let h = total_mins / 60;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

/// Format seconds as display timestamp `MM:SS` or `H:MM:SS`.
#[must_use]
#[allow(clippy::cast_sign_loss)]
pub fn format_display_time(secs: f64) -> String {
    let total_secs = secs.max(0.0).round() as u64;
    let s = total_secs % 60;
    let total_mins = total_secs / 60;
    let m = total_mins % 60;
    let h = total_mins / 60;
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

/// Parse SRT or VTT from a string, auto-detecting format.
pub fn parse_subtitles(input: &str) -> Result<SubtitleTrack> {
    let trimmed = strip_bom(input);
    if trimmed.starts_with("WEBVTT") {
        parse_vtt(trimmed)
    } else {
        parse_srt(trimmed)
    }
}

/// Strip UTF-8 BOM if present.
fn strip_bom(s: &str) -> &str {
    s.strip_prefix('\u{FEFF}').unwrap_or(s)
}

// ── SRT helpers ─────────────────────────────────────────────────

/// Normalize line endings to `\n` and split into non-empty blocks.
fn normalize_and_split(input: &str) -> Vec<String> {
    let normalized = input.replace("\r\n", "\n").replace('\r', "\n");
    normalized.split("\n\n").filter(|b| !b.trim().is_empty()).map(String::from).collect()
}

/// Find the index of the timestamp line (containing "-->") in a set of lines.
fn find_timestamp_index(lines: &[&str]) -> Option<usize> {
    lines.iter().position(|l| l.contains("-->"))
}

/// Find the timestamp index in an SRT block, requiring at least 2 lines.
fn find_srt_timestamp_index(lines: &[&str]) -> Option<usize> {
    find_timestamp_index(lines).filter(|_| lines.len() >= 2)
}

/// Parse the SRT sequence index from lines preceding the timestamp.
fn parse_srt_index(lines: &[&str], ts_idx: usize, fallback: usize) -> usize {
    if ts_idx > 0 {
        lines[0].trim().parse::<usize>().unwrap_or(fallback)
    } else {
        fallback
    }
}

/// Extract cue text from lines after the timestamp.
fn extract_cue_text(lines: &[&str], ts_idx: usize) -> String {
    lines[ts_idx + 1..].join("\n").trim().to_string()
}

/// Build a `SubtitleCue` from parsed SRT components, returning `None` if text is empty.
fn build_srt_cue(index: usize, start: f64, end: f64, text: String) -> Option<SubtitleCue> {
    if text.is_empty() {
        return None;
    }
    Some(SubtitleCue { index: index.saturating_sub(1), start_secs: start, end_secs: end, text })
}

/// Parse a single SRT block into a cue, returning `None` for invalid/empty blocks.
fn parse_srt_block(block: &str, fallback_index: usize) -> Result<Option<SubtitleCue>> {
    let lines: Vec<&str> = block.lines().collect();
    let Some(ts_idx) = find_srt_timestamp_index(&lines) else {
        return Ok(None);
    };

    let index = parse_srt_index(&lines, ts_idx, fallback_index);
    let (start, end) = parse_timestamp_line(lines[ts_idx], ',')?;
    let text = extract_cue_text(&lines, ts_idx);
    Ok(build_srt_cue(index, start, end, text))
}

/// Re-index cues sequentially starting from 0.
fn reindex_cues(cues: &mut [SubtitleCue]) {
    for (i, cue) in cues.iter_mut().enumerate() {
        cue.index = i;
    }
}

/// Parse SRT format.
fn parse_srt(input: &str) -> Result<SubtitleTrack> {
    let blocks = normalize_and_split(input);
    let mut cues = Vec::new();

    for block in &blocks {
        if let Some(cue) = parse_srt_block(block, cues.len())? {
            cues.push(cue);
        }
    }

    if cues.is_empty() {
        return Err(Error::InvalidInput("No valid SRT cues found".into()));
    }

    reindex_cues(&mut cues);

    Ok(SubtitleTrack { format: SubtitleFormat::Srt, cues })
}

// ── VTT helpers ─────────────────────────────────────────────────

/// Extract the VTT body after the WEBVTT header block.
fn vtt_body(normalized: &str) -> &str {
    normalized.split_once("\n\n").map(|x| x.1).unwrap_or("")
}

/// Extract and clean VTT cue text from lines after the timestamp.
fn extract_vtt_cue_text(lines: &[&str], ts_idx: usize) -> String {
    strip_vtt_tags(&lines[ts_idx + 1..].join("\n")).trim().to_string()
}

/// Build a `SubtitleCue` from parsed VTT components, returning `None` if text is empty.
fn build_vtt_cue(index: usize, start: f64, end: f64, text: String) -> Option<SubtitleCue> {
    if text.is_empty() {
        return None;
    }
    Some(SubtitleCue { index, start_secs: start, end_secs: end, text })
}

/// Parse a single VTT block into a cue, returning `None` for invalid/empty blocks.
fn parse_vtt_block(block: &str, index: usize) -> Result<Option<SubtitleCue>> {
    let lines: Vec<&str> = block.lines().collect();
    let Some(ts_idx) = find_timestamp_index(&lines) else {
        return Ok(None);
    };

    let (start, end) = parse_timestamp_line(lines[ts_idx], '.')?;
    let text = extract_vtt_cue_text(&lines, ts_idx);
    Ok(build_vtt_cue(index, start, end, text))
}

/// Parse VTT format.
fn parse_vtt(input: &str) -> Result<SubtitleTrack> {
    let normalized = input.replace("\r\n", "\n").replace('\r', "\n");
    let body = vtt_body(&normalized);
    let mut cues = Vec::new();

    for block in body.split("\n\n").filter(|b| !b.trim().is_empty()) {
        if let Some(cue) = parse_vtt_block(block, cues.len())? {
            cues.push(cue);
        }
    }

    if cues.is_empty() {
        return Err(Error::InvalidInput("No valid VTT cues found".into()));
    }

    Ok(SubtitleTrack { format: SubtitleFormat::Vtt, cues })
}

// ── Timestamp parsing helpers ───────────────────────────────────

/// Split a timestamp line on the `-->` arrow, returning `(start_str, end_half)`.
fn split_arrow(line: &str) -> Result<(&str, &str)> {
    line.split_once("-->")
        .ok_or_else(|| Error::InvalidInput(format!("Invalid timestamp line: {line}")))
}

/// Extract just the timestamp portion from the end half (strips VTT position settings).
fn extract_end_timestamp(end_half: &str) -> &str {
    end_half.split_whitespace().next().unwrap_or("")
}

/// Parse a timestamp line like "00:01:30,500 --> 00:02:00,000" or with dots.
fn parse_timestamp_line(line: &str, ms_sep: char) -> Result<(f64, f64)> {
    let (start_half, end_half) = split_arrow(line)?;
    let start = parse_time(start_half.trim(), ms_sep)?;
    let end = parse_time(extract_end_timestamp(end_half), ms_sep)?;
    Ok((start, end))
}

/// Parse one numeric field from a timestamp string, producing a contextual error.
fn parse_ts_field(field: &str, label: &str, raw: &str) -> Result<f64> {
    field.parse().map_err(|e| Error::InvalidInput(format!("Bad timestamp {label} '{raw}': {e}")))
}

/// Compute seconds from `MM:SS.mmm` parts.
fn secs_from_mm_ss(parts: &[&str], raw: &str) -> Result<f64> {
    let mins = parse_ts_field(parts[0], "minutes", raw)?;
    let secs = parse_ts_field(parts[1], "seconds", raw)?;
    Ok(mins * 60.0 + secs)
}

/// Compute seconds from `HH:MM:SS.mmm` parts.
fn secs_from_hh_mm_ss(parts: &[&str], raw: &str) -> Result<f64> {
    let hrs = parse_ts_field(parts[0], "hours", raw)?;
    let mins = parse_ts_field(parts[1], "minutes", raw)?;
    let secs = parse_ts_field(parts[2], "seconds", raw)?;
    Ok(hrs * 3600.0 + mins * 60.0 + secs)
}

/// Parse a single timestamp to seconds.
/// Accepts `HH:MM:SS,mmm` or `HH:MM:SS.mmm` or `MM:SS.mmm`.
fn parse_time(s: &str, ms_sep: char) -> Result<f64> {
    let normalized = s.replace(ms_sep, ".");
    let parts: Vec<&str> = normalized.split(':').collect();
    match parts.len() {
        2 => secs_from_mm_ss(&parts, s),
        3 => secs_from_hh_mm_ss(&parts, s),
        _ => Err(Error::InvalidInput(format!("Invalid timestamp: {s}"))),
    }
}

// ── VTT tag stripping helpers ───────────────────────────────────

/// Advance tag-tracking state for a single character.
/// Returns `true` if the character should be emitted to output.
fn vtt_tag_filter(ch: char, in_tag: &mut bool) -> bool {
    match ch {
        '<' => {
            *in_tag = true;
            false
        }
        '>' => {
            *in_tag = false;
            false
        }
        _other => !*in_tag,
    }
}

/// Strip VTT formatting tags like `<b>`, `<i>`, `<c.classname>`, etc.
fn strip_vtt_tags(s: &str) -> String {
    let mut in_tag = false;
    s.chars().filter(|&ch| vtt_tag_filter(ch, &mut in_tag)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── SRT parsing ─────────────────────────────────────────

    #[test]
    fn test_parse_srt_basic() {
        let srt = "\
1
00:00:01,000 --> 00:00:04,500
Welcome to this lecture.

2
00:00:05,000 --> 00:00:09,200
Today we cover supervised learning.
";
        let track = parse_subtitles(srt).unwrap();
        assert_eq!(track.format, SubtitleFormat::Srt);
        assert_eq!(track.cues.len(), 2);
        assert_eq!(track.cues[0].index, 0);
        assert!((track.cues[0].start_secs - 1.0).abs() < 0.01);
        assert!((track.cues[0].end_secs - 4.5).abs() < 0.01);
        assert_eq!(track.cues[0].text, "Welcome to this lecture.");
        assert!((track.cues[1].start_secs - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_srt_multiline_text() {
        let srt = "\
1
00:00:01,000 --> 00:00:04,500
Line one of the cue
and line two of the cue.
";
        let track = parse_subtitles(srt).unwrap();
        assert_eq!(track.cues.len(), 1);
        assert_eq!(track.cues[0].text, "Line one of the cue\nand line two of the cue.");
    }

    #[test]
    fn test_parse_srt_with_bom() {
        let srt = "\u{FEFF}1\n00:00:01,000 --> 00:00:04,500\nHello.\n";
        let track = parse_subtitles(srt).unwrap();
        assert_eq!(track.cues.len(), 1);
        assert_eq!(track.cues[0].text, "Hello.");
    }

    #[test]
    fn test_parse_srt_crlf() {
        let srt = "1\r\n00:00:01,000 --> 00:00:04,500\r\nHello.\r\n\r\n2\r\n00:00:05,000 --> 00:00:09,000\r\nWorld.\r\n";
        let track = parse_subtitles(srt).unwrap();
        assert_eq!(track.cues.len(), 2);
    }

    #[test]
    fn test_parse_srt_empty_cue_skipped() {
        let srt = "\
1
00:00:01,000 --> 00:00:04,500


2
00:00:05,000 --> 00:00:09,000
Actual text.
";
        let track = parse_subtitles(srt).unwrap();
        assert_eq!(track.cues.len(), 1);
        assert_eq!(track.cues[0].text, "Actual text.");
    }

    #[test]
    fn test_parse_srt_error_on_empty() {
        let result = parse_subtitles("");
        assert!(result.is_err());
    }

    // ── VTT parsing ─────────────────────────────────────────

    #[test]
    fn test_parse_vtt_basic() {
        let vtt = "\
WEBVTT

00:00:01.000 --> 00:00:04.500
Welcome to this lecture.

00:00:05.000 --> 00:00:09.200
Today we cover supervised learning.
";
        let track = parse_subtitles(vtt).unwrap();
        assert_eq!(track.format, SubtitleFormat::Vtt);
        assert_eq!(track.cues.len(), 2);
        assert!((track.cues[0].start_secs - 1.0).abs() < 0.01);
        assert!((track.cues[0].end_secs - 4.5).abs() < 0.01);
        assert_eq!(track.cues[0].text, "Welcome to this lecture.");
    }

    #[test]
    fn test_parse_vtt_with_cue_ids() {
        let vtt = "\
WEBVTT

intro-1
00:00:01.000 --> 00:00:04.500
Hello world.
";
        let track = parse_subtitles(vtt).unwrap();
        assert_eq!(track.cues.len(), 1);
        assert_eq!(track.cues[0].text, "Hello world.");
    }

    #[test]
    fn test_parse_vtt_with_metadata_header() {
        let vtt = "\
WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:04.500
Hello.
";
        let track = parse_subtitles(vtt).unwrap();
        assert_eq!(track.cues.len(), 1);
    }

    #[test]
    fn test_parse_vtt_strips_tags() {
        let vtt = "\
WEBVTT

00:00:01.000 --> 00:00:04.500
<b>Bold</b> and <i>italic</i> text.
";
        let track = parse_subtitles(vtt).unwrap();
        assert_eq!(track.cues[0].text, "Bold and italic text.");
    }

    #[test]
    fn test_parse_vtt_mm_ss_format() {
        let vtt = "\
WEBVTT

01:30.000 --> 02:00.000
Short timestamp format.
";
        let track = parse_subtitles(vtt).unwrap();
        assert!((track.cues[0].start_secs - 90.0).abs() < 0.01);
        assert!((track.cues[0].end_secs - 120.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_vtt_position_settings() {
        let vtt = "\
WEBVTT

00:00:01.000 --> 00:00:04.500 position:10% align:start
Positioned text.
";
        let track = parse_subtitles(vtt).unwrap();
        assert_eq!(track.cues.len(), 1);
        assert!((track.cues[0].end_secs - 4.5).abs() < 0.01);
    }

    // ── SubtitleTrack methods ───────────────────────────────

    #[test]
    fn test_track_duration() {
        let track = SubtitleTrack {
            format: SubtitleFormat::Srt,
            cues: vec![
                SubtitleCue { index: 0, start_secs: 0.0, end_secs: 5.0, text: "A".into() },
                SubtitleCue { index: 1, start_secs: 5.0, end_secs: 120.5, text: "B".into() },
            ],
        };
        assert!((track.duration_secs() - 120.5).abs() < 0.01);
    }

    #[test]
    fn test_track_duration_empty() {
        let track = SubtitleTrack { format: SubtitleFormat::Srt, cues: vec![] };
        assert!((track.duration_secs()).abs() < 0.01);
    }

    #[test]
    fn test_track_plain_text() {
        let track = SubtitleTrack {
            format: SubtitleFormat::Srt,
            cues: vec![
                SubtitleCue { index: 0, start_secs: 0.0, end_secs: 3.0, text: "Hello".into() },
                SubtitleCue { index: 1, start_secs: 3.0, end_secs: 6.0, text: "world".into() },
            ],
        };
        assert_eq!(track.to_plain_text(), "Hello world");
    }

    #[test]
    fn test_track_cues_in_range() {
        let track = SubtitleTrack {
            format: SubtitleFormat::Srt,
            cues: vec![
                SubtitleCue { index: 0, start_secs: 0.0, end_secs: 5.0, text: "A".into() },
                SubtitleCue { index: 1, start_secs: 5.0, end_secs: 10.0, text: "B".into() },
                SubtitleCue { index: 2, start_secs: 10.0, end_secs: 15.0, text: "C".into() },
            ],
        };
        // Range 4.0-11.0 overlaps A (ends at 5.0 > 4.0), B, and C (starts at 10.0 < 11.0)
        let range = track.cues_in_range(4.0, 11.0);
        assert_eq!(range.len(), 3);
        assert_eq!(range[0].text, "A");
        assert_eq!(range[1].text, "B");
        assert_eq!(range[2].text, "C");

        // Range 6.0-9.0 overlaps only B
        let range2 = track.cues_in_range(6.0, 9.0);
        assert_eq!(range2.len(), 1);
        assert_eq!(range2[0].text, "B");
    }

    #[test]
    fn test_srt_roundtrip() {
        let srt = "\
1
00:00:01,000 --> 00:00:04,500
Hello world.

2
00:01:30,500 --> 00:02:00,000
Second cue here.
";
        let track = parse_subtitles(srt).unwrap();
        let output = track.to_srt_string();
        let reparsed = parse_srt(&output).unwrap();
        assert_eq!(reparsed.cues.len(), track.cues.len());
        for (a, b) in track.cues.iter().zip(reparsed.cues.iter()) {
            assert!((a.start_secs - b.start_secs).abs() < 0.01);
            assert!((a.end_secs - b.end_secs).abs() < 0.01);
            assert_eq!(a.text, b.text);
        }
    }

    // ── Timestamp parsing edge cases ────────────────────────

    #[test]
    fn test_parse_time_zero() {
        let t = parse_time("00:00:00.000", '.').unwrap();
        assert!((t).abs() < 0.001);
    }

    #[test]
    fn test_parse_time_large() {
        let t = parse_time("99:59:59.999", '.').unwrap();
        let expected = 99.0 * 3600.0 + 59.0 * 60.0 + 59.999;
        assert!((t - expected).abs() < 0.01);
    }

    #[test]
    fn test_parse_time_mm_ss() {
        let t = parse_time("01:30.500", '.').unwrap();
        assert!((t - 90.5).abs() < 0.01);
    }

    #[test]
    fn test_parse_time_invalid() {
        assert!(parse_time("invalid", '.').is_err());
        assert!(parse_time("1:2:3:4", '.').is_err());
    }

    // ── Display formatting ──────────────────────────────────

    #[test]
    fn test_format_display_time() {
        assert_eq!(format_display_time(0.0), "0:00");
        assert_eq!(format_display_time(65.0), "1:05");
        assert_eq!(format_display_time(3661.0), "1:01:01");
        assert_eq!(format_display_time(90.4), "1:30");
    }

    #[test]
    fn test_format_srt_time() {
        assert_eq!(format_srt_time(0.0), "00:00:00,000");
        assert_eq!(format_srt_time(90.5), "00:01:30,500");
        assert_eq!(format_srt_time(3661.123), "01:01:01,123");
    }

    #[test]
    fn test_subtitle_format_display() {
        assert_eq!(SubtitleFormat::Srt.to_string(), "srt");
        assert_eq!(SubtitleFormat::Vtt.to_string(), "vtt");
    }

    // ── VTT tag stripping ───────────────────────────────────

    #[test]
    fn test_strip_vtt_tags_none() {
        assert_eq!(strip_vtt_tags("plain text"), "plain text");
    }

    #[test]
    fn test_strip_vtt_tags_bold() {
        assert_eq!(strip_vtt_tags("<b>bold</b>"), "bold");
    }

    #[test]
    fn test_strip_vtt_tags_nested() {
        assert_eq!(strip_vtt_tags("<b><i>text</i></b>"), "text");
    }

    #[test]
    fn test_strip_vtt_tags_class() {
        assert_eq!(strip_vtt_tags("<c.highlight>text</c>"), "text");
    }
}
