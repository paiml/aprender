//! Video probing via ffprobe.
//!
//! Extracts video metadata (codec, resolution, fps, duration) by
//! shelling out to ffprobe with JSON output.

use super::types::VideoProbe;
use crate::result::ProbarError;
use std::path::Path;

/// Build ffprobe command arguments for JSON output.
#[must_use]
pub fn build_ffprobe_args(video_path: &Path) -> Vec<String> {
    vec![
        "-v".to_string(),
        "quiet".to_string(),
        "-print_format".to_string(),
        "json".to_string(),
        "-show_format".to_string(),
        "-show_streams".to_string(),
        video_path.to_string_lossy().to_string(),
    ]
}

/// Probe a video file and extract metadata.
///
/// # Errors
///
/// Returns `ProbarError::FfmpegError` if ffprobe is not found or fails.
pub fn probe_video(video_path: &Path) -> Result<VideoProbe, ProbarError> {
    let args = build_ffprobe_args(video_path);

    let output = std::process::Command::new("ffprobe")
        .args(&args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .map_err(|e| ProbarError::FfmpegError {
            message: format!("Failed to execute ffprobe: {e}"),
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ProbarError::FfmpegError {
            message: format!("ffprobe exited with {}: {stderr}", output.status),
        });
    }

    let json_str = String::from_utf8_lossy(&output.stdout);
    parse_ffprobe_json(&json_str)
}

/// Parse ffprobe JSON output into a `VideoProbe`.
pub fn parse_ffprobe_json(json: &str) -> Result<VideoProbe, ProbarError> {
    let parsed: serde_json::Value =
        serde_json::from_str(json).map_err(|e| ProbarError::FfmpegError {
            message: format!("Failed to parse ffprobe JSON: {e}"),
        })?;

    let streams = parsed
        .get("streams")
        .and_then(|s| s.as_array())
        .ok_or_else(|| ProbarError::FfmpegError {
            message: "ffprobe output missing 'streams' array".to_string(),
        })?;

    // Find video stream
    let video_stream = streams
        .iter()
        .find(|s| s.get("codec_type").and_then(|t| t.as_str()) == Some("video"))
        .ok_or_else(|| ProbarError::FfmpegError {
            message: "No video stream found".to_string(),
        })?;

    // Find audio stream (optional)
    let audio_stream = streams
        .iter()
        .find(|s| s.get("codec_type").and_then(|t| t.as_str()) == Some("audio"));

    let codec = video_stream
        .get("codec_name")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let width = video_stream
        .get("width")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;

    let height = video_stream
        .get("height")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;

    let fps_fraction = video_stream
        .get("r_frame_rate")
        .and_then(|v| v.as_str())
        .unwrap_or("0/1")
        .to_string();

    let fps = parse_fps_fraction(&fps_fraction);

    let duration_secs = video_stream
        .get("duration")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<f64>().ok())
        .or_else(|| {
            parsed
                .get("format")
                .and_then(|f| f.get("duration"))
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<f64>().ok())
        })
        .unwrap_or(0.0);

    let bitrate_bps = parsed
        .get("format")
        .and_then(|f| f.get("bit_rate"))
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);

    let pixel_format = video_stream
        .get("pix_fmt")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let audio_codec = audio_stream
        .and_then(|s| s.get("codec_name"))
        .and_then(|v| v.as_str())
        .map(String::from);

    let audio_sample_rate = audio_stream
        .and_then(|s| s.get("sample_rate"))
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<u32>().ok());

    let audio_channels = audio_stream
        .and_then(|s| s.get("channels"))
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);

    Ok(VideoProbe {
        codec,
        width,
        height,
        fps_fraction,
        fps,
        duration_secs,
        bitrate_bps,
        pixel_format,
        audio_codec,
        audio_sample_rate,
        audio_channels,
    })
}

/// Parse an FPS fraction string like "24/1" or "30000/1001" into a float.
fn parse_fps_fraction(fraction: &str) -> f64 {
    let parts: Vec<&str> = fraction.split('/').collect();
    if parts.len() == 2 {
        let num: f64 = parts[0].parse().unwrap_or(0.0);
        let den: f64 = parts[1].parse().unwrap_or(1.0);
        if den > 0.0 {
            return num / den;
        }
    }
    fraction.parse().unwrap_or(0.0)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_build_ffprobe_args() {
        let args = build_ffprobe_args(Path::new("/tmp/video.mp4"));
        assert_eq!(args[0], "-v");
        assert_eq!(args[1], "quiet");
        assert_eq!(args[2], "-print_format");
        assert_eq!(args[3], "json");
        assert_eq!(args[4], "-show_format");
        assert_eq!(args[5], "-show_streams");
        assert_eq!(args[6], "/tmp/video.mp4");
    }

    #[test]
    fn test_build_ffprobe_args_length() {
        let args = build_ffprobe_args(Path::new("test.mp4"));
        assert_eq!(args.len(), 7);
    }

    #[test]
    fn test_parse_fps_fraction_integer() {
        assert!((parse_fps_fraction("24/1") - 24.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_fps_fraction_ntsc() {
        assert!((parse_fps_fraction("30000/1001") - 29.97).abs() < 0.01);
    }

    #[test]
    fn test_parse_fps_fraction_bare_number() {
        assert!((parse_fps_fraction("25") - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_fps_fraction_zero_denominator() {
        assert!(parse_fps_fraction("24/0") < 0.01);
    }

    #[test]
    fn test_parse_fps_fraction_invalid() {
        assert!(parse_fps_fraction("invalid") < 0.01);
    }

    #[test]
    fn test_parse_ffprobe_json_complete() {
        let json = r#"{
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "24/1",
                    "duration": "120.5",
                    "pix_fmt": "yuv420p"
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate": "48000",
                    "channels": 2
                }
            ],
            "format": {
                "duration": "120.5",
                "bit_rate": "5000000"
            }
        }"#;

        let probe = parse_ffprobe_json(json).unwrap();
        assert_eq!(probe.codec, "h264");
        assert_eq!(probe.width, 1920);
        assert_eq!(probe.height, 1080);
        assert!((probe.fps - 24.0).abs() < 0.01);
        assert!((probe.duration_secs - 120.5).abs() < 0.01);
        assert_eq!(probe.bitrate_bps, 5_000_000);
        assert_eq!(probe.pixel_format, "yuv420p");
        assert_eq!(probe.audio_codec.as_deref(), Some("aac"));
        assert_eq!(probe.audio_sample_rate, Some(48000));
        assert_eq!(probe.audio_channels, Some(2));
    }

    #[test]
    fn test_parse_ffprobe_json_no_audio() {
        let json = r#"{
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1280,
                    "height": 720,
                    "r_frame_rate": "30/1",
                    "duration": "60.0",
                    "pix_fmt": "yuv420p"
                }
            ],
            "format": {
                "duration": "60.0",
                "bit_rate": "2000000"
            }
        }"#;

        let probe = parse_ffprobe_json(json).unwrap();
        assert!(probe.audio_codec.is_none());
        assert!(probe.audio_sample_rate.is_none());
        assert!(probe.audio_channels.is_none());
    }

    #[test]
    fn test_parse_ffprobe_json_no_streams() {
        let json = r#"{"format": {}}"#;
        let result = parse_ffprobe_json(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_ffprobe_json_no_video_stream() {
        let json = r#"{
            "streams": [
                {"codec_type": "audio", "codec_name": "aac"}
            ],
            "format": {}
        }"#;
        let result = parse_ffprobe_json(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_ffprobe_json_invalid() {
        let result = parse_ffprobe_json("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_ffprobe_json_duration_from_format() {
        // When stream duration is missing, fall back to format duration
        let json = r#"{
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 640,
                    "height": 480,
                    "r_frame_rate": "24/1",
                    "pix_fmt": "yuv420p"
                }
            ],
            "format": {
                "duration": "90.0"
            }
        }"#;
        let probe = parse_ffprobe_json(json).unwrap();
        assert!((probe.duration_secs - 90.0).abs() < 0.01);
    }

    #[test]
    fn test_probe_video_missing_file() {
        let result = probe_video(Path::new("/nonexistent/video.mp4"));
        assert!(result.is_err());
    }
}
