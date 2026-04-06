//! Audio extraction from video files via ffmpeg.
//!
//! Shells out to ffmpeg to extract mono f32 PCM audio from video containers.
//! ffmpeg is a runtime dependency (already required by rmedia).

use crate::result::ProbarError;
use std::path::Path;

/// Default sample rate for extraction (matches AAC standard).
pub const DEFAULT_SAMPLE_RATE: u32 = 48000;

/// Build the ffmpeg command for audio extraction.
///
/// Returns the command arguments as a vector of strings.
#[must_use]
pub fn build_ffmpeg_args(video_path: &Path, sample_rate: u32) -> Vec<String> {
    vec![
        "-i".to_string(),
        video_path.to_string_lossy().to_string(),
        "-f".to_string(),
        "f32le".to_string(),
        "-acodec".to_string(),
        "pcm_f32le".to_string(),
        "-ac".to_string(),
        "1".to_string(),
        "-ar".to_string(),
        sample_rate.to_string(),
        "pipe:1".to_string(),
    ]
}

/// Extract audio from a video file as mono f32 PCM samples.
///
/// Shells out to ffmpeg and captures stdout as raw PCM data.
///
/// # Errors
///
/// Returns `ProbarError::FfmpegError` if ffmpeg is not found or fails.
pub fn extract_audio(video_path: &Path, sample_rate: u32) -> Result<Vec<f32>, ProbarError> {
    let args = build_ffmpeg_args(video_path, sample_rate);

    let output = std::process::Command::new("ffmpeg")
        .args(&args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .map_err(|e| ProbarError::FfmpegError {
            message: format!("Failed to execute ffmpeg: {e}"),
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ProbarError::FfmpegError {
            message: format!("ffmpeg exited with {}: {}", output.status, stderr),
        });
    }

    // Convert raw bytes to f32 samples (little-endian)
    let bytes = &output.stdout;
    if bytes.len() % 4 != 0 {
        return Err(ProbarError::FfmpegError {
            message: format!(
                "ffmpeg output length {} is not a multiple of 4 bytes",
                bytes.len()
            ),
        });
    }

    let samples: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| {
            let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
            f32::from_le_bytes(arr)
        })
        .collect();

    Ok(samples)
}

/// Derive the default EDL path from a video path.
///
/// Convention: `video.mp4` -> `video.edl.json`
#[must_use]
pub fn default_edl_path(video_path: &Path) -> std::path::PathBuf {
    let stem = video_path.file_stem().unwrap_or_default();
    let edl_name = format!("{}.edl.json", stem.to_string_lossy());
    match video_path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.join(edl_name),
        _ => std::path::PathBuf::from(edl_name),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_build_ffmpeg_args() {
        let path = Path::new("/tmp/video.mp4");
        let args = build_ffmpeg_args(path, 48000);
        assert_eq!(args[0], "-i");
        assert_eq!(args[1], "/tmp/video.mp4");
        assert_eq!(args[2], "-f");
        assert_eq!(args[3], "f32le");
        assert_eq!(args[4], "-acodec");
        assert_eq!(args[5], "pcm_f32le");
        assert_eq!(args[6], "-ac");
        assert_eq!(args[7], "1");
        assert_eq!(args[8], "-ar");
        assert_eq!(args[9], "48000");
        assert_eq!(args[10], "pipe:1");
    }

    #[test]
    fn test_build_ffmpeg_args_custom_rate() {
        let path = Path::new("test.mp4");
        let args = build_ffmpeg_args(path, 44100);
        assert_eq!(args[9], "44100");
    }

    #[test]
    fn test_build_ffmpeg_args_length() {
        let path = Path::new("test.mp4");
        let args = build_ffmpeg_args(path, 48000);
        assert_eq!(args.len(), 11);
    }

    #[test]
    fn test_default_edl_path_mp4() {
        let video = PathBuf::from("/output/demo-bench.mp4");
        let edl = default_edl_path(&video);
        assert_eq!(edl, PathBuf::from("/output/demo-bench.edl.json"));
    }

    #[test]
    fn test_default_edl_path_mov() {
        let video = PathBuf::from("/output/render.mov");
        let edl = default_edl_path(&video);
        assert_eq!(edl, PathBuf::from("/output/render.edl.json"));
    }

    #[test]
    fn test_default_edl_path_no_parent() {
        let video = PathBuf::from("video.mp4");
        let edl = default_edl_path(&video);
        assert_eq!(edl, PathBuf::from("video.edl.json"));
    }

    #[test]
    fn test_default_edl_path_nested() {
        let video = PathBuf::from("/a/b/c/test.mp4");
        let edl = default_edl_path(&video);
        assert_eq!(edl, PathBuf::from("/a/b/c/test.edl.json"));
    }

    #[test]
    fn test_default_sample_rate() {
        assert_eq!(DEFAULT_SAMPLE_RATE, 48000);
    }

    #[test]
    fn test_extract_audio_missing_ffmpeg() {
        // This test verifies error handling when ffmpeg is at a nonexistent path
        let result = extract_audio(Path::new("/nonexistent/video.mp4"), 48000);
        assert!(result.is_err());
    }
}
