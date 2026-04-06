//! Types for video quality verification.
//!
//! Provides structures for video probe results, quality expectations,
//! and verification reports.

use serde::{Deserialize, Serialize};

/// Video probe result from ffprobe.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VideoProbe {
    /// Video codec name (e.g., "h264", "hevc", "prores")
    pub codec: String,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Frame rate as a fraction (e.g., "24/1", "30000/1001")
    pub fps_fraction: String,
    /// Frame rate as a float
    pub fps: f64,
    /// Duration in seconds
    pub duration_secs: f64,
    /// Bitrate in bits per second (0 if unavailable)
    pub bitrate_bps: u64,
    /// Pixel format (e.g., "yuv420p")
    pub pixel_format: String,
    /// Audio codec (None if no audio stream)
    pub audio_codec: Option<String>,
    /// Audio sample rate (None if no audio stream)
    pub audio_sample_rate: Option<u32>,
    /// Audio channels (None if no audio stream)
    pub audio_channels: Option<u32>,
}

/// Expected video properties for validation.
#[derive(Clone, Debug)]
pub struct VideoExpectations {
    /// Expected width (None = skip check)
    pub width: Option<u32>,
    /// Expected height (None = skip check)
    pub height: Option<u32>,
    /// Expected FPS (None = skip check)
    pub fps: Option<f64>,
    /// Expected codec (None = skip check)
    pub codec: Option<String>,
    /// Minimum duration in seconds (None = skip check)
    pub min_duration_secs: Option<f64>,
    /// Maximum duration in seconds (None = skip check)
    pub max_duration_secs: Option<f64>,
    /// Whether audio stream must be present
    pub require_audio: bool,
    /// FPS tolerance for comparison (default: 0.01)
    pub fps_tolerance: f64,
}

impl Default for VideoExpectations {
    fn default() -> Self {
        Self {
            width: None,
            height: None,
            fps: None,
            codec: None,
            min_duration_secs: None,
            max_duration_secs: None,
            require_audio: false,
            fps_tolerance: 0.01,
        }
    }
}

impl VideoExpectations {
    /// Set expected resolution.
    #[must_use]
    pub const fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    /// Set expected FPS.
    #[must_use]
    pub fn with_fps(mut self, fps: f64) -> Self {
        self.fps = Some(fps);
        self
    }

    /// Set expected codec.
    #[must_use]
    pub fn with_codec(mut self, codec: impl Into<String>) -> Self {
        self.codec = Some(codec.into());
        self
    }

    /// Set minimum duration.
    #[must_use]
    pub fn with_min_duration(mut self, secs: f64) -> Self {
        self.min_duration_secs = Some(secs);
        self
    }

    /// Set maximum duration.
    #[must_use]
    pub fn with_max_duration(mut self, secs: f64) -> Self {
        self.max_duration_secs = Some(secs);
        self
    }

    /// Require audio stream.
    #[must_use]
    pub const fn with_require_audio(mut self, require: bool) -> Self {
        self.require_audio = require;
        self
    }
}

/// Video quality verification report.
#[derive(Clone, Debug, Serialize)]
pub struct VideoQualityReport {
    /// Source file path
    pub source: String,
    /// Overall verdict
    pub verdict: VideoVerdict,
    /// Probe results
    pub probe: VideoProbe,
    /// Individual check results
    pub checks: Vec<VideoCheck>,
    /// Number of passed checks
    pub passed_count: usize,
    /// Total number of checks
    pub total_count: usize,
}

/// Individual video quality check result.
#[derive(Clone, Debug, Serialize)]
pub struct VideoCheck {
    /// Check name
    pub name: String,
    /// Expected value
    pub expected: String,
    /// Actual value
    pub actual: String,
    /// Whether the check passed
    pub passed: bool,
}

/// Overall video quality verdict.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VideoVerdict {
    /// All checks passed
    Pass,
    /// One or more checks failed
    Fail,
    /// Could not probe video
    ProbeError,
}

impl std::fmt::Display for VideoVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pass => write!(f, "PASS"),
            Self::Fail => write!(f, "FAIL"),
            Self::ProbeError => write!(f, "PROBE ERROR"),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn sample_probe() -> VideoProbe {
        VideoProbe {
            codec: "h264".to_string(),
            width: 1920,
            height: 1080,
            fps_fraction: "24/1".to_string(),
            fps: 24.0,
            duration_secs: 120.0,
            bitrate_bps: 5_000_000,
            pixel_format: "yuv420p".to_string(),
            audio_codec: Some("aac".to_string()),
            audio_sample_rate: Some(48000),
            audio_channels: Some(2),
        }
    }

    #[test]
    fn test_video_verdict_display() {
        assert_eq!(VideoVerdict::Pass.to_string(), "PASS");
        assert_eq!(VideoVerdict::Fail.to_string(), "FAIL");
        assert_eq!(VideoVerdict::ProbeError.to_string(), "PROBE ERROR");
    }

    #[test]
    fn test_video_verdict_equality() {
        assert_eq!(VideoVerdict::Pass, VideoVerdict::Pass);
        assert_ne!(VideoVerdict::Pass, VideoVerdict::Fail);
    }

    #[test]
    fn test_expectations_default() {
        let exp = VideoExpectations::default();
        assert!(exp.width.is_none());
        assert!(exp.height.is_none());
        assert!(exp.fps.is_none());
        assert!(!exp.require_audio);
    }

    #[test]
    fn test_expectations_builders() {
        let exp = VideoExpectations::default()
            .with_resolution(1920, 1080)
            .with_fps(24.0)
            .with_codec("h264")
            .with_min_duration(10.0)
            .with_max_duration(300.0)
            .with_require_audio(true);
        assert_eq!(exp.width, Some(1920));
        assert_eq!(exp.height, Some(1080));
        assert!((exp.fps.unwrap() - 24.0).abs() < f64::EPSILON);
        assert_eq!(exp.codec.as_deref(), Some("h264"));
        assert!((exp.min_duration_secs.unwrap() - 10.0).abs() < f64::EPSILON);
        assert!((exp.max_duration_secs.unwrap() - 300.0).abs() < f64::EPSILON);
        assert!(exp.require_audio);
    }

    #[test]
    fn test_probe_serialization() {
        let probe = sample_probe();
        let json = serde_json::to_string(&probe).unwrap();
        assert!(json.contains("\"codec\":\"h264\""));
        assert!(json.contains("\"width\":1920"));
    }

    #[test]
    fn test_probe_deserialization() {
        let json = r#"{
            "codec": "h264",
            "width": 1920,
            "height": 1080,
            "fps_fraction": "24/1",
            "fps": 24.0,
            "duration_secs": 120.0,
            "bitrate_bps": 5000000,
            "pixel_format": "yuv420p",
            "audio_codec": "aac",
            "audio_sample_rate": 48000,
            "audio_channels": 2
        }"#;
        let probe: VideoProbe = serde_json::from_str(json).unwrap();
        assert_eq!(probe.codec, "h264");
        assert_eq!(probe.width, 1920);
    }

    #[test]
    fn test_video_check() {
        let check = VideoCheck {
            name: "resolution".to_string(),
            expected: "1920x1080".to_string(),
            actual: "1920x1080".to_string(),
            passed: true,
        };
        assert!(check.passed);
    }

    #[test]
    fn test_video_quality_report_serialization() {
        let report = VideoQualityReport {
            source: "test.mp4".to_string(),
            verdict: VideoVerdict::Pass,
            probe: sample_probe(),
            checks: vec![VideoCheck {
                name: "codec".to_string(),
                expected: "h264".to_string(),
                actual: "h264".to_string(),
                passed: true,
            }],
            passed_count: 1,
            total_count: 1,
        };
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"verdict\":\"Pass\""));
    }
}
