//! Video property validation against expectations.
//!
//! Compares probed video metadata against declared expectations
//! and produces a quality report.

use super::types::{VideoCheck, VideoExpectations, VideoProbe, VideoQualityReport, VideoVerdict};

/// Validate video properties against expectations.
///
/// Runs all configured checks and aggregates into a report.
#[must_use]
pub fn validate_video(
    probe: &VideoProbe,
    expectations: &VideoExpectations,
    source: &str,
) -> VideoQualityReport {
    let mut checks = Vec::new();

    if let Some(expected_width) = expectations.width {
        checks.push(VideoCheck {
            name: "width".to_string(),
            expected: expected_width.to_string(),
            actual: probe.width.to_string(),
            passed: probe.width == expected_width,
        });
    }

    if let Some(expected_height) = expectations.height {
        checks.push(VideoCheck {
            name: "height".to_string(),
            expected: expected_height.to_string(),
            actual: probe.height.to_string(),
            passed: probe.height == expected_height,
        });
    }

    if let Some(expected_fps) = expectations.fps {
        let fps_match = (probe.fps - expected_fps).abs() <= expectations.fps_tolerance;
        checks.push(VideoCheck {
            name: "fps".to_string(),
            expected: format!("{expected_fps:.2}"),
            actual: format!("{:.2}", probe.fps),
            passed: fps_match,
        });
    }

    if let Some(ref expected_codec) = expectations.codec {
        checks.push(VideoCheck {
            name: "codec".to_string(),
            expected: expected_codec.clone(),
            actual: probe.codec.clone(),
            passed: probe.codec == *expected_codec,
        });
    }

    if let Some(min_dur) = expectations.min_duration_secs {
        checks.push(VideoCheck {
            name: "min_duration".to_string(),
            expected: format!(">= {min_dur:.1}s"),
            actual: format!("{:.1}s", probe.duration_secs),
            passed: probe.duration_secs >= min_dur,
        });
    }

    if let Some(max_dur) = expectations.max_duration_secs {
        checks.push(VideoCheck {
            name: "max_duration".to_string(),
            expected: format!("<= {max_dur:.1}s"),
            actual: format!("{:.1}s", probe.duration_secs),
            passed: probe.duration_secs <= max_dur,
        });
    }

    if expectations.require_audio {
        checks.push(VideoCheck {
            name: "audio_present".to_string(),
            expected: "yes".to_string(),
            actual: if probe.audio_codec.is_some() {
                "yes".to_string()
            } else {
                "no".to_string()
            },
            passed: probe.audio_codec.is_some(),
        });
    }

    let passed_count = checks.iter().filter(|c| c.passed).count();
    let total_count = checks.len();
    let verdict = if passed_count == total_count {
        VideoVerdict::Pass
    } else {
        VideoVerdict::Fail
    };

    VideoQualityReport {
        source: source.to_string(),
        verdict,
        probe: probe.clone(),
        checks,
        passed_count,
        total_count,
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
    fn test_validate_all_pass() {
        let probe = sample_probe();
        let exp = VideoExpectations::default()
            .with_resolution(1920, 1080)
            .with_fps(24.0)
            .with_codec("h264")
            .with_min_duration(10.0)
            .with_max_duration(300.0)
            .with_require_audio(true);
        let report = validate_video(&probe, &exp, "test.mp4");
        assert_eq!(report.verdict, VideoVerdict::Pass);
        assert_eq!(report.passed_count, report.total_count);
    }

    #[test]
    fn test_validate_wrong_resolution() {
        let probe = sample_probe();
        let exp = VideoExpectations::default().with_resolution(3840, 2160);
        let report = validate_video(&probe, &exp, "test.mp4");
        assert_eq!(report.verdict, VideoVerdict::Fail);
        assert!(report.checks.iter().any(|c| c.name == "width" && !c.passed));
    }

    #[test]
    fn test_validate_wrong_fps() {
        let probe = sample_probe();
        let exp = VideoExpectations::default().with_fps(30.0);
        let report = validate_video(&probe, &exp, "test.mp4");
        assert_eq!(report.verdict, VideoVerdict::Fail);
    }

    #[test]
    fn test_validate_fps_within_tolerance() {
        let mut probe = sample_probe();
        probe.fps = 23.999;
        let exp = VideoExpectations::default().with_fps(24.0);
        let report = validate_video(&probe, &exp, "test.mp4");
        assert_eq!(report.verdict, VideoVerdict::Pass);
    }

    #[test]
    fn test_validate_wrong_codec() {
        let probe = sample_probe();
        let exp = VideoExpectations::default().with_codec("hevc");
        let report = validate_video(&probe, &exp, "test.mp4");
        assert_eq!(report.verdict, VideoVerdict::Fail);
    }

    #[test]
    fn test_validate_too_short() {
        let mut probe = sample_probe();
        probe.duration_secs = 5.0;
        let exp = VideoExpectations::default().with_min_duration(60.0);
        let report = validate_video(&probe, &exp, "test.mp4");
        assert_eq!(report.verdict, VideoVerdict::Fail);
    }

    #[test]
    fn test_validate_too_long() {
        let mut probe = sample_probe();
        probe.duration_secs = 600.0;
        let exp = VideoExpectations::default().with_max_duration(300.0);
        let report = validate_video(&probe, &exp, "test.mp4");
        assert_eq!(report.verdict, VideoVerdict::Fail);
    }

    #[test]
    fn test_validate_missing_audio() {
        let mut probe = sample_probe();
        probe.audio_codec = None;
        let exp = VideoExpectations::default().with_require_audio(true);
        let report = validate_video(&probe, &exp, "test.mp4");
        assert_eq!(report.verdict, VideoVerdict::Fail);
    }

    #[test]
    fn test_validate_no_expectations() {
        let probe = sample_probe();
        let exp = VideoExpectations::default();
        let report = validate_video(&probe, &exp, "test.mp4");
        assert_eq!(report.verdict, VideoVerdict::Pass);
        assert_eq!(report.total_count, 0);
    }

    #[test]
    fn test_validate_source_propagation() {
        let probe = sample_probe();
        let exp = VideoExpectations::default();
        let report = validate_video(&probe, &exp, "/output/demo.mp4");
        assert_eq!(report.source, "/output/demo.mp4");
    }

    #[test]
    fn test_validate_check_details() {
        let probe = sample_probe();
        let exp = VideoExpectations::default().with_codec("h264");
        let report = validate_video(&probe, &exp, "test.mp4");
        let codec_check = report.checks.iter().find(|c| c.name == "codec").unwrap();
        assert_eq!(codec_check.expected, "h264");
        assert_eq!(codec_check.actual, "h264");
        assert!(codec_check.passed);
    }

    #[test]
    fn test_validate_partial_fail() {
        let probe = sample_probe();
        let exp = VideoExpectations::default()
            .with_resolution(1920, 1080) // pass
            .with_fps(30.0); // fail
        let report = validate_video(&probe, &exp, "test.mp4");
        assert_eq!(report.verdict, VideoVerdict::Fail);
        assert_eq!(report.passed_count, 2); // width + height pass
        assert_eq!(report.total_count, 3); // width + height + fps
    }
}
