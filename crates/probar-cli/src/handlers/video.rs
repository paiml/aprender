//! Video quality command handler.
//!
//! Orchestrates: probe video -> validate against expectations -> render report.

use crate::commands::{OutputFormat, VideoCheckArgs};
use crate::config::CliConfig;
use crate::error::{CliError, CliResult};
use jugar_probar::video_quality::{probe_video, validate_video, VideoExpectations, VideoVerdict};

/// Execute the video check command for a single file.
pub fn execute_check(config: &CliConfig, args: &VideoCheckArgs) -> CliResult<()> {
    let video_path = &args.video;
    if !video_path.exists() {
        return Err(CliError::invalid_argument(format!(
            "Video file not found: {}",
            video_path.display()
        )));
    }

    if config.verbosity.is_verbose() {
        println!("Probing video: {}", video_path.display());
    }

    let probe = probe_video(video_path)
        .map_err(|e| CliError::test_execution(format!("Video probe failed: {e}")))?;

    let mut expectations = VideoExpectations::default();
    if let (Some(w), Some(h)) = (args.width, args.height) {
        expectations = expectations.with_resolution(w, h);
    } else if let Some(w) = args.width {
        expectations.width = Some(w);
    } else if let Some(h) = args.height {
        expectations.height = Some(h);
    }
    if let Some(fps) = args.fps {
        expectations = expectations.with_fps(fps);
    }
    if let Some(ref codec) = args.codec {
        expectations = expectations.with_codec(codec);
    }
    if let Some(min) = args.min_duration {
        expectations = expectations.with_min_duration(min);
    }
    if let Some(max) = args.max_duration {
        expectations = expectations.with_max_duration(max);
    }
    if args.require_audio {
        expectations = expectations.with_require_audio(true);
    }

    let report = validate_video(&probe, &expectations, &video_path.display().to_string());

    match args.format {
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&report)
                .map_err(|e| CliError::test_execution(format!("JSON serialization failed: {e}")))?;
            println!("{json}");
        }
        OutputFormat::Text => {
            render_text_report(&report);
        }
    }

    if report.verdict == VideoVerdict::Pass {
        Ok(())
    } else {
        Err(CliError::test_execution(format!(
            "Video quality check failed: {}",
            video_path.display()
        )))
    }
}

fn render_text_report(report: &jugar_probar::video_quality::VideoQualityReport) {
    println!("Video Quality: {} ({})", report.source, report.verdict);
    println!(
        "  Codec: {}  Resolution: {}x{}  FPS: {:.2}",
        report.probe.codec, report.probe.width, report.probe.height, report.probe.fps
    );
    println!(
        "  Duration: {:.1}s  Bitrate: {} bps  Pixel format: {}",
        report.probe.duration_secs, report.probe.bitrate_bps, report.probe.pixel_format
    );
    if let Some(ref ac) = report.probe.audio_codec {
        println!(
            "  Audio: {} @ {}Hz ({}ch)",
            ac,
            report.probe.audio_sample_rate.unwrap_or(0),
            report.probe.audio_channels.unwrap_or(0)
        );
    } else {
        println!("  Audio: none");
    }
    if !report.checks.is_empty() {
        println!("  Checks:");
        for check in &report.checks {
            println!(
                "    {}: expected={} actual={}  {}",
                check.name,
                check.expected,
                check.actual,
                if check.passed { "PASS" } else { "FAIL" }
            );
        }
    }
    println!(
        "Verdict: {} ({}/{} checks passed)",
        report.verdict, report.passed_count, report.total_count
    );
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use jugar_probar::video_quality::{VideoCheck, VideoProbe, VideoQualityReport, VideoVerdict};

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

    fn sample_report() -> VideoQualityReport {
        VideoQualityReport {
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
        }
    }

    #[test]
    fn test_render_text_report_pass() {
        let report = sample_report();
        render_text_report(&report);
    }

    #[test]
    fn test_render_text_report_no_audio() {
        let mut report = sample_report();
        report.probe.audio_codec = None;
        report.probe.audio_sample_rate = None;
        report.probe.audio_channels = None;
        render_text_report(&report);
    }

    #[test]
    fn test_render_text_report_fail() {
        let mut report = sample_report();
        report.verdict = VideoVerdict::Fail;
        report.checks.push(VideoCheck {
            name: "resolution".to_string(),
            expected: "3840x2160".to_string(),
            actual: "1920x1080".to_string(),
            passed: false,
        });
        report.passed_count = 1;
        report.total_count = 2;
        render_text_report(&report);
    }

    #[test]
    fn test_execute_check_missing_file() {
        let config = CliConfig::new();
        let args = VideoCheckArgs {
            video: std::path::PathBuf::from("/nonexistent/video.mp4"),
            width: None,
            height: None,
            fps: None,
            codec: None,
            min_duration: None,
            max_duration: None,
            require_audio: false,
            format: OutputFormat::Text,
        };
        let result = execute_check(&config, &args);
        assert!(result.is_err());
    }
}
