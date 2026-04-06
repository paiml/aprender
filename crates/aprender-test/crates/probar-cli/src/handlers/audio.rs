//! Audio quality command handler.
//!
//! Orchestrates: extract audio -> analyze levels/clipping/silence -> render report.

use crate::commands::{AudioCheckArgs, OutputFormat};
use crate::config::CliConfig;
use crate::error::{CliError, CliResult};
use jugar_probar::audio_quality::{analyze_audio, AudioQualityConfig, AudioVerdict};

/// Execute the audio check command for a single video.
pub fn execute_check(config: &CliConfig, args: &AudioCheckArgs) -> CliResult<()> {
    let video_path = &args.video;
    if !video_path.exists() {
        return Err(CliError::invalid_argument(format!(
            "Video file not found: {}",
            video_path.display()
        )));
    }

    let audio_config = AudioQualityConfig::default()
        .with_min_rms_dbfs(args.min_rms_dbfs)
        .with_max_peak_dbfs(args.max_peak_dbfs)
        .with_no_clipping(args.no_clipping);

    if config.verbosity.is_verbose() {
        println!("Analyzing audio quality: {}", video_path.display());
    }

    let report = analyze_audio(video_path, &audio_config, args.sample_rate)
        .map_err(|e| CliError::test_execution(format!("Audio extraction failed: {e}")))?;

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

    if report.verdict == AudioVerdict::Pass {
        Ok(())
    } else {
        Err(CliError::test_execution(format!(
            "Audio quality check failed: {}",
            video_path.display()
        )))
    }
}

fn render_text_report(report: &jugar_probar::audio_quality::AudioQualityReport) {
    println!("Audio Quality: {} ({})", report.source, report.verdict);
    println!(
        "  Duration: {:.1}s ({} samples @ {}Hz)",
        report.duration_secs, report.sample_count, report.sample_rate
    );
    println!("  Levels:");
    println!(
        "    Peak: {:.1} dBFS  {}",
        report.levels.peak_dbfs,
        if report.levels.passed { "PASS" } else { "FAIL" }
    );
    println!("    RMS:  {:.1} dBFS", report.levels.rms_dbfs);
    println!(
        "    Dynamic range: {:.1} dB",
        report.levels.dynamic_range_db
    );
    println!("  Clipping:");
    println!(
        "    Clipped samples: {} ({:.2}%)  {}",
        report.clipping.clipped_samples,
        report.clipping.clipped_pct,
        if report.clipping.passed {
            "PASS"
        } else {
            "FAIL"
        }
    );
    println!("  Silence:");
    println!(
        "    Total silence: {:.1}s ({:.1}%)  {}",
        report.silence.total_silence_secs,
        report.silence.silence_pct,
        if report.silence.passed {
            "PASS"
        } else {
            "FAIL"
        }
    );
    for region in &report.silence.regions {
        println!(
            "    Region: {:.1}s - {:.1}s ({:.1}s)",
            region.start_secs, region.end_secs, region.duration_secs
        );
    }
    println!("Verdict: {}", report.verdict);
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use jugar_probar::audio_quality::{
        AudioLevels, AudioQualityReport, AudioVerdict, ClippingReport, SilenceRegion, SilenceReport,
    };

    fn sample_report() -> AudioQualityReport {
        AudioQualityReport {
            source: "test.mp4".to_string(),
            verdict: AudioVerdict::Pass,
            levels: AudioLevels {
                peak: 0.5,
                peak_dbfs: -6.0,
                rms: 0.2,
                rms_dbfs: -14.0,
                dynamic_range_db: 40.0,
                passed: true,
            },
            clipping: ClippingReport {
                clipped_samples: 0,
                clipped_pct: 0.0,
                passed: true,
            },
            silence: SilenceReport {
                regions: vec![],
                total_silence_secs: 0.0,
                silence_pct: 0.0,
                passed: true,
            },
            duration_secs: 10.0,
            sample_rate: 48000,
            sample_count: 480_000,
        }
    }

    #[test]
    fn test_render_text_report_pass() {
        let report = sample_report();
        // Should not panic
        render_text_report(&report);
    }

    #[test]
    fn test_render_text_report_with_silence_regions() {
        let mut report = sample_report();
        report.silence.regions = vec![SilenceRegion {
            start_secs: 2.0,
            end_secs: 3.5,
            duration_secs: 1.5,
        }];
        report.silence.total_silence_secs = 1.5;
        report.silence.silence_pct = 15.0;
        render_text_report(&report);
    }

    #[test]
    fn test_render_text_report_fail() {
        let mut report = sample_report();
        report.verdict = AudioVerdict::Fail;
        report.clipping.passed = false;
        report.clipping.clipped_samples = 42;
        report.clipping.clipped_pct = 0.009;
        render_text_report(&report);
    }

    #[test]
    fn test_execute_check_missing_file() {
        let config = CliConfig::new();
        let args = AudioCheckArgs {
            video: std::path::PathBuf::from("/nonexistent/video.mp4"),
            sample_rate: 48000,
            min_rms_dbfs: -40.0,
            max_peak_dbfs: -0.1,
            no_clipping: true,
            format: OutputFormat::Text,
        };
        let result = execute_check(&config, &args);
        assert!(result.is_err());
    }
}
