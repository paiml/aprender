//! Audio Quality Verification: levels, clipping, silence analysis.
//!
//! Analyzes extracted audio to verify:
//! - Peak and RMS levels are within acceptable range
//! - No digital clipping (samples at +/- 1.0)
//! - Silence regions are identified and within acceptable limits
//!
//! # Usage
//!
//! ```text
//! Video ──→ av_sync::extract_audio ──→ analyze_audio
//!                                           │
//!                                    AudioQualityReport
//! ```

pub mod clipping;
pub mod levels;
pub mod silence;
pub mod types;

pub use clipping::detect_clipping;
pub use levels::{analyze_levels, check_levels};
pub use silence::{check_silence, detect_silence};
pub use types::{
    AudioLevels, AudioQualityConfig, AudioQualityReport, AudioVerdict, ClippingReport,
    SilenceRegion, SilenceReport,
};

use crate::av_sync::extract_audio;
use crate::result::ProbarError;
use std::path::Path;

/// Run complete audio quality analysis on a video file.
///
/// Extracts audio and runs all quality checks.
///
/// # Errors
///
/// Returns `ProbarError::FfmpegError` if audio extraction fails.
pub fn analyze_audio(
    video_path: &Path,
    config: &AudioQualityConfig,
    sample_rate: u32,
) -> Result<AudioQualityReport, ProbarError> {
    let samples = extract_audio(video_path, sample_rate)?;
    Ok(analyze_samples(&samples, video_path, config, sample_rate))
}

/// Run audio quality analysis on already-extracted PCM samples.
#[must_use]
pub fn analyze_samples(
    samples: &[f32],
    source_path: &Path,
    config: &AudioQualityConfig,
    sample_rate: u32,
) -> AudioQualityReport {
    if samples.is_empty() {
        return AudioQualityReport {
            source: source_path.display().to_string(),
            verdict: AudioVerdict::NoAudio,
            levels: AudioLevels {
                peak: 0.0,
                peak_dbfs: -120.0,
                rms: 0.0,
                rms_dbfs: -120.0,
                dynamic_range_db: 0.0,
                passed: false,
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
            duration_secs: 0.0,
            sample_rate,
            sample_count: 0,
        };
    }

    let mut audio_levels = analyze_levels(samples);
    audio_levels.passed = check_levels(&audio_levels, config.min_rms_dbfs, config.max_peak_dbfs);

    let mut clip_report = detect_clipping(samples);
    if !config.no_clipping {
        clip_report.passed = true; // skip clipping check if not required
    }

    let mut silence_report = detect_silence(
        samples,
        sample_rate,
        config.silence_threshold_dbfs,
        config.min_silence_duration_secs,
    );
    silence_report.passed = check_silence(&silence_report, config.max_silence_pct);

    #[allow(clippy::cast_precision_loss)]
    let duration_secs = samples.len() as f64 / f64::from(sample_rate);

    let verdict = if audio_levels.passed && clip_report.passed && silence_report.passed {
        AudioVerdict::Pass
    } else {
        AudioVerdict::Fail
    };

    AudioQualityReport {
        source: source_path.display().to_string(),
        verdict,
        levels: audio_levels,
        clipping: clip_report,
        silence: silence_report,
        duration_secs,
        sample_rate,
        sample_count: samples.len(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_samples_empty() {
        let config = AudioQualityConfig::default();
        let report = analyze_samples(&[], Path::new("test.mp4"), &config, 48000);
        assert_eq!(report.verdict, AudioVerdict::NoAudio);
    }

    #[test]
    fn test_analyze_samples_clean_signal() {
        let config = AudioQualityConfig::default();
        let samples = vec![0.3f32; 48000]; // 1s of clean signal
        let report = analyze_samples(&samples, Path::new("test.mp4"), &config, 48000);
        assert_eq!(report.verdict, AudioVerdict::Pass);
        assert!(report.levels.passed);
        assert!(report.clipping.passed);
        assert_eq!(report.sample_count, 48000);
        assert!((report.duration_secs - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_analyze_samples_clipped_signal() {
        let config = AudioQualityConfig::default();
        let mut samples = vec![0.3f32; 48000];
        samples[1000] = 1.0; // introduce clipping
        let report = analyze_samples(&samples, Path::new("test.mp4"), &config, 48000);
        assert_eq!(report.verdict, AudioVerdict::Fail);
        assert!(!report.clipping.passed);
    }

    #[test]
    fn test_analyze_samples_clipping_allowed() {
        let config = AudioQualityConfig::default().with_no_clipping(false);
        let mut samples = vec![0.3f32; 48000];
        samples[1000] = 1.0;
        let report = analyze_samples(&samples, Path::new("test.mp4"), &config, 48000);
        assert!(report.clipping.passed); // clipping check disabled
    }

    #[test]
    fn test_analyze_samples_too_quiet() {
        let config = AudioQualityConfig::default().with_min_rms_dbfs(-20.0);
        let samples = vec![0.01f32; 48000]; // very quiet
        let report = analyze_samples(&samples, Path::new("test.mp4"), &config, 48000);
        assert_eq!(report.verdict, AudioVerdict::Fail);
        assert!(!report.levels.passed);
    }

    #[test]
    fn test_analyze_samples_source_path() {
        let config = AudioQualityConfig::default();
        let samples = vec![0.3f32; 48000];
        let report = analyze_samples(&samples, Path::new("/output/demo.mp4"), &config, 48000);
        assert_eq!(report.source, "/output/demo.mp4");
    }

    #[test]
    fn test_analyze_samples_sample_rate() {
        let config = AudioQualityConfig::default();
        let samples = vec![0.3f32; 44100];
        let report = analyze_samples(&samples, Path::new("test.mp4"), &config, 44100);
        assert_eq!(report.sample_rate, 44100);
        assert!((report.duration_secs - 1.0).abs() < f64::EPSILON);
    }
}
