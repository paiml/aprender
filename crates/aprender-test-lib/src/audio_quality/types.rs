//! Types for audio quality verification.
//!
//! Provides report structures for audio level analysis, clipping detection,
//! and silence verification.

use serde::{Deserialize, Serialize};

/// Audio quality report for a rendered video.
#[derive(Clone, Debug, Serialize)]
pub struct AudioQualityReport {
    /// Source file path
    pub source: String,
    /// Overall verdict
    pub verdict: AudioVerdict,
    /// Audio level metrics
    pub levels: AudioLevels,
    /// Clipping analysis
    pub clipping: ClippingReport,
    /// Silence analysis
    pub silence: SilenceReport,
    /// Duration in seconds
    pub duration_secs: f64,
    /// Sample rate
    pub sample_rate: u32,
    /// Total sample count
    pub sample_count: usize,
}

/// Audio level metrics.
#[derive(Clone, Debug, Serialize)]
pub struct AudioLevels {
    /// Peak amplitude (0.0-1.0)
    pub peak: f64,
    /// Peak amplitude in dBFS
    pub peak_dbfs: f64,
    /// RMS amplitude (0.0-1.0)
    pub rms: f64,
    /// RMS amplitude in dBFS
    pub rms_dbfs: f64,
    /// Dynamic range in dB (peak - noise floor)
    pub dynamic_range_db: f64,
    /// Whether levels are within acceptable range
    pub passed: bool,
}

/// Clipping detection results.
#[derive(Clone, Debug, Serialize)]
pub struct ClippingReport {
    /// Number of clipped samples (at +/- 1.0)
    pub clipped_samples: usize,
    /// Percentage of clipped samples
    pub clipped_pct: f64,
    /// Whether clipping test passed (no clipping)
    pub passed: bool,
}

/// Silence detection results.
#[derive(Clone, Debug, Serialize)]
pub struct SilenceReport {
    /// Detected silence regions
    pub regions: Vec<SilenceRegion>,
    /// Total silence duration in seconds
    pub total_silence_secs: f64,
    /// Percentage of audio that is silence
    pub silence_pct: f64,
    /// Whether silence check passed
    pub passed: bool,
}

/// A region of silence in the audio.
#[derive(Clone, Debug, Serialize)]
pub struct SilenceRegion {
    /// Start time in seconds
    pub start_secs: f64,
    /// End time in seconds
    pub end_secs: f64,
    /// Duration in seconds
    pub duration_secs: f64,
}

/// Overall audio quality verdict.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioVerdict {
    /// All checks passed
    Pass,
    /// One or more checks failed
    Fail,
    /// No audio found
    NoAudio,
}

impl std::fmt::Display for AudioVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pass => write!(f, "PASS"),
            Self::Fail => write!(f, "FAIL"),
            Self::NoAudio => write!(f, "NO AUDIO"),
        }
    }
}

/// Configuration for audio quality checks.
#[derive(Clone, Debug)]
pub struct AudioQualityConfig {
    /// Minimum acceptable RMS in dBFS (default: -40.0)
    pub min_rms_dbfs: f64,
    /// Maximum acceptable peak in dBFS (default: -0.1, just below clipping)
    pub max_peak_dbfs: f64,
    /// Whether to fail on any clipping (default: true)
    pub no_clipping: bool,
    /// Silence threshold in dBFS (default: -60.0)
    pub silence_threshold_dbfs: f64,
    /// Minimum silence duration to report in seconds (default: 0.5)
    pub min_silence_duration_secs: f64,
    /// Maximum acceptable silence percentage (default: 80.0)
    pub max_silence_pct: f64,
}

impl Default for AudioQualityConfig {
    fn default() -> Self {
        Self {
            min_rms_dbfs: -40.0,
            max_peak_dbfs: -0.1,
            no_clipping: true,
            silence_threshold_dbfs: -60.0,
            min_silence_duration_secs: 0.5,
            max_silence_pct: 80.0,
        }
    }
}

impl AudioQualityConfig {
    /// Set minimum RMS level.
    #[must_use]
    pub fn with_min_rms_dbfs(mut self, dbfs: f64) -> Self {
        self.min_rms_dbfs = dbfs;
        self
    }

    /// Set maximum peak level.
    #[must_use]
    pub fn with_max_peak_dbfs(mut self, dbfs: f64) -> Self {
        self.max_peak_dbfs = dbfs;
        self
    }

    /// Set clipping policy.
    #[must_use]
    pub const fn with_no_clipping(mut self, no_clipping: bool) -> Self {
        self.no_clipping = no_clipping;
        self
    }

    /// Set silence threshold.
    #[must_use]
    pub fn with_silence_threshold_dbfs(mut self, dbfs: f64) -> Self {
        self.silence_threshold_dbfs = dbfs;
        self
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_verdict_display() {
        assert_eq!(AudioVerdict::Pass.to_string(), "PASS");
        assert_eq!(AudioVerdict::Fail.to_string(), "FAIL");
        assert_eq!(AudioVerdict::NoAudio.to_string(), "NO AUDIO");
    }

    #[test]
    fn test_audio_verdict_equality() {
        assert_eq!(AudioVerdict::Pass, AudioVerdict::Pass);
        assert_ne!(AudioVerdict::Pass, AudioVerdict::Fail);
    }

    #[test]
    fn test_config_defaults() {
        let config = AudioQualityConfig::default();
        assert!((config.min_rms_dbfs - (-40.0)).abs() < f64::EPSILON);
        assert!((config.max_peak_dbfs - (-0.1)).abs() < f64::EPSILON);
        assert!(config.no_clipping);
        assert!((config.silence_threshold_dbfs - (-60.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_builders() {
        let config = AudioQualityConfig::default()
            .with_min_rms_dbfs(-30.0)
            .with_max_peak_dbfs(-1.0)
            .with_no_clipping(false)
            .with_silence_threshold_dbfs(-50.0);
        assert!((config.min_rms_dbfs - (-30.0)).abs() < f64::EPSILON);
        assert!((config.max_peak_dbfs - (-1.0)).abs() < f64::EPSILON);
        assert!(!config.no_clipping);
        assert!((config.silence_threshold_dbfs - (-50.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_silence_region() {
        let region = SilenceRegion {
            start_secs: 1.0,
            end_secs: 2.5,
            duration_secs: 1.5,
        };
        assert!((region.duration_secs - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_audio_levels_serialization() {
        let levels = AudioLevels {
            peak: 0.95,
            peak_dbfs: -0.45,
            rms: 0.3,
            rms_dbfs: -10.46,
            dynamic_range_db: 50.0,
            passed: true,
        };
        let json = serde_json::to_string(&levels).unwrap();
        assert!(json.contains("\"peak\":0.95"));
    }

    #[test]
    fn test_clipping_report() {
        let report = ClippingReport {
            clipped_samples: 0,
            clipped_pct: 0.0,
            passed: true,
        };
        assert!(report.passed);
    }

    #[test]
    fn test_silence_report() {
        let report = SilenceReport {
            regions: vec![],
            total_silence_secs: 0.0,
            silence_pct: 0.0,
            passed: true,
        };
        assert!(report.passed);
        assert!(report.regions.is_empty());
    }

    #[test]
    fn test_audio_quality_report_serialization() {
        let report = AudioQualityReport {
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
        };
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"verdict\":\"Pass\""));
    }
}
