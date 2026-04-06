//! Silence detection in audio streams.
//!
//! Identifies regions of silence (energy below threshold) and reports
//! their positions and durations.

use super::types::{SilenceRegion, SilenceReport};

/// Detect silence regions in PCM samples.
///
/// Finds contiguous regions where RMS energy stays below the threshold.
/// Only reports regions longer than `min_duration_secs`.
pub fn detect_silence(
    samples: &[f32],
    sample_rate: u32,
    threshold_dbfs: f64,
    min_duration_secs: f64,
) -> SilenceReport {
    if samples.is_empty() {
        return SilenceReport {
            regions: vec![],
            total_silence_secs: 0.0,
            silence_pct: 0.0,
            passed: true,
        };
    }

    let window_size = (f64::from(sample_rate) * 0.01) as usize; // 10ms windows
    let window_size = if window_size == 0 { 1 } else { window_size };
    let threshold_linear = db_to_amplitude(threshold_dbfs);

    let mut regions = Vec::new();
    let mut silence_start: Option<usize> = None;

    let mut pos = 0;
    while pos + window_size <= samples.len() {
        let window = &samples[pos..pos + window_size];
        let rms = window_rms(window);

        if rms < threshold_linear {
            if silence_start.is_none() {
                silence_start = Some(pos);
            }
        } else if let Some(start) = silence_start.take() {
            let duration_samples = pos - start;
            let duration_secs = duration_samples as f64 / f64::from(sample_rate);
            if duration_secs >= min_duration_secs {
                regions.push(SilenceRegion {
                    start_secs: start as f64 / f64::from(sample_rate),
                    end_secs: pos as f64 / f64::from(sample_rate),
                    duration_secs,
                });
            }
        }

        pos += window_size;
    }

    // Handle trailing silence
    if let Some(start) = silence_start {
        let duration_samples = samples.len() - start;
        let duration_secs = duration_samples as f64 / f64::from(sample_rate);
        if duration_secs >= min_duration_secs {
            regions.push(SilenceRegion {
                start_secs: start as f64 / f64::from(sample_rate),
                end_secs: samples.len() as f64 / f64::from(sample_rate),
                duration_secs,
            });
        }
    }

    let total_silence_secs: f64 = regions.iter().map(|r| r.duration_secs).sum();
    let total_duration = samples.len() as f64 / f64::from(sample_rate);
    let silence_pct = if total_duration > 0.0 {
        (total_silence_secs / total_duration) * 100.0
    } else {
        0.0
    };

    SilenceReport {
        regions,
        total_silence_secs,
        silence_pct,
        passed: true, // caller determines pass/fail based on config
    }
}

/// Check if silence percentage exceeds the maximum.
#[must_use]
pub fn check_silence(report: &SilenceReport, max_silence_pct: f64) -> bool {
    report.silence_pct <= max_silence_pct
}

fn window_rms(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (sum_sq / samples.len() as f64).sqrt()
}

fn db_to_amplitude(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_silence_all_silent() {
        let silence = vec![0.0f32; 48000]; // 1 second at 48kHz
        let report = detect_silence(&silence, 48000, -60.0, 0.5);
        assert_eq!(report.regions.len(), 1);
        assert!((report.total_silence_secs - 1.0).abs() < 0.02);
    }

    #[test]
    fn test_detect_silence_no_silence() {
        let signal = vec![0.5f32; 48000];
        let report = detect_silence(&signal, 48000, -60.0, 0.5);
        assert!(report.regions.is_empty());
        assert!(report.total_silence_secs < f64::EPSILON);
    }

    #[test]
    fn test_detect_silence_gap_in_middle() {
        let mut samples = vec![0.5f32; 24000]; // 0.5s signal
        samples.extend(vec![0.0f32; 48000]); // 1.0s silence
        samples.extend(vec![0.5f32; 24000]); // 0.5s signal
        let report = detect_silence(&samples, 48000, -60.0, 0.5);
        assert_eq!(report.regions.len(), 1);
        assert!((report.regions[0].duration_secs - 1.0).abs() < 0.02);
    }

    #[test]
    fn test_detect_silence_short_gap_ignored() {
        let mut samples = vec![0.5f32; 24000];
        samples.extend(vec![0.0f32; 4800]); // 0.1s silence (below 0.5s minimum)
        samples.extend(vec![0.5f32; 24000]);
        let report = detect_silence(&samples, 48000, -60.0, 0.5);
        assert!(report.regions.is_empty());
    }

    #[test]
    fn test_detect_silence_empty() {
        let report = detect_silence(&[], 48000, -60.0, 0.5);
        assert!(report.regions.is_empty());
        assert!(report.passed);
    }

    #[test]
    fn test_detect_silence_trailing() {
        let mut samples = vec![0.5f32; 24000]; // 0.5s signal
        samples.extend(vec![0.0f32; 48000]); // 1.0s trailing silence
        let report = detect_silence(&samples, 48000, -60.0, 0.5);
        assert_eq!(report.regions.len(), 1);
    }

    #[test]
    fn test_silence_percentage() {
        let mut samples = vec![0.0f32; 48000]; // 1s silence
        samples.extend(vec![0.5f32; 48000]); // 1s signal
        let report = detect_silence(&samples, 48000, -60.0, 0.5);
        assert!((report.silence_pct - 50.0).abs() < 2.0);
    }

    #[test]
    fn test_check_silence_pass() {
        let report = SilenceReport {
            regions: vec![],
            total_silence_secs: 1.0,
            silence_pct: 10.0,
            passed: true,
        };
        assert!(check_silence(&report, 80.0));
    }

    #[test]
    fn test_check_silence_fail() {
        let report = SilenceReport {
            regions: vec![],
            total_silence_secs: 9.0,
            silence_pct: 90.0,
            passed: true,
        };
        assert!(!check_silence(&report, 80.0));
    }

    #[test]
    fn test_db_to_amplitude() {
        assert!((db_to_amplitude(0.0) - 1.0).abs() < 0.001);
        assert!((db_to_amplitude(-6.02) - 0.5).abs() < 0.01);
        assert!((db_to_amplitude(-20.0) - 0.1).abs() < 0.01);
    }
}
