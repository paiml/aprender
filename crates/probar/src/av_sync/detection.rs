//! Audio onset detection via RMS energy analysis.
//!
//! Detects percussive audio events (ticks/clicks) in PCM audio streams
//! using sliding-window RMS energy thresholding.

use super::types::AudioOnset;

/// Configuration for onset detection.
#[derive(Clone, Debug)]
pub struct DetectionConfig {
    /// Sample rate in Hz (e.g., 48000)
    pub sample_rate: u32,
    /// RMS window size in milliseconds (default: 10ms)
    pub window_ms: f64,
    /// Energy threshold in dB (default: -40.0)
    pub threshold_db: f64,
    /// Minimum gap between onsets in milliseconds (default: 200ms)
    pub min_gap_ms: f64,
    /// Look-back for onset refinement in milliseconds (default: 5ms)
    pub refine_lookback_ms: f64,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            window_ms: 10.0,
            threshold_db: -40.0,
            min_gap_ms: 200.0,
            refine_lookback_ms: 5.0,
        }
    }
}

impl DetectionConfig {
    /// Create a new detection config with the given sample rate.
    #[must_use]
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Set the energy threshold in dB.
    #[must_use]
    pub fn with_threshold_db(mut self, threshold_db: f64) -> Self {
        self.threshold_db = threshold_db;
        self
    }

    /// Set the minimum gap between onsets.
    #[must_use]
    pub fn with_min_gap_ms(mut self, min_gap_ms: f64) -> Self {
        self.min_gap_ms = min_gap_ms;
        self
    }

    /// Window size in samples.
    fn window_samples(&self) -> usize {
        ((self.window_ms / 1000.0) * f64::from(self.sample_rate)) as usize
    }

    /// Minimum gap in samples.
    fn min_gap_samples(&self) -> usize {
        ((self.min_gap_ms / 1000.0) * f64::from(self.sample_rate)) as usize
    }

    /// Lookback size in samples.
    fn lookback_samples(&self) -> usize {
        ((self.refine_lookback_ms / 1000.0) * f64::from(self.sample_rate)) as usize
    }
}

/// Compute RMS energy for a window of samples.
fn rms_energy(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (sum_sq / samples.len() as f64).sqrt()
}

/// Convert RMS to decibels.
fn rms_to_db(rms: f64) -> f64 {
    if rms <= 0.0 {
        return -120.0; // floor
    }
    20.0 * rms.log10()
}

/// Detect audio onsets in PCM samples.
///
/// Uses RMS energy windowing with threshold crossing detection.
/// Returns onsets sorted by time.
pub fn detect_onsets(samples: &[f32], config: &DetectionConfig) -> Vec<AudioOnset> {
    let window_size = config.window_samples();
    let min_gap = config.min_gap_samples();
    let lookback = config.lookback_samples();

    if samples.len() < window_size || window_size == 0 {
        return Vec::new();
    }

    let mut onsets = Vec::new();
    let mut last_onset_sample: Option<usize> = None;
    let mut was_below = true;

    // Slide window across the signal
    let step = window_size / 2; // 50% overlap
    let step = if step == 0 { 1 } else { step };
    let mut pos = 0;

    while pos + window_size <= samples.len() {
        let window = &samples[pos..pos + window_size];
        let rms = rms_energy(window);
        let db = rms_to_db(rms);

        if db >= config.threshold_db && was_below {
            // Threshold crossing detected
            let onset_sample = if lookback > 0 && pos >= lookback {
                refine_onset(
                    samples,
                    pos,
                    lookback,
                    config.threshold_db,
                    config.sample_rate,
                )
            } else {
                pos
            };

            // Enforce minimum gap
            let gap_ok = match last_onset_sample {
                Some(last) => onset_sample.saturating_sub(last) >= min_gap,
                None => true,
            };

            if gap_ok {
                let time_secs = onset_sample as f64 / f64::from(config.sample_rate);
                onsets.push(AudioOnset {
                    time_secs,
                    energy_db: db,
                    sample_index: onset_sample,
                });
                last_onset_sample = Some(onset_sample);
            }
            was_below = false;
        } else if db < config.threshold_db {
            was_below = true;
        }

        pos += step;
    }

    onsets
}

/// Refine onset position by looking back to find true start.
fn refine_onset(
    samples: &[f32],
    detected_pos: usize,
    lookback: usize,
    threshold_db: f64,
    sample_rate: u32,
) -> usize {
    let start = detected_pos.saturating_sub(lookback);
    let micro_window = (sample_rate as f64 * 0.002) as usize; // 2ms micro windows
    let micro_window = if micro_window == 0 { 1 } else { micro_window };

    let mut earliest = detected_pos;

    let mut pos = start;
    while pos + micro_window <= detected_pos {
        let window = &samples[pos..pos + micro_window];
        let rms = rms_energy(window);
        let db = rms_to_db(rms);
        if db >= threshold_db {
            earliest = pos;
            break;
        }
        pos += micro_window;
    }

    earliest
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Generate a synthetic PCM signal with ticks at known positions.
    fn synthetic_signal(sample_rate: u32, duration_secs: f64, tick_times: &[f64]) -> Vec<f32> {
        let total_samples = (duration_secs * f64::from(sample_rate)) as usize;
        let mut samples = vec![0.0f32; total_samples];
        let tick_duration_samples = (0.02 * f64::from(sample_rate)) as usize; // 20ms tick

        for &tick_time in tick_times {
            let start = (tick_time * f64::from(sample_rate)) as usize;
            for i in 0..tick_duration_samples {
                if start + i < total_samples {
                    // Generate a short burst at 0.5 amplitude
                    let phase =
                        (i as f64 / f64::from(sample_rate)) * 1000.0 * std::f64::consts::TAU;
                    samples[start + i] = (phase.sin() * 0.5) as f32;
                }
            }
        }

        samples
    }

    #[test]
    fn test_rms_energy_silence() {
        let silence = vec![0.0f32; 480];
        let rms = rms_energy(&silence);
        assert!(rms < f64::EPSILON);
    }

    #[test]
    fn test_rms_energy_constant() {
        let signal = vec![0.5f32; 480];
        let rms = rms_energy(&signal);
        assert!((rms - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_rms_energy_empty() {
        let empty: Vec<f32> = vec![];
        let rms = rms_energy(&empty);
        assert!(rms < f64::EPSILON);
    }

    #[test]
    fn test_rms_to_db_unity() {
        let db = rms_to_db(1.0);
        assert!(db.abs() < 0.01); // 0 dB
    }

    #[test]
    fn test_rms_to_db_half() {
        let db = rms_to_db(0.5);
        assert!((db - (-6.02)).abs() < 0.1); // ~-6 dB
    }

    #[test]
    fn test_rms_to_db_zero() {
        let db = rms_to_db(0.0);
        assert_eq!(db, -120.0);
    }

    #[test]
    fn test_rms_to_db_negative() {
        let db = rms_to_db(-1.0);
        assert_eq!(db, -120.0);
    }

    #[test]
    fn test_detect_onsets_empty() {
        let config = DetectionConfig::default();
        let onsets = detect_onsets(&[], &config);
        assert!(onsets.is_empty());
    }

    #[test]
    fn test_detect_onsets_silence() {
        let config = DetectionConfig::default();
        let silence = vec![0.0f32; 48000]; // 1 second of silence
        let onsets = detect_onsets(&silence, &config);
        assert!(onsets.is_empty());
    }

    #[test]
    fn test_detect_onsets_single_tick() {
        let config = DetectionConfig::default();
        let signal = synthetic_signal(48000, 2.0, &[1.0]);
        let onsets = detect_onsets(&signal, &config);
        assert_eq!(onsets.len(), 1, "expected exactly 1 onset");
        // Allow 15ms tolerance for detection precision
        assert!(
            (onsets[0].time_secs - 1.0).abs() < 0.015,
            "onset at {:.3}s, expected ~1.0s",
            onsets[0].time_secs
        );
    }

    #[test]
    fn test_detect_onsets_multiple_ticks() {
        let config = DetectionConfig::default();
        let signal = synthetic_signal(48000, 5.0, &[1.0, 2.0, 3.0]);
        let onsets = detect_onsets(&signal, &config);
        assert_eq!(onsets.len(), 3, "expected 3 onsets, got {}", onsets.len());

        for (i, expected_time) in [1.0, 2.0, 3.0].iter().enumerate() {
            assert!(
                (onsets[i].time_secs - expected_time).abs() < 0.015,
                "onset[{}] at {:.3}s, expected ~{:.1}s",
                i,
                onsets[i].time_secs,
                expected_time
            );
        }
    }

    #[test]
    fn test_detect_onsets_minimum_gap_enforcement() {
        // Two ticks 100ms apart should merge (min_gap=200ms)
        let config = DetectionConfig::default();
        let signal = synthetic_signal(48000, 2.0, &[1.0, 1.1]);
        let onsets = detect_onsets(&signal, &config);
        assert_eq!(
            onsets.len(),
            1,
            "ticks 100ms apart should merge, got {} onsets",
            onsets.len()
        );
    }

    #[test]
    fn test_detect_onsets_respects_threshold() {
        let mut config = DetectionConfig::default();
        config.threshold_db = 0.0; // Very high threshold
        let signal = synthetic_signal(48000, 2.0, &[1.0]);
        let onsets = detect_onsets(&signal, &config);
        assert!(
            onsets.is_empty(),
            "high threshold should reject quiet ticks"
        );
    }

    #[test]
    fn test_detect_onsets_too_short() {
        let config = DetectionConfig::default();
        let short = vec![0.5f32; 10]; // Way too short for a window
        let onsets = detect_onsets(&short, &config);
        assert!(onsets.is_empty());
    }

    #[test]
    fn test_detection_config_default() {
        let config = DetectionConfig::default();
        assert_eq!(config.sample_rate, 48000);
        assert!((config.window_ms - 10.0).abs() < f64::EPSILON);
        assert!((config.threshold_db - (-40.0)).abs() < f64::EPSILON);
        assert!((config.min_gap_ms - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_detection_config_builders() {
        let config = DetectionConfig::default()
            .with_sample_rate(44100)
            .with_threshold_db(-30.0)
            .with_min_gap_ms(100.0);
        assert_eq!(config.sample_rate, 44100);
        assert!((config.threshold_db - (-30.0)).abs() < f64::EPSILON);
        assert!((config.min_gap_ms - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_window_samples_calculation() {
        let config = DetectionConfig::default(); // 48000 Hz, 10ms window
        assert_eq!(config.window_samples(), 480);
    }

    #[test]
    fn test_min_gap_samples_calculation() {
        let config = DetectionConfig::default(); // 48000 Hz, 200ms gap
        assert_eq!(config.min_gap_samples(), 9600);
    }

    #[test]
    fn test_onset_ordering() {
        let config = DetectionConfig::default();
        let signal = synthetic_signal(48000, 5.0, &[3.0, 1.0, 2.0]);
        let onsets = detect_onsets(&signal, &config);
        // Onsets should be in chronological order
        for pair in onsets.windows(2) {
            assert!(pair[0].time_secs <= pair[1].time_secs);
        }
    }
}
