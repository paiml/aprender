//! Audio level analysis: peak, RMS, dynamic range.

use super::types::AudioLevels;

/// Compute audio levels from PCM samples.
///
/// Calculates peak amplitude, RMS level, and dynamic range.
pub fn analyze_levels(samples: &[f32]) -> AudioLevels {
    if samples.is_empty() {
        return AudioLevels {
            peak: 0.0,
            peak_dbfs: -120.0,
            rms: 0.0,
            rms_dbfs: -120.0,
            dynamic_range_db: 0.0,
            passed: false,
        };
    }

    let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);

    let sum_sq: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    let rms = (sum_sq / samples.len() as f64).sqrt();

    let peak_dbfs = amplitude_to_dbfs(f64::from(peak));
    let rms_dbfs = amplitude_to_dbfs(rms);

    // Estimate noise floor from quietest 10% of windowed RMS values
    let noise_floor_db = estimate_noise_floor(samples);
    let dynamic_range_db = peak_dbfs - noise_floor_db;

    AudioLevels {
        peak: f64::from(peak),
        peak_dbfs,
        rms,
        rms_dbfs,
        dynamic_range_db,
        passed: true, // caller sets based on config
    }
}

/// Check if levels pass the configured thresholds.
#[must_use]
pub fn check_levels(levels: &AudioLevels, min_rms_dbfs: f64, max_peak_dbfs: f64) -> bool {
    levels.rms_dbfs >= min_rms_dbfs && levels.peak_dbfs <= max_peak_dbfs
}

/// Convert linear amplitude to dBFS.
fn amplitude_to_dbfs(amplitude: f64) -> f64 {
    if amplitude <= 0.0 {
        -120.0
    } else {
        20.0 * amplitude.log10()
    }
}

/// Estimate noise floor by finding the quietest windowed RMS values.
fn estimate_noise_floor(samples: &[f32]) -> f64 {
    let window_size = 4800; // 100ms at 48kHz
    if samples.len() < window_size {
        return -120.0;
    }

    let mut rms_values: Vec<f64> = samples
        .chunks(window_size)
        .filter(|chunk| chunk.len() == window_size)
        .map(|chunk| {
            let sum_sq: f64 = chunk.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
            (sum_sq / chunk.len() as f64).sqrt()
        })
        .collect();

    rms_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Take the 10th percentile as noise floor
    let idx = rms_values.len() / 10;
    let noise_rms = rms_values.get(idx).copied().unwrap_or(0.0);
    amplitude_to_dbfs(noise_rms)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_levels_silence() {
        let silence = vec![0.0f32; 48000];
        let levels = analyze_levels(&silence);
        assert!(levels.peak < f64::EPSILON);
        assert_eq!(levels.peak_dbfs, -120.0);
        assert_eq!(levels.rms_dbfs, -120.0);
    }

    #[test]
    fn test_analyze_levels_unity() {
        let unity = vec![1.0f32; 48000];
        let levels = analyze_levels(&unity);
        assert!((levels.peak - 1.0).abs() < f64::EPSILON);
        assert!(levels.peak_dbfs.abs() < 0.01); // 0 dBFS
        assert!(levels.rms_dbfs.abs() < 0.01);
    }

    #[test]
    fn test_analyze_levels_half() {
        let half = vec![0.5f32; 48000];
        let levels = analyze_levels(&half);
        assert!((levels.peak - 0.5).abs() < f64::EPSILON);
        assert!((levels.peak_dbfs - (-6.02)).abs() < 0.1);
    }

    #[test]
    fn test_analyze_levels_empty() {
        let levels = analyze_levels(&[]);
        assert!(!levels.passed);
        assert_eq!(levels.peak_dbfs, -120.0);
    }

    #[test]
    fn test_analyze_levels_negative() {
        let signal = vec![-0.8f32; 48000];
        let levels = analyze_levels(&signal);
        // f32→f64 conversion introduces small rounding error
        assert!((levels.peak - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_amplitude_to_dbfs_zero() {
        assert_eq!(amplitude_to_dbfs(0.0), -120.0);
    }

    #[test]
    fn test_amplitude_to_dbfs_negative() {
        assert_eq!(amplitude_to_dbfs(-1.0), -120.0);
    }

    #[test]
    fn test_amplitude_to_dbfs_unity() {
        assert!(amplitude_to_dbfs(1.0).abs() < 0.01);
    }

    #[test]
    fn test_check_levels_pass() {
        let levels = AudioLevels {
            peak: 0.5,
            peak_dbfs: -6.0,
            rms: 0.2,
            rms_dbfs: -14.0,
            dynamic_range_db: 40.0,
            passed: true,
        };
        assert!(check_levels(&levels, -40.0, -0.1));
    }

    #[test]
    fn test_check_levels_too_quiet() {
        let levels = AudioLevels {
            peak: 0.01,
            peak_dbfs: -40.0,
            rms: 0.001,
            rms_dbfs: -60.0,
            dynamic_range_db: 20.0,
            passed: true,
        };
        assert!(!check_levels(&levels, -40.0, -0.1));
    }

    #[test]
    fn test_check_levels_too_hot() {
        let levels = AudioLevels {
            peak: 1.0,
            peak_dbfs: 0.0,
            rms: 0.5,
            rms_dbfs: -6.0,
            dynamic_range_db: 40.0,
            passed: true,
        };
        assert!(!check_levels(&levels, -40.0, -0.1));
    }

    #[test]
    fn test_dynamic_range_with_signal() {
        // Signal with both loud and quiet portions
        let mut samples = vec![0.001f32; 48000]; // quiet
        samples.extend(vec![0.8f32; 48000]); // loud
        let levels = analyze_levels(&samples);
        assert!(levels.dynamic_range_db > 0.0);
    }

    #[test]
    fn test_noise_floor_estimation() {
        let samples = vec![0.001f32; 48000];
        let floor = estimate_noise_floor(&samples);
        assert!(floor < -40.0);
    }
}
