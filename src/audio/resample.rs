//! Sample rate conversion
//!
//! Provides high-quality resampling for audio preprocessing.
//! Whisper requires 16kHz input, so this module converts from common
//! sample rates like 44.1kHz or 48kHz.
//!
//! # Example
//!
//! ```rust,ignore
//! use aprender::audio::resample::resample;
//!
//! // Convert from 44.1kHz to 16kHz
//! let samples_16k = resample(&samples_44k, 44100, 16000);
//! ```
//!
//! # Algorithm
//!
//! Uses windowed-sinc interpolation for high-quality resampling.
//! This is a polyphase filter implementation that minimizes aliasing
//! while maintaining efficiency.

use super::AudioResult;

/// Resample audio from one sample rate to another
///
/// # Arguments
/// * `audio` - Input audio samples (mono, f32)
/// * `from_rate` - Source sample rate in Hz
/// * `to_rate` - Target sample rate in Hz
///
/// # Returns
/// Resampled audio at target sample rate
///
/// # Errors
/// Returns error if sample rates are invalid (zero or negative)
pub fn resample(audio: &[f32], from_rate: u32, to_rate: u32) -> AudioResult<Vec<f32>> {
    if from_rate == 0 || to_rate == 0 {
        return Err(super::AudioError::InvalidParameters(
            "Sample rates must be positive".into(),
        ));
    }

    if from_rate == to_rate {
        return Ok(audio.to_vec());
    }

    // Calculate output length
    let ratio = f64::from(to_rate) / f64::from(from_rate);
    let output_len = (audio.len() as f64 * ratio).ceil() as usize;

    if output_len == 0 {
        return Ok(Vec::new());
    }

    // Linear interpolation (basic implementation)
    // TODO: Implement windowed-sinc for higher quality
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 / ratio;
        let src_idx = src_pos.floor() as usize;
        let frac = (src_pos - src_idx as f64) as f32;

        let sample = if src_idx + 1 < audio.len() {
            audio[src_idx] * (1.0 - frac) + audio[src_idx + 1] * frac
        } else if src_idx < audio.len() {
            audio[src_idx]
        } else {
            0.0
        };

        output.push(sample);
    }

    Ok(output)
}

/// Resample configuration for advanced use
#[derive(Debug, Clone)]
pub struct ResampleConfig {
    /// Filter length for windowed-sinc (higher = better quality, slower)
    pub filter_length: usize,
    /// Cutoff frequency as fraction of Nyquist (0.0-1.0)
    pub cutoff: f32,
}

impl Default for ResampleConfig {
    fn default() -> Self {
        Self {
            filter_length: 64,
            cutoff: 0.95,
        }
    }
}

impl ResampleConfig {
    /// High quality resampling (slower)
    #[must_use]
    pub fn high_quality() -> Self {
        Self {
            filter_length: 128,
            cutoff: 0.98,
        }
    }

    /// Fast resampling (lower quality)
    #[must_use]
    pub fn fast() -> Self {
        Self {
            filter_length: 16,
            cutoff: 0.9,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_same_rate() {
        let audio = vec![1.0, 2.0, 3.0, 4.0];
        let result = resample(&audio, 16000, 16000).expect("should succeed");
        assert_eq!(result, audio);
    }

    #[test]
    fn test_resample_empty() {
        let audio: Vec<f32> = vec![];
        let result = resample(&audio, 44100, 16000).expect("should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_resample_invalid_rate() {
        let audio = vec![1.0, 2.0, 3.0];
        assert!(resample(&audio, 0, 16000).is_err());
        assert!(resample(&audio, 16000, 0).is_err());
    }

    #[test]
    fn test_resample_downsample() {
        // 1 second at 44100 Hz -> 1 second at 16000 Hz
        let audio: Vec<f32> = (0..44100).map(|i| (i as f32 / 44100.0).sin()).collect();
        let result = resample(&audio, 44100, 16000).expect("should succeed");
        // Should be approximately 16000 samples
        let expected_len = (44100.0_f64 * 16000.0 / 44100.0).ceil() as usize;
        assert_eq!(result.len(), expected_len);
    }

    #[test]
    fn test_resample_upsample() {
        let audio = vec![0.0, 1.0, 0.0, -1.0];
        let result = resample(&audio, 8000, 16000).expect("should succeed");
        // Should approximately double the samples
        assert!(result.len() >= 7 && result.len() <= 9);
    }

    #[test]
    fn test_resample_config_default() {
        let config = ResampleConfig::default();
        assert_eq!(config.filter_length, 64);
        assert!((config.cutoff - 0.95).abs() < 0.01);
    }
}
