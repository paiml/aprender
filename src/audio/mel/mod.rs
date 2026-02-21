//! Mel spectrogram computation
//!
//! Implements mel filterbank for converting audio to mel spectrograms.
//! This is a critical component for ASR, TTS, and voice processing applications.
//!
//! # Algorithm
//!
//! 1. Apply Hann window to audio frames
//! 2. Compute FFT to get power spectrum
//! 3. Apply mel filterbank to convert to mel scale
//! 4. Apply log compression
//!
//! # Example
//!
//! ```rust
//! use aprender::audio::mel::{MelFilterbank, MelConfig};
//!
//! // Create filterbank with Whisper-compatible settings
//! let config = MelConfig::whisper();
//! let filterbank = MelFilterbank::new(&config);
//!
//! // Compute mel spectrogram
//! let audio = vec![0.0f32; 16000]; // 1 second at 16kHz
//! let mel_spec = filterbank.compute(&audio).expect("mel computation should succeed");
//! ```
//!
//! # References
//!
//! - Whisper paper: Radford et al. (2023)
//! - Mel scale: Stevens, Volkmann, & Newman (1937)

use super::{AudioError, AudioResult};
use std::f32::consts::PI;

// ============================================================================
// A11: Audio Clipping Detection
// ============================================================================

/// Result of audio clipping detection
#[derive(Debug, Clone, PartialEq)]
pub struct ClippingReport {
    /// Number of samples that exceed +1.0
    pub positive_clipped: usize,
    /// Number of samples that exceed -1.0
    pub negative_clipped: usize,
    /// Maximum sample value found
    pub max_value: f32,
    /// Minimum sample value found
    pub min_value: f32,
    /// Total number of samples analyzed
    pub total_samples: usize,
    /// Whether clipping was detected
    pub has_clipping: bool,
}

impl ClippingReport {
    /// Percentage of samples that are clipped
    #[must_use]
    pub fn clipping_percentage(&self) -> f32 {
        if self.total_samples == 0 {
            return 0.0;
        }
        let clipped = self.positive_clipped + self.negative_clipped;
        (clipped as f32 / self.total_samples as f32) * 100.0
    }
}

/// Detect audio clipping in samples (A11)
///
/// Audio samples should be normalized to the range [-1.0, 1.0].
/// Samples outside this range indicate clipping or improper normalization.
///
/// # Arguments
/// * `samples` - Audio samples to analyze
///
/// # Returns
/// Report containing clipping statistics
///
/// # Example
///
/// ```rust
/// use aprender::audio::mel::detect_clipping;
///
/// let samples = vec![0.5, 0.8, 1.2, -0.3, -1.5];
/// let report = detect_clipping(&samples);
/// assert!(report.has_clipping);
/// assert_eq!(report.positive_clipped, 1);
/// assert_eq!(report.negative_clipped, 1);
/// ```
#[must_use]
pub fn detect_clipping(samples: &[f32]) -> ClippingReport {
    if samples.is_empty() {
        return ClippingReport {
            positive_clipped: 0,
            negative_clipped: 0,
            max_value: 0.0,
            min_value: 0.0,
            total_samples: 0,
            has_clipping: false,
        };
    }

    let mut positive_clipped = 0_usize;
    let mut negative_clipped = 0_usize;
    let mut max_value = f32::NEG_INFINITY;
    let mut min_value = f32::INFINITY;

    for &sample in samples {
        if sample > max_value {
            max_value = sample;
        }
        if sample < min_value {
            min_value = sample;
        }
        if sample > 1.0 {
            positive_clipped += 1;
        } else if sample < -1.0 {
            negative_clipped += 1;
        }
    }

    let has_clipping = positive_clipped > 0 || negative_clipped > 0;

    ClippingReport {
        positive_clipped,
        negative_clipped,
        max_value,
        min_value,
        total_samples: samples.len(),
        has_clipping,
    }
}

/// Check if any sample contains NaN (A14)
///
/// # Arguments
/// * `samples` - Audio samples to check
///
/// # Returns
/// True if any sample is NaN
#[must_use]
pub fn has_nan(samples: &[f32]) -> bool {
    samples.iter().any(|s| s.is_nan())
}

/// Check if any sample contains Infinity (A15)
///
/// # Arguments
/// * `samples` - Audio samples to check
///
/// # Returns
/// True if any sample is positive or negative infinity
#[must_use]
pub fn has_inf(samples: &[f32]) -> bool {
    samples.iter().any(|s| s.is_infinite())
}

/// Convert stereo audio to mono by averaging channels (A12)
///
/// # Arguments
/// * `stereo` - Interleaved stereo samples [L0, R0, L1, R1, ...]
///
/// # Returns
/// Mono samples where each sample is (left + right) / 2
///
/// # Example
///
/// ```rust
/// use aprender::audio::mel::stereo_to_mono;
///
/// let stereo = vec![0.5, 0.3, 0.6, 0.4];  // 2 stereo frames
/// let mono = stereo_to_mono(&stereo);
/// assert_eq!(mono.len(), 2);
/// assert!((mono[0] - 0.4).abs() < 1e-6);  // (0.5 + 0.3) / 2
/// ```
#[must_use]
pub fn stereo_to_mono(stereo: &[f32]) -> Vec<f32> {
    if stereo.is_empty() {
        return Vec::new();
    }
    stereo
        .chunks(2)
        .map(|chunk| {
            if chunk.len() == 2 {
                (chunk[0] + chunk[1]) / 2.0
            } else {
                chunk[0] // Handle odd-length arrays (last sample is mono)
            }
        })
        .collect()
}

/// Validate audio samples for common issues
///
/// Checks for:
/// - Clipping (samples outside [-1.0, 1.0])
/// - NaN values (A14)
/// - Infinity values (A15)
/// - Empty audio (A13)
///
/// # Arguments
/// * `samples` - Audio samples to validate
///
/// # Returns
/// Ok(()) if audio is valid, Err with description otherwise
pub fn validate_audio(samples: &[f32]) -> AudioResult<()> {
    if samples.is_empty() {
        return Err(AudioError::InvalidParameters(
            "Audio cannot be empty".to_string(),
        ));
    }

    if has_nan(samples) {
        return Err(AudioError::InvalidParameters(
            "Audio contains NaN values".to_string(),
        ));
    }

    if has_inf(samples) {
        return Err(AudioError::InvalidParameters(
            "Audio contains Infinity values".to_string(),
        ));
    }

    let report = detect_clipping(samples);
    if report.has_clipping {
        return Err(AudioError::InvalidParameters(format!(
            "Audio clipping detected: {} samples exceed ±1.0 (max={:.3}, min={:.3}). \
             Normalize audio to [-1.0, 1.0] range.",
            report.positive_clipped + report.negative_clipped,
            report.max_value,
            report.min_value
        )));
    }

    Ok(())
}

/// Configuration for mel spectrogram computation
#[derive(Debug, Clone)]
pub struct MelConfig {
    /// Number of mel channels (typically 80 for Whisper, 128 for some TTS)
    pub n_mels: usize,
    /// FFT size (typically 400 for Whisper at 16kHz, 1024 for TTS)
    pub n_fft: usize,
    /// Hop length between frames (typically 160 for Whisper = 10ms at 16kHz)
    pub hop_length: usize,
    /// Sample rate in Hz (typically 16000 for Whisper, 22050 for TTS)
    pub sample_rate: u32,
    /// Minimum frequency for mel filterbank (Hz)
    pub fmin: f32,
    /// Maximum frequency for mel filterbank (Hz, typically sample_rate/2)
    pub fmax: f32,
    /// Whether to apply center padding (pad n_fft/2 zeros on each side)
    ///
    /// When true (librosa default, used by OpenAI Whisper / HuggingFace):
    ///   n_frames = audio_len / hop_length
    /// When false:
    ///   n_frames = (audio_len - n_fft) / hop_length + 1
    pub center_pad: bool,
}

impl MelConfig {
    /// Create configuration matching OpenAI Whisper
    ///
    /// Parameters: n_mels=80, n_fft=400, hop_length=160, sample_rate=16000, center_pad=true
    #[must_use]
    pub fn whisper() -> Self {
        Self {
            n_mels: 80,
            n_fft: 400,
            hop_length: 160,
            sample_rate: 16000,
            fmin: 0.0,
            fmax: 8000.0,
            center_pad: true,
        }
    }

    /// Create configuration for TTS applications (VITS-style)
    ///
    /// Parameters: n_mels=80, n_fft=1024, hop_length=256, sample_rate=22050
    #[must_use]
    pub fn tts() -> Self {
        Self {
            n_mels: 80,
            n_fft: 1024,
            hop_length: 256,
            sample_rate: 22050,
            fmin: 0.0,
            fmax: 11025.0,
            center_pad: false,
        }
    }

    /// Create custom configuration
    #[must_use]
    pub fn custom(
        n_mels: usize,
        n_fft: usize,
        hop_length: usize,
        sample_rate: u32,
        fmin: f32,
        fmax: f32,
        center_pad: bool,
    ) -> Self {
        Self {
            n_mels,
            n_fft,
            hop_length,
            sample_rate,
            fmin,
            fmax,
            center_pad,
        }
    }

    /// Number of frequency bins (n_fft / 2 + 1)
    #[must_use]
    pub fn n_freqs(&self) -> usize {
        self.n_fft / 2 + 1
    }
}

impl Default for MelConfig {
    fn default() -> Self {
        Self::whisper()
    }
}

/// Mel filterbank for spectrogram computation
///
/// Implements the mel-frequency filterbank used for audio preprocessing.
/// The filterbank converts linear frequency power spectra to mel-scale representations.
#[derive(Debug, Clone)]
pub struct MelFilterbank {
    /// Configuration
    config: MelConfig,
    /// Filterbank matrix (`n_mels` x `n_freqs`) stored in row-major order
    filters: Vec<f32>,
    /// Number of frequency bins (`n_fft` / 2 + 1)
    n_freqs: usize,
    /// Precomputed Hann window
    window: Vec<f32>,
}

include!("filterbank.rs");
