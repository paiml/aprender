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
//! let mel_spec = filterbank.compute(&audio).unwrap();
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
}

impl MelConfig {
    /// Create configuration matching OpenAI Whisper
    ///
    /// Parameters: n_mels=80, n_fft=400, hop_length=160, sample_rate=16000
    #[must_use]
    pub fn whisper() -> Self {
        Self {
            n_mels: 80,
            n_fft: 400,
            hop_length: 160,
            sample_rate: 16000,
            fmin: 0.0,
            fmax: 8000.0,
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
    ) -> Self {
        Self {
            n_mels,
            n_fft,
            hop_length,
            sample_rate,
            fmin,
            fmax,
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

impl MelFilterbank {
    /// Create a new mel filterbank from configuration
    ///
    /// # Arguments
    /// * `config` - Mel spectrogram configuration
    ///
    /// # Panics
    /// Panics if configuration parameters are zero
    #[must_use]
    pub fn new(config: &MelConfig) -> Self {
        assert!(config.n_mels > 0, "n_mels must be positive");
        assert!(config.n_fft > 0, "n_fft must be positive");
        assert!(config.sample_rate > 0, "sample_rate must be positive");

        let n_freqs = config.n_fft / 2 + 1;

        // Compute mel filterbank matrix
        let filters = Self::compute_filterbank(
            config.n_mels,
            config.n_fft,
            config.sample_rate,
            config.fmin,
            config.fmax,
        );

        // Precompute Hann window
        let window = Self::hann_window(config.n_fft);

        Self {
            config: config.clone(),
            filters,
            n_freqs,
            window,
        }
    }

    /// Create a mel filterbank from pre-computed filter weights
    ///
    /// This is useful when using filterbank weights from a model file
    /// that need to match exactly (e.g., for ASR parity testing).
    ///
    /// # Arguments
    /// * `filters` - Pre-computed filterbank matrix (n_mels x n_freqs) in row-major order
    /// * `config` - Configuration (n_mels, n_fft, etc.)
    ///
    /// # Panics
    /// Panics if filter dimensions don't match configuration
    #[must_use]
    pub fn from_filters(filters: Vec<f32>, config: &MelConfig) -> Self {
        let n_freqs = config.n_fft / 2 + 1;
        assert_eq!(
            filters.len(),
            config.n_mels * n_freqs,
            "filterbank size mismatch: expected {} x {} = {}, got {}",
            config.n_mels,
            n_freqs,
            config.n_mels * n_freqs,
            filters.len()
        );

        let window = Self::hann_window(config.n_fft);

        Self {
            config: config.clone(),
            filters,
            n_freqs,
            window,
        }
    }

    /// Compute the mel filterbank matrix with Slaney area normalization
    ///
    /// Creates triangular filters spaced on the mel scale, normalized so that
    /// each filter has unit area (Slaney normalization). This matches librosa's
    /// `norm='slaney'` and OpenAI Whisper's filterbank.
    ///
    /// # References
    /// - Slaney, M. (1998). Auditory Toolbox. Technical Report #1998-010.
    fn compute_filterbank(
        n_mels: usize,
        n_fft: usize,
        sample_rate: u32,
        fmin: f32,
        fmax: f32,
    ) -> Vec<f32> {
        let n_freqs = n_fft / 2 + 1;
        let mut filters = vec![0.0_f32; n_mels * n_freqs];

        // Convert to mel scale
        let mel_min = Self::hz_to_mel(fmin);
        let mel_max = Self::hz_to_mel(fmax);

        // Create n_mels + 2 points evenly spaced on mel scale
        let mel_points: Vec<f32> = (0..=n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * (i as f32) / ((n_mels + 1) as f32))
            .collect();

        // Convert mel points back to Hz
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| Self::mel_to_hz(m)).collect();

        // Convert Hz to FFT bin indices
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&f| ((n_fft as f32 + 1.0) * f / sample_rate as f32).floor() as usize)
            .collect();

        // Create triangular filters with Slaney area normalization
        for m in 0..n_mels {
            let f_m_minus = bin_points[m];
            let f_m = bin_points[m + 1];
            let f_m_plus = bin_points[m + 2];

            // Slaney normalization factor: 2 / (hz_high - hz_low)
            // This ensures each triangular filter has unit area
            let hz_low = hz_points[m];
            let hz_high = hz_points[m + 2];
            let bandwidth = hz_high - hz_low;
            let slaney_norm = if bandwidth > 0.0 {
                2.0 / bandwidth
            } else {
                1.0
            };

            // Rising slope
            for k in f_m_minus..f_m {
                if k < n_freqs && f_m > f_m_minus {
                    let slope = (k - f_m_minus) as f32 / (f_m - f_m_minus) as f32;
                    filters[m * n_freqs + k] = slope * slaney_norm;
                }
            }

            // Falling slope
            for k in f_m..f_m_plus {
                if k < n_freqs && f_m_plus > f_m {
                    let slope = (f_m_plus - k) as f32 / (f_m_plus - f_m) as f32;
                    filters[m * n_freqs + k] = slope * slaney_norm;
                }
            }
        }

        filters
    }

    /// Convert frequency in Hz to mel scale
    ///
    /// Uses the formula: mel = 2595 * log10(1 + f/700)
    #[inline]
    #[must_use]
    pub fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    /// Convert mel scale to frequency in Hz
    ///
    /// Uses the formula: f = 700 * (10^(mel/2595) - 1)
    #[inline]
    #[must_use]
    pub fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    /// Compute Hann window
    fn hann_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / size as f32).cos()))
            .collect()
    }

    /// Compute mel spectrogram from audio samples
    ///
    /// # Arguments
    /// * `audio` - Audio samples (mono, f32, at target sample rate)
    ///
    /// # Returns
    /// Mel spectrogram as a flattened 2D matrix (n_frames x n_mels) in row-major order
    ///
    /// # Errors
    /// Returns error if audio processing fails
    pub fn compute(&self, audio: &[f32]) -> AudioResult<Vec<f32>> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        let hop_length = self.config.hop_length;
        if hop_length == 0 {
            return Err(AudioError::InvalidParameters(
                "hop_length must be positive".into(),
            ));
        }

        // Calculate number of frames
        let n_frames = if audio.len() >= self.config.n_fft {
            (audio.len() - self.config.n_fft) / hop_length + 1
        } else {
            0
        };

        if n_frames == 0 {
            return Ok(Vec::new());
        }

        // Output buffer (n_frames x n_mels)
        let mut mel_spec = vec![0.0_f32; n_frames * self.config.n_mels];

        // Process each frame
        for frame_idx in 0..n_frames {
            let start = frame_idx * hop_length;

            // Apply window and compute power spectrum
            let power_spec = self.compute_power_spectrum(audio, start);

            // Apply mel filterbank
            for mel_idx in 0..self.config.n_mels {
                let mut mel_energy = 0.0_f32;
                for (freq_idx, &power) in power_spec.iter().enumerate() {
                    mel_energy += self.filters[mel_idx * self.n_freqs + freq_idx] * power;
                }

                // Apply log compression with floor to avoid log(0)
                let log_mel = (mel_energy.max(1e-10)).log10();
                mel_spec[frame_idx * self.config.n_mels + mel_idx] = log_mel;
            }
        }

        // Apply normalization (Whisper-style)
        self.normalize_whisper(&mut mel_spec);

        Ok(mel_spec)
    }

    /// Compute power spectrum for a single frame
    fn compute_power_spectrum(&self, audio: &[f32], start: usize) -> Vec<f32> {
        use rustfft::{num_complex::Complex, FftPlanner};

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(self.config.n_fft);

        // Apply window and prepare FFT input
        let mut fft_input: Vec<Complex<f32>> = (0..self.config.n_fft)
            .map(|i| {
                let sample = if start + i < audio.len() {
                    audio[start + i]
                } else {
                    0.0
                };
                Complex::new(sample * self.window[i], 0.0)
            })
            .collect();

        // Compute FFT
        fft.process(&mut fft_input);

        // Compute power spectrum (magnitude squared)
        fft_input
            .iter()
            .take(self.n_freqs)
            .map(Complex::norm_sqr)
            .collect()
    }

    /// Apply Whisper-style normalization
    #[allow(clippy::unused_self)]
    fn normalize_whisper(&self, mel_spec: &mut [f32]) {
        if mel_spec.is_empty() {
            return;
        }

        let max_val = mel_spec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        for x in mel_spec {
            *x = (*x).max(max_val - 8.0);
            *x = (*x + 4.0) / 4.0;
        }
    }

    /// Normalize mel spectrogram with global mean/std
    ///
    /// Applies: (x - mean) / std
    pub fn normalize_global(&self, mel_spec: &mut [f32]) {
        if mel_spec.is_empty() {
            return;
        }

        let mean = mel_spec.iter().sum::<f32>() / mel_spec.len() as f32;
        let variance =
            mel_spec.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / mel_spec.len() as f32;
        let std = variance.sqrt().max(1e-10);

        for x in mel_spec {
            *x = (*x - mean) / std;
        }
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &MelConfig {
        &self.config
    }

    /// Get the number of mel channels
    #[must_use]
    pub fn n_mels(&self) -> usize {
        self.config.n_mels
    }

    /// Get the FFT size
    #[must_use]
    pub fn n_fft(&self) -> usize {
        self.config.n_fft
    }

    /// Get the hop length
    #[must_use]
    pub fn hop_length(&self) -> usize {
        self.config.hop_length
    }

    /// Get the sample rate
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Get the number of frequency bins
    #[must_use]
    pub fn n_freqs(&self) -> usize {
        self.n_freqs
    }

    /// Get the filterbank matrix (n_mels x n_freqs) in row-major order
    #[must_use]
    pub fn filters(&self) -> &[f32] {
        &self.filters
    }

    /// Apply filterbank to power spectrum (scalar implementation)
    #[must_use]
    pub fn apply_filterbank(&self, power_spec: &[f32]) -> Vec<f32> {
        let mut mel_energies = vec![0.0_f32; self.config.n_mels];
        let spec_len = power_spec.len().min(self.n_freqs);

        for (mel_idx, mel_energy) in mel_energies.iter_mut().enumerate() {
            let mut energy = 0.0_f32;
            for (freq_idx, &spec_val) in power_spec.iter().take(spec_len).enumerate() {
                energy += self.filters[mel_idx * self.n_freqs + freq_idx] * spec_val;
            }
            *mel_energy = energy;
        }

        mel_energies
    }

    /// Calculate number of frames for given audio length
    #[must_use]
    pub fn num_frames(&self, audio_len: usize) -> usize {
        if audio_len >= self.config.n_fft {
            (audio_len - self.config.n_fft) / self.config.hop_length + 1
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests;
