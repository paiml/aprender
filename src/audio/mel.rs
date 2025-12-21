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

    /// Compute the mel filterbank matrix
    ///
    /// Creates triangular filters spaced on the mel scale.
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

        // Create triangular filters
        for m in 0..n_mels {
            let f_m_minus = bin_points[m];
            let f_m = bin_points[m + 1];
            let f_m_plus = bin_points[m + 2];

            // Rising slope
            for k in f_m_minus..f_m {
                if k < n_freqs && f_m > f_m_minus {
                    let slope = (k - f_m_minus) as f32 / (f_m - f_m_minus) as f32;
                    filters[m * n_freqs + k] = slope;
                }
            }

            // Falling slope
            for k in f_m..f_m_plus {
                if k < n_freqs && f_m_plus > f_m {
                    let slope = (f_m_plus - k) as f32 / (f_m_plus - f_m) as f32;
                    filters[m * n_freqs + k] = slope;
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
            .map(|c| c.norm_sqr())
            .collect()
    }

    /// Apply Whisper-style normalization
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

        for mel_idx in 0..self.config.n_mels {
            let mut energy = 0.0_f32;
            for freq_idx in 0..spec_len {
                energy += self.filters[mel_idx * self.n_freqs + freq_idx] * power_spec[freq_idx];
            }
            mel_energies[mel_idx] = energy;
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
mod tests {
    use super::*;

    // ============================================================
    // UNIT TESTS: Configuration
    // ============================================================

    #[test]
    fn test_mel_config_whisper() {
        let config = MelConfig::whisper();
        assert_eq!(config.n_mels, 80);
        assert_eq!(config.n_fft, 400);
        assert_eq!(config.hop_length, 160);
        assert_eq!(config.sample_rate, 16000);
    }

    #[test]
    fn test_mel_config_tts() {
        let config = MelConfig::tts();
        assert_eq!(config.n_mels, 80);
        assert_eq!(config.n_fft, 1024);
        assert_eq!(config.hop_length, 256);
        assert_eq!(config.sample_rate, 22050);
    }

    #[test]
    fn test_mel_config_n_freqs() {
        let config = MelConfig::whisper();
        assert_eq!(config.n_freqs(), 201); // 400/2 + 1
    }

    // ============================================================
    // UNIT TESTS: Mel scale conversion
    // ============================================================

    #[test]
    fn test_hz_to_mel_zero() {
        let mel = MelFilterbank::hz_to_mel(0.0);
        assert!((mel - 0.0).abs() < 1e-5, "0 Hz should map to 0 mel");
    }

    #[test]
    fn test_hz_to_mel_1000hz() {
        let mel = MelFilterbank::hz_to_mel(1000.0);
        assert!(
            (mel - 1000.0).abs() < 50.0,
            "1000 Hz should be close to 1000 mel, got {mel}"
        );
    }

    #[test]
    fn test_mel_to_hz_roundtrip() {
        let frequencies = [0.0, 100.0, 500.0, 1000.0, 4000.0, 8000.0];
        for &hz in &frequencies {
            let mel = MelFilterbank::hz_to_mel(hz);
            let recovered = MelFilterbank::mel_to_hz(mel);
            assert!(
                (hz - recovered).abs() < 0.1,
                "Roundtrip failed for {hz} Hz: got {recovered}"
            );
        }
    }

    #[test]
    fn test_mel_scale_monotonic() {
        let mut prev_mel = -1.0_f32;
        for hz in (0..8000).step_by(100) {
            let mel = MelFilterbank::hz_to_mel(hz as f32);
            assert!(
                mel > prev_mel,
                "Mel scale should be monotonically increasing"
            );
            prev_mel = mel;
        }
    }

    // ============================================================
    // UNIT TESTS: Filterbank creation
    // ============================================================

    #[test]
    fn test_mel_filterbank_new() {
        let config = MelConfig::whisper();
        let mel = MelFilterbank::new(&config);
        assert_eq!(mel.n_mels(), 80);
        assert_eq!(mel.n_fft(), 400);
        assert_eq!(mel.sample_rate(), 16000);
        assert_eq!(mel.n_freqs(), 201);
    }

    #[test]
    fn test_mel_filterbank_filters_shape() {
        let config = MelConfig::whisper();
        let mel = MelFilterbank::new(&config);
        assert_eq!(mel.filters.len(), 80 * 201);
    }

    #[test]
    fn test_mel_filterbank_filters_nonnegative() {
        let config = MelConfig::whisper();
        let mel = MelFilterbank::new(&config);
        for &f in &mel.filters {
            assert!(f >= 0.0, "Filter values should be non-negative");
        }
    }

    #[test]
    fn test_mel_filterbank_filters_bounded() {
        let config = MelConfig::whisper();
        let mel = MelFilterbank::new(&config);
        for &f in &mel.filters {
            assert!(f <= 1.0, "Filter values should be at most 1.0");
        }
    }

    #[test]
    fn test_hann_window_endpoints() {
        let window = MelFilterbank::hann_window(100);
        assert!(window[0] < 0.01, "Hann window should start near 0");
        assert!(window[99] < 0.01, "Hann window should end near 0");
    }

    #[test]
    fn test_hann_window_peak() {
        let window = MelFilterbank::hann_window(100);
        let mid = window[50];
        assert!(mid > 0.9, "Hann window should peak near 1.0 in middle");
    }

    // ============================================================
    // UNIT TESTS: Spectrogram computation
    // ============================================================

    #[test]
    fn test_mel_compute_empty() {
        let config = MelConfig::whisper();
        let mel = MelFilterbank::new(&config);
        let result = mel.compute(&[]);
        assert!(result.is_ok());
        assert!(result.map_or(false, |v| v.is_empty()));
    }

    #[test]
    fn test_mel_compute_short_audio() {
        let config = MelConfig::whisper();
        let mel = MelFilterbank::new(&config);
        let audio = vec![0.0; 100]; // Too short for even one frame
        let result = mel.compute(&audio);
        assert!(result.is_ok());
        assert!(result.map_or(false, |v| v.is_empty()));
    }

    #[test]
    fn test_mel_compute_exact_one_frame() {
        let config = MelConfig::whisper();
        let mel = MelFilterbank::new(&config);
        let audio = vec![0.0; 400]; // Exactly one FFT window
        let result = mel.compute(&audio).expect("compute should succeed");
        assert_eq!(result.len(), 80 * 1);
    }

    #[test]
    fn test_mel_compute_multiple_frames() {
        let config = MelConfig::whisper();
        let mel = MelFilterbank::new(&config);
        // 16000 samples = 1 second at 16kHz
        // With hop_length=160, we get (16000 - 400) / 160 + 1 = 98 frames
        let audio = vec![0.0; 16000];
        let result = mel.compute(&audio).expect("compute should succeed");
        let n_frames = result.len() / 80;
        assert_eq!(n_frames, 98);
    }

    #[test]
    fn test_mel_compute_sine_wave() {
        let config = MelConfig::whisper();
        let mel = MelFilterbank::new(&config);

        // Generate 1 second of 440 Hz sine wave
        let sample_rate = 16000.0;
        let freq = 440.0;
        let audio: Vec<f32> = (0..16000)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect();

        let result = mel.compute(&audio).expect("compute should succeed");

        let max_val = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_val = result.iter().cloned().fold(f32::INFINITY, f32::min);

        assert!(max_val.is_finite(), "Max should be finite");
        assert!(min_val.is_finite(), "Min should be finite");
        assert!(max_val > min_val, "Should have variation in output");
    }

    #[test]
    fn test_num_frames() {
        let config = MelConfig::whisper();
        let mel = MelFilterbank::new(&config);

        assert_eq!(mel.num_frames(0), 0);
        assert_eq!(mel.num_frames(100), 0);
        assert_eq!(mel.num_frames(400), 1);
        assert_eq!(mel.num_frames(560), 2);
        assert_eq!(mel.num_frames(16000), 98);
    }

    // ============================================================
    // UNIT TESTS: Normalization
    // ============================================================

    #[test]
    fn test_normalize_global_empty() {
        let config = MelConfig::whisper();
        let mel = MelFilterbank::new(&config);
        let mut data: Vec<f32> = vec![];
        mel.normalize_global(&mut data);
        assert!(data.is_empty());
    }

    #[test]
    fn test_normalize_global_mean_zero() {
        let config = MelConfig::whisper();
        let mel = MelFilterbank::new(&config);
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        mel.normalize_global(&mut data);

        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(mean.abs() < 1e-5, "Mean after normalization should be ~0");
    }

    #[test]
    fn test_normalize_global_std_one() {
        let config = MelConfig::whisper();
        let mel = MelFilterbank::new(&config);
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        mel.normalize_global(&mut data);

        let variance: f32 = data.iter().map(|&x| x.powi(2)).sum::<f32>() / data.len() as f32;
        let std = variance.sqrt();
        assert!(
            (std - 1.0).abs() < 1e-5,
            "Std after normalization should be ~1, got {std}"
        );
    }

    // ============================================================
    // UNIT TESTS: Apply filterbank
    // ============================================================

    #[test]
    fn test_apply_filterbank_shape() {
        let config = MelConfig::whisper();
        let mel = MelFilterbank::new(&config);
        let power_spec = vec![1.0; mel.n_freqs()];
        let result = mel.apply_filterbank(&power_spec);
        assert_eq!(result.len(), 80);
    }

    #[test]
    fn test_apply_filterbank_zeros() {
        let config = MelConfig::whisper();
        let mel = MelFilterbank::new(&config);
        let power_spec = vec![0.0; mel.n_freqs()];
        let result = mel.apply_filterbank(&power_spec);
        for &val in &result {
            assert!((val - 0.0).abs() < 1e-10, "Zero input should give zero output");
        }
    }
}
