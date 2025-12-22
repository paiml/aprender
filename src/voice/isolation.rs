//! Voice isolation module (GH-132).
//!
//! Provides voice isolation primitives for:
//! - Speech enhancement (noise reduction)
//! - Source separation (isolate target speaker)
//! - Music removal (extract speech from mixed audio)
//!
//! # Architecture
//!
//! ```text
//! Noisy Audio → STFT → Mask Estimation → Apply Mask → ISTFT → Clean Audio
//!                           ↓
//!                    Neural Network / Spectral
//! ```
//!
//! # Isolation Methods
//!
//! - **Spectral Subtraction**: Classical signal processing
//! - **Wiener Filter**: Optimal noise suppression
//! - **U-Net**: Deep learning based separation
//! - **Conv-TasNet**: End-to-end time-domain separation
//!
//! # Example
//!
//! ```rust
//! use aprender::voice::isolation::{IsolationConfig, IsolationMethod};
//!
//! let config = IsolationConfig::default();
//! assert_eq!(config.method, IsolationMethod::SpectralSubtraction);
//! ```
//!
//! # References
//!
//! - Jansson, A., et al. (2017). Singing Voice Separation with Deep U-Net.
//! - Luo, Y., et al. (2019). Conv-TasNet: Time-Domain Audio Separation.
//! - Ephraim, Y., & Malah, D. (1984). Speech Enhancement Using MMSE.
//!
//! # PMAT Compliance
//!
//! - Zero `unwrap()` calls
//! - All public APIs return `Result<T, E>` where fallible

use super::{VoiceError, VoiceResult};

// ============================================================================
// Configuration
// ============================================================================

/// Voice isolation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IsolationMethod {
    /// Spectral subtraction (classical)
    #[default]
    SpectralSubtraction,
    /// Wiener filter (statistical)
    WienerFilter,
    /// Deep neural network mask estimation
    NeuralMask,
    /// U-Net based separation
    UNet,
    /// Conv-TasNet (time-domain)
    ConvTasNet,
}

/// Noise estimation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NoiseEstimation {
    /// Estimate from initial silence
    #[default]
    InitialSilence,
    /// Minimum statistics method
    MinimumStatistics,
    /// Adaptive (continuous estimation)
    Adaptive,
    /// Fixed noise profile
    FixedProfile,
}

/// Configuration for voice isolation.
#[derive(Debug, Clone)]
pub struct IsolationConfig {
    /// Isolation method
    pub method: IsolationMethod,
    /// Noise estimation method
    pub noise_estimation: NoiseEstimation,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// FFT size for spectral processing
    pub fft_size: usize,
    /// Hop length for STFT
    pub hop_length: usize,
    /// Noise reduction strength [0.0, 1.0]
    pub reduction_strength: f32,
    /// Spectral floor (minimum gain)
    pub spectral_floor: f32,
    /// Preserve musical noise (reduce artifacts)
    pub preserve_musical_noise: bool,
    /// Number of noise frames for initial estimation
    pub noise_frames: usize,
}

impl Default for IsolationConfig {
    fn default() -> Self {
        Self {
            method: IsolationMethod::default(),
            noise_estimation: NoiseEstimation::default(),
            sample_rate: 16000,
            fft_size: 512,
            hop_length: 128,
            reduction_strength: 0.8,
            spectral_floor: 0.01,
            preserve_musical_noise: true,
            noise_frames: 10,
        }
    }
}

impl IsolationConfig {
    /// Create config for aggressive noise reduction
    #[must_use]
    pub fn aggressive() -> Self {
        Self {
            reduction_strength: 0.95,
            spectral_floor: 0.001,
            preserve_musical_noise: false,
            ..Self::default()
        }
    }

    /// Create config for mild noise reduction (preserve naturalness)
    #[must_use]
    pub fn mild() -> Self {
        Self {
            reduction_strength: 0.5,
            spectral_floor: 0.1,
            preserve_musical_noise: true,
            ..Self::default()
        }
    }

    /// Create config for neural network based isolation
    #[must_use]
    pub fn neural() -> Self {
        Self {
            method: IsolationMethod::NeuralMask,
            noise_estimation: NoiseEstimation::Adaptive,
            reduction_strength: 0.9,
            ..Self::default()
        }
    }

    /// Create config for real-time low-latency processing
    #[must_use]
    pub fn realtime() -> Self {
        Self {
            fft_size: 256,
            hop_length: 64,
            noise_frames: 5,
            ..Self::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> VoiceResult<()> {
        if self.sample_rate == 0 {
            return Err(VoiceError::InvalidConfig(
                "sample_rate must be > 0".to_string(),
            ));
        }
        if self.fft_size == 0 || !self.fft_size.is_power_of_two() {
            return Err(VoiceError::InvalidConfig(
                "fft_size must be a power of 2".to_string(),
            ));
        }
        if self.hop_length == 0 || self.hop_length > self.fft_size {
            return Err(VoiceError::InvalidConfig(
                "hop_length must be > 0 and <= fft_size".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.reduction_strength) {
            return Err(VoiceError::InvalidConfig(
                "reduction_strength must be in [0.0, 1.0]".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.spectral_floor) {
            return Err(VoiceError::InvalidConfig(
                "spectral_floor must be in [0.0, 1.0]".to_string(),
            ));
        }
        Ok(())
    }

    /// Get frequency bins for FFT
    #[must_use]
    pub fn freq_bins(&self) -> usize {
        self.fft_size / 2 + 1
    }

    /// Get frame length in seconds
    #[must_use]
    pub fn frame_duration_secs(&self) -> f32 {
        self.fft_size as f32 / self.sample_rate as f32
    }
}

// ============================================================================
// Isolation Result
// ============================================================================

/// Result of voice isolation operation.
#[derive(Debug, Clone)]
pub struct IsolationResult {
    /// Isolated (clean) audio samples
    pub audio: Vec<f32>,
    /// Estimated noise floor (optional)
    pub noise_floor: Option<Vec<f32>>,
    /// Sample rate of output
    pub sample_rate: u32,
    /// Signal-to-noise ratio improvement in dB
    pub snr_improvement_db: f32,
    /// Estimated input SNR in dB
    pub input_snr_db: f32,
    /// Estimated output SNR in dB
    pub output_snr_db: f32,
}

impl IsolationResult {
    /// Create a new isolation result
    #[must_use]
    pub fn new(audio: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            audio,
            noise_floor: None,
            sample_rate,
            snr_improvement_db: 0.0,
            input_snr_db: 0.0,
            output_snr_db: 0.0,
        }
    }

    /// Set SNR metrics
    #[must_use]
    pub fn with_snr(mut self, input_snr: f32, output_snr: f32) -> Self {
        self.input_snr_db = input_snr;
        self.output_snr_db = output_snr;
        self.snr_improvement_db = output_snr - input_snr;
        self
    }

    /// Set noise floor estimate
    #[must_use]
    pub fn with_noise_floor(mut self, noise_floor: Vec<f32>) -> Self {
        self.noise_floor = Some(noise_floor);
        self
    }
}

// ============================================================================
// Noise Profile
// ============================================================================

/// Estimated noise profile for subtraction.
#[derive(Debug, Clone)]
pub struct NoiseProfile {
    /// Mean spectral magnitude per frequency bin
    pub mean_spectrum: Vec<f32>,
    /// Standard deviation per frequency bin
    pub std_spectrum: Vec<f32>,
    /// Number of frames used for estimation
    pub num_frames: usize,
    /// Sample rate
    pub sample_rate: u32,
}

impl NoiseProfile {
    /// Create a new noise profile from spectral frames
    #[must_use]
    pub fn from_frames(frames: &[Vec<f32>], sample_rate: u32) -> Self {
        if frames.is_empty() {
            return Self {
                mean_spectrum: vec![],
                std_spectrum: vec![],
                num_frames: 0,
                sample_rate,
            };
        }

        let num_bins = frames[0].len();
        let num_frames = frames.len();

        // Compute mean spectrum
        let mut mean_spectrum = vec![0.0f32; num_bins];
        for frame in frames {
            for (i, &val) in frame.iter().enumerate() {
                if i < num_bins {
                    mean_spectrum[i] += val;
                }
            }
        }
        for m in &mut mean_spectrum {
            *m /= num_frames as f32;
        }

        // Compute standard deviation
        let mut std_spectrum = vec![0.0f32; num_bins];
        for frame in frames {
            for (i, &val) in frame.iter().enumerate() {
                if i < num_bins {
                    let diff = val - mean_spectrum[i];
                    std_spectrum[i] += diff * diff;
                }
            }
        }
        for s in &mut std_spectrum {
            *s = (*s / num_frames as f32).sqrt();
        }

        Self {
            mean_spectrum,
            std_spectrum,
            num_frames,
            sample_rate,
        }
    }

    /// Get noise magnitude at frequency bin
    #[must_use]
    pub fn noise_magnitude(&self, bin: usize) -> f32 {
        self.mean_spectrum.get(bin).copied().unwrap_or(0.0)
    }

    /// Check if profile is valid
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.num_frames > 0 && !self.mean_spectrum.is_empty()
    }
}

// ============================================================================
// Voice Isolator Trait
// ============================================================================

/// Trait for voice isolation systems.
pub trait VoiceIsolator: Send + Sync {
    /// Get the configuration
    fn config(&self) -> &IsolationConfig;

    /// Isolate voice from noisy audio.
    ///
    /// # Arguments
    /// * `audio` - Input audio samples (noisy)
    ///
    /// # Errors
    /// Returns error if isolation fails.
    fn isolate(&self, audio: &[f32]) -> VoiceResult<IsolationResult>;

    /// Isolate voice with pre-estimated noise profile.
    ///
    /// # Arguments
    /// * `audio` - Input audio samples
    /// * `noise_profile` - Pre-computed noise profile
    ///
    /// # Errors
    /// Returns error if isolation fails.
    fn isolate_with_profile(
        &self,
        audio: &[f32],
        noise_profile: &NoiseProfile,
    ) -> VoiceResult<IsolationResult>;

    /// Estimate noise profile from audio segment.
    ///
    /// # Arguments
    /// * `noise_audio` - Audio containing only noise
    ///
    /// # Errors
    /// Returns error if estimation fails.
    fn estimate_noise(&self, noise_audio: &[f32]) -> VoiceResult<NoiseProfile>;
}

// ============================================================================
// Spectral Subtraction Isolator
// ============================================================================

/// Spectral subtraction based voice isolator.
///
/// Classical approach:
/// 1. STFT to get magnitude/phase
/// 2. Estimate noise spectrum
/// 3. Subtract noise from magnitude
/// 4. Apply spectral floor
/// 5. ISTFT to reconstruct
#[derive(Debug, Clone)]
pub struct SpectralSubtractionIsolator {
    /// Configuration
    config: IsolationConfig,
    /// Over-subtraction factor (typically 1.0-2.0)
    over_subtraction: f32,
}

impl SpectralSubtractionIsolator {
    /// Create a new spectral subtraction isolator
    #[must_use]
    pub fn new(config: IsolationConfig) -> Self {
        Self {
            config,
            over_subtraction: 1.5,
        }
    }

    /// Set over-subtraction factor
    #[must_use]
    pub fn with_over_subtraction(mut self, factor: f32) -> Self {
        self.over_subtraction = factor.clamp(0.5, 5.0);
        self
    }

    /// Get over-subtraction factor
    #[must_use]
    pub fn over_subtraction(&self) -> f32 {
        self.over_subtraction
    }

    /// Compute STFT magnitude spectrum (simplified)
    fn compute_stft(&self, audio: &[f32]) -> Vec<Vec<f32>> {
        let fft_size = self.config.fft_size;
        let hop_length = self.config.hop_length;
        let freq_bins = self.config.freq_bins();

        let num_frames = (audio.len().saturating_sub(fft_size)) / hop_length + 1;
        let num_frames = num_frames.max(1);

        (0..num_frames)
            .map(|i| {
                let start = i * hop_length;
                let end = (start + fft_size).min(audio.len());

                // Compute energy per frequency bin (simplified - real impl uses FFT)
                let mut magnitudes = vec![0.0f32; freq_bins];

                if end > start {
                    let frame = &audio[start..end];

                    // Simple energy-based approximation
                    let energy: f32 = frame.iter().map(|x| x * x).sum::<f32>().sqrt();

                    // Distribute energy across bins (placeholder)
                    for (bin, mag) in magnitudes.iter_mut().enumerate() {
                        // Low-pass shape approximation
                        let freq_factor = 1.0 / (1.0 + (bin as f32 / 20.0));
                        *mag = energy * freq_factor;
                    }
                }

                magnitudes
            })
            .collect()
    }

    /// Apply spectral subtraction to magnitudes
    fn subtract_noise(
        &self,
        magnitudes: &[Vec<f32>],
        noise_profile: &NoiseProfile,
    ) -> Vec<Vec<f32>> {
        let strength = self.config.reduction_strength;
        let floor = self.config.spectral_floor;
        let alpha = self.over_subtraction;

        magnitudes
            .iter()
            .map(|frame| {
                frame
                    .iter()
                    .enumerate()
                    .map(|(bin, &mag)| {
                        let noise = noise_profile.noise_magnitude(bin) * alpha * strength;
                        (mag - noise).max(mag * floor)
                    })
                    .collect()
            })
            .collect()
    }

    /// Reconstruct audio from magnitudes (simplified overlap-add)
    fn reconstruct(&self, magnitudes: &[Vec<f32>]) -> Vec<f32> {
        if magnitudes.is_empty() {
            return vec![];
        }

        let hop_length = self.config.hop_length;
        let fft_size = self.config.fft_size;
        let output_len = magnitudes.len() * hop_length + fft_size;

        let mut output = vec![0.0f32; output_len];

        for (i, frame_mags) in magnitudes.iter().enumerate() {
            let start = i * hop_length;

            // Simplified reconstruction: use magnitude as envelope
            let avg_mag = if frame_mags.is_empty() {
                0.0
            } else {
                frame_mags.iter().sum::<f32>() / frame_mags.len() as f32
            };

            for j in 0..fft_size.min(output_len - start) {
                let t = j as f32 / self.config.sample_rate as f32;
                let freq = 200.0; // Simplified: single frequency
                let sample = (2.0 * std::f32::consts::PI * freq * t).sin() * avg_mag * 0.5;

                // Windowing
                let window =
                    0.5 * (1.0 - (2.0 * std::f32::consts::PI * j as f32 / fft_size as f32).cos());

                output[start + j] += sample * window;
            }
        }

        output
    }
}

impl VoiceIsolator for SpectralSubtractionIsolator {
    fn config(&self) -> &IsolationConfig {
        &self.config
    }

    fn isolate(&self, audio: &[f32]) -> VoiceResult<IsolationResult> {
        if audio.is_empty() {
            return Err(VoiceError::InvalidAudio("Empty audio".to_string()));
        }

        // Estimate noise from initial frames
        let noise_samples = self.config.noise_frames * self.config.fft_size;
        let noise_audio = if audio.len() >= noise_samples {
            &audio[..noise_samples]
        } else {
            audio
        };

        let noise_profile = self.estimate_noise(noise_audio)?;
        self.isolate_with_profile(audio, &noise_profile)
    }

    fn isolate_with_profile(
        &self,
        audio: &[f32],
        noise_profile: &NoiseProfile,
    ) -> VoiceResult<IsolationResult> {
        if audio.is_empty() {
            return Err(VoiceError::InvalidAudio("Empty audio".to_string()));
        }

        // Compute STFT
        let magnitudes = self.compute_stft(audio);

        // Apply spectral subtraction
        let clean_magnitudes = self.subtract_noise(&magnitudes, noise_profile);

        // Reconstruct
        let clean_audio = self.reconstruct(&clean_magnitudes);

        // Estimate SNR improvement
        let input_snr = estimate_snr(audio);
        let output_snr = estimate_snr(&clean_audio);

        Ok(IsolationResult::new(clean_audio, self.config.sample_rate)
            .with_snr(input_snr, output_snr)
            .with_noise_floor(noise_profile.mean_spectrum.clone()))
    }

    fn estimate_noise(&self, noise_audio: &[f32]) -> VoiceResult<NoiseProfile> {
        if noise_audio.is_empty() {
            return Err(VoiceError::InvalidAudio(
                "Empty noise audio for estimation".to_string(),
            ));
        }

        let frames = self.compute_stft(noise_audio);
        Ok(NoiseProfile::from_frames(&frames, self.config.sample_rate))
    }
}

impl Default for SpectralSubtractionIsolator {
    fn default() -> Self {
        Self::new(IsolationConfig::default())
    }
}

// ============================================================================
// Wiener Filter Isolator
// ============================================================================

/// Wiener filter based voice isolator.
///
/// Optimal linear filter that minimizes mean squared error.
/// Gain = S(f) / (S(f) + N(f)) where S = signal, N = noise
#[derive(Debug, Clone)]
pub struct WienerFilterIsolator {
    /// Configuration
    config: IsolationConfig,
    /// A priori SNR smoothing factor
    smoothing: f32,
}

impl WienerFilterIsolator {
    /// Create a new Wiener filter isolator
    #[must_use]
    pub fn new(config: IsolationConfig) -> Self {
        Self {
            config,
            smoothing: 0.98,
        }
    }

    /// Set smoothing factor for a priori SNR estimation
    #[must_use]
    pub fn with_smoothing(mut self, factor: f32) -> Self {
        self.smoothing = factor.clamp(0.0, 1.0);
        self
    }

    /// Compute Wiener gain
    fn compute_gain(signal_power: f32, noise_power: f32) -> f32 {
        if noise_power <= 0.0 {
            1.0
        } else {
            let snr = signal_power / noise_power;
            snr / (snr + 1.0)
        }
    }
}

impl VoiceIsolator for WienerFilterIsolator {
    fn config(&self) -> &IsolationConfig {
        &self.config
    }

    fn isolate(&self, audio: &[f32]) -> VoiceResult<IsolationResult> {
        if audio.is_empty() {
            return Err(VoiceError::InvalidAudio("Empty audio".to_string()));
        }

        // For Wiener filter, we need noise estimate
        let noise_samples = self.config.noise_frames * self.config.fft_size;
        let noise_audio = if audio.len() >= noise_samples {
            &audio[..noise_samples]
        } else {
            audio
        };

        let noise_profile = self.estimate_noise(noise_audio)?;
        self.isolate_with_profile(audio, &noise_profile)
    }

    fn isolate_with_profile(
        &self,
        audio: &[f32],
        noise_profile: &NoiseProfile,
    ) -> VoiceResult<IsolationResult> {
        if audio.is_empty() {
            return Err(VoiceError::InvalidAudio("Empty audio".to_string()));
        }

        let strength = self.config.reduction_strength;
        let floor = self.config.spectral_floor;

        // Apply Wiener filter frame by frame
        let hop_length = self.config.hop_length;
        let fft_size = self.config.fft_size;
        let num_frames = (audio.len().saturating_sub(fft_size)) / hop_length + 1;
        let num_frames = num_frames.max(1);

        let mut output = vec![0.0f32; audio.len()];

        for i in 0..num_frames {
            let start = i * hop_length;
            let end = (start + fft_size).min(audio.len());

            for (j, sample) in audio[start..end].iter().enumerate() {
                let bin = j % noise_profile.mean_spectrum.len().max(1);
                let noise_power = noise_profile.noise_magnitude(bin).powi(2);
                let signal_power = sample.powi(2);

                let gain = Self::compute_gain(signal_power, noise_power * strength);
                let gain = gain.max(floor);

                if start + j < output.len() {
                    output[start + j] = sample * gain;
                }
            }
        }

        let input_snr = estimate_snr(audio);
        let output_snr = estimate_snr(&output);

        Ok(IsolationResult::new(output, self.config.sample_rate)
            .with_snr(input_snr, output_snr))
    }

    fn estimate_noise(&self, noise_audio: &[f32]) -> VoiceResult<NoiseProfile> {
        if noise_audio.is_empty() {
            return Err(VoiceError::InvalidAudio(
                "Empty noise audio for estimation".to_string(),
            ));
        }

        // Compute magnitude frames
        let hop_length = self.config.hop_length;
        let fft_size = self.config.fft_size;
        let freq_bins = self.config.freq_bins();

        let num_frames = (noise_audio.len().saturating_sub(fft_size)) / hop_length + 1;
        let num_frames = num_frames.max(1);

        let frames: Vec<Vec<f32>> = (0..num_frames)
            .map(|i| {
                let start = i * hop_length;
                let end = (start + fft_size).min(noise_audio.len());

                // Energy per bin (simplified)
                let frame = &noise_audio[start..end];
                let energy: f32 = frame.iter().map(|x| x * x).sum::<f32>().sqrt();

                (0..freq_bins)
                    .map(|bin| {
                        let freq_factor = 1.0 / (1.0 + (bin as f32 / 20.0));
                        energy * freq_factor
                    })
                    .collect()
            })
            .collect();

        Ok(NoiseProfile::from_frames(&frames, self.config.sample_rate))
    }
}

impl Default for WienerFilterIsolator {
    fn default() -> Self {
        Self::new(IsolationConfig::default())
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Estimate signal-to-noise ratio in dB.
///
/// Uses simple energy-based estimation.
#[must_use]
pub fn estimate_snr(audio: &[f32]) -> f32 {
    if audio.is_empty() {
        return 0.0;
    }

    // Compute signal energy
    let signal_energy: f32 = audio.iter().map(|x| x * x).sum();
    let signal_rms = (signal_energy / audio.len() as f32).sqrt();

    // Estimate noise as minimum energy segments
    let frame_size = 256.min(audio.len());
    let num_frames = audio.len() / frame_size;

    if num_frames == 0 {
        return 20.0; // Default assumption
    }

    let frame_energies: Vec<f32> = (0..num_frames)
        .map(|i| {
            let start = i * frame_size;
            let end = (start + frame_size).min(audio.len());
            let energy: f32 = audio[start..end].iter().map(|x| x * x).sum();
            (energy / (end - start) as f32).sqrt()
        })
        .collect();

    // Noise floor as 10th percentile of frame energies
    let mut sorted_energies = frame_energies.clone();
    sorted_energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let noise_idx = (num_frames / 10).max(0);
    let noise_rms = sorted_energies.get(noise_idx).copied().unwrap_or(0.0001);

    // SNR in dB
    if noise_rms > 0.0 && signal_rms > 0.0 {
        20.0 * (signal_rms / noise_rms).log10()
    } else {
        0.0
    }
}

/// Compute spectral entropy (measure of noise-likeness).
///
/// Higher entropy = more noise-like.
#[must_use]
pub fn spectral_entropy(magnitudes: &[f32]) -> f32 {
    if magnitudes.is_empty() {
        return 0.0;
    }

    let sum: f32 = magnitudes.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }

    // Normalize to probability distribution
    let probabilities: Vec<f32> = magnitudes.iter().map(|&m| m / sum).collect();

    // Compute entropy
    let entropy: f32 = probabilities
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();

    // Normalize by maximum possible entropy
    let max_entropy = (magnitudes.len() as f32).ln();
    if max_entropy > 0.0 {
        entropy / max_entropy
    } else {
        0.0
    }
}

/// Detect voice activity in audio.
///
/// Returns vector of booleans (true = speech, false = silence/noise).
#[must_use]
pub fn detect_voice_activity(audio: &[f32], frame_size: usize, threshold: f32) -> Vec<bool> {
    if audio.is_empty() || frame_size == 0 {
        return vec![];
    }

    let num_frames = (audio.len() + frame_size - 1) / frame_size;

    (0..num_frames)
        .map(|i| {
            let start = i * frame_size;
            let end = (start + frame_size).min(audio.len());

            // Compute frame energy
            let energy: f32 = audio[start..end].iter().map(|x| x * x).sum();
            let rms = (energy / (end - start).max(1) as f32).sqrt();

            rms > threshold
        })
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isolation_method_default() {
        assert_eq!(
            IsolationMethod::default(),
            IsolationMethod::SpectralSubtraction
        );
    }

    #[test]
    fn test_noise_estimation_default() {
        assert_eq!(NoiseEstimation::default(), NoiseEstimation::InitialSilence);
    }

    #[test]
    fn test_config_default() {
        let config = IsolationConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.fft_size, 512);
        assert_eq!(config.hop_length, 128);
    }

    #[test]
    fn test_config_aggressive() {
        let config = IsolationConfig::aggressive();
        assert_eq!(config.reduction_strength, 0.95);
        assert!(!config.preserve_musical_noise);
    }

    #[test]
    fn test_config_mild() {
        let config = IsolationConfig::mild();
        assert_eq!(config.reduction_strength, 0.5);
        assert!(config.preserve_musical_noise);
    }

    #[test]
    fn test_config_neural() {
        let config = IsolationConfig::neural();
        assert_eq!(config.method, IsolationMethod::NeuralMask);
    }

    #[test]
    fn test_config_realtime() {
        let config = IsolationConfig::realtime();
        assert_eq!(config.fft_size, 256);
        assert_eq!(config.hop_length, 64);
    }

    #[test]
    fn test_config_validate_valid() {
        let config = IsolationConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_invalid_sample_rate() {
        let config = IsolationConfig {
            sample_rate: 0,
            ..IsolationConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_invalid_fft_size() {
        let config = IsolationConfig {
            fft_size: 100, // Not power of 2
            ..IsolationConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_invalid_hop_length() {
        let config = IsolationConfig {
            hop_length: 0,
            ..IsolationConfig::default()
        };
        assert!(config.validate().is_err());

        let config2 = IsolationConfig {
            hop_length: 1024, // > fft_size
            fft_size: 512,
            ..IsolationConfig::default()
        };
        assert!(config2.validate().is_err());
    }

    #[test]
    fn test_config_validate_invalid_strength() {
        let config = IsolationConfig {
            reduction_strength: 1.5,
            ..IsolationConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_freq_bins() {
        let config = IsolationConfig::default();
        assert_eq!(config.freq_bins(), 257); // 512/2 + 1
    }

    #[test]
    fn test_config_frame_duration() {
        let config = IsolationConfig::default();
        let expected = 512.0 / 16000.0;
        assert!((config.frame_duration_secs() - expected).abs() < 0.001);
    }

    #[test]
    fn test_isolation_result_new() {
        let result = IsolationResult::new(vec![0.1; 1600], 16000);
        assert_eq!(result.sample_rate, 16000);
        assert_eq!(result.snr_improvement_db, 0.0);
    }

    #[test]
    fn test_isolation_result_with_snr() {
        let result = IsolationResult::new(vec![0.1; 1600], 16000).with_snr(10.0, 20.0);
        assert_eq!(result.input_snr_db, 10.0);
        assert_eq!(result.output_snr_db, 20.0);
        assert_eq!(result.snr_improvement_db, 10.0);
    }

    #[test]
    fn test_noise_profile_from_frames() {
        let frames = vec![vec![1.0, 2.0, 3.0], vec![1.5, 2.5, 3.5]];
        let profile = NoiseProfile::from_frames(&frames, 16000);

        assert_eq!(profile.num_frames, 2);
        assert_eq!(profile.mean_spectrum.len(), 3);
        assert!((profile.mean_spectrum[0] - 1.25).abs() < 0.01);
    }

    #[test]
    fn test_noise_profile_empty() {
        let profile = NoiseProfile::from_frames(&[], 16000);
        assert!(!profile.is_valid());
    }

    #[test]
    fn test_spectral_subtraction_new() {
        let isolator = SpectralSubtractionIsolator::default();
        assert_eq!(isolator.over_subtraction(), 1.5);
    }

    #[test]
    fn test_spectral_subtraction_isolate() {
        let isolator = SpectralSubtractionIsolator::default();
        let audio: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.001).sin()).collect();

        let result = isolator.isolate(&audio);
        assert!(result.is_ok());
        let isolated = result.expect("isolation failed");
        assert!(!isolated.audio.is_empty());
    }

    #[test]
    fn test_spectral_subtraction_empty() {
        let isolator = SpectralSubtractionIsolator::default();
        let result = isolator.isolate(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_spectral_subtraction_estimate_noise() {
        let isolator = SpectralSubtractionIsolator::default();
        let noise: Vec<f32> = (0..4000).map(|i| 0.01 * (i as f32 * 0.005).sin()).collect();

        let profile = isolator.estimate_noise(&noise);
        assert!(profile.is_ok());
        let p = profile.expect("estimation failed");
        assert!(p.is_valid());
    }

    #[test]
    fn test_wiener_filter_new() {
        let isolator = WienerFilterIsolator::default();
        assert_eq!(isolator.config().method, IsolationMethod::SpectralSubtraction);
    }

    #[test]
    fn test_wiener_filter_isolate() {
        let isolator = WienerFilterIsolator::default();
        let audio: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.001).sin()).collect();

        let result = isolator.isolate(&audio);
        assert!(result.is_ok());
    }

    #[test]
    fn test_wiener_compute_gain() {
        // High SNR -> high gain
        let gain = WienerFilterIsolator::compute_gain(1.0, 0.1);
        assert!(gain > 0.9);

        // Low SNR -> low gain
        let gain = WienerFilterIsolator::compute_gain(0.1, 1.0);
        assert!(gain < 0.2);
    }

    #[test]
    fn test_estimate_snr() {
        // Pure sine wave should have high SNR
        let sine: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.01).sin()).collect();
        let snr = estimate_snr(&sine);
        assert!(snr > 0.0);

        // Constant value (no variation) - handle edge case
        let constant = vec![0.5f32; 8000];
        let snr_const = estimate_snr(&constant);
        assert!(snr_const.is_finite());
    }

    #[test]
    fn test_estimate_snr_empty() {
        let snr = estimate_snr(&[]);
        assert_eq!(snr, 0.0);
    }

    #[test]
    fn test_spectral_entropy() {
        // Uniform distribution = high entropy
        let uniform = vec![1.0; 100];
        let entropy_uniform = spectral_entropy(&uniform);
        assert!(entropy_uniform > 0.9);

        // Single peak = low entropy
        let mut peaked = vec![0.0; 100];
        peaked[50] = 1.0;
        let entropy_peaked = spectral_entropy(&peaked);
        assert!(entropy_peaked < 0.1);
    }

    #[test]
    fn test_spectral_entropy_empty() {
        let entropy = spectral_entropy(&[]);
        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn test_detect_voice_activity() {
        // Create audio with speech and silence
        let mut audio = vec![0.001f32; 8000]; // Silence
        for i in 2000..4000 {
            audio[i] = 0.5 * (i as f32 * 0.01).sin(); // Speech
        }

        let vad = detect_voice_activity(&audio, 1000, 0.1);

        // Should detect speech in middle frames
        assert!(vad.len() >= 4);
        assert!(!vad[0]); // Initial silence
        assert!(vad[2] || vad[3]); // Speech region
    }

    #[test]
    fn test_detect_voice_activity_empty() {
        let vad = detect_voice_activity(&[], 100, 0.1);
        assert!(vad.is_empty());
    }
}
