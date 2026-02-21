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

include!("wiener_filter_isolator.rs");
include!("isolation_tests.rs");
