//! Synthetic noise generation with ML-based spectral shaping
//!
//! This module provides a WASM-first noise generator using a small MLP
//! to predict magnitude spectra, enabling infinite customization of
//! noise color, texture, and modulation.
//!
//! # Features
//!
//! - **Noise Types**: White, pink, brown, blue, violet, and custom slopes
//! - **Binaural Beats**: Stereo output with frequency offset for brainwave entrainment
//! - **Real-time Control**: Update parameters without reloading the model
//! - **WASM Support**: Browser-compatible with Web Audio API integration
//!
//! # Example
//!
//! ```rust,ignore
//! use aprender::audio::noise::{NoiseGenerator, NoiseConfig, NoiseType};
//!
//! let config = NoiseConfig::brown();
//! let mut generator = NoiseGenerator::new(config)?;
//!
//! let mut buffer = vec![0.0f32; 1024];
//! generator.generate(&mut buffer)?;
//! ```

mod binaural;
mod config;
mod generator;
mod phase;
mod spectral;
mod train;

#[cfg(feature = "audio-noise-wasm")]
mod wasm;

// Re-exports
pub use binaural::BinauralGenerator;
pub use config::{BinauralPreset, NoiseConfig, NoiseType};
pub use generator::NoiseGenerator;
pub use phase::PhaseGenerator;
pub use spectral::SpectralMLP;
pub use train::{spectral_loss, NoiseTrainer, TrainingResult};

#[cfg(feature = "audio-noise-wasm")]
pub use wasm::{noise_version, NoiseGeneratorWasm};

use thiserror::Error;

/// Error type for noise generation operations
#[derive(Debug, Error)]
pub enum NoiseError {
    /// Invalid configuration parameter
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Model loading/saving error
    #[error("Model error: {0}")]
    ModelError(String),

    /// FFT computation error
    #[error("FFT error: {0}")]
    FftError(String),

    /// Buffer size mismatch
    #[error("Buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch { expected: usize, actual: usize },

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type for noise operations
pub type NoiseResult<T> = Result<T, NoiseError>;

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Error Type Tests ==========

    #[test]
    fn test_noise_error_invalid_config() {
        let err = NoiseError::InvalidConfig("bad value".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Invalid configuration"));
        assert!(msg.contains("bad value"));
    }

    #[test]
    fn test_noise_error_model_error() {
        let err = NoiseError::ModelError("weight mismatch".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Model error"));
        assert!(msg.contains("weight mismatch"));
    }

    #[test]
    fn test_noise_error_fft_error() {
        let err = NoiseError::FftError("planner failed".to_string());
        let msg = err.to_string();
        assert!(msg.contains("FFT error"));
    }

    #[test]
    fn test_noise_error_buffer_size_mismatch() {
        let err = NoiseError::BufferSizeMismatch {
            expected: 1024,
            actual: 512,
        };
        let msg = err.to_string();
        assert!(msg.contains("Buffer size mismatch"));
        assert!(msg.contains("1024"));
        assert!(msg.contains("512"));
    }

    #[test]
    fn test_noise_error_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: NoiseError = io_err.into();
        let msg = err.to_string();
        assert!(msg.contains("I/O error"));
    }

    #[test]
    fn test_noise_error_debug() {
        let err = NoiseError::InvalidConfig("test".to_string());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("InvalidConfig"));
    }
}
