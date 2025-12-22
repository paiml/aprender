//! Audio I/O and signal processing
//!
//! This module provides comprehensive audio processing capabilities in pure Rust:
//!
//! - **capture**: Platform-specific audio input (ALSA, CoreAudio, WASAPI, WebAudio)
//! - **playback**: Platform-specific audio output
//! - **codec**: Decode audio formats (WAV, MP3, AAC, FLAC, Opus, OGG)
//! - **resample**: High-quality sample rate conversion
//! - **mel**: Mel spectrogram computation for ASR/TTS
//! - **stream**: Chunked streaming primitives
//!
//! # Example
//!
//! ```rust,ignore
//! use aprender::audio::mel::{MelFilterbank, MelConfig};
//! use aprender::audio::resample::resample;
//!
//! // Resample to 16kHz (Whisper requirement)
//! let samples_16k = resample(&samples, 44100, 16000);
//!
//! // Compute mel spectrogram
//! let config = MelConfig::whisper();
//! let filterbank = MelFilterbank::new(&config);
//! let mel = filterbank.compute(&samples_16k);
//! ```
//!
//! # Platform Support
//!
//! | Platform | Capture | Playback |
//! |----------|---------|----------|
//! | Linux | ALSA | ALSA |
//! | macOS | CoreAudio | CoreAudio |
//! | Windows | WASAPI | WASAPI |
//! | WASM | Web Audio API | Web Audio API |

pub mod mel;
pub mod resample;

// Platform-specific modules
// Note: capture provides stubs by default, real backends require platform features
pub mod capture;

#[cfg(feature = "audio-playback")]
pub mod playback;

#[cfg(feature = "audio-codec")]
pub mod codec;

pub mod stream;

// Re-exports for convenience
pub use capture::{
    AudioCapture, AudioDevice, BufferCaptureSource, CaptureConfig, MockCaptureSource, MockSignal,
};
pub use mel::{
    detect_clipping, has_nan, validate_audio, ClippingReport, MelConfig, MelFilterbank,
};
pub use resample::resample;

use thiserror::Error;

/// Audio processing error type
#[derive(Debug, Error)]
pub enum AudioError {
    /// Invalid audio parameters
    #[error("Invalid audio parameters: {0}")]
    InvalidParameters(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Feature not implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// Capture/playback not running
    #[error("Audio stream not running")]
    NotRunning,

    /// Audio capture error
    #[error("Audio capture error: {0}")]
    CaptureError(String),

    /// Audio playback error
    #[error("Audio playback error: {0}")]
    PlaybackError(String),

    /// Codec error
    #[error("Codec error: {0}")]
    CodecError(String),

    /// Unsupported format
    #[error("Unsupported audio format: {0}")]
    UnsupportedFormat(String),

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type for audio operations
pub type AudioResult<T> = Result<T, AudioError>;

/// Decoded audio data
#[derive(Debug, Clone)]
pub struct DecodedAudio {
    /// Audio samples in f32 format, normalized to [-1.0, 1.0]
    pub samples: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: u8,
    /// Duration in milliseconds
    pub duration_ms: u64,
}

impl DecodedAudio {
    /// Create new decoded audio
    #[must_use]
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u8) -> Self {
        let duration_ms = if sample_rate > 0 {
            (samples.len() as u64 * 1000) / (u64::from(sample_rate) * u64::from(channels))
        } else {
            0
        };
        Self {
            samples,
            sample_rate,
            channels,
            duration_ms,
        }
    }

    /// Convert stereo to mono by averaging channels
    #[must_use]
    pub fn to_mono(&self) -> Self {
        if self.channels == 1 {
            return self.clone();
        }

        let mono_samples: Vec<f32> = self
            .samples
            .chunks(self.channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
            .collect();

        Self::new(mono_samples, self.sample_rate, 1)
    }
}
