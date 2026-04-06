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

// Noise generation module (ML-based spectral synthesis)
#[cfg(feature = "audio-noise")]
pub mod noise;

#[cfg(feature = "audio-playback")]
pub mod playback;

#[cfg(feature = "audio-codec")]
pub mod codec;

pub mod stream;

// Re-exports for convenience
pub use capture::{
    available_backend, has_native_backend, AudioCapture, AudioDevice, BufferCaptureSource,
    CaptureBackend, CaptureConfig, MockCaptureSource, MockSignal,
};
pub use mel::{
    detect_clipping, has_inf, has_nan, stereo_to_mono, validate_audio, ClippingReport, MelConfig,
    MelFilterbank,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoded_audio_new() {
        let samples = vec![0.0, 0.5, 1.0, -0.5];
        let audio = DecodedAudio::new(samples.clone(), 44100, 1);

        assert_eq!(audio.samples.len(), 4);
        assert_eq!(audio.sample_rate, 44100);
        assert_eq!(audio.channels, 1);
        // 4 samples / 44100 Hz = ~0.09ms
        assert!(audio.duration_ms < 1);
    }

    #[test]
    fn test_decoded_audio_stereo() {
        // Stereo: left, right, left, right
        let samples = vec![0.0, 1.0, 0.5, 0.5];
        let audio = DecodedAudio::new(samples.clone(), 44100, 2);

        assert_eq!(audio.channels, 2);
        assert_eq!(audio.samples.len(), 4);
    }

    #[test]
    fn test_decoded_audio_to_mono_from_stereo() {
        // Stereo samples: (L=0.0, R=1.0), (L=0.5, R=0.5)
        let samples = vec![0.0, 1.0, 0.5, 0.5];
        let stereo = DecodedAudio::new(samples, 44100, 2);
        let mono = stereo.to_mono();

        assert_eq!(mono.channels, 1);
        assert_eq!(mono.samples.len(), 2);
        // First sample: (0.0 + 1.0) / 2 = 0.5
        assert!((mono.samples[0] - 0.5).abs() < 0.001);
        // Second sample: (0.5 + 0.5) / 2 = 0.5
        assert!((mono.samples[1] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_decoded_audio_to_mono_already_mono() {
        let samples = vec![0.0, 0.5, 1.0];
        let mono = DecodedAudio::new(samples.clone(), 16000, 1);
        let result = mono.to_mono();

        // Should return clone
        assert_eq!(result.channels, 1);
        assert_eq!(result.samples.len(), 3);
        assert_eq!(result.samples, mono.samples);
    }

    #[test]
    fn test_decoded_audio_zero_sample_rate() {
        let samples = vec![0.0, 0.5];
        let audio = DecodedAudio::new(samples, 0, 1);

        assert_eq!(audio.duration_ms, 0);
    }

    #[test]
    fn test_audio_error_invalid_parameters() {
        let err = AudioError::InvalidParameters("bad param".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Invalid audio parameters"));
        assert!(msg.contains("bad param"));
    }

    #[test]
    fn test_audio_error_invalid_config() {
        let err = AudioError::InvalidConfig("bad config".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Invalid configuration"));
    }

    #[test]
    fn test_audio_error_not_implemented() {
        let err = AudioError::NotImplemented("feature X".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Not implemented"));
        assert!(msg.contains("feature X"));
    }

    #[test]
    fn test_audio_error_not_running() {
        let err = AudioError::NotRunning;
        let msg = err.to_string();
        assert!(msg.contains("not running"));
    }

    #[test]
    fn test_audio_error_capture() {
        let err = AudioError::CaptureError("mic error".to_string());
        let msg = err.to_string();
        assert!(msg.contains("capture error"));
    }

    #[test]
    fn test_audio_error_playback() {
        let err = AudioError::PlaybackError("speaker error".to_string());
        let msg = err.to_string();
        assert!(msg.contains("playback error"));
    }

    #[test]
    fn test_audio_error_codec() {
        let err = AudioError::CodecError("decode failed".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Codec error"));
    }

    #[test]
    fn test_audio_error_unsupported_format() {
        let err = AudioError::UnsupportedFormat("AIFF".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Unsupported audio format"));
        assert!(msg.contains("AIFF"));
    }

    #[test]
    fn test_audio_error_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: AudioError = io_err.into();
        let msg = err.to_string();
        assert!(msg.contains("I/O error"));
    }

    #[test]
    fn test_decoded_audio_clone() {
        let samples = vec![0.1, 0.2, 0.3];
        let audio = DecodedAudio::new(samples, 16000, 1);
        let cloned = audio.clone();

        assert_eq!(audio.samples, cloned.samples);
        assert_eq!(audio.sample_rate, cloned.sample_rate);
        assert_eq!(audio.channels, cloned.channels);
        assert_eq!(audio.duration_ms, cloned.duration_ms);
    }

    #[test]
    fn test_decoded_audio_debug() {
        let audio = DecodedAudio::new(vec![0.0], 16000, 1);
        let debug_str = format!("{:?}", audio);
        assert!(debug_str.contains("DecodedAudio"));
    }
}
