//! Speech processing module for ASR, TTS, VAD, and diarization.
//!
//! This module implements speech processing primitives for Whisper and other
//! speech models, following the spec in `apr-whisper-and-cookbook-support-eoy-2025.md`.
//!
//! # Modules
//!
//! - [`vad`]: Voice Activity Detection (Silero-style or energy-based)
//! - [`asr`]: Automatic Speech Recognition primitives
//! - [`diarization`]: Speaker diarization
//! - [`tts`]: Text-to-Speech primitives
//!
//! # Example
//!
//! ```rust
//! use aprender::speech::vad::{Vad, VadConfig};
//!
//! // Create VAD with default config
//! let config = VadConfig::default();
//! let vad = Vad::new(config).expect("default config is valid");
//!
//! // Detect voice activity in audio samples
//! let samples: Vec<f32> = vec![0.0; 16000]; // 1 second of silence at 16kHz
//! let segments = vad.detect(&samples, 16000).expect("valid input");
//! assert!(segments.is_empty()); // No speech detected in silence
//! ```
//!
//! # PMAT Compliance
//!
//! This module enforces zero-tolerance quality rules:
//! - No `unwrap()` - audio streams can't panic
//! - No `panic!()` - real-time processing requirement
//! - All public APIs return `Result<T, E>`
//! - `#[must_use]` on all Results
//!
//! # References
//!
//! - Silero VAD: <https://github.com/snakers4/silero-vad>
//! - WebRTC VAD: Energy-based voice activity detection
//! - Whisper: <https://github.com/openai/whisper>

pub mod asr;
pub mod diarization;
pub mod tts;
pub mod vad;

// Re-exports
pub use asr::{
    detect_language, is_language_supported, AsrConfig, AsrModel, AsrSession,
    CrossAttentionWeights, LanguageDetection, Segment, StreamingTranscription,
    Transcription, WordTiming, SUPPORTED_LANGUAGES,
};
pub use diarization::{DiarizationConfig, DiarizationResult, Speaker, SpeakerSegment};
pub use tts::{
    estimate_duration, normalize_text, split_sentences, AlignmentInfo, FastSpeech2Synthesizer,
    HifiGanVocoder, SpeechSynthesizer, SynthesisRequest, SynthesisResult, TtsConfig,
    VitsSynthesizer, Vocoder,
};
pub use vad::{Vad, VadConfig, VoiceSegment};

/// Speech processing error type
#[derive(Debug, Clone, PartialEq)]
pub enum SpeechError {
    /// Invalid configuration
    InvalidConfig(String),
    /// Invalid audio parameters
    InvalidAudio(String),
    /// Processing error
    ProcessingError(String),
    /// Insufficient samples
    InsufficientSamples { required: usize, provided: usize },
}

impl std::fmt::Display for SpeechError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "Invalid configuration: {msg}"),
            Self::InvalidAudio(msg) => write!(f, "Invalid audio: {msg}"),
            Self::ProcessingError(msg) => write!(f, "Processing error: {msg}"),
            Self::InsufficientSamples { required, provided } => {
                write!(
                    f,
                    "Insufficient samples: required {required}, provided {provided}"
                )
            }
        }
    }
}

impl std::error::Error for SpeechError {}

/// Result type for speech operations
pub type SpeechResult<T> = Result<T, SpeechError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speech_error_display() {
        let err = SpeechError::InvalidConfig("bad config".to_string());
        assert_eq!(err.to_string(), "Invalid configuration: bad config");

        let err = SpeechError::InsufficientSamples {
            required: 100,
            provided: 50,
        };
        assert_eq!(
            err.to_string(),
            "Insufficient samples: required 100, provided 50"
        );
    }

    #[test]
    fn test_speech_error_equality() {
        let err1 = SpeechError::InvalidAudio("test".to_string());
        let err2 = SpeechError::InvalidAudio("test".to_string());
        assert_eq!(err1, err2);
    }
}
