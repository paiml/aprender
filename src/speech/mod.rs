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
    detect_language, is_language_supported, AsrConfig, AsrModel, AsrSession, CrossAttentionWeights,
    LanguageDetection, Segment, StreamingTranscription, Transcription, WordTiming,
    SUPPORTED_LANGUAGES,
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

    #[test]
    fn test_speech_error_invalid_audio_display() {
        let err = SpeechError::InvalidAudio("corrupted data".to_string());
        assert_eq!(err.to_string(), "Invalid audio: corrupted data");
    }

    #[test]
    fn test_speech_error_processing_error_display() {
        let err = SpeechError::ProcessingError("failed to decode".to_string());
        assert_eq!(err.to_string(), "Processing error: failed to decode");
    }

    #[test]
    fn test_speech_error_debug() {
        let err = SpeechError::InvalidConfig("test".to_string());
        let debug = format!("{:?}", err);
        assert!(debug.contains("InvalidConfig"));
        assert!(debug.contains("test"));
    }

    #[test]
    fn test_speech_error_clone() {
        let err = SpeechError::ProcessingError("error".to_string());
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn test_speech_error_is_error() {
        // Test that SpeechError implements std::error::Error
        let err: &dyn std::error::Error = &SpeechError::InvalidConfig("test".to_string());
        assert!(err.to_string().contains("Invalid configuration"));
    }

    #[test]
    fn test_speech_error_inequality() {
        let err1 = SpeechError::InvalidAudio("test1".to_string());
        let err2 = SpeechError::InvalidAudio("test2".to_string());
        assert_ne!(err1, err2);
    }

    #[test]
    fn test_speech_error_different_variants() {
        let err1 = SpeechError::InvalidConfig("msg".to_string());
        let err2 = SpeechError::InvalidAudio("msg".to_string());
        assert_ne!(err1, err2);
    }

    #[test]
    fn test_speech_error_insufficient_samples_fields() {
        let err = SpeechError::InsufficientSamples {
            required: 1000,
            provided: 500,
        };
        if let SpeechError::InsufficientSamples { required, provided } = err {
            assert_eq!(required, 1000);
            assert_eq!(provided, 500);
        } else {
            panic!("Expected InsufficientSamples variant");
        }
    }

    #[test]
    fn test_speech_error_debug_all_variants() {
        let errors = [
            SpeechError::InvalidConfig("config".to_string()),
            SpeechError::InvalidAudio("audio".to_string()),
            SpeechError::ProcessingError("proc".to_string()),
            SpeechError::InsufficientSamples {
                required: 10,
                provided: 5,
            },
        ];

        for err in &errors {
            let debug = format!("{:?}", err);
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn test_speech_error_clone_all_variants() {
        let errors = [
            SpeechError::InvalidConfig("config".to_string()),
            SpeechError::InvalidAudio("audio".to_string()),
            SpeechError::ProcessingError("proc".to_string()),
            SpeechError::InsufficientSamples {
                required: 10,
                provided: 5,
            },
        ];

        for err in &errors {
            let cloned = err.clone();
            assert_eq!(*err, cloned);
        }
    }

    #[test]
    fn test_speech_result_ok() {
        let result: SpeechResult<i32> = Ok(42);
        assert!(result.is_ok());
        assert_eq!(result.expect("should be ok"), 42);
    }

    #[test]
    fn test_speech_result_err() {
        let result: SpeechResult<i32> = Err(SpeechError::InvalidConfig("bad".to_string()));
        assert!(result.is_err());
    }

    #[test]
    fn test_speech_error_source() {
        // std::error::Error requires source() method
        let err = SpeechError::InvalidConfig("test".to_string());
        let source = std::error::Error::source(&err);
        assert!(source.is_none()); // Default implementation returns None
    }

    #[test]
    fn test_speech_error_display_empty_message() {
        let err = SpeechError::InvalidConfig(String::new());
        assert_eq!(err.to_string(), "Invalid configuration: ");

        let err = SpeechError::InvalidAudio(String::new());
        assert_eq!(err.to_string(), "Invalid audio: ");

        let err = SpeechError::ProcessingError(String::new());
        assert_eq!(err.to_string(), "Processing error: ");
    }

    #[test]
    fn test_speech_error_insufficient_samples_zero() {
        let err = SpeechError::InsufficientSamples {
            required: 0,
            provided: 0,
        };
        assert_eq!(
            err.to_string(),
            "Insufficient samples: required 0, provided 0"
        );
    }

    #[test]
    fn test_speech_error_partial_eq_same_variant_different_content() {
        let err1 = SpeechError::InvalidConfig("foo".to_string());
        let err2 = SpeechError::InvalidConfig("bar".to_string());
        assert_ne!(err1, err2);
    }

    #[test]
    fn test_speech_error_partial_eq_struct_variant() {
        let err1 = SpeechError::InsufficientSamples {
            required: 100,
            provided: 50,
        };
        let err2 = SpeechError::InsufficientSamples {
            required: 100,
            provided: 50,
        };
        assert_eq!(err1, err2);

        let err3 = SpeechError::InsufficientSamples {
            required: 100,
            provided: 60,
        };
        assert_ne!(err1, err3);
    }
}
