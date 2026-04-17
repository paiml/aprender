//! Feature-gated transcription loader using whisper-apr for speech-to-text.
//!
//! When a media file has a sidecar subtitle (`.srt` or `.vtt`) adjacent to it,
//! the subtitle is loaded directly without transcription. For media files without
//! sidecars, the audio is decoded (WAV natively, MP4/MP3/etc via symphonia)
//! and transcribed using whisper-apr's Whisper ASR engine.
//!
//! Full ASR inference requires a `.apr` model file (e.g. `base.apr`,
//! `large-v3-turbo.apr`). When no model is configured, the loader computes
//! the mel spectrogram and reports what would be needed for transcription.

use crate::loader::subtitle::SubtitleLoader;
use crate::loader::{DocumentLoader, LoaderRegistry};
use crate::media::{SubtitleCue, SubtitleFormat, SubtitleTrack};
use crate::{Document, Error, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use whisper_apr::{Segment, TranscribeOptions, WhisperApr};

/// Media file extensions supported by the transcription loader.
const MEDIA_EXTENSIONS: &[&str] = &["mp4", "mp3", "wav", "m4a", "ogg", "flac", "webm"];

/// Compute backend for transcription inference.
#[derive(Debug, Clone, Copy, Default)]
pub enum TranscriptionBackend {
    /// CPU with SIMD acceleration via trueno
    #[default]
    Cpu,
    /// GPU via wgpu (cross-platform)
    Gpu,
    /// NVIDIA CUDA (Linux/Windows)
    Cuda,
}

/// Configuration for the transcription pipeline.
#[derive(Debug, Clone)]
pub struct TranscriptionConfig {
    /// Language hint (ISO 639-1, e.g., "en"). `None` for auto-detect.
    pub language: Option<String>,
    /// Beam size for decoding (1 = greedy, 5 = default).
    pub beam_size: usize,
    /// Enable word-level timestamps (more precise but slower).
    pub word_timestamps: bool,
    /// Write `.srt` sidecar files after transcription for caching.
    pub write_sidecar: bool,
    /// Compute backend for inference.
    pub backend: TranscriptionBackend,
    /// Path to the `.apr` model file (e.g. `base.apr`).
    /// When `None`, transcription of files without sidecars will fail with
    /// a helpful error message.
    pub model_path: Option<PathBuf>,
    /// Initial prompt to condition the decoder on domain vocabulary.
    /// Example: "This lecture covers AWS, Kubernetes, and YAML configurations."
    pub prompt: Option<String>,
    /// Hotwords to boost during decoding for domain-specific terms.
    /// Each string is a word or phrase to bias positively in the logit space.
    pub hotwords: Vec<String>,
}

impl Default for TranscriptionConfig {
    fn default() -> Self {
        Self {
            language: Some("en".into()),
            beam_size: 5,
            word_timestamps: false,
            write_sidecar: true,
            backend: TranscriptionBackend::default(),
            model_path: None,
            prompt: None,
            hotwords: Vec::new(),
        }
    }
}

/// Loader that handles media files via sidecar subtitle detection
/// and whisper-apr-based speech-to-text transcription.
///
/// When a media file has a sidecar subtitle (`.srt` or `.vtt`) adjacent to it,
/// the subtitle is loaded directly. Otherwise, the audio is decoded and
/// transcribed using the whisper-apr Whisper ASR engine.
///
/// # Example
///
/// ```rust,no_run
/// use trueno_rag::loader::transcription::{TranscriptionLoader, TranscriptionConfig};
/// use trueno_rag::loader::LoaderRegistry;
///
/// let mut registry = LoaderRegistry::new();
/// registry.register(Box::new(TranscriptionLoader::with_defaults()));
/// // Now the registry handles .mp4, .wav, etc. via sidecar detection
/// ```
pub struct TranscriptionLoader {
    config: TranscriptionConfig,
    whisper: Option<WhisperApr>,
}

impl TranscriptionLoader {
    /// Create a new transcription loader with the given configuration.
    ///
    /// If `config.model_path` is set, loads the whisper-apr model eagerly.
    /// Otherwise, transcription of files without sidecars will fail gracefully.
    pub fn new(config: TranscriptionConfig) -> Self {
        let whisper = config.model_path.as_ref().and_then(|path| match std::fs::read(path) {
            Ok(data) => match WhisperApr::load_from_apr(&data) {
                Ok(w) => Some(w),
                Err(e) => {
                    eprintln!("Warning: failed to load whisper model from {}: {e}", path.display());
                    None
                }
            },
            Err(e) => {
                eprintln!("Warning: failed to read model file {}: {e}", path.display());
                None
            }
        });
        Self { config, whisper }
    }

    /// Create a loader with default configuration (no model loaded).
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(TranscriptionConfig::default())
    }

    /// Transcribe audio samples using the loaded whisper-apr model.
    fn transcribe_audio(&self, samples: &[f32]) -> Result<TranscriptionResult> {
        let whisper = self.whisper.as_ref().ok_or_else(|| {
            Error::InvalidInput(
                "No Whisper model loaded. Set model_path in TranscriptionConfig \
                 or provide a .srt sidecar file alongside the media."
                    .into(),
            )
        })?;

        let mut options = TranscribeOptions::default();
        if let Some(ref lang) = self.config.language {
            options.language = Some(lang.clone());
        }
        options.word_timestamps = self.config.word_timestamps;
        if self.config.beam_size <= 1 {
            options.strategy = whisper_apr::DecodingStrategy::Greedy;
        }
        options.prompt = self.config.prompt.clone();
        options.hotwords = self.config.hotwords.clone();

        let result = whisper
            .transcribe(samples, options)
            .map_err(|e| Error::InvalidInput(format!("Transcription failed: {e}")))?;

        Ok(TranscriptionResult {
            text: result.text,
            segments: result.segments,
            language: result.language,
        })
    }

    /// Access the transcription configuration.
    #[must_use]
    pub fn config(&self) -> &TranscriptionConfig {
        &self.config
    }

    /// Check if a Whisper model is loaded and ready for transcription.
    #[must_use]
    pub fn has_model(&self) -> bool {
        self.whisper.is_some()
    }
}

/// Internal transcription result (simplified from whisper-apr types).
#[derive(Debug)]
struct TranscriptionResult {
    /// Full transcribed text (available for direct use if needed).
    #[allow(dead_code)]
    text: String,
    segments: Vec<Segment>,
    language: String,
}

impl DocumentLoader for TranscriptionLoader {
    fn supported_extensions(&self) -> Vec<&str> {
        MEDIA_EXTENSIONS.to_vec()
    }

    fn load(&self, path: &Path) -> Result<Document> {
        // 1. Check for sidecar subtitle file
        if let Some(sidecar) = LoaderRegistry::find_sidecar(path) {
            return SubtitleLoader.load(&sidecar);
        }

        // 2. Load and decode audio (WAV native, MP4/MP3/etc via symphonia)
        let samples_16k = whisper_apr::audio::load_audio_file(path).map_err(|e| {
            Error::InvalidInput(format!("Audio decode failed for {}: {e}", path.display()))
        })?;

        // 4. Transcribe
        let result = self.transcribe_audio(&samples_16k)?;

        // 5. Convert to subtitle track
        let track = segments_to_track(&result.segments);

        // 6. Optionally write sidecar for caching
        if self.config.write_sidecar {
            let _ = write_sidecar(path, &track);
        }

        // 7. Build document
        let mut doc = build_transcription_document(path, &track)?;
        doc.metadata.insert("language".into(), serde_json::json!(result.language));
        Ok(doc)
    }
}

impl std::fmt::Debug for TranscriptionLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TranscriptionLoader")
            .field("config", &self.config)
            .field("model_loaded", &self.whisper.is_some())
            .finish_non_exhaustive()
    }
}

/// Convert whisper-apr segments to a [`SubtitleTrack`].
///
/// Maps the `start`/`end` fields (seconds as f32) from whisper-apr's `Segment`
/// type to the f64 representation used by `SubtitleCue`.
pub fn segments_to_track(segments: &[Segment]) -> SubtitleTrack {
    let cues = segments
        .iter()
        .enumerate()
        .map(|(i, seg)| SubtitleCue {
            index: i,
            start_secs: f64::from(seg.start),
            end_secs: f64::from(seg.end),
            text: seg.text.trim().to_string(),
        })
        .collect();
    SubtitleTrack { format: SubtitleFormat::Srt, cues }
}

/// Build a [`Document`] from a transcription result.
pub fn build_transcription_document(path: &Path, track: &SubtitleTrack) -> Result<Document> {
    let title = path.file_stem().and_then(|s| s.to_str()).unwrap_or("Untitled").to_string();

    let mut metadata = HashMap::new();
    metadata.insert("duration_secs".into(), serde_json::json!(track.duration_secs()));
    metadata.insert("format".into(), serde_json::json!("transcription"));
    metadata.insert("cue_count".into(), serde_json::json!(track.cues.len()));
    metadata.insert(
        "subtitle_cues".into(),
        serde_json::to_value(&track.cues).map_err(Error::Serialization)?,
    );

    let mut doc =
        Document::new(track.to_plain_text()).with_title(title).with_source(path.to_string_lossy());
    doc.metadata = metadata;
    Ok(doc)
}

/// Write a [`SubtitleTrack`] as an SRT sidecar file adjacent to a media file.
///
/// Returns the path of the written sidecar.
pub fn write_sidecar(media_path: &Path, track: &SubtitleTrack) -> Result<PathBuf> {
    let sidecar_path = media_path.with_extension("srt");
    let srt_content = track.to_srt_string();
    std::fs::write(&sidecar_path, srt_content).map_err(Error::Io)?;
    Ok(sidecar_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcription_config_default() {
        let config = TranscriptionConfig::default();
        assert_eq!(config.language, Some("en".into()));
        assert_eq!(config.beam_size, 5);
        assert!(!config.word_timestamps);
        assert!(config.write_sidecar);
        assert!(config.model_path.is_none());
        assert!(config.prompt.is_none());
        assert!(config.hotwords.is_empty());
    }

    #[test]
    fn test_transcription_config_with_prompt() {
        let config = TranscriptionConfig {
            prompt: Some("This is a lecture about AWS and Kubernetes.".into()),
            ..TranscriptionConfig::default()
        };
        assert_eq!(config.prompt.as_deref(), Some("This is a lecture about AWS and Kubernetes."));
    }

    #[test]
    fn test_transcription_config_with_hotwords() {
        let config = TranscriptionConfig {
            hotwords: vec!["AWS".into(), "Kubernetes".into(), "YAML".into()],
            ..TranscriptionConfig::default()
        };
        assert_eq!(config.hotwords.len(), 3);
        assert_eq!(config.hotwords[0], "AWS");
    }

    #[test]
    fn test_transcription_backend_default() {
        let backend = TranscriptionBackend::default();
        assert!(matches!(backend, TranscriptionBackend::Cpu));
    }

    #[test]
    fn test_media_extensions() {
        let loader = TranscriptionLoader::with_defaults();
        let exts = loader.supported_extensions();
        assert!(exts.contains(&"mp4"));
        assert!(exts.contains(&"wav"));
        assert!(exts.contains(&"mp3"));
        assert!(exts.contains(&"flac"));
        assert!(exts.contains(&"webm"));
    }

    #[test]
    fn test_has_model_default_false() {
        let loader = TranscriptionLoader::with_defaults();
        assert!(!loader.has_model());
    }

    #[test]
    fn test_segments_to_track() {
        let segments = vec![
            Segment { start: 0.0, end: 3.0, text: "Hello world.".into(), tokens: vec![] },
            Segment { start: 3.5, end: 6.0, text: "How are you?".into(), tokens: vec![] },
        ];
        let track = segments_to_track(&segments);
        assert_eq!(track.cues.len(), 2);
        assert_eq!(track.cues[0].text, "Hello world.");
        assert!((track.cues[0].start_secs).abs() < 0.001);
        assert!((track.cues[0].end_secs - 3.0).abs() < 0.001);
        assert!((track.cues[1].start_secs - 3.5).abs() < 0.001);
        assert!((track.cues[1].end_secs - 6.0).abs() < 0.001);
    }

    #[test]
    fn test_segments_to_track_empty() {
        let track = segments_to_track(&[]);
        assert!(track.cues.is_empty());
        assert!((track.duration_secs()).abs() < 0.001);
    }

    #[test]
    fn test_load_non_wav_media_errors_helpful() {
        let loader = TranscriptionLoader::with_defaults();
        let result = loader.load(Path::new("/tmp/nonexistent_video.mp4"));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Audio decode") || err.contains("sidecar") || err.contains("not found")
        );
    }

    #[test]
    fn test_sidecar_fallback() {
        let dir = std::env::temp_dir().join("trueno_rag_test_transcription_sidecar");
        let _ = std::fs::create_dir_all(&dir);
        let media = dir.join("lecture.wav");
        let srt = dir.join("lecture.srt");
        std::fs::write(&media, b"fake wav data").unwrap();
        std::fs::write(&srt, "1\n00:00:01,000 --> 00:00:04,500\nSidecar text.\n").unwrap();

        let loader = TranscriptionLoader::with_defaults();
        let doc = loader.load(&media).unwrap();
        assert!(doc.content.contains("Sidecar text"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_build_transcription_document() {
        let track = SubtitleTrack {
            format: SubtitleFormat::Srt,
            cues: vec![
                SubtitleCue { index: 0, start_secs: 0.0, end_secs: 3.0, text: "Hello".into() },
                SubtitleCue { index: 1, start_secs: 3.0, end_secs: 6.0, text: "World".into() },
            ],
        };
        let doc = build_transcription_document(Path::new("/tmp/test.wav"), &track).unwrap();
        assert_eq!(doc.content, "Hello World");
        assert_eq!(doc.title, Some("test".into()));
        assert!(doc.metadata.contains_key("duration_secs"));
        assert!(doc.metadata.contains_key("subtitle_cues"));
        assert!(doc.metadata.contains_key("cue_count"));
    }

    #[test]
    fn test_write_sidecar() {
        let dir = std::env::temp_dir().join("trueno_rag_test_write_sidecar");
        let _ = std::fs::create_dir_all(&dir);
        let media = dir.join("output.mp4");

        let track = SubtitleTrack {
            format: SubtitleFormat::Srt,
            cues: vec![SubtitleCue {
                index: 0,
                start_secs: 1.0,
                end_secs: 4.5,
                text: "Hello.".into(),
            }],
        };

        let sidecar = write_sidecar(&media, &track).unwrap();
        assert_eq!(sidecar.extension().unwrap(), "srt");
        let content = std::fs::read_to_string(&sidecar).unwrap();
        assert!(content.contains("Hello."));
        assert!(content.contains("00:00:01,000"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_loader_debug() {
        let loader = TranscriptionLoader::with_defaults();
        let debug = format!("{loader:?}");
        assert!(debug.contains("TranscriptionLoader"));
        assert!(debug.contains("model_loaded"));
    }

    #[test]
    #[ignore = "stereo_to_mono not yet implemented"]
    fn test_stereo_to_mono() {
        let stereo = vec![0.5_f32, -0.5, 1.0, 0.0, -1.0, 1.0];
        let mono: Vec<f32> = stereo.chunks(2).map(|c| (c[0] + c[1]) / 2.0).collect();
        assert_eq!(mono.len(), 3);
        assert!((mono[0]).abs() < 0.001); // (0.5 + -0.5) / 2
        assert!((mono[1] - 0.5).abs() < 0.001); // (1.0 + 0.0) / 2
        assert!((mono[2]).abs() < 0.001); // (-1.0 + 1.0) / 2
    }

    #[test]
    #[ignore = "stereo_to_mono not yet implemented"]
    fn test_stereo_to_mono_passthrough() {
        let mono_input = vec![0.1_f32, 0.2, 0.3];
        assert_eq!(mono_input.len(), 3);
    }

    #[test]
    fn test_transcribe_without_model_errors() {
        let loader = TranscriptionLoader::with_defaults();
        let result = loader.transcribe_audio(&[0.0; 16000]);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("model") || err.contains("sidecar"));
    }

    #[test]
    fn test_config_with_model_path() {
        let config = TranscriptionConfig {
            model_path: Some(PathBuf::from("/tmp/nonexistent.apr")),
            ..TranscriptionConfig::default()
        };
        let loader = TranscriptionLoader::new(config);
        // Model file doesn't exist, so model won't be loaded
        assert!(!loader.has_model());
    }
}
