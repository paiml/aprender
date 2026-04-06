//! Audio transcription handler — speech-to-text via whisper-apr.
//!
//! With `speech` feature: real transcription using whisper-apr.
//! Without: dry-run response for API testing.

use axum::{extract::State, http::StatusCode, response::Json};
use serde::{Deserialize, Serialize};

use super::state::BancoState;
use super::types::ErrorResponse;

/// POST /api/v1/audio/transcriptions — transcribe audio to text.
pub async fn transcribe_handler(
    State(_state): State<BancoState>,
    Json(request): Json<TranscribeRequest>,
) -> Result<Json<TranscribeResponse>, (StatusCode, Json<ErrorResponse>)> {
    transcribe_audio(&request)
}

/// GET /api/v1/audio/formats — list supported audio formats.
pub async fn audio_formats_handler() -> Json<AudioFormatsResponse> {
    Json(AudioFormatsResponse {
        formats: vec![
            AudioFormat { extension: "wav".to_string(), mime: "audio/wav".to_string() },
            AudioFormat { extension: "mp3".to_string(), mime: "audio/mpeg".to_string() },
            AudioFormat { extension: "flac".to_string(), mime: "audio/flac".to_string() },
            AudioFormat { extension: "ogg".to_string(), mime: "audio/ogg".to_string() },
        ],
        sample_rate: 16000,
        engine: if cfg!(feature = "speech") { "whisper-apr" } else { "dry-run" }.to_string(),
    })
}

// ============================================================================
// whisper-apr transcription (speech feature)
// ============================================================================

#[cfg(feature = "speech")]
fn transcribe_audio(
    request: &TranscribeRequest,
) -> Result<Json<TranscribeResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Decode base64 audio data
    let audio_bytes = base64_decode(&request.audio_data).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(format!("Invalid base64 audio: {e}"), "invalid_audio", 400)),
        )
    })?;

    let ext = request.format.as_deref().unwrap_or("wav");

    // Load audio samples
    let samples = whisper_apr::audio::load_audio_samples(&audio_bytes, ext).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(format!("Audio decode failed: {e}"), "audio_error", 400)),
        )
    })?;

    // Create transcription options
    let options = whisper_apr::TranscribeOptions {
        language: request.language.clone(),
        task: if request.translate.unwrap_or(false) {
            whisper_apr::Task::Translate
        } else {
            whisper_apr::Task::Transcribe
        },
        ..Default::default()
    };

    // Create a tiny whisper model for transcription
    let model = whisper_apr::WhisperApr::tiny();
    let result = model.transcribe(&samples, options).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new(
                format!("Transcription failed: {e}"),
                "transcription_error",
                500,
            )),
        )
    })?;

    Ok(Json(TranscribeResponse {
        text: result.text,
        language: result.language,
        duration_secs: samples.len() as f32 / 16000.0,
        segments: result
            .segments
            .into_iter()
            .map(|s| TranscribeSegment { start: s.start, end: s.end, text: s.text })
            .collect(),
    }))
}

// ============================================================================
// Dry-run transcription (no speech feature)
// ============================================================================

#[cfg(not(feature = "speech"))]
fn transcribe_audio(
    request: &TranscribeRequest,
) -> Result<Json<TranscribeResponse>, (StatusCode, Json<ErrorResponse>)> {
    let audio_len = request.audio_data.len();
    // Estimate duration from base64 size (rough: 16kHz mono 16-bit = 32KB/sec)
    let estimated_bytes = audio_len * 3 / 4; // base64 → raw
    let estimated_duration = estimated_bytes as f32 / 32000.0;

    Ok(Json(TranscribeResponse {
        text: format!(
            "[dry-run] Would transcribe {} bytes of {} audio (~{:.1}s). Enable --features speech for real transcription.",
            audio_len,
            request.format.as_deref().unwrap_or("wav"),
            estimated_duration
        ),
        language: request.language.clone().unwrap_or_else(|| "en".to_string()),
        duration_secs: estimated_duration,
        segments: vec![],
    }))
}

/// Simple base64 decoder (no external dependency).
pub(crate) fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    // Use the standard alphabet
    let table: Vec<u8> =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".to_vec();

    let input = input.trim().replace(['\n', '\r', ' '], "");
    let mut output = Vec::with_capacity(input.len() * 3 / 4);
    let mut buf: u32 = 0;
    let mut bits: u32 = 0;

    for c in input.bytes() {
        if c == b'=' {
            break;
        }
        let val = table.iter().position(|&b| b == c).ok_or("Invalid base64 character")?;
        buf = (buf << 6) | val as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            output.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }
    Ok(output)
}

// ============================================================================
// Types
// ============================================================================

/// Transcription request.
#[derive(Debug, Clone, Deserialize)]
pub struct TranscribeRequest {
    /// Base64-encoded audio data.
    pub audio_data: String,
    /// Audio format: "wav", "mp3", "flac", "ogg".
    #[serde(default)]
    pub format: Option<String>,
    /// Language code (e.g., "en", "es"). Auto-detected if not specified.
    #[serde(default)]
    pub language: Option<String>,
    /// Translate to English instead of transcribing.
    #[serde(default)]
    pub translate: Option<bool>,
}

/// Transcription response.
#[derive(Debug, Clone, Serialize)]
pub struct TranscribeResponse {
    pub text: String,
    pub language: String,
    pub duration_secs: f32,
    pub segments: Vec<TranscribeSegment>,
}

/// A timestamped segment.
#[derive(Debug, Clone, Serialize)]
pub struct TranscribeSegment {
    pub start: f32,
    pub end: f32,
    pub text: String,
}

/// Supported audio formats.
#[derive(Debug, Serialize)]
pub struct AudioFormatsResponse {
    pub formats: Vec<AudioFormat>,
    pub sample_rate: u32,
    pub engine: String,
}

/// Audio format info.
#[derive(Debug, Serialize)]
pub struct AudioFormat {
    pub extension: String,
    pub mime: String,
}
