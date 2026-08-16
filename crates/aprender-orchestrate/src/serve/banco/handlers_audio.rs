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
        engine: "none".to_string(),
    })
}

// ============================================================================
// whisper-apr transcription (speech feature)
// ============================================================================

/// Transcription is not part of aprender.
///
/// This used to dispatch to `whisper-apr` behind a `speech` feature, with a
/// `[dry-run]` stub when the feature was off. whisper-apr is a standalone
/// project and no longer a dependency here, so the honest answer is a refusal
/// naming where the capability actually lives -- not a body that looks like a
/// transcription and is not one, and not an instruction to enable a feature
/// that no longer exists.
fn transcribe_audio(
    request: &TranscribeRequest,
) -> Result<Json<TranscribeResponse>, (StatusCode, Json<ErrorResponse>)> {
    let _ = request;
    Err((
        StatusCode::NOT_IMPLEMENTED,
        Json(ErrorResponse {
            error: "transcription_not_supported".to_string(),
            message: "aprender does not transcribe audio. whisper-apr is a \
                      standalone project; use it directly."
                .to_string(),
        }),
    ))
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
